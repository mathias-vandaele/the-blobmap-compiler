//! Answers computed from the reference alone (Dijkstra matrices, the inputs' catalog and
//! taxonomy), in the golden JSON form of FORMAT §11. Golden answers are these.

use blobmap_format::{INF, NONE16};
use blobmap_overture::Id;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::reference::Reference;

pub fn uuid(id: &Id) -> String {
    blobmap_overture::uuid_string(id)
}

fn time(t: u16) -> Value {
    if t == INF { json!("BEYOND") } else { json!(t) }
}

pub fn reason_name(code: u8) -> &'static str {
    match code {
        1 => "NoEligibleRoad",
        2 => "NoRoad",
        3 => "TrappedComponent",
        _ => "FilteredPlace",
    }
}

/// The reference with its all-pairs matrices, per mode in `Reference::modes` order.
pub struct Expected<'a> {
    pub r: &'a Reference,
    pub m: Vec<Vec<Vec<u16>>>,
}

pub enum Cat {
    Taxonomy(u16),
    Basic(u16),
}

impl Expected<'_> {
    fn mode(&self, name: &str) -> usize {
        self.r.modes.iter().position(|m| m.name == name).expect("mode")
    }

    /// An ID string resolved and served, or its reason answer.
    fn served(&self, s: &str, m: usize) -> Result<usize, Value> {
        let unknown = || json!({"reason": "UnknownId", "id": s});
        let id = blobmap_overture::parse_uuid(s).map_err(|_| unknown())?;
        let i = self.r.lookup(&id).ok_or_else(unknown)? as usize;
        match self.r.modes[m].at[i] {
            Ok(_) => Ok(i),
            Err((c, _)) => Err(json!({"reason": reason_name(c), "id": s})),
        }
    }

    fn expert(&self, m: usize, i: usize) -> u32 {
        self.r.modes[m].at[i].as_ref().map(|e| e.expert).unwrap_or(u32::MAX)
    }

    pub fn travel_time(&self, a: &str, b: &str, mode: &str) -> Value {
        let m = self.mode(mode);
        let ia = match self.served(a, m) {
            Ok(i) => i,
            Err(r) => return r,
        };
        let ib = match self.served(b, m) {
            Ok(i) => i,
            Err(r) => return r,
        };
        if ia == ib {
            return json!({"time": 0});
        }
        if self.expert(m, ia) != self.expert(m, ib) {
            return json!({"reason": "DifferentComponent", "id": b});
        }
        json!({"time": time(self.m[m][ia][ib])})
    }

    /// The column of (source, start) pairs in a direction ("from" or "to").
    pub fn column(&self, m: usize, src: &[(usize, u16)], dir: &str) -> Vec<u16> {
        (0..self.r.n())
            .map(|i| {
                src.iter()
                    .map(|&(s, st)| {
                        let t = if dir == "from" { self.m[m][s][i] } else { self.m[m][i][s] };
                        st.saturating_add(t)
                    })
                    .min()
                    .unwrap_or(INF)
            })
            .collect()
    }

    fn sources(&self, v: &Value, m: usize) -> Result<Vec<(usize, u16)>, Value> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|s| {
                let i = self.served(s["id"].as_str().unwrap(), m)?;
                Ok((i, s.get("start").and_then(Value::as_u64).unwrap_or(0) as u16))
            })
            .collect()
    }

    pub fn ids_answer(&self, idx: impl Iterator<Item = usize>) -> Value {
        let mut ids: Vec<Id> = idx.map(|i| self.r.ids[i]).collect();
        ids.sort_unstable();
        let mut h = Sha256::new();
        for id in &ids {
            h.update(id);
        }
        json!({"count": ids.len(), "ids_sha256": hex::encode(h.finalize())})
    }

    pub fn category(&self, v: &Value) -> Option<Cat> {
        let t = &self.r.catalog.taxonomy;
        if let Some(k) = v.get("category").and_then(Value::as_str) {
            return t.resolve(k).map(Cat::Taxonomy);
        }
        let k = v.get("basic").and_then(Value::as_str)?;
        t.resolve(k).filter(|&n| t.is_basic[n as usize]).map(Cat::Basic)
    }

    /// Whether internal index i is a kept place matching the category.
    pub fn matches(&self, i: usize, c: &Cat) -> bool {
        let Some(p) = self.r.place_of[i] else { return false };
        let p = &self.r.catalog.places[p];
        let t = &self.r.catalog.taxonomy;
        if p.status != 0 || p.primary == NONE16 {
            return false;
        }
        match *c {
            Cat::Basic(b) => p.basic == b,
            Cat::Taxonomy(x) => t.is_under(p.primary, x) || p.alt.iter().any(|&a| t.is_under(a, x)),
        }
    }

    pub fn answer(&self, op: &str, a: &Value) -> Value {
        let mode = || self.mode(a["mode"].as_str().unwrap());
        match op {
            "travel_time" => self.travel_time(a["a"].as_str().unwrap(), a["b"].as_str().unwrap(), a["mode"].as_str().unwrap()),
            "nearest" => {
                let m = mode();
                let id = a["id"].as_str().unwrap();
                let i = match self.served(id, m) {
                    Ok(i) => i,
                    Err(r) => return r,
                };
                let c = self.category(&a["category"]).expect("a known category");
                let col = self.column(m, &[(i, 0)], a["dir"].as_str().unwrap());
                let best = (0..self.r.n()).filter(|&p| col[p] != INF && self.matches(p, &c)).map(|p| (col[p], self.r.ids[p])).min();
                match best {
                    Some((t, p)) => json!({"time": t, "place": uuid(&p)}),
                    None => json!({"time": "BEYOND", "place": null}),
                }
            }
            "isochrone" | "column" => {
                let m = mode();
                let src = match self.sources(&a["sources"], m) {
                    Ok(s) => s,
                    Err(r) => return r,
                };
                let col = self.column(m, &src, a["dir"].as_str().unwrap());
                if op == "column" {
                    let mut h = Sha256::new();
                    for t in &col {
                        h.update(t.to_le_bytes());
                    }
                    return json!({"times_sha256": hex::encode(h.finalize()), "beyond": col.iter().filter(|&&t| t == INF).count()});
                }
                let t = a["t"].as_u64().unwrap();
                self.ids_answer((0..col.len()).filter(|&i| col[i] != INF && col[i] as u64 <= t))
            }
            "reachable_from_all" | "reachable_from_any" => {
                let m = mode();
                let src = match self.sources(&a["sources"], m) {
                    Ok(s) => s,
                    Err(r) => return r,
                };
                let t = a["t"].as_u64().unwrap();
                let cols: Vec<Vec<u16>> = src.iter().map(|s| self.column(m, std::slice::from_ref(s), "from")).collect();
                let ok = |c: &Vec<u16>, i: usize| c[i] != INF && c[i] as u64 <= t;
                let all = op == "reachable_from_all";
                self.ids_answer((0..self.r.n()).filter(|&i| if all { !cols.is_empty() && cols.iter().all(|c| ok(c, i)) } else { cols.iter().any(|c| ok(c, i)) }))
            }
            "multi_criteria" => self.multi_criteria(a),
            "chained" | "chained_column" => self.chained(op, a),
            _ => unreachable!("op {op}"),
        }
    }

    /// SPEC §9 chained trips, from the matrices: car(x, p) ⊕ penalty ⊕ walk(p, y).
    fn chained(&self, op: &str, a: &Value) -> Value {
        let (car, walk) = (self.mode("car"), self.mode("walk"));
        let c = self.category(&a["transfer_category"]).expect("a known category");
        let pen = a["penalty"].as_u64().unwrap() as u16;
        let transfers: Vec<usize> =
            (0..self.r.n()).filter(|&p| self.matches(p, &c) && self.r.modes[car].at[p].is_ok() && self.r.modes[walk].at[p].is_ok()).collect();
        let trip = |x: usize, y: usize, p: usize| self.m[car][x][p].saturating_add(pen).saturating_add(self.m[walk][p][y]);
        if op == "chained" {
            let (sa, sb) = (a["a"].as_str().unwrap(), a["b"].as_str().unwrap());
            let x = match self.served(sa, car) {
                Ok(i) => i,
                Err(r) => return r,
            };
            let y = match self.served(sb, walk) {
                Ok(i) => i,
                Err(r) => return r,
            };
            return match transfers.iter().map(|&p| (trip(x, y, p), self.r.ids[p])).filter(|t| t.0 != INF).min() {
                Some((t, p)) => json!({"time": t, "place": uuid(&p)}),
                None => json!({"time": "BEYOND", "place": null}),
            };
        }
        let from = a["dir"] == "from";
        let id = a["id"].as_str().unwrap();
        let i = match self.served(id, if from { car } else { walk }) {
            Ok(i) => i,
            Err(r) => return r,
        };
        let col: Vec<u16> = (0..self.r.n())
            .map(|y| transfers.iter().map(|&p| if from { trip(i, y, p) } else { trip(y, i, p) }).min().unwrap_or(INF))
            .collect();
        let mut h = Sha256::new();
        for t in &col {
            h.update(t.to_le_bytes());
        }
        json!({"times_sha256": hex::encode(h.finalize()), "beyond": col.iter().filter(|&&t| t == INF).count()})
    }

    fn multi_criteria(&self, a: &Value) -> Value {
        let expr = &a["expr"];
        let mut terms = Vec::new();
        collect(expr, &mut terms);
        let mut modes = Vec::new();
        let mut times = Vec::new();
        for t in &terms {
            let m = self.mode(t["mode"].as_str().unwrap());
            let dir = t["dir"].as_str().unwrap();
            let target = &t["target"];
            let col = if let Some(s) = target.get("sources") {
                let starts = target.get("start").and_then(Value::as_array);
                let mut src = Vec::new();
                for (k, id) in s.as_array().unwrap().iter().enumerate() {
                    let id = id.as_str().unwrap();
                    match self.served(id, m) {
                        Ok(i) => src.push((i, starts.and_then(|x| x.get(k)).and_then(Value::as_u64).unwrap_or(0) as u16)),
                        Err(r) => return r,
                    }
                }
                self.column(m, &src, dir)
            } else {
                let c = self.category(target).expect("a known category");
                let places: Vec<(usize, u16)> = (0..self.r.n()).filter(|&p| self.matches(p, &c) && self.r.modes[m].at[p].is_ok()).map(|p| (p, 0)).collect();
                self.column(m, &places, if dir == "from" { "to" } else { "from" })
            };
            if !modes.contains(&m) {
                modes.push(m);
            }
            times.push(col);
        }
        let na = self.r.place_of.iter().filter(|p| p.is_none()).count();
        let n = self.r.n();
        let range = match a.get("kind").and_then(Value::as_str).unwrap_or("addresses") {
            "addresses" => 0..na,
            "places" => na..n,
            _ => 0..n,
        };
        let matching: Vec<usize> =
            range.filter(|&i| modes.iter().all(|&m| self.r.modes[m].at[i].is_ok())).filter(|&i| eval(expr, &mut 0, &times, i)).collect();
        match a.get("k") {
            None => self.ids_answer(matching.into_iter()),
            Some(k) => {
                let w: Vec<u64> = terms.iter().map(|t| t["weight"].as_u64().unwrap()).collect();
                let mut s: Vec<(u64, Id)> = matching
                    .into_iter()
                    .filter(|&i| times.iter().all(|c| c[i] != INF))
                    .map(|i| (times.iter().zip(&w).map(|(c, w)| w * c[i] as u64).sum(), self.r.ids[i]))
                    .collect();
                s.sort_unstable();
                s.truncate(k.as_u64().unwrap() as usize);
                json!({"top": s.iter().map(|(sc, id)| json!({"id": uuid(id), "score": sc.to_string()})).collect::<Vec<_>>()})
            }
        }
    }
}

fn collect<'a>(e: &'a Value, out: &mut Vec<&'a Value>) {
    if let Some(t) = e.get("term") {
        out.push(t);
    } else if let Some(xs) = e.get("and").or_else(|| e.get("or")) {
        for x in xs.as_array().unwrap() {
            collect(x, out);
        }
    } else {
        collect(&e["not"], out);
    }
}

fn eval(e: &Value, next: &mut usize, times: &[Vec<u16>], i: usize) -> bool {
    if let Some(t) = e.get("term") {
        let c = times[*next][i];
        *next += 1;
        return c != INF && c as u64 <= t["max"].as_u64().unwrap();
    }
    if let Some(xs) = e.get("and") {
        return xs.as_array().unwrap().iter().fold(true, |acc, x| eval(x, next, times, i) & acc);
    }
    if let Some(xs) = e.get("or") {
        return xs.as_array().unwrap().iter().fold(false, |acc, x| eval(x, next, times, i) | acc);
    }
    !eval(&e["not"], next, times, i)
}
