//! The operations of SPEC §9, with arguments and answers in the golden JSON form of
//! FORMAT §11.

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::runtime::model::{Dir, Id, Reason, Runtime};
use crate::runtime::query::{Source, Work};
use crate::{INF, NONE16};

pub fn parse_id(s: &str) -> Result<Id> {
    let b = s.as_bytes();
    let ok = b.len() == 36
        && b.iter().enumerate().all(|(i, &c)| if matches!(i, 8 | 13 | 18 | 23) { c == b'-' } else { c.is_ascii_digit() || (b'a'..=b'f').contains(&c) });
    ensure!(ok, "{s:?} is not a canonical UUID");
    let hex: String = s.chars().filter(|&c| c != '-').collect();
    let mut id = [0u8; 16];
    for (k, x) in id.iter_mut().enumerate() {
        *x = u8::from_str_radix(&hex[2 * k..2 * k + 2], 16)?;
    }
    Ok(id)
}

pub fn id_string(id: &Id) -> String {
    let h = hex::encode(id);
    format!("{}-{}-{}-{}-{}", &h[0..8], &h[8..12], &h[12..16], &h[16..20], &h[20..32])
}

pub fn time_json(t: u16) -> Value {
    if t == INF { json!("BEYOND") } else { json!(t) }
}

fn reason_json(r: Reason, id: &str) -> Value {
    json!({"reason": r.name(), "id": id})
}

/// A category of a query (SPEC §6): a taxonomy node or a basic category.
#[derive(Clone, Copy, Debug)]
pub enum Category {
    Taxonomy(u16),
    Basic(u16),
}

impl Runtime {
    /// Resolves an ID string: its internal index, or `UnknownId`.
    pub fn resolve(&self, s: &str) -> Result<u32, Reason> {
        let id = parse_id(s).map_err(|_| Reason::UnknownId)?;
        self.vocab.lookup(&id).ok_or(Reason::UnknownId)
    }

    /// A served source, or the reason (SPEC §9 "Reasons").
    fn source(&self, s: &str, mode: usize) -> Result<u32, Reason> {
        let i = self.resolve(s)?;
        self.served(i, mode)?;
        Ok(i)
    }

    /// A category key after Overture's redirects (FORMAT §7.4); unknown keys are errors.
    pub fn category(&self, v: &Value) -> Result<Category> {
        let (key, basic) = match (v.get("category"), v.get("basic")) {
            (Some(k), None) => (k.as_str().context("category key")?, false),
            (None, Some(k)) => (k.as_str().context("basic key")?, true),
            _ => bail!("a category is {{\"category\": key}} or {{\"basic\": key}}"),
        };
        let t = &self.vocab;
        let key = match t.redirect.binary_search_by(|x| x.as_str().cmp(key)) {
            Ok(k) => t.tax_name[t.redirect_to[k] as usize].as_str(),
            Err(_) => key,
        };
        let node = t.tax_name.binary_search_by(|x| x.as_str().cmp(key)).ok().with_context(|| format!("no category {key:?}"))? as u16;
        if basic {
            ensure!(t.tax_is_basic[node as usize] == 1, "{key:?} is not a basic category");
            Ok(Category::Basic(node))
        } else {
            Ok(Category::Taxonomy(node))
        }
    }

    /// Whether place p (index into places) matches the category.
    pub fn matches(&self, p: usize, c: Category) -> bool {
        let v = &self.vocab;
        if v.status[p] != 0 || v.primary[p] == NONE16 {
            return false;
        }
        match c {
            Category::Basic(b) => v.basic[p] == b,
            Category::Taxonomy(x) => {
                let reaches = |mut n: u16| {
                    while n != NONE16 {
                        if n == x {
                            return true;
                        }
                        n = v.tax_parent[n as usize];
                    }
                    false
                };
                reaches(v.primary[p]) || v.alt[v.alt_first[p] as usize..v.alt_first[p + 1] as usize].iter().any(|&a| reaches(a))
            }
        }
    }

    /// The served places of a category, as column sources.
    fn places_of(&self, c: Category, mode: usize) -> Vec<Source> {
        let na = self.vocab.address.len();
        (0..self.vocab.place.len())
            .filter(|&p| self.matches(p, c) && self.served((na + p) as u32, mode).is_ok())
            .map(|p| Source { id: (na + p) as u32, start: 0 })
            .collect()
    }

    fn sources(&self, v: &Value, mode: usize) -> Result<Result<Vec<Source>, Value>> {
        let mut out = Vec::new();
        for s in v.as_array().context("sources")? {
            let id = s["id"].as_str().context("source id")?;
            let start = s.get("start").map_or(Ok(0), |x| x.as_u64().context("start").map(|x| x.min(INF as u64) as u16))?;
            match self.source(id, mode) {
                Ok(i) => out.push(Source { id: i, start }),
                Err(r) => return Ok(Err(reason_json(r, id))),
            }
        }
        Ok(Ok(out))
    }

    /// IDs, sorted by their bytes, and their SHA-256 (FORMAT §11 `ids_sha256`).
    pub fn id_set(&self, idx: impl Iterator<Item = u32>) -> (Vec<Id>, String) {
        let mut ids: Vec<Id> = idx.map(|i| self.vocab.id(i)).collect();
        ids.sort_unstable();
        let mut h = Sha256::new();
        for id in &ids {
            h.update(id);
        }
        (ids, hex::encode(h.finalize()))
    }

    fn id_answer(&self, idx: impl Iterator<Item = u32>, with_ids: bool) -> Value {
        let (ids, sha) = self.id_set(idx);
        let mut a = json!({"count": ids.len(), "ids_sha256": sha});
        if with_ids {
            a["ids"] = ids.iter().map(id_string).collect();
        }
        a
    }

    /// Smaller time, then smaller ID (SPEC §7), or larger ID under the fault.
    fn better(&self, a: (u16, Id), b: (u16, Id)) -> bool {
        if self.faults.largest_id { a.0 < b.0 || (a.0 == b.0 && a.1 > b.1) } else { a < b }
    }

    pub fn travel_time(&self, a: &str, b: &str, mode: usize) -> Result<Value> {
        self.travel_time_in(&mut Work::default(), a, b, mode)
    }

    fn travel_time_in(&self, w: &mut Work, a: &str, b: &str, mode: usize) -> Result<Value> {
        let ia = match self.source(a, mode) {
            Ok(i) => i,
            Err(r) => return Ok(reason_json(r, a)),
        };
        let ib = match self.source(b, mode) {
            Ok(i) => i,
            Err(r) => return Ok(reason_json(r, b)),
        };
        if ia == ib {
            return Ok(json!({"time": 0}));
        }
        let mv = &self.vocab.modes[mode];
        if mv.component[ia as usize] != mv.component[ib as usize] {
            return Ok(reason_json(Reason::DifferentComponent, b));
        }
        Ok(json!({"time": time_json(self.pair_in(w, ia, ib, mode)?)}))
    }

    pub fn nearest(&self, c: Category, id: &str, mode: usize, dir: Dir) -> Result<Value> {
        self.nearest_in(&mut Work::default(), c, id, mode, dir)
    }

    fn nearest_in(&self, w: &mut Work, c: Category, id: &str, mode: usize, dir: Dir) -> Result<Value> {
        let i = match self.source(id, mode) {
            Ok(i) => i,
            Err(r) => return Ok(reason_json(r, id)),
        };
        let col = self.column_in(w, &[Source { id: i, start: 0 }], mode, dir)?;
        let na = self.vocab.address.len();
        let mut best: Option<(u16, Id)> = None;
        for p in 0..self.vocab.place.len() {
            let t = col[na + p];
            if t == INF || !self.matches(p, c) {
                continue;
            }
            let cand = (t, self.vocab.place[p]);
            if best.is_none_or(|b| self.better(cand, b)) {
                best = Some(cand);
            }
        }
        w.give(col);
        Ok(match best {
            Some((t, p)) => json!({"time": t, "place": id_string(&p)}),
            None => json!({"time": "BEYOND", "place": null}),
        })
    }

    /// One golden query's answer (FORMAT §11).
    pub fn answer(&self, op: &str, args: &Value) -> Result<Value> {
        self.answer_in(&mut Work::default(), op, args)
    }

    /// `answer` with the working state `w`, reused across a thread's queries.
    pub fn answer_in(&self, w: &mut Work, op: &str, args: &Value) -> Result<Value> {
        let mode = || self.mode(args["mode"].as_str().context("mode")?);
        let dir = || Dir::parse(args["dir"].as_str().context("dir")?);
        let with_ids = args.get("ids").and_then(Value::as_bool).unwrap_or(false);
        match op {
            "travel_time" => self.travel_time_in(w, args["a"].as_str().context("a")?, args["b"].as_str().context("b")?, mode()?),
            "nearest" => self.nearest_in(w, self.category(&args["category"])?, args["id"].as_str().context("id")?, mode()?, dir()?),
            "isochrone" | "column" => {
                let m = mode()?;
                let src = match self.sources(&args["sources"], m)? {
                    Ok(s) => s,
                    Err(r) => return Ok(r),
                };
                let col = self.column_in(w, &src, m, dir()?)?;
                let a = if op == "column" {
                    column_json(&col)
                } else {
                    let t = args["t"].as_u64().context("t")?;
                    self.id_answer((0..col.len() as u32).filter(|&i| col[i as usize] != INF && col[i as usize] as u64 <= t), with_ids)
                };
                w.give(col);
                Ok(a)
            }
            "reachable_from_all" | "reachable_from_any" => {
                let m = mode()?;
                let src = match self.sources(&args["sources"], m)? {
                    Ok(s) => s,
                    Err(r) => return Ok(r),
                };
                let t = args["t"].as_u64().context("t")?;
                let n = self.vocab.n();
                let all = op == "reachable_from_all";
                let mut hit: Vec<bool> = vec![all && !src.is_empty(); n];
                for s in &src {
                    let col = self.column_in(w, std::slice::from_ref(s), m, Dir::From)?;
                    for i in 0..n {
                        let ok = col[i] != INF && col[i] as u64 <= t;
                        hit[i] = if all { hit[i] && ok } else { hit[i] || ok };
                    }
                    w.give(col);
                }
                Ok(self.id_answer((0..n as u32).filter(|&i| hit[i as usize]), with_ids))
            }
            "multi_criteria" => self.multi_criteria(w, args, with_ids),
            "chained" | "chained_column" => {
                let c = self.category(&args["transfer_category"])?;
                let penalty = args["penalty"].as_u64().context("penalty")?;
                ensure!(penalty < INF as u64, "penalty must be below {INF}");
                if op == "chained" {
                    self.chained(w, args["a"].as_str().context("a")?, args["b"].as_str().context("b")?, c, penalty as u16)
                } else {
                    self.chained_column(w, args["id"].as_str().context("id")?, c, penalty as u16, dir()?)
                }
            }
            _ => bail!("unknown op {op:?}"),
        }
    }

    /// Transfer places of a chained trip (SPEC §9): kept places of the category served for
    /// both car and walk, as internal indices.
    fn transfers(&self, c: Category, car: usize, walk: usize) -> Vec<u32> {
        let na = self.vocab.address.len();
        (0..self.vocab.place.len())
            .map(|p| (p, (na + p) as u32))
            .filter(|&(p, i)| self.matches(p, c) && self.served(i, car).is_ok() && self.served(i, walk).is_ok())
            .map(|(_, i)| i)
            .collect()
    }

    /// `chained(a, b)`: min over transfer places p of car(a, p) ⊕ penalty ⊕ walk(p, b), and p.
    fn chained(&self, w: &mut Work, a: &str, b: &str, c: Category, penalty: u16) -> Result<Value> {
        let (car, walk) = (self.mode("car")?, self.mode("walk")?);
        let ia = match self.source(a, car) {
            Ok(i) => i,
            Err(r) => return Ok(reason_json(r, a)),
        };
        let ib = match self.source(b, walk) {
            Ok(i) => i,
            Err(r) => return Ok(reason_json(r, b)),
        };
        // The same ID (SPEC §9): 0 s, through no transfer place.
        if ia == ib {
            return Ok(json!({"time": 0, "place": null}));
        }
        let drive = self.column_in(w, &[Source { id: ia, start: 0 }], car, Dir::From)?;
        let on_foot = self.column_in(w, &[Source { id: ib, start: 0 }], walk, Dir::To)?;
        let mut best: Option<(u16, Id)> = None;
        for p in self.transfers(c, car, walk) {
            let t = self.faults.add(self.faults.add(drive[p as usize], penalty), on_foot[p as usize]);
            let cand = (t, self.vocab.id(p));
            if t != INF && best.is_none_or(|b| self.better(cand, b)) {
                best = Some(cand);
            }
        }
        w.give(drive);
        w.give(on_foot);
        Ok(match best {
            Some((t, p)) => json!({"time": t, "place": id_string(&p)}),
            None => json!({"time": "BEYOND", "place": null}),
        })
    }

    /// The chained column (SPEC §9): `From` car from `id` to each transfer place, then walk
    /// on to every ID; `To` car from every ID to each transfer place, then walk on to `id`.
    fn chained_column(&self, w: &mut Work, id: &str, c: Category, penalty: u16, dir: Dir) -> Result<Value> {
        let (car, walk) = (self.mode("car")?, self.mode("walk")?);
        let (first, second) = if dir == Dir::From { (car, walk) } else { (walk, car) };
        let i = match self.source(id, first) {
            Ok(i) => i,
            Err(r) => return Ok(reason_json(r, id)),
        };
        // The leg at `id`'s end, to or from each transfer place; then the other leg as a
        // multi-source column started at those times plus the penalty.
        let leg = self.column_in(w, &[Source { id: i, start: 0 }], first, dir)?;
        let src: Vec<Source> = self
            .transfers(c, car, walk)
            .into_iter()
            .map(|p| Source { id: p, start: self.faults.add(leg[p as usize], penalty) })
            .filter(|s| s.start != INF)
            .collect();
        w.give(leg);
        let mut col = self.column_in(w, &src, second, dir)?;
        // `id` itself: 0 s by the same-ID rule, when it is served for the other leg too.
        if self.served(i, second).is_ok() {
            col[i as usize] = 0;
        }
        let a = column_json(&col);
        w.give(col);
        Ok(a)
    }

    fn multi_criteria(&self, w: &mut Work, args: &Value, with_ids: bool) -> Result<Value> {
        let expr = &args["expr"];
        let kind = args.get("kind").and_then(Value::as_str).unwrap_or("addresses");
        let mut terms: Vec<&Value> = Vec::new();
        collect_terms(expr, &mut terms)?;
        let n = self.vocab.n();
        let na = self.vocab.address.len();
        let mut modes: Vec<usize> = Vec::new();
        let mut times: Vec<Vec<u16>> = Vec::new();
        for t in &terms {
            let m = self.mode(t["mode"].as_str().context("term mode")?)?;
            let dir = Dir::parse(t["dir"].as_str().context("term dir")?)?;
            let target = &t["target"];
            let col = if let Some(s) = target.get("sources") {
                let starts = target.get("start").and_then(Value::as_array);
                let mut src = Vec::new();
                for (k, id) in s.as_array().context("sources")?.iter().enumerate() {
                    let id = id.as_str().context("source id")?;
                    let start = starts.and_then(|a| a.get(k)).and_then(Value::as_u64).unwrap_or(0).min(INF as u64) as u16;
                    match self.source(id, m) {
                        Ok(i) => src.push(Source { id: i, start }),
                        Err(r) => return Ok(reason_json(r, id)),
                    }
                }
                self.column_in(w, &src, m, dir)?
            } else {
                // The time to the nearest place, as `nearest`: the column of the category's
                // places in the other direction.
                let c = self.category(target)?;
                self.column_in(w, &self.places_of(c, m), m, dir.flip())?
            };
            if !modes.contains(&m) {
                modes.push(m);
            }
            times.push(col);
        }
        let range: Box<dyn Iterator<Item = u32>> = match kind {
            "addresses" => Box::new(0..na as u32),
            "places" => Box::new(na as u32..n as u32),
            "both" => Box::new(0..n as u32),
            _ => bail!("kind must be addresses, places or both"),
        };
        let candidates: Vec<u32> = range.filter(|&i| modes.iter().all(|&m| self.served(i, m).is_ok())).collect();
        let matching: Vec<u32> = candidates.into_iter().filter(|&i| eval(expr, &mut 0, &times, i as usize).unwrap_or(false)).collect();
        let a = match args.get("k") {
            None => self.id_answer(matching.into_iter(), with_ids),
            Some(kv) => {
                let k = kv.as_u64().context("k")? as usize;
                let weights: Vec<u64> = terms.iter().map(|t| t["weight"].as_u64().context("weight")).collect::<Result<_>>()?;
                let mut scored: Vec<(u64, Id)> = matching
                    .into_iter()
                    .filter(|&i| times.iter().all(|c| c[i as usize] != INF))
                    .map(|i| (times.iter().zip(&weights).map(|(c, w)| w * c[i as usize] as u64).sum(), self.vocab.id(i)))
                    .collect();
                if self.faults.largest_id {
                    scored.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
                } else {
                    scored.sort_unstable();
                }
                scored.truncate(k);
                json!({"top": scored.iter().map(|(s, id)| json!({"id": id_string(id), "score": s.to_string()})).collect::<Vec<_>>()})
            }
        };
        for c in times {
            w.give(c);
        }
        Ok(a)
    }
}

/// A column's golden answer (FORMAT §11).
fn column_json(col: &[u16]) -> Value {
    let mut h = Sha256::new();
    for t in col {
        h.update(t.to_le_bytes());
    }
    json!({"times_sha256": hex::encode(h.finalize()), "beyond": col.iter().filter(|&&t| t == INF).count()})
}

fn collect_terms<'a>(e: &'a Value, out: &mut Vec<&'a Value>) -> Result<()> {
    if let Some(t) = e.get("term") {
        out.push(t);
    } else if let Some(xs) = e.get("and").or_else(|| e.get("or")) {
        for x in xs.as_array().context("and/or")? {
            collect_terms(x, out)?;
        }
    } else if let Some(x) = e.get("not") {
        collect_terms(x, out)?;
    } else {
        bail!("bad expression {e}");
    }
    Ok(())
}

/// Evaluates the expression for ID i; `next` walks the terms in `collect_terms` order.
fn eval(e: &Value, next: &mut usize, times: &[Vec<u16>], i: usize) -> Result<bool> {
    if let Some(t) = e.get("term") {
        let c = times[*next][i];
        *next += 1;
        let max = t["max"].as_u64().context("max")?;
        return Ok(c != INF && c as u64 <= max);
    }
    if let Some(xs) = e.get("and") {
        let mut all = true;
        for x in xs.as_array().context("and")? {
            all &= eval(x, next, times, i)?;
        }
        return Ok(all);
    }
    if let Some(xs) = e.get("or") {
        let mut any = false;
        for x in xs.as_array().context("or")? {
            any |= eval(x, next, times, i)?;
        }
        return Ok(any);
    }
    Ok(!eval(&e["not"], next, times, i)?)
}
