//! The operations of SPEC §9, typed, and their golden JSON form (FORMAT §11).

use anyhow::{Context, Result, bail, ensure};
use blobmap_model::INF;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::heads::{Times, Want};
use crate::runtime::{ColReq, Runtime, Src, bound_of};
use crate::vocab::{Category, Dir, Id, Reason, id_string, parse_id};

/// An answer that is not a time: the reason and the ID it is about (SPEC §9 "Reasons").
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub reason: Reason,
    pub id: String,
}

pub type Answer<T> = Result<Result<T, Refusal>>;

/// A source as a query names it: an ID and a start time in seconds.
#[derive(Clone, Debug)]
pub struct Source {
    pub id: String,
    pub start: u16,
}

/// A time, or `None` for BEYOND.
pub type Time = Option<u16>;

fn time(t: u16) -> Time {
    (t != INF).then_some(t)
}

/// The best of a search over places: its time and place (`None` for the same-ID case).
pub type Best = Option<(u16, Option<Id>)>;

#[derive(Clone, Debug)]
pub enum Expr {
    Term(Term),
    And(Vec<Expr>),
    Or(Vec<Expr>),
    Not(Box<Expr>),
}

#[derive(Clone, Debug)]
pub struct Term {
    pub target: Target,
    pub mode: String,
    pub dir: Dir,
    pub max: u64,
    pub weight: Option<u64>,
}

#[derive(Clone, Debug)]
pub enum Target {
    Sources(Vec<Source>),
    /// A category key, basic or not, resolved in term order.
    Category { key: String, basic: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Addresses,
    Places,
    Both,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Matches {
    /// The matching IDs, sorted by their bytes.
    Ids(Vec<Id>),
    /// The k best by weighted score, then ID.
    Top(Vec<(Id, u64)>),
}

impl Runtime {
    /// Resolves an ID string: its internal index, or `UnknownId`.
    pub fn resolve(&self, s: &str) -> Result<u32, Reason> {
        let id = parse_id(s).ok_or(Reason::UnknownId)?;
        self.vocab.lookup(&id).ok_or(Reason::UnknownId)
    }

    fn source(&self, s: &str, mode: usize) -> Result<u32, Refusal> {
        let r = |reason| Refusal { reason, id: s.to_string() };
        let i = self.resolve(s).map_err(r)?;
        self.served(i, mode).map_err(r)?;
        Ok(i)
    }

    fn sources(&self, v: &[Source], mode: usize) -> Result<Vec<Src>, Refusal> {
        v.iter().map(|s| Ok(Src { id: self.source(&s.id, mode)?, start: s.start })).collect()
    }

    pub fn category(&self, key: &str, basic: bool) -> Result<Category> {
        self.vocab.category_key(key, basic)
    }

    fn places_of(&self, c: Category, mode: usize) -> Vec<Src> {
        let na = self.vocab.address.len();
        (0..self.vocab.place.len()).filter(|&p| self.vocab.matches(p, c) && self.served((na + p) as u32, mode).is_ok()).map(|p| Src { id: (na + p) as u32, start: 0 }).collect()
    }

    fn sorted_ids(&self, idx: impl Iterator<Item = u32>) -> Vec<Id> {
        let mut ids: Vec<Id> = idx.map(|i| self.vocab.id(i)).collect();
        ids.sort_unstable();
        ids
    }

    pub fn travel_time(&self, a: &str, b: &str, mode: &str) -> Answer<Time> {
        let m = self.mode(mode)?;
        let ia = match self.source(a, m) {
            Ok(i) => i,
            Err(r) => return Ok(Err(r)),
        };
        let ib = match self.source(b, m) {
            Ok(i) => i,
            Err(r) => return Ok(Err(r)),
        };
        if ia == ib {
            return Ok(Ok(Some(0)));
        }
        let mv = &self.vocab.modes[m];
        if mv.component(ia as usize) != mv.component(ib as usize) {
            return Ok(Err(Refusal { reason: Reason::DifferentComponent, id: b.to_string() }));
        }
        Ok(Ok(time(self.pair(ia, ib, m)?)))
    }

    pub fn nearest(&self, c: Category, id: &str, mode: &str, dir: Dir) -> Answer<Best> {
        let m = self.mode(mode)?;
        let i = match self.source(id, m) {
            Ok(i) => i,
            Err(r) => return Ok(Err(r)),
        };
        let na = self.vocab.address.len();
        let col = self.column(&[Src { id: i, start: 0 }], m, dir, INF, Want::range(na, self.vocab.n()))?;
        let mut best: Option<(u16, Id)> = None;
        for p in 0..self.vocab.place.len() {
            let t = col.get((na + p) as u32);
            if t == INF || !self.vocab.matches(p, c) {
                continue;
            }
            let cand = (t, self.vocab.place[p]);
            if best.is_none_or(|b| cand < b) {
                best = Some(cand);
            }
        }
        Ok(Ok(best.map(|(t, p)| (t, Some(p)))))
    }

    /// A full column: every ID's time, by internal index.
    pub fn column_of(&self, sources: &[Source], mode: &str, dir: Dir) -> Answer<Times> {
        let m = self.mode(mode)?;
        let src = match self.sources(sources, m) {
            Ok(s) => s,
            Err(r) => return Ok(Err(r)),
        };
        Ok(Ok(self.column(&src, m, dir, INF, Want::all(self.vocab.n()))?))
    }

    /// The IDs within t seconds, sorted.
    pub fn isochrone(&self, sources: &[Source], t: u64, mode: &str, dir: Dir) -> Answer<Vec<Id>> {
        let m = self.mode(mode)?;
        let src = match self.sources(sources, m) {
            Ok(s) => s,
            Err(r) => return Ok(Err(r)),
        };
        let bound = if self.opts.prune { bound_of(t) } else { INF };
        let col = self.column(&src, m, dir, bound, Want::all(self.vocab.n()).within(t))?;
        Ok(Ok(self.sorted_ids(col.finite().into_iter())))
    }

    /// The IDs within t seconds from every source (`all`) or from any.
    pub fn reachable(&self, sources: &[Source], t: u64, mode: &str, all: bool) -> Answer<Vec<Id>> {
        let m = self.mode(mode)?;
        let src = match self.sources(sources, m) {
            Ok(s) => s,
            Err(r) => return Ok(Err(r)),
        };
        let bound = if self.opts.prune { bound_of(t) } else { INF };
        let want = Want::all(self.vocab.n()).within(t);
        let reqs: Vec<ColReq> = src.iter().map(|s| ColReq { sources: vec![*s], mode: m, dir: Dir::From, bound, want }).collect();
        let mut lists = self.columns(&reqs)?.iter().map(|c| c.finite()).collect::<Vec<_>>().into_iter();
        let mut hit = lists.next().unwrap_or_default();
        for l in lists {
            if all {
                hit.retain(|i| l.binary_search(i).is_ok());
            } else {
                hit.extend(l);
            }
        }
        if !all {
            hit.sort_unstable();
            hit.dedup();
        }
        Ok(Ok(self.sorted_ids(hit.into_iter())))
    }

    pub fn multi_criteria(&self, expr: &Expr, kind: Kind, k: Option<usize>) -> Answer<Matches> {
        let mut terms: Vec<&Term> = Vec::new();
        collect(expr, &mut terms);
        let (n, na) = (self.vocab.n() as u32, self.vocab.address.len() as u32);
        let range = match kind {
            Kind::Addresses => 0..na,
            Kind::Places => na..n,
            Kind::Both => 0..n,
        };
        // Without k, and when an ID beyond every term does not match, only the IDs within
        // some term's limit can match: each column answers those alone.
        let sparse = k.is_none() && !eval(expr, &mut 0, &|_| false);
        let mut modes: Vec<usize> = Vec::new();
        let mut reqs = Vec::new();
        for t in &terms {
            let m = self.mode(&t.mode)?;
            let bound = if self.opts.prune && k.is_none() { bound_of(t.max) } else { INF };
            let mut want = Want::range(range.start as usize, range.end as usize);
            if sparse {
                want = want.within(t.max);
            }
            reqs.push(match &t.target {
                Target::Sources(s) => match self.sources(s, m) {
                    Ok(src) => ColReq { sources: src, mode: m, dir: t.dir, bound, want },
                    Err(r) => return Ok(Err(r)),
                },
                // The time to the nearest place, as `nearest`: the column of the category's
                // places in the other direction.
                Target::Category { key, basic } => ColReq { sources: self.places_of(self.category(key, *basic)?, m), mode: m, dir: t.dir.flip(), bound, want },
            });
            if !modes.contains(&m) {
                modes.push(m);
            }
        }
        let times = self.columns(&reqs)?;
        let (times, terms) = (&times, &terms);
        let ok = |&i: &u32| {
            modes.iter().all(|&m| self.served(i, m).is_ok())
                && eval(expr, &mut 0, &|j: usize| {
                    let c = times[j].get(i);
                    c != INF && c as u64 <= terms[j].max
                })
        };
        let matching: Vec<u32> = if sparse {
            let mut cand: Vec<u32> = times.iter().flat_map(|c| c.finite()).collect();
            cand.sort_unstable();
            cand.dedup();
            cand.into_iter().filter(ok).collect()
        } else {
            range.filter(ok).collect()
        };
        Ok(Ok(match k {
            None => Matches::Ids(self.sorted_ids(matching.into_iter())),
            Some(k) => {
                let weights: Vec<u64> = terms.iter().map(|t| t.weight.context("weight")).collect::<Result<_>>()?;
                let mut scored: Vec<(u64, Id)> = matching
                    .into_iter()
                    .filter(|&i| times.iter().all(|c| c.get(i) != INF))
                    .map(|i| (times.iter().zip(&weights).map(|(c, w)| w * c.get(i) as u64).sum(), self.vocab.id(i)))
                    .collect();
                scored.sort_unstable();
                scored.truncate(k);
                Matches::Top(scored.into_iter().map(|(s, id)| (id, s)).collect())
            }
        }))
    }

    /// Kept places of the category served for both car and walk (SPEC §9).
    fn transfers(&self, c: Category, car: usize, walk: usize) -> Vec<u32> {
        let na = self.vocab.address.len();
        (0..self.vocab.place.len())
            .map(|p| (p, (na + p) as u32))
            .filter(|&(p, i)| self.vocab.matches(p, c) && self.served(i, car).is_ok() && self.served(i, walk).is_ok())
            .map(|(_, i)| i)
            .collect()
    }

    /// Car from a to a transfer place, the penalty, then walk to b: the best, and the place.
    pub fn chained(&self, a: &str, b: &str, c: Category, penalty: u16) -> Answer<Best> {
        let (car, walk) = (self.mode("car")?, self.mode("walk")?);
        let ia = match self.source(a, car) {
            Ok(i) => i,
            Err(r) => return Ok(Err(r)),
        };
        let ib = match self.source(b, walk) {
            Ok(i) => i,
            Err(r) => return Ok(Err(r)),
        };
        if ia == ib {
            return Ok(Ok(Some((0, None))));
        }
        let places = Want::range(self.vocab.address.len(), self.vocab.n());
        let cols = self.columns(&[
            ColReq { sources: vec![Src { id: ia, start: 0 }], mode: car, dir: Dir::From, bound: INF, want: places },
            ColReq { sources: vec![Src { id: ib, start: 0 }], mode: walk, dir: Dir::To, bound: INF, want: places },
        ])?;
        let (drive, on_foot) = (&cols[0], &cols[1]);
        let mut best: Option<(u16, Id)> = None;
        for p in self.transfers(c, car, walk) {
            let t = drive.get(p).saturating_add(penalty).saturating_add(on_foot.get(p));
            let cand = (t, self.vocab.id(p));
            if t != INF && best.is_none_or(|b| cand < b) {
                best = Some(cand);
            }
        }
        Ok(Ok(best.map(|(t, p)| (t, Some(p)))))
    }

    /// The chained column (SPEC §9): `From` car from `id` to each transfer place, then walk
    /// on to every ID; `To` car from every ID to each transfer place, then walk on to `id`.
    pub fn chained_column(&self, id: &str, c: Category, penalty: u16, dir: Dir) -> Answer<Times> {
        let (car, walk) = (self.mode("car")?, self.mode("walk")?);
        let (first, second) = if dir == Dir::From { (car, walk) } else { (walk, car) };
        let i = match self.source(id, first) {
            Ok(i) => i,
            Err(r) => return Ok(Err(r)),
        };
        let leg = self.column(&[Src { id: i, start: 0 }], first, dir, INF, Want::range(self.vocab.address.len(), self.vocab.n()))?;
        let src: Vec<Src> = self.transfers(c, car, walk).into_iter().map(|p| Src { id: p, start: leg.get(p).saturating_add(penalty) }).filter(|s| s.start != INF).collect();
        let mut col = self.column(&src, second, dir, INF, Want::all(self.vocab.n()))?;
        if self.served(i, second).is_ok() {
            col.set(i, 0);
        }
        Ok(Ok(col))
    }

    /// One golden query's answer (FORMAT §11).
    pub fn answer(&self, op: &str, args: &Value) -> Result<Value> {
        let s = |k: &str| args[k].as_str().with_context(|| k.to_string());
        let dir = || Dir::parse(s("dir")?);
        let with_ids = args.get("ids").and_then(Value::as_bool).unwrap_or(false);
        let refusal = |r: Refusal| json!({"reason": r.reason.name(), "id": r.id});
        let best = |b: Best| match b {
            Some((t, p)) => json!({"time": t, "place": p.map(|p| id_string(&p))}),
            None => json!({"time": "BEYOND", "place": null}),
        };
        let ids = |v: Vec<Id>| id_answer(&v, with_ids);
        Ok(match op {
            "travel_time" => match self.travel_time(s("a")?, s("b")?, s("mode")?)? {
                Ok(t) => json!({"time": t.map_or(json!("BEYOND"), |t| json!(t))}),
                Err(r) => refusal(r),
            },
            "nearest" => {
                let c = category_json(self, &args["category"])?;
                match self.nearest(c, s("id")?, s("mode")?, dir()?)? {
                    Ok(b) => best(b),
                    Err(r) => refusal(r),
                }
            }
            "column" => match self.column_of(&sources_json(&args["sources"])?, s("mode")?, dir()?)? {
                Ok(c) => column_json(&c, self.vocab.n()),
                Err(r) => refusal(r),
            },
            "isochrone" => {
                let src = sources_json(&args["sources"])?;
                let t = args["t"].as_u64().context("t")?;
                match self.isochrone(&src, t, s("mode")?, dir()?)? {
                    Ok(v) => ids(v),
                    Err(r) => refusal(r),
                }
            }
            "reachable_from_all" | "reachable_from_any" => {
                let src = sources_json(&args["sources"])?;
                let t = args["t"].as_u64().context("t")?;
                match self.reachable(&src, t, s("mode")?, op == "reachable_from_all")? {
                    Ok(v) => ids(v),
                    Err(r) => refusal(r),
                }
            }
            "multi_criteria" => {
                let expr = expr_json(&args["expr"])?;
                let kind = match args.get("kind").and_then(Value::as_str).unwrap_or("addresses") {
                    "addresses" => Kind::Addresses,
                    "places" => Kind::Places,
                    "both" => Kind::Both,
                    _ => bail!("kind must be addresses, places or both"),
                };
                let k = match args.get("k") {
                    None => None,
                    Some(v) => Some(v.as_u64().context("k")? as usize),
                };
                match self.multi_criteria(&expr, kind, k)? {
                    Ok(Matches::Ids(v)) => ids(v),
                    Ok(Matches::Top(v)) => json!({"top": v.iter().map(|(id, s)| json!({"id": id_string(id), "score": s.to_string()})).collect::<Vec<_>>()}),
                    Err(r) => refusal(r),
                }
            }
            "chained" | "chained_column" => {
                let c = category_json(self, &args["transfer_category"])?;
                let penalty = args["penalty"].as_u64().context("penalty")?;
                ensure!(penalty < INF as u64, "penalty must be below {INF}");
                if op == "chained" {
                    match self.chained(s("a")?, s("b")?, c, penalty as u16)? {
                        Ok(b) => best(b),
                        Err(r) => refusal(r),
                    }
                } else {
                    match self.chained_column(s("id")?, c, penalty as u16, dir()?)? {
                        Ok(col) => column_json(&col, self.vocab.n()),
                        Err(r) => refusal(r),
                    }
                }
            }
            _ => bail!("unknown op {op:?}"),
        })
    }
}

fn collect<'a>(e: &'a Expr, out: &mut Vec<&'a Term>) {
    match e {
        Expr::Term(t) => out.push(t),
        Expr::And(xs) | Expr::Or(xs) => xs.iter().for_each(|x| collect(x, out)),
        Expr::Not(x) => collect(x, out),
    }
}

/// Evaluates the expression, `term(j)` telling whether the j-th term holds; `next` walks
/// the terms in `collect` order.
fn eval(e: &Expr, next: &mut usize, term: &impl Fn(usize) -> bool) -> bool {
    match e {
        Expr::Term(_) => {
            let x = term(*next);
            *next += 1;
            x
        }
        Expr::And(xs) => {
            let mut all = true;
            for x in xs {
                all &= eval(x, next, term);
            }
            all
        }
        Expr::Or(xs) => {
            let mut any = false;
            for x in xs {
                any |= eval(x, next, term);
            }
            any
        }
        Expr::Not(x) => !eval(x, next, term),
    }
}

pub fn column_json(col: &Times, n: usize) -> Value {
    let mut h = Sha256::new();
    let mut buf: Vec<u8> = Vec::with_capacity(1 << 16);
    let mut beyond = 0usize;
    col.for_each(n as u32, |t| {
        beyond += (t == INF) as usize;
        buf.extend_from_slice(&t.to_le_bytes());
        if buf.len() >= 1 << 16 {
            h.update(&buf);
            buf.clear();
        }
    });
    h.update(&buf);
    json!({"times_sha256": hex::encode(h.finalize()), "beyond": beyond})
}

pub fn id_answer(ids: &[Id], with_ids: bool) -> Value {
    let mut h = Sha256::new();
    for id in ids {
        h.update(id);
    }
    let mut a = json!({"count": ids.len(), "ids_sha256": hex::encode(h.finalize())});
    if with_ids {
        a["ids"] = ids.iter().map(id_string).collect();
    }
    a
}

fn category_json(rt: &Runtime, v: &Value) -> Result<Category> {
    rt.vocab.category(v)
}

pub fn sources_json(v: &Value) -> Result<Vec<Source>> {
    v.as_array()
        .context("sources")?
        .iter()
        .map(|s| {
            let id = s["id"].as_str().context("source id")?.to_string();
            let start = s.get("start").map_or(Ok(0), |x| x.as_u64().context("start").map(|x| x.min(INF as u64) as u16))?;
            Ok(Source { id, start })
        })
        .collect()
}

pub fn expr_json(e: &Value) -> Result<Expr> {
    if let Some(t) = e.get("term") {
        let target = &t["target"];
        let target = if let Some(s) = target.get("sources") {
            let starts = target.get("start").and_then(Value::as_array);
            let src = s
                .as_array()
                .context("sources")?
                .iter()
                .enumerate()
                .map(|(k, id)| {
                    let start = starts.and_then(|a| a.get(k)).and_then(Value::as_u64).unwrap_or(0).min(INF as u64) as u16;
                    Ok(Source { id: id.as_str().context("source id")?.to_string(), start })
                })
                .collect::<Result<_>>()?;
            Target::Sources(src)
        } else {
            match (target.get("category"), target.get("basic")) {
                (Some(k), None) => Target::Category { key: k.as_str().context("category key")?.to_string(), basic: false },
                (None, Some(k)) => Target::Category { key: k.as_str().context("basic key")?.to_string(), basic: true },
                _ => bail!("a category is {{\"category\": key}} or {{\"basic\": key}}"),
            }
        };
        return Ok(Expr::Term(Term {
            target,
            mode: t["mode"].as_str().context("term mode")?.to_string(),
            dir: Dir::parse(t["dir"].as_str().context("term dir")?)?,
            max: t["max"].as_u64().context("max")?,
            weight: t.get("weight").and_then(Value::as_u64),
        }));
    }
    if let Some(xs) = e.get("and") {
        return Ok(Expr::And(xs.as_array().context("and")?.iter().map(expr_json).collect::<Result<_>>()?));
    }
    if let Some(xs) = e.get("or") {
        return Ok(Expr::Or(xs.as_array().context("or")?.iter().map(expr_json).collect::<Result<_>>()?));
    }
    if let Some(x) = e.get("not") {
        return Ok(Expr::Not(Box::new(expr_json(x)?)));
    }
    bail!("bad expression {e}")
}
