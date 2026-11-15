//! Overture's place taxonomy of the pinned release (ported from the old
//! `overture/taxonomy.rs`), indexed as FORMAT §7.4 stores it: nodes sorted by key, every
//! basic category a node, and the redirects of `mapping_sheet.csv` resolved to the node they
//! end at (a removal without a target is dropped, so a query naming it is an error).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use blobmap_model::NONE16;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Taxonomy {
    /// Keys in byte order; a node's index is its position.
    pub name: Vec<String>,
    pub parent: Vec<u16>,
    pub basic: Vec<u16>,
    pub is_basic: Vec<bool>,
    /// Retired keys in byte order, and the node each one resolves to.
    pub redirect: Vec<String>,
    pub redirect_to: Vec<u16>,
}

fn csv_rows(path: &Path) -> Result<(Vec<String>, Vec<Vec<String>>)> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(!text.contains('"'), "{}: quoted fields are not supported", path.display());
    let mut lines = text.lines();
    let head: Vec<String> = lines.next().context("empty csv")?.split(',').map(str::to_string).collect();
    let rows: Vec<Vec<String>> = lines.filter(|l| !l.is_empty()).map(|l| l.split(',').map(str::to_string).collect()).collect();
    ensure!(rows.iter().all(|r| r.len() == head.len()), "{}: ragged rows", path.display());
    Ok((head, rows))
}

fn col(head: &[String], name: &str) -> Result<usize> {
    head.iter().position(|h| h == name).with_context(|| format!("column {name} missing"))
}

impl Taxonomy {
    /// From the release's `taxonomy/` directory: `taxonomy.csv` and, if present,
    /// `mapping_sheet.csv`.
    pub fn read(dir: &Path) -> Result<Taxonomy> {
        let (head, rows) = csv_rows(&dir.join("taxonomy.csv"))?;
        let (c_path, c_prim, c_basic, c_isb) =
            (col(&head, "taxonomy")?, col(&head, "primary")?, col(&head, "basicCategory")?, col(&head, "isBasic")?);
        let mut name: Vec<String> = rows.iter().map(|r| r[c_prim].clone()).collect();
        name.sort();
        let by_name: HashMap<&str, u16> = name.iter().enumerate().map(|(i, n)| (n.as_str(), i as u16)).collect();
        ensure!(by_name.len() == name.len(), "category keys are not unique");
        ensure!(name.len() < NONE16 as usize, "too many categories for U16");
        let n = name.len();
        let (mut parent, mut basic, mut is_basic) = (vec![NONE16; n], vec![NONE16; n], vec![false; n]);
        for r in &rows {
            let i = by_name[r[c_prim].as_str()] as usize;
            let path: Vec<&str> = r[c_path].split(" > ").collect();
            ensure!(path.last() == Some(&r[c_prim].as_str()), "path {} does not end in {}", r[c_path], r[c_prim]);
            if path.len() > 1 {
                parent[i] = *by_name.get(path[path.len() - 2]).with_context(|| format!("unknown parent in {}", r[c_path]))?;
            }
            if !r[c_basic].is_empty() {
                // FORMAT §7.4: every basic category must be a node with the same key.
                basic[i] =
                    *by_name.get(r[c_basic].as_str()).with_context(|| format!("basic category {} is not a taxonomy node", r[c_basic]))?;
            }
            is_basic[i] = r[c_isb] == "true";
        }
        for i in 0..n {
            ensure!(
                basic[i] == NONE16 || is_basic[basic[i] as usize],
                "{}: basic category {} is not flagged basic",
                name[i],
                name[basic[i] as usize]
            );
        }
        let mut raw: HashMap<String, Option<String>> = HashMap::new();
        let sheet = dir.join("mapping_sheet.csv");
        if sheet.exists() {
            let (head, rows) = csv_rows(&sheet)?;
            let (c_old, c_new, c_rm, c_to) = (
                col(&head, "Old Primary Category")?,
                col(&head, "New Primary Category")?,
                col(&head, "PC Removed")?,
                col(&head, "PC Redirect To")?,
            );
            for r in rows {
                let (old, new, to) = (&r[c_old], &r[c_new], r[c_to].trim());
                if old.is_empty() || by_name.contains_key(old.as_str()) {
                    continue;
                }
                let target = if !to.is_empty() {
                    (to != "n/a").then(|| to.to_string())
                } else if r[c_rm] == "TRUE" {
                    None
                } else if old != new {
                    Some(new.clone())
                } else {
                    continue;
                };
                raw.entry(old.clone()).or_insert(target);
            }
        }
        let mut redirects: Vec<(String, u16)> = Vec::new();
        for old in raw.keys() {
            let mut cur = old.clone();
            for _ in 0..8 {
                if let Some(&i) = by_name.get(cur.as_str()) {
                    redirects.push((old.clone(), i));
                    break;
                }
                match raw.get(&cur) {
                    Some(Some(next)) => cur = next.clone(),
                    _ => break,
                }
            }
        }
        redirects.sort();
        Ok(Taxonomy {
            redirect: redirects.iter().map(|r| r.0.clone()).collect(),
            redirect_to: redirects.iter().map(|r| r.1).collect(),
            name,
            parent,
            basic,
            is_basic,
        })
    }

    pub fn id(&self, key: &str) -> Option<u16> {
        self.name.binary_search_by(|n| n.as_str().cmp(key)).ok().map(|i| i as u16)
    }

    /// A query key after the redirects (SPEC §6), or None.
    pub fn resolve(&self, key: &str) -> Option<u16> {
        match self.redirect.binary_search_by(|n| n.as_str().cmp(key)) {
            Ok(i) => Some(self.redirect_to[i]),
            Err(_) => self.id(key),
        }
    }

    /// Whether node c is `node` or below it.
    pub fn is_under(&self, mut c: u16, node: u16) -> bool {
        loop {
            if c == node {
                return true;
            }
            c = self.parent[c as usize];
            if c == NONE16 {
                return false;
            }
        }
    }
}
