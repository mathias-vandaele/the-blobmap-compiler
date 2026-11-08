//! Check outcomes: how many cases each check covered and its mismatches.

use std::fmt;

/// Examples kept per check; every mismatch is counted.
const EXAMPLES: usize = 8;

#[derive(Clone, Debug, Default)]
pub struct Outcome {
    pub name: String,
    pub checked: u64,
    pub mismatches: u64,
    pub examples: Vec<String>,
}

impl Outcome {
    pub fn count(&mut self) {
        self.checked += 1;
    }

    pub fn fail(&mut self, msg: String) {
        self.mismatches += 1;
        if self.examples.len() < EXAMPLES {
            self.examples.push(msg);
        }
    }

    pub fn expect(&mut self, ok: bool, msg: impl FnOnce() -> String) {
        if !ok {
            self.fail(msg());
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub outcomes: Vec<Outcome>,
}

impl Report {
    pub fn check(&mut self, name: &str) -> &mut Outcome {
        if let Some(k) = self.outcomes.iter().position(|o| o.name == name) {
            return &mut self.outcomes[k];
        }
        self.outcomes.push(Outcome { name: name.to_string(), ..Outcome::default() });
        self.outcomes.last_mut().unwrap()
    }

    pub fn mismatches(&self) -> u64 {
        self.outcomes.iter().map(|o| o.mismatches).sum()
    }

    /// Mismatches outside the given checks.
    pub fn mismatches_except(&self, skip: &[&str]) -> u64 {
        self.outcomes.iter().filter(|o| !skip.iter().any(|s| o.name.starts_with(s))).map(|o| o.mismatches).sum()
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for o in &self.outcomes {
            writeln!(f, "{:>8} checked {:>6} mismatches  {}", o.checked, o.mismatches, o.name)?;
            for e in &o.examples {
                writeln!(f, "             - {e}")?;
            }
        }
        Ok(())
    }
}
