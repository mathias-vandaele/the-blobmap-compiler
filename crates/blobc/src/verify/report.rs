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

    /// Adds the cases and mismatches of a part of the check run on its own: absorbing the
    /// parts in order keeps the examples a single run would have kept.
    pub fn absorb(&mut self, part: Outcome) {
        self.checked += part.checked;
        self.mismatches += part.mismatches;
        let room = EXAMPLES.saturating_sub(self.examples.len());
        self.examples.extend(part.examples.into_iter().take(room));
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
