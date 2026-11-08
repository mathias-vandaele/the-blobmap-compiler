//! Writes tier files (SPEC §10, verify):
//! - `tier tiny <data dir> <tier file>`: writes the fixtures under the data directory and
//!   lists them;
//! - `tier add <tier file> <tier> <data dir> <name> <pbf> <overture dir> <config dir>
//!   [<extent>]`: adds or replaces one set of inputs, paths relative to the data directory.

use std::path::Path;

use anyhow::{Result, bail};
use blobmap_conformance::tier::{InputSet, TierFile, tiny};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        ["tiny", data, out] => std::fs::write(out, tiny(Path::new(data))?.to_toml()?)?,
        ["add", file, tier, data, name, pbf, overture, config, ref rest @ ..] if rest.len() <= 1 => {
            let path = Path::new(file);
            let mut tf = if path.exists() { TierFile::read(path)? } else { TierFile { tier: tier.into(), inputs: Vec::new() } };
            if tf.tier != tier {
                bail!("{file} is the {} tier", tf.tier);
            }
            let set = InputSet::scan(Path::new(data), name, pbf, overture, config, rest.first().copied())?;
            tf.inputs.retain(|s| s.name != name);
            tf.inputs.push(set);
            tf.inputs.sort_by(|x, y| x.name.cmp(&y.name));
            std::fs::write(path, tf.to_toml()?)?;
        }
        _ => bail!("usage: tier tiny <data dir> <tier file> | tier add <tier file> <tier> <data dir> <name> <pbf> <overture> <config> [<extent>]"),
    }
    Ok(())
}
