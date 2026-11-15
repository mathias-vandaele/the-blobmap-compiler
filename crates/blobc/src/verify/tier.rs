//! Tier files (SPEC §10, verify): `tiers/<tier>.toml` lists the tier's sets of inputs, each
//! with the paths of its `.pbf`, Overture release, config and extent under the data
//! directory, and every file it uses with its SHA-256.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use blobmap_model::manifest::Manifest;
use serde::{Deserialize, Serialize};

use crate::build::sha256_file;
use crate::verify::reference::Inputs;

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct TierFile {
    pub tier: String,
    #[serde(default)]
    pub inputs: Vec<InputSet>,
}

/// One set of inputs. Paths are relative to the data directory, with `/` separators.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct InputSet {
    pub name: String,
    pub pbf: String,
    pub overture: String,
    pub config: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extent: Option<String>,
    /// Every file of the set: path → SHA-256.
    pub files: BTreeMap<String, String>,
}

/// Every file under `dir`, as `/`-separated paths relative to `root`, sorted.
fn files_under(root: &Path, dir: &Path, out: &mut Vec<String>) -> Result<()> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?.map(|e| e.map(|e| e.path())).collect::<std::io::Result<_>>()?;
    entries.sort();
    for p in entries {
        if p.is_dir() {
            files_under(root, &p, out)?;
        } else {
            out.push(relative(root, &p)?);
        }
    }
    Ok(())
}

impl TierFile {
    pub fn read(path: &Path) -> Result<TierFile> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading the tier file {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn to_toml(&self) -> Result<String> {
        Ok(format!("# Inputs of the {} tier (SPEC §10, verify): paths under `blobc verify --data`.\n{}", self.tier, toml::to_string(self)?))
    }

    /// The set whose `.pbf` and extent have the model's SHA-256s.
    pub fn find(&self, m: &Manifest) -> Result<&InputSet> {
        let extent = m.inputs.extent.as_ref().map(|e| e.sha256.as_str());
        let hit = self.inputs.iter().find(|s| {
            s.files.get(&s.pbf).map(String::as_str) == Some(m.inputs.pbf.sha256.as_str())
                && s.extent.as_ref().map(|e| s.files.get(e).map(String::as_str)) == extent.map(Some)
        });
        hit.with_context(|| format!("the {} tier lists no inputs with the model's .pbf ({}) and extent", self.tier, m.inputs.pbf.sha256))
    }
}

impl InputSet {
    /// A set from files on disk: `overture` and `config` are directories, every file under
    /// them is listed.
    pub fn scan(data: &Path, name: &str, pbf: &str, overture: &str, config: &str, extent: Option<&str>) -> Result<InputSet> {
        let mut paths = vec![pbf.to_string()];
        files_under(data, &data.join(overture), &mut paths)?;
        files_under(data, &data.join(config), &mut paths)?;
        paths.extend(extent.map(str::to_string));
        let mut files = BTreeMap::new();
        for p in paths {
            let sha = sha256_file(&data.join(&p))?.0;
            files.insert(p, sha);
        }
        Ok(InputSet { name: name.into(), pbf: pbf.into(), overture: overture.into(), config: config.into(), extent: extent.map(str::to_string), files })
    }

    /// Checks every file under `data` and the config against the model's manifest, and gives
    /// the inputs' paths.
    pub fn check(&self, data: &Path, m: &Manifest, work: &Path) -> Result<Inputs> {
        for (p, want) in &self.files {
            let got = sha256_file(&data.join(p))?.0;
            ensure!(&got == want, "{}: SHA-256 {got}, the tier file lists {want}", data.join(p).display());
        }
        let cfg = &m.config;
        for (what, f) in [("car profile", &cfg.profiles.car), ("active profile", &cfg.profiles.active), ("place rules", &cfg.place_rules)] {
            let p = format!("{}/{}", self.config, f.name);
            match self.files.get(&p) {
                Some(sha) if sha == &f.sha256 => {}
                Some(_) => bail!("{p}: the {what} differ from the model's"),
                None => bail!("{p}: the tier file does not list the model's {what}"),
            }
        }
        Ok(Inputs {
            pbf: data.join(&self.pbf),
            overture: data.join(&self.overture),
            config: data.join(&self.config),
            extent: self.extent.as_ref().map(|e| data.join(e)),
            work: work.to_path_buf(),
        })
    }
}

fn relative(root: &Path, p: &Path) -> Result<String> {
    Ok(p.strip_prefix(root)?.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/"))
}

/// Writes every fixture scenario under `data` and gives the tiny tier file that lists them,
/// each with its default config.
pub fn tiny(data: &Path) -> Result<TierFile> {
    let mut tf = TierFile { tier: "tiny".into(), inputs: Vec::new() };
    for s in crate::fixtures::scenarios::all() {
        let w = s.write(data)?;
        let ext = w.extent.as_ref().map(|e| relative(data, e)).transpose()?;
        tf.inputs.push(InputSet::scan(data, s.name, &relative(data, &w.pbf)?, &relative(data, &w.overture)?, &relative(data, &w.configs[0].1)?, ext.as_deref())?);
    }
    Ok(tf)
}

/// Writes tier files (SPEC §10, verify):
/// - `blobc tier tiny <data dir> <tier file>`: writes the fixtures under the data directory and
///   lists them;
/// - `blobc tier add <tier file> <tier> <data dir> <name> <pbf> <overture dir> <config dir>
///   [<extent>]`: adds or replaces one set of inputs, paths relative to the data directory.
pub fn main(a: Vec<String>) -> Result<()> {
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
        _ => bail!("usage: blobc tier tiny <data dir> <tier file> | blobc tier add <tier file> <tier> <data dir> <name> <pbf> <overture> <config> [<extent>]"),
    }
    Ok(())
}
