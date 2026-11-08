//! Writes every tiny-tier scenario's inputs: `fixtures <out dir> [scenario …]`.

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(out) = args.first() else { anyhow::bail!("usage: fixtures <out dir> [scenario ...]") };
    for s in blobmap_fixtures::scenarios::all() {
        if args.len() > 1 && !args[1..].iter().any(|a| a == s.name) {
            continue;
        }
        let w = s.write(std::path::Path::new(out))?;
        println!("{}\t{}", s.name, w.pbf.display());
    }
    Ok(())
}
