//! Records the compiler's git commit (40 hex) and whether the working tree is dirty, for
//! the manifest's `compiler` (FORMAT §3); zeros and dirty outside a git checkout.

fn main() {
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
    };
    let commit = git(&["rev-parse", "HEAD"]).map(|s| s.trim().to_string()).unwrap_or_else(|| "0".repeat(40));
    let dirty = git(&["status", "--porcelain", "--untracked-files=no"]).is_none_or(|s| !s.trim().is_empty());
    println!("cargo:rustc-env=BLOBC_COMMIT={commit}");
    println!("cargo:rustc-env=BLOBC_DIRTY={dirty}");
    for p in ["../../.git/HEAD", "../../.git/index", "../../crates", "../../config", "../../SPEC.md", "../../Cargo.toml"] {
        println!("cargo:rerun-if-changed={p}");
    }
}
