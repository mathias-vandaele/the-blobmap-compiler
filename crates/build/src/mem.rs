//! Peak resident set size of this process (Linux: `VmHWM` of `/proc/self/status`; the
//! high-water mark is reset by writing 5 to `/proc/self/clear_refs`).

fn status_bytes(key: &str) -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with(key))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// Peak RSS in bytes since the process start or the last `reset_peak`.
pub fn peak_rss() -> Option<u64> {
    status_bytes("VmHWM:")
}

/// Reset the high-water mark to the current RSS (no-op where unsupported).
pub fn reset_peak() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
}
