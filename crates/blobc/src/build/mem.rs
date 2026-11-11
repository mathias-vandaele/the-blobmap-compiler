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

/// Resident set size in bytes now.
pub fn rss() -> Option<u64> {
    status_bytes("VmRSS:")
}

/// Anonymous resident memory in bytes now: the resident set without mapped files' pages,
/// which the kernel can drop and read back.
pub fn anon() -> Option<u64> {
    status_bytes("RssAnon:")
}

/// Gives the allocator's free memory back to the system (glibc keeps it per thread arena
/// otherwise, and a later stage cannot reuse another arena's holes).
pub fn trim() {
    // SAFETY: malloc_trim only releases free heap memory; it has no preconditions.
    unsafe {
        libc::malloc_trim(0);
    }
}

/// Reset the high-water mark to the current RSS (no-op where unsupported).
pub fn reset_peak() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
}
