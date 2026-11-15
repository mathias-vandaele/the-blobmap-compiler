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

/// CPU time of this process so far (user and system, every thread), in seconds.
pub fn cpu_seconds() -> f64 {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let f: Vec<&str> = s.rsplit(')').next().unwrap_or("").split_whitespace().collect();
    let ticks = |k: usize| f.get(k).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0);
    // SAFETY: sysconf has no preconditions.
    let hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as f64;
    (ticks(11) + ticks(12)) / hz.max(1.0)
}

/// Physical cores (one per set of SMT siblings), from sysfs; `None` where it cannot be read.
pub fn physical_cores() -> Option<usize> {
    let mut cores = std::collections::BTreeSet::new();
    for e in std::fs::read_dir("/sys/devices/system/cpu").ok()?.flatten() {
        let name = e.file_name();
        let name = name.to_string_lossy();
        if name.strip_prefix("cpu").is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
            && let Ok(s) = std::fs::read_to_string(e.path().join("topology/core_cpus_list"))
        {
            cores.insert(s.trim().to_string());
        }
    }
    (!cores.is_empty()).then_some(cores.len())
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

/// Memory this process may still take without swapping, in bytes: the least of the system's
/// available memory and, under a cgroup limit (a `systemd-run` scope), the limit less this
/// process's anonymous memory, which the kernel cannot drop when the scope has no swap. None
/// where unknown.
pub fn free() -> Option<u64> {
    let m = std::fs::read_to_string("/proc/meminfo").ok()?;
    let avail = m.lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse::<u64>().ok()? * 1024;
    Some(match cgroup_limit() {
        Some(l) => avail.min(l.saturating_sub(anon()?)),
        None => avail,
    })
}

/// The tightest `memory.max` of this process's cgroup and its ancestors (cgroup v2).
fn cgroup_limit() -> Option<u64> {
    let c = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = c.lines().find_map(|l| l.strip_prefix("0::"))?;
    let root = std::path::Path::new("/sys/fs/cgroup");
    let mut dir = root.join(path.trim_start_matches('/'));
    let mut least: Option<u64> = None;
    loop {
        if let Some(x) = std::fs::read_to_string(dir.join("memory.max")).ok().and_then(|v| v.trim().parse::<u64>().ok()) {
            least = Some(least.map_or(x, |l| l.min(x)));
        }
        if dir == root || !dir.pop() {
            return least;
        }
    }
}
