//! The research CUDA kernels (`cu/research/sweep.cu`, unmodified), through their C API, as
//! the milestone 5 smoke test calls them: the trial's reference for speed.

use std::ffi::CStr;
use std::os::raw::c_char;

use anyhow::{Result, bail};

use crate::layered::{Entry, Layered};

#[repr(C)]
struct RawGpu {
    _p: [u8; 0],
}

unsafe extern "C" {
    fn cchgpu_last_error() -> *const c_char;
    fn cchgpu_create(n: u32, m: u32, levels: u32, level_first: *const u32, up_first: *const u32, up_head: *const u32, w_fwd: *const u16, w_rev: *const u16, ids: *const u32, top_levels: u32, max_cols: u32, out: *mut *mut RawGpu) -> i32;
    fn cchgpu_destroy(g: *mut RawGpu);
    fn cchgpu_set_top_resident(g: *mut RawGpu, on: i32) -> usize;
    fn cchgpu_sweep(g: *mut RawGpu, cols: u32, e: *const Entry, count: u32, reversed: i32, mode: i32, t: *mut f32) -> i32;
    fn cchgpu_download(g: *mut RawGpu, out: *mut u16) -> i32;
    fn cchgpu_mem_info(free: *mut usize, total: *mut usize) -> i32;
}

fn err() -> String {
    unsafe { CStr::from_ptr(cchgpu_last_error()).to_string_lossy().into_owned() }
}

/// Research schedules: one launch per wide level, or the same captured as a CUDA graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    PerLevel = 0,
    Graph = 1,
}

pub struct ResearchGpu {
    g: *mut RawGpu,
    n: usize,
    pub resident_smem: usize,
}

// The handle is used from one thread at a time (behind a mutex).
unsafe impl Send for ResearchGpu {}

impl ResearchGpu {
    pub fn new(c: &Layered, max_cols: u32) -> Result<ResearchGpu> {
        let ids: Vec<u32> = (0..c.n as u32).collect();
        let mut g: *mut RawGpu = std::ptr::null_mut();
        let rc = unsafe {
            cchgpu_create(c.n as u32, c.up_head.len() as u32, c.levels() as u32, c.level_first.as_ptr(), c.up_first.as_ptr(), c.up_head.as_ptr(), c.w_fwd.as_ptr(), c.w_rev.as_ptr(), ids.as_ptr(), c.top as u32, max_cols, &mut g)
        };
        if rc != 0 {
            bail!("research create: {rc} {}", err());
        }
        let resident_smem = unsafe { cchgpu_set_top_resident(g, 1) };
        Ok(ResearchGpu { g, n: c.n, resident_smem })
    }

    /// One sweep of `cols` columns from `entries`; returns the research's event timings
    /// (seed, top, wide, total) in ms. In `Graph`, top is 0 and wide is top and wide.
    pub fn sweep(&mut self, cols: u32, entries: &[Entry], reversed: bool, s: Schedule) -> Result<[f32; 4]> {
        let mut t = [0f32; 4];
        if unsafe { cchgpu_sweep(self.g, cols, entries.as_ptr(), entries.len() as u32, reversed as i32, s as i32, t.as_mut_ptr()) } != 0 {
            bail!("research sweep: {}", err());
        }
        Ok(t)
    }

    /// Column 0's labels, in new ids (requires one column per sweep).
    pub fn download(&mut self) -> Result<Vec<u16>> {
        let mut out = vec![0u16; self.n];
        if unsafe { cchgpu_download(self.g, out.as_mut_ptr()) } != 0 {
            bail!("research download: {}", err());
        }
        Ok(out)
    }
}

impl Drop for ResearchGpu {
    fn drop(&mut self) {
        unsafe { cchgpu_destroy(self.g) };
    }
}

/// Free and total device memory.
pub fn mem_info() -> Result<(usize, usize)> {
    let (mut f, mut t) = (0, 0);
    if unsafe { cchgpu_mem_info(&mut f, &mut t) } != 0 {
        bail!("mem info: {}", err());
    }
    Ok((f, t))
}
