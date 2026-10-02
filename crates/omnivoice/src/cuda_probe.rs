//! dlopen pre-probe for the CUDA runtime stack.
//!
//! cudarc is built with `dynamic-loading` (see candle's workspace
//! `Cargo.toml`): it dlopens the CUDA libraries on first use and *panics* —
//! rather than returning an error — when one is missing. A bare
//! `Device::new_cuda` on a machine without a full CUDA stack would therefore
//! abort the process instead of letting the caller fall back to CPU. The
//! probe mirrors cudarc's loader (same library names, same dlopen search
//! paths), so a failure here means cudarc would panic and the caller skips
//! CUDA up front. A false negative only costs a CPU fallback.

use libloading::Library;

/// Driver name group: at least one must load. `cuda` is the Linux/macOS
/// soname (and the Windows import library), `nvcuda` the Windows driver.
const DRIVER_GROUP: &[&str] = &["cuda", "nvcuda"];

/// Toolkit libraries cudarc loads for candle's feature set. cuDNN joins via
/// `required_groups()` when the `cudnn` feature pulls it into candle's conv
/// kernels.
const TOOLKIT_LIBS: &[&str] = &["cublas", "cublasLt", "nvrtc", "curand"];

fn required_groups() -> Vec<Vec<&'static str>> {
    let mut groups = vec![DRIVER_GROUP.to_vec(), TOOLKIT_LIBS.to_vec()];
    if cfg!(feature = "cudnn") {
        groups.push(vec!["cudnn"]);
    }
    groups
}

/// dlopen names on Unix: the unversioned form, the driver-style `.so.1`, and
/// the major-version sonames the toolkit ships (cuDNN 9, cuRAND 10, CUDA
/// 12/13-era cuBLAS/NVRTC). macOS dylib variants are not listed: CUDA on
/// macOS has no supported driver, so a cuda build there just stays on CPU.
fn unix_candidates(lib_name: &str) -> Vec<String> {
    let mut out = vec![format!("lib{lib_name}.so"), format!("lib{lib_name}.so.1")];
    for major in ["9", "10", "11", "12", "13"] {
        out.push(format!("lib{lib_name}.so.{major}"));
    }
    out
}

/// dlopen names on Windows: the plain DLL plus the toolkit's `64_<major>`
/// convention (`cublas64_12.dll`, `cudnn64_9.dll`).
fn windows_candidates(lib_name: &str) -> Vec<String> {
    let mut out = vec![format!("{lib_name}.dll")];
    for major in ["9", "10", "11", "12", "13"] {
        out.push(format!("{lib_name}64_{major}.dll"));
    }
    out
}

fn candidates(lib_name: &str) -> Vec<String> {
    if cfg!(target_os = "windows") {
        windows_candidates(lib_name)
    } else {
        unix_candidates(lib_name)
    }
}

// `Library::new` runs library constructors on dlopen, hence `unsafe` in the
// libloading releases in play here. The allow keeps this compiling should a
// future version re-mark it safe.
fn try_dlopen(name: &str) -> Result<Library, libloading::Error> {
    #[allow(unused_unsafe)]
    let lib = unsafe { Library::new(name) };
    lib
}

fn loadable(lib_name: &str) -> bool {
    candidates(lib_name).iter().any(|cand| try_dlopen(cand).is_ok())
}

/// `None` when every required library group loads through the normal dlopen
/// search paths (`LD_LIBRARY_PATH`, ldconfig cache, rpath); otherwise a
/// human-readable list of the missing groups.
pub(crate) fn cuda_stack_missing() -> Option<String> {
    let missing: Vec<String> = required_groups()
        .into_iter()
        .filter(|group| group.iter().all(|name| !loadable(name)))
        .map(|group| group.join("|"))
        .collect();
    if missing.is_empty() {
        None
    } else {
        Some(missing.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unix_candidates_cover_documented_sonames() {
        assert!(unix_candidates("cuda").contains(&"libcuda.so.1".to_string()));
        assert!(unix_candidates("cublas").contains(&"libcublas.so.12".to_string()));
        assert!(unix_candidates("cublasLt").contains(&"libcublasLt.so.12".to_string()));
        assert!(unix_candidates("nvrtc").contains(&"libnvrtc.so.12".to_string()));
        assert!(unix_candidates("curand").contains(&"libcurand.so.10".to_string()));
        assert!(unix_candidates("cudnn").contains(&"libcudnn.so.9".to_string()));
    }

    #[test]
    fn windows_candidates_cover_toolkit_convention() {
        assert!(windows_candidates("nvcuda").contains(&"nvcuda.dll".to_string()));
        assert!(windows_candidates("cublas").contains(&"cublas64_12.dll".to_string()));
        assert!(windows_candidates("cublasLt").contains(&"cublasLt64_12.dll".to_string()));
        assert!(windows_candidates("cudnn").contains(&"cudnn64_9.dll".to_string()));
    }

    #[test]
    fn driver_names_are_alternatives_not_requirements() {
        // `nvcuda` never exists on Unix; the group must still count as
        // present when `cuda` loads, so probe failures are per-group.
        let groups = required_groups();
        assert_eq!(groups[0], vec!["cuda", "nvcuda"]);
        assert!(groups.contains(&vec!["cublas", "cublasLt", "nvrtc", "curand"]));
    }
}
