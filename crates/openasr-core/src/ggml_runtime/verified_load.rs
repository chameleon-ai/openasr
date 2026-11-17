//! Rust-side verified loading for ggml backend plugins.
//!
//! Stock ggml 1.0.0 has no verified-load API (the fork's
//! `ggml_backend_load_verified_*` is gone). The activation policy is
//! unchanged: a module is only ever loaded after its signed file identity was
//! verified against the signed bundle manifest (bundled CPU rescue modules)
//! or the signed catalog + installed-pack record (optional accelerator
//! packs), and openasr never calls the stock `ggml_backend_load*` family.
//! This module performs the load itself, initializes the module's
//! `ggml_backend_init`, checks basic sanity on the returned
//! `ggml_backend_reg_t`, and registers it with the stock registry via
//! `ggml_backend_register`.
//!
//! What is intentionally NOT reproduced:
//!
//! * No in-plugin attestation symbols are read. Stock plugins do not export
//!   them, and the signed-file-identity check performed before load is the
//!   single authority for "is this exact module".
//! * No synthetic "live hardware" is fabricated. Stock ggml cannot report a
//!   device's PCI target (sm_86) or its driver version to Rust. The
//!   qualification probes are therefore unavailable on stock ggml and return
//!   [`VerifiedLoadError::LiveProbeUnavailable`]. Consequence: optional
//!   downloadable GPU packs cannot be qualified or freshly activated until a
//!   hardware attestation story exists. Activation of an already persisted,
//!   exact signed record remains possible because its target/driver binding
//!   is carried in the record, not probed.
//! * TOCTOU between the file-identity check and the load is not worse than
//!   upstream's: both happen inside one activation critical section after the
//!   plugin files are write/delete-locked by the caller. A local attacker
//!   able to swap files under a running loader already owns the process.
//!
//! Module lifetime: on the success path the module handle intentionally
//! leaks (the registry owns the module for the lifetime of the process,
//! matching upstream's plugin contract). On any failure after
//! `ggml_backend_init` has run, the handle also leaks rather than unloading
//! a module whose backends may exist; `ggml_backend_unload` cannot be used
//! for a backend that was never registered.

use std::{
    ffi::{CStr, CString, c_int, c_void},
    ptr::NonNull,
};

use thiserror::Error;

use super::ffi;

#[derive(Debug, Error)]
pub(crate) enum VerifiedLoadError {
    #[error("verified load: {provider} plugin failed its pre-load sanity check (score {score})")]
    ScoreRejected { provider: String, score: i32 },
    #[error(
        "verified load: {provider} plugin does not export ggml_backend_init or it returned null"
    )]
    InitFailed { provider: String },
    #[error("verified load: {provider} plugin resolved to unexpected backend '{actual}'")]
    ProviderMismatch { provider: String, actual: String },
    #[error("verified load: {provider} plugin load failed: {detail}")]
    LoadFailed { provider: String, detail: String },
}

#[cfg(target_os = "windows")]
type ModuleHandle = *mut c_void;
#[cfg(not(target_os = "windows"))]
type ModuleHandle = *mut c_void;

/// Load `path` with the same restricted search policy the fork used: the
/// plugin's own directory, the explicitly verified dependency directories,
/// and System32 / the default loader search path. The application directory
/// is deliberately not consulted.
fn module_path(
    path: &CStr,
    dependency_dirs: &[CString],
    provider: ProviderKind,
) -> Result<ModuleHandle, VerifiedLoadError> {
    #[cfg(target_os = "windows")]
    {
        use windows_sys::Win32::System::LibraryLoader::{
            AddDllDirectory, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32,
            LOAD_LIBRARY_SEARCH_USER_DIRS, LoadLibraryExW, RemoveDllDirectory,
        };
        let to_wide = |value: &CStr| -> Result<Vec<u16>, VerifiedLoadError> {
            value
                .to_str()
                .map(|path| path.encode_utf16().chain(std::iter::once(0)).collect())
                .map_err(|_| VerifiedLoadError::LoadFailed {
                    provider: provider.as_str().to_string(),
                    detail: "path is not valid UTF-8".to_string(),
                })
        };
        let wide = to_wide(path)?;
        let mut cookies = Vec::with_capacity(dependency_dirs.len());
        let mut setup_ok = true;
        for dir in dependency_dirs {
            let wide_dir = to_wide(dir)?;
            // SAFETY: `wide_dir` is a null-terminated UTF-16 path.
            let cookie = unsafe { AddDllDirectory(wide_dir.as_ptr()) };
            if cookie.is_null() {
                setup_ok = false;
                break;
            }
            cookies.push(cookie);
        }
        let handle = if setup_ok {
            let flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
                | LOAD_LIBRARY_SEARCH_SYSTEM32
                | LOAD_LIBRARY_SEARCH_USER_DIRS;
            // SAFETY: `wide` is a null-terminated UTF-16 path; the flags
            // select the restricted search path.
            unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), flags) }
        } else {
            std::ptr::null_mut()
        };
        for cookie in cookies.into_iter().rev() {
            // SAFETY: cookie was returned by AddDllDirectory above.
            unsafe { RemoveDllDirectory(cookie) };
        }
        if handle.is_null() {
            let detail = if setup_ok {
                format!("LoadLibraryExW failed with GetLastError={}", unsafe {
                    windows_sys::Win32::Foundation::GetLastError()
                })
            } else {
                "failed to register a verified dependency directory".to_string()
            };
            return Err(VerifiedLoadError::LoadFailed {
                provider: provider.as_str().to_string(),
                detail,
            });
        }
        Ok(handle)
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = dependency_dirs;
        // SAFETY: `path` is a valid NUL-terminated C string; RTLD_LOCAL keeps
        // the module's symbols out of the global namespace; RTLD_NOW fails
        // the load eagerly on unresolved imports.
        let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            let detail = unsafe { CStr::from_ptr(libc::dlerror()) }
                .to_string_lossy()
                .into_owned();
            return Err(VerifiedLoadError::LoadFailed {
                provider: provider.as_str().to_string(),
                detail,
            });
        }
        Ok(handle)
    }
}

#[cfg(target_os = "windows")]
fn module_symbol(handle: ModuleHandle, name: &CStr) -> Option<*mut c_void> {
    // SAFETY: `handle` is a live module handle returned by LoadLibraryExW.
    unsafe {
        windows_sys::Win32::System::LibraryLoader::GetProcAddress(handle, name.as_ptr().cast())
    }
    .map(|proc| proc as *mut c_void)
}

#[cfg(not(target_os = "windows"))]
fn module_symbol(handle: ModuleHandle, name: &CStr) -> Option<*mut c_void> {
    // SAFETY: `handle` is a live module handle returned by dlopen.
    let symbol = unsafe { libc::dlsym(handle, name.as_ptr()) };
    if symbol.is_null() { None } else { Some(symbol) }
}

fn module_close(handle: ModuleHandle) {
    #[cfg(target_os = "windows")]
    // SAFETY: `handle` is a live module handle owned by this call and no
    // backend was registered from it.
    unsafe {
        windows_sys::Win32::Foundation::FreeLibrary(handle.cast())
    };
    #[cfg(not(target_os = "windows"))]
    // SAFETY: `handle` is a live module handle owned by this call and no
    // backend was registered from it.
    unsafe {
        libc::dlclose(handle)
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ProviderKind {
    Cpu,
    Cuda,
    Hip,
    Vulkan,
}

impl ProviderKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Hip => "hip",
            Self::Vulkan => "vulkan",
        }
    }

    fn expected_name_fragments(self) -> &'static [&'static str] {
        match self {
            Self::Cpu => &["cpu"],
            Self::Cuda => &["cuda"],
            Self::Hip => &["hip", "rocm"],
            Self::Vulkan => &["vulkan"],
        }
    }
}

struct Candidate {
    path: CString,
    score: i32,
}

fn score_candidate(
    path: &CStr,
    provider: ProviderKind,
    dependency_dirs: &[CString],
) -> Result<Candidate, VerifiedLoadError> {
    let handle = module_path(path, dependency_dirs, provider)?;
    let mut score = 1i32;
    if let Some(symbol) = module_symbol(handle, c"ggml_backend_score") {
        // SAFETY: ggml_backend_score is `int (*)(void)` in every ggml plugin.
        let score_fn: unsafe extern "C" fn() -> c_int = unsafe { std::mem::transmute(symbol) };
        // SAFETY: the plugin owns this entry point; it performs no
        // registration and is called before any backend exists.
        score = unsafe { score_fn() };
    }
    // Close the scoring handle: no registry entry references it yet.
    module_close(handle);
    if score <= 0 {
        return Err(VerifiedLoadError::ScoreRejected {
            provider: provider.as_str().to_string(),
            score,
        });
    }
    Ok(Candidate {
        path: path.to_owned(),
        score,
    })
}

/// Load the best-scoring CPU rescue module from a caller-verified absolute
/// path list. Every entry must already have passed the bundle's signed
/// file-identity verification; this function only establishes that the
/// module is a loadable, self-initializing ggml backend of the expected
/// kind and registers it.
pub(crate) fn load_best_verified(
    paths: &[CString],
    provider: ProviderKind,
) -> Result<(), VerifiedLoadError> {
    let mut scored = Vec::with_capacity(paths.len());
    for path in paths {
        match score_candidate(path, provider, &[]) {
            Ok(candidate) => scored.push(candidate),
            Err(VerifiedLoadError::ScoreRejected { .. }) => continue,
            Err(error) => return Err(error),
        }
    }
    scored.sort_by_key(|candidate| std::cmp::Reverse(candidate.score));
    let Some(best) = scored.into_iter().next() else {
        return Err(VerifiedLoadError::LoadFailed {
            provider: provider.as_str().to_string(),
            detail: "no candidate plugin scored above zero".to_string(),
        });
    };
    load_and_register(&best.path, provider, &[])
}

/// Load one exact plugin from a caller-verified absolute path. Used for the
/// optional accelerator pack lane; the caller has already re-verified the
/// installed pack (catalog binding, file inventory, PE image hashes, load
/// lock) before this function runs.
pub(crate) fn load_exact_verified(
    path: &CStr,
    provider: ProviderKind,
    dependency_dirs: &[CString],
) -> Result<(), VerifiedLoadError> {
    load_and_register(path, provider, dependency_dirs)
}

fn load_and_register(
    path: &CStr,
    provider: ProviderKind,
    dependency_dirs: &[CString],
) -> Result<(), VerifiedLoadError> {
    let handle = module_path(path, dependency_dirs, provider)?;
    let Some(init_ptr) = module_symbol(handle, c"ggml_backend_init") else {
        return Err(VerifiedLoadError::InitFailed {
            provider: provider.as_str().to_string(),
        });
    };
    // SAFETY: ggml_backend_init is `ggml_backend_reg_t (*)(void)`; a foreign
    // or broken module is rejected by the name check below.
    let init_fn: unsafe extern "C" fn() -> ffi::GgmlBackendRegRaw =
        unsafe { std::mem::transmute(init_ptr) };
    let reg = unsafe { init_fn() };
    let Some(reg) = NonNull::new(reg) else {
        // Leak the module handle rather than unload a module whose
        // backend_init ran (see module-lifetime note).
        return Err(VerifiedLoadError::InitFailed {
            provider: provider.as_str().to_string(),
        });
    };
    let name_ptr = unsafe { ffi::ggml_backend_reg_name(reg.as_ptr()) };
    let actual = if name_ptr.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(name_ptr) }
            .to_string_lossy()
            .into_owned()
    };
    let normalized = actual.to_ascii_lowercase();
    let matches = provider
        .expected_name_fragments()
        .iter()
        .any(|fragment| normalized.contains(fragment));
    if !matches {
        // Registered nowhere yet; the instance is orphaned but unreachable
        // from the registry and from every openasr activation route. Leak
        // the handle rather than risk unloading a live backend.
        return Err(VerifiedLoadError::ProviderMismatch {
            provider: provider.as_str().to_string(),
            actual,
        });
    }
    // Register with the stock registry. The handle intentionally leaks: the
    // registry owns the module for the lifetime of the process, matching
    // upstream's plugin lifetime contract.
    unsafe { ffi::ggml_backend_register(reg.as_ptr()) };
    Ok(())
}
