use anyhow::Result;
use ik_llama_cpp;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::ffi::CStr;
use std::os::raw::{c_char, c_void, c_uint};


static LLAMA_BACKEND_INITIALIZED: AtomicBool = AtomicBool::new(false);

pub struct Backend;

pub fn init() -> Result<Backend> {
    if LLAMA_BACKEND_INITIALIZED
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        unsafe {
            ik_llama_cpp::llama_backend_init();
            ik_llama_cpp::llama_log_set(Some(llama_log_callback), std::ptr::null_mut());
        }
    }
    Ok(Backend {})
}

unsafe extern "C" fn llama_log_callback(
    level: c_uint,
    text: *const c_char,
    _user_data: *mut c_void,
) {
    if text.is_null() {
        return;
    }

    let c_str = unsafe { CStr::from_ptr(text) };
    let msg = c_str.to_string_lossy().trim_end().to_string();

    // Маппим уровни логов GGML на tracing
    match level {
        ik_llama_cpp::GGML_LOG_LEVEL_ERROR => tracing::error!("ik_llama.cpp: {}", msg),
        ik_llama_cpp::GGML_LOG_LEVEL_WARN => tracing::warn!("ik_llama.cpp: {}", msg),
        _ => tracing::trace!("ik_llama.cpp: {}", msg),
    }
}
