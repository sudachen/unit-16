use std::ffi::CStr;
use std::os::raw::{c_char, c_void, c_uint};

// Пустой колбэк, который просто выбрасывает все сообщения
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
        llama_cpp_sys_2::GGML_LOG_LEVEL_ERROR => tracing::error!("llama.cpp: {}", msg),
        llama_cpp_sys_2::GGML_LOG_LEVEL_WARN => tracing::warn!("llama.cpp: {}", msg),
        _ => tracing::trace!("llama.cpp: {}", msg),
    }
}

pub fn catch_traces() {
    unsafe {
        llama_cpp_sys_2::llama_log_set(Some(llama_log_callback), std::ptr::null_mut());
    }
}

