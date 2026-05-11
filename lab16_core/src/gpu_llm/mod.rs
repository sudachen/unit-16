mod model;
mod context;
mod llama_traces;

use hf_hub::api::sync::ApiBuilder;
use std::path::PathBuf;
use std::sync::OnceLock;
use llama_cpp_2::llama_backend::LlamaBackend;
use tracing::debug;

pub use model::Model;
pub use context::Context;

pub fn get_or_download_model(repo: &str, filename: &str) -> anyhow::Result<PathBuf> {
    // This will use default HF cache directory (~/.cache/huggingface/hub)
    let api = ApiBuilder::new()
        .with_progress(true) // Enable progress bars automatically
        .build()?;

    let api_repo = api.model(repo.to_string());

    debug!("Checking for model: {} in repo: {}...", filename, repo);

    // If it's already in the cache, it returns the local path instantly.
    // If not, it starts the download.
    let model_path = api_repo.get(filename)?;

    debug!("Model ready at: {:?}", model_path);
    Ok(model_path)
}

static LLAMA_BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

pub fn get_backend() -> anyhow::Result<&'static LlamaBackend> {
    if let Some(b) = LLAMA_BACKEND.get() {
        return Ok(b);
    }
    let backend = LlamaBackend::init()?;
    llama_traces::catch_traces();
    Ok(LLAMA_BACKEND.get_or_init(|| backend))
}
