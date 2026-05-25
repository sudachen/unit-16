use hf_hub::api::sync::ApiBuilder;
use std::path::PathBuf;
use tracing::debug;

use crate::inference::{ModelConfig, Model};

pub struct ModelInfo {
    pub repo: &'static str,
    pub filename: &'static str,
}

impl ModelInfo {
    pub fn get_or_download(&self) -> anyhow::Result<PathBuf> {
        get_or_download_model(self.repo, self.filename)
    }

    pub fn load(&self, cfg :ModelConfig) -> anyhow::Result<Model>{
        let path = self.get_or_download()?;
        cfg.load_fom_file(path)
    }
}

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
