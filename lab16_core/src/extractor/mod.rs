#![allow(unused_imports)]
use crate::fb2scan::Fb2Scan;
use crate::inference::{Model,ModelConfig,KVType};
use crate::hf::ModelInfo;

use anyhow::{Context as _, Result};
use tracing::debug;
mod prompts;


const EXTRACTOR_LLM_MODEL: ModelInfo = ModelInfo {
    repo: "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
    filename: "Mistral-Nemo-Instruct-2407-Q5_K_M.gguf",
};

const EXTRACTOR_LLM_CONTEXT: u32 = 16384;
const EXTRACTOR_LLM_MAX_TOKENS: u32 = 4096;

pub struct Extractor {
    model: Model,
}

impl Extractor {
    pub fn new() -> Result<Self> {
        debug!("Initializing extractor model");
        let config = ModelConfig::default().with_n_gpu_layers(1000);
        let model = EXTRACTOR_LLM_MODEL.load(config)?;
        Ok(Self { model })
    }

    pub fn extract(&mut self, fb2: Fb2Scan) -> Result<()> {
        debug!("Initializing entity extractor context");
        let mut context =
            self.model
            .context()
            .with_n_ctx(EXTRACTOR_LLM_CONTEXT)
            .with_k_cache_hadamard()
            .with_type_kv(KVType::Q4_0, KVType::Q5_0)
            .build()?;

        for section in fb2.sections() {
            if section.text().trim().len() < 42 {
                continue;
            }
            println!("Section: {}", section.title().unwrap_or("Untitled"));
            let lang = section.language().unwrap_or("TEXT LANGUAGE").to_uppercase();
            println!("Section language: {lang}");
            let mut prompt = String::from(prompts::prompt1::PROMPT_BEGIN);
            prompt.push_str(&section.text());
            prompt.push_str(prompts::prompt1::PROMPT_END);
            context.clear_kv_cache();
            let response_json = context.inference(&prompt, EXTRACTOR_LLM_MAX_TOKENS)?;
            println!("{}", response_json);
        }

        Ok(())
    }

    pub fn extract_from_file(&mut self, path: &std::path::PathBuf) -> Result<()> {
        let fb2 = Fb2Scan::from_file(path).expect("Failed to parse FB2");
        self.extract(fb2)
    }
}
