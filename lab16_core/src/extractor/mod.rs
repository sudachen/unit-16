use tracing::debug;
use crate::fb2scan::Fb2Scan;
use crate::gpu_llm::Model;
use anyhow::{Result, Context as _};

const EXTRACTOR_LLM_MODEL: (&str, &str) = (
    "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
    "Mistral-Nemo-Instruct-2407-Q5_K_M.gguf",
);

const EXTRACTOR_LLM_CONTEXT: u32 = 1024*20;
const EXTRACTOR_LLM_MAX_TOKENS: u32 = 4096;

mod prompts;
const EXTRACTOR_LLM_PROMPT: &str = prompts::prompt1::PROMPT;

enum Languages {
    Russian,
    English,
}

impl Languages {
    fn to_str(&self) -> &'static str {
        match self {
            Languages::Russian => "RUSSIAN",
            Languages::English => "ENGLISH",
        }
    }
}


pub struct Extractor {
    model: Option<Model>,
}

impl Extractor {

    pub fn new() -> Self {
        Self { model: None }
    }

    pub fn extract(&mut self, fb2: Fb2Scan) -> Result<()>{

        if self.model.is_none() {
            debug!("Initializing entity extractor model");
            self.model = Some(Model::new(&crate::gpu_llm::get_or_download_model(
                EXTRACTOR_LLM_MODEL.0,
                EXTRACTOR_LLM_MODEL.1)?)?)
        }

        debug!("Initializing entity extractor context");
        let mut context = self.model.as_ref().unwrap().context().
            with_n_batch(std::num::NonZeroU32::new(EXTRACTOR_LLM_CONTEXT).unwrap()).
            with_n_ctx(std::num::NonZeroU32::new(EXTRACTOR_LLM_CONTEXT).unwrap()).
            build()?;


        for section in fb2.sections() {
            if section.text().trim().len() < 42 { continue; }
            println!("Section: {}", section.title().unwrap_or("Untitled"));
            let lang = section.language().unwrap_or("TEXT LANGUAGE").to_uppercase();
            println!("Section language: {lang}");
            let mut prompt = String::from(prompts::PROMPT_PREFIX);
            prompt.push_str(prompts::prompt1::PROMPT);
            prompt.push_str(&section.text());
            prompt.push_str(prompts::PROMPT_SUFFIX);
            context.clear_kv_cache();
            let response_json = context.inference(&prompt, EXTRACTOR_LLM_MAX_TOKENS)?;
            println!("{}", response_json);
        }

        Ok(())
    }

    pub fn extract_from_file(&mut self, path: &std::path::PathBuf) -> Result<()>{
        let fb2 = Fb2Scan::from_file(path).expect("Failed to parse FB2");
        self.extract(fb2)
    }
}

