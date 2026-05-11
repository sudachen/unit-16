mod scanner;
mod extractor;

const ENTITY_EXTRACTOR_LLM_MODEL: (&str, &str) = (
    "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
    "Mistral-Nemo-Instruct-2407-Q5_K_M.gguf",
);

pub use extractor::*;