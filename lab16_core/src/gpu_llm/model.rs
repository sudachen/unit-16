use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use std::path::Path;

pub struct Model {
    model: LlamaModel,
}

impl Model {
    pub fn new(model_path: &Path) -> anyhow::Result<Self> {
        let backend = super::get_backend()?;

        // Setting parameters for 16GB VRAM
        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(1000); // 1000 means "offload everything to GPU"

        let model = LlamaModel::load_from_file(backend, model_path, &model_params)?;

        Ok(Self { model })
    }

    pub fn context(&self) -> super::context::Builder<'_> {
        super::context::Builder::new(&self.model)
    }
}
