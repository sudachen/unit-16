use std::ffi::CString;
use std::fmt::{Debug, Formatter};
use anyhow::{Result, Context as _};
use std::path::Path;
use std::ptr::NonNull;
use ik_llama_cpp;
use super::context::Builder;
pub struct Model {
    m: Option<NonNull<ik_llama_cpp::llama_model>>
}

impl Model {
    pub fn as_ptr(&self) -> Result<*const ik_llama_cpp::llama_model> {
        Ok(self.m.as_ref().map(|x| x.as_ptr()).context("Uninitialized model")?)
    }
    pub fn as_mut_ptr(&mut self) -> Result<*mut ik_llama_cpp::llama_model> {
        Ok(self.m.as_ref().map(|x| x.as_ptr()).context("Uninitialized model")?)
    }

    pub fn context(&mut self) -> Builder<'_> {
        Builder::new(self)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if let Some(model) = self.m.as_ref(){
            unsafe { ik_llama_cpp::llama_free_model(model.as_ptr()) }
        }
    }
}

pub struct ModelConfig {
    pub params: ik_llama_cpp::llama_model_params,
}

impl Debug for ModelConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelConfig")
            .field("n_gpu_layers", &self.params.n_gpu_layers)
            .field("main_gpu", &self.params.main_gpu)
            .field("vocab_only", &self.params.vocab_only)
            .field("use_mmap", &self.params.use_mmap)
            .field("use_mlock", &self.params.use_mlock)
            .finish()
    }
}


impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            params: unsafe { ik_llama_cpp::llama_model_default_params() }
        }
    }
}

impl ModelConfig {
    #[allow(dead_code)]
    pub fn with_n_gpu_layers(mut self, n_gpu_layers: u32) -> Self {
        let n_gpu_layers = i32::try_from(n_gpu_layers).unwrap_or(i32::MAX);
        self.params.n_gpu_layers = n_gpu_layers;
        self
    }
    #[allow(dead_code)]
    pub fn with_main_gpu(mut self, main_gpu: i32) -> Self {
        self.params.main_gpu = main_gpu;
        self
    }

    #[allow(dead_code)]
    pub fn with_vocab_only(mut self, vocab_only: bool) -> Self {
        self.params.vocab_only = vocab_only;
        self
    }

    #[allow(dead_code)]
    pub fn with_use_mmap(mut self, use_mmap: bool) -> Self {
        self.params.use_mmap = use_mmap;
        self
    }

    #[allow(dead_code)]
    pub fn with_use_mlock(mut self, use_mlock: bool) -> Self {
        self.params.use_mlock = use_mlock;
        self
    }

    #[allow(dead_code)]
    pub fn with_split_mode(mut self, split_mode: SplitMode) -> Self {
        self.params.split_mode = split_mode as ik_llama_cpp::llama_split_mode;
        self
    }

}

#[repr(i8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SplitMode {
    /// Single GPU
    None = ik_llama_cpp::LLAMA_SPLIT_MODE_NONE as i8,
    /// Split layers and KV across GPUs
    Layer = ik_llama_cpp::LLAMA_SPLIT_MODE_LAYER as i8,
    /// Split layers and KV across GPUs, use tensor parallelism if supported
    Row = ik_llama_cpp::LLAMA_SPLIT_MODE_ATTN as i8,
    /// Experimental tensor parallelism across GPUs
    Tensor = ik_llama_cpp::LLAMA_SPLIT_MODE_GRAPH as i8,
}

impl ModelConfig {
    pub fn load_fom_file(&self, model_path: impl AsRef<Path>) -> Result<Model> {
        super::backend::init()?;
        let path_ref = model_path.as_ref();
        let path = path_ref
            .to_str()
            .context("Failed to convert path to string")?;
        let cstr = CString::new(path)?;
        let llama_model =
            unsafe { ik_llama_cpp::llama_model_load_from_file(cstr.as_ptr(), self.params) };
        Ok(Model {
            m: Some(NonNull::new(llama_model).context("Failed to load model")?)
        })
    }
}
