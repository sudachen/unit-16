use anyhow::Result;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use tracing::{error};

pub struct Context<'a>(LlamaContext<'a>);

impl<'a> Context<'a> {
    pub fn clear_kv_cache(&mut self) {
        self.0.clear_kv_cache();
    }
    pub fn inference(&mut self, prompt: &str, max_tokens: u32) -> Result<String> {
        let model = self.0.model;

        let tokens_list = model.str_to_token(prompt, AddBos::Always)?;
        let mut batch = LlamaBatch::new(tokens_list.len(), 1);

        let last_index = (tokens_list.len() - 1) as i32;
        for (i, token) in tokens_list.iter().enumerate() {
            batch.add(*token, i as i32, &[0], i as i32 == last_index)?;
        }

        self.0.decode(&mut batch)?;

        let mut sampler = LlamaSampler::greedy();
        let mut n_cur = tokens_list.len() as i32;

        let mut response_bytes: Vec<u8> = Vec::new();

        for _ in 0..max_tokens {
            let token_id = sampler.sample(&self.0, batch.n_tokens() - 1);
            if token_id == self.0.model.token_eos() {
                break;
            }

            match model.token_to_piece_bytes(token_id, 64, false, None) {
                Ok(piece_bytes) => {
                    response_bytes.extend(&piece_bytes);
                    //let piece_text = String::from_utf8_lossy(&piece_bytes);
                }
                Err(e) => {
                    error!("Failed to extract piece bytes: {:?}", e);
                    break;
                }
            }

            batch.clear();
            batch.add(token_id, n_cur, &[0], true)?;
            self.0.decode(&mut batch)?;
            n_cur += 1;
        }

        Ok(String::from_utf8_lossy(&response_bytes).into_owned())
    }
}

#[derive(Default)]
pub struct Params {
    pub n_ctx: Option<std::num::NonZeroU32>,
    pub n_batch: Option<std::num::NonZeroU32>,
}

pub struct Builder<'a> {
    model: &'a LlamaModel,
    params: Params,
}

impl<'a> Builder<'a> {
    pub fn new(model: &'a LlamaModel) -> Self {
        Self {
            model,
            params: Params::default(),
        }
    }

    pub fn with_n_ctx(mut self, n_ctx: std::num::NonZeroU32) -> Self {
        self.params.n_ctx = Some(n_ctx);
        self
    }

    pub fn with_n_batch(mut self, n_batch: std::num::NonZeroU32) -> Self {
        self.params.n_batch = Some(n_batch);
        self
    }

    pub fn build(&self) -> Result<Context<'a>> {
        let backend = super::get_backend()?;
        let mut ctx_params = LlamaContextParams::default();
        if let Some(n_ctx) = self.params.n_ctx {
            ctx_params = ctx_params.with_n_ctx(Some(n_ctx));
        }
        if let Some(n_batch) = self.params.n_batch {
            ctx_params = ctx_params.with_n_batch(n_batch.get());
        }
        let ctx = self.model.new_context(backend, ctx_params)?;
        Ok(Context(ctx))
    }
}
