use std::ffi::CString;
use std::marker::PhantomData;
use std::num::{NonZeroI32, NonZeroU16};
use std::os::raw::c_int;
use anyhow::Result;

use super::model::Model;
use super::batch::Batch;
use tracing::{error};

use super::errors::{DecodeError,TokenToStringError};

pub struct Context<'a>{
    ctx: *mut ik_llama_cpp::llama_context,
    model_ptr: *const ik_llama_cpp::llama_model,
    vocab_ptr: *const ik_llama_cpp::llama_vocab,
    n_vocab: usize,
    initialized_logits: Vec<i32>,
    //embeddings_enabled: bool,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    _lifetime: PhantomData<&'a Model>,
}

impl<'a> Context<'a> {
    pub fn clear_kv_cache(&mut self) {
        unsafe { ik_llama_cpp::llama_kv_cache_clear(self.ctx) };
    }

    pub fn decode(&mut self, batch: &mut Batch) -> std::result::Result<(), DecodeError> {
        let result =
            unsafe { ik_llama_cpp::llama_decode(self.ctx, batch.llama_batch) };

        match NonZeroI32::new(result) {
            None => {
                self.initialized_logits
                    .clone_from(&batch.initialized_logits);
                Ok(())
            }
            Some(error) => Err(DecodeError::from(error)),
        }
    }


    pub fn inference(&mut self, prompt: &str, max_tokens: u32) -> Result<String> {
        let tokens_list = self.str_to_token(prompt, true, true)?;
        let mut batch = Batch::new(tokens_list.len(), 1);

        let last_index = (tokens_list.len() - 1) as i32;
        for (i, token) in tokens_list.iter().enumerate() {
            batch.add(*token, i as i32, &[0], i as i32 == last_index)?;
        }

        self.decode(&mut batch)?;
        let mut n_cur = tokens_list.len() as i32;
        let mut response_bytes: Vec<u8> = Vec::new();

        for _ in 0..max_tokens {
            let token = self.sample(batch.n_tokens()-1)?;
            if unsafe {ik_llama_cpp::llama_vocab_is_eog(self.vocab_ptr, token)} {
                break;
            }

            match self.token_to_piece_bytes(token, 64, false, None) {
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
            batch.add(token, n_cur, &[0], true)?;
            self.decode(&mut batch)?;
            n_cur += 1;
        }

        Ok(String::from_utf8_lossy(&response_bytes).into_owned())
    }

    pub fn sample(&mut self, logit_idx: i32) -> Result<ik_llama_cpp::llama_token> {
        unsafe {
            let logits_ptr = ik_llama_cpp::llama_get_logits_ith(self.ctx, logit_idx);
            let mut candidates_vec = Vec::with_capacity(self.n_vocab);
            for id in 0..self.n_vocab {
                candidates_vec.push(ik_llama_cpp::llama_token_data {
                    id: id as i32,
                    logit: *logits_ptr.add(id),
                    p: 0.0,
                });
            }

            let mut candidates_array = ik_llama_cpp::llama_token_data_array {
                data: candidates_vec.as_mut_ptr(),
                size: self.n_vocab,
                selected: 0,
                sorted: false,
            };

            /*if !self.prev_tokens.is_empty() && self.penalty_repeat > 1.0 {
                // Берем последние 64 токена из истории (или сколько есть)
                let last_n = std::cmp::min(self.prev_tokens.len(), 64);
                let start_ptr = self.prev_tokens.as_ptr().add(self.prev_tokens.len() - last_n);

                llama_sample_repetition_penalties(
                    ctx,
                    &mut candidates_array,
                    start_ptr,
                    last_n,
                    self.penalty_repeat,
                    0.0, // frequency penalty
                    0.0, // presence penalty
                );
            }*/

            let token_id = if self.temperature <= 0.0 {
                ik_llama_cpp::llama_sample_token_greedy(self.ctx, &mut candidates_array)
            } else {
                ik_llama_cpp::llama_sample_top_k(self.ctx, &mut candidates_array, self.top_k, 1);
                ik_llama_cpp::llama_sample_top_p(self.ctx, &mut candidates_array, self.top_p, 1);
                ik_llama_cpp::llama_sample_temp(self.ctx, &mut candidates_array, self.temperature);
                ik_llama_cpp::llama_sample_token(self.ctx, &mut candidates_array)
            };

            //self.prev_tokens.push(token_id);
            Ok(token_id)
        }
    }

    pub fn token_to_piece_bytes(
        &self,
        token: ik_llama_cpp::llama_token,
        buffer_size: usize,
        special: bool,
        lstrip: Option<NonZeroU16>,
    ) -> std::result::Result<Vec<u8>, TokenToStringError> {
        let string = CString::new(vec![b'*'; buffer_size]).expect("no null");
        let len = string.as_bytes().len();
        let len = c_int::try_from(len).expect("length fits into c_int");
        let buf = string.into_raw();
        let lstrip = lstrip.map_or(0, |it| i32::from(it.get()));
        let size = unsafe {
            ik_llama_cpp::llama_token_to_piece(
                self.model_ptr,
                token,
                buf,
                len,
                lstrip,
                special,
            )
        };

        match size {
            0 => Err(TokenToStringError::UnknownTokenType),
            i if i.is_negative() => Err(TokenToStringError::InsufficientBufferSpace(i)),
            size => {
                let string = unsafe { CString::from_raw(buf) };
                let mut bytes = string.into_bytes();
                let len = usize::try_from(size).expect("size is positive and fits into usize");
                bytes.truncate(len);
                Ok(bytes)
            }
        }
    }

    pub fn str_to_token(
        &self,
        str: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<ik_llama_cpp::llama_token>> {
        let add_bos = match add_special {
            true => 1,
            _ => 0
        };
        let tokens_estimation = std::cmp::max(8, (str.len() / 2) + add_bos);
        let mut buffer: Vec<ik_llama_cpp::llama_token> = Vec::with_capacity(tokens_estimation);
        let c_string = CString::new(str)?;
        let buffer_capacity =
            c_int::try_from(buffer.capacity()).expect("buffer capacity should fit into a c_int");

        let size = unsafe {
            ik_llama_cpp::llama_tokenize(
                self.model_ptr,
                c_string.as_ptr(),
                c_int::try_from(c_string.as_bytes().len())?,
                buffer.as_mut_ptr().cast::<ik_llama_cpp::llama_token>(),
                buffer_capacity,
                add_special,
                parse_special,
            )
        };

        // if we fail the first time we can resize the vector to the correct size and try again. This should never fail.
        // as a result - size is guaranteed to be positive here.
        let size = if size.is_negative() {
            buffer.reserve_exact(usize::try_from(-size).expect("usize's are larger "));
            unsafe {
                ik_llama_cpp::llama_tokenize(
                    self.model_ptr,
                    c_string.as_ptr(),
                    c_int::try_from(c_string.as_bytes().len())?,
                    buffer.as_mut_ptr().cast::<ik_llama_cpp::llama_token>(),
                    -size,
                    add_special,
                    parse_special,
                )
            }
        } else {
            size
        };

        let size = usize::try_from(size).expect("size is positive and usize ");

        // Safety: `size` < `capacity` and llama-cpp has initialized elements up to `size`
        unsafe { buffer.set_len(size) }
        Ok(buffer)
    }
}

pub struct Builder<'a> {
    model: &'a mut Model,
    params: ik_llama_cpp::llama_context_params,
}

impl<'a> Builder<'a> {
    pub fn new(model: &'a mut Model) -> Self {
        Self {
            model,
            params: unsafe { ik_llama_cpp::llama_context_default_params() },
        }
    }

    pub fn with_n_ctx(mut self, n_ctx:u32) -> Self {
        self.params.n_ctx = n_ctx;
        self
    }

    pub fn with_n_batch(mut self, n_batch: u32) -> Self {
        self.params.n_batch = n_batch;
        self
    }

    pub fn build(&mut self) -> Result<Context<'a>> {
        let ctx = unsafe {
            ik_llama_cpp::llama_init_from_model(self.model.as_mut_ptr()?, self.params)
        };
        let model_ptr = self.model.as_ptr()?;
        Ok(Context{
            ctx,
            model_ptr,
            n_vocab: unsafe { ik_llama_cpp::llama_n_vocab(model_ptr) } as usize,
            vocab_ptr: unsafe { ik_llama_cpp::llama_model_get_vocab(model_ptr) },
            initialized_logits: vec![],
            //embeddings_enabled: false,
            temperature: 0.0,
            top_k: 0,
            top_p: 0.0,
            _lifetime: Default::default(),
        })
    }
}
