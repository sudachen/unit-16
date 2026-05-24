use std::marker::PhantomData;
use super::errors::BatchAddError;

/// A safe wrapper around `llama_batch`.
#[derive(Debug)]
pub struct Batch<'a> {
    /// The number of tokens the batch was allocated with. they are safe to write to - but not necessarily read from as they are not necessarily initialized
    allocated: usize,
    /// The logits that are initialized. Used by [`Context`] to ensure that only initialized logits are accessed.
    pub(crate) initialized_logits: Vec<ik_llama_cpp::llama_token>,
    #[allow(clippy::doc_markdown)]
    /// The llama_cpp batch. always initialize by `llama_batch_init(allocated, <unknown>, <unknown>)`
    pub(crate) llama_batch: ik_llama_cpp::llama_batch,
    phantom: PhantomData<&'a [ik_llama_cpp::llama_token]>,
}

impl<'a> Batch<'a> {
    pub fn clear(&mut self) {
        self.llama_batch.n_tokens = 0;
        self.initialized_logits.clear();
    }

    pub fn add(
        &mut self,
        token: ik_llama_cpp::llama_token,
        pos: ik_llama_cpp::llama_pos,
        seq_ids: &[i32],
        logits: bool,
    ) -> core::result::Result<(), BatchAddError> {
        if self.allocated
            < usize::try_from(self.n_tokens() + 1).expect("cannot fit n_tokens into a usize")
        {
            return Err(BatchAddError::InsufficientSpace(self.allocated));
        }
        let offset = self.llama_batch.n_tokens;
        let offset_usize = usize::try_from(offset).expect("cannot fit n_tokens into a usize");
        unsafe {
            // batch.token   [batch.n_tokens] = id;
            self.llama_batch.token.add(offset_usize).write(token);
            // batch.pos     [batch.n_tokens] = pos,
            self.llama_batch.pos.add(offset_usize).write(pos);
            // batch.n_seq_id[batch.n_tokens] = seq_ids.size();
            self.llama_batch.n_seq_id.add(offset_usize).write(
                ik_llama_cpp::llama_seq_id::try_from(seq_ids.len())
                    .expect("cannot fit seq_ids.len() into a llama_seq_id"),
            );
            // for (size_t i = 0; i < seq_ids.size(); ++i) {
            //     batch.seq_id[batch.n_tokens][i] = seq_ids[i];
            // }
            for (i, seq_id) in seq_ids.iter().enumerate() {
                let tmp = *self.llama_batch.seq_id.add(offset_usize);
                tmp.add(i).write(*seq_id);
            }
            // batch.logits  [batch.n_tokens] = logits;
            self.llama_batch
                .logits
                .add(offset_usize)
                .write(i8::from(logits));
        }

        if logits {
            self.initialized_logits.push(offset);
        } else {
            self.initialized_logits.retain(|l| l != &offset);
        }

        // batch.n_tokens++;
        self.llama_batch.n_tokens += 1;

        Ok(())
    }

    pub fn add_sequence(
        &mut self,
        tokens: &[ik_llama_cpp::llama_token],
        seq_id: i32,
        logits_all: bool,
    ) -> Result<(), BatchAddError> {
        let n_tokens_0 =
            usize::try_from(self.llama_batch.n_tokens).expect("cannot fit n_tokens into a usize");
        let n_tokens = tokens.len();

        if self.allocated < n_tokens_0 + n_tokens {
            return Err(BatchAddError::InsufficientSpace(self.allocated));
        }

        let last_index = ik_llama_cpp::llama_pos::try_from(n_tokens.saturating_sub(1))
            .expect("cannot fit n_tokens into a llama_pos");
        for (i, token) in (0..).zip(tokens.iter()) {
            self.add(*token, i, &[seq_id], logits_all || i == last_index)?;
        }

        Ok(())
    }

    #[must_use]
    pub fn new(n_tokens: usize, n_seq_max: i32) -> Self {
        let n_tokens_i32 = i32::try_from(n_tokens).expect("cannot fit n_tokens into a i32");
        let batch = unsafe { ik_llama_cpp::llama_batch_init(n_tokens_i32, 0, n_seq_max) };

        Batch {
            allocated: n_tokens,
            initialized_logits: vec![],
            llama_batch: batch,
            phantom: PhantomData,
        }
    }

    pub fn get_one(tokens: &'a [ik_llama_cpp::llama_token],pos: ik_llama_cpp::llama_pos, seq_id: i32) -> Result<Self, BatchAddError> {
        if tokens.is_empty() {
            return Err(BatchAddError::EmptyBuffer);
        }
        let batch = unsafe {
            let ptr = tokens.as_ptr() as *mut i32;
            ik_llama_cpp::llama_batch_get_one(
                ptr,
                tokens
                    .len()
                    .try_into()
                    .expect("number of tokens exceeds i32::MAX"),
                pos,
                seq_id,

            )
        };
        let batch = Self {
            allocated: 0,
            initialized_logits: vec![(tokens.len() - 1)
                .try_into()
                .expect("number of tokens exceeds i32::MAX + 1")],
            llama_batch: batch,
            phantom: PhantomData,
        };
        Ok(batch)
    }

    /// Returns the number of tokens in the batch.
    #[must_use]
    pub fn n_tokens(&self) -> i32 {
        self.llama_batch.n_tokens
    }
}

impl<'a> Drop for Batch<'a> {
    fn drop(&mut self) {
        unsafe {
            if self.allocated > 0 {
                ik_llama_cpp::llama_batch_free(self.llama_batch);
            }
        }
    }
}
