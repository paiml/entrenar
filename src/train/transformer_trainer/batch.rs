//! Language model batch utilities

/// A batch of tokenized sequences for language model training
#[derive(Debug, Clone)]
pub struct LMBatch {
    /// Input token IDs (batch_size x seq_len flattened)
    pub input_ids: Vec<u32>,
    /// Target token IDs (batch_size x seq_len flattened, shifted by 1)
    pub target_ids: Vec<u32>,
    /// Batch size
    pub batch_size: usize,
    /// Sequence length
    pub seq_len: usize,
}

impl LMBatch {
    /// Create a new LM batch from token sequences
    ///
    /// For causal LM, targets are inputs shifted by 1:
    /// input:  [BOS, A, B, C, D]
    /// target: [A, B, C, D, EOS]
    pub fn from_sequences(sequences: &[Vec<u32>], pad_id: u32, eos_id: u32) -> Self {
        if sequences.is_empty() {
            return Self {
                input_ids: Vec::new(),
                target_ids: Vec::new(),
                batch_size: 0,
                seq_len: 0,
            };
        }

        let batch_size = sequences.len();
        let max_len = sequences.iter().map(Vec::len).max().unwrap_or(0);
        let seq_len = max_len.saturating_sub(1).max(1);

        let mut input_ids = Vec::with_capacity(batch_size * seq_len);
        let mut target_ids = Vec::with_capacity(batch_size * seq_len);

        for seq in sequences {
            // Input: all tokens except last
            for i in 0..seq_len {
                if i < seq.len() - 1 {
                    input_ids.push(seq[i]);
                } else {
                    input_ids.push(pad_id);
                }
            }

            // Target: all tokens except first (shifted by 1)
            for i in 0..seq_len {
                if i + 1 < seq.len() {
                    target_ids.push(seq[i + 1]);
                } else if i + 1 == seq.len() {
                    target_ids.push(eos_id);
                } else {
                    target_ids.push(pad_id);
                }
            }
        }

        Self { input_ids, target_ids, batch_size, seq_len }
    }

    /// Create a batch from a single sequence (for testing)
    pub fn single(input_ids: Vec<u32>, target_ids: Vec<u32>) -> Self {
        let seq_len = input_ids.len();
        Self { input_ids, target_ids, batch_size: 1, seq_len }
    }

    /// Get input IDs for a specific batch item
    pub fn get_input(&self, batch_idx: usize) -> Option<&[u32]> {
        if batch_idx >= self.batch_size {
            return None;
        }
        let start = batch_idx * self.seq_len;
        let end = start + self.seq_len;
        Some(&self.input_ids[start..end])
    }

    /// Get target IDs for a specific batch item
    pub fn get_target(&self, batch_idx: usize) -> Option<&[u32]> {
        if batch_idx >= self.batch_size {
            return None;
        }
        let start = batch_idx * self.seq_len;
        let end = start + self.seq_len;
        Some(&self.target_ids[start..end])
    }

    /// Total number of tokens in batch
    pub fn num_tokens(&self) -> usize {
        self.batch_size * self.seq_len
    }
}
