//! Embedding layer module
//!
//! This module provides token embedding layers for transformer models.

use crate::Tensor;
use std::collections::HashMap;

/// Embedding layer
pub struct Embedding {
    /// Embedding weight (vocab_size x hidden_size)
    pub weight: Tensor,
    /// Vocabulary size
    vocab_size: usize,
    /// Hidden dimension
    hidden_size: usize,
}

impl Embedding {
    /// Create new embedding layer with initialized weights
    pub fn new(vocab_size: usize, hidden_size: usize) -> Self {
        let scale = (1.0 / hidden_size as f32).sqrt();
        Self {
            weight: Tensor::from_vec(
                (0..vocab_size * hidden_size)
                    .map(|i| ((i as f32 * 0.111).sin() * scale))
                    .collect(),
                true,
            ),
            vocab_size,
            hidden_size,
        }
    }

    /// Create from parameters
    pub fn from_params(
        params: &HashMap<String, Tensor>,
        name: &str,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Option<Self> {
        let weight = params.get(name)?.clone();
        Some(Self {
            weight,
            vocab_size,
            hidden_size,
        })
    }

    /// Forward pass - lookup embeddings for token IDs
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs to look up
    ///
    /// # Returns
    /// Embedded vectors (seq_len * hidden_size, flattened)
    pub fn forward(&self, token_ids: &[u32]) -> Tensor {
        let mut output = Vec::with_capacity(token_ids.len() * self.hidden_size);

        for &token_id in token_ids {
            let idx = token_id as usize;
            if idx >= self.vocab_size {
                // Out of vocabulary - use zeros
                output.extend(std::iter::repeat_n(0.0, self.hidden_size));
            } else {
                let start = idx * self.hidden_size;
                let end = start + self.hidden_size;
                output.extend_from_slice(&self.weight.data().as_slice().unwrap()[start..end]);
            }
        }

        Tensor::from_vec(output, true)
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get hidden dimension
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_forward() {
        let embed = Embedding::new(100, 8);
        let tokens = vec![0, 5, 10];
        let output = embed.forward(&tokens);
        assert_eq!(output.len(), 3 * 8);
    }

    #[test]
    fn test_embedding_out_of_vocab() {
        let embed = Embedding::new(100, 8);
        let tokens = vec![0, 200]; // 200 is out of vocab
        let output = embed.forward(&tokens);
        assert_eq!(output.len(), 2 * 8);
        // Out of vocab should be zeros
        let data = output.data();
        for i in 8..16 {
            assert_eq!(data[i], 0.0);
        }
    }

    #[test]
    fn test_embedding_vocab_and_hidden_size() {
        let embed = Embedding::new(500, 16);
        assert_eq!(embed.vocab_size(), 500);
        assert_eq!(embed.hidden_size(), 16);
    }

    #[test]
    fn test_embedding_single_token() {
        let embed = Embedding::new(100, 8);
        let tokens = vec![42];
        let output = embed.forward(&tokens);
        assert_eq!(output.len(), 8);
        assert!(output.requires_grad());
    }

    #[test]
    fn test_embedding_requires_grad() {
        let embed = Embedding::new(100, 8);
        assert!(embed.weight.requires_grad());
    }

    #[test]
    fn test_embedding_from_params() {
        let mut params = HashMap::new();
        params.insert(
            "embed.weight".to_string(),
            Tensor::from_vec(vec![0.1; 100 * 8], true),
        );
        let embed = Embedding::from_params(&params, "embed.weight", 100, 8);
        assert!(embed.is_some());
        let embed = embed.unwrap();
        assert_eq!(embed.vocab_size(), 100);
        assert_eq!(embed.hidden_size(), 8);
    }

    #[test]
    fn test_embedding_from_params_missing() {
        let params: HashMap<String, Tensor> = HashMap::new();
        let embed = Embedding::from_params(&params, "missing.weight", 100, 8);
        assert!(embed.is_none());
    }
}
