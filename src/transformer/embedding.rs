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
    ///
    /// # Contract (PMAT-326)
    /// Validates weight.len() == vocab_size * hidden_size.
    /// Returns None if key is missing or shape is wrong.
    pub fn from_params(
        params: &HashMap<String, Tensor>,
        name: &str,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Option<Self> {
        let weight = params.get(name)?.clone();
        let expected = vocab_size * hidden_size;
        if weight.len() != expected {
            eprintln!(
                "[PMAT-326] Embedding '{name}': shape mismatch — got {} elements, expected {expected} ({vocab_size}x{hidden_size})",
                weight.len()
            );
            return None;
        }
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
                // N-09: OOB token → zeros. Contract: embedding-lookup-v1.yaml
                eprintln!(
                    "Warning: Embedding::forward token_id {} >= vocab_size {}. N-09 OOB escape.",
                    token_id, self.vocab_size
                );
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

    // =========================================================================
    // FALSIFY-E7: Entrenar embedding contract gap analysis (Refs PMAT-326)
    //
    // Five-Whys: §2.1.1 "What Are Embeddings" falsification sweep
    //   Why 1: Trained model could have garbage embeddings
    //   Why 2: No data quality validation during training
    //   Why 3: Embedding uses raw Tensor, not ValidatedEmbedding
    //   Why 4: entrenar predates the ValidatedEmbedding contract
    //   Why 5: No cross-crate contract enforcement test existed
    //
    // Popper (1959): "These tests try to break the claim that
    // entrenar's embedding pipeline prevents degenerate models."
    // =========================================================================

    /// FALSIFY-E7a: Embedding initialization produces non-degenerate values
    ///
    /// The init formula `(i * 0.111).sin() * scale` MUST produce varied,
    /// finite values. If it doesn't, freshly-initialized models are DOA.
    #[test]
    fn falsify_e7a_init_produces_valid_embedding() {
        let embed = Embedding::new(100, 64);
        let data = embed.weight.data();
        let slice = data.as_slice().expect("data as slice");

        // No NaN
        let nan_count = slice.iter().filter(|v| v.is_nan()).count();
        assert_eq!(nan_count, 0, "FALSIFY-E7a: Init must not produce NaN");

        // No Inf
        let inf_count = slice.iter().filter(|v| v.is_infinite()).count();
        assert_eq!(inf_count, 0, "FALSIFY-E7a: Init must not produce Inf");

        // Not all zeros (<50% zeros per embedding contract)
        let zero_count = slice.iter().filter(|v| v.abs() < 1e-10).count();
        let zero_pct = 100.0 * zero_count as f64 / slice.len() as f64;
        assert!(zero_pct < 50.0,
            "FALSIFY-E7a: Init has {zero_pct:.1}% zeros — exceeds embedding contract threshold (50%)");

        // Values vary (not constant)
        let min = slice.iter().copied().fold(f32::INFINITY, f32::min);
        let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!((max - min).abs() > 1e-6,
            "FALSIFY-E7a: Init values are constant ({min}..{max}) — degenerate embedding");
    }

    /// FALSIFY-E7b: Embedding shape matches vocab * hidden
    #[test]
    fn falsify_e7b_shape_matches_dimensions() {
        let vocab_size = 151;
        let hidden_size = 32;
        let embed = Embedding::new(vocab_size, hidden_size);
        assert_eq!(embed.weight.len(), vocab_size * hidden_size,
            "FALSIFY-E7b: Embedding length must be vocab_size * hidden_size");
    }

    /// FALSIFY-E7c: from_params rejects wrong-shape tensor (PMAT-326 fix)
    ///
    /// from_params now validates weight.len() == vocab_size * hidden_size.
    /// A tensor of 50 elements is rejected when 100*8=800 is expected.
    #[test]
    fn falsify_e7c_from_params_rejects_wrong_shape() {
        let mut params = HashMap::new();
        // Intentionally wrong size: 50 elements for 100*8=800 expected
        params.insert(
            "embed.weight".to_string(),
            Tensor::from_vec(vec![0.1; 50], true),
        );
        let embed = Embedding::from_params(&params, "embed.weight", 100, 8);
        // FIXED (PMAT-326): now rejected
        assert!(embed.is_none(),
            "FALSIFY-E7c: PMAT-326 fix — from_params MUST reject wrong-shape embedding");
    }

    /// FALSIFY-E7d: OOB token_id produces zeros (not panic)
    ///
    /// Contract divergence: aprender skips OOB tokens, realizar/entrenar zero-fill.
    /// This test documents entrenar's behavior.
    #[test]
    fn falsify_e7d_oob_token_produces_zeros_not_panic() {
        let embed = Embedding::new(100, 8);
        let tokens = vec![0, 999]; // 999 is way OOB
        let output = embed.forward(&tokens);
        assert_eq!(output.len(), 2 * 8);
        // Token 0 should have non-zero values
        let data = output.data();
        let token0_l2: f32 = (0..8).map(|i| data[i] * data[i]).sum::<f32>().sqrt();
        assert!(token0_l2 > 1e-6, "Token 0 should have non-zero embedding");
        // Token 999 should be all zeros
        let token999_l2: f32 = (8..16).map(|i| data[i] * data[i]).sum::<f32>().sqrt();
        assert!(token999_l2 < 1e-10, "OOB token should be zero-filled");
    }

    /// FALSIFY-E7e: Embedding init is deterministic (reproducible)
    #[test]
    fn falsify_e7e_init_deterministic() {
        let embed1 = Embedding::new(100, 64);
        let embed2 = Embedding::new(100, 64);
        let d1 = embed1.weight.data();
        let d2 = embed2.weight.data();
        assert_eq!(d1.as_slice().unwrap(), d2.as_slice().unwrap(),
            "FALSIFY-E7e: Same vocab+hidden must produce identical initialization");
    }
}
