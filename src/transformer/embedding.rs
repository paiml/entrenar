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

    // =========================================================================
    // FALSIFY-EM-001..004: embedding-lookup-v1.yaml contract mapping
    //
    // Five-Whys (PMAT-354):
    //   Why 1: entrenar has E7a-e init tests but no forward-path EM-* tests
    //   Why 2: E7 tests validate initialization, not the lookup/forward contract
    //   Why 3: no mapping from embedding-lookup-v1.yaml to entrenar test names
    //   Why 4: entrenar predates the provable-contracts YAML
    //   Why 5: forward() was assumed correct because it's "just slicing"
    //
    // References:
    //   - provable-contracts/contracts/embedding-lookup-v1.yaml
    //   - src/transformer/embedding.rs::forward()
    // =========================================================================

    /// FALSIFY-EM-001: forward output shape = seq_len * hidden_size
    #[test]
    fn falsify_em_001_forward_output_shape() {
        let embed = Embedding::new(100, 32);

        for seq_len in [1, 3, 10, 50] {
            let tokens: Vec<u32> = (0..seq_len).collect();
            let output = embed.forward(&tokens);
            assert_eq!(
                output.len(),
                seq_len as usize * 32,
                "FALSIFIED EM-001: forward({seq_len} tokens) produced {} elements, expected {}",
                output.len(),
                seq_len as usize * 32
            );
        }
    }

    /// FALSIFY-EM-001b: empty input produces empty output
    #[test]
    fn falsify_em_001b_forward_empty_input() {
        let embed = Embedding::new(100, 32);
        let output = embed.forward(&[]);
        assert_eq!(
            output.len(),
            0,
            "FALSIFIED EM-001b: empty input should produce 0 elements"
        );
    }

    /// FALSIFY-EM-002: OOB token → zeros, no panic (N-09 escape)
    ///
    /// Contract: token_id >= vocab_size produces zero-filled output, not a panic.
    /// Valid tokens alongside OOB tokens must still produce correct results.
    #[test]
    fn falsify_em_002_oob_safety() {
        let vocab_size = 50;
        let hidden = 8;
        let embed = Embedding::new(vocab_size, hidden);

        // Pure OOB tokens
        let oob_output = embed.forward(&[999, 50, 100]);
        let oob_data = oob_output.data();
        for (i, &v) in oob_data.iter().enumerate() {
            assert!(
                v.abs() < 1e-10,
                "FALSIFIED EM-002: OOB output[{i}] = {v}, expected 0.0"
            );
        }

        // Mixed valid + OOB: valid tokens must still be correct
        let mixed_output = embed.forward(&[0, 999, 49]);
        let mixed_data = mixed_output.data();
        let weight_data = embed.weight.data();

        // Token 0 (valid): should match weight row 0
        for d in 0..hidden {
            assert_eq!(
                mixed_data[d],
                weight_data[d],
                "FALSIFIED EM-002: valid token 0 corrupted at dim {d}"
            );
        }

        // Token 999 (OOB): should be zeros
        for d in 0..hidden {
            assert!(
                mixed_data[hidden + d].abs() < 1e-10,
                "FALSIFIED EM-002: OOB token 999 at dim {d} = {}, expected 0.0",
                mixed_data[hidden + d]
            );
        }

        // Token 49 (valid boundary): should match weight row 49
        for d in 0..hidden {
            assert_eq!(
                mixed_data[2 * hidden + d],
                weight_data[49 * hidden + d],
                "FALSIFIED EM-002: valid boundary token 49 corrupted at dim {d}"
            );
        }
    }

    /// FALSIFY-EM-003: forward determinism (same tokens → bit-identical output)
    #[test]
    fn falsify_em_003_forward_determinism() {
        let embed = Embedding::new(100, 64);
        let tokens = vec![5u32, 42, 0, 99, 17];

        let o1 = embed.forward(&tokens);
        let o2 = embed.forward(&tokens);

        assert_eq!(
            o1.data().as_slice().unwrap(),
            o2.data().as_slice().unwrap(),
            "FALSIFIED EM-003: forward() is non-deterministic"
        );
    }

    /// FALSIFY-EM-004: forward output is finite (no NaN, no Inf)
    #[test]
    fn falsify_em_004_forward_finite_output() {
        let embed = Embedding::new(200, 16);
        let tokens: Vec<u32> = (0..200).collect();
        let output = embed.forward(&tokens);
        let data = output.data();

        let nan_count = data.iter().filter(|v| v.is_nan()).count();
        let inf_count = data.iter().filter(|v| v.is_infinite()).count();

        assert_eq!(
            nan_count, 0,
            "FALSIFIED EM-004: forward output contains {} NaN values",
            nan_count
        );
        assert_eq!(
            inf_count, 0,
            "FALSIFIED EM-004: forward output contains {} Inf values",
            inf_count
        );
    }

    /// FALSIFY-EM-005: forward value correctness (extractive — output[i] = W[token_id])
    #[test]
    fn falsify_em_005_forward_value_correctness() {
        let embed = Embedding::new(50, 8);
        let tokens = vec![0u32, 10, 49];
        let output = embed.forward(&tokens);
        let out_data = output.data();
        let weight_data = embed.weight.data();

        // Token 0: output[0..8] == weight[0..8]
        for i in 0..8 {
            assert_eq!(
                out_data[i], weight_data[i],
                "FALSIFIED EM-005: output[{i}] != weight[{i}] for token 0"
            );
        }
        // Token 10: output[8..16] == weight[80..88]
        for i in 0..8 {
            assert_eq!(
                out_data[8 + i],
                weight_data[80 + i],
                "FALSIFIED EM-005: output[{}] != weight[{}] for token 10",
                8 + i,
                80 + i
            );
        }
    }

    // =========================================================================
    // FALSIFY-EMB-005: Non-zero embeddings (embedding-algebra-v1.yaml)
    //
    // Five-Whys (PMAT-354):
    //   Why 1: entrenar had E7a init tests but no FALSIFY-EMB-005 tagged test
    //   Why 2: E7a covers init validity, not the EMB "non-zero" algebra claim
    //   Why 3: no mapping from embedding-algebra-v1.yaml to entrenar test names
    //   Why 4: entrenar predates the provable-contracts YAML
    //   Why 5: forward output non-zero was assumed from init non-zero
    // =========================================================================

    /// FALSIFY-EMB-005: forward output is non-zero for valid tokens
    #[test]
    fn falsify_emb_005_forward_non_zero() {
        let embed = Embedding::new(100, 64);
        let tokens = vec![0u32, 42, 99];
        let output = embed.forward(&tokens);
        let data = output.data();

        let l2_norm: f32 = data.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            l2_norm > 1e-6,
            "FALSIFIED EMB-005: forward output is all-zero (L2={l2_norm})"
        );
    }
}
