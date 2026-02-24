//! Complete transformer model module
//!
//! This module provides the full transformer model for language modeling.

use crate::autograd::matmul;
use crate::Tensor;
use std::collections::HashMap;

use super::block::TransformerBlock;
use super::config::TransformerConfig;
use super::embedding::Embedding;
use super::norm::RMSNorm;

/// Complete transformer model
pub struct Transformer {
    /// Configuration
    pub config: TransformerConfig,
    /// Token embedding layer
    pub embed_tokens: Embedding,
    /// Transformer layers
    pub layers: Vec<TransformerBlock>,
    /// Final layer normalization
    pub norm: RMSNorm,
    /// Language model head (tied to embeddings or separate)
    pub lm_head: Option<Tensor>,
}

impl Transformer {
    /// Create new transformer with initialized weights
    pub fn new(config: &TransformerConfig) -> Self {
        let layers: Vec<TransformerBlock> = (0..config.num_hidden_layers)
            .map(|i| TransformerBlock::new(config, i))
            .collect();

        Self {
            config: config.clone(),
            embed_tokens: Embedding::new(config.vocab_size, config.hidden_size),
            layers,
            norm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            lm_head: None, // Use tied embeddings by default
        }
    }

    /// Create transformer from parameter map
    ///
    /// Expected parameter names (following HuggingFace LLaMA convention):
    /// - `model.embed_tokens.weight`
    /// - `model.layers.{i}.*`
    /// - `model.norm.weight`
    /// - `lm_head.weight` (optional, uses tied embeddings if not present)
    pub fn from_params(
        config: &TransformerConfig,
        params: &HashMap<String, Tensor>,
    ) -> Option<Self> {
        let embed_tokens = Embedding::from_params(
            params,
            "model.embed_tokens.weight",
            config.vocab_size,
            config.hidden_size,
        )?;

        let layers: Option<Vec<TransformerBlock>> = (0..config.num_hidden_layers)
            .map(|i| TransformerBlock::from_params(config, params, i))
            .collect();
        let layers = layers?;

        let norm = RMSNorm::from_params(params, "model.norm", config.rms_norm_eps, config.hidden_size)?;

        // PMAT-329: Validate lm_head shape if present
        let lm_head = if let Some(tensor) = params.get("lm_head.weight") {
            let expected = config.hidden_size * config.vocab_size;
            if tensor.len() != expected {
                eprintln!(
                    "[PMAT-329] lm_head.weight: shape mismatch — got {} elements, expected {expected} ({hidden}x{vocab})",
                    tensor.len(),
                    hidden = config.hidden_size,
                    vocab = config.vocab_size,
                );
                return None;
            }
            Some(tensor.clone())
        } else {
            None
        };

        Some(Self {
            config: config.clone(),
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    /// Forward pass for language modeling
    ///
    /// # Arguments
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    /// Logits tensor (seq_len * vocab_size, flattened)
    pub fn forward(&self, token_ids: &[u32]) -> Tensor {
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;

        // Embed tokens
        let mut hidden = self.embed_tokens.forward(token_ids);

        // Pass through transformer layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, seq_len);
        }

        // Final normalization
        let normalized = self.norm.forward_batched(&hidden, seq_len, hidden_size);

        // Language model head
        let lm_weight = self.lm_head.as_ref().unwrap_or(&self.embed_tokens.weight);

        // (seq_len, hidden_size) @ (hidden_size, vocab_size) = (seq_len, vocab_size)
        // Note: If embedding is (vocab_size, hidden_size), we need to transpose
        // For tied weights, we use embedding.T effectively
        matmul(
            &normalized,
            lm_weight,
            seq_len,
            hidden_size,
            self.config.vocab_size,
        )
    }

    /// Forward pass returning hidden states (before lm_head)
    ///
    /// # Arguments
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    /// Hidden states tensor (seq_len * hidden_size, flattened)
    pub fn forward_hidden(&self, token_ids: &[u32]) -> Tensor {
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;

        // Embed tokens
        let mut hidden = self.embed_tokens.forward(token_ids);

        // Pass through transformer layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, seq_len);
        }

        // Final normalization
        self.norm.forward_batched(&hidden, seq_len, hidden_size)
    }

    /// Get the last token's logits (for generation)
    pub fn forward_last(&self, token_ids: &[u32]) -> Tensor {
        let logits = self.forward(token_ids);
        let seq_len = token_ids.len();
        let vocab_size = self.config.vocab_size;

        // Extract last position
        let start = (seq_len - 1) * vocab_size;
        let end = start + vocab_size;
        let last_logits: Vec<f32> = logits.data().as_slice().unwrap()[start..end].to_vec();

        Tensor::from_vec(last_logits, logits.requires_grad())
    }

    /// Get all parameters as a vector
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.embed_tokens.weight, &self.norm.weight];
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        if let Some(lm_head) = &self.lm_head {
            params.push(lm_head);
        }
        params
    }

    /// Get all parameters as mutable references for optimizer
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params: Vec<&mut Tensor> = Vec::new();
        params.push(&mut self.embed_tokens.weight);
        params.push(&mut self.norm.weight);
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        if let Some(lm_head) = &mut self.lm_head {
            params.push(lm_head);
        }
        params
    }

    /// Get configuration
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_tiny_forward() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);
        let tokens = vec![1, 2, 3];
        let logits = transformer.forward(&tokens);
        assert_eq!(logits.len(), 3 * config.vocab_size);
    }

    #[test]
    fn test_transformer_tiny_forward_last() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);
        let tokens = vec![1, 2, 3];
        let logits = transformer.forward_last(&tokens);
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_transformer_parameters() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);
        let params = transformer.parameters();
        // embed_tokens + norm + (layers * (input_norm + post_attn_norm + 4 attn weights + 3 ffn weights))
        // = 2 + 2 * (2 + 4 + 3) = 2 + 2 * 9 = 20
        assert_eq!(params.len(), 20);
    }

    #[test]
    fn test_transformer_config_accessor() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);
        assert_eq!(transformer.config().hidden_size, config.hidden_size);
        assert_eq!(transformer.config().vocab_size, config.vocab_size);
    }

    #[test]
    fn test_transformer_single_token() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);
        let tokens = vec![42];
        let logits = transformer.forward(&tokens);
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_output_finite_values() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);
        let tokens = vec![1, 2, 3, 4, 5];
        let logits = transformer.forward(&tokens);
        // All outputs should be finite (no NaN or Inf)
        assert!(logits.data().iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_transformer_empty_lm_head_uses_tied_weights() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);
        // Default transformer should have no separate lm_head
        assert!(transformer.lm_head.is_none());
        // But should still produce valid logits
        let tokens = vec![1, 2];
        let logits = transformer.forward(&tokens);
        assert_eq!(logits.len(), 2 * config.vocab_size);
    }

    #[test]
    fn test_from_params_returns_none_on_missing() {
        let config = TransformerConfig::tiny();
        let params: HashMap<String, Tensor> = HashMap::new();
        let result = Transformer::from_params(&config, &params);
        assert!(result.is_none());
    }

    #[test]
    fn test_transformer_from_params_with_lm_head() {
        let config = TransformerConfig::tiny();
        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;
        let kv_hidden_size = config.num_kv_heads * config.head_dim();
        let intermediate_size = config.intermediate_size;

        let mut params = HashMap::new();

        // Embedding
        params.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::from_vec(vec![0.1; vocab_size * hidden_size], true),
        );

        // All layers
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer_idx}");
            params.insert(
                format!("{prefix}.input_layernorm.weight"),
                Tensor::from_vec(vec![1.0; hidden_size], true),
            );
            params.insert(
                format!("{prefix}.self_attn.q_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
            );
            params.insert(
                format!("{prefix}.self_attn.k_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
            );
            params.insert(
                format!("{prefix}.self_attn.v_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
            );
            params.insert(
                format!("{prefix}.self_attn.o_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
            );
            params.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                Tensor::from_vec(vec![1.0; hidden_size], true),
            );
            params.insert(
                format!("{prefix}.mlp.gate_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * intermediate_size], true),
            );
            params.insert(
                format!("{prefix}.mlp.up_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * intermediate_size], true),
            );
            params.insert(
                format!("{prefix}.mlp.down_proj.weight"),
                Tensor::from_vec(vec![0.1; intermediate_size * hidden_size], true),
            );
        }

        // Final norm
        params.insert(
            "model.norm.weight".to_string(),
            Tensor::from_vec(vec![1.0; hidden_size], true),
        );

        // LM head (separate, not tied)
        params.insert(
            "lm_head.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * vocab_size], true),
        );

        let transformer = Transformer::from_params(&config, &params);
        assert!(transformer.is_some());
        let transformer = transformer.unwrap();
        assert!(transformer.lm_head.is_some());
        assert_eq!(transformer.layers.len(), config.num_hidden_layers);
    }

    #[test]
    fn test_transformer_from_params_without_lm_head() {
        let config = TransformerConfig::tiny();
        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;
        let kv_hidden_size = config.num_kv_heads * config.head_dim();
        let intermediate_size = config.intermediate_size;

        let mut params = HashMap::new();

        // Embedding
        params.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::from_vec(vec![0.1; vocab_size * hidden_size], true),
        );

        // All layers
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer_idx}");
            params.insert(
                format!("{prefix}.input_layernorm.weight"),
                Tensor::from_vec(vec![1.0; hidden_size], true),
            );
            params.insert(
                format!("{prefix}.self_attn.q_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
            );
            params.insert(
                format!("{prefix}.self_attn.k_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
            );
            params.insert(
                format!("{prefix}.self_attn.v_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
            );
            params.insert(
                format!("{prefix}.self_attn.o_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
            );
            params.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                Tensor::from_vec(vec![1.0; hidden_size], true),
            );
            params.insert(
                format!("{prefix}.mlp.gate_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * intermediate_size], true),
            );
            params.insert(
                format!("{prefix}.mlp.up_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * intermediate_size], true),
            );
            params.insert(
                format!("{prefix}.mlp.down_proj.weight"),
                Tensor::from_vec(vec![0.1; intermediate_size * hidden_size], true),
            );
        }

        // Final norm - no lm_head
        params.insert(
            "model.norm.weight".to_string(),
            Tensor::from_vec(vec![1.0; hidden_size], true),
        );

        let transformer = Transformer::from_params(&config, &params);
        assert!(transformer.is_some());
        let transformer = transformer.unwrap();
        assert!(transformer.lm_head.is_none()); // Should use tied embeddings
    }

    #[test]
    fn test_transformer_parameters_with_lm_head() {
        let config = TransformerConfig::tiny();
        let mut transformer = Transformer::new(&config);

        // Add a separate lm_head
        transformer.lm_head = Some(Tensor::from_vec(
            vec![0.1; config.hidden_size * config.vocab_size],
            true,
        ));

        let params = transformer.parameters();
        // embed_tokens + norm + (layers * 9) + lm_head
        // = 1 + 1 + (2 * 9) + 1 = 21
        assert_eq!(params.len(), 21);
    }

    #[test]
    fn test_transformer_forward_with_lm_head() {
        let config = TransformerConfig::tiny();
        let mut transformer = Transformer::new(&config);

        // Add a separate lm_head
        transformer.lm_head = Some(Tensor::from_vec(
            vec![0.1; config.hidden_size * config.vocab_size],
            true,
        ));

        let tokens = vec![1, 2, 3];
        let logits = transformer.forward(&tokens);
        assert_eq!(logits.len(), 3 * config.vocab_size);
        assert!(logits.data().iter().all(|&v| v.is_finite()));
    }

    // =========================================================================
    // FALSIFY-L: §2.1.2 LM Head Contract — Five-Whys Gap Analysis (Refs PMAT-329)
    //
    // Contract: tensor-layout-v1.yaml §tensors.lm_head
    //   critical: "true"
    //   note: "GH-202 root cause - wrong shape caused [PAD] garbage output"
    //
    // Five-Whys:
    //   Why 1: entrenar-trained model's lm_head could corrupt inference
    //   Why 2: lm_head save/load has no shape validation
    //   Why 3: from_params accepts ANY tensor for lm_head (like embedding)
    //   Why 4: entrenar predates ValidatedWeight contract
    //   Why 5: No cross-crate contract enforcement for trained models
    //
    // Popper (1959): "These tests attempt to falsify the claim that
    // entrenar's lm_head handling prevents garbage output after training."
    // =========================================================================

    /// FALSIFY-L1e: from_params rejects wrong-shape lm_head (PMAT-329 fix)
    ///
    /// from_params now validates lm_head shape against vocab*hidden.
    /// A tensor of 50 elements is rejected when vocab*hidden is expected.
    #[test]
    fn falsify_l1e_from_params_rejects_wrong_shape_lm_head() {
        let config = TransformerConfig::tiny();
        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;
        let kv_hidden_size = config.num_kv_heads * config.head_dim();
        let intermediate_size = config.intermediate_size;

        let mut params = HashMap::new();

        // Valid embedding + layers + norm
        params.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::from_vec(vec![0.1; vocab_size * hidden_size], true),
        );
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer_idx}");
            params.insert(format!("{prefix}.input_layernorm.weight"), Tensor::from_vec(vec![1.0; hidden_size], true));
            params.insert(format!("{prefix}.self_attn.q_proj.weight"), Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true));
            params.insert(format!("{prefix}.self_attn.k_proj.weight"), Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true));
            params.insert(format!("{prefix}.self_attn.v_proj.weight"), Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true));
            params.insert(format!("{prefix}.self_attn.o_proj.weight"), Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true));
            params.insert(format!("{prefix}.post_attention_layernorm.weight"), Tensor::from_vec(vec![1.0; hidden_size], true));
            params.insert(format!("{prefix}.mlp.gate_proj.weight"), Tensor::from_vec(vec![0.1; hidden_size * intermediate_size], true));
            params.insert(format!("{prefix}.mlp.up_proj.weight"), Tensor::from_vec(vec![0.1; hidden_size * intermediate_size], true));
            params.insert(format!("{prefix}.mlp.down_proj.weight"), Tensor::from_vec(vec![0.1; intermediate_size * hidden_size], true));
        }
        params.insert("model.norm.weight".to_string(), Tensor::from_vec(vec![1.0; hidden_size], true));

        // WRONG-SHAPE lm_head: 50 elements for hidden*vocab expected
        params.insert(
            "lm_head.weight".to_string(),
            Tensor::from_vec(vec![0.1; 50], true),
        );

        let transformer = Transformer::from_params(&config, &params);
        // FIXED (PMAT-329): now rejected
        assert!(transformer.is_none(),
            "FALSIFY-L1e: PMAT-329 fix — from_params MUST reject wrong-shape lm_head");
    }

    /// FALSIFY-L2e: Tied embeddings produce valid logit dimensions
    ///
    /// When lm_head is None, the embedding weight [vocab, hidden] is used as lm_head.
    /// The matmul must produce [seq_len, vocab_size] logits.
    #[test]
    fn falsify_l2e_tied_embeddings_produce_correct_logit_dims() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);
        assert!(transformer.lm_head.is_none(), "Default should use tied embeddings");

        let tokens = vec![1, 2, 3];
        let logits = transformer.forward(&tokens);
        assert_eq!(logits.len(), 3 * config.vocab_size,
            "FALSIFY-L2e: Tied embedding logits must be seq_len * vocab_size");

        // All logits must be finite (not NaN/Inf)
        let data = logits.data();
        let nan_count = data.iter().filter(|v| v.is_nan()).count();
        let inf_count = data.iter().filter(|v| v.is_infinite()).count();
        assert_eq!(nan_count, 0, "FALSIFY-L2e: Tied logits must not contain NaN");
        assert_eq!(inf_count, 0, "FALSIFY-L2e: Tied logits must not contain Inf");
    }

    /// FALSIFY-L3e: Separate lm_head produces valid logit dimensions
    #[test]
    fn falsify_l3e_separate_lm_head_produces_correct_logit_dims() {
        let config = TransformerConfig::tiny();
        let mut transformer = Transformer::new(&config);
        transformer.lm_head = Some(Tensor::from_vec(
            vec![0.1; config.hidden_size * config.vocab_size],
            true,
        ));

        let tokens = vec![1, 2, 3];
        let logits = transformer.forward(&tokens);
        assert_eq!(logits.len(), 3 * config.vocab_size,
            "FALSIFY-L3e: Separate lm_head logits must be seq_len * vocab_size");
        let data = logits.data();
        assert!(data.iter().all(|v| v.is_finite()),
            "FALSIFY-L3e: Separate lm_head logits must all be finite");
    }

    /// FALSIFY-L4e: lm_head is included in parameters() and parameters_mut()
    ///
    /// If lm_head is present but not returned by parameters(), the optimizer
    /// won't update it during training → frozen lm_head → garbage after finetuning.
    #[test]
    fn falsify_l4e_lm_head_in_parameter_list() {
        let config = TransformerConfig::tiny();
        let mut transformer = Transformer::new(&config);

        // Without lm_head: N params
        let n_without = transformer.parameters().len();

        // With lm_head: N+1 params
        transformer.lm_head = Some(Tensor::from_vec(
            vec![0.1; config.hidden_size * config.vocab_size],
            true,
        ));
        let n_with = transformer.parameters().len();
        assert_eq!(n_with, n_without + 1,
            "FALSIFY-L4e: lm_head must be included in parameters() — optimizer needs it");

        // Also check parameters_mut
        let n_mut = transformer.parameters_mut().len();
        assert_eq!(n_mut, n_with,
            "FALSIFY-L4e: parameters_mut() must include lm_head for gradient updates");
    }

    /// FALSIFY-L5e: forward_last returns exactly vocab_size logits
    ///
    /// The last token's logits are used for next-token prediction.
    /// Off-by-one in the slice extraction → wrong token generated.
    #[test]
    fn falsify_l5e_forward_last_correct_size() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);

        let tokens = vec![1, 2, 3, 4, 5];
        let logits = transformer.forward_last(&tokens);
        assert_eq!(logits.len(), config.vocab_size,
            "FALSIFY-L5e: forward_last must return exactly vocab_size logits");
        let data = logits.data();
        assert!(data.iter().all(|v| v.is_finite()),
            "FALSIFY-L5e: forward_last logits must all be finite");
    }

    #[test]
    fn test_causal_lm_loss_backward() {
        use crate::train::CausalLMLoss;
        use crate::train::LossFn;

        let vocab_size = 100;
        let seq_len = 3;
        let loss_fn = CausalLMLoss::new(vocab_size);

        // Create some logits
        let logits = Tensor::from_vec(
            (0..seq_len * vocab_size)
                .map(|i| (i as f32 * 0.01).sin())
                .collect(),
            true,
        );

        // Target token IDs
        let targets = Tensor::from_vec(vec![5.0, 10.0, 15.0], false);

        let mut loss = loss_fn.forward(&logits, &targets);

        // Backward
        crate::autograd::backward(&mut loss, None);

        // Loss should be positive
        assert!(loss.data()[0] > 0.0);
        assert!(loss.data()[0].is_finite());

        // Logits should have gradient
        assert!(logits.grad().is_some());
        let grad = logits.grad().unwrap();
        assert!(grad.iter().all(|&v| v.is_finite()));
    }
}
