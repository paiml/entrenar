//! Complete encoder model (BERT/RoBERTa/CodeBERT)
//!
//! Provides the full encoder pipeline:
//! ```text
//! token_ids → Embedding + PositionEmbedding + TokenTypeEmbedding → LayerNorm
//!           → N × EncoderBlock → [seq_len, hidden_size]
//! ```
//!
//! For classification, use CLS pooling on the output (position 0).

use crate::autograd::add;
use crate::error::{Error, Result};
use crate::Tensor;
use std::collections::HashMap;
use std::path::Path;

use super::config::TransformerConfig;
use super::embedding::{Embedding, LearnedPositionEmbedding};
use super::encoder_block::EncoderBlock;
use super::norm::LayerNorm;
use super::weights::{load_safetensors_weights, Architecture};

/// Complete encoder model (BERT/RoBERTa/CodeBERT).
pub struct EncoderModel {
    /// Configuration
    pub config: TransformerConfig,
    /// Token embedding
    pub embed_tokens: Embedding,
    /// Position embedding (learned absolute)
    pub position_embeddings: LearnedPositionEmbedding,
    /// Token type embedding (segment A/B, optional for RoBERTa but present in weights)
    pub token_type_embeddings: Option<Embedding>,
    /// Post-embedding LayerNorm
    pub embeddings_layernorm: LayerNorm,
    /// Encoder layers
    pub layers: Vec<EncoderBlock>,
}

impl EncoderModel {
    /// Create new encoder with default initialization.
    pub fn new(config: &TransformerConfig) -> Self {
        let max_positions = config.max_position_embeddings;
        let eps = config.rms_norm_eps;
        let layers = (0..config.num_hidden_layers)
            .map(|i| EncoderBlock::new(config, i))
            .collect();

        Self {
            config: config.clone(),
            embed_tokens: Embedding::new(config.vocab_size, config.hidden_size),
            position_embeddings: LearnedPositionEmbedding::new(max_positions, config.hidden_size),
            token_type_embeddings: Some(Embedding::new(2, config.hidden_size)),
            embeddings_layernorm: LayerNorm::new(config.hidden_size, eps),
            layers,
        }
    }

    /// Create encoder from pre-trained parameters (after RoBERTa name mapping).
    ///
    /// Expected parameter names:
    /// - `encoder.embed_tokens.weight`
    /// - `encoder.position_embeddings.weight`
    /// - `encoder.token_type_embeddings.weight` (optional)
    /// - `encoder.embeddings_layernorm.{weight,bias}`
    /// - `encoder.layers.{i}.*`
    pub fn from_params(
        config: &TransformerConfig,
        params: &HashMap<String, Tensor>,
    ) -> Option<Self> {
        let max_positions = config.max_position_embeddings;

        let embed_tokens = Embedding::from_params(
            params,
            "encoder.embed_tokens.weight",
            config.vocab_size,
            config.hidden_size,
        )?;

        let position_embeddings = LearnedPositionEmbedding::from_params(
            params,
            "encoder.position_embeddings.weight",
            max_positions,
            config.hidden_size,
        )?;

        // Token type embeddings are optional (RoBERTa has them but sets to zero)
        // CodeBERT has type_vocab_size=1, standard BERT/RoBERTa has 2.
        // Infer from actual tensor shape rather than hardcoding.
        let token_type_embeddings = params
            .get("encoder.token_type_embeddings.weight")
            .and_then(|tensor| {
                let type_vocab_size = tensor.len() / config.hidden_size;
                if type_vocab_size == 0 || tensor.len() != type_vocab_size * config.hidden_size {
                    return None;
                }
                Embedding::from_params(
                    params,
                    "encoder.token_type_embeddings.weight",
                    type_vocab_size,
                    config.hidden_size,
                )
            });

        let embeddings_layernorm = LayerNorm::from_params(
            params,
            "encoder.embeddings_layernorm",
            config.rms_norm_eps,
            config.hidden_size,
        )?;

        let layers: Option<Vec<EncoderBlock>> = (0..config.num_hidden_layers)
            .map(|i| EncoderBlock::from_params(config, params, i))
            .collect();
        let layers = layers?;

        Some(Self {
            config: config.clone(),
            embed_tokens,
            position_embeddings,
            token_type_embeddings,
            embeddings_layernorm,
            layers,
        })
    }

    /// Load encoder from SafeTensors file(s).
    pub fn from_safetensors(config: &TransformerConfig, model_path: &Path) -> Result<Self> {
        let weights = load_safetensors_weights(model_path, Architecture::RoBERTa)?;
        Self::from_params(config, &weights).ok_or_else(|| {
            Error::ConfigError("Failed to construct encoder from loaded weights".into())
        })
    }

    /// Forward pass: token_ids → hidden states [seq_len × hidden_size].
    ///
    /// # Arguments
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    /// Hidden states tensor (seq_len * hidden_size, flattened)
    pub fn forward(&self, token_ids: &[u32]) -> Tensor {
        let seq_len = token_ids.len();
        let h = self.config.hidden_size;

        // Token embeddings
        let token_emb = self.embed_tokens.forward(token_ids);

        // Position embeddings
        let pos_emb = self.position_embeddings.forward(seq_len);

        // Add token + position embeddings
        let mut combined = add(&token_emb, &pos_emb);

        // Add token type embeddings (all zeros = segment A)
        if let Some(ref tte) = self.token_type_embeddings {
            let type_ids: Vec<u32> = vec![0; seq_len];
            let type_emb = tte.forward(&type_ids);
            combined = add(&combined, &type_emb);
        }

        // Post-embedding LayerNorm
        let mut hidden = self.embeddings_layernorm.forward_batched(&combined, seq_len, h);

        // Pass through encoder layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, seq_len);
        }

        hidden
    }

    /// Extract [CLS] embedding (position 0) from hidden states.
    ///
    /// For classification, the [CLS] token at position 0 attends bidirectionally
    /// to all other tokens, making it a summary representation.
    pub fn cls_embedding(&self, token_ids: &[u32]) -> Tensor {
        let hidden = self.forward(token_ids);
        let h = self.config.hidden_size;
        let data = hidden.data();
        let slice = data.as_slice().expect("hidden contiguous");
        Tensor::from_vec(slice[..h].to_vec(), false)
    }

    /// Get total parameter count.
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        count += self.embed_tokens.vocab_size() * self.embed_tokens.hidden_size();
        count += self.position_embeddings.weight.len();
        if let Some(ref tte) = self.token_type_embeddings {
            count += tte.vocab_size() * tte.hidden_size();
        }
        count += self.embeddings_layernorm.weight.len() * 2; // weight + bias
        for layer in &self.layers {
            count += layer.parameters().iter().map(|p| p.len()).sum::<usize>();
        }
        count
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::transformer::ModelArchitecture;

    fn tiny_encoder_config() -> TransformerConfig {
        // Minimal config for testing
        TransformerConfig {
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: 4,
            intermediate_size: 64,
            vocab_size: 100,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-5,
            architecture: ModelArchitecture::Encoder,
            ..TransformerConfig::tiny()
        }
    }

    #[test]
    fn clf_001_encoder_model_forward_shape() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        let token_ids = vec![1, 2, 3, 4];
        let output = model.forward(&token_ids);
        assert_eq!(output.len(), 4 * config.hidden_size);
    }

    #[test]
    fn clf_001_encoder_model_forward_finite() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        let token_ids = vec![10, 20, 30];
        let output = model.forward(&token_ids);
        let data = output.data();
        let slice = data.as_slice().unwrap();
        assert!(slice.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn clf_001_encoder_cls_embedding_shape() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        let token_ids = vec![5, 10, 15];
        let cls = model.cls_embedding(&token_ids);
        assert_eq!(cls.len(), config.hidden_size);
    }

    #[test]
    fn clf_001_encoder_cls_embedding_deterministic() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        let token_ids = vec![1, 2, 3];
        let cls1 = model.cls_embedding(&token_ids);
        let cls2 = model.cls_embedding(&token_ids);
        let d1 = cls1.data();
        let d2 = cls2.data();
        let s1 = d1.as_slice().unwrap();
        let s2 = d2.as_slice().unwrap();
        assert_eq!(s1, s2, "CLS embedding must be deterministic");
    }

    #[test]
    fn clf_001_encoder_num_parameters() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        let count = model.num_parameters();
        // Should be > 0 and reasonable
        assert!(count > 1000, "encoder should have substantial params, got {count}");
    }
}
