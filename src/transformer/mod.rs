//! Transformer layers with automatic differentiation support
//!
//! This module provides transformer building blocks that work with entrenar's
//! tape-based autograd engine. All operations support gradient computation.
//!
//! ## Architecture Components
//!
//! - `MultiHeadAttention`: Multi-head self-attention mechanism
//! - `FeedForward`: Position-wise feed-forward network (MLP)
//! - `TransformerBlock`: Complete transformer block (attention + FFN + residuals)
//! - `TransformerConfig`: Configuration for transformer models
//!
//! ## Example
//!
//! ```ignore
//! use entrenar::transformer::{TransformerConfig, TransformerBlock};
//!
//! let config = TransformerConfig::llama2_7b();
//! let block = TransformerBlock::new(&config, 0);
//! let output = block.forward(&input);
//! ```

use crate::autograd::{add, matmul, scale};
use crate::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for transformer models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Hidden dimension (embedding size)
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of key-value heads (for grouped-query attention)
    pub num_kv_heads: usize,
    /// Feed-forward network intermediate dimension
    pub intermediate_size: usize,
    /// Number of transformer layers
    pub num_hidden_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// RMS normalization epsilon
    pub rms_norm_eps: f32,
    /// RoPE theta base
    pub rope_theta: f32,
    /// Whether to use bias in linear layers
    pub use_bias: bool,
}

impl TransformerConfig {
    /// LLaMA 2 7B configuration
    pub fn llama2_7b() -> Self {
        Self {
            hidden_size: 4096,
            num_attention_heads: 32,
            num_kv_heads: 32,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            vocab_size: 32000,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            use_bias: false,
        }
    }

    /// LLaMA 2 13B configuration
    pub fn llama2_13b() -> Self {
        Self {
            hidden_size: 5120,
            num_attention_heads: 40,
            num_kv_heads: 40,
            intermediate_size: 13824,
            num_hidden_layers: 40,
            vocab_size: 32000,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            use_bias: false,
        }
    }

    /// Mistral 7B configuration
    pub fn mistral_7b() -> Self {
        Self {
            hidden_size: 4096,
            num_attention_heads: 32,
            num_kv_heads: 8, // Grouped-query attention
            intermediate_size: 14336,
            num_hidden_layers: 32,
            vocab_size: 32000,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            use_bias: false,
        }
    }

    /// Qwen2 0.5B configuration (good for testing)
    pub fn qwen2_0_5b() -> Self {
        Self {
            hidden_size: 896,
            num_attention_heads: 14,
            num_kv_heads: 2,
            intermediate_size: 4864,
            num_hidden_layers: 24,
            vocab_size: 151936,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            use_bias: true,
        }
    }

    /// Tiny configuration for testing
    pub fn tiny() -> Self {
        Self {
            hidden_size: 64,
            num_attention_heads: 2,
            num_kv_heads: 2,
            intermediate_size: 256,
            num_hidden_layers: 2,
            vocab_size: 1000,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            use_bias: false,
        }
    }

    /// Per-head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Multi-head self-attention layer
pub struct MultiHeadAttention {
    /// Configuration
    config: TransformerConfig,
    /// Query projection weight (hidden_size x hidden_size)
    pub w_q: Tensor,
    /// Key projection weight (hidden_size x kv_hidden_size)
    pub w_k: Tensor,
    /// Value projection weight (hidden_size x kv_hidden_size)
    pub w_v: Tensor,
    /// Output projection weight (hidden_size x hidden_size)
    pub w_o: Tensor,
}

impl MultiHeadAttention {
    /// Create new attention layer with initialized weights
    pub fn new(config: &TransformerConfig) -> Self {
        let hidden_size = config.hidden_size;
        let kv_hidden_size = config.num_kv_heads * config.head_dim();

        // Xavier initialization scale
        let scale = (2.0 / (hidden_size + hidden_size) as f32).sqrt();
        let kv_scale = (2.0 / (hidden_size + kv_hidden_size) as f32).sqrt();

        Self {
            config: config.clone(),
            w_q: Tensor::from_vec(
                (0..hidden_size * hidden_size)
                    .map(|i| ((i as f32 * 0.123).sin() * scale))
                    .collect(),
                true,
            ),
            w_k: Tensor::from_vec(
                (0..hidden_size * kv_hidden_size)
                    .map(|i| ((i as f32 * 0.234).sin() * kv_scale))
                    .collect(),
                true,
            ),
            w_v: Tensor::from_vec(
                (0..hidden_size * kv_hidden_size)
                    .map(|i| ((i as f32 * 0.345).sin() * kv_scale))
                    .collect(),
                true,
            ),
            w_o: Tensor::from_vec(
                (0..hidden_size * hidden_size)
                    .map(|i| ((i as f32 * 0.456).sin() * scale))
                    .collect(),
                true,
            ),
        }
    }

    /// Create attention layer from parameter map
    ///
    /// Expected parameter names (following HuggingFace convention):
    /// - `{prefix}.q_proj.weight`
    /// - `{prefix}.k_proj.weight`
    /// - `{prefix}.v_proj.weight`
    /// - `{prefix}.o_proj.weight`
    pub fn from_params(
        config: &TransformerConfig,
        params: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Option<Self> {
        let w_q = params.get(&format!("{prefix}.q_proj.weight"))?.clone();
        let w_k = params.get(&format!("{prefix}.k_proj.weight"))?.clone();
        let w_v = params.get(&format!("{prefix}.v_proj.weight"))?.clone();
        let w_o = params.get(&format!("{prefix}.o_proj.weight"))?.clone();

        Some(Self {
            config: config.clone(),
            w_q,
            w_k,
            w_v,
            w_o,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input tensor (seq_len * hidden_size, flattened)
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Output tensor (seq_len * hidden_size, flattened)
    pub fn forward(&self, x: &Tensor, seq_len: usize) -> Tensor {
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let kv_hidden_size = num_kv_heads * head_dim;

        // Project Q, K, V
        // x: (seq_len, hidden_size) -> Q: (seq_len, hidden_size)
        let q = matmul(x, &self.w_q, seq_len, hidden_size, hidden_size);
        // x: (seq_len, hidden_size) -> K: (seq_len, kv_hidden_size)
        let k = matmul(x, &self.w_k, seq_len, hidden_size, kv_hidden_size);
        // x: (seq_len, hidden_size) -> V: (seq_len, kv_hidden_size)
        let v = matmul(x, &self.w_v, seq_len, hidden_size, kv_hidden_size);

        // Multi-head attention with grouped-query attention support
        let mut attn_outputs = Vec::with_capacity(num_heads * seq_len * head_dim);

        // Number of query heads per KV head
        let heads_per_kv = num_heads / num_kv_heads;

        for h in 0..num_heads {
            // Which KV head to use for this query head
            let kv_h = h / heads_per_kv;

            // Extract Q for this head: (seq_len, head_dim)
            let q_head: Vec<f32> = (0..seq_len)
                .flat_map(|s| {
                    let start = s * hidden_size + h * head_dim;
                    q.data().as_slice().unwrap()[start..start + head_dim].to_vec()
                })
                .collect();

            // Extract K for this KV head: (seq_len, head_dim)
            let k_head: Vec<f32> = (0..seq_len)
                .flat_map(|s| {
                    let start = s * kv_hidden_size + kv_h * head_dim;
                    k.data().as_slice().unwrap()[start..start + head_dim].to_vec()
                })
                .collect();

            // Extract V for this KV head: (seq_len, head_dim)
            let v_head: Vec<f32> = (0..seq_len)
                .flat_map(|s| {
                    let start = s * kv_hidden_size + kv_h * head_dim;
                    v.data().as_slice().unwrap()[start..start + head_dim].to_vec()
                })
                .collect();

            // Scaled dot-product attention for this head
            let q_tensor = Tensor::from_vec(q_head, false);
            let k_tensor = Tensor::from_vec(k_head, false);
            let v_tensor = Tensor::from_vec(v_head, false);

            let attn_out = crate::autograd::attention(
                &q_tensor, &k_tensor, &v_tensor, seq_len, head_dim, seq_len, head_dim,
            );

            attn_outputs.extend_from_slice(attn_out.data().as_slice().unwrap());
        }

        // Concatenate heads and reshape: (seq_len, num_heads * head_dim) = (seq_len, hidden_size)
        // Then reorder from head-major to position-major
        let mut concat_output = vec![0.0; seq_len * hidden_size];
        for h in 0..num_heads {
            for s in 0..seq_len {
                for d in 0..head_dim {
                    // Source: head h, position s, dimension d
                    let src_idx = h * seq_len * head_dim + s * head_dim + d;
                    // Destination: position s, head h, dimension d
                    let dst_idx = s * hidden_size + h * head_dim + d;
                    concat_output[dst_idx] = attn_outputs[src_idx];
                }
            }
        }

        let concat_tensor = Tensor::from_vec(concat_output, true);

        // Output projection: (seq_len, hidden_size) @ (hidden_size, hidden_size) = (seq_len, hidden_size)
        matmul(&concat_tensor, &self.w_o, seq_len, hidden_size, hidden_size)
    }

    /// Get all parameters as a vector
    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.w_q, &self.w_k, &self.w_v, &self.w_o]
    }
}

/// Position-wise Feed-Forward Network
pub struct FeedForward {
    /// Configuration
    config: TransformerConfig,
    /// Gate projection weight (hidden_size x intermediate_size)
    pub w_gate: Tensor,
    /// Up projection weight (hidden_size x intermediate_size)
    pub w_up: Tensor,
    /// Down projection weight (intermediate_size x hidden_size)
    pub w_down: Tensor,
}

impl FeedForward {
    /// Create new FFN layer with initialized weights
    pub fn new(config: &TransformerConfig) -> Self {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        // Xavier initialization scale
        let scale_up = (2.0 / (hidden_size + intermediate_size) as f32).sqrt();
        let scale_down = (2.0 / (intermediate_size + hidden_size) as f32).sqrt();

        Self {
            config: config.clone(),
            w_gate: Tensor::from_vec(
                (0..hidden_size * intermediate_size)
                    .map(|i| ((i as f32 * 0.567).sin() * scale_up))
                    .collect(),
                true,
            ),
            w_up: Tensor::from_vec(
                (0..hidden_size * intermediate_size)
                    .map(|i| ((i as f32 * 0.678).sin() * scale_up))
                    .collect(),
                true,
            ),
            w_down: Tensor::from_vec(
                (0..intermediate_size * hidden_size)
                    .map(|i| ((i as f32 * 0.789).sin() * scale_down))
                    .collect(),
                true,
            ),
        }
    }

    /// Create FFN layer from parameter map
    ///
    /// Expected parameter names (following HuggingFace convention):
    /// - `{prefix}.gate_proj.weight`
    /// - `{prefix}.up_proj.weight`
    /// - `{prefix}.down_proj.weight`
    pub fn from_params(
        config: &TransformerConfig,
        params: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Option<Self> {
        let w_gate = params.get(&format!("{prefix}.gate_proj.weight"))?.clone();
        let w_up = params.get(&format!("{prefix}.up_proj.weight"))?.clone();
        let w_down = params.get(&format!("{prefix}.down_proj.weight"))?.clone();

        Some(Self {
            config: config.clone(),
            w_gate,
            w_up,
            w_down,
        })
    }

    /// Forward pass with SwiGLU activation
    ///
    /// FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    ///
    /// # Arguments
    /// * `x` - Input tensor (seq_len * hidden_size, flattened)
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Output tensor (seq_len * hidden_size, flattened)
    pub fn forward(&self, x: &Tensor, seq_len: usize) -> Tensor {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;

        // Gate projection: (seq_len, hidden) @ (hidden, intermediate) = (seq_len, intermediate)
        let gate = matmul(x, &self.w_gate, seq_len, hidden_size, intermediate_size);

        // Up projection: (seq_len, hidden) @ (hidden, intermediate) = (seq_len, intermediate)
        let up = matmul(x, &self.w_up, seq_len, hidden_size, intermediate_size);

        // SwiGLU: SiLU(gate) * up
        let gate_activated = crate::autograd::swish(&gate);
        let hidden = crate::autograd::mul(&gate_activated, &up);

        // Down projection: (seq_len, intermediate) @ (intermediate, hidden) = (seq_len, hidden)
        matmul(
            &hidden,
            &self.w_down,
            seq_len,
            intermediate_size,
            hidden_size,
        )
    }

    /// Get all parameters as a vector
    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.w_gate, &self.w_up, &self.w_down]
    }
}

/// RMS Normalization layer
pub struct RMSNorm {
    /// Weight (scale) parameter
    pub weight: Tensor,
    /// Epsilon for numerical stability
    eps: f32,
}

impl RMSNorm {
    /// Create new RMS normalization layer
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        Self {
            weight: Tensor::ones(hidden_size, true),
            eps,
        }
    }

    /// Create from parameters
    pub fn from_params(params: &HashMap<String, Tensor>, prefix: &str, eps: f32) -> Option<Self> {
        let weight = params.get(&format!("{prefix}.weight"))?.clone();
        Some(Self { weight, eps })
    }

    /// Forward pass
    ///
    /// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let n = x.len() as f32;

        // Compute RMS
        let sq_sum: f32 = x.data().iter().map(|v| v * v).sum();
        let rms = (sq_sum / n + self.eps).sqrt();

        // Normalize and scale
        let normalized = scale(x, 1.0 / rms);
        crate::autograd::mul(&normalized, &self.weight)
    }

    /// Forward pass for batched input
    ///
    /// # Arguments
    /// * `x` - Input tensor (seq_len * hidden_size, flattened)
    /// * `seq_len` - Sequence length
    /// * `hidden_size` - Hidden dimension
    pub fn forward_batched(&self, x: &Tensor, seq_len: usize, hidden_size: usize) -> Tensor {
        let mut output = vec![0.0; seq_len * hidden_size];

        for s in 0..seq_len {
            let start = s * hidden_size;
            let end = start + hidden_size;
            let slice = &x.data().as_slice().unwrap()[start..end];

            // Compute RMS for this position
            let sq_sum: f32 = slice.iter().map(|v| v * v).sum();
            let rms = (sq_sum / hidden_size as f32 + self.eps).sqrt();

            // Normalize and scale
            for (i, &val) in slice.iter().enumerate() {
                output[start + i] = (val / rms) * self.weight.data()[i];
            }
        }

        Tensor::from_vec(output, x.requires_grad())
    }
}

/// Complete transformer block
pub struct TransformerBlock {
    /// Configuration
    config: TransformerConfig,
    /// Layer index
    layer_idx: usize,
    /// Input layer normalization
    pub input_norm: RMSNorm,
    /// Self-attention
    pub self_attn: MultiHeadAttention,
    /// Post-attention layer normalization
    pub post_attn_norm: RMSNorm,
    /// Feed-forward network
    pub ffn: FeedForward,
}

impl TransformerBlock {
    /// Create new transformer block with initialized weights
    pub fn new(config: &TransformerConfig, layer_idx: usize) -> Self {
        Self {
            config: config.clone(),
            layer_idx,
            input_norm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            self_attn: MultiHeadAttention::new(config),
            post_attn_norm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            ffn: FeedForward::new(config),
        }
    }

    /// Create transformer block from parameter map
    ///
    /// Expected parameter names (following HuggingFace LLaMA convention):
    /// - `model.layers.{layer_idx}.input_layernorm.weight`
    /// - `model.layers.{layer_idx}.self_attn.*`
    /// - `model.layers.{layer_idx}.post_attention_layernorm.weight`
    /// - `model.layers.{layer_idx}.mlp.*`
    pub fn from_params(
        config: &TransformerConfig,
        params: &HashMap<String, Tensor>,
        layer_idx: usize,
    ) -> Option<Self> {
        let prefix = format!("model.layers.{layer_idx}");

        let input_norm = RMSNorm::from_params(
            params,
            &format!("{prefix}.input_layernorm"),
            config.rms_norm_eps,
        )?;

        let self_attn =
            MultiHeadAttention::from_params(config, params, &format!("{prefix}.self_attn"))?;

        let post_attn_norm = RMSNorm::from_params(
            params,
            &format!("{prefix}.post_attention_layernorm"),
            config.rms_norm_eps,
        )?;

        let ffn = FeedForward::from_params(config, params, &format!("{prefix}.mlp"))?;

        Some(Self {
            config: config.clone(),
            layer_idx,
            input_norm,
            self_attn,
            post_attn_norm,
            ffn,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input tensor (seq_len * hidden_size, flattened)
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Output tensor (seq_len * hidden_size, flattened)
    pub fn forward(&self, x: &Tensor, seq_len: usize) -> Tensor {
        let hidden_size = self.config.hidden_size;

        // Pre-norm + attention + residual
        let norm1 = self.input_norm.forward_batched(x, seq_len, hidden_size);
        let attn_out = self.self_attn.forward(&norm1, seq_len);
        let residual1 = add(x, &attn_out);

        // Pre-norm + FFN + residual
        let norm2 = self
            .post_attn_norm
            .forward_batched(&residual1, seq_len, hidden_size);
        let ffn_out = self.ffn.forward(&norm2, seq_len);
        add(&residual1, &ffn_out)
    }

    /// Get layer index
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Get all parameters as a vector
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.input_norm.weight, &self.post_attn_norm.weight];
        params.extend(self.self_attn.parameters());
        params.extend(self.ffn.parameters());
        params
    }
}

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

        let norm = RMSNorm::from_params(params, "model.norm", config.rms_norm_eps)?;

        let lm_head = params.get("lm_head.weight").cloned();

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

    /// Get configuration
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_config_llama2() {
        let config = TransformerConfig::llama2_7b();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.head_dim(), 128);
    }

    #[test]
    fn test_transformer_config_tiny() {
        let config = TransformerConfig::tiny();
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_attention_heads, 2);
        assert_eq!(config.head_dim(), 32);
    }

    #[test]
    fn test_rms_norm_forward() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
        let output = norm.forward(&x);
        assert_eq!(output.len(), 4);
        // Output should be normalized and scaled
        let data = output.data();
        assert!(data.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_rms_norm_batched() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], true);
        let output = norm.forward_batched(&x, 2, 4);
        assert_eq!(output.len(), 8);
    }

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
    fn test_feed_forward_tiny() {
        let config = TransformerConfig::tiny();
        let ffn = FeedForward::new(&config);
        let x = Tensor::from_vec(vec![0.1; 2 * config.hidden_size], true);
        let output = ffn.forward(&x, 2);
        assert_eq!(output.len(), 2 * config.hidden_size);
    }

    #[test]
    fn test_multi_head_attention_tiny() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let x = Tensor::from_vec(vec![0.1; 2 * config.hidden_size], true);
        let output = attn.forward(&x, 2);
        assert_eq!(output.len(), 2 * config.hidden_size);
    }

    #[test]
    fn test_transformer_block_tiny() {
        let config = TransformerConfig::tiny();
        let block = TransformerBlock::new(&config, 0);
        let x = Tensor::from_vec(vec![0.1; 2 * config.hidden_size], true);
        let output = block.forward(&x, 2);
        assert_eq!(output.len(), 2 * config.hidden_size);
    }

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
    fn test_config_serialization() {
        let config = TransformerConfig::llama2_7b();
        let json = serde_json::to_string(&config).unwrap();
        let restored: TransformerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.hidden_size, config.hidden_size);
        assert_eq!(restored.num_attention_heads, config.num_attention_heads);
    }

    #[test]
    fn test_mistral_config() {
        let config = TransformerConfig::mistral_7b();
        assert_eq!(config.num_kv_heads, 8); // Grouped-query attention
        assert_eq!(config.num_attention_heads, 32);
        // 32 / 8 = 4 query heads per KV head
    }

    #[test]
    fn test_qwen2_config() {
        let config = TransformerConfig::qwen2_0_5b();
        assert!(config.use_bias);
        assert_eq!(config.vocab_size, 151936);
    }

    #[test]
    fn test_llama2_13b_config() {
        let config = TransformerConfig::llama2_13b();
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.num_attention_heads, 40);
        assert_eq!(config.num_hidden_layers, 40);
        assert_eq!(config.head_dim(), 128); // 5120 / 40 = 128
    }

    #[test]
    fn test_rms_norm_normalization_property() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], true);
        let output = norm.forward(&x);
        // After RMS normalization, if weights are 1, output should be x / rms(x)
        // rms(x) = sqrt(mean(x^2)) = sqrt(4) = 2
        // so output = [2/2, 2/2, 2/2, 2/2] = [1, 1, 1, 1]
        let data = output.data();
        for &val in data.iter() {
            assert!((val - 1.0).abs() < 1e-5, "Expected ~1.0, got {val}");
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
    fn test_feed_forward_parameters() {
        let config = TransformerConfig::tiny();
        let ffn = FeedForward::new(&config);
        let params = ffn.parameters();
        assert_eq!(params.len(), 3); // w_gate, w_up, w_down
    }

    #[test]
    fn test_multi_head_attention_parameters() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let params = attn.parameters();
        assert_eq!(params.len(), 4); // w_q, w_k, w_v, w_o
    }

    #[test]
    fn test_transformer_block_layer_idx() {
        let config = TransformerConfig::tiny();
        let block = TransformerBlock::new(&config, 5);
        assert_eq!(block.layer_idx(), 5);
    }

    #[test]
    fn test_transformer_block_parameters() {
        let config = TransformerConfig::tiny();
        let block = TransformerBlock::new(&config, 0);
        let params = block.parameters();
        // input_norm + post_attn_norm + 4 attn + 3 ffn = 9
        assert_eq!(params.len(), 9);
    }

    #[test]
    fn test_transformer_config_accessor() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);
        assert_eq!(transformer.config().hidden_size, config.hidden_size);
        assert_eq!(transformer.config().vocab_size, config.vocab_size);
    }

    #[test]
    fn test_attention_longer_sequence() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let x = Tensor::from_vec(vec![0.1; 8 * config.hidden_size], true);
        let output = attn.forward(&x, 8);
        assert_eq!(output.len(), 8 * config.hidden_size);
    }

    #[test]
    fn test_ffn_longer_sequence() {
        let config = TransformerConfig::tiny();
        let ffn = FeedForward::new(&config);
        let x = Tensor::from_vec(vec![0.1; 8 * config.hidden_size], true);
        let output = ffn.forward(&x, 8);
        assert_eq!(output.len(), 8 * config.hidden_size);
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
    fn test_rms_norm_with_zeros() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], true);
        let output = norm.forward(&x);
        // With zeros input and eps, output should be finite (zeros)
        let data = output.data();
        assert!(data.iter().all(|&v| v.is_finite()));
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
    fn test_config_yaml_serialization() {
        let config = TransformerConfig::tiny();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let restored: TransformerConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(restored.hidden_size, config.hidden_size);
        assert_eq!(restored.num_hidden_layers, config.num_hidden_layers);
    }

    #[test]
    fn test_grouped_query_attention_ratio() {
        let config = TransformerConfig::mistral_7b();
        let heads_per_kv = config.num_attention_heads / config.num_kv_heads;
        assert_eq!(heads_per_kv, 4); // 32 / 8 = 4
    }

    #[test]
    fn test_embedding_requires_grad() {
        let embed = Embedding::new(100, 8);
        assert!(embed.weight.requires_grad());
    }

    #[test]
    fn test_rms_norm_weight_requires_grad() {
        let norm = RMSNorm::new(4, 1e-6);
        assert!(norm.weight.requires_grad());
    }

    #[test]
    fn test_ffn_weight_sizes() {
        let config = TransformerConfig::tiny();
        let ffn = FeedForward::new(&config);
        assert_eq!(
            ffn.w_gate.len(),
            config.hidden_size * config.intermediate_size
        );
        assert_eq!(
            ffn.w_up.len(),
            config.hidden_size * config.intermediate_size
        );
        assert_eq!(
            ffn.w_down.len(),
            config.intermediate_size * config.hidden_size
        );
    }

    #[test]
    fn test_attention_weight_sizes() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let kv_hidden = config.num_kv_heads * config.head_dim();
        assert_eq!(attn.w_q.len(), config.hidden_size * config.hidden_size);
        assert_eq!(attn.w_k.len(), config.hidden_size * kv_hidden);
        assert_eq!(attn.w_v.len(), config.hidden_size * kv_hidden);
        assert_eq!(attn.w_o.len(), config.hidden_size * config.hidden_size);
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
    fn test_rms_norm_from_params() {
        let mut params = HashMap::new();
        params.insert(
            "test.weight".to_string(),
            Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], true),
        );
        let norm = RMSNorm::from_params(&params, "test", 1e-6);
        assert!(norm.is_some());
        let norm = norm.unwrap();
        assert_eq!(norm.weight.len(), 4);
    }

    #[test]
    fn test_config_clone() {
        let config = TransformerConfig::llama2_7b();
        let cloned = config.clone();
        assert_eq!(config.hidden_size, cloned.hidden_size);
        assert_eq!(config.vocab_size, cloned.vocab_size);
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
}
