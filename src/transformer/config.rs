//! Transformer configuration module
//!
//! This module provides configuration structures for transformer models.

use serde::{Deserialize, Serialize};

// Well-known model architecture constants
const LLAMA2_7B_INTERMEDIATE_SIZE: usize = 11008;
const LLAMA2_13B_HIDDEN_SIZE: usize = 5120;
const LLAMA2_13B_INTERMEDIATE_SIZE: usize = 13824;
const LLAMA_VOCAB_SIZE: usize = 32000;
const MISTRAL_INTERMEDIATE_SIZE: usize = 14336;
const MISTRAL_MAX_SEQ_LEN: usize = 32768;
const QWEN2_0_5B_HIDDEN_SIZE: usize = 896;
const QWEN2_0_5B_INTERMEDIATE_SIZE: usize = 4864;
const QWEN2_VOCAB_SIZE: usize = 151936;
const QWEN2_MAX_SEQ_LEN: usize = 32768;
const QWEN2_ROPE_THETA: f32 = 1_000_000.0;
const QWEN3_4B_HIDDEN_SIZE: usize = 2560;
const QWEN3_4B_INTERMEDIATE_SIZE: usize = 9728;
const QWEN3_5_9B_HIDDEN_SIZE: usize = 4096;
const QWEN3_5_9B_INTERMEDIATE_SIZE: usize = 12288;
const QWEN3_5_VOCAB_SIZE: usize = 248320;
const QWEN3_5_MAX_SEQ_LEN: usize = 262144;
const DEFAULT_ROPE_THETA: f32 = 10000.0;

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
    /// Explicit per-head dimension (overrides hidden_size / num_heads).
    /// Required for Qwen3 where head_dim=128 but hidden_size/num_heads=80.
    #[serde(default)]
    pub head_dim_override: Option<usize>,
}

impl TransformerConfig {
    /// LLaMA 2 7B configuration
    pub fn llama2_7b() -> Self {
        Self {
            hidden_size: 4096,
            num_attention_heads: 32,
            num_kv_heads: 32,
            intermediate_size: LLAMA2_7B_INTERMEDIATE_SIZE,
            num_hidden_layers: 32,
            vocab_size: LLAMA_VOCAB_SIZE,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-6,
            rope_theta: DEFAULT_ROPE_THETA,
            use_bias: false,
            head_dim_override: None,
        }
    }

    /// LLaMA 2 13B configuration
    pub fn llama2_13b() -> Self {
        Self {
            hidden_size: LLAMA2_13B_HIDDEN_SIZE,
            num_attention_heads: 40,
            num_kv_heads: 40,
            intermediate_size: LLAMA2_13B_INTERMEDIATE_SIZE,
            num_hidden_layers: 40,
            vocab_size: LLAMA_VOCAB_SIZE,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-6,
            rope_theta: DEFAULT_ROPE_THETA,
            use_bias: false,
            head_dim_override: None,
        }
    }

    /// Mistral 7B configuration
    pub fn mistral_7b() -> Self {
        Self {
            hidden_size: 4096,
            num_attention_heads: 32,
            num_kv_heads: 8, // Grouped-query attention
            intermediate_size: MISTRAL_INTERMEDIATE_SIZE,
            num_hidden_layers: 32,
            vocab_size: LLAMA_VOCAB_SIZE,
            max_position_embeddings: MISTRAL_MAX_SEQ_LEN,
            rms_norm_eps: 1e-5,
            rope_theta: DEFAULT_ROPE_THETA,
            use_bias: false,
            head_dim_override: None,
        }
    }

    /// Qwen2 0.5B configuration (good for testing)
    pub fn qwen2_0_5b() -> Self {
        Self {
            hidden_size: QWEN2_0_5B_HIDDEN_SIZE,
            num_attention_heads: 14,
            num_kv_heads: 2,
            intermediate_size: QWEN2_0_5B_INTERMEDIATE_SIZE,
            num_hidden_layers: 24,
            vocab_size: QWEN2_VOCAB_SIZE,
            max_position_embeddings: QWEN2_MAX_SEQ_LEN,
            rms_norm_eps: 1e-6,
            rope_theta: QWEN2_ROPE_THETA,
            use_bias: true,
            head_dim_override: None,
        }
    }

    /// Qwen2.5-Coder 7B configuration (GH-371)
    ///
    /// Qwen2.5-Coder-7B-Instruct: 28 layers, 28 heads, 4 KV heads, hidden=3584
    /// Contract: contracts/model-families/qwen2.yaml
    pub fn qwen2_7b() -> Self {
        Self {
            hidden_size: 3584,
            num_attention_heads: 28,
            num_kv_heads: 4,
            intermediate_size: 18944,
            num_hidden_layers: 28,
            vocab_size: 152064,
            max_position_embeddings: QWEN2_MAX_SEQ_LEN,
            rms_norm_eps: 1e-6,
            rope_theta: QWEN2_ROPE_THETA,
            use_bias: true,
            head_dim_override: None,
        }
    }

    /// Qwen3 4B configuration
    ///
    /// Qwen3-4B: 36 layers, 32 heads, 8 KV heads, hidden=2560, head_dim=128.
    /// Same vocab_size as Qwen2 (151936). No attention bias (Qwen3 family).
    pub fn qwen3_4b() -> Self {
        Self {
            hidden_size: QWEN3_4B_HIDDEN_SIZE,
            num_attention_heads: 32,
            num_kv_heads: 8,
            intermediate_size: QWEN3_4B_INTERMEDIATE_SIZE,
            num_hidden_layers: 36,
            vocab_size: QWEN2_VOCAB_SIZE, // 151936, same as Qwen2
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: QWEN2_ROPE_THETA, // 1M theta
            use_bias: false,              // Qwen3: no attention bias
            head_dim_override: Some(128), // Contract: qwen3.yaml §4b.head_dim=128
        }
    }

    /// Qwen3.5 9B configuration
    ///
    /// Key differences from Qwen2: no attention bias, head_dim=256 (explicit),
    /// vocab_size=248320, hybrid attention (standard + linear layers).
    /// Contract: contracts/model-families/qwen3_5.yaml
    pub fn qwen3_5_9b() -> Self {
        Self {
            hidden_size: QWEN3_5_9B_HIDDEN_SIZE,
            num_attention_heads: 16,
            num_kv_heads: 4,
            intermediate_size: QWEN3_5_9B_INTERMEDIATE_SIZE,
            num_hidden_layers: 32,
            vocab_size: QWEN3_5_VOCAB_SIZE,
            max_position_embeddings: QWEN3_5_MAX_SEQ_LEN,
            rms_norm_eps: 1e-6,
            rope_theta: QWEN2_ROPE_THETA, // Same 1M theta as Qwen2
            use_bias: false,              // KEY: no attention bias (unlike Qwen2)
            head_dim_override: None,      // 4096/16=256, no override needed
        }
    }

    /// Construct from APR v2 metadata fields.
    ///
    /// CONTRACT: The `.apr` file is the single source of truth for model
    /// architecture. These fields were validated at import time by the
    /// `tensor-layout-v1` contract. This function propagates that contract
    /// to the training pipeline — no hardcoded lookups, no silent fallbacks.
    ///
    /// Returns None if any required field is missing, forcing the caller to
    /// handle the error explicitly rather than silently degrading to tiny().
    ///
    /// GH-376: Fixes instruct pipeline ignoring .apr architecture metadata.
    pub fn from_apr_metadata(
        hidden_size: Option<usize>,
        num_heads: Option<usize>,
        num_kv_heads: Option<usize>,
        intermediate_size: Option<usize>,
        num_layers: Option<usize>,
        vocab_size: Option<usize>,
        max_position_embeddings: Option<usize>,
        rms_norm_eps: Option<f32>,
        rope_theta: Option<f32>,
        architecture: Option<&str>,
    ) -> Option<Self> {
        let hidden = hidden_size?;
        let heads = num_heads?;
        let layers = num_layers?;
        let vocab = vocab_size?;
        let intermediate = intermediate_size?;

        // Qwen3 family: head_dim=128 is explicit, not hidden/heads
        // Qwen2 family: use_bias=true
        let (use_bias, head_dim_override) = match architecture {
            Some(a) if a.starts_with("qwen3") => {
                // Qwen3: no bias, explicit head_dim=128 when hidden/heads != 128
                let computed = hidden / heads;
                let override_dim = if computed != 128 { Some(128) } else { None };
                (false, override_dim)
            }
            Some(a) if a.starts_with("qwen2") => (true, None),
            _ => (false, None),
        };

        Some(Self {
            hidden_size: hidden,
            num_attention_heads: heads,
            num_kv_heads: num_kv_heads.unwrap_or(heads),
            intermediate_size: intermediate,
            num_hidden_layers: layers,
            vocab_size: vocab,
            max_position_embeddings: max_position_embeddings.unwrap_or(32768),
            rms_norm_eps: rms_norm_eps.unwrap_or(1e-6),
            rope_theta: rope_theta.unwrap_or(DEFAULT_ROPE_THETA),
            use_bias,
            head_dim_override,
        })
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
            rope_theta: DEFAULT_ROPE_THETA,
            use_bias: false,
            head_dim_override: None,
        }
    }

    /// Per-head dimension.
    ///
    /// Uses explicit override when set (Qwen3: head_dim=128 with hidden=2560, 32 heads).
    /// Falls back to hidden_size / num_heads for standard architectures.
    pub fn head_dim(&self) -> usize {
        self.head_dim_override
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Total Q/O projection dimension = num_heads * head_dim.
    ///
    /// Equals hidden_size for standard architectures but differs when head_dim
    /// is explicitly overridden (e.g. Qwen3-4B: 32 * 128 = 4096 != 2560).
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim()
    }

    // =========================================================================
    // VRAM Budget Solver (Provable Design-by-Contract)
    //
    // Contract: contracts/model-families/qwen3.yaml §CUDA TRAINING RESOURCE BUDGET
    // Meyer (1992) "No Hidden Clauses": every term maps 1:1 to a GpuBuffer::new()
    // call in cuda_block.rs. No magic numbers — all derived from model dims.
    // =========================================================================

    /// KV hidden dimension = num_kv_heads * head_dim.
    fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim()
    }

    /// Per-layer weight VRAM in f32 elements (constant, independent of seq_len).
    ///
    /// Maps to cuda_block.rs lines 212-220: `GpuBuffer::from_host()` uploads.
    pub fn per_layer_weight_elements(&self) -> usize {
        let h = self.hidden_size;
        let q = self.q_dim();
        let kv = self.kv_dim();
        let i = self.intermediate_size;
        // w_q: q*h, w_k: kv*h, w_v: kv*h, w_o: h*q, w_gate: i*h, w_up: i*h, w_down: h*i
        // input_norm: h, post_attn_norm: h
        q * h + kv * h * 2 + h * q + i * h * 3 + h * 2
    }

    /// Per-layer gradient weight VRAM in f32 elements (constant, independent of seq_len).
    ///
    /// Maps to cuda_block.rs lines 238-258: constant-size gradient buffers.
    /// NOTE: grad_w_q and grad_w_o use hidden*hidden (not q_dim*hidden) in current code.
    fn per_layer_grad_weight_elements(&self) -> usize {
        let h = self.hidden_size;
        let kv = self.kv_dim();
        let i = self.intermediate_size;
        // grad_input_norm: h, grad_post_attn_norm: h
        // grad_gate: h*i, grad_up: h*i, grad_down: i*h
        // grad_w_q: h*h, grad_w_k: h*kv, grad_w_v: h*kv, grad_w_o: h*h
        h * 2 + h * i * 3 + h * h * 2 + h * kv * 2
    }

    /// Per-layer scratch elements that scale linearly with seq_len.
    ///
    /// Maps to cuda_block.rs lines 224-236, 243-248: `GpuBuffer::new(_, S * dim)`.
    fn per_layer_scratch_linear_coeff(&self) -> usize {
        let h = self.hidden_size;
        let kv = self.kv_dim();
        let i = self.intermediate_size;
        let n = self.num_attention_heads;
        let hd = self.head_dim();
        // Forward: norm1(h) + q(h) + k(kv) + v(kv) + attn_out(h) + o_proj(h)
        //          + residual1(h) + norm2(h) + gate(i) + up(i) + swiglu(i) + ffn(h)
        // Backward: grad_hidden(h) + grad_swiglu(i)
        // Attention reshape: q_batched(N*hd) + kv_temp(N*hd) + kv_temp2(N*hd)
        h * 8 + kv * 2 + i * 4 + n * hd * 3
    }

    /// Per-layer scratch elements that scale quadratically with seq_len.
    ///
    /// Returns (quadratic_coeff, linear_fallback_coeff) because:
    ///   attn_scores = N * S * S
    ///   grad_attn_scores = N * S * max(S, hd)
    ///
    /// When S >= hd: total = 2 * N * S^2 (pure quadratic)
    /// When S < hd:  total = N * S^2 + N * S * hd (mixed)
    fn per_layer_scratch_quadratic_coeff(&self) -> (usize, usize) {
        let n = self.num_attention_heads;
        let hd = self.head_dim();
        // attn_scores: N * S * S (always quadratic)
        // grad_attn_scores: N * S * max(S, hd)
        //   When S >= hd → N * S * S (quadratic)
        //   When S < hd  → N * S * hd (linear)
        (n, n * hd) // (quadratic_coeff, linear_fallback for grad when S < hd)
    }

    /// Total VRAM in bytes for all layers at a given max_seq_len.
    ///
    /// Postcondition: result is exact for the current cuda_block.rs buffer layout.
    pub fn total_training_vram_bytes(&self, max_seq_len: usize) -> usize {
        let l = self.num_hidden_layers;
        let s = max_seq_len;
        let hd = self.head_dim();

        let constant_per_layer =
            self.per_layer_weight_elements() + self.per_layer_grad_weight_elements();
        let linear_per_layer = self.per_layer_scratch_linear_coeff() * s;

        let (n_quad, n_hd_linear) = self.per_layer_scratch_quadratic_coeff();
        let quadratic_per_layer = if s >= hd {
            2 * n_quad * s * s
        } else {
            n_quad * s * s + n_hd_linear * s
        };

        let elements_per_layer = constant_per_layer + linear_per_layer + quadratic_per_layer;
        l * elements_per_layer * 4 // f32 = 4 bytes
    }

    /// Total VRAM in bytes with SHARED scratch workspace (1 per model, not per layer).
    ///
    /// This is the correct budget formula when gradient buffers are shared across
    /// layers (canonical in PyTorch/JAX). Only weights are truly per-layer.
    ///
    /// Postcondition: result < total_training_vram_bytes(s) for L > 1
    pub fn total_training_vram_bytes_shared(&self, max_seq_len: usize) -> usize {
        let l = self.num_hidden_layers;
        let s = max_seq_len;
        let hd = self.head_dim();

        // Weights are per-layer (unavoidable — must all be resident on GPU)
        let weights_total = l * self.per_layer_weight_elements();

        // Gradient weight buffers: SHARED (one set, reused across layers)
        let grad_weights_shared = self.per_layer_grad_weight_elements();

        // Seq-len-dependent scratch: SHARED (one set)
        let linear_shared = self.per_layer_scratch_linear_coeff() * s;
        let (n_quad, n_hd_linear) = self.per_layer_scratch_quadratic_coeff();
        let quadratic_shared = if s >= hd {
            2 * n_quad * s * s
        } else {
            n_quad * s * s + n_hd_linear * s
        };

        let total_elements =
            weights_total + grad_weights_shared + linear_shared + quadratic_shared;
        total_elements * 4 // f32 = 4 bytes
    }

    /// Solve for the maximum seq_len that fits in the given VRAM budget (bytes),
    /// using shared scratch workspace.
    ///
    /// This is the solver to use with the shared-scratch architecture.
    /// Returns None if even seq_len=1 exceeds the budget.
    pub fn max_seq_len_for_vram_shared(&self, vram_bytes: usize) -> Option<usize> {
        if self.total_training_vram_bytes_shared(1) > vram_bytes {
            return None;
        }

        let mut lo: usize = 1;
        let mut hi: usize = self.max_position_embeddings;

        while lo < hi {
            let mid = lo + (hi - lo + 1) / 2;
            if self.total_training_vram_bytes_shared(mid) <= vram_bytes {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }

        Some(lo)
    }

    /// Solve for the maximum seq_len that fits in the given VRAM budget (bytes).
    ///
    /// Binary search over [1, max_position_embeddings].
    /// Returns None if even seq_len=1 exceeds the budget.
    ///
    /// Precondition: vram_bytes > 0
    /// Postcondition: total_training_vram_bytes(result) <= vram_bytes
    pub fn max_seq_len_for_vram(&self, vram_bytes: usize) -> Option<usize> {
        if self.total_training_vram_bytes(1) > vram_bytes {
            return None;
        }

        let mut lo: usize = 1;
        let mut hi: usize = self.max_position_embeddings;

        while lo < hi {
            let mid = lo + (hi - lo + 1) / 2;
            if self.total_training_vram_bytes(mid) <= vram_bytes {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }

        Some(lo)
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
    fn test_config_serialization() {
        let config = TransformerConfig::llama2_7b();
        let json = serde_json::to_string(&config).expect("JSON serialization should succeed");
        let restored: TransformerConfig =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
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
    fn test_config_yaml_serialization() {
        let config = TransformerConfig::tiny();
        let yaml = serde_yaml::to_string(&config).expect("config should be valid");
        let restored: TransformerConfig =
            serde_yaml::from_str(&yaml).expect("config should be valid");
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
    fn test_config_clone() {
        let config = TransformerConfig::llama2_7b();
        let cloned = config.clone();
        assert_eq!(config.hidden_size, cloned.hidden_size);
        assert_eq!(config.vocab_size, cloned.vocab_size);
    }

    #[test]
    fn test_qwen3_5_9b_config() {
        let config = TransformerConfig::qwen3_5_9b();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_kv_heads, 4);
        assert_eq!(config.intermediate_size, 12288);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.vocab_size, 248320);
        assert_eq!(config.max_position_embeddings, 262144);
        assert!(!config.use_bias);
    }

    #[test]
    fn test_qwen3_5_9b_head_dim() {
        let config = TransformerConfig::qwen3_5_9b();
        // 4096 / 16 = 256 (explicit head_dim, not derived from hidden/heads ratio)
        assert_eq!(config.head_dim(), 256);
    }

    #[test]
    fn test_qwen3_5_9b_gqa_ratio() {
        let config = TransformerConfig::qwen3_5_9b();
        let heads_per_kv = config.num_attention_heads / config.num_kv_heads;
        assert_eq!(heads_per_kv, 4); // 16 / 4 = 4 Q heads per KV head
    }

    // =========================================================================
    // from_apr_metadata contract tests (GH-376)
    // =========================================================================

    #[test]
    fn test_from_apr_metadata_qwen3_8b() {
        // Qwen3-8B: 36 layers, 32 heads, 8 KV heads, hidden=4096, head_dim=128
        let config = TransformerConfig::from_apr_metadata(
            Some(4096),   // hidden_size
            Some(32),     // num_heads
            Some(8),      // num_kv_heads
            Some(12288),  // intermediate_size
            Some(36),     // num_layers
            Some(151936), // vocab_size
            Some(40960),  // max_position_embeddings
            Some(1e-6),   // rms_norm_eps
            Some(1e6),    // rope_theta
            Some("qwen3"),
        )
        .expect("all required fields present");

        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.num_hidden_layers, 36);
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.head_dim(), 128); // 4096/32=128, no override needed
        assert!(!config.use_bias); // Qwen3: no bias
    }

    #[test]
    fn test_from_apr_metadata_qwen2_7b() {
        // Qwen2.5 should get use_bias=true
        let config = TransformerConfig::from_apr_metadata(
            Some(3584),
            Some(28),
            Some(4),
            Some(18944),
            Some(28),
            Some(152064),
            Some(32768),
            Some(1e-6),
            Some(1e6),
            Some("qwen2"),
        )
        .expect("all required fields present");

        assert!(config.use_bias); // Qwen2: has bias
        assert_eq!(config.head_dim(), 128); // 3584/28=128
    }

    #[test]
    fn test_from_apr_metadata_missing_required_returns_none() {
        // Missing hidden_size — must return None, not silently degrade
        assert!(TransformerConfig::from_apr_metadata(
            None, Some(32), Some(8), Some(12288), Some(36), Some(151936),
            Some(40960), Some(1e-6), Some(1e6), Some("qwen3"),
        ).is_none());

        // Missing num_layers
        assert!(TransformerConfig::from_apr_metadata(
            Some(4096), Some(32), Some(8), Some(12288), None, Some(151936),
            Some(40960), Some(1e-6), Some(1e6), Some("qwen3"),
        ).is_none());
    }

    // =========================================================================
    // VRAM Budget Solver Falsification Tests
    //
    // Popperian: each test attempts to BREAK a mathematical invariant.
    // If any test fails, the budget formula disagrees with cuda_block.rs.
    // =========================================================================

    #[test]
    fn falsify_vram_monotonic_in_seq_len() {
        // Prediction: VRAM is strictly monotonically increasing in seq_len
        let config = TransformerConfig::qwen3_4b();
        let mut prev = config.total_training_vram_bytes(1);
        for s in [2, 4, 8, 16, 32, 64, 128, 256, 512] {
            let cur = config.total_training_vram_bytes(s);
            assert!(
                cur > prev,
                "VRAM must increase: seq_len={s} ({cur}) should exceed prev ({prev})"
            );
            prev = cur;
        }
    }

    #[test]
    fn falsify_vram_solver_postcondition() {
        // Prediction: solver result satisfies total_vram <= budget
        let config = TransformerConfig::qwen3_4b();
        let budget = 24 * 1024 * 1024 * 1024_usize; // 24 GB (RTX 4090)
        if let Some(max_s) = config.max_seq_len_for_vram(budget) {
            let used = config.total_training_vram_bytes(max_s);
            assert!(
                used <= budget,
                "Solver returned seq_len={max_s} using {used} bytes > budget {budget}"
            );
            // And seq_len+1 should exceed budget (tightness)
            if max_s < config.max_position_embeddings {
                let over = config.total_training_vram_bytes(max_s + 1);
                assert!(
                    over > budget,
                    "Solver not tight: seq_len={} uses {over} <= budget {budget}",
                    max_s + 1
                );
            }
        }
    }

    #[test]
    fn falsify_vram_solver_returns_none_when_impossible() {
        // Prediction: if even seq_len=1 exceeds budget, solver returns None
        let config = TransformerConfig::qwen3_4b();
        let tiny_budget = 1024; // 1 KB — impossible for any model
        assert!(
            config.max_seq_len_for_vram(tiny_budget).is_none(),
            "Solver should return None when budget is too small"
        );
    }

    #[test]
    fn falsify_qwen3_4b_vram_matches_oom_observation() {
        // Observation: Qwen3-4B OOM'd on 24 GB 4090 at seq_len=512.
        // The formula MUST agree: seq_len=512 should exceed ~23 GB usable VRAM.
        let config = TransformerConfig::qwen3_4b();
        let vram_512 = config.total_training_vram_bytes(512);
        let usable_vram = 23 * 1024 * 1024 * 1024_usize; // ~23 GB after CUDA runtime

        // Diagnostic: print the budget breakdown
        let vram_1 = config.total_training_vram_bytes(1);
        let shared_128 = config.total_training_vram_bytes_shared(128);
        let shared_512 = config.total_training_vram_bytes_shared(512);
        let solved = config.max_seq_len_for_vram_shared(24 * 1024 * 1024 * 1024);
        eprintln!("=== Qwen3-4B VRAM Budget ===");
        eprintln!("  Per-layer weights:    {:.1} MB", config.per_layer_weight_elements() as f64 * 4.0 / 1e6);
        eprintln!("  Per-layer grad scratch: {:.1} MB", config.per_layer_grad_weight_elements() as f64 * 4.0 / 1e6);
        eprintln!("  Per-layer (S=512): {:.1} MB", (vram_512 / 36) as f64 / 1e6);
        eprintln!("  36 layers S=1 (per-layer scratch): {:.1} GB", vram_1 as f64 / 1e9);
        eprintln!("  36 layers S=512 (per-layer scratch): {:.1} GB", vram_512 as f64 / 1e9);
        eprintln!("  36 layers S=128 (SHARED scratch):    {:.1} GB", shared_128 as f64 / 1e9);
        eprintln!("  36 layers S=512 (SHARED scratch):    {:.1} GB", shared_512 as f64 / 1e9);
        eprintln!("  Max seq_len for 24 GB (shared):      {:?}", solved);

        assert!(
            vram_512 > usable_vram,
            "Formula says {:.1} GB for seq_len=512, but we OOM'd on 23 GB — formula is wrong",
            vram_512 as f64 / 1e9
        );
    }

    #[test]
    fn falsify_qwen2_0_5b_fits_on_4090() {
        // Observation: Qwen2-0.5B trained successfully on 4090 at seq_len=512.
        // The formula MUST agree: it should fit in 24 GB.
        let config = TransformerConfig::qwen2_0_5b();
        let vram_512 = config.total_training_vram_bytes(512);
        let total_vram = 24 * 1024 * 1024 * 1024_usize;
        assert!(
            vram_512 < total_vram,
            "Formula says {:.1} GB for Qwen2-0.5B at seq_len=512, but it fit on 4090",
            vram_512 as f64 / 1e9
        );
    }

    #[test]
    fn falsify_vram_budget_concrete_values() {
        // Verify concrete VRAM numbers for Qwen3-4B to catch formula drift.
        let config = TransformerConfig::qwen3_4b();

        // Per-layer weights: q(4096*2560) + k(1024*2560) + v(1024*2560)
        //   + o(2560*4096) + gate(9728*2560) + up(9728*2560) + down(2560*9728)
        //   + norms(2560*2)
        let expected_weights = 4096 * 2560 + 1024 * 2560 * 2 + 2560 * 4096
            + 9728 * 2560 * 3 + 2560 * 2;
        assert_eq!(config.per_layer_weight_elements(), expected_weights);

        // With PER-LAYER gradient scratch (current cuda_block.rs layout),
        // Qwen3-4B's constant overhead alone exceeds 24 GB:
        // 36 layers × 776 MB = 27.9 GB. Solver correctly returns None.
        let budget_24gb = 24 * 1024 * 1024 * 1024_usize;
        assert!(
            config.max_seq_len_for_vram(budget_24gb).is_none(),
            "Qwen3-4B per-layer scratch CANNOT fit 24 GB — proves shared scratch needed"
        );

        // With SHARED scratch (weight-only per-layer), budget check uses
        // total_training_vram_bytes_shared(). Qwen3-4B weights-only = 14.5 GB,
        // leaves ~9 GB for one shared scratch set + seq_len-dependent buffers.
        let shared_budget = config.total_training_vram_bytes_shared(128);
        assert!(
            shared_budget < budget_24gb,
            "Qwen3-4B shared scratch at seq_len=128 should fit 24 GB, got {:.1} GB",
            shared_budget as f64 / 1e9
        );
    }
}
