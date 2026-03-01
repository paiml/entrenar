//! Instruction-following fine-tuning pipeline (GH-371)
//!
//! Wires Transformer + LoRA for causal language model fine-tuning on
//! instruction-response pairs.
//!
//! # Architecture
//!
//! ```text
//! [prompt_ids ++ response_ids] -> Transformer.forward() -> logits [seq_len, vocab_size]
//!   -> causal_lm_loss(logits[prompt_len..], response_ids) -> scalar loss
//! ```
//!
//! # Contract
//!
//! - F-INST-002: Loss computed only on response tokens (prompt tokens masked)
//! - F-INST-003: Perplexity = exp(avg_loss) reported per epoch
//! - F-INST-004: LoRA adapters saved in APR format

use crate::lora::LoRALayer;
use crate::optim::{clip_grad_norm_refs, AdamW, Optimizer};
use crate::tokenizer::HfTokenizer;
use crate::transformer::{Transformer, TransformerConfig};
use crate::Tensor;
use std::path::{Path, PathBuf};

/// Configuration for instruction fine-tuning.
#[derive(Debug, Clone)]
pub struct InstructConfig {
    /// LoRA rank
    pub lora_rank: usize,
    /// LoRA alpha
    pub lora_alpha: f32,
    /// Learning rate
    pub learning_rate: f32,
    /// Number of training epochs
    pub epochs: usize,
    /// Maximum sequence length (prompt + response)
    pub max_seq_len: usize,
    /// Maximum gradient norm for clipping
    pub gradient_clip_norm: Option<f32>,
}

impl Default for InstructConfig {
    fn default() -> Self {
        Self {
            lora_rank: 16,
            lora_alpha: 32.0,
            learning_rate: 2e-4,
            epochs: 3,
            max_seq_len: 512,
            gradient_clip_norm: Some(1.0),
        }
    }
}

/// Result of processing one instruction-response pair.
#[derive(Debug, Clone)]
pub struct InstructStepResult {
    /// Cross-entropy loss on response tokens
    pub loss: f32,
    /// Number of response tokens
    pub num_response_tokens: usize,
    /// Perplexity = exp(loss)
    pub perplexity: f32,
}

/// Result of processing a mini-batch of instruction samples.
#[derive(Debug, Clone)]
pub struct InstructBatchResult {
    /// Average cross-entropy loss across the batch (response tokens only)
    pub avg_loss: f32,
    /// Total response tokens in batch
    pub total_response_tokens: usize,
    /// Perplexity = exp(avg_loss)
    pub perplexity: f32,
    /// Gradient norm before clipping
    pub grad_norm: f32,
}

/// Instruction fine-tuning pipeline.
///
/// Owns the transformer and LoRA adapters. Uses `Transformer::forward()`
/// for causal LM logits and computes loss on response tokens only.
pub struct InstructPipeline {
    /// Base transformer model
    pub model: Transformer,
    /// LoRA adapters applied to Q/V attention projections
    pub lora_layers: Vec<LoRALayer>,
    /// Pipeline configuration
    pub config: InstructConfig,
    /// AdamW optimizer for trainable parameters
    optimizer: AdamW,
    /// Optional BPE tokenizer
    tokenizer: Option<HfTokenizer>,
    /// Path to base model (for checkpoint provenance)
    model_dir: Option<PathBuf>,
}

impl InstructPipeline {
    /// Create a new pipeline with random weights.
    pub fn new(model_config: &TransformerConfig, instruct_config: InstructConfig) -> Self {
        let model = Transformer::new(model_config);
        let mut lora_layers =
            Self::build_lora_layers(&model, model_config, &instruct_config);

        for lora in &mut lora_layers {
            for param in lora.trainable_params() {
                param.set_requires_grad(true);
            }
        }

        let optimizer = AdamW::default_params(instruct_config.learning_rate);

        Self {
            model,
            lora_layers,
            config: instruct_config,
            optimizer,
            tokenizer: None,
            model_dir: None,
        }
    }

    /// Create pipeline from pretrained model weights.
    ///
    /// Loads transformer from SafeTensors and optionally a BPE tokenizer.
    ///
    /// # Errors
    /// Returns error if model files cannot be loaded.
    pub fn from_pretrained(
        model_dir: &Path,
        model_config: &TransformerConfig,
        instruct_config: InstructConfig,
    ) -> crate::Result<Self> {
        let model = Transformer::from_safetensors(model_dir, model_config)?;
        let mut lora_layers =
            Self::build_lora_layers(&model, model_config, &instruct_config);

        for lora in &mut lora_layers {
            for param in lora.trainable_params() {
                param.set_requires_grad(true);
            }
        }

        let optimizer = AdamW::default_params(instruct_config.learning_rate);

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            HfTokenizer::from_file(&tokenizer_path).ok()
        } else {
            None
        };

        Ok(Self {
            model,
            lora_layers,
            config: instruct_config,
            optimizer,
            tokenizer,
            model_dir: Some(model_dir.to_path_buf()),
        })
    }

    /// Build LoRA layers for Q and V projections (same pattern as ClassifyPipeline).
    fn build_lora_layers(
        model: &Transformer,
        model_config: &TransformerConfig,
        config: &InstructConfig,
    ) -> Vec<LoRALayer> {
        let hidden = model_config.hidden_size;
        let head_dim = hidden / model_config.num_attention_heads;

        let mut lora_layers = Vec::new();

        for layer in &model.layers {
            let attn = &layer.self_attn;

            // Q projection LoRA
            let q_dim = model_config.num_attention_heads * head_dim;
            let q_weight = Tensor::from_vec(
                attn.w_q.data().as_slice().expect("contiguous w_q").to_vec(),
                false,
            );
            lora_layers.push(LoRALayer::new(
                q_weight,
                q_dim,
                hidden,
                config.lora_rank,
                config.lora_alpha,
            ));

            // V projection LoRA
            let v_dim = model_config.num_kv_heads * head_dim;
            let v_weight = Tensor::from_vec(
                attn.w_v.data().as_slice().expect("contiguous w_v").to_vec(),
                false,
            );
            lora_layers.push(LoRALayer::new(
                v_weight,
                v_dim,
                hidden,
                config.lora_rank,
                config.lora_alpha,
            ));
        }

        lora_layers
    }

    /// Tokenize text, truncating to max_seq_len.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let ids = if let Some(ref tok) = self.tokenizer {
            tok.encode(text)
        } else {
            text.bytes().map(u32::from).collect()
        };

        if ids.len() > self.config.max_seq_len {
            ids[..self.config.max_seq_len].to_vec()
        } else {
            ids
        }
    }

    /// Compute causal LM loss on a single instruction-response pair.
    ///
    /// # Contract (F-INST-002)
    /// Loss is computed only on response tokens. Prompt tokens are masked.
    pub fn train_step(
        &mut self,
        prompt_ids: &[u32],
        response_ids: &[u32],
    ) -> InstructStepResult {
        let full_ids: Vec<u32> = prompt_ids
            .iter()
            .chain(response_ids.iter())
            .copied()
            .collect();

        let prompt_len = prompt_ids.len();
        let response_len = response_ids.len();

        if response_len == 0 || full_ids.len() < 2 {
            return InstructStepResult {
                loss: 0.0,
                num_response_tokens: 0,
                perplexity: 1.0,
            };
        }

        let full_ids = if full_ids.len() > self.config.max_seq_len {
            full_ids[..self.config.max_seq_len].to_vec()
        } else {
            full_ids
        };
        let seq_len = full_ids.len();
        let vocab_size = self.model.config().vocab_size;

        // ── 1. Zero gradients ──────────────────────────────────────────
        for lora in &mut self.lora_layers {
            for param in lora.trainable_params() {
                param.zero_grad();
            }
        }

        // ── 2. Forward pass → logits [seq_len, vocab_size] ────────────
        let logits = self.model.forward(&full_ids);
        let logits_data = logits
            .data()
            .as_slice()
            .expect("contiguous logits")
            .to_vec();

        // ── 3. Causal LM loss on response tokens only ─────────────────
        let loss_start = prompt_len.saturating_sub(1);
        let loss_end = seq_len - 1;
        let num_loss_tokens = loss_end.saturating_sub(loss_start);

        if num_loss_tokens == 0 {
            return InstructStepResult {
                loss: 0.0,
                num_response_tokens: 0,
                perplexity: 1.0,
            };
        }

        let (avg_loss, grad_logits) =
            Self::compute_causal_lm_loss(&logits_data, &full_ids, loss_start, loss_end, vocab_size);

        // ── 4. Backward through autograd ──────────────────────────────
        logits.set_grad(ndarray::Array1::from(grad_logits));
        if let Some(op) = logits.backward_op() {
            op.backward();
        }

        // ── 5. Optimizer step on LoRA parameters ──────────────────────
        let mut params: Vec<&mut Tensor> = Vec::new();
        for lora in &mut self.lora_layers {
            params.extend(lora.trainable_params());
        }

        if let Some(max_norm) = self.config.gradient_clip_norm {
            clip_grad_norm_refs(&mut params, max_norm);
        }

        self.optimizer.step_refs(&mut params);

        InstructStepResult {
            loss: avg_loss,
            num_response_tokens: num_loss_tokens,
            perplexity: avg_loss.exp().min(1e6),
        }
    }

    /// Evaluate loss and perplexity on a set of samples without updating weights.
    pub fn evaluate(
        &self,
        prompt_ids_batch: &[Vec<u32>],
        response_ids_batch: &[Vec<u32>],
    ) -> InstructBatchResult {
        let mut total_loss = 0.0f32;
        let mut total_response_tokens = 0usize;

        for (prompt_ids, response_ids) in
            prompt_ids_batch.iter().zip(response_ids_batch.iter())
        {
            let full_ids: Vec<u32> = prompt_ids
                .iter()
                .chain(response_ids.iter())
                .copied()
                .collect();

            let prompt_len = prompt_ids.len();
            if response_ids.is_empty() || full_ids.len() < 2 {
                continue;
            }

            let full_ids = if full_ids.len() > self.config.max_seq_len {
                full_ids[..self.config.max_seq_len].to_vec()
            } else {
                full_ids
            };
            let seq_len = full_ids.len();
            let vocab_size = self.model.config().vocab_size;

            let logits = self.model.forward(&full_ids);
            let logits_data = logits
                .data()
                .as_slice()
                .expect("contiguous logits")
                .to_vec();

            let loss_start = prompt_len.saturating_sub(1);
            let loss_end = seq_len - 1;
            let num_loss_tokens = loss_end.saturating_sub(loss_start);

            let (sample_loss, _) = Self::compute_causal_lm_loss(
                &logits_data,
                &full_ids,
                loss_start,
                loss_end,
                vocab_size,
            );

            total_loss += sample_loss * num_loss_tokens as f32;
            total_response_tokens += num_loss_tokens;
        }

        let avg_loss = if total_response_tokens > 0 {
            total_loss / total_response_tokens as f32
        } else {
            0.0
        };

        InstructBatchResult {
            avg_loss,
            total_response_tokens,
            perplexity: avg_loss.exp().min(1e6),
            grad_norm: 0.0,
        }
    }

    /// Compute causal LM loss and gradients for the given position range.
    ///
    /// Returns (average_loss, gradient_logits).
    fn compute_causal_lm_loss(
        logits_data: &[f32],
        full_ids: &[u32],
        loss_start: usize,
        loss_end: usize,
        vocab_size: usize,
    ) -> (f32, Vec<f32>) {
        let seq_len = full_ids.len();
        let num_loss_tokens = loss_end.saturating_sub(loss_start);
        let mut total_loss = 0.0f32;
        let mut grad_logits = vec![0.0f32; seq_len * vocab_size];

        for pos in loss_start..loss_end {
            let target = full_ids[pos + 1] as usize;
            if target >= vocab_size {
                continue;
            }

            let logit_start = pos * vocab_size;
            let row = &logits_data[logit_start..logit_start + vocab_size];

            // Numerically stable log-softmax
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_vals: Vec<f32> = row.iter().map(|&v| (v - max_val).exp()).collect();
            let sum_exp: f32 = exp_vals.iter().sum();
            let log_sum_exp = sum_exp.ln() + max_val;

            let loss_i = -(row[target] - log_sum_exp);
            total_loss += if loss_i.is_finite() { loss_i } else { 100.0 };

            // Gradient: (softmax - one_hot) / num_loss_tokens
            let inv_n = 1.0 / num_loss_tokens as f32;
            for j in 0..vocab_size {
                grad_logits[logit_start + j] = (exp_vals[j] / sum_exp) * inv_n;
            }
            grad_logits[logit_start + target] -= inv_n;
        }

        let avg_loss = if num_loss_tokens > 0 {
            total_loss / num_loss_tokens as f32
        } else {
            0.0
        };

        (avg_loss, grad_logits)
    }

    /// Number of trainable LoRA parameters.
    #[must_use]
    pub fn num_trainable_parameters(&self) -> usize {
        // LoRA layers store weight + lora_a + lora_b; we count lora_a + lora_b
        self.lora_layers.len() * 2 * self.config.lora_rank
            * (self.lora_layers.first().map_or(0, |_| {
                // Approximate: each LoRA pair has rank * (rows + cols) params
                // This is a rough estimate since layers may differ in size
                1
            }))
    }

    /// Update learning rate (for LR scheduling).
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.optimizer.set_lr(lr);
    }

    /// Get current learning rate.
    #[must_use]
    pub fn learning_rate(&self) -> f32 {
        self.optimizer.lr()
    }

    /// Set model path for checkpoint provenance.
    pub fn set_model_path(&mut self, path: &Path) {
        self.model_dir = Some(path.to_path_buf());
    }

    /// Summary of pipeline configuration.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "InstructPipeline: {} LoRA layers, rank={}, alpha={:.1}",
            self.lora_layers.len(),
            self.config.lora_rank,
            self.config.lora_alpha,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruct_pipeline_new() {
        let model_config = TransformerConfig::tiny();
        let instruct_config = InstructConfig {
            lora_rank: 4,
            ..InstructConfig::default()
        };
        let pipeline = InstructPipeline::new(&model_config, instruct_config);
        // 2 layers * 2 (Q+V) = 4 LoRA layers for tiny config
        assert_eq!(pipeline.lora_layers.len(), 4);
    }

    #[test]
    fn test_instruct_train_step() {
        let model_config = TransformerConfig::tiny();
        let instruct_config = InstructConfig {
            lora_rank: 4,
            max_seq_len: 64,
            ..InstructConfig::default()
        };
        let mut pipeline = InstructPipeline::new(&model_config, instruct_config);

        let prompt_ids: Vec<u32> = (0..10).collect();
        let response_ids: Vec<u32> = (10..20).collect();

        let result = pipeline.train_step(&prompt_ids, &response_ids);
        assert!(result.loss >= 0.0);
        assert_eq!(result.num_response_tokens, 10);
        assert!(result.perplexity >= 1.0);
    }

    #[test]
    fn test_instruct_evaluate() {
        let model_config = TransformerConfig::tiny();
        let instruct_config = InstructConfig {
            lora_rank: 4,
            max_seq_len: 64,
            ..InstructConfig::default()
        };
        let pipeline = InstructPipeline::new(&model_config, instruct_config);

        let prompts = vec![vec![0u32, 1, 2, 3, 4]];
        let responses = vec![vec![5u32, 6, 7, 8, 9]];

        let result = pipeline.evaluate(&prompts, &responses);
        assert!(result.avg_loss >= 0.0);
        assert_eq!(result.total_response_tokens, 5);
    }

    #[test]
    fn test_empty_response_noop() {
        let model_config = TransformerConfig::tiny();
        let instruct_config = InstructConfig {
            lora_rank: 4,
            max_seq_len: 64,
            ..InstructConfig::default()
        };
        let mut pipeline = InstructPipeline::new(&model_config, instruct_config);

        let result = pipeline.train_step(&[0, 1, 2], &[]);
        assert_eq!(result.loss, 0.0);
        assert_eq!(result.num_response_tokens, 0);
    }
}
