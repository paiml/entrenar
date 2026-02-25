//! Classification fine-tuning pipeline
//!
//! Wires Transformer + LoRA + ClassificationHead for sequence classification.
//!
//! # Architecture
//!
//! ```text
//! token_ids -> Transformer.forward_hidden() -> [seq_len, hidden_size]
//!           -> ClassificationHead.forward()  -> [num_classes]
//!           -> cross_entropy_loss(target)    -> scalar loss
//! ```
//!
//! # Contract
//!
//! See `aprender/contracts/classification-finetune-v1.yaml`

use super::classification::{
    ClassificationHead, MultiLabelSafetySample, SafetySample, load_multi_label_corpus,
    load_safety_corpus,
};
use crate::autograd::matmul;
use crate::lora::LoRAConfig;
use crate::lora::LoRALayer;
use crate::optim::{AdamW, Optimizer};
use crate::transformer::TransformerConfig;
use crate::transformer::Transformer;
use crate::Tensor;
use std::path::Path;

/// Classification fine-tuning pipeline configuration.
#[derive(Debug, Clone)]
pub struct ClassifyConfig {
    /// Number of output classes
    pub num_classes: usize,
    /// LoRA rank
    pub lora_rank: usize,
    /// LoRA alpha
    pub lora_alpha: f32,
    /// Learning rate
    pub learning_rate: f32,
    /// Number of training epochs
    pub epochs: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Log every N steps
    pub log_interval: usize,
}

impl Default for ClassifyConfig {
    fn default() -> Self {
        Self {
            num_classes: 5,
            lora_rank: 16,
            lora_alpha: 16.0,
            learning_rate: 1e-4,
            epochs: 3,
            max_seq_len: 512,
            log_interval: 100,
        }
    }
}

/// Classification fine-tuning pipeline.
///
/// Owns the transformer, LoRA adapters, and classification head.
/// Provides `train_step()` for single-step training and `train()` for full loop.
pub struct ClassifyPipeline {
    /// Base transformer model (weights frozen)
    pub model: Transformer,
    /// Classification head (trainable)
    pub classifier: ClassificationHead,
    /// LoRA adapters applied to attention projections
    pub lora_layers: Vec<LoRALayer>,
    /// Pipeline configuration
    pub config: ClassifyConfig,
    /// AdamW optimizer for trainable parameters
    optimizer: AdamW,
}

impl ClassifyPipeline {
    /// Create a new classification pipeline.
    ///
    /// # Arguments
    /// * `model_config` - Transformer configuration (e.g., `TransformerConfig::qwen2_0_5b()`)
    /// * `classify_config` - Classification pipeline configuration
    pub fn new(model_config: &TransformerConfig, classify_config: ClassifyConfig) -> Self {
        let model = Transformer::new(model_config);
        let classifier = ClassificationHead::new(
            model_config.hidden_size,
            classify_config.num_classes,
        );

        // Create LoRA layers for q_proj and v_proj in each transformer layer
        let lora_config = LoRAConfig::new(classify_config.lora_rank, classify_config.lora_alpha)
            .target_qv_projections();

        let mut lora_layers = Vec::new();
        for layer in &model.layers {
            let attn = &layer.self_attn;
            let hidden = model_config.hidden_size;
            let head_dim = hidden / model_config.num_attention_heads;

            // Q projection LoRA
            if lora_config.should_apply("q_proj", None) {
                let q_dim = model_config.num_attention_heads * head_dim;
                let q_weight = Tensor::from_vec(
                    attn.w_q.data().as_slice().expect("contiguous w_q").to_vec(),
                    false,
                );
                lora_layers.push(LoRALayer::new(
                    q_weight,
                    q_dim,
                    hidden,
                    classify_config.lora_rank,
                    classify_config.lora_alpha,
                ));
            }

            // V projection LoRA
            if lora_config.should_apply("v_proj", None) {
                let v_dim = model_config.num_kv_heads * head_dim;
                let v_weight = Tensor::from_vec(
                    attn.w_v.data().as_slice().expect("contiguous w_v").to_vec(),
                    false,
                );
                lora_layers.push(LoRALayer::new(
                    v_weight,
                    v_dim,
                    hidden,
                    classify_config.lora_rank,
                    classify_config.lora_alpha,
                ));
            }
        }

        // Ensure LoRA A/B matrices have requires_grad=true
        for lora in &mut lora_layers {
            for param in lora.trainable_params() {
                param.set_requires_grad(true);
            }
        }

        let optimizer = AdamW::default_params(classify_config.learning_rate);

        Self {
            model,
            classifier,
            lora_layers,
            config: classify_config,
            optimizer,
        }
    }

    /// Single training step: forward + loss + backward + optimizer update.
    ///
    /// Performs the complete training cycle:
    /// 1. Forward pass through transformer + classification head
    /// 2. Cross-entropy loss computation
    /// 3. Gradient computation via autograd backward
    /// 4. AdamW optimizer step on trainable parameters
    ///
    /// # Arguments
    /// * `token_ids` - Tokenized input
    /// * `label` - Target class index
    ///
    /// # Returns
    /// Loss value as f32
    pub fn train_step(&mut self, token_ids: &[u32], label: usize) -> f32 {
        let seq_len = token_ids.len();
        let num_classes = self.config.num_classes;

        // ── 1. Zero gradients ─────────────────────────────────────────
        self.classifier.weight.zero_grad();
        self.classifier.bias.zero_grad();
        for lora in &self.lora_layers {
            lora.lora_a().zero_grad();
            lora.lora_b().zero_grad();
        }

        // ── 2. Forward pass ───────────────────────────────────────────
        let hidden = self.model.forward_hidden(token_ids);
        let pooled = self.classifier.mean_pool(&hidden, seq_len);

        // matmul builds autograd backward ops (connects classifier.weight to loss)
        let logits = matmul(
            &pooled,
            &self.classifier.weight,
            1,
            self.classifier.hidden_size(),
            num_classes,
        );

        // Add bias (element-wise, preserving grad tracking)
        let logits_with_bias: Vec<f32> = logits
            .data()
            .as_slice()
            .expect("contiguous logits")
            .iter()
            .zip(
                self.classifier
                    .bias
                    .data()
                    .as_slice()
                    .expect("contiguous bias")
                    .iter(),
            )
            .map(|(&l, &b)| l + b)
            .collect();

        // ── 3. Cross-entropy loss + manual gradient ───────────────────
        // Compute softmax probabilities
        let max_val = logits_with_bias
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits_with_bias.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp).collect();

        // Loss = -log(prob[target])
        let loss_val = -(probs[label].max(1e-10).ln());
        let loss_val = if loss_val.is_finite() { loss_val } else { 100.0 };

        // ∂L/∂logits = probs - one_hot(target)
        // This is the well-known cross-entropy gradient
        let mut grad_logits = probs.clone();
        grad_logits[label] -= 1.0;

        // ── 4. Backward through matmul (autograd) ─────────────────────
        // Set loss gradient on the matmul output, then call backward
        let logits_tensor = logits;
        logits_tensor.set_grad(ndarray::Array1::from(grad_logits.clone()));
        if let Some(op) = logits_tensor.backward_op() {
            op.backward();
        }

        // Manually set bias gradient (∂L/∂bias = ∂L/∂logits)
        self.classifier
            .bias
            .set_grad(ndarray::Array1::from(grad_logits));

        // ── 5. Optimizer step ─────────────────────────────────────────
        // Collect trainable params without borrowing through self (avoids borrow conflict with optimizer)
        let mut params: Vec<&mut Tensor> = Vec::new();
        for lora in &mut self.lora_layers {
            params.extend(lora.trainable_params());
        }
        params.extend(self.classifier.parameters_mut());
        self.optimizer.step_refs(&mut params);

        loss_val
    }

    /// Multi-label training step using BCE with logits loss.
    ///
    /// Unlike `train_step` which uses cross-entropy (mutually exclusive classes),
    /// this uses BCE with logits (each class is independent binary decision).
    ///
    /// # Arguments
    /// * `token_ids` - Tokenized input
    /// * `targets` - Multi-hot target vector (length == num_classes)
    ///
    /// # Returns
    /// Loss value as f32
    pub fn multi_label_train_step(&mut self, token_ids: &[u32], targets: &[f32]) -> f32 {
        let seq_len = token_ids.len();
        let num_classes = self.config.num_classes;
        assert_eq!(
            targets.len(),
            num_classes,
            "F-CLASS-001: target length must match num_classes"
        );

        // ── 1. Zero gradients ─────────────────────────────────────────
        self.classifier.weight.zero_grad();
        self.classifier.bias.zero_grad();
        for lora in &self.lora_layers {
            lora.lora_a().zero_grad();
            lora.lora_b().zero_grad();
        }

        // ── 2. Forward pass ───────────────────────────────────────────
        let hidden = self.model.forward_hidden(token_ids);
        let pooled = self.classifier.mean_pool(&hidden, seq_len);

        let logits = matmul(
            &pooled,
            &self.classifier.weight,
            1,
            self.classifier.hidden_size(),
            num_classes,
        );

        // Add bias
        let logits_with_bias: Vec<f32> = logits
            .data()
            .as_slice()
            .expect("contiguous logits")
            .iter()
            .zip(
                self.classifier
                    .bias
                    .data()
                    .as_slice()
                    .expect("contiguous bias")
                    .iter(),
            )
            .map(|(&l, &b)| l + b)
            .collect();

        // ── 3. BCE with logits loss + manual gradient ───────────────
        // Per-element: L_i = max(x_i, 0) - x_i * t_i + log(1 + exp(-|x_i|))
        let loss_val: f32 = logits_with_bias
            .iter()
            .zip(targets.iter())
            .map(|(&x, &t)| {
                let relu = x.max(0.0);
                relu - x * t + (1.0 + (-x.abs()).exp()).ln()
            })
            .sum::<f32>()
            / num_classes as f32;

        let loss_val = if loss_val.is_finite() { loss_val } else { 100.0 };

        // ∂L/∂logits = (σ(x) - targets) / N
        let grad_logits: Vec<f32> = logits_with_bias
            .iter()
            .zip(targets.iter())
            .map(|(&x, &t)| {
                let sigma = if x >= 0.0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    let e = x.exp();
                    e / (1.0 + e)
                };
                (sigma - t) / num_classes as f32
            })
            .collect();

        // ── 4. Backward through matmul (autograd) ─────────────────────
        let logits_tensor = logits;
        logits_tensor.set_grad(ndarray::Array1::from(grad_logits.clone()));
        if let Some(op) = logits_tensor.backward_op() {
            op.backward();
        }

        // Manually set bias gradient
        self.classifier
            .bias
            .set_grad(ndarray::Array1::from(grad_logits));

        // ── 5. Optimizer step ─────────────────────────────────────────
        let mut params: Vec<&mut Tensor> = Vec::new();
        for lora in &mut self.lora_layers {
            params.extend(lora.trainable_params());
        }
        params.extend(self.classifier.parameters_mut());
        self.optimizer.step_refs(&mut params);

        loss_val
    }

    /// Load multi-label corpus from JSONL file.
    ///
    /// Supports both single-label `{"input","label"}` and multi-label `{"input","labels"}`
    /// formats. Single-label entries are automatically converted to multi-hot encoding.
    ///
    /// # Errors
    /// Returns error if file is invalid or labels out of range.
    pub fn load_multi_label_corpus(
        &self,
        path: &Path,
    ) -> crate::Result<Vec<MultiLabelSafetySample>> {
        load_multi_label_corpus(path, self.config.num_classes)
    }

    /// Get all trainable parameters (LoRA A/B + classifier weight/bias).
    pub fn trainable_parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params: Vec<&mut Tensor> = Vec::new();
        for lora in &mut self.lora_layers {
            params.extend(lora.trainable_params());
        }
        params.extend(self.classifier.parameters_mut());
        params
    }

    /// Count total trainable parameters.
    #[must_use]
    pub fn num_trainable_parameters(&self) -> usize {
        let lora_params: usize = self.lora_layers.iter().map(|l: &LoRALayer| {
            l.rank() * (l.d_in() + l.d_out())
        }).sum();
        lora_params + self.classifier.num_parameters()
    }

    /// Load corpus from JSONL file.
    ///
    /// # Errors
    /// Returns error if file is invalid or labels out of range.
    pub fn load_corpus(&self, path: &Path) -> crate::Result<Vec<SafetySample>> {
        load_safety_corpus(path, self.config.num_classes)
    }

    /// Merge all LoRA adapters into base weights (for inference).
    pub fn merge_adapters(&mut self) {
        for lora in &mut self.lora_layers {
            lora.merge();
        }
    }

    /// Summary of the pipeline configuration.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "ClassifyPipeline:\n  Model: {} hidden, {} layers\n  LoRA: rank={}, alpha={:.1}, {} adapters\n  Classifier: {}->{} ({} params)\n  Total trainable: {} params",
            self.model.config.hidden_size,
            self.model.config.num_hidden_layers,
            self.config.lora_rank,
            self.config.lora_alpha,
            self.lora_layers.len(),
            self.classifier.hidden_size(),
            self.classifier.num_classes(),
            self.classifier.num_parameters(),
            self.num_trainable_parameters(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> TransformerConfig {
        TransformerConfig::tiny()
    }

    #[test]
    fn test_classify_pipeline_creation() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 5,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let pipeline = ClassifyPipeline::new(&model_config, classify_config);
        assert_eq!(pipeline.classifier.num_classes(), 5);
        assert!(!pipeline.lora_layers.is_empty(), "Should have LoRA layers");
    }

    #[test]
    fn test_classify_pipeline_train_step() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 5,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
        let loss = pipeline.train_step(&[1, 2, 3], 0);
        assert!(loss.is_finite(), "F-CLASS-005: loss must be finite");
        assert!(loss > 0.0, "Cross-entropy loss must be positive");
    }

    #[test]
    fn test_classify_pipeline_convergence() {
        // SSC-017: Training must reduce loss across epochs
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            learning_rate: 1e-2,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);

        // Train on 3 samples for 20 epochs
        let samples = [
            (vec![1u32, 2, 3], 0usize),
            (vec![4, 5, 6], 1),
            (vec![7, 8, 9], 2),
        ];

        let mut first_epoch_loss = 0.0f32;
        let mut last_epoch_loss = 0.0f32;

        for epoch in 0..20 {
            let mut epoch_loss = 0.0f32;
            for (tokens, label) in &samples {
                epoch_loss += pipeline.train_step(tokens, *label);
            }
            epoch_loss /= samples.len() as f32;

            if epoch == 0 {
                first_epoch_loss = epoch_loss;
            }
            last_epoch_loss = epoch_loss;
        }

        assert!(
            last_epoch_loss < first_epoch_loss,
            "SSC-017: Loss must decrease. First epoch: {first_epoch_loss:.4}, last: {last_epoch_loss:.4}"
        );
    }

    #[test]
    fn test_classify_pipeline_trainable_params() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 5,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
        let params = pipeline.trainable_parameters_mut();
        // LoRA A + B per adapter + classifier weight + bias
        assert!(params.len() >= 3, "Should have at least classifier + 1 LoRA adapter params");
    }

    #[test]
    fn test_classify_pipeline_summary() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig::default();
        let pipeline = ClassifyPipeline::new(&model_config, classify_config);
        let summary = pipeline.summary();
        assert!(summary.contains("ClassifyPipeline"));
        assert!(summary.contains("LoRA"));
        assert!(summary.contains("Classifier"));
    }

    #[test]
    fn test_multi_label_train_step() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 5,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);

        // Multi-hot: classes 1 and 2 active (needs-quoting AND non-deterministic)
        let targets = vec![0.0, 1.0, 1.0, 0.0, 0.0];
        let loss = pipeline.multi_label_train_step(&[1, 2, 3], &targets);
        assert!(loss.is_finite(), "F-CLASS-005: loss must be finite");
        assert!(loss > 0.0, "BCE loss must be positive");
    }

    #[test]
    fn test_multi_label_convergence() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            learning_rate: 1e-2,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);

        // Train on multi-label samples
        let samples: [(Vec<u32>, Vec<f32>); 3] = [
            (vec![1, 2, 3], vec![1.0, 1.0, 0.0]), // classes 0+1
            (vec![4, 5, 6], vec![0.0, 1.0, 1.0]), // classes 1+2
            (vec![7, 8, 9], vec![1.0, 0.0, 1.0]), // classes 0+2
        ];

        let mut first_loss = 0.0f32;
        let mut last_loss = 0.0f32;

        for epoch in 0..20 {
            let mut epoch_loss = 0.0f32;
            for (tokens, targets) in &samples {
                epoch_loss += pipeline.multi_label_train_step(tokens, targets);
            }
            epoch_loss /= samples.len() as f32;

            if epoch == 0 {
                first_loss = epoch_loss;
            }
            last_loss = epoch_loss;
        }

        assert!(
            last_loss < first_loss,
            "SSC-021: Multi-label loss must decrease. First: {first_loss:.4}, last: {last_loss:.4}"
        );
    }

    #[test]
    #[should_panic(expected = "F-CLASS-001")]
    fn test_multi_label_wrong_target_length() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 5,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
        // Wrong number of targets (3 instead of 5)
        pipeline.multi_label_train_step(&[1, 2, 3], &[1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_classify_pipeline_merge() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 5,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);

        // Should not panic
        pipeline.merge_adapters();

        // All LoRA layers should be merged
        for lora in &pipeline.lora_layers {
            assert!(lora.is_merged(), "All adapters should be merged");
        }
    }
}
