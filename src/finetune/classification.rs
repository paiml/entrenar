//! Classification head and corpus loader for fine-tuning
//!
//! Provides a classifier head that attaches to a transformer's hidden states
//! (mean pooling → linear) and a JSONL corpus loader for safety labels.
//!
//! # Contract
//!
//! See `aprender/contracts/classification-finetune-v1.yaml`:
//! - F-CLASS-001: Logit shape == num_classes
//! - F-CLASS-002: Label index < num_classes
//! - F-CLASS-004: Weight shape == hidden_size * num_classes
//!
//! # Architecture
//!
//! ```text
//! hidden_states [seq_len, hidden_size]
//!   → mean pool → [hidden_size]
//!   → linear    → [num_classes]
//!   → softmax   → class probabilities
//! ```

use crate::autograd::{matmul, Tensor};
use serde::Deserialize;
use std::path::Path;

/// Classification head: mean pool + linear projection.
///
/// Maps transformer hidden states to class logits.
/// Weight shape: [hidden_size * num_classes] (flattened row-major).
/// Bias shape: [num_classes].
pub struct ClassificationHead {
    /// Linear weight [hidden_size, num_classes] flattened
    pub weight: Tensor,
    /// Bias [num_classes]
    pub bias: Tensor,
    /// Input dimension (model hidden_size)
    hidden_size: usize,
    /// Output dimension (number of classes)
    num_classes: usize,
}

impl ClassificationHead {
    /// Create a new classification head with Xavier-initialized weights.
    ///
    /// # Arguments
    /// * `hidden_size` - Transformer hidden dimension (e.g., 896 for Qwen2-0.5B)
    /// * `num_classes` - Number of output classes (e.g., 5 for shell safety)
    ///
    /// # Contract (F-CLASS-004)
    /// Validates hidden_size > 0 and num_classes >= 2.
    pub fn new(hidden_size: usize, num_classes: usize) -> Self {
        assert!(hidden_size > 0, "F-CLASS-004: hidden_size must be > 0");
        assert!(num_classes >= 2, "F-CLASS-004: num_classes must be >= 2");

        // Xavier uniform initialization: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
        let scale = (6.0 / (hidden_size + num_classes) as f32).sqrt();
        let mut rng_state: u64 = 42;
        let weight_data: Vec<f32> = (0..hidden_size * num_classes)
            .map(|_| {
                // Simple LCG for deterministic init
                rng_state = rng_state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let u = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
                (2.0 * u - 1.0) * scale
            })
            .collect();

        let weight = Tensor::from_vec(weight_data, true);
        let bias = Tensor::zeros(num_classes, true);

        Self {
            weight,
            bias,
            hidden_size,
            num_classes,
        }
    }

    /// Forward pass: hidden_states → mean pool → linear → logits.
    ///
    /// # Arguments
    /// * `hidden_states` - Transformer output [seq_len * hidden_size] flattened
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Logits tensor [num_classes]
    ///
    /// # Contract (F-CLASS-001)
    /// Output always has exactly num_classes elements.
    pub fn forward(&self, hidden_states: &Tensor, seq_len: usize) -> Tensor {
        // Mean pool across sequence dimension
        let pooled = self.mean_pool(hidden_states, seq_len);

        // Linear: pooled [1, hidden_size] @ weight [hidden_size, num_classes] = [1, num_classes]
        let logits = matmul(&pooled, &self.weight, 1, self.hidden_size, self.num_classes);

        // Add bias
        let logits_data: Vec<f32> = logits
            .data()
            .as_slice()
            .expect("contiguous logits data")
            .iter()
            .zip(
                self.bias
                    .data()
                    .as_slice()
                    .expect("contiguous bias data")
                    .iter(),
            )
            .map(|(&l, &b)| l + b)
            .collect();

        Tensor::from_vec(logits_data, logits.requires_grad())
    }

    /// Mean pool hidden states across sequence length.
    ///
    /// hidden_states: [seq_len * hidden_size] → [hidden_size]
    pub fn mean_pool(&self, hidden_states: &Tensor, seq_len: usize) -> Tensor {
        let data = hidden_states.data();
        let slice = data.as_slice().expect("contiguous hidden states");
        let h = self.hidden_size;

        let mut pooled = vec![0.0f32; h];
        for pos in 0..seq_len {
            let start = pos * h;
            for j in 0..h {
                pooled[j] += slice[start + j];
            }
        }
        let inv_len = 1.0 / seq_len as f32;
        for v in &mut pooled {
            *v *= inv_len;
        }

        Tensor::from_vec(pooled, hidden_states.requires_grad())
    }

    /// Get trainable parameters (weight + bias).
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }

    /// Get parameters (immutable).
    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    /// Number of classes.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Hidden size.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Total trainable parameter count.
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        self.hidden_size * self.num_classes + self.num_classes
    }
}

// =============================================================================
// CORPUS LOADER
// =============================================================================

/// A single shell safety corpus sample (single-label).
#[derive(Debug, Clone, Deserialize)]
pub struct SafetySample {
    /// Shell script content
    pub input: String,
    /// Safety class index (0-4)
    pub label: usize,
}

impl SafetySample {
    /// Convert the input text to token IDs using byte-level encoding.
    ///
    /// Each byte of the UTF-8 representation is mapped to a `u32` token ID.
    /// This provides a simple, deterministic tokenization suitable for the
    /// classification pipeline. For production use with large vocabularies,
    /// an external tokenizer (BPE, SentencePiece) should be used before
    /// calling `train_step` directly.
    #[must_use]
    pub fn input_ids(&self) -> Vec<u32> {
        self.input.bytes().map(u32::from).collect()
    }
}

/// A multi-label shell safety corpus sample.
///
/// A script can have multiple active labels (e.g., both non-deterministic AND needs-quoting).
/// Labels are a multi-hot vector: `[0.0, 1.0, 1.0, 0.0, 0.0]` means classes 1 and 2 are active.
#[derive(Debug, Clone, Deserialize)]
pub struct MultiLabelSafetySample {
    /// Shell script content
    pub input: String,
    /// Multi-hot label vector (length == num_classes)
    pub labels: Vec<f32>,
}

impl MultiLabelSafetySample {
    /// Create from a single-label sample by converting to multi-hot encoding.
    pub fn from_single_label(sample: &SafetySample, num_classes: usize) -> Self {
        let mut labels = vec![0.0f32; num_classes];
        if sample.label < num_classes {
            labels[sample.label] = 1.0;
        }
        Self {
            input: sample.input.clone(),
            labels,
        }
    }

    /// Active class indices (where labels[i] > 0.5).
    pub fn active_classes(&self) -> Vec<usize> {
        self.labels
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 0.5)
            .map(|(i, _)| i)
            .collect()
    }
}

/// Corpus statistics.
#[derive(Debug, Clone)]
pub struct SafetyCorpusStats {
    /// Total samples
    pub total: usize,
    /// Samples per class
    pub class_counts: Vec<usize>,
    /// Average input length (chars)
    pub avg_input_len: usize,
}

/// Load shell safety corpus from JSONL file.
///
/// Each line is `{"input": "...", "label": N}` where N is 0-4.
///
/// # Contract (F-CLASS-002)
/// All labels must be < num_classes.
///
/// # Errors
/// Returns error if file cannot be read or contains invalid labels.
pub fn load_safety_corpus(path: &Path, num_classes: usize) -> crate::Result<Vec<SafetySample>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| crate::Error::Io(format!("Corpus file not found: {}: {e}", path.display())))?;

    let mut samples = Vec::new();
    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let sample: SafetySample = serde_json::from_str(line).map_err(|e| {
            crate::Error::ConfigError(format!("Invalid JSONL at line {}: {e}", line_num + 1))
        })?;

        // F-CLASS-002: label bounds check
        if sample.label >= num_classes {
            return Err(crate::Error::ConfigError(format!(
                "F-CLASS-002: label {} at line {} out of range (num_classes={num_classes})",
                sample.label,
                line_num + 1,
            )));
        }

        samples.push(sample);
    }

    Ok(samples)
}

/// Compute corpus statistics.
pub fn corpus_stats(samples: &[SafetySample], num_classes: usize) -> SafetyCorpusStats {
    let mut class_counts = vec![0usize; num_classes];
    let mut total_len = 0usize;

    for s in samples {
        if s.label < num_classes {
            class_counts[s.label] += 1;
        }
        total_len += s.input.len();
    }

    SafetyCorpusStats {
        total: samples.len(),
        class_counts,
        avg_input_len: if samples.is_empty() {
            0
        } else {
            total_len / samples.len()
        },
    }
}

/// Load multi-label corpus from JSONL file.
///
/// Supports two formats:
/// - Single-label: `{"input": "...", "label": N}` → converts to multi-hot
/// - Multi-label: `{"input": "...", "labels": [0.0, 1.0, 1.0, 0.0, 0.0]}`
///
/// # Errors
/// Returns error if file cannot be read or labels are invalid.
pub fn load_multi_label_corpus(
    path: &Path,
    num_classes: usize,
) -> crate::Result<Vec<MultiLabelSafetySample>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| crate::Error::Io(format!("Corpus file not found: {}: {e}", path.display())))?;

    let mut samples = Vec::new();
    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Try multi-label format first
        if let Ok(sample) = serde_json::from_str::<MultiLabelSafetySample>(line) {
            if sample.labels.len() != num_classes {
                return Err(crate::Error::ConfigError(format!(
                    "F-CLASS-001: labels length {} at line {} != num_classes {num_classes}",
                    sample.labels.len(),
                    line_num + 1,
                )));
            }
            samples.push(sample);
        } else if let Ok(single) = serde_json::from_str::<SafetySample>(line) {
            if single.label >= num_classes {
                return Err(crate::Error::ConfigError(format!(
                    "F-CLASS-002: label {} at line {} out of range (num_classes={num_classes})",
                    single.label,
                    line_num + 1,
                )));
            }
            samples.push(MultiLabelSafetySample::from_single_label(
                &single,
                num_classes,
            ));
        } else {
            return Err(crate::Error::ConfigError(format!(
                "Invalid JSONL at line {}: unrecognized format",
                line_num + 1,
            )));
        }
    }

    Ok(samples)
}

/// BCE with logits loss for multi-label classification.
///
/// Per-element: `L_i = max(x_i, 0) - x_i * t_i + log(1 + exp(-|x_i|))`
/// Total: `L = mean(L_i)`
///
/// # Contract (F-CLASS-005)
/// Output is finite (no NaN/Inf).
pub fn bce_with_logits_loss(logits: &Tensor, targets: &[f32], num_classes: usize) -> Tensor {
    let data = logits.data();
    let slice = data.as_slice().expect("contiguous logits");
    assert_eq!(
        slice.len(),
        num_classes,
        "F-CLASS-001: logit shape mismatch"
    );
    assert_eq!(
        targets.len(),
        num_classes,
        "F-CLASS-001: target shape mismatch"
    );

    let total_loss: f32 = slice
        .iter()
        .zip(targets.iter())
        .map(|(&x, &t)| {
            let relu = x.max(0.0);
            relu - x * t + (1.0 + (-x.abs()).exp()).ln()
        })
        .sum::<f32>()
        / num_classes as f32;

    // F-CLASS-005: finite check
    let total_loss = if total_loss.is_finite() {
        total_loss
    } else {
        100.0
    };

    Tensor::from_vec(vec![total_loss], logits.requires_grad())
}

/// Cross-entropy loss for classification.
///
/// # Arguments
/// * `logits` - Raw logits [num_classes]
/// * `target` - Target class index
/// * `num_classes` - Number of classes
///
/// # Returns
/// Scalar loss value (as single-element Tensor)
///
/// # Contract (F-CLASS-005)
/// Output is finite (no NaN/Inf).
pub fn cross_entropy_loss(logits: &Tensor, target: usize, num_classes: usize) -> Tensor {
    let data = logits.data();
    let slice = data.as_slice().expect("contiguous logits");
    assert_eq!(
        slice.len(),
        num_classes,
        "F-CLASS-001: logit shape mismatch"
    );
    assert!(target < num_classes, "F-CLASS-002: label out of range");

    // Numerically stable log-softmax: log(softmax(x_i)) = x_i - max - log(sum(exp(x_j - max)))
    let max_val = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp: f32 = slice.iter().map(|&v| (v - max_val).exp()).sum::<f32>().ln() + max_val;
    let loss = -(slice[target] - log_sum_exp);

    // F-CLASS-005: finite check
    let loss = if loss.is_finite() { loss } else { 100.0 };

    Tensor::from_vec(vec![loss], logits.requires_grad())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classification_head_shape() {
        let head = ClassificationHead::new(128, 5);
        assert_eq!(head.num_classes(), 5);
        assert_eq!(head.hidden_size(), 128);
        assert_eq!(head.num_parameters(), 128 * 5 + 5);
    }

    #[test]
    fn test_classification_head_forward() {
        let head = ClassificationHead::new(64, 5);
        // Simulate hidden states: 3 tokens, hidden_size=64
        let hidden = Tensor::from_vec(vec![0.1f32; 3 * 64], false);
        let logits = head.forward(&hidden, 3);
        assert_eq!(logits.len(), 5, "F-CLASS-001: must produce 5 logits");
    }

    #[test]
    fn test_classification_head_parameters() {
        let mut head = ClassificationHead::new(64, 5);
        assert_eq!(head.parameters().len(), 2); // weight + bias
        assert_eq!(head.parameters_mut().len(), 2);
    }

    #[test]
    fn test_cross_entropy_loss_finite() {
        let logits = Tensor::from_vec(vec![1.0, 2.0, -1.0, 0.5, 3.0], false);
        let loss = cross_entropy_loss(&logits, 2, 5);
        let loss_val = loss.data()[0];
        assert!(loss_val.is_finite(), "F-CLASS-005: loss must be finite");
        assert!(loss_val > 0.0, "Cross-entropy loss must be positive");
    }

    #[test]
    fn test_cross_entropy_loss_correct_class() {
        // If logit for target class is much larger, loss should be small
        let logits = Tensor::from_vec(vec![-100.0, -100.0, 100.0, -100.0, -100.0], false);
        let loss = cross_entropy_loss(&logits, 2, 5);
        let loss_val = loss.data()[0];
        assert!(
            loss_val < 0.01,
            "Loss for correct high-confidence should be ~0"
        );
    }

    #[test]
    fn test_cross_entropy_loss_wrong_class() {
        // If logit for target class is much smaller, loss should be large
        let logits = Tensor::from_vec(vec![100.0, -100.0, -100.0, -100.0, -100.0], false);
        let loss = cross_entropy_loss(&logits, 2, 5);
        let loss_val = loss.data()[0];
        assert!(loss_val > 1.0, "Loss for wrong class should be large");
    }

    #[test]
    fn test_mean_pool() {
        let head = ClassificationHead::new(4, 2);
        // 2 tokens, hidden_size=4: [[1,2,3,4], [5,6,7,8]] → mean = [3,4,5,6]
        let hidden = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], false);
        let pooled = head.mean_pool(&hidden, 2);
        let data = pooled.data();
        let slice = data.as_slice().expect("contiguous");
        assert_eq!(slice.len(), 4);
        assert!((slice[0] - 3.0).abs() < 1e-6);
        assert!((slice[1] - 4.0).abs() < 1e-6);
        assert!((slice[2] - 5.0).abs() < 1e-6);
        assert!((slice[3] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_corpus_stats_empty() {
        let stats = corpus_stats(&[], 5);
        assert_eq!(stats.total, 0);
        assert_eq!(stats.class_counts, vec![0; 5]);
    }

    #[test]
    fn test_corpus_stats_distribution() {
        let samples = vec![
            SafetySample {
                input: "echo hello".into(),
                label: 0,
            },
            SafetySample {
                input: "echo $HOME".into(),
                label: 1,
            },
            SafetySample {
                input: "echo $RANDOM".into(),
                label: 2,
            },
            SafetySample {
                input: "mkdir /tmp/x".into(),
                label: 3,
            },
            SafetySample {
                input: "eval $x".into(),
                label: 4,
            },
            SafetySample {
                input: "ls".into(),
                label: 0,
            },
        ];
        let stats = corpus_stats(&samples, 5);
        assert_eq!(stats.total, 6);
        assert_eq!(stats.class_counts, vec![2, 1, 1, 1, 1]);
    }

    #[test]
    #[should_panic(expected = "F-CLASS-001")]
    fn test_cross_entropy_wrong_logit_count() {
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        let _ = cross_entropy_loss(&logits, 0, 5);
    }

    #[test]
    #[should_panic(expected = "F-CLASS-002")]
    fn test_cross_entropy_label_out_of_range() {
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], false);
        let _ = cross_entropy_loss(&logits, 5, 5);
    }

    // ── Multi-label tests ──────────────────────────────────────────

    #[test]
    fn test_multi_label_from_single_label() {
        let single = SafetySample {
            input: "echo $RANDOM".into(),
            label: 2,
        };
        let multi = MultiLabelSafetySample::from_single_label(&single, 5);
        assert_eq!(multi.labels, vec![0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(multi.active_classes(), vec![2]);
    }

    #[test]
    fn test_multi_label_active_classes() {
        let sample = MultiLabelSafetySample {
            input: "echo $RANDOM $HOME".into(),
            labels: vec![0.0, 1.0, 1.0, 0.0, 0.0],
        };
        assert_eq!(sample.active_classes(), vec![1, 2]);
    }

    #[test]
    fn test_multi_label_no_active_classes() {
        let sample = MultiLabelSafetySample {
            input: "".into(),
            labels: vec![0.0, 0.0, 0.0, 0.0, 0.0],
        };
        assert!(sample.active_classes().is_empty());
    }

    #[test]
    fn test_multi_label_all_active() {
        let sample = MultiLabelSafetySample {
            input: "eval $RANDOM; mkdir /x".into(),
            labels: vec![1.0, 1.0, 1.0, 1.0, 1.0],
        };
        assert_eq!(sample.active_classes(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_bce_with_logits_loss_basic() {
        let logits = Tensor::from_vec(vec![2.0, -1.0, 0.5, -2.0, 3.0], false);
        let targets = [1.0, 0.0, 1.0, 0.0, 0.0];
        let loss = bce_with_logits_loss(&logits, &targets, 5);
        let loss_val = loss.data()[0];
        assert!(loss_val.is_finite(), "F-CLASS-005: loss must be finite");
        assert!(loss_val > 0.0, "BCE loss must be positive");
    }

    #[test]
    fn test_bce_with_logits_loss_perfect() {
        let logits = Tensor::from_vec(vec![100.0, -100.0, 100.0, -100.0, -100.0], false);
        let targets = [1.0, 0.0, 1.0, 0.0, 0.0];
        let loss = bce_with_logits_loss(&logits, &targets, 5);
        assert!(
            loss.data()[0] < 0.01,
            "Perfect prediction should have near-zero loss"
        );
    }

    #[test]
    #[should_panic(expected = "F-CLASS-001")]
    fn test_bce_logit_shape_mismatch() {
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        let _ = bce_with_logits_loss(&logits, &[1.0, 0.0, 1.0, 0.0, 0.0], 5);
    }

    #[test]
    #[should_panic(expected = "F-CLASS-001")]
    fn test_bce_target_shape_mismatch() {
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], false);
        let _ = bce_with_logits_loss(&logits, &[1.0, 0.0, 1.0], 5);
    }
}
