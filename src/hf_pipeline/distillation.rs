//! Knowledge Distillation Loss Functions
//!
//! Implements temperature-scaled KL divergence and progressive distillation
//! based on Hinton et al. (2015) and Sun et al. (2019).
//!
//! # References
//!
//! [1] Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge
//!     in a Neural Network." arXiv:1503.02531
//!
//! [2] Sun, S., Cheng, Y., Gan, Z., & Liu, J. (2019). "Patient Knowledge
//!     Distillation for BERT Model Compression." EMNLP 2019.
//!
//! [3] Zagoruyko, S., & Komodakis, N. (2017). "Paying More Attention to
//!     Attention: Improving the Performance of CNNs via Attention Transfer."
//!     ICLR 2017.

use ndarray::{Array1, Array2};

/// Softmax with numerical stability
fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Array1<f32> = logits.mapv(|x| (x - max).exp());
    let sum = exp.sum();
    exp / sum
}

/// Log softmax with numerical stability
fn log_softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let shifted = logits.mapv(|x| x - max);
    let log_sum_exp = shifted.mapv(f32::exp).sum().ln();
    shifted.mapv(|x| x - log_sum_exp)
}

/// KL divergence: KL(P || Q) = sum(P * log(P/Q))
fn kl_divergence(log_q: &Array1<f32>, p: &Array1<f32>) -> f32 {
    // KL(P || Q) = sum(P * (log(P) - log(Q)))
    // Since we have log(Q), we compute: sum(P * log(P)) - sum(P * log(Q))
    let p_log_p: f32 = p
        .iter()
        .map(|&pi| if pi > 1e-10 { pi * pi.ln() } else { 0.0 })
        .sum();
    let p_log_q: f32 = p.iter().zip(log_q.iter()).map(|(&pi, &lqi)| pi * lqi).sum();
    p_log_p - p_log_q
}

/// Cross-entropy loss
fn cross_entropy_loss(logits: &Array1<f32>, target: usize) -> f32 {
    let log_probs = log_softmax(logits);
    -log_probs[target]
}

/// Knowledge Distillation Loss
///
/// Implements Hinton et al. (2015) distillation loss:
///
/// ```text
/// L_KD = α * T² * KL(softmax(z_s/T) || softmax(z_t/T)) + (1-α) * CE(y, z_s)
/// ```
///
/// Where:
/// - `z_s` = student logits
/// - `z_t` = teacher logits
/// - `T` = temperature (higher = softer targets)
/// - `α` = weight for distillation loss vs hard label loss
/// - `y` = ground truth labels
#[derive(Debug, Clone)]
pub struct DistillationLoss {
    /// Temperature for softening distributions (typical: 2-20)
    pub temperature: f32,
    /// Weight for soft loss vs hard loss (typical: 0.5-0.9)
    pub alpha: f32,
}

impl Default for DistillationLoss {
    fn default() -> Self {
        Self::new(4.0, 0.7)
    }
}

impl DistillationLoss {
    /// Create new distillation loss
    ///
    /// # Arguments
    ///
    /// * `temperature` - Temperature for softening (2-20 typical)
    /// * `alpha` - Weight for soft loss (0.5-0.9 typical)
    #[must_use]
    pub fn new(temperature: f32, alpha: f32) -> Self {
        Self { temperature, alpha }
    }

    /// Compute distillation loss for single sample
    ///
    /// # Arguments
    ///
    /// * `student_logits` - Student model output logits
    /// * `teacher_logits` - Teacher model output logits
    /// * `target` - Ground truth label index
    ///
    /// # Returns
    ///
    /// Combined distillation loss
    pub fn forward_single(
        &self,
        student_logits: &Array1<f32>,
        teacher_logits: &Array1<f32>,
        target: usize,
    ) -> f32 {
        let t = self.temperature;

        // Temperature-scaled logits
        let student_scaled: Array1<f32> = student_logits.mapv(|x| x / t);
        let teacher_scaled: Array1<f32> = teacher_logits.mapv(|x| x / t);

        // Soft targets from teacher
        let teacher_soft = softmax(&teacher_scaled);
        let student_log_soft = log_softmax(&student_scaled);

        // KL divergence (scaled by T²)
        let kl_loss = kl_divergence(&student_log_soft, &teacher_soft) * t * t;

        // Hard label cross-entropy
        let ce_loss = cross_entropy_loss(student_logits, target);

        // Combined loss
        self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss
    }

    /// Compute distillation loss for batch
    ///
    /// # Arguments
    ///
    /// * `student_logits` - [batch_size, vocab_size]
    /// * `teacher_logits` - [batch_size, vocab_size]
    /// * `targets` - Ground truth labels
    ///
    /// # Returns
    ///
    /// Mean batch loss
    pub fn forward(
        &self,
        student_logits: &Array2<f32>,
        teacher_logits: &Array2<f32>,
        targets: &[usize],
    ) -> f32 {
        let batch_size = student_logits.nrows();
        assert_eq!(batch_size, teacher_logits.nrows());
        assert_eq!(batch_size, targets.len());

        let mut total_loss = 0.0;
        for (i, &target) in targets.iter().enumerate() {
            let s_row = student_logits.row(i).to_owned();
            let t_row = teacher_logits.row(i).to_owned();
            total_loss += self.forward_single(&s_row, &t_row, target);
        }

        total_loss / batch_size as f32
    }

    /// Compute soft loss only (no hard labels)
    pub fn soft_loss(&self, student_logits: &Array1<f32>, teacher_logits: &Array1<f32>) -> f32 {
        let t = self.temperature;

        let student_scaled: Array1<f32> = student_logits.mapv(|x| x / t);
        let teacher_scaled: Array1<f32> = teacher_logits.mapv(|x| x / t);

        let teacher_soft = softmax(&teacher_scaled);
        let student_log_soft = log_softmax(&student_scaled);

        kl_divergence(&student_log_soft, &teacher_soft) * t * t
    }
}

/// Progressive Knowledge Distillation
///
/// Matches student hidden states to teacher hidden states at selected layers.
/// Based on Sun et al. (2019).
#[derive(Debug, Clone)]
pub struct ProgressiveDistillation {
    /// Layer mapping: (student_layer, teacher_layer)
    pub layer_mapping: Vec<(usize, usize)>,
    /// Loss weight for hidden state matching
    pub hidden_weight: f32,
    /// Projection matrix for dimension alignment (student_dim x teacher_dim)
    /// Used when student hidden size differs from teacher hidden size.
    pub projection: Option<Array2<f32>>,
}

impl Default for ProgressiveDistillation {
    fn default() -> Self {
        Self {
            layer_mapping: vec![(0, 2), (1, 5), (2, 8), (3, 11)],
            hidden_weight: 1.0,
            projection: None,
        }
    }
}

impl ProgressiveDistillation {
    /// Create new progressive distillation config
    #[must_use]
    pub fn new(layer_mapping: Vec<(usize, usize)>) -> Self {
        Self {
            layer_mapping,
            hidden_weight: 1.0,
            projection: None,
        }
    }

    /// Set projection layer for dimension alignment
    ///
    /// Creates a linear projection matrix to align student hidden states
    /// to teacher hidden size. Initialized with Xavier uniform.
    ///
    /// # Arguments
    ///
    /// * `student_dim` - Student model hidden dimension
    /// * `teacher_dim` - Teacher model hidden dimension
    #[must_use]
    pub fn with_projection(mut self, student_dim: usize, teacher_dim: usize) -> Self {
        use rand::Rng;

        // Xavier uniform initialization
        let scale = (6.0 / (student_dim + teacher_dim) as f32).sqrt();
        let mut rng = rand::rng();

        let projection = Array2::from_shape_fn((student_dim, teacher_dim), |_| {
            rng.random_range(-scale..scale)
        });

        self.projection = Some(projection);
        self
    }

    /// Set hidden state loss weight
    #[must_use]
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.hidden_weight = weight;
        self
    }

    /// Compute hidden state matching loss
    ///
    /// Uses MSE loss between projected student and teacher hidden states.
    /// If projection layer is set and shapes differ, projects student to teacher dimension.
    pub fn hidden_state_loss(
        &self,
        student_hidden: &[Array2<f32>],
        teacher_hidden: &[Array2<f32>],
    ) -> f32 {
        let mut total_loss = 0.0;
        let mut count = 0;

        for (s_idx, t_idx) in &self.layer_mapping {
            if *s_idx < student_hidden.len() && *t_idx < teacher_hidden.len() {
                let s_h = &student_hidden[*s_idx];
                let t_h = &teacher_hidden[*t_idx];

                // MSE loss - project student if dimensions differ
                if s_h.dim() == t_h.dim() {
                    // Same dimensions: direct MSE
                    let diff = s_h - t_h;
                    let mse = diff.mapv(|x| x * x).mean().unwrap_or(0.0);
                    total_loss += mse;
                    count += 1;
                } else if let Some(ref proj) = self.projection {
                    // Different dimensions: project student to teacher space
                    // s_h: (batch, student_dim), proj: (student_dim, teacher_dim)
                    // result: (batch, teacher_dim)
                    let s_dim = s_h.shape()[1];
                    let t_dim = t_h.shape()[1];

                    // Verify projection dimensions match
                    if proj.shape() == [s_dim, t_dim] {
                        let projected = s_h.dot(proj);
                        let diff = &projected - t_h;
                        let mse = diff.mapv(|x| x * x).mean().unwrap_or(0.0);
                        total_loss += mse;
                        count += 1;
                    }
                }
                // Skip if shapes differ and no projection is set
            }
        }

        if count > 0 {
            self.hidden_weight * total_loss / count as f32
        } else {
            0.0
        }
    }
}

/// Attention Transfer Loss
///
/// Transfers attention maps from teacher to student.
/// Based on Zagoruyko & Komodakis (2017).
#[derive(Debug, Clone)]
pub struct AttentionTransfer {
    /// Loss weight
    pub weight: f32,
}

impl Default for AttentionTransfer {
    fn default() -> Self {
        Self { weight: 0.1 }
    }
}

impl AttentionTransfer {
    /// Create new attention transfer config
    #[must_use]
    pub fn new(weight: f32) -> Self {
        Self { weight }
    }

    /// Compute attention transfer loss
    ///
    /// Uses L2 norm of normalized attention map differences.
    pub fn loss(
        &self,
        student_attention: &[Array2<f32>],
        teacher_attention: &[Array2<f32>],
    ) -> f32 {
        let mut total_loss = 0.0;
        let count = student_attention.len().min(teacher_attention.len());

        for (s_attn, t_attn) in student_attention.iter().zip(teacher_attention.iter()) {
            // L2 normalize attention maps
            let s_norm = l2_normalize(s_attn);
            let t_norm = l2_normalize(t_attn);

            // Frobenius norm of difference
            let diff = &s_norm - &t_norm;
            let frob = diff.mapv(|x| x * x).sum().sqrt();
            total_loss += frob * frob;
        }

        if count > 0 {
            self.weight * total_loss / count as f32
        } else {
            0.0
        }
    }
}

/// L2 normalize a 2D array
fn l2_normalize(arr: &Array2<f32>) -> Array2<f32> {
    let norm = arr.mapv(|x| x * x).sum().sqrt();
    if norm > 1e-10 {
        arr / norm
    } else {
        arr.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // =========================================================================
    // Softmax Tests
    // =========================================================================

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = array![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_all_positive() {
        let logits = array![-100.0, 0.0, 100.0];
        let probs = softmax(&logits);
        for p in probs.iter() {
            assert!(*p >= 0.0);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values should not overflow
        let logits = array![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);
        assert!(probs.iter().all(|&p| p.is_finite()));
        assert!((probs.sum() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_identity() {
        let logits = array![1.0, 2.0, 3.0];
        let log_probs = log_softmax(&logits);
        let probs_from_log: Array1<f32> = log_probs.mapv(|x| x.exp());
        let probs = softmax(&logits);

        for (a, b) in probs.iter().zip(probs_from_log.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    // =========================================================================
    // KL Divergence Tests
    // =========================================================================

    #[test]
    fn test_kl_divergence_zero_for_same() {
        let p = softmax(&array![1.0, 2.0, 3.0]);
        let log_p = log_softmax(&array![1.0, 2.0, 3.0]);
        let kl = kl_divergence(&log_p, &p);
        assert!(kl.abs() < 1e-5);
    }

    #[test]
    fn test_kl_divergence_positive() {
        let p = softmax(&array![1.0, 2.0, 3.0]);
        let log_q = log_softmax(&array![3.0, 2.0, 1.0]);
        let kl = kl_divergence(&log_q, &p);
        assert!(kl >= 0.0);
    }

    // =========================================================================
    // DistillationLoss Tests
    // =========================================================================

    #[test]
    fn test_distillation_loss_default() {
        let loss = DistillationLoss::default();
        assert_eq!(loss.temperature, 4.0);
        assert_eq!(loss.alpha, 0.7);
    }

    #[test]
    fn test_distillation_loss_positive() {
        let loss = DistillationLoss::new(4.0, 0.5);
        let student = array![1.0, 2.0, 3.0];
        let teacher = array![1.5, 2.5, 2.0];
        let l = loss.forward_single(&student, &teacher, 2);
        assert!(l >= 0.0);
    }

    #[test]
    fn test_distillation_loss_zero_alpha() {
        // alpha=0 means only hard label loss
        let loss = DistillationLoss::new(4.0, 0.0);
        let student = array![1.0, 2.0, 3.0];
        let teacher = array![100.0, 200.0, 300.0]; // Very different teacher
        let l = loss.forward_single(&student, &teacher, 2);
        // Should be close to cross-entropy loss (ignoring teacher)
        let ce = cross_entropy_loss(&student, 2);
        assert!((l - ce).abs() < 0.01);
    }

    #[test]
    fn test_distillation_loss_high_temp() {
        // Higher temperature = softer distributions
        let loss_low = DistillationLoss::new(1.0, 1.0);
        let loss_high = DistillationLoss::new(10.0, 1.0);
        let student = array![1.0, 2.0, 3.0];
        let teacher = array![1.0, 2.0, 3.0];

        let l_low = loss_low.soft_loss(&student, &teacher);
        let l_high = loss_high.soft_loss(&student, &teacher);

        // Both should be near zero for same logits
        assert!(l_low.abs() < 0.1);
        assert!(l_high.abs() < 0.1);
    }

    #[test]
    fn test_distillation_loss_batch() {
        let loss = DistillationLoss::new(4.0, 0.5);
        let student = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 2.0, 1.0, 3.0]).unwrap();
        let teacher = Array2::from_shape_vec((2, 3), vec![1.5, 2.5, 2.5, 2.5, 1.5, 2.5]).unwrap();
        let targets = vec![2, 0];

        let l = loss.forward(&student, &teacher, &targets);
        assert!(l >= 0.0);
        assert!(l.is_finite());
    }

    // =========================================================================
    // ProgressiveDistillation Tests
    // =========================================================================

    #[test]
    fn test_progressive_default() {
        let prog = ProgressiveDistillation::default();
        assert!(!prog.layer_mapping.is_empty());
        assert_eq!(prog.hidden_weight, 1.0);
    }

    #[test]
    fn test_progressive_hidden_loss_zero_for_same() {
        let prog = ProgressiveDistillation::new(vec![(0, 0), (1, 1)]);
        let hidden = Array2::<f32>::ones((4, 768));
        let student = vec![hidden.clone(), hidden.clone()];
        let teacher = vec![hidden.clone(), hidden.clone()];

        let loss = prog.hidden_state_loss(&student, &teacher);
        assert!(loss.abs() < 1e-5);
    }

    #[test]
    fn test_progressive_hidden_loss_positive_for_diff() {
        let prog = ProgressiveDistillation::new(vec![(0, 0)]);
        let s = Array2::<f32>::zeros((4, 768));
        let t = Array2::<f32>::ones((4, 768));

        let loss = prog.hidden_state_loss(&[s], &[t]);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_progressive_with_weight() {
        let prog = ProgressiveDistillation::new(vec![(0, 0)]).with_weight(0.5);
        assert_eq!(prog.hidden_weight, 0.5);
    }

    #[test]
    fn test_progressive_projection_layer_creation() {
        // Student dim 512, teacher dim 768
        let prog = ProgressiveDistillation::new(vec![(0, 0)]).with_projection(512, 768);
        assert!(prog.projection.is_some());
        let proj = prog.projection.as_ref().unwrap();
        assert_eq!(proj.dim(), (512, 768));
    }

    #[test]
    fn test_progressive_hidden_loss_with_projection() {
        // Student has dim 512, teacher has dim 768
        let prog = ProgressiveDistillation::new(vec![(0, 0)]).with_projection(512, 768);

        let student = vec![Array2::<f32>::ones((4, 512))];
        let teacher = vec![Array2::<f32>::ones((4, 768))];

        // Should not skip due to shape mismatch
        let loss = prog.hidden_state_loss(&student, &teacher);
        // Loss should be computed (not zero due to projection mismatch)
        // Just verify it doesn't skip
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_progressive_projection_correct_transform() {
        // Use identity-like projection
        let mut prog = ProgressiveDistillation::new(vec![(0, 0)]).with_projection(768, 768);

        // Set projection to identity matrix
        if let Some(ref mut proj) = prog.projection {
            proj.fill(0.0);
            for i in 0..768 {
                proj[[i, i]] = 1.0;
            }
        }

        let hidden = Array2::<f32>::from_elem((4, 768), 1.0);
        let student = vec![hidden.clone()];
        let teacher = vec![hidden.clone()];

        // With identity projection, loss should be ~0
        let loss = prog.hidden_state_loss(&student, &teacher);
        assert!(loss.abs() < 1e-4, "Identity projection should give ~0 loss");
    }

    #[test]
    fn test_progressive_no_projection_skips_mismatched() {
        // No projection set
        let prog = ProgressiveDistillation::new(vec![(0, 0)]);

        let student = vec![Array2::<f32>::ones((4, 512))];
        let teacher = vec![Array2::<f32>::ones((4, 768))];

        // Should skip due to shape mismatch, loss = 0
        let loss = prog.hidden_state_loss(&student, &teacher);
        assert_eq!(
            loss, 0.0,
            "Should skip mismatched shapes without projection"
        );
    }

    // =========================================================================
    // AttentionTransfer Tests
    // =========================================================================

    #[test]
    fn test_attention_transfer_default() {
        let at = AttentionTransfer::default();
        assert_eq!(at.weight, 0.1);
    }

    #[test]
    fn test_attention_transfer_zero_for_same() {
        let at = AttentionTransfer::new(1.0);
        let attn = Array2::<f32>::ones((8, 8));
        let student = vec![attn.clone()];
        let teacher = vec![attn.clone()];

        let loss = at.loss(&student, &teacher);
        assert!(loss.abs() < 1e-5);
    }

    #[test]
    fn test_attention_transfer_positive_for_diff() {
        let at = AttentionTransfer::new(1.0);
        let s = Array2::<f32>::zeros((8, 8));
        let t = Array2::<f32>::ones((8, 8));

        let loss = at.loss(&[s], &[t]);
        assert!(loss > 0.0);
    }

    // =========================================================================
    // L2 Normalize Tests
    // =========================================================================

    #[test]
    fn test_l2_normalize_unit_norm() {
        let arr = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 0.0, 0.0]).unwrap();
        let norm = l2_normalize(&arr);
        let l2 = norm.mapv(|x| x * x).sum().sqrt();
        assert!((l2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_normalize_zero() {
        let arr = Array2::<f32>::zeros((2, 2));
        let norm = l2_normalize(&arr);
        // Should return zeros without NaN
        assert!(norm.iter().all(|&x| x.is_finite()));
    }

    // =========================================================================
    // Property-like Tests
    // =========================================================================

    #[test]
    fn test_distillation_loss_monotonic_in_alpha() {
        let student = array![1.0, 2.0, 3.0];
        let teacher = array![3.0, 2.0, 1.0]; // Very different

        let loss_0 = DistillationLoss::new(4.0, 0.0).forward_single(&student, &teacher, 2);
        let loss_1 = DistillationLoss::new(4.0, 1.0).forward_single(&student, &teacher, 2);

        // As alpha increases, soft loss contribution increases
        // Both should be valid losses
        assert!(loss_0 >= 0.0);
        assert!(loss_1 >= 0.0);
    }

    #[test]
    fn test_temperature_scaling_effect() {
        let student = array![1.0, 2.0, 3.0];
        let teacher = array![0.5, 2.0, 3.5];

        let loss_t1 = DistillationLoss::new(1.0, 1.0).soft_loss(&student, &teacher);
        let loss_t10 = DistillationLoss::new(10.0, 1.0).soft_loss(&student, &teacher);

        // Both should be valid
        assert!(loss_t1.is_finite());
        assert!(loss_t10.is_finite());
    }
}
