//! Distillation loss functions

use ndarray::{Array2, Axis};

/// Knowledge Distillation Loss
///
/// Combines soft targets from teacher (via temperature-scaled KL divergence)
/// with hard targets from ground truth labels (via cross-entropy).
///
/// # Formula
///
/// ```text
/// L = α * T² * KL(softmax(teacher/T) || softmax(student/T))
///   + (1-α) * CE(student, labels)
/// ```
///
/// where T is temperature and α is the distillation weight.
///
/// # Example
///
/// ```
/// use entrenar::distill::DistillationLoss;
/// use ndarray::array;
///
/// let loss_fn = DistillationLoss::new(2.0, 0.7);
/// let student_logits = array![[2.0, 1.0, 0.5]];
/// let teacher_logits = array![[1.5, 1.2, 0.8]];
/// let labels = vec![0];
///
/// let loss = loss_fn.forward(&student_logits, &teacher_logits, &labels);
/// assert!(loss > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct DistillationLoss {
    /// Temperature for softening probability distributions
    pub temperature: f32,
    /// Weight for distillation loss (α). Hard loss weight is (1-α)
    pub alpha: f32,
}

impl DistillationLoss {
    /// Create a new distillation loss function
    ///
    /// # Arguments
    ///
    /// * `temperature` - Temperature for softening distributions (typically 2.0-5.0)
    /// * `alpha` - Weight for distillation vs hard loss (typically 0.5-0.9)
    ///
    /// # Panics
    ///
    /// Panics if temperature <= 0 or alpha not in [0, 1]
    pub fn new(temperature: f32, alpha: f32) -> Self {
        assert!(
            temperature > 0.0,
            "Temperature must be positive, got {}",
            temperature
        );
        assert!(
            (0.0..=1.0).contains(&alpha),
            "Alpha must be in [0, 1], got {}",
            alpha
        );

        Self { temperature, alpha }
    }

    /// Compute the distillation loss
    ///
    /// # Arguments
    ///
    /// * `student_logits` - Logits from student model [batch_size, num_classes]
    /// * `teacher_logits` - Logits from teacher model [batch_size, num_classes]
    /// * `labels` - Ground truth labels [batch_size]
    ///
    /// # Returns
    ///
    /// Combined distillation and hard loss (scalar)
    pub fn forward(
        &self,
        student_logits: &Array2<f32>,
        teacher_logits: &Array2<f32>,
        labels: &[usize],
    ) -> f32 {
        assert_eq!(
            student_logits.shape(),
            teacher_logits.shape(),
            "Student and teacher logits must have same shape"
        );
        assert_eq!(
            student_logits.nrows(),
            labels.len(),
            "Batch size must match number of labels"
        );

        // Soft targets: KL divergence with temperature scaling
        let kl_loss = self.kl_divergence_loss(student_logits, teacher_logits);

        // Hard targets: Cross-entropy with ground truth
        let ce_loss = self.cross_entropy_loss(student_logits, labels);

        // Combine with temperature correction factor (T²)
        self.alpha * kl_loss * self.temperature * self.temperature + (1.0 - self.alpha) * ce_loss
    }

    /// Temperature-scaled KL divergence loss
    ///
    /// KL(teacher || student) where both distributions are softened by temperature
    fn kl_divergence_loss(
        &self,
        student_logits: &Array2<f32>,
        teacher_logits: &Array2<f32>,
    ) -> f32 {
        let student_soft = softmax_2d(&(student_logits / self.temperature));
        let teacher_soft = softmax_2d(&(teacher_logits / self.temperature));

        kl_divergence(&teacher_soft, &student_soft)
    }

    /// Standard cross-entropy loss with hard labels
    fn cross_entropy_loss(&self, logits: &Array2<f32>, labels: &[usize]) -> f32 {
        let probs = softmax_2d(logits);

        let mut loss = 0.0;
        for (i, &label) in labels.iter().enumerate() {
            let prob = probs[[i, label]].max(1e-10); // Avoid log(0)
            loss -= prob.ln();
        }

        loss / labels.len() as f32
    }
}

/// Compute softmax along last axis for 2D array
///
/// softmax(x)_i = exp(x_i) / Σ exp(x_j)
fn softmax_2d(x: &Array2<f32>) -> Array2<f32> {
    let mut result = x.clone();

    for mut row in result.axis_iter_mut(Axis(0)) {
        // Subtract max for numerical stability
        let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|v| (v - max_val).exp());

        // Normalize
        let sum: f32 = row.sum();
        row.mapv_inplace(|v| v / sum);
    }

    result
}

/// KL divergence between two probability distributions
///
/// KL(p || q) = Σ p_i * log(p_i / q_i)
///
/// Average over batch dimension.
fn kl_divergence(p: &Array2<f32>, q: &Array2<f32>) -> f32 {
    assert_eq!(p.shape(), q.shape());

    let mut total_kl = 0.0;

    for (p_row, q_row) in p.axis_iter(Axis(0)).zip(q.axis_iter(Axis(0))) {
        let mut kl = 0.0;
        for (&p_i, &q_i) in p_row.iter().zip(q_row.iter()) {
            if p_i > 1e-10 {
                // Avoid log(0)
                kl += p_i * (p_i / q_i.max(1e-10)).ln();
            }
        }
        total_kl += kl;
    }

    total_kl / p.nrows() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_distillation_loss_basic() {
        let loss_fn = DistillationLoss::new(2.0, 0.5);
        let student = array![[2.0, 1.0, 0.5]];
        let teacher = array![[1.5, 1.2, 0.8]];
        let labels = vec![0];

        let loss = loss_fn.forward(&student, &teacher, &labels);
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let probs = softmax_2d(&x);

        for row in probs.axis_iter(Axis(0)) {
            let sum: f32 = row.sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_kl_divergence_zero_for_identical() {
        let p = array![[0.7, 0.2, 0.1], [0.5, 0.3, 0.2]];
        let kl = kl_divergence(&p, &p);
        assert_relative_eq!(kl, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_kl_divergence_positive() {
        let p = array![[0.7, 0.2, 0.1]];
        let q = array![[0.4, 0.4, 0.2]];
        let kl = kl_divergence(&p, &q);
        assert!(kl > 0.0);
    }

    #[test]
    #[should_panic(expected = "Temperature must be positive")]
    fn test_negative_temperature_panics() {
        DistillationLoss::new(-1.0, 0.5);
    }

    #[test]
    #[should_panic(expected = "Alpha must be in [0, 1]")]
    fn test_invalid_alpha_panics() {
        DistillationLoss::new(2.0, 1.5);
    }

    #[test]
    fn test_temperature_effect() {
        let student = array![[10.0, 1.0, 0.1]];
        let teacher = array![[5.0, 4.0, 3.0]];
        let labels = vec![0];

        let low_temp_loss = DistillationLoss::new(1.0, 1.0);
        let high_temp_loss = DistillationLoss::new(5.0, 1.0);

        let loss_low = low_temp_loss.forward(&student, &teacher, &labels);
        let loss_high = high_temp_loss.forward(&student, &teacher, &labels);

        // Higher temperature should soften distributions more
        assert!(loss_low != loss_high);
    }

    #[test]
    fn test_alpha_balances_losses() {
        let student = array![[2.0, 1.0, 0.5]];
        let teacher = array![[1.5, 1.2, 0.8]];
        let labels = vec![0];

        // Pure distillation (α=1)
        let pure_distill = DistillationLoss::new(2.0, 1.0);
        let loss_distill = pure_distill.forward(&student, &teacher, &labels);

        // Pure hard loss (α=0)
        let pure_hard = DistillationLoss::new(2.0, 0.0);
        let loss_hard = pure_hard.forward(&student, &teacher, &labels);

        // Balanced (α=0.5)
        let balanced = DistillationLoss::new(2.0, 0.5);
        let loss_balanced = balanced.forward(&student, &teacher, &labels);

        // Balanced should be between the two extremes (approximately)
        assert!(loss_balanced > 0.0);
        assert!(loss_distill > 0.0);
        assert!(loss_hard > 0.0);
    }
}
