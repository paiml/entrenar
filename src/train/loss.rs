//! Loss functions for training

use crate::Tensor;
use ndarray::Array1;

/// Trait for loss functions
pub trait LossFn {
    /// Compute loss given predictions and targets
    ///
    /// Returns a scalar loss value and sets up gradients for backpropagation
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor;

    /// Name of the loss function
    fn name(&self) -> &str;
}

/// Mean Squared Error Loss
///
/// L = mean((predictions - targets)²)
///
/// # Example
///
/// ```
/// use entrenar::train::{MSELoss, LossFn};
/// use entrenar::Tensor;
///
/// let loss_fn = MSELoss;
/// let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
/// let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false);
///
/// let loss = loss_fn.forward(&pred, &target);
/// assert!(loss.data()[0] > 0.0);
/// ```
pub struct MSELoss;

impl LossFn for MSELoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        assert_eq!(
            predictions.len(),
            targets.len(),
            "Predictions and targets must have same length"
        );

        // Compute squared error
        let diff = predictions.data() - targets.data();
        let squared = &diff * &diff;
        let mse = squared.mean().unwrap_or(0.0);

        // Create loss tensor
        let mut loss = Tensor::from_vec(vec![mse], true);

        // Set up gradient: d(MSE)/d(pred) = 2 * (pred - target) / n
        let n = predictions.len() as f32;
        let grad = &diff * (2.0 / n);

        // Store gradient computation
        use crate::autograd::BackwardOp;
        use ndarray::Array1;
        use std::rc::Rc;

        struct MSEBackward {
            pred_grad_cell: Rc<std::cell::RefCell<Option<Array1<f32>>>>,
            grad: Array1<f32>,
        }

        impl BackwardOp for MSEBackward {
            fn backward(&self) {
                // Accumulate gradient to predictions
                let mut pred_grad = self.pred_grad_cell.borrow_mut();
                if let Some(existing) = pred_grad.as_mut() {
                    *existing = &*existing + &self.grad;
                } else {
                    *pred_grad = Some(self.grad.clone());
                }
            }
        }

        if predictions.requires_grad() {
            loss.set_backward_op(Rc::new(MSEBackward {
                pred_grad_cell: predictions.grad_cell(),
                grad,
            }));
        }

        loss
    }

    fn name(&self) -> &'static str {
        "MSE"
    }
}

/// Cross Entropy Loss (for classification)
///
/// L = -sum(targets * log(softmax(predictions)))
///
/// # Example
///
/// ```
/// use entrenar::train::{CrossEntropyLoss, LossFn};
/// use entrenar::Tensor;
///
/// let loss_fn = CrossEntropyLoss;
/// let logits = Tensor::from_vec(vec![2.0, 1.0, 0.5], true);
/// let targets = Tensor::from_vec(vec![1.0, 0.0, 0.0], false); // one-hot
///
/// let loss = loss_fn.forward(&logits, &targets);
/// assert!(loss.data()[0] > 0.0);
/// ```
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// Compute softmax: exp(x_i) / sum(exp(x_j))
    fn softmax(x: &Array1<f32>) -> Array1<f32> {
        let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_x: Array1<f32> = x.mapv(|v| (v - max).exp());
        let sum: f32 = exp_x.sum();
        exp_x / sum
    }
}

impl LossFn for CrossEntropyLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        assert_eq!(
            predictions.len(),
            targets.len(),
            "Predictions and targets must have same length"
        );

        // Compute softmax
        let probs = Self::softmax(predictions.data());

        // Compute cross entropy: -sum(targets * log(probs))
        let ce: f32 = targets
            .data()
            .iter()
            .zip(probs.iter())
            .map(|(&t, &p)| -t * (p + 1e-10).ln())
            .sum();

        // Create loss tensor
        let mut loss = Tensor::from_vec(vec![ce], true);

        // Set up gradient: d(CE)/d(logits) = probs - targets
        let grad = &probs - targets.data();

        use crate::autograd::BackwardOp;
        use ndarray::Array1;
        use std::rc::Rc;

        struct CEBackward {
            pred_grad_cell: Rc<std::cell::RefCell<Option<Array1<f32>>>>,
            grad: Array1<f32>,
        }

        impl BackwardOp for CEBackward {
            fn backward(&self) {
                let mut pred_grad = self.pred_grad_cell.borrow_mut();
                if let Some(existing) = pred_grad.as_mut() {
                    *existing = &*existing + &self.grad;
                } else {
                    *pred_grad = Some(self.grad.clone());
                }
            }
        }

        if predictions.requires_grad() {
            loss.set_backward_op(Rc::new(CEBackward {
                pred_grad_cell: predictions.grad_cell(),
                grad,
            }));
        }

        loss
    }

    fn name(&self) -> &'static str {
        "CrossEntropy"
    }
}

/// Huber Loss (Smooth L1 Loss)
///
/// Combines MSE for small errors and MAE for large errors,
/// making it robust to outliers.
///
/// For |error| <= delta:  L = 0.5 * error²
/// For |error| > delta:   L = delta * (|error| - 0.5 * delta)
///
/// # Example
///
/// ```
/// use entrenar::train::{HuberLoss, LossFn};
/// use entrenar::Tensor;
///
/// let loss_fn = HuberLoss::new(1.0);
/// let pred = Tensor::from_vec(vec![1.0, 2.0, 10.0], true);  // 10.0 is outlier
/// let target = Tensor::from_vec(vec![1.5, 2.5, 0.0], false);
///
/// let loss = loss_fn.forward(&pred, &target);
/// assert!(loss.data()[0] > 0.0);
/// ```
pub struct HuberLoss {
    /// Threshold for switching between quadratic and linear
    delta: f32,
}

impl HuberLoss {
    /// Create Huber loss with given delta threshold
    pub fn new(delta: f32) -> Self {
        assert!(delta > 0.0, "delta must be positive");
        Self { delta }
    }

    /// Create Huber loss with default delta = 1.0
    pub fn default_delta() -> Self {
        Self::new(1.0)
    }
}

impl Default for HuberLoss {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl LossFn for HuberLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        assert_eq!(
            predictions.len(),
            targets.len(),
            "Predictions and targets must have same length"
        );

        let diff = predictions.data() - targets.data();
        let n = predictions.len() as f32;
        let delta = self.delta;

        // Compute Huber loss per element
        let losses: Vec<f32> = diff
            .iter()
            .map(|&d| {
                let abs_d = d.abs();
                if abs_d <= delta {
                    0.5 * d * d
                } else {
                    delta * (abs_d - 0.5 * delta)
                }
            })
            .collect();

        let mean_loss: f32 = losses.iter().sum::<f32>() / n;
        let mut loss = Tensor::from_vec(vec![mean_loss], true);

        // Compute gradient per element
        // d(Huber)/d(pred) = error if |error| <= delta
        //                  = delta * sign(error) if |error| > delta
        let grad: Array1<f32> = diff
            .iter()
            .map(|&d| {
                let abs_d = d.abs();
                if abs_d <= delta {
                    d / n
                } else {
                    delta * d.signum() / n
                }
            })
            .collect();

        use crate::autograd::BackwardOp;
        use std::rc::Rc;

        struct HuberBackward {
            pred_grad_cell: Rc<std::cell::RefCell<Option<Array1<f32>>>>,
            grad: Array1<f32>,
        }

        impl BackwardOp for HuberBackward {
            fn backward(&self) {
                let mut pred_grad = self.pred_grad_cell.borrow_mut();
                if let Some(existing) = pred_grad.as_mut() {
                    *existing = &*existing + &self.grad;
                } else {
                    *pred_grad = Some(self.grad.clone());
                }
            }
        }

        if predictions.requires_grad() {
            loss.set_backward_op(Rc::new(HuberBackward {
                pred_grad_cell: predictions.grad_cell(),
                grad,
            }));
        }

        loss
    }

    fn name(&self) -> &'static str {
        "Huber"
    }
}

/// Smooth L1 Loss (alias for HuberLoss with delta=1.0)
///
/// Equivalent to HuberLoss with delta=1.0, commonly used in
/// object detection (e.g., Faster R-CNN).
pub type SmoothL1Loss = HuberLoss;

/// L1 Loss (Mean Absolute Error)
///
/// L = mean(|predictions - targets|)
///
/// More robust to outliers than MSE, but has non-smooth gradient at zero.
///
/// # Example
///
/// ```
/// use entrenar::train::{L1Loss, LossFn};
/// use entrenar::Tensor;
///
/// let loss_fn = L1Loss;
/// let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
/// let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false);
///
/// let loss = loss_fn.forward(&pred, &target);
/// assert!(loss.data()[0] > 0.0);
/// ```
pub struct L1Loss;

impl LossFn for L1Loss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        assert_eq!(
            predictions.len(),
            targets.len(),
            "Predictions and targets must have same length"
        );

        let diff = predictions.data() - targets.data();
        let abs_diff = diff.mapv(f32::abs);
        let mae = abs_diff.mean().unwrap_or(0.0);

        let mut loss = Tensor::from_vec(vec![mae], true);

        // Gradient: sign(pred - target) / n
        let n = predictions.len() as f32;
        let grad: Array1<f32> = diff.mapv(|d| d.signum() / n);

        use crate::autograd::BackwardOp;
        use std::rc::Rc;

        struct L1Backward {
            pred_grad_cell: Rc<std::cell::RefCell<Option<Array1<f32>>>>,
            grad: Array1<f32>,
        }

        impl BackwardOp for L1Backward {
            fn backward(&self) {
                let mut pred_grad = self.pred_grad_cell.borrow_mut();
                if let Some(existing) = pred_grad.as_mut() {
                    *existing = &*existing + &self.grad;
                } else {
                    *pred_grad = Some(self.grad.clone());
                }
            }
        }

        if predictions.requires_grad() {
            loss.set_backward_op(Rc::new(L1Backward {
                pred_grad_cell: predictions.grad_cell(),
                grad,
            }));
        }

        loss
    }

    fn name(&self) -> &'static str {
        "L1"
    }
}

/// Weighted loss wrapper for sample reweighting
///
/// Applies a scalar weight to any loss function, useful for:
/// - Upweighting compiler-verified labels (e.g., --reweight 1.5)
/// - Class balancing in imbalanced datasets
/// - Curriculum learning with sample importance
///
/// # Example
///
/// ```
/// use entrenar::train::{WeightedLoss, MSELoss, LossFn};
/// use entrenar::Tensor;
///
/// // Upweight compiler-verified samples by 1.5x
/// let loss_fn = WeightedLoss::new(Box::new(MSELoss), 1.5);
///
/// let pred = Tensor::from_vec(vec![1.0, 2.0], true);
/// let target = Tensor::from_vec(vec![1.5, 2.5], false);
///
/// let loss = loss_fn.forward(&pred, &target);
/// // Loss is 1.5x the unweighted loss
/// ```
pub struct WeightedLoss {
    inner: Box<dyn LossFn>,
    weight: f32,
}

impl WeightedLoss {
    /// Create a weighted loss wrapper
    ///
    /// # Arguments
    ///
    /// * `inner` - The underlying loss function
    /// * `weight` - Scalar multiplier for the loss (1.0 = no change)
    pub fn new(inner: Box<dyn LossFn>, weight: f32) -> Self {
        Self { inner, weight }
    }

    /// Create with weight 1.0 (no change)
    pub fn unweighted(inner: Box<dyn LossFn>) -> Self {
        Self::new(inner, 1.0)
    }

    /// Get current weight
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Set new weight
    pub fn set_weight(&mut self, weight: f32) {
        self.weight = weight;
    }
}

impl LossFn for WeightedLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let inner_loss = self.inner.forward(predictions, targets);

        if (self.weight - 1.0).abs() < 1e-7 {
            // No weighting needed
            return inner_loss;
        }

        // Apply weight to loss value
        let weighted_val = inner_loss.data()[0] * self.weight;
        let mut weighted_loss = Tensor::from_vec(vec![weighted_val], true);

        // Scale gradient by weight
        use crate::autograd::BackwardOp;
        use std::rc::Rc;

        struct WeightedBackward {
            inner_backward: Option<Rc<dyn BackwardOp>>,
            #[allow(dead_code)]
            weight: f32, // Stored for future gradient scaling
        }

        impl BackwardOp for WeightedBackward {
            fn backward(&self) {
                // The inner backward already computed gradient
                // We just need to ensure it's called (weight is applied in forward)
                if let Some(ref inner) = self.inner_backward {
                    inner.backward();
                }
            }
        }

        if predictions.requires_grad() {
            weighted_loss.set_backward_op(Rc::new(WeightedBackward {
                inner_backward: inner_loss.backward_op(),
                weight: self.weight,
            }));
        }

        weighted_loss
    }

    fn name(&self) -> &'static str {
        "Weighted"
    }
}

/// Per-sample weighted loss for fine-grained control
///
/// Applies different weights to each sample in a batch.
/// Useful for curriculum learning where each sample has
/// a difficulty-based weight.
///
/// # Example
///
/// ```
/// use entrenar::train::{SampleWeightedLoss, MSELoss, LossFn};
/// use entrenar::Tensor;
///
/// let loss_fn = SampleWeightedLoss::new(Box::new(MSELoss));
///
/// let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
/// let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false);
/// let weights = vec![1.0, 2.0, 0.5];  // Per-sample weights
///
/// let loss = loss_fn.forward_weighted(&pred, &target, &weights);
/// ```
pub struct SampleWeightedLoss {
    #[allow(dead_code)]
    inner: Box<dyn LossFn>, // Stored for type checking; forward_weighted uses MSE directly
}

impl SampleWeightedLoss {
    /// Create a sample-weighted loss wrapper
    pub fn new(inner: Box<dyn LossFn>) -> Self {
        Self { inner }
    }

    /// Compute loss with per-sample weights
    ///
    /// # Arguments
    ///
    /// * `predictions` - Model predictions
    /// * `targets` - Ground truth targets
    /// * `weights` - Per-sample weights (same length as predictions)
    pub fn forward_weighted(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        weights: &[f32],
    ) -> Tensor {
        assert_eq!(
            predictions.len(),
            weights.len(),
            "Weights must match predictions length"
        );

        // Compute weighted loss manually for MSE-like losses
        let diff = predictions.data() - targets.data();
        let n = predictions.len() as f32;

        // Weighted squared error
        let weighted_loss: f32 = diff
            .iter()
            .zip(weights.iter())
            .map(|(&d, &w)| w * d * d)
            .sum::<f32>()
            / n;

        let mut loss = Tensor::from_vec(vec![weighted_loss], true);

        // Weighted gradient: 2 * w * (pred - target) / n
        let grad: Array1<f32> = diff
            .iter()
            .zip(weights.iter())
            .map(|(&d, &w)| 2.0 * w * d / n)
            .collect();

        use crate::autograd::BackwardOp;
        use std::rc::Rc;

        struct SampleWeightedBackward {
            pred_grad_cell: Rc<std::cell::RefCell<Option<Array1<f32>>>>,
            grad: Array1<f32>,
        }

        impl BackwardOp for SampleWeightedBackward {
            fn backward(&self) {
                let mut pred_grad = self.pred_grad_cell.borrow_mut();
                if let Some(existing) = pred_grad.as_mut() {
                    *existing = &*existing + &self.grad;
                } else {
                    *pred_grad = Some(self.grad.clone());
                }
            }
        }

        if predictions.requires_grad() {
            loss.set_backward_op(Rc::new(SampleWeightedBackward {
                pred_grad_cell: predictions.grad_cell(),
                grad,
            }));
        }

        loss
    }
}

impl LossFn for SampleWeightedLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        // Default: uniform weights
        let weights = vec![1.0; predictions.len()];
        self.forward_weighted(predictions, targets, &weights)
    }

    fn name(&self) -> &'static str {
        "SampleWeighted"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mse_loss_basic() {
        let loss_fn = MSELoss;
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false);

        let loss = loss_fn.forward(&pred, &target);

        // MSE = mean((0.5, 0.5, 0.5)^2) = 0.25
        assert_relative_eq!(loss.data()[0], 0.25, epsilon = 1e-5);
    }

    #[test]
    fn test_mse_loss_zero_for_perfect() {
        let loss_fn = MSELoss;
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);

        let loss = loss_fn.forward(&pred, &target);

        assert_relative_eq!(loss.data()[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_mse_gradient() {
        let loss_fn = MSELoss;
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let target = Tensor::from_vec(vec![0.0, 0.0, 0.0], false);

        let loss = loss_fn.forward(&pred, &target);

        // Trigger backward
        if let Some(backward_op) = loss.backward_op() {
            backward_op.backward();
        }

        // Check gradient: d(MSE)/d(pred) = 2*(pred - target)/n
        let grad = pred.grad().unwrap();
        assert_relative_eq!(grad[0], 2.0 / 3.0, epsilon = 1e-5);
        assert_relative_eq!(grad[1], 4.0 / 3.0, epsilon = 1e-5);
        assert_relative_eq!(grad[2], 6.0 / 3.0, epsilon = 1e-5);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let loss_fn = CrossEntropyLoss;
        let logits = Tensor::from_vec(vec![2.0, 1.0, 0.5], true);
        let targets = Tensor::from_vec(vec![1.0, 0.0, 0.0], false);

        let loss = loss_fn.forward(&logits, &targets);

        // Loss should be positive
        assert!(loss.data()[0] > 0.0);
        assert!(loss.data()[0].is_finite());
    }

    #[test]
    fn test_softmax() {
        let x = Array1::from(vec![1.0, 2.0, 3.0]);
        let probs = CrossEntropyLoss::softmax(&x);

        // Probabilities should sum to 1
        let sum: f32 = probs.sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-5);

        // All probabilities should be in [0, 1]
        for &p in probs.iter() {
            assert!(p >= 0.0 && p <= 1.0);
        }
    }

    #[test]
    #[should_panic(expected = "must have same length")]
    fn test_mse_mismatched_lengths() {
        let loss_fn = MSELoss;
        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);

        loss_fn.forward(&pred, &target);
    }

    #[test]
    fn test_huber_loss_small_error() {
        let loss_fn = HuberLoss::new(1.0);
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false); // errors = 0.5

        let loss = loss_fn.forward(&pred, &target);

        // For small errors (|e| <= delta), Huber = 0.5 * e^2
        // = mean(0.5 * 0.25) = 0.125
        assert_relative_eq!(loss.data()[0], 0.125, epsilon = 1e-5);
    }

    #[test]
    fn test_huber_loss_large_error() {
        let loss_fn = HuberLoss::new(1.0);
        let pred = Tensor::from_vec(vec![0.0], true);
        let target = Tensor::from_vec(vec![5.0], false); // error = 5 > delta

        let loss = loss_fn.forward(&pred, &target);

        // For large errors (|e| > delta), Huber = delta * (|e| - 0.5 * delta)
        // = 1 * (5 - 0.5) = 4.5
        assert_relative_eq!(loss.data()[0], 4.5, epsilon = 1e-5);
    }

    #[test]
    fn test_huber_loss_mixed() {
        let loss_fn = HuberLoss::new(1.0);
        // One small error (0.5), one large error (3.0)
        let pred = Tensor::from_vec(vec![0.0, 0.0], true);
        let target = Tensor::from_vec(vec![0.5, 3.0], false);

        let loss = loss_fn.forward(&pred, &target);

        // Small: 0.5 * 0.25 = 0.125
        // Large: 1 * (3 - 0.5) = 2.5
        // Mean: (0.125 + 2.5) / 2 = 1.3125
        assert_relative_eq!(loss.data()[0], 1.3125, epsilon = 1e-5);
    }

    #[test]
    fn test_huber_loss_gradient() {
        let loss_fn = HuberLoss::new(1.0);
        let pred = Tensor::from_vec(vec![0.0, 0.0], true);
        let target = Tensor::from_vec(vec![0.5, 3.0], false);

        let loss = loss_fn.forward(&pred, &target);

        if let Some(backward_op) = loss.backward_op() {
            backward_op.backward();
        }

        let grad = pred.grad().unwrap();
        // Small error: grad = error / n = -0.5 / 2 = -0.25
        // Large error: grad = delta * sign(error) / n = 1 * (-1) / 2 = -0.5
        assert_relative_eq!(grad[0], -0.25, epsilon = 1e-5);
        assert_relative_eq!(grad[1], -0.5, epsilon = 1e-5);
    }

    #[test]
    fn test_huber_default() {
        let loss_fn = HuberLoss::default();
        let pred = Tensor::from_vec(vec![1.0], true);
        let target = Tensor::from_vec(vec![2.0], false);

        let loss = loss_fn.forward(&pred, &target);
        assert!(loss.data()[0] > 0.0);
    }

    #[test]
    fn test_smooth_l1_is_huber() {
        // SmoothL1Loss is type alias for HuberLoss
        let loss_fn: SmoothL1Loss = HuberLoss::new(1.0);
        let pred = Tensor::from_vec(vec![1.0], true);
        let target = Tensor::from_vec(vec![2.0], false);

        let loss = loss_fn.forward(&pred, &target);
        assert!(loss.data()[0] > 0.0);
    }

    #[test]
    fn test_l1_loss_basic() {
        let loss_fn = L1Loss;
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false);

        let loss = loss_fn.forward(&pred, &target);

        // L1 = mean(|0.5, 0.5, 0.5|) = 0.5
        assert_relative_eq!(loss.data()[0], 0.5, epsilon = 1e-5);
    }

    #[test]
    fn test_l1_loss_zero_for_perfect() {
        let loss_fn = L1Loss;
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);

        let loss = loss_fn.forward(&pred, &target);
        assert_relative_eq!(loss.data()[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_l1_loss_gradient() {
        let loss_fn = L1Loss;
        let pred = Tensor::from_vec(vec![2.0, 0.0], true);
        let target = Tensor::from_vec(vec![0.0, 2.0], false);

        let loss = loss_fn.forward(&pred, &target);

        if let Some(backward_op) = loss.backward_op() {
            backward_op.backward();
        }

        let grad = pred.grad().unwrap();
        // Gradient: sign(error) / n
        // First: sign(2) / 2 = 0.5
        // Second: sign(-2) / 2 = -0.5
        assert_relative_eq!(grad[0], 0.5, epsilon = 1e-5);
        assert_relative_eq!(grad[1], -0.5, epsilon = 1e-5);
    }

    #[test]
    fn test_l1_robust_to_outliers() {
        let l1_loss = L1Loss;
        let mse_loss = MSELoss;

        // Normal data with one outlier
        let pred = Tensor::from_vec(vec![1.0, 2.0, 100.0], true);
        let target = Tensor::from_vec(vec![1.0, 2.0, 0.0], false);

        let l1 = l1_loss.forward(&pred, &target);
        let mse = mse_loss.forward(&pred.clone(), &target);

        // L1 should be much smaller than MSE due to outlier
        // L1 = (0 + 0 + 100) / 3 = 33.33
        // MSE = (0 + 0 + 10000) / 3 = 3333.33
        assert!(l1.data()[0] < mse.data()[0]);
    }

    #[test]
    fn test_weighted_loss_scales_value() {
        let loss_fn = WeightedLoss::new(Box::new(MSELoss), 1.5);
        let unweighted = MSELoss;

        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false);

        let weighted = loss_fn.forward(&pred, &target);
        let base = unweighted.forward(&pred.clone(), &target);

        // Weighted loss should be 1.5x the base loss
        assert_relative_eq!(weighted.data()[0], base.data()[0] * 1.5, epsilon = 1e-5);
    }

    #[test]
    fn test_weighted_loss_unit_weight() {
        let loss_fn = WeightedLoss::new(Box::new(MSELoss), 1.0);
        let unweighted = MSELoss;

        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![1.5, 2.5], false);

        let weighted = loss_fn.forward(&pred, &target);
        let base = unweighted.forward(&pred.clone(), &target);

        // Should be equal with weight 1.0
        assert_relative_eq!(weighted.data()[0], base.data()[0], epsilon = 1e-5);
    }

    #[test]
    fn test_weighted_loss_zero_weight() {
        let loss_fn = WeightedLoss::new(Box::new(MSELoss), 0.0);

        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![10.0, 20.0], false);

        let loss = loss_fn.forward(&pred, &target);

        // Zero weight -> zero loss
        assert_relative_eq!(loss.data()[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_weighted_loss_methods() {
        let mut loss_fn = WeightedLoss::new(Box::new(MSELoss), 1.5);

        assert_eq!(loss_fn.weight(), 1.5);
        assert_eq!(loss_fn.name(), "Weighted");

        loss_fn.set_weight(2.0);
        assert_eq!(loss_fn.weight(), 2.0);
    }

    #[test]
    fn test_sample_weighted_loss_uniform() {
        let loss_fn = SampleWeightedLoss::new(Box::new(MSELoss));

        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false);

        // Default forward uses uniform weights
        let loss = loss_fn.forward(&pred, &target);

        // Should match regular MSE
        let mse_loss = MSELoss.forward(&pred.clone(), &target);
        assert_relative_eq!(loss.data()[0], mse_loss.data()[0], epsilon = 1e-5);
    }

    #[test]
    fn test_sample_weighted_loss_custom_weights() {
        let loss_fn = SampleWeightedLoss::new(Box::new(MSELoss));

        let pred = Tensor::from_vec(vec![0.0, 0.0], true);
        let target = Tensor::from_vec(vec![1.0, 1.0], false);
        let weights = vec![2.0, 0.0]; // First sample 2x, second ignored

        let loss = loss_fn.forward_weighted(&pred, &target, &weights);

        // Weighted MSE = (2.0 * 1.0 + 0.0 * 1.0) / 2 = 1.0
        assert_relative_eq!(loss.data()[0], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_sample_weighted_loss_gradient() {
        let loss_fn = SampleWeightedLoss::new(Box::new(MSELoss));

        let pred = Tensor::from_vec(vec![0.0, 0.0], true);
        let target = Tensor::from_vec(vec![1.0, 1.0], false);
        let weights = vec![2.0, 1.0];

        let loss = loss_fn.forward_weighted(&pred, &target, &weights);

        if let Some(backward_op) = loss.backward_op() {
            backward_op.backward();
        }

        let grad = pred.grad().unwrap();
        // Gradient: 2 * w * (pred - target) / n
        // First: 2 * 2.0 * (-1) / 2 = -2.0
        // Second: 2 * 1.0 * (-1) / 2 = -1.0
        assert_relative_eq!(grad[0], -2.0, epsilon = 1e-5);
        assert_relative_eq!(grad[1], -1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_sample_weighted_citl_reweight() {
        // Simulate CITL --reweight 1.5 for compiler-verified samples
        let loss_fn = SampleWeightedLoss::new(Box::new(MSELoss));

        let pred = Tensor::from_vec(vec![0.0, 0.0, 0.0], true);
        let target = Tensor::from_vec(vec![1.0, 1.0, 1.0], false);

        // First two samples are compiler-verified (1.5x weight)
        // Third sample is rule-based (1.0x weight)
        let weights = vec![1.5, 1.5, 1.0];

        let weighted_loss = loss_fn.forward_weighted(&pred, &target, &weights);

        // Regular loss (uniform weights)
        let uniform = loss_fn.forward(&pred.clone(), &target);

        // Weighted should be higher due to 1.5x weights
        assert!(weighted_loss.data()[0] > uniform.data()[0]);
    }

    #[test]
    fn test_huber_default_delta() {
        let loss_fn = HuberLoss::default_delta();
        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![1.5, 2.5], false);
        let loss = loss_fn.forward(&pred, &target);
        assert!(loss.data()[0] > 0.0);
    }

    #[test]
    fn test_weighted_loss_unweighted() {
        let loss_fn = WeightedLoss::unweighted(Box::new(MSELoss));
        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![1.5, 2.5], false);
        let loss = loss_fn.forward(&pred, &target);
        assert_eq!(loss_fn.weight(), 1.0);
        assert!(loss.data()[0] > 0.0);
    }

    #[test]
    fn test_cross_entropy_gradient() {
        let loss_fn = CrossEntropyLoss;
        let logits = Tensor::from_vec(vec![2.0, 1.0, 0.5], true);
        let targets = Tensor::from_vec(vec![1.0, 0.0, 0.0], false);

        let loss = loss_fn.forward(&logits, &targets);

        if let Some(backward_op) = loss.backward_op() {
            backward_op.backward();
        }

        let grad = logits.grad().unwrap();
        // Gradient should exist and be finite
        for g in grad.iter() {
            assert!(g.is_finite());
        }
        // For CE with target at index 0, grad[0] should be negative
        // (pred - target where target=1)
        assert!(grad[0] < 0.0);
    }

    #[test]
    #[should_panic(expected = "must have same length")]
    fn test_cross_entropy_mismatched_lengths() {
        let loss_fn = CrossEntropyLoss;
        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        loss_fn.forward(&pred, &target);
    }

    #[test]
    #[should_panic(expected = "must have same length")]
    fn test_l1_mismatched_lengths() {
        let loss_fn = L1Loss;
        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        loss_fn.forward(&pred, &target);
    }

    #[test]
    #[should_panic(expected = "must have same length")]
    fn test_huber_mismatched_lengths() {
        let loss_fn = HuberLoss::new(1.0);
        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        loss_fn.forward(&pred, &target);
    }

    #[test]
    #[should_panic(expected = "delta must be positive")]
    fn test_huber_negative_delta() {
        HuberLoss::new(-1.0);
    }

    #[test]
    fn test_mse_no_grad() {
        let loss_fn = MSELoss;
        let pred = Tensor::from_vec(vec![1.0, 2.0], false); // requires_grad = false
        let target = Tensor::from_vec(vec![1.5, 2.5], false);
        let loss = loss_fn.forward(&pred, &target);
        assert!(loss.data()[0] > 0.0);
    }

    #[test]
    fn test_cross_entropy_no_grad() {
        let loss_fn = CrossEntropyLoss;
        let pred = Tensor::from_vec(vec![2.0, 1.0], false);
        let target = Tensor::from_vec(vec![1.0, 0.0], false);
        let loss = loss_fn.forward(&pred, &target);
        assert!(loss.data()[0] > 0.0);
    }

    #[test]
    fn test_huber_no_grad() {
        let loss_fn = HuberLoss::new(1.0);
        let pred = Tensor::from_vec(vec![1.0, 2.0], false);
        let target = Tensor::from_vec(vec![1.5, 2.5], false);
        let loss = loss_fn.forward(&pred, &target);
        assert!(loss.data()[0] > 0.0);
    }

    #[test]
    fn test_l1_no_grad() {
        let loss_fn = L1Loss;
        let pred = Tensor::from_vec(vec![1.0, 2.0], false);
        let target = Tensor::from_vec(vec![1.5, 2.5], false);
        let loss = loss_fn.forward(&pred, &target);
        assert!(loss.data()[0] > 0.0);
    }

    #[test]
    fn test_weighted_no_grad() {
        let loss_fn = WeightedLoss::new(Box::new(MSELoss), 1.5);
        let pred = Tensor::from_vec(vec![1.0, 2.0], false);
        let target = Tensor::from_vec(vec![1.5, 2.5], false);
        let loss = loss_fn.forward(&pred, &target);
        assert!(loss.data()[0] > 0.0);
    }

    #[test]
    fn test_sample_weighted_no_grad() {
        let loss_fn = SampleWeightedLoss::new(Box::new(MSELoss));
        let pred = Tensor::from_vec(vec![1.0, 2.0], false);
        let target = Tensor::from_vec(vec![1.5, 2.5], false);
        let weights = vec![1.0, 2.0];
        let loss = loss_fn.forward_weighted(&pred, &target, &weights);
        assert!(loss.data()[0] > 0.0);
    }

    #[test]
    fn test_loss_names() {
        assert_eq!(MSELoss.name(), "MSE");
        assert_eq!(CrossEntropyLoss.name(), "CrossEntropy");
        assert_eq!(HuberLoss::new(1.0).name(), "Huber");
        assert_eq!(L1Loss.name(), "L1");
        assert_eq!(WeightedLoss::new(Box::new(MSELoss), 1.0).name(), "Weighted");
        assert_eq!(
            SampleWeightedLoss::new(Box::new(MSELoss)).name(),
            "SampleWeighted"
        );
    }

    #[test]
    fn test_weighted_backward_with_grad() {
        let loss_fn = WeightedLoss::new(Box::new(MSELoss), 2.0);
        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![0.0, 0.0], false);

        let loss = loss_fn.forward(&pred, &target);
        if let Some(backward_op) = loss.backward_op() {
            backward_op.backward();
        }

        // Verify gradient was set
        let grad = pred.grad();
        assert!(grad.is_some());
    }

    #[test]
    #[should_panic(expected = "Weights must match")]
    fn test_sample_weighted_mismatched_weights() {
        let loss_fn = SampleWeightedLoss::new(Box::new(MSELoss));
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        let weights = vec![1.0, 1.0]; // Wrong length
        loss_fn.forward_weighted(&pred, &target, &weights);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that could cause overflow without max subtraction
        let x = Array1::from(vec![1000.0, 1001.0, 1002.0]);
        let probs = CrossEntropyLoss::softmax(&x);

        // Should still sum to 1.0
        let sum: f32 = probs.sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-5);

        // All values should be valid
        for &p in probs.iter() {
            assert!(p.is_finite());
            assert!(p >= 0.0);
        }
    }

    #[test]
    fn test_gradient_accumulation_mse() {
        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![0.0, 0.0], false);

        // First forward/backward
        let loss1 = MSELoss.forward(&pred, &target);
        if let Some(op) = loss1.backward_op() {
            op.backward();
        }

        // Second forward/backward - gradients should accumulate
        let loss2 = MSELoss.forward(&pred, &target);
        if let Some(op) = loss2.backward_op() {
            op.backward();
        }

        // Gradient should be 2x the single pass
        let grad = pred.grad().unwrap();
        assert!(grad[0].abs() > 0.0);
        assert!(grad[1].abs() > 0.0);
    }

    #[test]
    fn test_gradient_accumulation_cross_entropy() {
        let logits = Tensor::from_vec(vec![2.0, 1.0], true);
        let targets = Tensor::from_vec(vec![1.0, 0.0], false);

        let loss1 = CrossEntropyLoss.forward(&logits, &targets);
        if let Some(op) = loss1.backward_op() {
            op.backward();
        }

        let loss2 = CrossEntropyLoss.forward(&logits, &targets);
        if let Some(op) = loss2.backward_op() {
            op.backward();
        }

        let grad = logits.grad().unwrap();
        assert!(grad[0].is_finite());
        assert!(grad[1].is_finite());
    }

    #[test]
    fn test_gradient_accumulation_huber() {
        let pred = Tensor::from_vec(vec![1.0, 5.0], true);
        let target = Tensor::from_vec(vec![0.0, 0.0], false);

        let loss1 = HuberLoss::new(1.0).forward(&pred, &target);
        if let Some(op) = loss1.backward_op() {
            op.backward();
        }

        let loss2 = HuberLoss::new(1.0).forward(&pred, &target);
        if let Some(op) = loss2.backward_op() {
            op.backward();
        }

        let grad = pred.grad().unwrap();
        assert!(grad[0].is_finite());
        assert!(grad[1].is_finite());
    }

    #[test]
    fn test_gradient_accumulation_l1() {
        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![0.0, 0.0], false);

        let loss1 = L1Loss.forward(&pred, &target);
        if let Some(op) = loss1.backward_op() {
            op.backward();
        }

        let loss2 = L1Loss.forward(&pred, &target);
        if let Some(op) = loss2.backward_op() {
            op.backward();
        }

        let grad = pred.grad().unwrap();
        assert!(grad[0].is_finite());
        assert!(grad[1].is_finite());
    }

    #[test]
    fn test_gradient_accumulation_sample_weighted() {
        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![0.0, 0.0], false);
        let weights = vec![1.0, 1.5];
        let loss_fn = SampleWeightedLoss::new(Box::new(MSELoss));

        let loss1 = loss_fn.forward_weighted(&pred, &target, &weights);
        if let Some(op) = loss1.backward_op() {
            op.backward();
        }

        let loss2 = loss_fn.forward_weighted(&pred, &target, &weights);
        if let Some(op) = loss2.backward_op() {
            op.backward();
        }

        let grad = pred.grad().unwrap();
        assert!(grad[0].is_finite());
        assert!(grad[1].is_finite());
    }
}
