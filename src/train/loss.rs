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

    fn name(&self) -> &str {
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

    fn name(&self) -> &str {
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

    fn name(&self) -> &str {
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
        let abs_diff = diff.mapv(|x| x.abs());
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

    fn name(&self) -> &str {
        "L1"
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
}
