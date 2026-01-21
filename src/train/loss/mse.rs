//! Mean Squared Error, Mean Absolute Error, and Huber losses

use crate::Tensor;
use ndarray::Array1;

use super::LossFn;

/// Mean Squared Error Loss
///
/// L = mean((predictions - targets)^2)
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

/// Huber Loss (Smooth L1 Loss)
///
/// Combines MSE for small errors and MAE for large errors,
/// making it robust to outliers.
///
/// For |error| <= delta:  L = 0.5 * error^2
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
    #[should_panic(expected = "must have same length")]
    fn test_mse_mismatched_lengths() {
        let loss_fn = MSELoss;
        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);

        loss_fn.forward(&pred, &target);
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
    #[should_panic(expected = "must have same length")]
    fn test_l1_mismatched_lengths() {
        let loss_fn = L1Loss;
        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        loss_fn.forward(&pred, &target);
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
    fn test_huber_default_delta() {
        let loss_fn = HuberLoss::default_delta();
        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![1.5, 2.5], false);
        let loss = loss_fn.forward(&pred, &target);
        assert!(loss.data()[0] > 0.0);
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
    fn test_huber_no_grad() {
        let loss_fn = HuberLoss::new(1.0);
        let pred = Tensor::from_vec(vec![1.0, 2.0], false);
        let target = Tensor::from_vec(vec![1.5, 2.5], false);
        let loss = loss_fn.forward(&pred, &target);
        assert!(loss.data()[0] > 0.0);
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
}
