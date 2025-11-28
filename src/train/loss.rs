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
}
