//! Cross Entropy Loss for classification

use crate::Tensor;
use ndarray::Array1;

use super::LossFn;

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
    pub(crate) fn softmax(x: &Array1<f32>) -> Array1<f32> {
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
            .map(|(&t, &p)| -t * (p + 1e-10).max(f32::MIN_POSITIVE).ln())
            .sum();

        // Create loss tensor
        let mut loss = Tensor::from_vec(vec![ce], true);

        // Set up gradient: d(CE)/d(logits) = probs - targets
        let grad = &probs - targets.data();

        use crate::autograd::BackwardOp;
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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
        for &p in &probs {
            assert!((0.0..=1.0).contains(&p));
        }
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
        for g in &grad {
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
    fn test_cross_entropy_no_grad() {
        let loss_fn = CrossEntropyLoss;
        let pred = Tensor::from_vec(vec![2.0, 1.0], false);
        let target = Tensor::from_vec(vec![1.0, 0.0], false);
        let loss = loss_fn.forward(&pred, &target);
        assert!(loss.data()[0] > 0.0);
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
        for &p in &probs {
            assert!(p.is_finite());
            assert!(p >= 0.0);
        }
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
}
