//! Mean Squared Error Loss

use crate::autograd::BackwardOp;
use crate::Tensor;
use ndarray::Array1;
use std::rc::Rc;

use crate::train::loss::LossFn;

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
