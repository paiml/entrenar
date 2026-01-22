//! Normalization autograd operations: layer_norm

use crate::autograd::{BackwardOp, Tensor};
use ndarray::Array1;
use std::cell::RefCell;
use std::rc::Rc;

/// Layer Normalization
///
/// Normalizes input to have mean=0 and variance=1, then applies learned scale (gamma) and shift (beta)
/// LayerNorm(x) = gamma * (x - mean) / sqrt(var + epsilon) + beta
pub fn layer_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor, epsilon: f32) -> Tensor {
    let n = x.len() as f32;

    // Compute mean
    let mean = x.data().sum() / n;

    // Compute variance
    let variance = x.data().mapv(|val| (val - mean).powi(2)).sum() / n;
    let std = (variance + epsilon).sqrt();

    // Normalize
    let normalized = x.data().mapv(|val| (val - mean) / std);

    // Scale and shift
    let data = &normalized * gamma.data() + beta.data();

    let requires_grad = x.requires_grad() || gamma.requires_grad() || beta.requires_grad();
    let mut result = Tensor::new(data, requires_grad);

    if requires_grad {
        let x_clone = x.clone();
        let gamma_clone = gamma.clone();
        let beta_clone = beta.clone();
        let backward_op = Rc::new(LayerNormBackward {
            x: x_clone,
            gamma: gamma_clone,
            beta: beta_clone,
            normalized: normalized.clone(),
            std,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    result
}

struct LayerNormBackward {
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    normalized: Array1<f32>,
    std: f32,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for LayerNormBackward {
    fn backward(&self) {
        if let Some(grad_output) = self.result_grad.borrow().as_ref() {
            let n = self.x.len() as f32;

            // ∂L/∂beta = ∂L/∂y (gradient flows directly through addition)
            if self.beta.requires_grad() {
                self.beta.accumulate_grad(grad_output.clone());
            }

            // ∂L/∂gamma = ∂L/∂y * x_normalized
            if self.gamma.requires_grad() {
                let grad_gamma = grad_output * &self.normalized;
                self.gamma.accumulate_grad(grad_gamma);
            }

            // ∂L/∂x is complex due to mean and variance dependencies
            if self.x.requires_grad() {
                // Gradient through scale: grad_normalized = grad_output * gamma
                let grad_normalized = grad_output * self.gamma.data();

                // Sum of gradients (for mean term)
                let sum_grad = grad_normalized.sum();

                // Sum of gradients weighted by normalized values (for variance term)
                let sum_grad_normalized = (&grad_normalized * &self.normalized).sum();

                // Full gradient formula:
                // ∂L/∂x_i = (1/std) * [grad_normalized_i - (1/n)*sum_grad - (1/n)*normalized_i*sum_grad_normalized]
                let grad_x: Vec<f32> = grad_normalized
                    .iter()
                    .zip(self.normalized.iter())
                    .map(|(&grad_norm, &norm)| {
                        (grad_norm - sum_grad / n - norm * sum_grad_normalized / n) / self.std
                    })
                    .collect();

                self.x.accumulate_grad(Array1::from(grad_x));
            }

            // Continue backward through the graph
            if let Some(op) = self.x.backward_op() {
                op.backward();
            }
            if let Some(op) = self.gamma.backward_op() {
                op.backward();
            }
            if let Some(op) = self.beta.backward_op() {
                op.backward();
            }
        }
    }
}
