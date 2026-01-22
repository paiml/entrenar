//! Activation function autograd operations: relu, gelu, swish, softmax

use crate::autograd::{BackwardOp, Tensor};
use ndarray::Array1;
use std::cell::RefCell;
use std::rc::Rc;

/// ReLU activation
pub fn relu(a: &Tensor) -> Tensor {
    let data = a.data().mapv(|x| x.max(0.0));
    let requires_grad = a.requires_grad();

    let mut result = Tensor::new(data, requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let backward_op = Rc::new(ReluBackward {
            a: a_clone,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    result
}

struct ReluBackward {
    a: Tensor,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for ReluBackward {
    fn backward(&self) {
        if let Some(grad) = self.result_grad.borrow().as_ref() {
            if self.a.requires_grad() {
                // ∂L/∂a = ∂L/∂out * (a > 0)
                let grad_a = grad * &self.a.data().mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                self.a.accumulate_grad(grad_a);
            }

            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
        }
    }
}

/// GELU activation (Gaussian Error Linear Unit)
///
/// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
pub fn gelu(a: &Tensor) -> Tensor {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // √(2/π)
    const COEFF: f32 = 0.044_715;

    let data = a.data().mapv(|x| {
        let x3 = x * x * x;
        let inner = SQRT_2_OVER_PI * (x + COEFF * x3);
        0.5 * x * (1.0 + inner.tanh())
    });

    let requires_grad = a.requires_grad();
    let mut result = Tensor::new(data, requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let backward_op = Rc::new(GeluBackward {
            a: a_clone,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    result
}

struct GeluBackward {
    a: Tensor,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for GeluBackward {
    fn backward(&self) {
        if let Some(grad_output) = self.result_grad.borrow().as_ref() {
            if self.a.requires_grad() {
                const SQRT_2_OVER_PI: f32 = 0.797_884_6;
                const COEFF: f32 = 0.044_715;

                // ∂GELU/∂x = 0.5 * (1 + tanh(z)) + 0.5 * x * sech²(z) * dz/dx
                // where z = √(2/π) * (x + 0.044715 * x³)
                // and dz/dx = √(2/π) * (1 + 3 * 0.044715 * x²)
                let grad_a: Vec<f32> = self
                    .a
                    .data()
                    .iter()
                    .zip(grad_output.iter())
                    .map(|(&x, &grad)| {
                        let x2 = x * x;
                        let x3 = x2 * x;
                        let z = SQRT_2_OVER_PI * (x + COEFF * x3);
                        let tanh_z = z.tanh();
                        let sech2_z = 1.0 - tanh_z * tanh_z;
                        let dz_dx = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x2);

                        let gelu_grad = 0.5 * (1.0 + tanh_z) + 0.5 * x * sech2_z * dz_dx;
                        grad * gelu_grad
                    })
                    .collect();

                self.a.accumulate_grad(Array1::from(grad_a));
            }

            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
        }
    }
}

/// Swish activation (also known as SiLU - Sigmoid Linear Unit)
///
/// Swish(x) = x * sigmoid(x) = x / (1 + e^(-x))
pub fn swish(a: &Tensor) -> Tensor {
    let data = a.data().mapv(|x| {
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        x * sigmoid
    });

    let requires_grad = a.requires_grad();
    let mut result = Tensor::new(data, requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let output_clone = result.clone();
        let backward_op = Rc::new(SwishBackward {
            a: a_clone,
            output: output_clone,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    result
}

struct SwishBackward {
    a: Tensor,
    output: Tensor,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for SwishBackward {
    fn backward(&self) {
        if let Some(grad_output) = self.result_grad.borrow().as_ref() {
            if self.a.requires_grad() {
                // ∂Swish/∂x = Swish(x) + sigmoid(x) * (1 - Swish(x))
                // This can be simplified to: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                let grad_a: Vec<f32> = self
                    .a
                    .data()
                    .iter()
                    .zip(self.output.data().iter())
                    .zip(grad_output.iter())
                    .map(|((&x, &swish_x), &grad)| {
                        let sigmoid = 1.0 / (1.0 + (-x).exp());
                        let swish_grad = swish_x + sigmoid * (1.0 - swish_x);
                        grad * swish_grad
                    })
                    .collect();

                self.a.accumulate_grad(Array1::from(grad_a));
            }

            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
        }
    }
}

/// Softmax activation
pub fn softmax(a: &Tensor) -> Tensor {
    let max_val = a.data().iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals = a.data().mapv(|x| (x - max_val).exp());
    let sum_exp = exp_vals.sum();
    let data = exp_vals / sum_exp;

    let requires_grad = a.requires_grad();
    let mut result = Tensor::new(data, requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let output_clone = result.clone();
        let backward_op = Rc::new(SoftmaxBackward {
            a: a_clone,
            output: output_clone,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    result
}

struct SoftmaxBackward {
    a: Tensor,
    output: Tensor,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for SoftmaxBackward {
    fn backward(&self) {
        if let Some(grad_output) = self.result_grad.borrow().as_ref() {
            if self.a.requires_grad() {
                // ∂L/∂x = y ⊙ (∂L/∂y - (y · ∂L/∂y))
                let y = self.output.data();
                let dot = (y * grad_output).sum();
                let grad_a = y * &(grad_output - dot);
                self.a.accumulate_grad(grad_a);
            }

            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
        }
    }
}
