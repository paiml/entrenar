//! Basic autograd operations: add, mul, scale, sum

use crate::autograd::{BackwardOp, Tensor};
use ndarray::Array1;
use std::cell::RefCell;
use std::rc::Rc;

/// Add two tensors
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let data = a.data() + b.data();
    let requires_grad = a.requires_grad() || b.requires_grad();

    let mut result = Tensor::new(data, requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let b_clone = b.clone();
        let backward_op = Rc::new(AddBackward {
            a: a_clone,
            b: b_clone,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    result
}

struct AddBackward {
    a: Tensor,
    b: Tensor,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for AddBackward {
    fn backward(&self) {
        if let Some(grad) = self.result_grad.borrow().as_ref() {
            if self.a.requires_grad() {
                self.a.accumulate_grad(grad.clone());
            }
            if self.b.requires_grad() {
                self.b.accumulate_grad(grad.clone());
            }

            // Recursively call backward on inputs
            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
            if let Some(op) = self.b.backward_op() {
                op.backward();
            }
        }
    }
}

/// Multiply two tensors element-wise
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let data = a.data() * b.data();
    let requires_grad = a.requires_grad() || b.requires_grad();

    let mut result = Tensor::new(data, requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let b_clone = b.clone();
        let backward_op = Rc::new(MulBackward {
            a: a_clone,
            b: b_clone,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    result
}

struct MulBackward {
    a: Tensor,
    b: Tensor,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for MulBackward {
    fn backward(&self) {
        if let Some(grad) = self.result_grad.borrow().as_ref() {
            if self.a.requires_grad() {
                // ∂L/∂a = ∂L/∂out * b
                let grad_a = grad * self.b.data();
                self.a.accumulate_grad(grad_a);
            }
            if self.b.requires_grad() {
                // ∂L/∂b = ∂L/∂out * a
                let grad_b = grad * self.a.data();
                self.b.accumulate_grad(grad_b);
            }

            // Recursively call backward on inputs
            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
            if let Some(op) = self.b.backward_op() {
                op.backward();
            }
        }
    }
}

/// Scale tensor by a scalar
pub fn scale(a: &Tensor, factor: f32) -> Tensor {
    let data = a.data() * factor;
    let requires_grad = a.requires_grad();

    let mut result = Tensor::new(data, requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let backward_op = Rc::new(ScaleBackward {
            a: a_clone,
            factor,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    result
}

struct ScaleBackward {
    a: Tensor,
    factor: f32,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for ScaleBackward {
    fn backward(&self) {
        if let Some(grad) = self.result_grad.borrow().as_ref() {
            if self.a.requires_grad() {
                // ∂L/∂a = ∂L/∂out * factor
                let grad_a = grad * self.factor;
                self.a.accumulate_grad(grad_a);
            }

            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
        }
    }
}

/// Add tensor b scaled by a factor to tensor a: result = a + scale * b
///
/// This is useful for LoRA: y = Wx + scale * (BA)x
pub fn add_scaled(a: &Tensor, b: &Tensor, scale_factor: f32) -> Tensor {
    assert_eq!(a.len(), b.len(), "Tensors must have same length");

    // Compute a + scale * b
    let a_data = a.data();
    let b_data = b.data();
    let result_data: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(&a_val, &b_val)| a_val + scale_factor * b_val)
        .collect();

    let requires_grad = a.requires_grad() || b.requires_grad();
    let mut result = Tensor::new(Array1::from(result_data), requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let b_clone = b.clone();
        let backward_op = Rc::new(AddScaledBackward {
            a: a_clone,
            b: b_clone,
            scale: scale_factor,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    result
}

struct AddScaledBackward {
    a: Tensor,
    b: Tensor,
    scale: f32,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for AddScaledBackward {
    fn backward(&self) {
        if let Some(grad) = self.result_grad.borrow().as_ref() {
            // ∂L/∂a = ∂L/∂result * 1 = grad
            if self.a.requires_grad() {
                self.a.accumulate_grad(grad.clone());
            }

            // ∂L/∂b = ∂L/∂result * scale = grad * scale
            if self.b.requires_grad() {
                let grad_b = grad * self.scale;
                self.b.accumulate_grad(grad_b);
            }

            // Recursively call backward on inputs
            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
            if let Some(op) = self.b.backward_op() {
                op.backward();
            }
        }
    }
}

/// Sum all elements
pub fn sum(a: &Tensor) -> Tensor {
    let data = Array1::from(vec![a.data().sum()]);
    let requires_grad = a.requires_grad();

    let mut result = Tensor::new(data, requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let backward_op = Rc::new(SumBackward {
            a: a_clone,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    result
}

struct SumBackward {
    a: Tensor,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for SumBackward {
    fn backward(&self) {
        if let Some(grad) = self.result_grad.borrow().as_ref() {
            if self.a.requires_grad() {
                // ∂L/∂a = ∂L/∂sum * 1 (broadcast)
                let grad_val = grad[0];
                let grad_a = Array1::from(vec![grad_val; self.a.len()]);
                self.a.accumulate_grad(grad_a);
            }

            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
        }
    }
}
