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
