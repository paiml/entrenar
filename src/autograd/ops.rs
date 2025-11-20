//! Autograd operations with backward passes

use super::{BackwardOp, Tensor};
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

/// Softmax activation
pub fn softmax(a: &Tensor) -> Tensor {
    let max_val = a.data().iter().cloned().fold(f32::NEG_INFINITY, f32::max);
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

/// Matrix multiplication
///
/// Computes C = A @ B where:
/// - A is m×k (flattened to length m*k)
/// - B is k×n (flattened to length k*n)
/// - C is m×n (flattened to length m*n)
///
/// # Arguments
/// * `a` - Left matrix (m×k flattened)
/// * `b` - Right matrix (k×n flattened)
/// * `m` - Number of rows in A
/// * `k` - Number of columns in A (= rows in B)
/// * `n` - Number of columns in B
pub fn matmul(a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Tensor {
    assert_eq!(a.len(), m * k, "Matrix A size mismatch");
    assert_eq!(b.len(), k * n, "Matrix B size mismatch");

    // Compute C = A @ B
    let mut result_data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a.data()[i * k + p] * b.data()[p * n + j];
            }
            result_data[i * n + j] = sum;
        }
    }

    let requires_grad = a.requires_grad() || b.requires_grad();
    let mut result = Tensor::new(Array1::from(result_data), requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let b_clone = b.clone();
        let backward_op = Rc::new(MatmulBackward {
            a: a_clone,
            b: b_clone,
            m,
            k,
            n,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    result
}

struct MatmulBackward {
    a: Tensor,
    b: Tensor,
    m: usize,
    k: usize,
    n: usize,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for MatmulBackward {
    fn backward(&self) {
        if let Some(grad_output) = self.result_grad.borrow().as_ref() {
            // ∂L/∂A = ∂L/∂C @ B^T
            // ∂L/∂B = A^T @ ∂L/∂C

            if self.a.requires_grad() {
                let mut grad_a = vec![0.0; self.m * self.k];
                // grad_A[i,p] = sum_j grad_C[i,j] * B[p,j]
                for i in 0..self.m {
                    for p in 0..self.k {
                        let mut sum = 0.0;
                        for j in 0..self.n {
                            sum += grad_output[i * self.n + j] * self.b.data()[p * self.n + j];
                        }
                        grad_a[i * self.k + p] = sum;
                    }
                }
                self.a.accumulate_grad(Array1::from(grad_a));
            }

            if self.b.requires_grad() {
                let mut grad_b = vec![0.0; self.k * self.n];
                // grad_B[p,j] = sum_i A[i,p] * grad_C[i,j]
                for p in 0..self.k {
                    for j in 0..self.n {
                        let mut sum = 0.0;
                        for i in 0..self.m {
                            sum += self.a.data()[i * self.k + p] * grad_output[i * self.n + j];
                        }
                        grad_b[p * self.n + j] = sum;
                    }
                }
                self.b.accumulate_grad(Array1::from(grad_b));
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
