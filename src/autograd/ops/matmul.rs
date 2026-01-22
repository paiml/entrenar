//! Matrix multiplication autograd operations

use crate::autograd::{BackwardOp, Tensor};
use ndarray::Array1;
use std::cell::RefCell;
use std::rc::Rc;

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
