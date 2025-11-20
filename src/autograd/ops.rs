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

/// Scaled Dot-Product Attention
///
/// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
///
/// Parameters:
/// - q: Query matrix (seq_len x d_k, stored flattened)
/// - k: Key matrix (seq_len x d_k, stored flattened)
/// - v: Value matrix (seq_len x d_v, stored flattened)
/// - seq_len: Sequence length
/// - d_k: Dimension of queries and keys
/// - d_v: Dimension of values
///
/// Returns: Tensor of shape (seq_len x d_v, stored flattened)
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_len: usize,
    d_k: usize,
    _k_seq_len: usize, // Kept for API compatibility, assumes same as seq_len
    d_v: usize,
) -> Tensor {
    let scale = (d_k as f32).sqrt();

    // Step 1: Compute Q @ K^T (seq_len x seq_len)
    let mut scores = vec![0.0; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut dot = 0.0;
            for p in 0..d_k {
                dot += q.data()[i * d_k + p] * k.data()[j * d_k + p];
            }
            scores[i * seq_len + j] = dot / scale;
        }
    }

    // Step 2: Apply softmax row-wise
    let mut attention_weights = vec![0.0; seq_len * seq_len];
    for i in 0..seq_len {
        let row_start = i * seq_len;
        let row_end = row_start + seq_len;
        let row = &scores[row_start..row_end];

        // Softmax for numerical stability
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();

        for (j, &exp_val) in exp_vals.iter().enumerate() {
            attention_weights[row_start + j] = exp_val / sum_exp;
        }
    }

    // Step 3: Compute attention_weights @ V (seq_len x d_v)
    let mut output_data = vec![0.0; seq_len * d_v];
    for i in 0..seq_len {
        for j in 0..d_v {
            let mut sum = 0.0;
            for p in 0..seq_len {
                sum += attention_weights[i * seq_len + p] * v.data()[p * d_v + j];
            }
            output_data[i * d_v + j] = sum;
        }
    }

    let requires_grad = q.requires_grad() || k.requires_grad() || v.requires_grad();
    let mut result = Tensor::new(Array1::from(output_data), requires_grad);

    if requires_grad {
        let q_clone = q.clone();
        let k_clone = k.clone();
        let v_clone = v.clone();
        let backward_op = Rc::new(AttentionBackward {
            q: q_clone,
            k: k_clone,
            v: v_clone,
            attention_weights: Array1::from(attention_weights),
            seq_len,
            d_k,
            d_v,
            scale,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    result
}

struct AttentionBackward {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_weights: Array1<f32>,
    seq_len: usize,
    d_k: usize,
    d_v: usize,
    scale: f32,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for AttentionBackward {
    fn backward(&self) {
        if let Some(grad_output) = self.result_grad.borrow().as_ref() {
            let seq_len = self.seq_len;
            let d_k = self.d_k;
            let d_v = self.d_v;

            // Gradient w.r.t. V: attention_weights^T @ grad_output
            if self.v.requires_grad() {
                let mut grad_v = vec![0.0; seq_len * d_v];
                for i in 0..seq_len {
                    for j in 0..d_v {
                        let mut sum = 0.0;
                        for p in 0..seq_len {
                            // attention_weights^T[i,p] = attention_weights[p,i]
                            sum +=
                                self.attention_weights[p * seq_len + i] * grad_output[p * d_v + j];
                        }
                        grad_v[i * d_v + j] = sum;
                    }
                }
                self.v.accumulate_grad(Array1::from(grad_v));
            }

            // Gradient w.r.t. attention_weights: grad_output @ V^T
            let mut grad_attention_weights = vec![0.0; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut sum = 0.0;
                    for p in 0..d_v {
                        // V^T[p,j] = V[j,p]
                        sum += grad_output[i * d_v + p] * self.v.data()[j * d_v + p];
                    }
                    grad_attention_weights[i * seq_len + j] = sum;
                }
            }

            // Gradient through softmax (row-wise)
            let mut grad_scores = vec![0.0; seq_len * seq_len];
            for i in 0..seq_len {
                let row_start = i * seq_len;
                for j in 0..seq_len {
                    let idx = row_start + j;
                    let p_j = self.attention_weights[idx];

                    // Softmax gradient: p_j * (grad_j - sum_k(p_k * grad_k))
                    let mut sum_pk_gradk = 0.0;
                    for k in 0..seq_len {
                        let k_idx = row_start + k;
                        sum_pk_gradk +=
                            self.attention_weights[k_idx] * grad_attention_weights[k_idx];
                    }

                    grad_scores[idx] = p_j * (grad_attention_weights[idx] - sum_pk_gradk);
                }
            }

            // Gradient through scaling
            let grad_scaled: Vec<f32> = grad_scores.iter().map(|&g| g / self.scale).collect();

            // Gradient w.r.t. Q: grad_qk @ K
            if self.q.requires_grad() {
                let mut grad_q = vec![0.0; seq_len * d_k];
                for i in 0..seq_len {
                    for j in 0..d_k {
                        let mut sum = 0.0;
                        for p in 0..seq_len {
                            sum += grad_scaled[i * seq_len + p] * self.k.data()[p * d_k + j];
                        }
                        grad_q[i * d_k + j] = sum;
                    }
                }
                self.q.accumulate_grad(Array1::from(grad_q));
            }

            // Gradient w.r.t. K: grad_qk^T @ Q
            if self.k.requires_grad() {
                let mut grad_k = vec![0.0; seq_len * d_k];
                for i in 0..seq_len {
                    for j in 0..d_k {
                        let mut sum = 0.0;
                        for p in 0..seq_len {
                            // grad_qk^T[i,p] = grad_qk[p,i]
                            sum += grad_scaled[p * seq_len + i] * self.q.data()[p * d_k + j];
                        }
                        grad_k[i * d_k + j] = sum;
                    }
                }
                self.k.accumulate_grad(Array1::from(grad_k));
            }

            // Continue backward through the graph
            if let Some(op) = self.q.backward_op() {
                op.backward();
            }
            if let Some(op) = self.k.backward_op() {
                op.backward();
            }
            if let Some(op) = self.v.backward_op() {
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
