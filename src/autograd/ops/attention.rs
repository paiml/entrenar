//! Attention autograd operations: scaled dot-product attention

use crate::autograd::{BackwardOp, Tensor};
use ndarray::Array1;
use std::cell::RefCell;
use std::rc::Rc;

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
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
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
