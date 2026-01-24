//! Attention autograd operations: scaled dot-product attention
//!
//! Uses CUDA GEMM for Q@K^T and Attn@V operations when available.

use crate::autograd::{BackwardOp, Tensor};
use ndarray::Array1;
use std::cell::RefCell;
use std::rc::Rc;

// Import matmul_compute from sibling module for GPU-accelerated matrix operations
use super::matmul::{matmul_compute, transpose};

/// Scaled Dot-Product Attention (GPU-accelerated)
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

    // Step 1: Compute Q @ K^T (seq_len x seq_len) using GPU GEMM
    // Q is (seq_len, d_k), K is (seq_len, d_k), K^T is (d_k, seq_len)
    // Result: (seq_len, d_k) @ (d_k, seq_len) = (seq_len, seq_len)
    let q_slice = q.data().as_slice().unwrap();
    let k_slice = k.data().as_slice().unwrap();
    let k_t = transpose(k_slice, seq_len, d_k); // K^T: (d_k, seq_len)
    let mut scores = matmul_compute(q_slice, &k_t, seq_len, d_k, seq_len);

    // Apply scaling
    for score in &mut scores {
        *score /= scale;
    }

    // Step 2: Apply softmax row-wise (CPU for numerical stability)
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

    // Step 3: Compute attention_weights @ V (seq_len x d_v) using GPU GEMM
    // attention_weights is (seq_len, seq_len), V is (seq_len, d_v)
    // Result: (seq_len, seq_len) @ (seq_len, d_v) = (seq_len, d_v)
    let v_slice = v.data().as_slice().unwrap();
    let output_data = matmul_compute(&attention_weights, v_slice, seq_len, seq_len, d_v);

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
            let grad_out_slice = grad_output.as_slice().unwrap();
            let attn_slice = self.attention_weights.as_slice().unwrap();

            // Gradient w.r.t. V: attention_weights^T @ grad_output
            // attention_weights is (seq_len, seq_len), grad_output is (seq_len, d_v)
            // attention_weights^T is (seq_len, seq_len)
            // Result: (seq_len, seq_len) @ (seq_len, d_v) = (seq_len, d_v)
            if self.v.requires_grad() {
                let attn_t = transpose(attn_slice, seq_len, seq_len);
                let grad_v = matmul_compute(&attn_t, grad_out_slice, seq_len, seq_len, d_v);
                self.v.accumulate_grad(Array1::from(grad_v));
            }

            // Gradient w.r.t. attention_weights: grad_output @ V^T
            // grad_output is (seq_len, d_v), V is (seq_len, d_v), V^T is (d_v, seq_len)
            // Result: (seq_len, d_v) @ (d_v, seq_len) = (seq_len, seq_len)
            let v_slice = self.v.data().as_slice().unwrap();
            let v_t = transpose(v_slice, seq_len, d_v);
            let grad_attention_weights =
                matmul_compute(grad_out_slice, &v_t, seq_len, d_v, seq_len);

            // Gradient through softmax (row-wise) - must be CPU for numerical stability
            let mut grad_scores = vec![0.0; seq_len * seq_len];
            for i in 0..seq_len {
                let row_start = i * seq_len;
                for j in 0..seq_len {
                    let idx = row_start + j;
                    let p_j = attn_slice[idx];

                    // Softmax gradient: p_j * (grad_j - sum_k(p_k * grad_k))
                    let mut sum_pk_gradk = 0.0;
                    for k in 0..seq_len {
                        let k_idx = row_start + k;
                        sum_pk_gradk += attn_slice[k_idx] * grad_attention_weights[k_idx];
                    }

                    grad_scores[idx] = p_j * (grad_attention_weights[idx] - sum_pk_gradk);
                }
            }

            // Gradient through scaling
            for g in &mut grad_scores {
                *g /= self.scale;
            }

            // Gradient w.r.t. Q: grad_scaled @ K
            // grad_scaled is (seq_len, seq_len), K is (seq_len, d_k)
            // Result: (seq_len, seq_len) @ (seq_len, d_k) = (seq_len, d_k)
            if self.q.requires_grad() {
                let k_slice = self.k.data().as_slice().unwrap();
                let grad_q = matmul_compute(&grad_scores, k_slice, seq_len, seq_len, d_k);
                self.q.accumulate_grad(Array1::from(grad_q));
            }

            // Gradient w.r.t. K: grad_scaled^T @ Q
            // grad_scaled is (seq_len, seq_len), grad_scaled^T is (seq_len, seq_len)
            // Q is (seq_len, d_k)
            // Result: (seq_len, seq_len) @ (seq_len, d_k) = (seq_len, d_k)
            if self.k.requires_grad() {
                let grad_t = transpose(&grad_scores, seq_len, seq_len);
                let q_slice = self.q.data().as_slice().unwrap();
                let grad_k = matmul_compute(&grad_t, q_slice, seq_len, seq_len, d_k);
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
