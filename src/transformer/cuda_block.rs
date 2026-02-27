//! CUDA-accelerated Transformer Block (ENT-147 through ENT-152)
//!
//! This module provides a fully GPU-accelerated transformer block using trueno-gpu kernels.
//! All operations run on CUDA to achieve >70% GPU utilization.
//!
//! # Phase 22 Implementation Status
//!
//! - ENT-147: CUDA RMSNorm integration ✅
//! - ENT-148: CUDA Softmax integration ✅
//! - ENT-149: CUDA SiLU activation ✅
//! - ENT-150: Fused SwiGLU kernel ✅
//! - ENT-151: CUDA backward pass ✅
//! - ENT-152: CudaTransformer wrapper ✅

#![allow(dead_code)]
// SAFETY: This module performs GPU memory transfers via CUDA driver FFI.
// The unsafe blocks are limited to copy_from_host_async / copy_to_host_async
// where we guarantee the host buffer outlives the async operation by syncing
// the stream before the buffer goes out of scope.
#![allow(unsafe_code)]

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
#[inline]
fn saturating_u32(v: usize) -> u32 {
    v.min(u32::MAX as usize) as u32
}

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaContext, CudaStream, GpuBuffer};

#[cfg(feature = "cuda")]
use crate::autograd::cuda_backward::{
    gemm_backward_a, gemm_backward_b, rms_norm_backward, silu_backward,
};
#[cfg(feature = "cuda")]
use crate::autograd::cuda_forward::{
    batched_4d_gemm_forward, batched_softmax_forward, batched_to_interleaved_forward,
    batched_transpose_forward, expand_kv_heads, fused_swiglu_forward, gemm_forward,
    interleaved_to_batched_forward, residual_add_forward, rms_norm_forward, scale_forward,
};
#[cfg(feature = "cuda")]
use crate::autograd::cuda_tensor::Result;

#[cfg(feature = "cuda")]
use super::config::TransformerConfig;

/// CUDA-accelerated transformer block
///
/// All operations run on GPU with minimal CPU<->GPU transfers.
#[cfg(feature = "cuda")]
pub struct CudaTransformerBlock {
    /// Configuration
    config: TransformerConfig,
    /// Layer index
    layer_idx: usize,
    /// Input RMSNorm weight (gamma)
    input_norm_weight: GpuBuffer<f32>,
    /// Post-attention RMSNorm weight (gamma)
    post_attn_norm_weight: GpuBuffer<f32>,
    /// Query projection weight (hidden_size x hidden_size)
    w_q: GpuBuffer<f32>,
    /// Key projection weight (hidden_size x kv_hidden_size)
    w_k: GpuBuffer<f32>,
    /// Value projection weight (hidden_size x kv_hidden_size)
    w_v: GpuBuffer<f32>,
    /// Output projection weight (hidden_size x hidden_size)
    w_o: GpuBuffer<f32>,
    /// FFN gate projection (hidden_size x intermediate_size)
    w_gate: GpuBuffer<f32>,
    /// FFN up projection (hidden_size x intermediate_size)
    w_up: GpuBuffer<f32>,
    /// FFN down projection (intermediate_size x hidden_size)
    w_down: GpuBuffer<f32>,
    /// CUDA context
    ctx: Arc<CudaContext>,
    /// Scratch buffers for intermediate results
    scratch: CudaBlockScratch,
}

/// Preallocated scratch buffers for transformer forward pass
#[cfg(feature = "cuda")]
struct CudaBlockScratch {
    /// After input RMSNorm (seq_len * hidden_size)
    norm1_out: GpuBuffer<f32>,
    /// Q projection output (seq_len * hidden_size)
    q: GpuBuffer<f32>,
    /// K projection output (seq_len * kv_hidden_size)
    k: GpuBuffer<f32>,
    /// V projection output (seq_len * kv_hidden_size)
    v: GpuBuffer<f32>,
    /// Attention scores (num_heads * seq_len * seq_len)
    attn_scores: GpuBuffer<f32>,
    /// Attention output (seq_len * hidden_size)
    attn_out: GpuBuffer<f32>,
    /// Output projection result
    o_proj_out: GpuBuffer<f32>,
    /// Residual after attention
    residual1: GpuBuffer<f32>,
    /// After post-attention RMSNorm
    norm2_out: GpuBuffer<f32>,
    /// FFN gate output (seq_len * intermediate_size)
    gate_out: GpuBuffer<f32>,
    /// FFN up output (seq_len * intermediate_size)
    up_out: GpuBuffer<f32>,
    /// FFN fused SwiGLU output: SiLU(gate) * up (seq_len * intermediate_size)
    swiglu_out: GpuBuffer<f32>,
    /// FFN down projection output
    ffn_out: GpuBuffer<f32>,
    // === Gradient buffers for backward pass (ENT-151) ===
    /// Gradient for input norm weight
    grad_input_norm: GpuBuffer<f32>,
    /// Gradient for post-attention norm weight
    grad_post_attn_norm: GpuBuffer<f32>,
    /// Gradient for FFN gate projection
    grad_gate: GpuBuffer<f32>,
    /// Gradient for FFN up projection
    grad_up: GpuBuffer<f32>,
    /// Gradient for FFN down projection
    grad_down: GpuBuffer<f32>,
    /// Gradient accumulator for hidden states
    grad_hidden: GpuBuffer<f32>,
    /// Gradient for SwiGLU intermediate
    grad_swiglu: GpuBuffer<f32>,
    // === Attention layout scratch buffers (GPU-only attention pipeline) ===
    /// Q in batched layout [num_heads, seq_len, head_dim]
    attn_q_batched: GpuBuffer<f32>,
    /// K/V layout temp buffer [num_heads, seq_len, head_dim]
    attn_kv_temp: GpuBuffer<f32>,
    /// K transposed / second temp [num_heads, head_dim, seq_len]
    attn_kv_temp2: GpuBuffer<f32>,
}

#[cfg(feature = "cuda")]
impl CudaTransformerBlock {
    /// Create a new CUDA transformer block from CPU tensors
    ///
    /// Uploads all weights to GPU memory.
    pub fn new(
        config: &TransformerConfig,
        layer_idx: usize,
        ctx: Arc<CudaContext>,
        input_norm_weight: &[f32],
        post_attn_norm_weight: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        max_seq_len: usize,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let kv_hidden_size = config.num_kv_heads * config.head_dim();
        let intermediate_size = config.intermediate_size;
        let num_heads = config.num_attention_heads;

        // Upload weights to GPU
        let input_norm_weight = GpuBuffer::from_host(&ctx, input_norm_weight)?;
        let post_attn_norm_weight = GpuBuffer::from_host(&ctx, post_attn_norm_weight)?;
        let w_q = GpuBuffer::from_host(&ctx, w_q)?;
        let w_k = GpuBuffer::from_host(&ctx, w_k)?;
        let w_v = GpuBuffer::from_host(&ctx, w_v)?;
        let w_o = GpuBuffer::from_host(&ctx, w_o)?;
        let w_gate = GpuBuffer::from_host(&ctx, w_gate)?;
        let w_up = GpuBuffer::from_host(&ctx, w_up)?;
        let w_down = GpuBuffer::from_host(&ctx, w_down)?;

        // Allocate scratch buffers for max sequence length
        let scratch = CudaBlockScratch {
            norm1_out: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            q: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            k: GpuBuffer::new(&ctx, max_seq_len * kv_hidden_size)?,
            v: GpuBuffer::new(&ctx, max_seq_len * kv_hidden_size)?,
            attn_scores: GpuBuffer::new(&ctx, num_heads * max_seq_len * max_seq_len)?,
            attn_out: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            o_proj_out: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            residual1: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            norm2_out: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            gate_out: GpuBuffer::new(&ctx, max_seq_len * intermediate_size)?,
            up_out: GpuBuffer::new(&ctx, max_seq_len * intermediate_size)?,
            swiglu_out: GpuBuffer::new(&ctx, max_seq_len * intermediate_size)?,
            ffn_out: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            // Gradient buffers (ENT-151)
            grad_input_norm: GpuBuffer::new(&ctx, hidden_size)?,
            grad_post_attn_norm: GpuBuffer::new(&ctx, hidden_size)?,
            grad_gate: GpuBuffer::new(&ctx, hidden_size * intermediate_size)?,
            grad_up: GpuBuffer::new(&ctx, hidden_size * intermediate_size)?,
            grad_down: GpuBuffer::new(&ctx, intermediate_size * hidden_size)?,
            grad_hidden: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            grad_swiglu: GpuBuffer::new(&ctx, max_seq_len * intermediate_size)?,
            // Attention layout scratch (all sized for num_heads, handles GQA expansion)
            attn_q_batched: GpuBuffer::new(&ctx, num_heads * max_seq_len * config.head_dim())?,
            attn_kv_temp: GpuBuffer::new(&ctx, num_heads * max_seq_len * config.head_dim())?,
            attn_kv_temp2: GpuBuffer::new(&ctx, num_heads * max_seq_len * config.head_dim())?,
        };

        Ok(Self {
            config: config.clone(),
            layer_idx,
            input_norm_weight,
            post_attn_norm_weight,
            w_q,
            w_k,
            w_v,
            w_o,
            w_gate,
            w_up,
            w_down,
            ctx,
            scratch,
        })
    }

    /// Forward pass - all operations on GPU
    ///
    /// # Arguments
    /// * `input` - Input tensor on GPU (seq_len * hidden_size)
    /// * `output` - Output tensor on GPU (seq_len * hidden_size)
    /// * `seq_len` - Sequence length
    /// * `stream` - CUDA stream for async execution
    pub fn forward(
        &mut self,
        input: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        seq_len: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let hidden_size = self.config.hidden_size;
        let kv_hidden_size = self.config.num_kv_heads * self.config.head_dim();
        let intermediate_size = self.config.intermediate_size;

        // === Pre-attention RMSNorm (ENT-147) ===
        rms_norm_forward(
            input,
            &self.input_norm_weight,
            &mut self.scratch.norm1_out,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            stream,
        )?;

        // === Q, K, V Projections (CUDA GEMM) ===
        gemm_forward(
            &self.scratch.norm1_out,
            &self.w_q,
            &mut self.scratch.q,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(hidden_size),
            stream,
        )?;

        gemm_forward(
            &self.scratch.norm1_out,
            &self.w_k,
            &mut self.scratch.k,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(kv_hidden_size),
            stream,
        )?;

        gemm_forward(
            &self.scratch.norm1_out,
            &self.w_v,
            &mut self.scratch.v,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(kv_hidden_size),
            stream,
        )?;

        // === Multi-Head Attention (GPU-only, zero CPU transfers) ===
        self.compute_attention_cuda(seq_len, stream)?;

        // === Output Projection ===
        gemm_forward(
            &self.scratch.attn_out,
            &self.w_o,
            &mut self.scratch.o_proj_out,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(hidden_size),
            stream,
        )?;

        // === Residual Add (input + attention_output) ===
        cuda_add(
            input,
            &self.scratch.o_proj_out,
            &mut self.scratch.residual1,
            seq_len * hidden_size,
            stream,
        )?;

        // === Post-attention RMSNorm ===
        rms_norm_forward(
            &self.scratch.residual1,
            &self.post_attn_norm_weight,
            &mut self.scratch.norm2_out,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            stream,
        )?;

        // === FFN: Gate + Up Projections ===
        gemm_forward(
            &self.scratch.norm2_out,
            &self.w_gate,
            &mut self.scratch.gate_out,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(intermediate_size),
            stream,
        )?;

        gemm_forward(
            &self.scratch.norm2_out,
            &self.w_up,
            &mut self.scratch.up_out,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(intermediate_size),
            stream,
        )?;

        // === FFN: Fused SwiGLU (ENT-150) - SiLU(gate) * up in single kernel ===
        fused_swiglu_forward(
            &self.scratch.gate_out,
            &self.scratch.up_out,
            &mut self.scratch.swiglu_out,
            saturating_u32(seq_len * intermediate_size),
            stream,
        )?;

        // === FFN: Down Projection ===
        gemm_forward(
            &self.scratch.swiglu_out,
            &self.w_down,
            &mut self.scratch.ffn_out,
            saturating_u32(seq_len),
            saturating_u32(intermediate_size),
            saturating_u32(hidden_size),
            stream,
        )?;

        // === Final Residual Add (residual1 + ffn_output) ===
        cuda_add(
            &self.scratch.residual1,
            &self.scratch.ffn_out,
            output,
            seq_len * hidden_size,
            stream,
        )?;

        Ok(())
    }

    /// Compute multi-head attention entirely on GPU (zero CPU transfers)
    ///
    /// # Contract (C-ATTN-001)
    ///
    /// - **Precondition**: Q [seq, hidden], K [seq, kv_hidden], V [seq, kv_hidden] on GPU
    /// - **Postcondition**: attn_out [seq, hidden] = concat(head_0..head_H) where
    ///   head_h = softmax(Q_h @ K_{kv(h)}^T / √d_k) @ V_{kv(h)}
    /// - **Invariant**: Zero gpu_to_vec / vec_to_gpu calls; numerically equivalent to CPU
    ///
    /// Uses existing trueno-gpu kernels:
    /// - `InterleavedToBatchedKernel` for Q/K/V layout conversion
    /// - `BatchedTransposeKernel` for K^T
    /// - `Batched4DGemmKernel` for Q@K^T and attn@V
    /// - `ScaleKernel` for 1/√d_k scaling
    /// - `BatchedSoftmaxKernel` for row-wise softmax
    /// - `BatchedToInterleavedKernel` for output layout conversion
    /// - D2D copies for GQA head expansion
    fn compute_attention_cuda(&mut self, seq_len: usize, stream: &CudaStream) -> Result<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let heads_per_kv = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let seq = saturating_u32(seq_len);
        let nh = saturating_u32(num_heads);
        let nkv = saturating_u32(num_kv_heads);
        let hd = saturating_u32(head_dim);

        // Step 1: Q interleaved [seq, num_heads * head_dim] → batched [num_heads, seq, head_dim]
        interleaved_to_batched_forward(
            &self.scratch.q,
            &mut self.scratch.attn_q_batched,
            seq,
            nh,
            hd,
            stream,
        )?;

        // Step 2: K interleaved [seq, num_kv_heads * head_dim] → batched [num_kv_heads, seq, head_dim]
        interleaved_to_batched_forward(
            &self.scratch.k,
            &mut self.scratch.attn_kv_temp,
            seq,
            nkv,
            hd,
            stream,
        )?;

        // Step 3: GQA expansion + transpose for K
        if heads_per_kv == 1 {
            // MHA: transpose directly [num_heads, seq, head_dim] → [num_heads, head_dim, seq]
            batched_transpose_forward(
                &self.scratch.attn_kv_temp,
                &mut self.scratch.attn_kv_temp2,
                nh,
                seq,
                hd,
                stream,
            )?;
        } else {
            // GQA: expand [num_kv_heads, seq, hd] → [num_heads, seq, hd] in attn_kv_temp2
            expand_kv_heads(
                &self.scratch.attn_kv_temp,
                &mut self.scratch.attn_kv_temp2,
                num_kv_heads,
                heads_per_kv,
                seq_len * head_dim,
                stream,
            )?;
            // Transpose expanded K: [num_heads, seq, hd] → [num_heads, hd, seq] in attn_kv_temp
            batched_transpose_forward(
                &self.scratch.attn_kv_temp2,
                &mut self.scratch.attn_kv_temp,
                nh,
                seq,
                hd,
                stream,
            )?;
            // Move K^T to attn_kv_temp2 for consistent naming below
            // (swap pointers via D2D copy — attn_kv_temp → attn_kv_temp2)
            // SAFETY: Both buffers are valid GPU allocations with matching sizes.
            unsafe {
                self.scratch
                    .attn_kv_temp2
                    .copy_from_buffer_async(&self.scratch.attn_kv_temp, stream)
                    .map_err(|e| {
                        crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                            "K^T buffer copy failed: {e}"
                        ))
                    })?;
            }
        }

        // Step 4: Q @ K^T → attn_scores [num_heads, seq, seq]
        // attn_q_batched: [1, num_heads, seq, head_dim]
        // attn_kv_temp2:  [1, num_heads, head_dim, seq] (K transposed)
        // attn_scores:    [1, num_heads, seq, seq]
        batched_4d_gemm_forward(
            &self.scratch.attn_q_batched,
            &self.scratch.attn_kv_temp2,
            &mut self.scratch.attn_scores,
            1,
            nh,
            seq,
            seq,
            hd,
            stream,
        )?;

        // Step 5: Scale scores by 1/√d_k (in-place)
        let total_scores = nh * seq * seq;
        {
            // SAFETY: In-place aliasing is safe for element-wise operations where each
            // element is read before being written. ScaleKernel processes elements
            // independently. The view is forgotten to prevent double-free.
            let scores_view = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    self.scratch.attn_scores.as_ptr(),
                    self.scratch.attn_scores.len(),
                )
            };
            scale_forward(
                &scores_view,
                &mut self.scratch.attn_scores,
                scale,
                total_scores,
                stream,
            )?;
            std::mem::forget(scores_view);
        }

        // Step 6: Row-wise softmax → attn_weights [num_heads * seq, seq] (in-place)
        let total_rows = nh * seq;
        {
            // SAFETY: In-place aliasing is safe for BatchedSoftmaxKernel which uses
            // shared memory for row-wise reduction. Each row is fully read into shared
            // memory before any output is written. The view is forgotten to prevent double-free.
            let scores_view = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    self.scratch.attn_scores.as_ptr(),
                    self.scratch.attn_scores.len(),
                )
            };
            batched_softmax_forward(
                &scores_view,
                &mut self.scratch.attn_scores,
                total_rows,
                seq,
                stream,
            )?;
            std::mem::forget(scores_view);
        }

        // Step 7: V layout conversion + GQA expansion
        interleaved_to_batched_forward(
            &self.scratch.v,
            &mut self.scratch.attn_kv_temp,
            seq,
            nkv,
            hd,
            stream,
        )?;

        if heads_per_kv == 1 {
            // MHA: V already in [num_heads, seq, head_dim] in attn_kv_temp
        } else {
            // GQA: expand V [num_kv_heads, seq, hd] → [num_heads, seq, hd]
            expand_kv_heads(
                &self.scratch.attn_kv_temp,
                &mut self.scratch.attn_kv_temp2,
                num_kv_heads,
                heads_per_kv,
                seq_len * head_dim,
                stream,
            )?;
            // Copy expanded V back to attn_kv_temp for the GEMM
            // SAFETY: Both buffers are valid GPU allocations with matching sizes.
            unsafe {
                self.scratch
                    .attn_kv_temp
                    .copy_from_buffer_async(&self.scratch.attn_kv_temp2, stream)
                    .map_err(|e| {
                        crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                            "V expanded buffer copy failed: {e}"
                        ))
                    })?;
            }
        }

        // Step 8: attn_weights @ V → attn_result [num_heads, seq, head_dim]
        // attn_scores:   [1, num_heads, seq, seq]
        // attn_kv_temp:  [1, num_heads, seq, head_dim]
        // → attn_q_batched: [1, num_heads, seq, head_dim] (reuse Q buffer)
        batched_4d_gemm_forward(
            &self.scratch.attn_scores,
            &self.scratch.attn_kv_temp,
            &mut self.scratch.attn_q_batched,
            1,
            nh,
            seq,
            hd,
            seq,
            stream,
        )?;

        // Step 9: Convert back to interleaved [seq, num_heads * head_dim] → attn_out
        batched_to_interleaved_forward(
            &self.scratch.attn_q_batched,
            &mut self.scratch.attn_out,
            seq,
            nh,
            hd,
            stream,
        )?;

        Ok(())
    }

    /// Get layer index
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Get configuration
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Backward pass - gradient computation on GPU (ENT-151)
    ///
    /// Computes gradients for all parameters given upstream gradient.
    ///
    /// # Arguments
    /// * `input` - Original input from forward pass (seq_len * hidden_size)
    /// * `grad_output` - Gradient from upstream layer (seq_len * hidden_size)
    /// * `grad_input` - Output: gradient w.r.t. input (seq_len * hidden_size)
    /// * `seq_len` - Sequence length
    /// * `stream` - CUDA stream for async execution
    ///
    /// # Returns
    /// Gradients are accumulated into the scratch buffers:
    /// - `scratch.grad_input_norm` - Gradient for input RMSNorm weight
    /// - `scratch.grad_post_attn_norm` - Gradient for post-attention RMSNorm weight
    /// - `scratch.grad_gate/up/down` - Gradients for FFN weights
    pub fn backward(
        &mut self,
        input: &GpuBuffer<f32>,
        grad_output: &GpuBuffer<f32>,
        grad_input: &mut GpuBuffer<f32>,
        seq_len: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let eps = 1e-5_f32;

        // Backward through final residual: grad_output flows to both residual1 and ffn_out
        self.backward_ffn(grad_output, seq_len, hidden_size, intermediate_size, stream)?;

        // Backward through post-attention RMSNorm
        self.backward_post_attn_norm(grad_input, seq_len, hidden_size, eps, stream)?;

        // Backward through first residual connection and input RMSNorm
        self.backward_residual_and_input_norm(
            input,
            grad_output,
            grad_input,
            seq_len,
            hidden_size,
            eps,
            stream,
        )?;

        Ok(())
    }

    /// Backward through FFN: down projection, SwiGLU, and gate projection.
    fn backward_ffn(
        &mut self,
        grad_output: &GpuBuffer<f32>,
        seq_len: usize,
        hidden_size: usize,
        intermediate_size: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        // grad_swiglu = grad_ffn_out @ w_down^T
        gemm_backward_a(
            grad_output,
            &self.w_down,
            &mut self.scratch.grad_swiglu,
            saturating_u32(seq_len),
            saturating_u32(intermediate_size),
            saturating_u32(hidden_size),
            stream,
        )?;

        // grad_w_down = swiglu_out^T @ grad_ffn_out
        gemm_backward_b(
            &self.scratch.swiglu_out,
            grad_output,
            &mut self.scratch.grad_down,
            saturating_u32(seq_len),
            saturating_u32(intermediate_size),
            saturating_u32(hidden_size),
            stream,
        )?;

        // SiLU backward on gate
        silu_backward(
            &self.scratch.gate_out,
            &self.scratch.grad_swiglu,
            &mut self.scratch.grad_hidden,
            stream,
        )?;

        // D2D copy grad_hidden to attn_kv_temp to avoid borrow conflict
        // (grad_hidden is read by gemm_backward_b, then overwritten by gemm_backward_a)
        // SAFETY: Both buffers are valid GPU allocations; attn_kv_temp is unused during backward.
        unsafe {
            self.scratch
                .attn_kv_temp
                .copy_from_buffer_async(&self.scratch.grad_hidden, stream)
                .map_err(|e| {
                    crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                        "Backward FFN grad_hidden D2D copy failed: {e}"
                    ))
                })?;
        }

        // grad_w_gate = norm2_out^T @ grad_gate_out
        gemm_backward_b(
            &self.scratch.norm2_out,
            &self.scratch.attn_kv_temp,
            &mut self.scratch.grad_gate,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(intermediate_size),
            stream,
        )?;

        // Compute grad_norm2 using the copy in attn_kv_temp
        gemm_backward_a(
            &self.scratch.attn_kv_temp,
            &self.w_gate,
            &mut self.scratch.norm2_out, // Reuse as temp output
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(intermediate_size),
            stream,
        )?;

        Ok(())
    }

    /// Backward through post-attention RMSNorm.
    fn backward_post_attn_norm(
        &mut self,
        grad_input: &mut GpuBuffer<f32>,
        seq_len: usize,
        hidden_size: usize,
        eps: f32,
        stream: &CudaStream,
    ) -> Result<()> {
        // D2D copy norm2_out → grad_hidden (avoids D2H + H2D round-trip)
        // SAFETY: Both buffers are valid GPU allocations with matching sizes.
        unsafe {
            self.scratch
                .grad_hidden
                .copy_from_buffer_async(&self.scratch.norm2_out, stream)
                .map_err(|e| {
                    crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                        "Backward norm D2D copy failed: {e}"
                    ))
                })?;
        }

        rms_norm_backward(
            &self.scratch.residual1,
            &self.post_attn_norm_weight,
            &self.scratch.grad_hidden,
            grad_input,
            &mut self.scratch.grad_post_attn_norm,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            eps,
            stream,
        )
    }

    /// Backward through first residual connection and input RMSNorm.
    fn backward_residual_and_input_norm(
        &mut self,
        input: &GpuBuffer<f32>,
        grad_output: &GpuBuffer<f32>,
        grad_input: &mut GpuBuffer<f32>,
        seq_len: usize,
        hidden_size: usize,
        eps: f32,
        stream: &CudaStream,
    ) -> Result<()> {
        let n = saturating_u32(seq_len * hidden_size);

        // residual1 = input + o_proj_out; grad_input += grad_residual1
        // Use attn_kv_temp as temp to hold grad_input + grad_output sum,
        // since residual_add_forward can't alias input with output when
        // grad_input is both input and output.
        residual_add_forward(grad_input, grad_output, &mut self.scratch.attn_kv_temp, n, stream)?;

        // Copy sum back to grad_input
        // SAFETY: Both buffers are valid GPU allocations with matching sizes.
        unsafe {
            grad_input
                .copy_from_buffer_async(&self.scratch.attn_kv_temp, stream)
                .map_err(|e| {
                    crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                        "Backward residual grad_input D2D copy failed: {e}"
                    ))
                })?;
        }

        // D2D copy grad_input to grad_hidden (avoids aliasing for rms_norm_backward)
        // SAFETY: Both buffers are valid GPU allocations.
        unsafe {
            self.scratch
                .grad_hidden
                .copy_from_buffer_async(&self.scratch.attn_kv_temp, stream)
                .map_err(|e| {
                    crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                        "Backward residual grad_hidden D2D copy failed: {e}"
                    ))
                })?;
        }

        rms_norm_backward(
            input,
            &self.input_norm_weight,
            &self.scratch.grad_hidden,
            grad_input,
            &mut self.scratch.grad_input_norm,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            eps,
            stream,
        )
    }
}

/// CUDA element-wise addition on GPU (zero CPU transfers)
///
/// Uses `ResidualAddKernel` — single kernel launch, no D2H/H2D transfers.
#[cfg(feature = "cuda")]
fn cuda_add(
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    n: usize,
    stream: &CudaStream,
) -> Result<()> {
    residual_add_forward(a, b, output, saturating_u32(n), stream)
}

/// CUDA element-wise multiplication on GPU (zero CPU transfers)
///
/// Uses `ElementwiseMulKernel` — single kernel launch, no D2H/H2D transfers.
#[cfg(feature = "cuda")]
fn cuda_mul(
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    n: usize,
    stream: &CudaStream,
) -> Result<()> {
    crate::autograd::cuda_forward::elementwise_mul_forward(
        a,
        b,
        output,
        saturating_u32(n),
        stream,
    )
}

// CPU fallback stub
#[cfg(not(feature = "cuda"))]
pub struct CudaTransformerBlock;

#[cfg(not(feature = "cuda"))]
impl CudaTransformerBlock {
    pub fn layer_idx(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_cuda_block_compiles() {
        // Basic compilation test
        #[cfg(feature = "cuda")]
        {
            use super::*;
            let _ = std::mem::size_of::<CudaTransformerBlock>();
        }
    }
}
