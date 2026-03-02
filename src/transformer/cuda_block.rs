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
    batched_softmax_backward, gemm_backward_a, gemm_backward_b, rms_norm_backward, silu_backward,
};
#[cfg(feature = "cuda")]
use crate::autograd::cuda_optim::adamw_step_cuda;
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

/// Preallocated scratch buffers for transformer forward/backward pass (per-layer).
///
/// These are per-layer because the backward pass reads forward activations
/// stored during the forward pass. Seq_len-dependent sizes.
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
    // === Seq-dependent backward scratch (per-layer for activation reuse) ===
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
    // === Attention backward scratch (seq-dependent) ===
    /// Gradient for attention scores [num_heads * seq_len * seq_len]
    /// Kept separate from attn_scores because softmax backward reads y while writing grad_x
    grad_attn_scores: GpuBuffer<f32>,
}

/// Shared gradient workspace for weight gradients (one per model, NOT per layer).
///
/// # Contract (C-GRADWS-001)
///
/// Backward processes layers sequentially — only one layer's weight gradients
/// are computed at a time. Sharing this workspace across layers saves
/// `(L-1) * per_layer_grad_weight_elements * 4` bytes of VRAM.
///
/// For Qwen3-4B: saves 35 * 372 MB = 13.0 GB.
///
/// - **Precondition**: Allocated once before training loop starts
/// - **Postcondition**: After backward() for layer i, contains layer i's weight gradients
/// - **Invariant**: Buffer sizes match model config; never reallocated during training
#[cfg(feature = "cuda")]
pub struct CudaGradWorkspace {
    /// Gradient for input norm weight [hidden_size]
    pub(crate) grad_input_norm: GpuBuffer<f32>,
    /// Gradient for post-attention norm weight [hidden_size]
    pub(crate) grad_post_attn_norm: GpuBuffer<f32>,
    /// Gradient for FFN gate projection [hidden_size * intermediate_size]
    pub(crate) grad_gate: GpuBuffer<f32>,
    /// Gradient for FFN up projection [hidden_size * intermediate_size]
    pub(crate) grad_up: GpuBuffer<f32>,
    /// Gradient for FFN down projection [intermediate_size * hidden_size]
    pub(crate) grad_down: GpuBuffer<f32>,
    /// Gradient for Q projection weight [hidden_size * hidden_size]
    pub(crate) grad_w_q: GpuBuffer<f32>,
    /// Gradient for K projection weight [hidden_size * kv_hidden_size]
    pub(crate) grad_w_k: GpuBuffer<f32>,
    /// Gradient for V projection weight [hidden_size * kv_hidden_size]
    pub(crate) grad_w_v: GpuBuffer<f32>,
    /// Gradient for output projection weight [hidden_size * hidden_size]
    pub(crate) grad_w_o: GpuBuffer<f32>,
}

#[cfg(feature = "cuda")]
impl CudaGradWorkspace {
    /// Allocate shared gradient workspace for the given model config.
    ///
    /// Called once per training run. All 9 buffers are zero-initialized.
    pub fn new(ctx: &Arc<CudaContext>, config: &TransformerConfig) -> Result<Self> {
        let h = config.hidden_size;
        let kv = config.num_kv_heads * config.head_dim();
        let i = config.intermediate_size;

        Ok(Self {
            grad_input_norm: GpuBuffer::new(ctx, h)?,
            grad_post_attn_norm: GpuBuffer::new(ctx, h)?,
            grad_gate: GpuBuffer::new(ctx, h * i)?,
            grad_up: GpuBuffer::new(ctx, h * i)?,
            grad_down: GpuBuffer::new(ctx, i * h)?,
            grad_w_q: GpuBuffer::new(ctx, h * h)?,
            grad_w_k: GpuBuffer::new(ctx, h * kv)?,
            grad_w_v: GpuBuffer::new(ctx, h * kv)?,
            grad_w_o: GpuBuffer::new(ctx, h * h)?,
        })
    }
}

/// GPU-resident AdamW optimizer state for one transformer block.
///
/// Stores first (m) and second (v) moment estimates for all 9 weight tensors:
/// 7 matmul weights + 2 RMSNorm weights. All buffers live on GPU to avoid
/// CPU↔GPU transfers during training.
///
/// # Contract (C-OPTSTATE-001)
///
/// - **Precondition**: CUDA context valid, all buffers allocated to match weight dimensions
/// - **Postcondition**: m and v buffers initialized to zero (unbiased start)
/// - **Invariant**: Buffer sizes immutable after creation; m/v never reallocated
#[cfg(feature = "cuda")]
pub struct GpuBlockOptimizerState {
    // Attention projection optimizer states
    m_w_q: GpuBuffer<f32>,
    v_w_q: GpuBuffer<f32>,
    m_w_k: GpuBuffer<f32>,
    v_w_k: GpuBuffer<f32>,
    m_w_v: GpuBuffer<f32>,
    v_w_v: GpuBuffer<f32>,
    m_w_o: GpuBuffer<f32>,
    v_w_o: GpuBuffer<f32>,
    // FFN projection optimizer states
    m_w_gate: GpuBuffer<f32>,
    v_w_gate: GpuBuffer<f32>,
    m_w_up: GpuBuffer<f32>,
    v_w_up: GpuBuffer<f32>,
    m_w_down: GpuBuffer<f32>,
    v_w_down: GpuBuffer<f32>,
    // RMSNorm weight optimizer states
    m_input_norm: GpuBuffer<f32>,
    v_input_norm: GpuBuffer<f32>,
    m_post_attn_norm: GpuBuffer<f32>,
    v_post_attn_norm: GpuBuffer<f32>,
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
        let q_dim = config.q_dim(); // num_heads * head_dim (may differ from hidden_size)
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

        // Allocate scratch buffers — Q and attn_out need q_dim, not hidden_size
        let scratch = CudaBlockScratch {
            norm1_out: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            q: GpuBuffer::new(&ctx, max_seq_len * q_dim)?,
            k: GpuBuffer::new(&ctx, max_seq_len * kv_hidden_size)?,
            v: GpuBuffer::new(&ctx, max_seq_len * kv_hidden_size)?,
            attn_scores: GpuBuffer::new(&ctx, num_heads * max_seq_len * max_seq_len)?,
            attn_out: GpuBuffer::new(&ctx, max_seq_len * q_dim)?,
            o_proj_out: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            residual1: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            norm2_out: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            gate_out: GpuBuffer::new(&ctx, max_seq_len * intermediate_size)?,
            up_out: GpuBuffer::new(&ctx, max_seq_len * intermediate_size)?,
            swiglu_out: GpuBuffer::new(&ctx, max_seq_len * intermediate_size)?,
            ffn_out: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            // Seq-dependent backward scratch
            grad_hidden: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            grad_swiglu: GpuBuffer::new(&ctx, max_seq_len * intermediate_size)?,
            // Attention layout scratch (all sized for num_heads, handles GQA expansion)
            attn_q_batched: GpuBuffer::new(&ctx, num_heads * max_seq_len * config.head_dim())?,
            attn_kv_temp: GpuBuffer::new(&ctx, num_heads * max_seq_len * config.head_dim())?,
            attn_kv_temp2: GpuBuffer::new(&ctx, num_heads * max_seq_len * config.head_dim())?,
            // Attention backward gradient buffers (ENT-151b)
            // grad_attn_scores needs max(H*S*S, H*S*hd) for buffer reuse safety
            grad_attn_scores: GpuBuffer::new(
                &ctx,
                num_heads * max_seq_len * max_seq_len.max(config.head_dim()),
            )?,
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
        let q_dim = self.config.q_dim();
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
        // C[seq,q_dim] = A[seq,hidden] @ B[hidden,q_dim]
        gemm_forward(
            &self.scratch.norm1_out,
            &self.w_q,
            &mut self.scratch.q,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(q_dim),
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
        // C[seq,hidden] = A[seq,q_dim] @ B[q_dim,hidden]
        gemm_forward(
            &self.scratch.attn_out,
            &self.w_o,
            &mut self.scratch.o_proj_out,
            saturating_u32(seq_len),
            saturating_u32(q_dim),
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
    /// - `scratch.grad_w_q/w_k/w_v/w_o` - Gradients for attention projection weights
    pub fn backward(
        &mut self,
        input: &GpuBuffer<f32>,
        grad_output: &GpuBuffer<f32>,
        grad_input: &mut GpuBuffer<f32>,
        seq_len: usize,
        stream: &CudaStream,
        grad_ws: &mut CudaGradWorkspace,
    ) -> Result<()> {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let eps = 1e-5_f32;

        // Backward through final residual: grad_output flows to both residual1 and ffn_out
        self.backward_ffn(grad_output, seq_len, hidden_size, intermediate_size, stream, grad_ws)?;

        // Backward through post-attention RMSNorm
        self.backward_post_attn_norm(grad_input, seq_len, hidden_size, eps, stream, grad_ws)?;

        // Backward through attention: output projection, attention weights, Q/K/V projections
        // (ENT-151b: previously missing — attention params received no gradients)
        self.backward_attention(grad_input, seq_len, stream, grad_ws)?;

        // Backward through first residual connection and input RMSNorm
        self.backward_residual_and_input_norm(
            input,
            grad_output,
            grad_input,
            seq_len,
            hidden_size,
            eps,
            stream,
            grad_ws,
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
        grad_ws: &mut CudaGradWorkspace,
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
            &mut grad_ws.grad_down,
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
            &mut grad_ws.grad_gate,
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
        grad_ws: &mut CudaGradWorkspace,
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
            &mut grad_ws.grad_post_attn_norm,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            eps,
            stream,
        )
    }

    /// Backward through multi-head attention (ENT-151b)
    ///
    /// Reverses the forward attention pipeline:
    /// output_proj → layout → attn_weights@V → softmax → scale → Q@K^T → layout → Q/K/V proj
    ///
    /// # Contract (C-ATTN-BACK-001)
    ///
    /// - **Precondition**: grad_input contains gradient from post-attention norm backward,
    ///   scratch.{q, k, v, attn_scores, attn_out, norm1_out} contain forward pass values
    /// - **Postcondition**: grad_hidden contains gradient w.r.t. norm1_out (input to Q/K/V proj),
    ///   grad_w_{q,k,v,o} contain weight gradients for attention projections
    /// - **Invariant**: Zero CPU-side data transfers; all operations on GPU
    fn backward_attention(
        &mut self,
        grad_input: &mut GpuBuffer<f32>,
        seq_len: usize,
        stream: &CudaStream,
        grad_ws: &mut CudaGradWorkspace,
    ) -> Result<()> {
        let hidden_size = self.config.hidden_size;
        let kv_hidden_size = self.config.num_kv_heads * self.config.head_dim();
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let heads_per_kv = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let seq = saturating_u32(seq_len);
        let nh = saturating_u32(num_heads);
        let nkv = saturating_u32(num_kv_heads);
        let hd = saturating_u32(head_dim);

        // === Step 4.1: Output projection backward ===
        // grad_input currently holds gradient from post_attn_norm backward.
        // grad_attn_out = grad_input @ w_o^T → grad_hidden
        gemm_backward_a(
            grad_input,
            &self.w_o,
            &mut self.scratch.grad_hidden,
            seq,
            saturating_u32(hidden_size),
            saturating_u32(hidden_size),
            stream,
        )?;

        // grad_w_o = attn_out^T @ grad_input
        gemm_backward_b(
            &self.scratch.attn_out,
            grad_input,
            &mut grad_ws.grad_w_o,
            seq,
            saturating_u32(hidden_size),
            saturating_u32(hidden_size),
            stream,
        )?;

        // === Step 4.2: Layout conversion ===
        // grad_attn_out [seq, hidden] → grad_attn_batched [num_heads, seq, head_dim]
        // Reuse attn_q_batched for grad_attn_batched
        interleaved_to_batched_forward(
            &self.scratch.grad_hidden,
            &mut self.scratch.attn_q_batched,
            seq,
            nh,
            hd,
            stream,
        )?;

        // === Step 4.3: Backward through attn_weights @ V ===
        // Forward was: attn_result = attn_weights @ V_batched
        // Reconstruct V_batched from preserved v
        interleaved_to_batched_forward(
            &self.scratch.v,
            &mut self.scratch.attn_kv_temp,
            seq,
            nkv,
            hd,
            stream,
        )?;

        // GQA expand V if needed
        if heads_per_kv > 1 {
            expand_kv_heads(
                &self.scratch.attn_kv_temp,
                &mut self.scratch.attn_kv_temp2,
                num_kv_heads,
                heads_per_kv,
                seq_len * head_dim,
                stream,
            )?;
            // SAFETY: Both buffers are valid GPU allocations with matching sizes.
            unsafe {
                self.scratch
                    .attn_kv_temp
                    .copy_from_buffer_async(&self.scratch.attn_kv_temp2, stream)
                    .map_err(|e| {
                        crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                            "Attn backward V expand D2D copy failed: {e}"
                        ))
                    })?;
            }
        }
        // attn_kv_temp now has V_batched [num_heads, seq, head_dim]

        // Transpose V: [num_heads, seq, head_dim] → [num_heads, head_dim, seq]
        batched_transpose_forward(
            &self.scratch.attn_kv_temp,
            &mut self.scratch.attn_kv_temp2,
            nh,
            seq,
            hd,
            stream,
        )?;
        // attn_kv_temp2 = V^T [num_heads, head_dim, seq]

        // grad_attn_weights = grad_attn_batched @ V^T → grad_attn_scores [H, seq, seq]
        batched_4d_gemm_forward(
            &self.scratch.attn_q_batched,
            &self.scratch.attn_kv_temp2,
            &mut self.scratch.grad_attn_scores,
            1,
            nh,
            seq,
            seq,
            hd,
            stream,
        )?;

        // grad_V = attn_weights^T @ grad_attn_batched → attn_kv_temp [H, seq, hd]
        // First transpose attn_weights (attn_scores): [H, seq, seq] → [H, seq, seq] (symmetric dims)
        // Actually we need attn_scores^T which is just transpose of [H, seq, seq]
        batched_transpose_forward(
            &self.scratch.attn_scores,
            &mut self.scratch.attn_kv_temp2,
            nh,
            seq,
            seq,
            stream,
        )?;
        // attn_kv_temp2 = attn_weights^T [H, seq, seq]

        batched_4d_gemm_forward(
            &self.scratch.attn_kv_temp2,
            &self.scratch.attn_q_batched,
            &mut self.scratch.attn_kv_temp,
            1,
            nh,
            seq,
            hd,
            seq,
            stream,
        )?;
        // attn_kv_temp = grad_V [num_heads, seq, head_dim]

        // === Step 4.4: Softmax backward ===
        // attn_scores contains softmax output from forward pass
        // In-place: grad_attn_scores is both input (grad_output) and output (grad_input)
        // This is safe because the kernel reads all elements in pass 1 before writing in pass 2.
        let total_rows = nh * seq;
        {
            // SAFETY: In-place aliasing is safe for BatchedSoftmaxBackwardKernel which uses
            // a two-pass approach: pass 1 reads all y[i]*gy[i] to compute dot product,
            // pass 2 writes grad_x[i]. The view is forgotten to prevent double-free.
            let grad_scores_view = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    self.scratch.grad_attn_scores.as_ptr(),
                    self.scratch.grad_attn_scores.len(),
                )
            };
            batched_softmax_backward(
                &self.scratch.attn_scores,
                &grad_scores_view,
                &mut self.scratch.grad_attn_scores,
                total_rows,
                seq,
                stream,
            )?;
            std::mem::forget(grad_scores_view);
        }
        // grad_attn_scores now contains gradient through softmax

        // === Step 4.5: Scale backward ===
        // Forward scaled by 1/√d_k, backward is same scale (linear operation)
        let total_scores = nh * seq * seq;
        {
            // SAFETY: In-place aliasing safe for element-wise scale (independent elements).
            let scores_view = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    self.scratch.grad_attn_scores.as_ptr(),
                    self.scratch.grad_attn_scores.len(),
                )
            };
            scale_forward(
                &scores_view,
                &mut self.scratch.grad_attn_scores,
                scale,
                total_scores,
                stream,
            )?;
            std::mem::forget(scores_view);
        }

        // === Step 4.6: Backward through Q @ K^T ===
        // Forward was: scores = Q_batched @ K^T
        // Reconstruct K_batched and expand for GQA
        interleaved_to_batched_forward(
            &self.scratch.k,
            &mut self.scratch.attn_kv_temp2,
            seq,
            nkv,
            hd,
            stream,
        )?;

        if heads_per_kv > 1 {
            // SAFETY: Both buffers are valid GPU allocations; attn_q_batched is about to be
            // overwritten anyway.
            unsafe {
                self.scratch
                    .attn_q_batched
                    .copy_from_buffer_async(&self.scratch.attn_kv_temp2, stream)
                    .map_err(|e| {
                        crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                            "Attn backward K copy for GQA expand failed: {e}"
                        ))
                    })?;
            }
            expand_kv_heads(
                &self.scratch.attn_q_batched,
                &mut self.scratch.attn_kv_temp2,
                num_kv_heads,
                heads_per_kv,
                seq_len * head_dim,
                stream,
            )?;
        }
        // attn_kv_temp2 = K_expanded [num_heads, seq, head_dim]

        // grad_Q = grad_scores @ K_expanded → attn_q_batched [H, seq, hd]
        batched_4d_gemm_forward(
            &self.scratch.grad_attn_scores,
            &self.scratch.attn_kv_temp2,
            &mut self.scratch.attn_q_batched,
            1,
            nh,
            seq,
            hd,
            seq,
            stream,
        )?;

        // grad_K^T = Q^T @ grad_scores
        // First reconstruct Q_batched from preserved q
        // Reconstruct Q_batched into o_proj_out (attn_q_batched already overwritten by grad_Q).
        interleaved_to_batched_forward(
            &self.scratch.q,
            &mut self.scratch.o_proj_out, // temp buffer for Q_batched
            seq,
            nh,
            hd,
            stream,
        )?;

        // Transpose Q: [H, seq, hd] → [H, hd, seq]
        batched_transpose_forward(
            &self.scratch.o_proj_out,
            &mut self.scratch.attn_kv_temp2, // reuse for Q^T
            nh,
            seq,
            hd,
            stream,
        )?;

        // grad_K^T = Q^T @ grad_scores → ffn_out as temp [H, hd, seq]
        batched_4d_gemm_forward(
            &self.scratch.attn_kv_temp2,
            &self.scratch.grad_attn_scores,
            &mut self.scratch.ffn_out, // reuse as temp for grad_K^T [H, hd, seq]
            1,
            nh,
            hd,
            seq,
            seq,
            stream,
        )?;

        // Transpose grad_K^T → grad_K: [H, hd, seq] → [H, seq, hd]
        batched_transpose_forward(
            &self.scratch.ffn_out,
            &mut self.scratch.attn_kv_temp2, // grad_K [H, seq, hd]
            nh,
            hd,
            seq,
            stream,
        )?;

        // === Step 4.7: GQA gradient reduction ===
        // grad_K and grad_V are in [num_heads, seq, hd], need to reduce to [num_kv_heads, seq, hd]
        if heads_per_kv > 1 {
            self.reduce_gqa_gradients(num_kv_heads, heads_per_kv, seq_len, head_dim, stream)?;
        }

        // === Step 4.8: Convert gradients back to interleaved layout ===
        // grad_Q: attn_q_batched [H, seq, hd] → o_proj_out [seq, hidden] (interleaved)
        batched_to_interleaved_forward(
            &self.scratch.attn_q_batched,
            &mut self.scratch.o_proj_out,
            seq,
            nh,
            hd,
            stream,
        )?;

        // grad_K: attn_kv_temp2 [nkv, seq, hd] → norm2_out [seq, kv_hidden] (interleaved)
        batched_to_interleaved_forward(
            &self.scratch.attn_kv_temp2,
            &mut self.scratch.norm2_out,
            seq,
            nkv,
            hd,
            stream,
        )?;

        // grad_V: attn_kv_temp [nkv, seq, hd] → ffn_out [seq, kv_hidden] (interleaved)
        batched_to_interleaved_forward(
            &self.scratch.attn_kv_temp,
            &mut self.scratch.ffn_out,
            seq,
            nkv,
            hd,
            stream,
        )?;

        // === Step 4.9: Q/K/V projection backward ===
        // grad_norm1 = grad_q @ w_q^T → grad_hidden
        gemm_backward_a(
            &self.scratch.o_proj_out, // grad_q interleaved
            &self.w_q,
            &mut self.scratch.grad_hidden,
            seq,
            saturating_u32(hidden_size),
            saturating_u32(hidden_size),
            stream,
        )?;

        // grad_norm1 += grad_k @ w_k^T
        // GEMM result stored in grad_attn_scores scratch space
        gemm_backward_a(
            &self.scratch.norm2_out, // grad_k interleaved
            &self.w_k,
            &mut self.scratch.grad_attn_scores, // temp for grad_k @ w_k^T
            seq,
            saturating_u32(hidden_size),
            saturating_u32(kv_hidden_size),
            stream,
        )?;
        // grad_hidden += grad_attn_scores (through o_proj_out scratch — same hidden_size)
        let n_hidden = saturating_u32(seq_len * hidden_size);
        residual_add_forward(
            &self.scratch.grad_hidden,
            &self.scratch.grad_attn_scores,
            &mut self.scratch.o_proj_out, // temp output (hidden_size, matches grad_hidden)
            n_hidden,
            stream,
        )?;
        // SAFETY: Both buffers are valid GPU allocations with matching sizes.
        unsafe {
            self.scratch
                .grad_hidden
                .copy_from_buffer_async(&self.scratch.o_proj_out, stream)
                .map_err(|e| {
                    crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                        "Attn backward grad_hidden accumulate K D2D copy failed: {e}"
                    ))
                })?;
        }

        // grad_norm1 += grad_v @ w_v^T
        gemm_backward_a(
            &self.scratch.ffn_out, // grad_v interleaved
            &self.w_v,
            &mut self.scratch.grad_attn_scores, // temp for grad_v @ w_v^T
            seq,
            saturating_u32(hidden_size),
            saturating_u32(kv_hidden_size),
            stream,
        )?;
        // grad_hidden += grad_attn_scores (through o_proj_out scratch)
        residual_add_forward(
            &self.scratch.grad_hidden,
            &self.scratch.grad_attn_scores,
            &mut self.scratch.o_proj_out, // temp output (hidden_size)
            n_hidden,
            stream,
        )?;
        // SAFETY: Both buffers are valid GPU allocations with matching sizes.
        unsafe {
            self.scratch
                .grad_hidden
                .copy_from_buffer_async(&self.scratch.o_proj_out, stream)
                .map_err(|e| {
                    crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                        "Attn backward grad_hidden accumulate V D2D copy failed: {e}"
                    ))
                })?;
        }

        // Weight gradients: grad_w_q = norm1_out^T @ grad_q
        gemm_backward_b(
            &self.scratch.norm1_out,
            &self.scratch.o_proj_out, // grad_q
            &mut grad_ws.grad_w_q,
            seq,
            saturating_u32(hidden_size),
            saturating_u32(hidden_size),
            stream,
        )?;

        // grad_w_k = norm1_out^T @ grad_k
        gemm_backward_b(
            &self.scratch.norm1_out,
            &self.scratch.norm2_out, // grad_k
            &mut grad_ws.grad_w_k,
            seq,
            saturating_u32(hidden_size),
            saturating_u32(kv_hidden_size),
            stream,
        )?;

        // grad_w_v = norm1_out^T @ grad_v
        gemm_backward_b(
            &self.scratch.norm1_out,
            &self.scratch.ffn_out, // grad_v
            &mut grad_ws.grad_w_v,
            seq,
            saturating_u32(hidden_size),
            saturating_u32(kv_hidden_size),
            stream,
        )?;

        // Copy grad_hidden → grad_input for downstream (residual backward)
        // SAFETY: Both buffers are valid GPU allocations with matching sizes.
        unsafe {
            grad_input
                .copy_from_buffer_async(&self.scratch.grad_hidden, stream)
                .map_err(|e| {
                    crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                        "Attn backward grad_hidden → grad_input D2D copy failed: {e}"
                    ))
                })?;
        }

        Ok(())
    }

    /// Reduce GQA head gradients from [num_heads] to [num_kv_heads] by summing groups.
    ///
    /// Reads grad_K from `attn_kv_temp2` and grad_V from `attn_kv_temp` (both [H, seq, hd]).
    /// Writes reduced grad_K to `attn_kv_temp2` and reduced grad_V to `attn_kv_temp`
    /// (both [nkv, seq, hd]).
    ///
    /// Uses `grad_attn_scores`, `ffn_out`, `o_proj_out`, `grad_hidden` as scratch.
    fn reduce_gqa_gradients(
        &mut self,
        num_kv_heads: usize,
        heads_per_kv: usize,
        seq_len: usize,
        head_dim: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let elems_per_head = seq_len * head_dim;

        // Reduce grad_K: attn_kv_temp2 [H] → grad_attn_scores [nkv]
        self.reduce_single_gqa_gradient(
            true, num_kv_heads, heads_per_kv, elems_per_head, stream,
        )?;

        // Reduce grad_V: attn_kv_temp [H] → ffn_out [nkv]
        self.reduce_single_gqa_gradient(
            false, num_kv_heads, heads_per_kv, elems_per_head, stream,
        )?;

        // Copy reduced results to known locations for step 4.8
        let kv_elems = num_kv_heads * elems_per_head;
        // SAFETY: Valid GPU allocations with sufficient size.
        unsafe {
            self.scratch
                .attn_kv_temp2
                .copy_from_buffer_at_async(&self.scratch.grad_attn_scores, 0, 0, kv_elems, stream)
                .map_err(|e| {
                    crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                        "GQA grad_K reduced final copy failed: {e}"
                    ))
                })?;
            self.scratch
                .attn_kv_temp
                .copy_from_buffer_at_async(&self.scratch.ffn_out, 0, 0, kv_elems, stream)
                .map_err(|e| {
                    crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                        "GQA grad_V reduced final copy failed: {e}"
                    ))
                })?;
        }
        Ok(())
    }

    /// Reduce one gradient tensor from [num_heads] to [num_kv_heads] by summing groups.
    ///
    /// When `is_k=true`: reads from `attn_kv_temp2`, writes to `grad_attn_scores`.
    /// When `is_k=false`: reads from `attn_kv_temp`, writes to `ffn_out`.
    /// Uses `o_proj_out` and `grad_hidden` as scratch.
    fn reduce_single_gqa_gradient(
        &mut self,
        is_k: bool,
        num_kv_heads: usize,
        heads_per_kv: usize,
        elems_per_head: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let label = if is_k { "K" } else { "V" };

        for kv_h in 0..num_kv_heads {
            let dst_offset = kv_h * elems_per_head;
            let first_h = kv_h * heads_per_kv;
            let src_offset = first_h * elems_per_head;

            // Copy first head of group as base
            // SAFETY: All offsets are within buffer bounds.
            unsafe {
                let (dst, src) = if is_k {
                    (&mut self.scratch.grad_attn_scores, &self.scratch.attn_kv_temp2)
                } else {
                    (&mut self.scratch.ffn_out, &self.scratch.attn_kv_temp)
                };
                dst.copy_from_buffer_at_async(src, dst_offset, src_offset, elems_per_head, stream)
                    .map_err(|e| {
                        crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                            "GQA grad_{label} reduce base copy failed: {e}"
                        ))
                    })?;
            }

            // Add remaining heads in group
            for rep in 1..heads_per_kv {
                let h = kv_h * heads_per_kv + rep;
                let h_offset = h * elems_per_head;

                // Head extraction into o_proj_out buffer
                // SAFETY: Valid GPU allocations with sufficient size.
                unsafe {
                    let src = if is_k {
                        &self.scratch.attn_kv_temp2
                    } else {
                        &self.scratch.attn_kv_temp
                    };
                    self.scratch
                        .o_proj_out
                        .copy_from_buffer_at_async(src, 0, h_offset, elems_per_head, stream)
                        .map_err(|e| {
                            crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                                "GQA grad_{label} reduce head copy failed: {e}"
                            ))
                        })?;
                }

                // Add: dst[dst_offset..] += o_proj_out[0..elems_per_head]
                // SAFETY: Creating non-owning views for arithmetic; forgotten to prevent double-free.
                unsafe {
                    let dst_buf = if is_k {
                        &self.scratch.grad_attn_scores
                    } else {
                        &self.scratch.ffn_out
                    };
                    let dst_view = GpuBuffer::<f32>::from_raw_parts(
                        dst_buf.as_ptr() + (dst_offset as u64 * 4),
                        elems_per_head,
                    );
                    let src_view = GpuBuffer::<f32>::from_raw_parts(
                        self.scratch.o_proj_out.as_ptr(),
                        elems_per_head,
                    );
                    let mut sum_view = GpuBuffer::<f32>::from_raw_parts(
                        self.scratch.grad_hidden.as_ptr(),
                        elems_per_head,
                    );
                    residual_add_forward(
                        &dst_view,
                        &src_view,
                        &mut sum_view,
                        saturating_u32(elems_per_head),
                        stream,
                    )?;
                    // Copy sum back to dst at dst_offset
                    let dst_buf = if is_k {
                        &mut self.scratch.grad_attn_scores
                    } else {
                        &mut self.scratch.ffn_out
                    };
                    dst_buf
                        .copy_from_buffer_at_async(
                            &self.scratch.grad_hidden, dst_offset, 0, elems_per_head, stream,
                        )
                        .map_err(|e| {
                            crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                                "GQA grad_{label} reduce sum copy failed: {e}"
                            ))
                        })?;
                    std::mem::forget(dst_view);
                    std::mem::forget(src_view);
                    std::mem::forget(sum_view);
                }
            }
        }
        Ok(())
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
        grad_ws: &mut CudaGradWorkspace,
    ) -> Result<()> {
        let n = saturating_u32(seq_len * hidden_size);

        // residual1 = input + o_proj_out; grad_input += grad_residual1
        // attn_kv_temp holds grad_input + grad_output sum (avoids alias
        // between input and output in residual_add_forward).
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
            &mut grad_ws.grad_input_norm,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            eps,
            stream,
        )
    }

    /// Initialize GPU-resident AdamW optimizer state for all block weights.
    ///
    /// Allocates zero-initialized first and second moment buffers for each of the
    /// 9 weight tensors (4 attention projections + 3 FFN projections + 2 RMSNorm).
    ///
    /// # Contract (C-OPTINIT-001)
    ///
    /// - **Precondition**: CUDA context is valid, sufficient GPU memory available
    /// - **Postcondition**: All m/v buffers are zero-initialized with dimensions
    ///   matching the corresponding weight tensors
    /// - **Invariant**: Total GPU memory for optimizer state = 2 × sum(weight_sizes) × 4 bytes
    pub fn init_optimizer_state(&self) -> Result<GpuBlockOptimizerState> {
        let hidden = self.config.hidden_size;
        let kv_hidden = self.config.num_kv_heads * self.config.head_dim();
        let intermediate = self.config.intermediate_size;

        Ok(GpuBlockOptimizerState {
            m_w_q: GpuBuffer::new(&self.ctx, hidden * hidden)?,
            v_w_q: GpuBuffer::new(&self.ctx, hidden * hidden)?,
            m_w_k: GpuBuffer::new(&self.ctx, hidden * kv_hidden)?,
            v_w_k: GpuBuffer::new(&self.ctx, hidden * kv_hidden)?,
            m_w_v: GpuBuffer::new(&self.ctx, hidden * kv_hidden)?,
            v_w_v: GpuBuffer::new(&self.ctx, hidden * kv_hidden)?,
            m_w_o: GpuBuffer::new(&self.ctx, hidden * hidden)?,
            v_w_o: GpuBuffer::new(&self.ctx, hidden * hidden)?,
            m_w_gate: GpuBuffer::new(&self.ctx, hidden * intermediate)?,
            v_w_gate: GpuBuffer::new(&self.ctx, hidden * intermediate)?,
            m_w_up: GpuBuffer::new(&self.ctx, hidden * intermediate)?,
            v_w_up: GpuBuffer::new(&self.ctx, hidden * intermediate)?,
            m_w_down: GpuBuffer::new(&self.ctx, intermediate * hidden)?,
            v_w_down: GpuBuffer::new(&self.ctx, intermediate * hidden)?,
            m_input_norm: GpuBuffer::new(&self.ctx, hidden)?,
            v_input_norm: GpuBuffer::new(&self.ctx, hidden)?,
            m_post_attn_norm: GpuBuffer::new(&self.ctx, hidden)?,
            v_post_attn_norm: GpuBuffer::new(&self.ctx, hidden)?,
        })
    }

    /// Run GPU-resident AdamW optimizer step on all block weights.
    ///
    /// Updates weights in-place using gradients computed by `backward()`.
    /// All operations run on GPU — zero CPU↔GPU data transfers.
    ///
    /// # Contract (C-OPTSTEP-001)
    ///
    /// - **Precondition**: `backward()` completed for this block (scratch grad buffers valid),
    ///   `state` initialized via `init_optimizer_state()`, `step > 0`
    /// - **Postcondition**: All 9 weight tensors updated by AdamW rule,
    ///   m/v states updated with current gradient statistics
    /// - **Invariant**: Weight dimensions unchanged; no GPU memory allocated or freed
    pub fn optimizer_step(
        &mut self,
        state: &mut GpuBlockOptimizerState,
        step: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        stream: &CudaStream,
        grad_ws: &CudaGradWorkspace,
    ) -> Result<()> {
        debug_assert!(step > 0, "C-OPTSTEP-001: step must be > 0 for bias correction");

        // Pre-capture lengths to avoid borrow conflicts (len is immutable borrow,
        // adamw_step_cuda takes mutable borrow on same buffer)
        let n_wq = self.w_q.len() as u32;
        let n_wk = self.w_k.len() as u32;
        let n_wv = self.w_v.len() as u32;
        let n_wo = self.w_o.len() as u32;
        let n_gate = self.w_gate.len() as u32;
        let n_up = self.w_up.len() as u32;
        let n_down = self.w_down.len() as u32;
        let n_inorm = self.input_norm_weight.len() as u32;
        let n_panorm = self.post_attn_norm_weight.len() as u32;

        // Attention projection weights
        adamw_step_cuda(
            &mut self.w_q, &grad_ws.grad_w_q,
            &mut state.m_w_q, &mut state.v_w_q,
            lr, beta1, beta2, eps, weight_decay, step, n_wq, stream,
        )?;
        adamw_step_cuda(
            &mut self.w_k, &grad_ws.grad_w_k,
            &mut state.m_w_k, &mut state.v_w_k,
            lr, beta1, beta2, eps, weight_decay, step, n_wk, stream,
        )?;
        adamw_step_cuda(
            &mut self.w_v, &grad_ws.grad_w_v,
            &mut state.m_w_v, &mut state.v_w_v,
            lr, beta1, beta2, eps, weight_decay, step, n_wv, stream,
        )?;
        adamw_step_cuda(
            &mut self.w_o, &grad_ws.grad_w_o,
            &mut state.m_w_o, &mut state.v_w_o,
            lr, beta1, beta2, eps, weight_decay, step, n_wo, stream,
        )?;

        // FFN projection weights
        adamw_step_cuda(
            &mut self.w_gate, &grad_ws.grad_gate,
            &mut state.m_w_gate, &mut state.v_w_gate,
            lr, beta1, beta2, eps, weight_decay, step, n_gate, stream,
        )?;
        adamw_step_cuda(
            &mut self.w_up, &grad_ws.grad_up,
            &mut state.m_w_up, &mut state.v_w_up,
            lr, beta1, beta2, eps, weight_decay, step, n_up, stream,
        )?;
        adamw_step_cuda(
            &mut self.w_down, &grad_ws.grad_down,
            &mut state.m_w_down, &mut state.v_w_down,
            lr, beta1, beta2, eps, weight_decay, step, n_down, stream,
        )?;

        // RMSNorm weights
        adamw_step_cuda(
            &mut self.input_norm_weight, &grad_ws.grad_input_norm,
            &mut state.m_input_norm, &mut state.v_input_norm,
            lr, beta1, beta2, eps, weight_decay, step, n_inorm, stream,
        )?;
        adamw_step_cuda(
            &mut self.post_attn_norm_weight, &grad_ws.grad_post_attn_norm,
            &mut state.m_post_attn_norm, &mut state.v_post_attn_norm,
            lr, beta1, beta2, eps, weight_decay, step, n_panorm, stream,
        )?;

        Ok(())
    }

    /// Download all weight data from GPU to host vectors.
    ///
    /// Used to synchronize GPU-updated weights back to CPU model for checkpointing.
    ///
    /// # Contract (C-DLWEIGHTS-001)
    ///
    /// - **Precondition**: Block weights are valid GPU allocations
    /// - **Postcondition**: Returned vectors have exact same length and content as GPU buffers
    /// - **Invariant**: GPU buffers are not modified
    pub fn download_weights(&self) -> Result<BlockWeights> {
        let download = |buf: &GpuBuffer<f32>| -> Result<Vec<f32>> {
            let mut host = vec![0.0f32; buf.len()];
            buf.copy_to_host(&mut host).map_err(|e| {
                crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                    "Weight download failed: {e}"
                ))
            })?;
            Ok(host)
        };

        Ok(BlockWeights {
            w_q: download(&self.w_q)?,
            w_k: download(&self.w_k)?,
            w_v: download(&self.w_v)?,
            w_o: download(&self.w_o)?,
            w_gate: download(&self.w_gate)?,
            w_up: download(&self.w_up)?,
            w_down: download(&self.w_down)?,
            input_norm_weight: download(&self.input_norm_weight)?,
            post_attn_norm_weight: download(&self.post_attn_norm_weight)?,
        })
    }
}

/// Downloaded weight data from a CUDA transformer block.
///
/// # Contract (C-BLOCKWT-001)
///
/// - **Invariant**: Vector lengths match original weight dimensions
#[cfg(feature = "cuda")]
pub struct BlockWeights {
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
    pub input_norm_weight: Vec<f32>,
    pub post_attn_norm_weight: Vec<f32>,
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

// =============================================================================
// CudaBlock — enum dispatching fp32 or NF4 transformer blocks
// =============================================================================

/// Unified enum for CUDA transformer blocks (fp32 or NF4-quantized).
///
/// The classify pipeline stores `Vec<CudaBlock>` and calls `forward()` without
/// caring which quantization format the frozen weights use.
#[cfg(feature = "cuda")]
pub enum CudaBlock {
    /// Standard fp32 weights (full precision, ~16 GB for Qwen3-4B)
    Fp32(CudaTransformerBlock),
    /// NF4 quantized weights (~2 GB for Qwen3-4B, ~8x compression)
    Nf4(CudaNf4TransformerBlock),
}

#[cfg(feature = "cuda")]
impl CudaBlock {
    /// Forward pass through the transformer block.
    ///
    /// Dispatches to the underlying fp32 or NF4 implementation.
    pub fn forward(
        &mut self,
        input: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        seq_len: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        match self {
            CudaBlock::Fp32(b) => b.forward(input, output, seq_len, stream),
            CudaBlock::Nf4(b) => b.forward(input, output, seq_len, stream),
        }
    }

    /// Get the layer index of this block.
    pub fn layer_idx(&self) -> usize {
        match self {
            CudaBlock::Fp32(b) => b.layer_idx(),
            CudaBlock::Nf4(b) => b.layer_idx,
        }
    }

    /// Backward pass (only supported for fp32 blocks).
    ///
    /// NF4 blocks are frozen — backward is never called when `quantize_nf4` is active
    /// because `gpu_training` is set to `None`.
    pub fn backward(
        &mut self,
        input: &GpuBuffer<f32>,
        grad_output: &GpuBuffer<f32>,
        grad_input: &mut GpuBuffer<f32>,
        seq_len: usize,
        stream: &CudaStream,
        grad_ws: &mut CudaGradWorkspace,
    ) -> Result<()> {
        match self {
            CudaBlock::Fp32(b) => b.backward(input, grad_output, grad_input, seq_len, stream, grad_ws),
            CudaBlock::Nf4(_) => Err(crate::autograd::cuda_tensor::CudaTensorError::KernelError("backward not supported on NF4 blocks (frozen weights)".into())),
        }
    }

    /// Initialize optimizer state (only supported for fp32 blocks).
    pub fn init_optimizer_state(&self) -> Result<GpuBlockOptimizerState> {
        match self {
            CudaBlock::Fp32(b) => b.init_optimizer_state(),
            CudaBlock::Nf4(_) => Err(crate::autograd::cuda_tensor::CudaTensorError::KernelError("init_optimizer_state not supported on NF4 blocks".into())),
        }
    }

    /// Download weights from GPU (only supported for fp32 blocks).
    pub fn download_weights(&self) -> Result<BlockWeights> {
        match self {
            CudaBlock::Fp32(b) => b.download_weights(),
            CudaBlock::Nf4(_) => Err(crate::autograd::cuda_tensor::CudaTensorError::KernelError("download_weights not supported on NF4 blocks".into())),
        }
    }

    /// Optimizer step (only supported for fp32 blocks).
    pub fn optimizer_step(
        &mut self,
        state: &mut GpuBlockOptimizerState,
        step: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        stream: &CudaStream,
        grad_ws: &CudaGradWorkspace,
    ) -> Result<()> {
        match self {
            CudaBlock::Fp32(b) => b.optimizer_step(state, step, lr, beta1, beta2, eps, weight_decay, stream, grad_ws),
            CudaBlock::Nf4(_) => Err(crate::autograd::cuda_tensor::CudaTensorError::KernelError("optimizer_step not supported on NF4 blocks (frozen weights)".into())),
        }
    }
}

/// CPU fallback stub for CudaBlock.
#[cfg(not(feature = "cuda"))]
pub enum CudaBlock {
    Fp32(CudaTransformerBlock),
}

// =============================================================================
// NF4 Quantized Transformer Block (trueno#108: QLoRA support)
// =============================================================================

/// CUDA-accelerated transformer block with NF4-quantized frozen weights.
///
/// Stores the 7 projection weights as packed NF4 (4-bit) + per-block scales instead
/// of fp32, achieving ~8x compression. Norm weights remain fp32 (negligible size).
///
/// # VRAM Savings (Qwen3-4B example)
///
/// | Component | fp32 | NF4 |
/// |-----------|------|-----|
/// | Frozen weights (36L × 7 projections) | 16.0 GB | 2.1 GB |
///
/// # Forward Only
///
/// NF4 blocks are frozen — no backward pass needed. LoRA adapters (fp32) handle
/// the trainable parameters separately. The forward pass uses fused dequant+GEMM
/// kernels that read NF4 directly without materializing fp32 weights.
#[cfg(feature = "cuda")]
pub struct CudaNf4TransformerBlock {
    config: TransformerConfig,
    layer_idx: usize,
    // Norm weights stay fp32 (tiny: 2 × hidden_size floats)
    input_norm_weight: GpuBuffer<f32>,
    post_attn_norm_weight: GpuBuffer<f32>,
    // Projection weights: NF4 quantized (packed data + per-block scales)
    w_q_nf4: GpuBuffer<u8>,
    w_q_scales: GpuBuffer<f32>,
    w_k_nf4: GpuBuffer<u8>,
    w_k_scales: GpuBuffer<f32>,
    w_v_nf4: GpuBuffer<u8>,
    w_v_scales: GpuBuffer<f32>,
    w_o_nf4: GpuBuffer<u8>,
    w_o_scales: GpuBuffer<f32>,
    w_gate_nf4: GpuBuffer<u8>,
    w_gate_scales: GpuBuffer<f32>,
    w_up_nf4: GpuBuffer<u8>,
    w_up_scales: GpuBuffer<f32>,
    w_down_nf4: GpuBuffer<u8>,
    w_down_scales: GpuBuffer<f32>,
    ctx: Arc<CudaContext>,
    scratch: CudaBlockScratch,
}

#[cfg(feature = "cuda")]
impl CudaNf4TransformerBlock {
    /// Create a new NF4 transformer block from fp32 CPU tensors.
    ///
    /// Quantizes all 7 projection weights to NF4 on CPU, then uploads the packed
    /// data and scales to GPU. Norm weights are uploaded as fp32.
    #[allow(clippy::too_many_arguments)]
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
        use trueno_gpu::kernels::{quantize_nf4, NF4_BLOCK_SIZE};

        let hidden_size = config.hidden_size;
        let q_dim = config.q_dim(); // num_heads * head_dim (may differ from hidden_size)
        let kv_hidden_size = config.num_kv_heads * config.head_dim();
        let intermediate_size = config.intermediate_size;
        let num_heads = config.num_attention_heads;

        // ── C-NF4SHAPE-001: Weight shape contracts ──────────────────────
        // Ground truth: PMAT-331 validation in attention.rs from_pretrained()
        //   Q: [q_dim, hidden], K: [kv_hidden, hidden], V: [kv_hidden, hidden], O: [hidden, q_dim]
        //   gate: [intermediate, hidden], up: [intermediate, hidden], down: [hidden, intermediate]
        assert_eq!(w_q.len(), q_dim * hidden_size,
            "C-NF4SHAPE-001: w_q expected {}, got {} (q_dim={q_dim}, hidden={hidden_size})",
            q_dim * hidden_size, w_q.len());
        assert_eq!(w_k.len(), kv_hidden_size * hidden_size,
            "C-NF4SHAPE-001: w_k expected {}, got {}", kv_hidden_size * hidden_size, w_k.len());
        assert_eq!(w_v.len(), kv_hidden_size * hidden_size,
            "C-NF4SHAPE-001: w_v expected {}, got {}", kv_hidden_size * hidden_size, w_v.len());
        assert_eq!(w_o.len(), hidden_size * q_dim,
            "C-NF4SHAPE-001: w_o expected {}, got {}", hidden_size * q_dim, w_o.len());
        assert_eq!(w_gate.len(), intermediate_size * hidden_size,
            "C-NF4SHAPE-001: w_gate expected {}, got {}", intermediate_size * hidden_size, w_gate.len());
        assert_eq!(w_up.len(), intermediate_size * hidden_size,
            "C-NF4SHAPE-001: w_up expected {}, got {}", intermediate_size * hidden_size, w_up.len());
        assert_eq!(w_down.len(), hidden_size * intermediate_size,
            "C-NF4SHAPE-001: w_down expected {}, got {}", hidden_size * intermediate_size, w_down.len());

        // Upload norm weights as fp32
        let input_norm_weight = GpuBuffer::from_host(&ctx, input_norm_weight)?;
        let post_attn_norm_weight = GpuBuffer::from_host(&ctx, post_attn_norm_weight)?;

        // Helper: quantize fp32 weight to NF4, upload packed data + scales to GPU
        let quantize_and_upload =
            |weights: &[f32], total: usize| -> Result<(GpuBuffer<u8>, GpuBuffer<f32>)> {
                assert_eq!(weights.len(), total, "weight length mismatch");
                assert!(
                    total % NF4_BLOCK_SIZE == 0,
                    "weight count {total} not divisible by NF4 block size {NF4_BLOCK_SIZE}"
                );

                // NF4 quantization operates on flat buffer — rows/cols only matter for
                // block alignment. Use (total/NF4_BLOCK_SIZE, NF4_BLOCK_SIZE) to ensure
                // every block is full.
                let q = quantize_nf4(weights, total / NF4_BLOCK_SIZE, NF4_BLOCK_SIZE);
                let nf4_buf = GpuBuffer::from_host(&ctx, &q.data)?;
                let scales_buf = GpuBuffer::from_host(&ctx, &q.scales)?;
                Ok((nf4_buf, scales_buf))
            };

        // Quantize all 7 projection weights (shape contracts already verified above)
        let (w_q_nf4, w_q_scales) = quantize_and_upload(w_q, q_dim * hidden_size)?;
        let (w_k_nf4, w_k_scales) = quantize_and_upload(w_k, kv_hidden_size * hidden_size)?;
        let (w_v_nf4, w_v_scales) = quantize_and_upload(w_v, kv_hidden_size * hidden_size)?;
        let (w_o_nf4, w_o_scales) = quantize_and_upload(w_o, hidden_size * q_dim)?;
        let (w_gate_nf4, w_gate_scales) =
            quantize_and_upload(w_gate, intermediate_size * hidden_size)?;
        let (w_up_nf4, w_up_scales) =
            quantize_and_upload(w_up, intermediate_size * hidden_size)?;
        let (w_down_nf4, w_down_scales) =
            quantize_and_upload(w_down, hidden_size * intermediate_size)?;

        // Allocate scratch buffers — Q and attn_out need q_dim, not hidden_size
        // C-NF4SCRATCH-001: q_dim = num_heads * head_dim (may differ from hidden_size)
        let scratch = CudaBlockScratch {
            norm1_out: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            q: GpuBuffer::new(&ctx, max_seq_len * q_dim)?,
            k: GpuBuffer::new(&ctx, max_seq_len * kv_hidden_size)?,
            v: GpuBuffer::new(&ctx, max_seq_len * kv_hidden_size)?,
            attn_scores: GpuBuffer::new(&ctx, num_heads * max_seq_len * max_seq_len)?,
            attn_out: GpuBuffer::new(&ctx, max_seq_len * q_dim)?,
            o_proj_out: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            residual1: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            norm2_out: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            gate_out: GpuBuffer::new(&ctx, max_seq_len * intermediate_size)?,
            up_out: GpuBuffer::new(&ctx, max_seq_len * intermediate_size)?,
            swiglu_out: GpuBuffer::new(&ctx, max_seq_len * intermediate_size)?,
            ffn_out: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            grad_hidden: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            grad_swiglu: GpuBuffer::new(&ctx, max_seq_len * intermediate_size)?,
            attn_q_batched: GpuBuffer::new(&ctx, num_heads * max_seq_len * config.head_dim())?,
            attn_kv_temp: GpuBuffer::new(&ctx, num_heads * max_seq_len * config.head_dim())?,
            attn_kv_temp2: GpuBuffer::new(&ctx, num_heads * max_seq_len * config.head_dim())?,
            grad_attn_scores: GpuBuffer::new(
                &ctx,
                num_heads * max_seq_len * max_seq_len.max(config.head_dim()),
            )?,
        };

        Ok(Self {
            config: config.clone(),
            layer_idx,
            input_norm_weight,
            post_attn_norm_weight,
            w_q_nf4,
            w_q_scales,
            w_k_nf4,
            w_k_scales,
            w_v_nf4,
            w_v_scales,
            w_o_nf4,
            w_o_scales,
            w_gate_nf4,
            w_gate_scales,
            w_up_nf4,
            w_up_scales,
            w_down_nf4,
            w_down_scales,
            ctx,
            scratch,
        })
    }

    /// Forward pass using NF4 fused dequant+GEMM kernels.
    ///
    /// Identical pipeline to `CudaTransformerBlock::forward()` but all 7 GEMM
    /// calls use `gemm_nf4_forward()` which reads NF4 weights directly.
    pub fn forward(
        &mut self,
        input: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        seq_len: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        use crate::autograd::cuda_forward::gemm_nf4_forward;

        let hidden_size = self.config.hidden_size;
        let q_dim = self.config.q_dim();
        let kv_hidden_size = self.config.num_kv_heads * self.config.head_dim();
        let intermediate_size = self.config.intermediate_size;

        // === Pre-attention RMSNorm ===
        rms_norm_forward(
            input,
            &self.input_norm_weight,
            &mut self.scratch.norm1_out,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            stream,
        )?;

        // === Q, K, V Projections (NF4 fused dequant + GEMM) ===
        // C-NF4GEMM-001: Q proj is C[seq,q_dim] = A[seq,hidden] @ B[hidden,q_dim]
        gemm_nf4_forward(
            &self.scratch.norm1_out,
            &self.w_q_nf4,
            &self.w_q_scales,
            &mut self.scratch.q,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(q_dim),
            stream,
        )?;

        gemm_nf4_forward(
            &self.scratch.norm1_out,
            &self.w_k_nf4,
            &self.w_k_scales,
            &mut self.scratch.k,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(kv_hidden_size),
            stream,
        )?;

        gemm_nf4_forward(
            &self.scratch.norm1_out,
            &self.w_v_nf4,
            &self.w_v_scales,
            &mut self.scratch.v,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(kv_hidden_size),
            stream,
        )?;

        // === Multi-Head Attention (GPU-only, zero CPU transfers) ===
        // Reuses the same attention pipeline as fp32 (activations are fp32)
        self.compute_attention_cuda(seq_len, stream)?;

        // === Output Projection ===
        // C-NF4GEMM-002: O proj is C[seq,hidden] = A[seq,q_dim] @ B[q_dim,hidden]
        gemm_nf4_forward(
            &self.scratch.attn_out,
            &self.w_o_nf4,
            &self.w_o_scales,
            &mut self.scratch.o_proj_out,
            saturating_u32(seq_len),
            saturating_u32(q_dim),
            saturating_u32(hidden_size),
            stream,
        )?;

        // === Residual Add ===
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

        // === FFN: Gate + Up Projections (NF4) ===
        gemm_nf4_forward(
            &self.scratch.norm2_out,
            &self.w_gate_nf4,
            &self.w_gate_scales,
            &mut self.scratch.gate_out,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(intermediate_size),
            stream,
        )?;

        gemm_nf4_forward(
            &self.scratch.norm2_out,
            &self.w_up_nf4,
            &self.w_up_scales,
            &mut self.scratch.up_out,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(intermediate_size),
            stream,
        )?;

        // === FFN: Fused SwiGLU ===
        fused_swiglu_forward(
            &self.scratch.gate_out,
            &self.scratch.up_out,
            &mut self.scratch.swiglu_out,
            saturating_u32(seq_len * intermediate_size),
            stream,
        )?;

        // === FFN: Down Projection (NF4) ===
        gemm_nf4_forward(
            &self.scratch.swiglu_out,
            &self.w_down_nf4,
            &self.w_down_scales,
            &mut self.scratch.ffn_out,
            saturating_u32(seq_len),
            saturating_u32(intermediate_size),
            saturating_u32(hidden_size),
            stream,
        )?;

        // === Final Residual Add ===
        cuda_add(
            &self.scratch.residual1,
            &self.scratch.ffn_out,
            output,
            seq_len * hidden_size,
            stream,
        )?;

        Ok(())
    }

    /// Layer index accessor.
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }
}

/// Helper: delegate attention computation to the shared implementation.
///
/// `CudaNf4TransformerBlock` reuses the same attention pipeline as the fp32 block
/// since attention operates on fp32 activations (Q/K/V are already dequantized by GEMM).
#[cfg(feature = "cuda")]
impl CudaNf4TransformerBlock {
    fn compute_attention_cuda(&mut self, seq_len: usize, stream: &CudaStream) -> Result<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let heads_per_kv = num_heads / num_kv_heads;

        let s = saturating_u32(seq_len);
        let nh = saturating_u32(num_heads);
        let nkv = saturating_u32(num_kv_heads);
        let hd = saturating_u32(head_dim);

        // Q: interleaved → batched layout
        interleaved_to_batched_forward(
            &self.scratch.q,
            &mut self.scratch.attn_q_batched,
            s, nh, hd,
            stream,
        )?;

        // K: interleaved → batched, then GQA expand if needed
        interleaved_to_batched_forward(
            &self.scratch.k,
            &mut self.scratch.attn_kv_temp,
            s, nkv, hd,
            stream,
        )?;

        if heads_per_kv > 1 {
            expand_kv_heads(
                &self.scratch.attn_kv_temp,
                &mut self.scratch.attn_kv_temp2,
                num_kv_heads, heads_per_kv,
                seq_len * head_dim,
                stream,
            )?;
        } else {
            // SAFETY: D2D copy with matching buffer sizes
            unsafe {
                self.scratch.attn_kv_temp2
                    .copy_from_buffer_async(&self.scratch.attn_kv_temp, stream)
                    .map_err(|e| {
                        crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(
                            format!("K copy failed: {e:?}"),
                        )
                    })?;
            }
        }

        // K^T: transpose for attention scores
        batched_transpose_forward(
            &self.scratch.attn_kv_temp2,
            &mut self.scratch.attn_kv_temp,
            nh, s, hd,
            stream,
        )?;

        // Q @ K^T → attention scores
        batched_4d_gemm_forward(
            &self.scratch.attn_q_batched,
            &self.scratch.attn_kv_temp,
            &mut self.scratch.attn_scores,
            1, nh, s, s, hd,
            stream,
        )?;

        // Scale by 1/sqrt(head_dim)
        let scale_factor = 1.0 / (head_dim as f32).sqrt();
        let total_scores = num_heads * seq_len * seq_len;
        let scores_view = unsafe {
            GpuBuffer::<f32>::from_raw_parts(
                self.scratch.attn_scores.as_ptr(),
                self.scratch.attn_scores.len(),
            )
        };
        scale_forward(
            &scores_view,
            &mut self.scratch.attn_scores,
            scale_factor,
            saturating_u32(total_scores),
            stream,
        )?;
        std::mem::forget(scores_view);

        // Softmax (in-place: input aliased with output via unsafe view)
        // SAFETY: The softmax kernel reads each row completely into shared memory / registers
        // before writing output. The view is forgotten to prevent double-free.
        let scores_view = unsafe {
            GpuBuffer::<f32>::from_raw_parts(
                self.scratch.attn_scores.as_ptr(),
                self.scratch.attn_scores.len(),
            )
        };
        batched_softmax_forward(
            &scores_view,
            &mut self.scratch.attn_scores,
            saturating_u32(num_heads * seq_len),
            s,
            stream,
        )?;
        std::mem::forget(scores_view);

        // V: interleaved → batched, then GQA expand
        interleaved_to_batched_forward(
            &self.scratch.v,
            &mut self.scratch.attn_kv_temp,
            s, nkv, hd,
            stream,
        )?;

        if heads_per_kv > 1 {
            expand_kv_heads(
                &self.scratch.attn_kv_temp,
                &mut self.scratch.attn_kv_temp2,
                num_kv_heads, heads_per_kv,
                seq_len * head_dim,
                stream,
            )?;
        } else {
            unsafe {
                self.scratch.attn_kv_temp2
                    .copy_from_buffer_async(&self.scratch.attn_kv_temp, stream)
                    .map_err(|e| {
                        crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(
                            format!("V copy failed: {e:?}"),
                        )
                    })?;
            }
        }

        // attn_scores @ V → attention output
        batched_4d_gemm_forward(
            &self.scratch.attn_scores,
            &self.scratch.attn_kv_temp2,
            &mut self.scratch.attn_q_batched,
            1, nh, s, hd, s,
            stream,
        )?;

        // Batched → interleaved layout
        batched_to_interleaved_forward(
            &self.scratch.attn_q_batched,
            &mut self.scratch.attn_out,
            s, nh, hd,
            stream,
        )?;

        Ok(())
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
            let _ = std::mem::size_of::<CudaNf4TransformerBlock>();
        }
    }
}
