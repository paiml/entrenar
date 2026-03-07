//! GPU-resident transformer trainer (ALB-040)
//!
//! Wires the existing `CudaTransformerBlock` forward/backward/optimizer_step
//! into the pretraining path. Follows the proven `classify_pipeline.rs` pattern.
//!
//! # Architecture
//!
//! ```text
//! CudaTransformerTrainer
//! ├── model: Transformer                 (CPU — embed + save)
//! ├── cuda_trainer: CudaTrainer          (GPU device context)
//! ├── cuda_blocks: Vec<CudaTransformerBlock>
//! ├── cuda_grad_workspace: CudaGradWorkspace
//! ├── gpu_training: GpuPretrainState     (layer_inputs, grad bufs, opt states)
//! ├── lm_head_weight_gpu: GpuBuffer      (V × H on GPU)
//! ├── lm_head_grad_gpu: GpuBuffer        (V × H gradient scratch)
//! ├── lm_head_m/v: GpuBuffer             (AdamW moment states)
//! └── config: TransformerTrainConfig
//! ```
//!
//! # Transfer budget (C-GPUTRAIN-002, updated KAIZEN-050/052)
//!
//! 1 PCIe transfer per training step (+ tiny control transfers):
//! 1. H2D: hidden states after embedding (seq×H×4 bytes)
//! 2. H2D: target_ids for fused cross-entropy (seq×4 bytes — ~512B)
//! 3. D2H: loss_partials from fused cross-entropy (seq×4 bytes — ~512B)
//!
//! Eliminated by KAIZEN-050:
//! - D2H logits (was seq×V×4 = 77.8MB for Qwen3-4B)
//! - H2D grad_logits (was seq×V×4 = 77.8MB)
//!
//! Eliminated by KAIZEN-052:
//! - grad_gpu buffer allocation (was seq×V×4 = 77.8MB per step)

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaStream, GpuBuffer};

#[cfg(feature = "cuda")]
use crate::autograd::cuda_backward::{gemm_backward_a, gemm_backward_b, rms_norm_backward};
#[cfg(feature = "cuda")]
use crate::autograd::cuda_forward::{gemm_forward, pre_warm_forward_kernels, rms_norm_forward};
#[cfg(feature = "cuda")]
use crate::autograd::cuda_optim::{
    adamw_step_cuda, clip_scale_reduce_cuda, fused_cross_entropy_cuda, gradient_clip_cuda,
    gradient_clip_gpu_scale_cuda, squared_sum_collect, squared_sum_cuda, squared_sum_launch_cuda,
    squared_sum_launch_into, FusedClipState,
};
#[cfg(feature = "cuda")]
use crate::autograd::cuda_training::{cuda_training_available, CudaTrainer};
#[cfg(feature = "cuda")]
use crate::autograd::precision::GradScaler;
#[cfg(feature = "cuda")]
use crate::autograd::Tensor;
#[cfg(feature = "cuda")]
use crate::io::{save_model, Model, ModelFormat, ModelMetadata, SaveConfig};
#[cfg(feature = "cuda")]
use crate::optim::{AdamW, Optimizer};
#[cfg(feature = "cuda")]
use crate::train::MetricsTracker;
#[cfg(feature = "cuda")]
use crate::transformer::{
    BlockWeights, CudaGradWorkspace, CudaTransformerBlock, GpuBlockOptimizerState, Transformer,
};

#[cfg(feature = "cuda")]
use super::batch::LMBatch;
#[cfg(feature = "cuda")]
use super::config::TransformerTrainConfig;
#[cfg(feature = "cuda")]
use super::step_profiler::StepProfiler;

/// Compute gradient L2 norm of the shared workspace via GPU reduction (KAIZEN-054).
///
/// Uses `squared_sum_cuda` per buffer (~1KB D2H each) instead of downloading entire
/// gradient buffers to CPU (was 58 MB+ per block, disabled in ALB-067).
///
/// Free function to avoid borrow conflicts with `&mut self`.
#[cfg(feature = "cuda")]
fn compute_workspace_clip_scale_gpu(
    ws: &CudaGradWorkspace,
    max_norm: f32,
    stream: &CudaStream,
) -> (f32, f32) {
    use crate::autograd::cuda_optim::PendingSquaredSum;

    let all_bufs: [&GpuBuffer<f32>; 9] = [
        &ws.grad_w_q,
        &ws.grad_w_k,
        &ws.grad_w_v,
        &ws.grad_w_o,
        &ws.grad_gate,
        &ws.grad_up,
        &ws.grad_down,
        &ws.grad_input_norm,
        &ws.grad_post_attn_norm,
    ];

    // KAIZEN-055: Launch all 9 squared_sum kernels back-to-back without syncing.
    // Single sync after all launches — reduces 9 pipeline flushes to 1 per block.
    let mut pending: Vec<PendingSquaredSum> = Vec::with_capacity(9);
    for buf in &all_bufs {
        let n = buf.len() as u32;
        if n == 0 {
            continue;
        }
        match squared_sum_launch_cuda(buf, n, stream) {
            Ok(p) => pending.push(p),
            Err(_) => {} // Skip buffer on error (e.g., if kernel cache not init)
        }
    }

    // Single sync point for all 9 kernel launches.
    if let Err(_) = stream.synchronize() {
        return (1.0, 0.0);
    }

    // Collect results: download partial sums (~1KB each) and combine.
    let mut total_sq = 0.0f64;
    for p in &pending {
        if let Ok(norm) = squared_sum_collect(p) {
            total_sq += f64::from(norm) * f64::from(norm);
        }
    }

    let grad_norm = total_sq.sqrt() as f32;
    let scale = if grad_norm > max_norm { max_norm / grad_norm } else { 1.0 };
    (scale, grad_norm)
}

/// Clip all gradient buffers in the shared workspace using GPU-computed L2 norm (KAIZEN-054).
///
/// R-004: Returns pre-clip gradient L2 norm for observability logging.
#[cfg(feature = "cuda")]
fn clip_workspace_gradients(ws: &mut CudaGradWorkspace, max_norm: f32, stream: &CudaStream) -> f32 {
    let (scale, grad_norm) = compute_workspace_clip_scale_gpu(ws, max_norm, stream);
    if (scale - 1.0).abs() < 1e-7 {
        return grad_norm;
    }

    let n_wq = ws.grad_w_q.len() as u32;
    let n_wk = ws.grad_w_k.len() as u32;
    let n_wv = ws.grad_w_v.len() as u32;
    let n_wo = ws.grad_w_o.len() as u32;
    let n_gate = ws.grad_gate.len() as u32;
    let n_up = ws.grad_up.len() as u32;
    let n_down = ws.grad_down.len() as u32;
    let n_inorm = ws.grad_input_norm.len() as u32;
    let n_panorm = ws.grad_post_attn_norm.len() as u32;

    let _ = gradient_clip_cuda(&mut ws.grad_w_q, scale, n_wq, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_w_k, scale, n_wk, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_w_v, scale, n_wv, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_w_o, scale, n_wo, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_gate, scale, n_gate, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_up, scale, n_up, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_down, scale, n_down, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_input_norm, scale, n_inorm, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_post_attn_norm, scale, n_panorm, stream);
    grad_norm
}

/// ALB-078: Fused gradient clipping — entire pipeline stays on GPU.
///
/// Replaces `clip_workspace_gradients` by eliminating the stream.synchronize()
/// and D2H partial-sum download. All computation happens on GPU:
///
/// 1. 9× SquaredSumKernel → write partials to pre-allocated contiguous buffer
/// 2. 1× ClipScaleReduceKernel → reduce partials, compute scale on GPU
/// 3. 9× GradientClipGpuScaleKernel → read scale from GPU, apply to gradients
///
/// Zero sync points, zero D2H transfers per block.
#[cfg(feature = "cuda")]
fn fused_clip_workspace_gradients(
    ws: &mut CudaGradWorkspace,
    max_norm: f32,
    state: &FusedClipState,
    stream: &CudaStream,
) {
    let all_bufs: [&GpuBuffer<f32>; 9] = [
        &ws.grad_w_q,
        &ws.grad_w_k,
        &ws.grad_w_v,
        &ws.grad_w_o,
        &ws.grad_gate,
        &ws.grad_up,
        &ws.grad_down,
        &ws.grad_input_norm,
        &ws.grad_post_attn_norm,
    ];

    // Phase 1: Launch 9 squared_sum kernels into contiguous partials buffer.
    // Each writes to state.partials_buf at its pre-computed offset.
    for (i, buf) in all_bufs.iter().enumerate() {
        let n = buf.len() as u32;
        if n == 0 {
            continue;
        }
        let output_ptr = state.partials_buf.as_ptr() + u64::from(state.offsets[i]) * 4;
        let _ = squared_sum_launch_into(buf, n, output_ptr, stream);
    }

    // Phase 2: Reduce all partials and compute clip_scale on GPU.
    // Stream ordering guarantees all squared_sum kernels complete before this runs.
    let _ = clip_scale_reduce_cuda(
        &state.partials_buf,
        state.total_partials,
        max_norm,
        &state.scale_buf,
        stream,
    );

    // Phase 3: Apply clip scale to all 9 gradient buffers.
    // Scale is read from GPU memory — no D2H needed.
    let scale_ptr = state.scale_buf.as_ptr(); // output[0] = clip_scale
    let mut all_bufs_mut: [&mut GpuBuffer<f32>; 9] = [
        &mut ws.grad_w_q,
        &mut ws.grad_w_k,
        &mut ws.grad_w_v,
        &mut ws.grad_w_o,
        &mut ws.grad_gate,
        &mut ws.grad_up,
        &mut ws.grad_down,
        &mut ws.grad_input_norm,
        &mut ws.grad_post_attn_norm,
    ];
    for buf in &mut all_bufs_mut {
        let n = buf.len() as u32;
        if n == 0 {
            continue;
        }
        let _ = gradient_clip_gpu_scale_cuda(buf, scale_ptr, n, stream);
    }
}

/// R-004: Compute gradient L2 norm without clipping (for observability only).
///
/// Uses GPU reduction (KAIZEN-054). Only ~9KB D2H per call.
#[cfg(feature = "cuda")]
fn compute_workspace_grad_norm(ws: &CudaGradWorkspace, stream: &CudaStream) -> f32 {
    let (_, norm) = compute_workspace_clip_scale_gpu(ws, f32::MAX, stream);
    norm
}

/// ALB-072: Unscale all gradient buffers in the shared workspace by `inv_scale`.
///
/// In fp16 AMP, the fused cross-entropy kernel multiplies loss_scale into the
/// gradient output. All subsequent backward gradients carry this scaling. The
/// GPU block optimizer (AdamW) must receive unscaled gradients — otherwise the
/// second moment `v` overflows f32, producing NaN in early layers.
///
/// This is the GPU-side equivalent of `GradScaler::unscale_and_check()` used
/// for CPU embedding gradients.
#[cfg(feature = "cuda")]
fn unscale_workspace_gradients(ws: &mut CudaGradWorkspace, inv_scale: f32, stream: &CudaStream) {
    if (inv_scale - 1.0).abs() < 1e-7 {
        return;
    }

    let n_wq = ws.grad_w_q.len() as u32;
    let n_wk = ws.grad_w_k.len() as u32;
    let n_wv = ws.grad_w_v.len() as u32;
    let n_wo = ws.grad_w_o.len() as u32;
    let n_gate = ws.grad_gate.len() as u32;
    let n_up = ws.grad_up.len() as u32;
    let n_down = ws.grad_down.len() as u32;
    let n_inorm = ws.grad_input_norm.len() as u32;
    let n_panorm = ws.grad_post_attn_norm.len() as u32;

    let _ = gradient_clip_cuda(&mut ws.grad_w_q, inv_scale, n_wq, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_w_k, inv_scale, n_wk, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_w_v, inv_scale, n_wv, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_w_o, inv_scale, n_wo, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_gate, inv_scale, n_gate, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_up, inv_scale, n_up, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_down, inv_scale, n_down, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_input_norm, inv_scale, n_inorm, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_post_attn_norm, inv_scale, n_panorm, stream);
}

/// GPU-resident training state for pretraining.
///
/// # Contract (C-GPUTRAIN-001)
///
/// - `layer_inputs.len() == num_layers`
/// - All buffers preallocated at init; zero GPU allocations during training
/// - `step` increments monotonically
#[cfg(feature = "cuda")]
struct GpuPretrainState {
    /// Saved layer inputs for backward [num_layers][seq_len * hidden_size]
    layer_inputs: Vec<GpuBuffer<f32>>,
    /// Which layer inputs were saved during forward (activation checkpointing).
    /// When checkpointing is enabled, only checkpoint boundary layers are saved.
    /// Non-saved layers are recomputed from the nearest checkpoint before backward.
    saved_layer_mask: Vec<bool>,
    /// Temporary buffer for activation recomputation [seq_len * hidden_size].
    /// Used as the initial input when recomputing from a checkpoint boundary.
    /// Only allocated when activation checkpointing is enabled.
    recompute_buf: Option<GpuBuffer<f32>>,
    /// Final RMSNorm weight on GPU [hidden_size]
    final_norm_weight: GpuBuffer<f32>,
    /// Final block output (pre-norm) for RMSNorm backward [seq_len * hidden_size]
    blocks_output: GpuBuffer<f32>,
    /// Alternating gradient buffer A [seq_len * hidden_size]
    grad_buf_a: GpuBuffer<f32>,
    /// Alternating gradient buffer B [seq_len * hidden_size]
    grad_buf_b: GpuBuffer<f32>,
    /// Gradient for final norm weight [hidden_size]
    grad_final_norm_weight: GpuBuffer<f32>,
    /// RMSNorm output buffer (reused each step) [seq_len * hidden_size]
    norm_output: GpuBuffer<f32>,
    /// Logits buffer (reused each step) [seq_len * vocab_size]
    logits_buf: GpuBuffer<f32>,
    /// LM head gradient buffer [seq_len * hidden_size] (grad w.r.t. normed hidden)
    lm_head_grad_hidden: GpuBuffer<f32>,
    /// Per-block optimizer states
    optimizer_states: Vec<GpuBlockOptimizerState>,
    /// Optimizer step counter
    step: u32,
}

/// GPU-resident transformer trainer for pretraining.
///
/// Uses `CudaTransformerBlock` forward/backward/optimizer_step on GPU,
/// keeping only embedding lookup and cross-entropy loss on CPU.
///
/// # Contract (C-GPUTRAIN-002)
///
/// - Exactly 3 PCIe transfers per training step
/// - Graceful fallback to CPU `TransformerTrainer` on any CUDA failure
/// - Weight sync via `sync_weights_to_cpu()` before save
#[cfg(feature = "cuda")]
pub struct CudaTransformerTrainer {
    /// CPU model (for embedding, saving, fallback)
    model: Transformer,
    /// CUDA device context
    cuda_trainer: CudaTrainer,
    /// GPU-resident transformer blocks
    cuda_blocks: Vec<CudaTransformerBlock>,
    /// Shared gradient workspace (one set, reused across layers)
    cuda_grad_workspace: CudaGradWorkspace,
    /// GPU training state (layer inputs, grad bufs, optimizer states)
    gpu_training: GpuPretrainState,
    /// LM head weight on GPU [vocab_size * hidden_size]
    lm_head_weight_gpu: GpuBuffer<f32>,
    /// LM head weight gradient on GPU [vocab_size * hidden_size]
    lm_head_grad_gpu: GpuBuffer<f32>,
    /// LM head AdamW first moment [vocab_size * hidden_size]
    lm_head_m: GpuBuffer<f32>,
    /// LM head AdamW second moment [vocab_size * hidden_size]
    lm_head_v: GpuBuffer<f32>,
    /// Final norm weight AdamW first moment [hidden_size]
    final_norm_m: GpuBuffer<f32>,
    /// Final norm weight AdamW second moment [hidden_size]
    final_norm_v: GpuBuffer<f32>,
    /// CPU optimizer for embedding weights only
    embed_optimizer: AdamW,
    /// Training configuration
    config: TransformerTrainConfig,
    /// Metrics tracker
    pub metrics: MetricsTracker,
    /// Current optimizer step
    step: usize,
    /// Accumulated loss (for gradient accumulation)
    accumulated_loss: f32,
    /// Accumulated batch count
    accumulated_batches: usize,
    /// R-004: Last observed LM head gradient L2 norm (proxy for global grad norm)
    last_grad_norm: f32,
    /// R-040: Last observed embedding activation gradient L2 norm
    last_embed_grad_norm: f32,
    /// R-038: Per-block gradient accumulation for true multi-step gradient accumulation.
    /// Only allocated when accumulation_steps > 1. CPU-side buffers (~335 MB for 350M).
    grad_accum: Option<super::grad_accumulator::PerBlockGradientAccumulator>,
    /// ALB-091: GPU-resident gradient accumulation (replaces CPU accum when available).
    /// Eliminates 24 × ga stream.synchronize() + D2H transfers per optimizer step.
    gpu_grad_accum: Option<super::gpu_grad_accumulator::GpuGradientAccumulator>,
    /// R-002: Gradient scaler for mixed-precision training.
    /// For BF16: no-op (scale=1.0, dynamic=false).
    /// For FP16: dynamic loss scaling to prevent gradient underflow.
    grad_scaler: GradScaler,
    /// KAIZEN-047: Per-step wall-clock profiler.
    /// Reports timing breakdown for each training phase.
    profiler: StepProfiler,
    /// KAIZEN-053: Pre-allocated forward scratch buffers [max_seq_len * hidden_size].
    /// Reused every step — eliminates 2 × cuMemAlloc/Free per training step.
    fwd_scratch_a: GpuBuffer<f32>,
    fwd_scratch_b: GpuBuffer<f32>,
    /// KAIZEN-056: Pre-allocated CPU staging buffer for H2D hidden state upload.
    /// Eliminates vec![0.0; max_seq_len * hidden_size] allocation per step.
    h2d_staging: Vec<f32>,
    /// KAIZEN-059: Pre-allocated CPU staging buffer for D2H gradient downloads
    /// during gradient accumulation. Sized to max(h*intermediate, vocab*h).
    /// Eliminates ~15GB of per-step heap churn (36 × vec![0.0; h*i] + vec![0.0; vocab*h]
    /// per micro-batch × accumulation_steps).
    d2h_staging: Vec<f32>,
    /// ALB-078: Pre-allocated state for fused gradient clipping pipeline.
    /// Eliminates 24 stream.synchronize() calls per step.
    fused_clip: Option<FusedClipState>,
}

#[cfg(feature = "cuda")]
impl CudaTransformerTrainer {
    /// Create a new GPU-resident trainer.
    ///
    /// # Errors
    ///
    /// Returns `Err` if CUDA initialization, kernel pre-warming, or block upload fails.
    /// Caller should fall back to CPU `TransformerTrainer` on error.
    pub fn new(config: TransformerTrainConfig) -> crate::Result<Self> {
        let model = Transformer::new(&config.model_config);
        Self::with_model(model, config)
    }

    /// ALB-089: Load SafeTensors checkpoint for GPU inference (forward-only).
    ///
    /// Creates a `CudaTransformerTrainer` in inference mode. The optimizer
    /// state is allocated (wasteful but simple), but `forward_logits()` only
    /// uses the forward path. Call `forward_logits(&tokens)` to generate.
    ///
    /// # Arguments
    /// * `checkpoint_dir` - Directory containing model.safetensors + config.json
    /// * `model_config` - Transformer architecture config
    ///
    /// # Errors
    ///
    /// Returns `Err` if SafeTensors loading or CUDA initialization fails.
    pub fn for_inference(
        checkpoint_dir: impl AsRef<std::path::Path>,
        model_config: crate::transformer::TransformerConfig,
    ) -> crate::Result<Self> {
        let model = Transformer::from_safetensors(checkpoint_dir.as_ref(), &model_config)?;
        let mut config = TransformerTrainConfig::new(model_config);
        config.max_seq_len = config.model_config.max_position_embeddings;
        Self::with_model(model, config)
    }

    /// Create a GPU-resident trainer from an existing model.
    ///
    /// # Errors
    ///
    /// Returns `Err` if CUDA initialization fails.
    pub fn with_model(model: Transformer, config: TransformerTrainConfig) -> crate::Result<Self> {
        if !cuda_training_available() {
            return Err(crate::error::Error::ConfigError("CUDA not available".into()));
        }

        let mc = &config.model_config;
        let max_seq_len = config.max_seq_len;
        let hidden_size = mc.hidden_size;
        let vocab_size = mc.vocab_size;
        let num_layers = mc.num_hidden_layers;

        // Step 1: Create CUDA trainer (initializes kernel caches)
        let cuda_trainer = CudaTrainer::new().map_err(|e| {
            crate::error::Error::ConfigError(format!("CUDA trainer init failed: {e:?}"))
        })?;

        println!(
            "  GPU: {} ({:.1} GB)",
            cuda_trainer.device_name(),
            cuda_trainer.total_memory() as f64 / 1e9
        );

        let ctx = cuda_trainer.context().clone();
        let stream = cuda_trainer.stream();

        // Step 2: Pre-warm forward kernels (C-PREWARM-001)
        // Must happen before block upload — JIT compilation needs free VRAM
        pre_warm_forward_kernels(
            hidden_size,
            mc.intermediate_size,
            mc.num_attention_heads,
            mc.num_kv_heads,
            mc.head_dim(),
            max_seq_len,
        )
        .map_err(|e| crate::error::Error::ConfigError(format!("Kernel pre-warm failed: {e:?}")))?;

        // Step 2b: Bind cuBLAS handles to training stream (ALB-075)
        // Must happen after kernel cache init, before any GEMM calls.
        if let Err(e) = crate::autograd::cuda_forward::set_forward_cublas_stream(stream) {
            println!("[WARN] cuBLAS forward stream bind failed: {e:?} — falling back to PTX");
        }
        if let Err(e) = crate::autograd::cuda_backward::set_backward_cublas_stream(stream) {
            println!("[WARN] cuBLAS backward stream bind failed: {e:?} — falling back to PTX");
        }

        // Step 3: Upload transformer blocks to GPU
        let mut cuda_blocks = Vec::with_capacity(num_layers);
        for (i, layer) in model.layers.iter().enumerate() {
            let block = CudaTransformerBlock::new(
                mc,
                i,
                ctx.clone(),
                layer.input_norm.weight.data().as_slice().expect("contiguous"),
                layer.post_attn_norm.weight.data().as_slice().expect("contiguous"),
                layer.self_attn.w_q.data().as_slice().expect("contiguous"),
                layer.self_attn.w_k.data().as_slice().expect("contiguous"),
                layer.self_attn.w_v.data().as_slice().expect("contiguous"),
                layer.self_attn.w_o.data().as_slice().expect("contiguous"),
                layer.ffn.w_gate.data().as_slice().expect("contiguous"),
                layer.ffn.w_up.data().as_slice().expect("contiguous"),
                layer.ffn.w_down.data().as_slice().expect("contiguous"),
                max_seq_len,
            )
            .map_err(|e| {
                crate::error::Error::ConfigError(format!("Block {i} upload failed: {e:?}"))
            })?;
            cuda_blocks.push(block);
        }
        println!("  ✓ {} transformer blocks uploaded to GPU", num_layers);

        // Step 4: Allocate shared gradient workspace
        let cuda_grad_workspace = CudaGradWorkspace::new(&ctx, mc).map_err(|e| {
            crate::error::Error::ConfigError(format!("Grad workspace alloc failed: {e:?}"))
        })?;

        // Step 5: Allocate GPU training state
        let buf_size = max_seq_len * hidden_size;
        let logits_size = max_seq_len * vocab_size;

        // Activation checkpointing: determine which layers save their inputs.
        // Checkpoint boundary layers (every segment_size layers) are always saved.
        // Non-boundary layers are recomputed from the nearest checkpoint during backward.
        let checkpointing = config.checkpoint_config.enabled;
        let segment_size = if checkpointing {
            let ns = config.checkpoint_config.num_segments.max(1);
            (num_layers + ns - 1) / ns
        } else {
            1 // Every layer is a checkpoint (no recomputation)
        };
        let saved_layer_mask: Vec<bool> =
            (0..num_layers).map(|i| !checkpointing || i % segment_size == 0).collect();

        let mut layer_inputs = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layer_inputs.push(GpuBuffer::new(&ctx, buf_size).map_err(|e| {
                crate::error::Error::ConfigError(format!("Layer input alloc failed: {e:?}"))
            })?);
        }

        // Allocate recompute buffer if checkpointing is enabled
        let recompute_buf = if checkpointing {
            Some(GpuBuffer::new(&ctx, buf_size).map_err(|e| {
                crate::error::Error::ConfigError(format!("Recompute buf alloc failed: {e:?}"))
            })?)
        } else {
            None
        };

        if checkpointing {
            let saved_count = saved_layer_mask.iter().filter(|&&x| x).count();
            println!(
                "  ✓ Activation checkpointing: {} segments, saving {}/{} layer inputs",
                config.checkpoint_config.num_segments, saved_count, num_layers
            );
        }

        // Upload final RMSNorm weight
        let norm_slice = model.norm.weight.data().as_slice().expect("contiguous");
        let final_norm_weight = GpuBuffer::from_host(&ctx, norm_slice).map_err(|e| {
            crate::error::Error::ConfigError(format!("Norm weight upload failed: {e:?}"))
        })?;

        let blocks_output = GpuBuffer::new(&ctx, buf_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("Blocks output alloc failed: {e:?}"))
        })?;
        let grad_buf_a = GpuBuffer::new(&ctx, buf_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("Grad buf A alloc failed: {e:?}"))
        })?;
        let grad_buf_b = GpuBuffer::new(&ctx, buf_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("Grad buf B alloc failed: {e:?}"))
        })?;
        let grad_final_norm_weight = GpuBuffer::new(&ctx, hidden_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("Grad norm alloc failed: {e:?}"))
        })?;
        let norm_output = GpuBuffer::new(&ctx, buf_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("Norm output alloc failed: {e:?}"))
        })?;
        let logits_buf = GpuBuffer::new(&ctx, logits_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("Logits buf alloc failed: {e:?}"))
        })?;
        let lm_head_grad_hidden = GpuBuffer::new(&ctx, buf_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("LM head grad alloc failed: {e:?}"))
        })?;

        // Initialize per-block optimizer states
        let mut optimizer_states = Vec::with_capacity(num_layers);
        for (i, block) in cuda_blocks.iter().enumerate() {
            optimizer_states.push(block.init_optimizer_state().map_err(|e| {
                crate::error::Error::ConfigError(format!("Block {i} opt state failed: {e:?}"))
            })?);
        }

        let gpu_training = GpuPretrainState {
            layer_inputs,
            saved_layer_mask,
            recompute_buf,
            final_norm_weight,
            blocks_output,
            grad_buf_a,
            grad_buf_b,
            grad_final_norm_weight,
            norm_output,
            logits_buf,
            lm_head_grad_hidden,
            optimizer_states,
            step: 0,
        };

        // Step 6: Upload LM head weight to GPU
        // Use tied weights (embed_tokens.weight) or separate lm_head
        let lm_head_data = model.lm_head.as_ref().unwrap_or(&model.embed_tokens.weight).data();
        let lm_head_slice = lm_head_data.as_slice().expect("contiguous");
        let lm_head_weight_gpu = GpuBuffer::from_host(&ctx, lm_head_slice).map_err(|e| {
            crate::error::Error::ConfigError(format!("LM head upload failed: {e:?}"))
        })?;
        let lm_head_grad_gpu = GpuBuffer::new(&ctx, vocab_size * hidden_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("LM head grad alloc failed: {e:?}"))
        })?;
        // CRITICAL: Must zero-initialize m/v buffers. GpuBuffer::new() does NOT
        // zero memory (cuMemAlloc returns uninitialized VRAM).
        let lm_head_m = GpuBuffer::from_host(&ctx, &vec![0.0f32; vocab_size * hidden_size])
            .map_err(|e| {
                crate::error::Error::ConfigError(format!("LM head m alloc failed: {e:?}"))
            })?;
        let lm_head_v = GpuBuffer::from_host(&ctx, &vec![0.0f32; vocab_size * hidden_size])
            .map_err(|e| {
                crate::error::Error::ConfigError(format!("LM head v alloc failed: {e:?}"))
            })?;

        // Final norm optimizer states
        let final_norm_m = GpuBuffer::from_host(&ctx, &vec![0.0f32; hidden_size]).map_err(|e| {
            crate::error::Error::ConfigError(format!("Final norm m alloc failed: {e:?}"))
        })?;
        let final_norm_v = GpuBuffer::from_host(&ctx, &vec![0.0f32; hidden_size]).map_err(|e| {
            crate::error::Error::ConfigError(format!("Final norm v alloc failed: {e:?}"))
        })?;

        // KAIZEN-053: Pre-allocate forward scratch buffers (reused every step)
        let buf_size = max_seq_len * hidden_size;
        let fwd_scratch_a = GpuBuffer::new(&ctx, buf_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("Fwd scratch A alloc failed: {e:?}"))
        })?;
        let fwd_scratch_b = GpuBuffer::new(&ctx, buf_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("Fwd scratch B alloc failed: {e:?}"))
        })?;

        // Sync to ensure all uploads completed
        stream
            .synchronize()
            .map_err(|e| crate::error::Error::ConfigError(format!("Stream sync failed: {e:?}")))?;

        println!(
            "  ✓ GPU training state allocated (LM head: {:.1} MB)",
            (vocab_size * hidden_size * 4) as f64 / 1e6
        );

        // KAIZEN-050: loss_fn removed — cross-entropy computed by fused GPU kernel
        // C-EMBED-GRAD-001: CPU optimizer must match YAML hyperparams (not defaults)
        let embed_optimizer =
            AdamW::new(config.lr, config.beta1, config.beta2, 1e-8, config.weight_decay);

        // R-038: Allocate per-block gradient accumulation buffers (CPU-side)
        // when accumulation_steps > 1 for true gradient accumulation.
        let grad_accum = if config.accumulation_steps > 1 {
            let kv_hidden = mc.num_kv_heads * mc.head_dim();
            let block_sizes =
                super::grad_accumulator::PerBlockGradientAccumulator::compute_block_sizes(
                    hidden_size,
                    kv_hidden,
                    mc.intermediate_size,
                );
            let accum = super::grad_accumulator::PerBlockGradientAccumulator::new(
                num_layers,
                block_sizes,
                vocab_size,
                hidden_size,
            );
            println!(
                "  ✓ Gradient accumulation: {} steps, CPU buffers ({:.1} MB)",
                config.accumulation_steps,
                (accum.block_grads.iter().map(|b| b.total_elements()).sum::<usize>()
                    + accum.lm_head_grad.len()
                    + accum.final_norm_grad.len()
                    + accum.embedding_grad.len()) as f64
                    * 4.0
                    / 1e6,
            );
            Some(accum)
        } else {
            None
        };

        // ALB-091: GPU-resident gradient accumulation (eliminates D2H bottleneck).
        // Falls back to CPU accum if GPU allocation fails.
        let gpu_grad_accum = if config.accumulation_steps > 1 {
            match super::gpu_grad_accumulator::GpuGradientAccumulator::new(&ctx, &mc) {
                Ok(accum) => {
                    println!("  ✓ GPU gradient accumulation enabled (ALB-091)");
                    Some(accum)
                }
                Err(e) => {
                    eprintln!(
                        "  [WARN] GPU gradient accumulation failed ({e}), using CPU fallback"
                    );
                    None
                }
            }
        } else {
            None
        };

        // KAIZEN-059: Pre-allocate D2H staging buffer for gradient accumulation
        // downloads. Only needed when GPU accum is unavailable (CPU fallback path).
        let d2h_staging = if config.accumulation_steps > 1 && gpu_grad_accum.is_none() {
            let ws_max = hidden_size * mc.intermediate_size;
            let lm_max = vocab_size * hidden_size;
            vec![0.0f32; ws_max.max(lm_max)]
        } else {
            Vec::new()
        };

        // ALB-078: Pre-allocate fused gradient clipping state.
        // Eliminates 24 stream syncs per step by keeping norm+clip on GPU.
        let kv_hidden = mc.num_kv_heads * mc.head_dim();
        let fused_clip = Self::init_fused_clip(&ctx, &config, hidden_size, kv_hidden, mc);

        // R-002: Initialize gradient scaler from precision config
        let grad_scaler = GradScaler::from_config(&config.precision_config);
        if config.precision_config.is_mixed() {
            println!(
                "  ✓ Mixed precision: {} (loss scale={}, dynamic={})",
                config.precision_config.compute_precision,
                grad_scaler.scale(),
                grad_scaler.is_dynamic(),
            );
        }

        Ok(Self {
            model,
            cuda_trainer,
            cuda_blocks,
            cuda_grad_workspace,
            gpu_training,
            lm_head_weight_gpu,
            lm_head_grad_gpu,
            lm_head_m,
            lm_head_v,
            final_norm_m,
            final_norm_v,
            embed_optimizer,
            // KAIZEN-047: Read profile_interval before moving config into struct.
            profiler: if config.profile_interval > 0 {
                StepProfiler::new(true, config.profile_interval)
            } else {
                StepProfiler::disabled()
            },
            config,
            metrics: MetricsTracker::new(),
            step: 0,
            accumulated_loss: 0.0,
            accumulated_batches: 0,
            last_grad_norm: 0.0,
            last_embed_grad_norm: 0.0,
            grad_accum,
            gpu_grad_accum,
            grad_scaler,
            fwd_scratch_a,
            fwd_scratch_b,
            h2d_staging: vec![0.0f32; max_seq_len * hidden_size],
            d2h_staging,
            fused_clip,
        })
    }

    /// ALB-078: Initialize fused gradient clipping state (extracted for complexity).
    fn init_fused_clip(
        ctx: &std::sync::Arc<trueno_gpu::driver::CudaContext>,
        config: &TransformerTrainConfig,
        hidden_size: usize,
        kv_hidden: usize,
        mc: &crate::transformer::TransformerConfig,
    ) -> Option<FusedClipState> {
        config.base.max_grad_norm?;
        let grad_sizes: [u32; 9] = [
            (hidden_size * hidden_size) as u32,
            (hidden_size * kv_hidden) as u32,
            (hidden_size * kv_hidden) as u32,
            (hidden_size * hidden_size) as u32,
            (hidden_size * mc.intermediate_size) as u32,
            (hidden_size * mc.intermediate_size) as u32,
            (mc.intermediate_size * hidden_size) as u32,
            hidden_size as u32,
            hidden_size as u32,
        ];
        match FusedClipState::new(ctx, &grad_sizes) {
            Ok(state) => {
                println!(
                    "  ✓ Fused gradient clipping: {} partials ({:.1} KB)",
                    state.total_partials,
                    state.total_partials as f64 * 4.0 / 1024.0,
                );
                Some(state)
            }
            Err(e) => {
                println!("  ⚠ Fused clip alloc failed ({e:?}), using sync fallback");
                None
            }
        }
    }

    /// Run one forward+backward step for a single sequence.
    ///
    /// # Contract (C-GPUSTEP-001)
    ///
    /// - Precondition: `input_ids.len() == target_ids.len() <= max_seq_len`
    /// - Postcondition: If `accumulate_only` is false, all GPU weights updated.
    ///   If true, gradients accumulated into CPU buffers (no weight updates).
    /// - Transfer count: 1 PCIe H2D + ~1KB control (KAIZEN-050, + 24×9 D2H if accumulating)
    fn train_step_single(
        &mut self,
        input_ids: &[u32],
        target_ids: &[u32],
        accumulate_only: bool,
    ) -> Option<f32> {
        self.profiler.begin_step();
        let result = self.train_step_inner(input_ids, target_ids, accumulate_only);
        self.profiler.finish_step();
        result
    }

    /// Inner training step — separated so profiler always records the step.
    fn train_step_inner(
        &mut self,
        input_ids: &[u32],
        target_ids: &[u32],
        accumulate_only: bool,
    ) -> Option<f32> {
        let hidden_size = self.config.model_config.hidden_size;
        let vocab_size = self.config.model_config.vocab_size;

        // Truncate to max_seq_len — GPU buffers are pre-allocated for this size
        let max_sl = self.config.max_seq_len;
        let input_ids = if input_ids.len() > max_sl { &input_ids[..max_sl] } else { input_ids };
        let target_ids = if target_ids.len() > max_sl { &target_ids[..max_sl] } else { target_ids };
        let seq_len = input_ids.len();

        // Steps 1-6: GPU forward pass — logits stay GPU-resident (KAIZEN-050)
        // (sub-phases embed, h2d, forward, norm_lm instrumented inside gpu_forward)
        self.gpu_forward(input_ids, seq_len, hidden_size, vocab_size)?;

        // Step 7: Fused GPU cross-entropy loss + softmax backward (KAIZEN-050)
        // Eliminates: logits D2H (77.8MB) + CPU softmax (40ms) + grad H2D (77.8MB)
        self.profiler.begin(StepProfiler::LOSS);
        let stream = self.cuda_trainer.stream();

        // Compute combined scale: (1/seq_len) * (1/accum_steps)
        //
        // ALB-072: Do NOT multiply by grad_scaler.scale() here. All backward
        // computation uses f32 GpuBuffers — there is no fp16 gradient underflow
        // risk. The 65536x loss scaling caused gradient overflow in early layers
        // (blocks 0-1 went NaN). The GradScaler remains active for the CPU
        // embedding path (unscale_and_check in optimizer_step) as a safety check,
        // but it operates with scale=1.0 effective for GPU gradients.
        let mut loss_scale = 1.0 / seq_len as f32;
        if self.config.accumulation_steps > 1 {
            loss_scale /= self.config.accumulation_steps as f32;
        }

        // KAIZEN-052: In-place — gradient written directly to logits_buf.
        let loss_val = fused_cross_entropy_cuda(
            &mut self.gpu_training.logits_buf,
            target_ids,
            seq_len as u32,
            vocab_size as u32,
            loss_scale,
            stream,
        )
        .ok()?;

        // NaN guard (replaces logits NaN check — NaN logits → NaN loss via kernel)
        if !loss_val.is_finite() {
            return None;
        }
        self.profiler.end(StepProfiler::LOSS);

        // Steps 8-11: GPU backward pass (with or without optimizer)
        // (sub-phases lm_bwd, norm_bwd, blk_bwd instrumented inside gpu_backward)
        // KAIZEN-050: grad_logits on GPU. KAIZEN-052: grad lives in logits_buf (in-place).
        let grad_output_is_a =
            self.gpu_backward(seq_len, hidden_size, vocab_size, accumulate_only)?;

        // Step 12: Embedding backward (CPU scatter-add always accumulates)
        self.profiler.begin(StepProfiler::EMBED_BWD);
        self.embed_backward(input_ids, seq_len, hidden_size, vocab_size, grad_output_is_a)?;
        self.profiler.end(StepProfiler::EMBED_BWD);

        Some(loss_val)
    }

    /// GPU forward pass: embed → blocks → norm → LM head.
    ///
    /// Logits stay GPU-resident in `self.gpu_training.logits_buf` (KAIZEN-050).
    /// Transfers: 1 H2D (hidden states). No D2H — logits consumed by fused kernel.
    #[allow(unsafe_code)]
    fn gpu_forward(
        &mut self,
        input_ids: &[u32],
        seq_len: usize,
        hidden_size: usize,
        vocab_size: usize,
    ) -> Option<()> {
        let stream = self.cuda_trainer.stream();

        // Embedding lookup (CPU)
        self.profiler.begin(StepProfiler::EMBED);
        let hidden = self.model.embed_tokens.forward(input_ids);
        let hidden_slice = hidden.data().as_slice()?;
        self.profiler.end(StepProfiler::EMBED);

        // Upload hidden states to GPU (Transfer 1: H2D)
        // Pad to max_seq_len so D2D copies to pre-allocated layer_inputs match.
        // KAIZEN-053: Reuse pre-allocated scratch buffers instead of cuMemAlloc per step.
        // KAIZEN-056: Reuse pre-allocated h2d_staging instead of alloc per step.
        self.profiler.begin(StepProfiler::H2D);
        self.h2d_staging[..hidden_slice.len()].copy_from_slice(hidden_slice);
        self.h2d_staging[hidden_slice.len()..].fill(0.0);
        self.fwd_scratch_a.copy_from_host(&self.h2d_staging).ok()?;
        self.profiler.end(StepProfiler::H2D);

        // Forward through CUDA blocks using pre-allocated ping-pong buffers.
        // KAIZEN-053: fwd_scratch_a/b are top-level fields (not in gpu_training)
        // so borrowing them doesn't conflict with gpu_training.layer_inputs.
        self.profiler.begin(StepProfiler::FORWARD);
        let mut input_is_a = true; // Track which scratch buffer is "input"
        for (i, block) in self.cuda_blocks.iter_mut().enumerate() {
            // Use raw pointers for the ping-pong to avoid borrow conflicts
            // with self.gpu_training.layer_inputs
            let (input_ptr, output_ptr): (*const GpuBuffer<f32>, *mut GpuBuffer<f32>) =
                if input_is_a {
                    (
                        std::ptr::from_ref(&self.fwd_scratch_a),
                        std::ptr::from_mut(&mut self.fwd_scratch_b),
                    )
                } else {
                    (
                        std::ptr::from_ref(&self.fwd_scratch_b),
                        std::ptr::from_mut(&mut self.fwd_scratch_a),
                    )
                };
            if self.gpu_training.saved_layer_mask[i] {
                // SAFETY: Both buffers are valid GPU allocations with matching max_seq_len size.
                // Copy completes before block.forward() reads from input (same stream ordering).
                unsafe {
                    self.gpu_training.layer_inputs[i]
                        .copy_from_buffer_async(&*input_ptr, stream)
                        .ok()?;
                }
            }
            // SAFETY: input_ptr and output_ptr point to disjoint fwd_scratch_{a,b}.
            unsafe {
                block.forward(&*input_ptr, &mut *output_ptr, seq_len, stream).ok()?;
            }
            input_is_a = !input_is_a;
        }
        self.profiler.end(StepProfiler::FORWARD);

        // After the loop, input_is_a tells us which buffer has the final output
        let final_output: &GpuBuffer<f32> =
            if input_is_a { &self.fwd_scratch_a } else { &self.fwd_scratch_b };

        // Save blocks output for final norm backward
        // SAFETY: Disjoint GPU buffers with matching max_seq_len sizes.
        self.profiler.begin(StepProfiler::NORM_LM);
        unsafe {
            self.gpu_training.blocks_output.copy_from_buffer_async(final_output, stream).ok()?;
        }

        // Final RMSNorm forward (GPU)
        rms_norm_forward(
            final_output,
            &self.gpu_training.final_norm_weight,
            &mut self.gpu_training.norm_output,
            seq_len as u32,
            hidden_size as u32,
            stream,
        )
        .ok()?;

        // LM head GEMM forward (GPU)
        // gemm_forward treats flat (V,H) memory as (H,V) row-major, which
        // implicitly transposes — matching the CPU matmul's tied-weight behavior.
        gemm_forward(
            &self.gpu_training.norm_output,
            &self.lm_head_weight_gpu,
            &mut self.gpu_training.logits_buf,
            seq_len as u32,
            hidden_size as u32,
            vocab_size as u32,
            stream,
        )
        .ok()?;

        // KAIZEN-050: Logits stay GPU-resident — no D2H transfer.
        // Fused cross-entropy kernel reads logits_buf directly on GPU.
        self.profiler.end(StepProfiler::NORM_LM);

        Some(())
    }

    /// ALB-089: Forward-only pass that returns last-position logits on CPU.
    ///
    /// Runs the same GPU forward as training but downloads only the last
    /// position's logits (vocab_size floats) for token sampling. No backward
    /// pass, no loss computation.
    ///
    /// # Contract (C-CUDA-INF-001)
    ///
    /// - Same forward path as `gpu_forward()` — identical logits
    /// - Only downloads `logits[seq_len-1, :]` (128 KB for 32K vocab)
    /// - stream.synchronize() before D2H (C-STREAMSYNC-001)
    pub fn forward_logits(&mut self, input_ids: &[u32]) -> Option<Vec<f32>> {
        let seq_len = input_ids.len();
        let hidden_size = self.config.model_config.hidden_size;
        let vocab_size = self.config.model_config.vocab_size;

        if seq_len == 0 || seq_len > self.config.max_seq_len {
            return None;
        }

        // Reuse gpu_forward for the actual computation
        self.gpu_forward(input_ids, seq_len, hidden_size, vocab_size)?;

        // C-STREAMSYNC-001: synchronize before D2H
        let stream = self.cuda_trainer.stream();
        stream.synchronize().ok()?;

        // Download last position logits only: logits_buf[seq_len-1, :]
        let offset = (seq_len - 1) * vocab_size;
        let mut logits = vec![0.0f32; vocab_size];
        self.gpu_training.logits_buf.copy_to_host_at(&mut logits, offset).ok()?;

        Some(logits)
    }

    /// GPU backward pass with interleaved per-block optimizer step.
    ///
    /// Each block's backward writes weight gradients to the shared `CudaGradWorkspace`.
    /// Recompute layer inputs for a segment during backward (activation checkpointing).
    ///
    /// When checkpointing is enabled, non-checkpoint layers don't save their inputs
    /// during forward. Before their backward pass, we recompute from the nearest
    /// checkpoint by re-running forward through intermediate blocks.
    ///
    /// This recomputes the entire segment [checkpoint..=target_layer], storing
    /// intermediate layer_inputs so subsequent layers in the same segment don't
    /// need redundant recomputation.
    ///
    /// # Contract (R-021)
    ///
    /// After this call, `layer_inputs[i]` is valid for all i in [checkpoint..=target_layer].
    #[allow(unsafe_code)]
    fn recompute_segment(
        gpu_training: &mut GpuPretrainState,
        cuda_blocks: &mut [crate::transformer::CudaTransformerBlock],
        target_layer: usize,
        seq_len: usize,
        stream: &CudaStream,
    ) -> Option<()> {
        // Find nearest saved checkpoint at or before target
        let seg_start = (0..=target_layer).rev().find(|&i| gpu_training.saved_layer_mask[i])?;

        if seg_start == target_layer {
            return Some(()); // Already saved
        }

        // Copy checkpoint input to recompute_buf as starting point.
        // SAFETY: recompute_buf and layer_inputs are disjoint allocations.
        let recompute_buf = gpu_training.recompute_buf.as_mut()?;
        unsafe {
            recompute_buf
                .copy_from_buffer_async(&gpu_training.layer_inputs[seg_start], stream)
                .ok()?;
        }

        // Forward through blocks [seg_start..target_layer], saving intermediate inputs.
        // For block i, input → block i → output becomes input for block i+1.
        // We save output (= input to block i+1) in layer_inputs[i+1].
        //
        // Buffer pattern:
        //   i == seg_start: input = recompute_buf, output = layer_inputs[seg_start+1]
        //   i > seg_start:  input = layer_inputs[i], output = layer_inputs[i+1]
        //
        // SAFETY: split_at_mut ensures non-overlapping borrows of layer_inputs.
        // recompute_buf is separate from layer_inputs.
        for i in seg_start..target_layer {
            if i == seg_start {
                // Input is in recompute_buf, output goes to layer_inputs[i+1]
                let recompute_ptr: *const GpuBuffer<f32> = recompute_buf;
                let li = &mut gpu_training.layer_inputs;
                unsafe {
                    cuda_blocks[i]
                        .forward(&*recompute_ptr, &mut li[i + 1], seq_len, stream)
                        .ok()?;
                }
            } else {
                // Input = layer_inputs[i], output = layer_inputs[i+1]
                let li = &mut gpu_training.layer_inputs;
                let (left, right) = li.split_at_mut(i + 1);
                cuda_blocks[i].forward(&left[i], &mut right[0], seq_len, stream).ok()?;
            }
        }

        Some(())
    }

    /// Since `gemm_backward_b` overwrites (not accumulates), we must run each block's
    /// optimizer step immediately after its backward, before the next block overwrites
    /// the workspace. This also enables per-block gradient clipping.
    ///
    /// When `accumulate_only` is true (R-038 gradient accumulation), the per-block
    /// optimizer steps are skipped and workspace gradients are downloaded to CPU-side
    /// `PerBlockGradientAccumulator` instead. LM head and final norm gradients are
    /// also downloaded and accumulated. The optimizer step is deferred until
    /// `gpu_optimizer_from_accum()` is called.
    ///
    /// Returns `grad_output_is_a` flag for embedding backward.
    /// Transfer: 0 H2D (KAIZEN-050/052: grad in logits_buf) + 24×9 D2H if accumulating.
    #[allow(unsafe_code)]
    fn gpu_backward(
        &mut self,
        seq_len: usize,
        hidden_size: usize,
        vocab_size: usize,
        accumulate_only: bool,
    ) -> Option<bool> {
        let stream = self.cuda_trainer.stream();
        let max_grad_norm = self.config.base.max_grad_norm;
        let lr = self.current_lr();
        // ALB-072: No inv_scale needed — loss_scale no longer includes grad_scaler.
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let weight_decay = self.config.weight_decay;

        // KAIZEN-050: grad_logits GPU-resident. KAIZEN-052: grad lives in logits_buf (in-place).
        // No separate grad buffer. No GRAD_H2D transfer.

        // LM head GEMM backward
        self.profiler.begin(StepProfiler::LM_BWD);
        gemm_backward_a(
            &self.gpu_training.logits_buf,
            &self.lm_head_weight_gpu,
            &mut self.gpu_training.lm_head_grad_hidden,
            seq_len as u32,
            hidden_size as u32,
            vocab_size as u32,
            stream,
        )
        .ok()?;

        gemm_backward_b(
            &self.gpu_training.norm_output,
            &self.gpu_training.logits_buf,
            &mut self.lm_head_grad_gpu,
            seq_len as u32,
            hidden_size as u32,
            vocab_size as u32,
            stream,
        )
        .ok()?;

        // Clip LM head weight gradient
        // KAIZEN-049: GPU norm reduction.
        // KAIZEN-051: No explicit sync needed — same stream ordering.
        // ALB-071: Always compute LM head grad norm for observability (R-004).
        let lm_norm =
            squared_sum_cuda(&self.lm_head_grad_gpu, self.lm_head_grad_gpu.len() as u32, stream)
                .unwrap_or(0.0);
        self.last_grad_norm = lm_norm; // R-004: capture for observability (now on unscaled grads)
        if let Some(max_norm) = max_grad_norm {
            let clip_scale = if lm_norm > max_norm { max_norm / lm_norm } else { 1.0 };
            let n = self.lm_head_grad_gpu.len() as u32;
            let _ = gradient_clip_cuda(&mut self.lm_head_grad_gpu, clip_scale, n, stream);
        }
        self.profiler.end(StepProfiler::LM_BWD);

        // Final RMSNorm backward
        self.profiler.begin(StepProfiler::NORM_BWD);
        rms_norm_backward(
            &self.gpu_training.blocks_output,
            &self.gpu_training.final_norm_weight,
            &self.gpu_training.lm_head_grad_hidden,
            &mut self.gpu_training.grad_buf_a,
            &mut self.gpu_training.grad_final_norm_weight,
            seq_len as u32,
            hidden_size as u32,
            1e-5_f32,
            stream,
        )
        .ok()?;

        // Clip final norm weight gradient
        // KAIZEN-051: No explicit sync needed — same stream ordering as LM head clip.
        if let Some(max_norm) = max_grad_norm {
            let (scale, _) = Self::compute_clip_scale_with_norm(
                &self.gpu_training.grad_final_norm_weight,
                max_norm,
                stream,
            );
            let n = self.gpu_training.grad_final_norm_weight.len() as u32;
            let _ =
                gradient_clip_cuda(&mut self.gpu_training.grad_final_norm_weight, scale, n, stream);
        }
        self.profiler.end(StepProfiler::NORM_BWD);

        // R-038: Either accumulate non-block grads or run non-block optimizer.
        if accumulate_only {
            // ALB-091: GPU-resident accumulation (no sync, no D2H) or CPU fallback.
            if let Some(ref mut gpu_accum) = self.gpu_grad_accum {
                let _ = gpu_accum.accumulate_nonblock(
                    &self.lm_head_grad_gpu,
                    &self.gpu_training.grad_final_norm_weight,
                    stream,
                );
            } else {
                stream.synchronize().ok()?;
                Self::download_nonblock_grads_to_accum(
                    &self.lm_head_grad_gpu,
                    &self.gpu_training.grad_final_norm_weight,
                    &mut self.grad_accum,
                    &mut self.d2h_staging,
                )?;
            }
        } else {
            Self::run_nonblock_optimizer_step(
                &mut self.gpu_training,
                &mut self.lm_head_weight_gpu,
                &self.lm_head_grad_gpu,
                &mut self.lm_head_m,
                &mut self.lm_head_v,
                &mut self.final_norm_m,
                &mut self.final_norm_v,
                lr,
                beta1,
                beta2,
                weight_decay,
                stream,
            );
        }

        // Backward through blocks in reverse, with interleaved clip + optimizer.
        // Each block's backward writes weight gradients to shared CudaGradWorkspace.
        //
        // SAFETY: grad_buf_a and grad_buf_b are disjoint fields. Raw pointers
        // allow alternating read/write without violating aliasing rules.
        self.profiler.begin(StepProfiler::BLK_BWD);
        let grad_a_ptr: *mut GpuBuffer<f32> = &mut self.gpu_training.grad_buf_a;
        let grad_b_ptr: *mut GpuBuffer<f32> = &mut self.gpu_training.grad_buf_b;
        let mut grad_output_is_a = true;

        for layer_idx in (0..self.cuda_blocks.len()).rev() {
            // Activation checkpointing: if this layer's input wasn't saved during
            // forward, recompute the segment from the nearest checkpoint.
            if !self.gpu_training.saved_layer_mask[layer_idx] {
                Self::recompute_segment(
                    &mut self.gpu_training,
                    &mut self.cuda_blocks,
                    layer_idx,
                    seq_len,
                    stream,
                )?;
            }

            let (grad_output, grad_input) = unsafe {
                if grad_output_is_a {
                    (&*grad_a_ptr, &mut *grad_b_ptr)
                } else {
                    (&*grad_b_ptr, &mut *grad_a_ptr)
                }
            };
            self.cuda_blocks[layer_idx]
                .backward(
                    &self.gpu_training.layer_inputs[layer_idx],
                    grad_output,
                    grad_input,
                    seq_len,
                    stream,
                    &mut self.cuda_grad_workspace,
                )
                .ok()?;

            // KAIZEN-054: Per-block gradient clipping re-enabled via GPU-side norm.
            // squared_sum_cuda computes L2 norm on GPU (~1KB D2H per buffer, 9 buffers).
            // ALB-067 disabled this due to CPU D2H bottleneck (864 transfers/step);
            // now only 9 tiny partial-sum downloads per block.
            //
            // No stream.synchronize() needed: backward, unscale, clip, and optimizer_step
            // are all GPU-side operations on the same stream.
            // ALB-078: Fused GPU clip (zero sync) or sync-based fallback.
            if let Some(max_norm) = max_grad_norm {
                if let Some(ref state) = self.fused_clip {
                    fused_clip_workspace_gradients(
                        &mut self.cuda_grad_workspace,
                        max_norm,
                        state,
                        stream,
                    );
                } else {
                    clip_workspace_gradients(&mut self.cuda_grad_workspace, max_norm, stream);
                }
            }

            // R-038: Either accumulate workspace grads or run optimizer per-block.
            if accumulate_only {
                // ALB-091: GPU-resident accumulation (no sync, no D2H) or CPU fallback.
                if let Some(ref mut gpu_accum) = self.gpu_grad_accum {
                    let _ =
                        gpu_accum.accumulate_block(&self.cuda_grad_workspace, layer_idx, stream);
                } else {
                    // CPU fallback: SYNC + D2H (ALB-065 / Rule 6).
                    stream.synchronize().ok()?;
                    if let Some(accum) = &mut self.grad_accum {
                        Self::download_workspace_to_accum(
                            &self.cuda_grad_workspace,
                            accum,
                            layer_idx,
                            &mut self.d2h_staging,
                        )?;
                    }
                }
            } else {
                // Per-block optimizer step: consume workspace gradients before next block overwrites
                let step = self.gpu_training.step;
                let _ = self.cuda_blocks[layer_idx].optimizer_step(
                    &mut self.gpu_training.optimizer_states[layer_idx],
                    step,
                    lr,
                    beta1,
                    beta2,
                    1e-8,
                    weight_decay,
                    stream,
                    &self.cuda_grad_workspace,
                );
            }

            grad_output_is_a = !grad_output_is_a;
        }

        stream.synchronize().ok()?;
        self.profiler.end(StepProfiler::BLK_BWD);

        Some(grad_output_is_a)
    }

    /// R-038: Download non-block (LM head + final norm) gradients to CPU accumulator.
    /// Static method to avoid borrow conflicts.
    // KAIZEN-044: Pre-allocate single buffer for LM head + norm D2H downloads.
    // lm_head_grad is vocab×hidden (389M elements = 1.5 GB for Qwen3-4B).
    // KAIZEN-059: Host buffer now passed in (d2h_staging) — zero per-call allocations.
    fn download_nonblock_grads_to_accum(
        lm_head_grad: &GpuBuffer<f32>,
        final_norm_grad: &GpuBuffer<f32>,
        grad_accum: &mut Option<super::grad_accumulator::PerBlockGradientAccumulator>,
        host: &mut [f32],
    ) -> Option<()> {
        let accum = grad_accum.as_mut()?;

        let lm_slice = &mut host[..lm_head_grad.len()];
        lm_head_grad.copy_to_host_at(lm_slice, 0).ok()?;
        for (d, s) in accum.lm_head_grad.iter_mut().zip(lm_slice.iter()) {
            *d += s;
        }

        let norm_slice = &mut host[..final_norm_grad.len()];
        final_norm_grad.copy_to_host_at(norm_slice, 0).ok()?;
        for (d, s) in accum.final_norm_grad.iter_mut().zip(norm_slice.iter()) {
            *d += s;
        }
        Some(())
    }

    /// Run LM head + final norm optimizer step (non-accumulating path).
    /// Static method to avoid borrow conflicts with `stream`.
    #[allow(clippy::too_many_arguments)]
    fn run_nonblock_optimizer_step(
        gpu_training: &mut GpuPretrainState,
        lm_head_weight_gpu: &mut GpuBuffer<f32>,
        lm_head_grad_gpu: &GpuBuffer<f32>,
        lm_head_m: &mut GpuBuffer<f32>,
        lm_head_v: &mut GpuBuffer<f32>,
        final_norm_m: &mut GpuBuffer<f32>,
        final_norm_v: &mut GpuBuffer<f32>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
        stream: &CudaStream,
    ) {
        gpu_training.step += 1;
        let step = gpu_training.step;

        let n_lm = lm_head_weight_gpu.len() as u32;
        let _ = adamw_step_cuda(
            lm_head_weight_gpu,
            lm_head_grad_gpu,
            lm_head_m,
            lm_head_v,
            lr,
            beta1,
            beta2,
            1e-8,
            weight_decay,
            step,
            n_lm,
            stream,
        );

        let n_norm = gpu_training.final_norm_weight.len() as u32;
        let _ = adamw_step_cuda(
            &mut gpu_training.final_norm_weight,
            &gpu_training.grad_final_norm_weight,
            final_norm_m,
            final_norm_v,
            lr,
            beta1,
            beta2,
            1e-8,
            weight_decay,
            step,
            n_norm,
            stream,
        );
    }

    /// R-038: Download shared CudaGradWorkspace to CPU per-block accumulation buffers.
    ///
    /// Static method to avoid borrow conflicts with `stream` (same pattern as
    /// `recompute_segment`). Must be called after stream.synchronize() (ALB-065 / Rule 6).
    // KAIZEN-044: Pre-allocate a single host buffer for all D2H downloads
    // in download_workspace_to_accum. Was allocating vec![0.0f32; len] × 9 buffers.
    // KAIZEN-059: Host buffer now passed in (d2h_staging) — zero per-call allocations.
    fn download_workspace_to_accum(
        ws: &CudaGradWorkspace,
        accum: &mut super::grad_accumulator::PerBlockGradientAccumulator,
        layer_idx: usize,
        host: &mut [f32],
    ) -> Option<()> {
        let bg = &mut accum.block_grads[layer_idx];

        use super::grad_accumulator::component;
        let bufs_and_components: [(&GpuBuffer<f32>, usize); 9] = [
            (&ws.grad_w_q, component::W_Q),
            (&ws.grad_w_k, component::W_K),
            (&ws.grad_w_v, component::W_V),
            (&ws.grad_w_o, component::W_O),
            (&ws.grad_gate, component::GATE),
            (&ws.grad_up, component::UP),
            (&ws.grad_down, component::DOWN),
            (&ws.grad_input_norm, component::INPUT_NORM),
            (&ws.grad_post_attn_norm, component::POST_ATTN_NORM),
        ];

        for (gpu_buf, comp_idx) in &bufs_and_components {
            let slice = &mut host[..gpu_buf.len()];
            gpu_buf.copy_to_host_at(slice, 0).ok()?;
            for (d, s) in bg.components[*comp_idx].iter_mut().zip(slice.iter()) {
                *d += s;
            }
        }
        Some(())
    }

    /// R-038: Upload averaged CPU accumulation buffers to GPU workspace and run
    /// optimizer step for all blocks + LM head + final norm.
    ///
    /// Called once after `accumulation_steps` micro-batches have been accumulated.
    /// ALB-091: Run optimizer step from GPU-resident accumulated gradients.
    /// D2D copy accum → workspace, then run per-block optimizer. Zero accum after.
    fn gpu_optimizer_from_gpu_accum(&mut self) -> Option<()> {
        let stream = self.cuda_trainer.stream();
        let lr = self.current_lr();
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let weight_decay = self.config.weight_decay;

        // Sync once to ensure all accumulation kernels complete
        stream.synchronize().ok()?;

        self.gpu_training.step += 1;
        let step = self.gpu_training.step;

        // Upload GPU accum → workspace (D2D) and run optimizer per block
        let gpu_accum = self.gpu_grad_accum.as_ref()?;
        for layer_idx in 0..self.cuda_blocks.len() {
            gpu_accum.upload_to_workspace(&mut self.cuda_grad_workspace, layer_idx).ok()?;

            let _ = self.cuda_blocks[layer_idx].optimizer_step(
                &mut self.gpu_training.optimizer_states[layer_idx],
                step,
                lr,
                beta1,
                beta2,
                1e-8,
                weight_decay,
                stream,
                &self.cuda_grad_workspace,
            );
        }

        // LM head: D2D copy accum → grad buffer, then optimizer step
        gpu_accum
            .upload_nonblock(
                &mut self.lm_head_grad_gpu,
                &mut self.gpu_training.grad_final_norm_weight,
            )
            .ok()?;

        let n_lm = self.lm_head_weight_gpu.len() as u32;
        let _ = adamw_step_cuda(
            &mut self.lm_head_weight_gpu,
            &self.lm_head_grad_gpu,
            &mut self.lm_head_m,
            &mut self.lm_head_v,
            lr,
            beta1,
            beta2,
            1e-8,
            weight_decay,
            step,
            n_lm,
            stream,
        );

        // Final norm optimizer step
        let n_norm = self.gpu_training.final_norm_weight.len() as u32;
        let _ = adamw_step_cuda(
            &mut self.gpu_training.final_norm_weight,
            &self.gpu_training.grad_final_norm_weight,
            &mut self.final_norm_m,
            &mut self.final_norm_v,
            lr,
            beta1,
            beta2,
            1e-8,
            weight_decay,
            step,
            n_norm,
            stream,
        );

        stream.synchronize().ok()?;

        // Zero accum for next window
        if let Some(ref mut gpu_accum) = self.gpu_grad_accum {
            let _ = gpu_accum.zero_all();
        }

        Some(())
    }

    #[allow(unsafe_code)]
    fn gpu_optimizer_from_accum(&mut self) -> Option<()> {
        let stream = self.cuda_trainer.stream();
        let lr = self.current_lr();
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let weight_decay = self.config.weight_decay;

        // Average accumulated gradients
        let accum = self.grad_accum.as_mut()?;
        accum.average();

        // Jidoka: check for NaN/Inf before applying
        if accum.has_non_finite() {
            println!("[WARN] R-038: NaN/Inf in accumulated gradients, skipping optimizer step");
            accum.zero_all();
            return Some(());
        }

        self.gpu_training.step += 1;
        let step = self.gpu_training.step;

        // Upload accumulated gradients and run optimizer for each block
        use super::grad_accumulator::component;
        for layer_idx in 0..self.cuda_blocks.len() {
            let bg = &accum.block_grads[layer_idx];

            // Upload accumulated gradients to shared workspace
            unsafe {
                self.cuda_grad_workspace
                    .grad_w_q
                    .copy_from_host_async(&bg.components[component::W_Q], stream)
                    .ok()?;
                self.cuda_grad_workspace
                    .grad_w_k
                    .copy_from_host_async(&bg.components[component::W_K], stream)
                    .ok()?;
                self.cuda_grad_workspace
                    .grad_w_v
                    .copy_from_host_async(&bg.components[component::W_V], stream)
                    .ok()?;
                self.cuda_grad_workspace
                    .grad_w_o
                    .copy_from_host_async(&bg.components[component::W_O], stream)
                    .ok()?;
                self.cuda_grad_workspace
                    .grad_gate
                    .copy_from_host_async(&bg.components[component::GATE], stream)
                    .ok()?;
                self.cuda_grad_workspace
                    .grad_up
                    .copy_from_host_async(&bg.components[component::UP], stream)
                    .ok()?;
                self.cuda_grad_workspace
                    .grad_down
                    .copy_from_host_async(&bg.components[component::DOWN], stream)
                    .ok()?;
                self.cuda_grad_workspace
                    .grad_input_norm
                    .copy_from_host_async(&bg.components[component::INPUT_NORM], stream)
                    .ok()?;
                self.cuda_grad_workspace
                    .grad_post_attn_norm
                    .copy_from_host_async(&bg.components[component::POST_ATTN_NORM], stream)
                    .ok()?;
            }

            // Run optimizer step with uploaded averaged gradients
            let _ = self.cuda_blocks[layer_idx].optimizer_step(
                &mut self.gpu_training.optimizer_states[layer_idx],
                step,
                lr,
                beta1,
                beta2,
                1e-8,
                weight_decay,
                stream,
                &self.cuda_grad_workspace,
            );
        }

        // Upload accumulated LM head gradients and run AdamW step
        unsafe {
            self.lm_head_grad_gpu.copy_from_host_async(&accum.lm_head_grad, stream).ok()?;
        }
        let n_lm = self.lm_head_weight_gpu.len() as u32;
        let _ = adamw_step_cuda(
            &mut self.lm_head_weight_gpu,
            &self.lm_head_grad_gpu,
            &mut self.lm_head_m,
            &mut self.lm_head_v,
            lr,
            beta1,
            beta2,
            1e-8,
            weight_decay,
            step,
            n_lm,
            stream,
        );

        // Upload accumulated final norm gradients and run AdamW step
        unsafe {
            self.gpu_training
                .grad_final_norm_weight
                .copy_from_host_async(&accum.final_norm_grad, stream)
                .ok()?;
        }
        let n_norm = self.gpu_training.final_norm_weight.len() as u32;
        let _ = adamw_step_cuda(
            &mut self.gpu_training.final_norm_weight,
            &self.gpu_training.grad_final_norm_weight,
            &mut self.final_norm_m,
            &mut self.final_norm_v,
            lr,
            beta1,
            beta2,
            1e-8,
            weight_decay,
            step,
            n_norm,
            stream,
        );

        stream.synchronize().ok()?;

        // Zero accum for next window
        accum.zero_all();
        Some(())
    }

    /// Compute gradient L2 norm via GPU reduction kernel (KAIZEN-049).
    ///
    /// Runs `SquaredSumKernel` on GPU, downloads only `num_blocks` partial sums (~1KB)
    /// instead of the full buffer (128MB for lm_head). Falls back to CPU download on error.
    ///
    /// # Contract (C-CLIPNORM-GPU-001)
    ///
    /// - **Precondition**: `buf.len() > 0`, stream is synchronized with prior kernel
    /// - **Postcondition**: `grad_norm ≈ sqrt(sum(buf[i]^2))`, `scale = min(1, max_norm/norm)`
    /// - **Transfer**: ~1KB D2H (num_blocks × 4B) vs n×4B (128MB for 32M elements)
    ///
    /// R-004: Returns `(clip_scale, grad_norm)` for observability.
    fn compute_clip_scale_with_norm(
        buf: &GpuBuffer<f32>,
        max_norm: f32,
        stream: &CudaStream,
    ) -> (f32, f32) {
        let n = buf.len() as u32;
        // Try GPU reduction first — ~1KB D2H instead of n×4 bytes
        let grad_norm = match squared_sum_cuda(buf, n, stream) {
            Ok(norm) => norm,
            Err(_) => {
                // Fallback: full D2H (original path)
                let mut host = vec![0.0f32; buf.len()];
                if buf.copy_to_host_at(&mut host, 0).is_err() {
                    return (1.0, 0.0);
                }
                let sq_sum: f64 = host.iter().map(|&x| (x as f64) * (x as f64)).sum();
                sq_sum.sqrt() as f32
            }
        };
        let scale = if grad_norm > max_norm { max_norm / grad_norm } else { 1.0 };
        (scale, grad_norm)
    }

    /// Download embedding gradient from GPU, clip, and scatter-add into CPU weight.
    ///
    /// # Contract (C-EMBED-GRAD-001)
    ///
    /// The activation gradient from block[0]'s backward is unclipped (per-block clipping
    /// only applies to weight gradients in the shared workspace). For deep networks with
    /// random init, this gradient can overflow f32, producing NaN in the CPU AdamW.
    /// We clip the activation gradient to max_grad_norm before scatter-adding.
    #[allow(unsafe_code)]
    fn embed_backward(
        &mut self,
        input_ids: &[u32],
        _seq_len: usize,
        hidden_size: usize,
        vocab_size: usize,
        grad_output_is_a: bool,
    ) -> Option<()> {
        // The final backward output is in whichever buffer was last written
        let grad_a_ptr: *const GpuBuffer<f32> = &self.gpu_training.grad_buf_a;
        let grad_b_ptr: *const GpuBuffer<f32> = &self.gpu_training.grad_buf_b;
        let embed_grad_buf = unsafe {
            if grad_output_is_a {
                &*grad_a_ptr
            } else {
                &*grad_b_ptr
            }
        };
        let mut embed_grad_data = self.cuda_trainer.download(embed_grad_buf).ok()?;

        // C-EMBED-GRAD-001: ALWAYS clip activation gradient before scatter-add.
        // Without this, 24-layer random-init backward amplifies gradients to ~1e35,
        // which overflows the CPU AdamW's second moment buffer.
        //
        // ALB-071: Decoupled from general grad_clip config. Embed activation gradient
        // clipping is a SAFETY constraint (prevents NaN), not a training hyperparameter.
        // Uses dedicated max_embed_grad_norm (default 1.0) independent of weight grad_clip.
        let embed_clip_norm = self.config.base.max_grad_norm.unwrap_or(1.0);
        {
            let sq_sum: f64 = embed_grad_data.iter().map(|&x| (x as f64) * (x as f64)).sum();
            let grad_norm = sq_sum.sqrt() as f32;
            self.last_embed_grad_norm = grad_norm; // R-040: per-parameter-group tracking
            if grad_norm > embed_clip_norm {
                let scale = embed_clip_norm / grad_norm;
                for g in &mut embed_grad_data {
                    *g *= scale;
                }
            }
        }

        // KAIZEN-048: In-place scatter-add via grad_cell().borrow_mut().
        // Before: 3 × 128MB clones per step (grad() deep-copies Array1).
        // After: zero clones — mutate existing gradient buffer directly.
        let embed_weight = &mut self.model.embed_tokens.weight;
        let grad_cell = embed_weight.grad_cell();
        let mut grad_ref = grad_cell.borrow_mut();
        if grad_ref.is_none() {
            *grad_ref = Some(ndarray::Array1::zeros(embed_weight.len()));
        }
        if let Some(grad) = grad_ref.as_mut() {
            for (pos, &token_id) in input_ids.iter().enumerate() {
                let tid = token_id as usize;
                if tid < vocab_size {
                    let src = pos * hidden_size;
                    let dst = tid * hidden_size;
                    for h in 0..hidden_size {
                        grad[dst + h] += embed_grad_data[src + h];
                    }
                }
            }
        }
        Some(())
    }

    /// Apply optimizer step to CPU embedding and update metrics.
    ///
    /// GPU block optimizer steps now run interleaved with backward in `gpu_backward()`.
    /// LM head and final norm optimizer steps also run in `gpu_backward()`.
    /// This method handles only CPU embedding and bookkeeping.
    fn optimizer_step(&mut self) {
        // ALB-072: Gradients are no longer scaled by grad_scaler (loss_scale excludes
        // grad_scaler.scale()). All backward computation uses f32 — no fp16 underflow
        // risk. Skip unscaling; just update scaler as successful.
        self.grad_scaler.update(true);

        // ALB-079: Sync CPU embedding optimizer lr with cosine schedule
        self.embed_optimizer.set_lr(self.current_lr());
        // CPU optimizer step for embedding weight
        let mut embed_params = vec![&mut self.model.embed_tokens.weight];
        self.embed_optimizer.step_refs(&mut embed_params);

        self.step += 1;
        self.metrics.losses.push(self.accumulated_loss);
        self.metrics.increment_step();

        self.accumulated_loss = 0.0;
        self.accumulated_batches = 0;
    }

    /// Process a batch (forward + backward + optimizer step with accumulation).
    ///
    /// R-038: When `accumulation_steps > 1`, runs forward+backward without optimizer
    /// for each micro-batch, downloading per-block weight gradients to CPU-side
    /// `PerBlockGradientAccumulator`. After `accumulation_steps` batches, averages
    /// the accumulated gradients, uploads them to GPU, and runs a single optimizer step.
    ///
    /// When `accumulation_steps == 1` (default), runs forward+backward+optimizer
    /// immediately per sequence (original behavior).
    ///
    /// Returns average loss for the batch.
    pub fn train_batch(&mut self, batch: &LMBatch) -> f32 {
        if batch.batch_size == 0 {
            return 0.0;
        }

        let accumulating = self.grad_accum.is_some() || self.gpu_grad_accum.is_some();

        if self.accumulated_batches == 0 {
            // Zero embedding gradients at start of accumulation window
            self.embed_optimizer.zero_grad_refs(&mut vec![&mut self.model.embed_tokens.weight]);
        }

        let mut total_loss = 0.0;
        let mut valid_count = 0;

        for i in 0..batch.batch_size {
            let Some(input_ids) = batch.get_input(i) else {
                continue;
            };
            let Some(target_ids) = batch.get_target(i) else {
                continue;
            };

            // R-038: When accumulating, run backward without optimizer (accumulate_only=true).
            // Gradients are downloaded to CPU per-block accum buffers. Embedding grads are
            // scatter-added normally (they're already on CPU).
            if let Some(loss) = self.train_step_single(input_ids, target_ids, accumulating) {
                total_loss += loss;
                valid_count += 1;
                if accumulating {
                    if let Some(accum) = &mut self.gpu_grad_accum {
                        accum.accumulated_count += 1;
                    } else if let Some(accum) = &mut self.grad_accum {
                        accum.accumulated_count += 1;
                    }
                }
            }
        }

        let avg_loss = if valid_count > 0 { total_loss / valid_count as f32 } else { 0.0 };

        self.accumulated_loss += avg_loss / self.config.accumulation_steps as f32;
        self.accumulated_batches += 1;

        if self.accumulated_batches >= self.config.accumulation_steps {
            if accumulating {
                // ALB-091: Prefer GPU-resident accum path (zero D2H), fall back to CPU.
                if self.gpu_grad_accum.is_some() {
                    self.gpu_optimizer_from_gpu_accum();
                } else {
                    self.gpu_optimizer_from_accum();
                }
            }
            self.optimizer_step();
        }

        avg_loss
    }

    /// R-005: Evaluate a batch without backward pass or weight updates.
    /// Returns average cross-entropy loss, or 0.0 if no valid items.
    /// KAIZEN-050: Uses fused GPU cross-entropy (no logits D2H).
    pub fn eval_batch(&mut self, batch: &LMBatch) -> f32 {
        let hidden_size = self.config.model_config.hidden_size;
        let vocab_size = self.config.model_config.vocab_size;
        let max_sl = self.config.max_seq_len;
        let mut total_loss = 0.0;
        let mut valid_count = 0;
        for i in 0..batch.batch_size {
            if let Some(loss) = self.eval_single_sequence(batch, i, max_sl, hidden_size, vocab_size)
            {
                total_loss += loss;
                valid_count += 1;
            }
        }
        if valid_count > 0 {
            total_loss / valid_count as f32
        } else {
            0.0
        }
    }

    /// Evaluate a single sequence from a batch. Returns None if invalid.
    fn eval_single_sequence(
        &mut self,
        batch: &LMBatch,
        i: usize,
        max_sl: usize,
        hidden_size: usize,
        vocab_size: usize,
    ) -> Option<f32> {
        let input_ids = batch.get_input(i)?;
        let target_ids = batch.get_target(i)?;
        // Truncate to max_seq_len — GPU buffers are pre-allocated for this size
        let input_ids = if input_ids.len() > max_sl { &input_ids[..max_sl] } else { input_ids };
        let target_ids = if target_ids.len() > max_sl { &target_ids[..max_sl] } else { target_ids };
        let seq_len = input_ids.len();
        self.gpu_forward(input_ids, seq_len, hidden_size, vocab_size)?;
        let stream = self.cuda_trainer.stream();
        let scale = 1.0 / seq_len as f32;
        let loss = fused_cross_entropy_cuda(
            &mut self.gpu_training.logits_buf,
            target_ids,
            seq_len as u32,
            vocab_size as u32,
            scale,
            stream,
        )
        .ok()?;
        if loss.is_finite() {
            Some(loss)
        } else {
            None
        }
    }

    /// Train for one epoch over batches.
    pub fn train_epoch(&mut self, batches: &[LMBatch]) -> f32 {
        self.train_epoch_with_callback(batches, |_, _, _| {})
    }

    /// Train for one epoch with a per-step callback.
    ///
    /// Stops early if `max_steps` is set and reached.
    pub fn train_epoch_with_callback<F>(&mut self, batches: &[LMBatch], mut on_batch: F) -> f32
    where
        F: FnMut(usize, f32, &Self),
    {
        if batches.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let mut batches_processed = 0;

        for (i, batch) in batches.iter().enumerate() {
            if let Some(max) = self.config.max_steps {
                if self.step >= max {
                    break;
                }
            }

            let batch_loss = self.train_batch(batch);
            total_loss += batch_loss;
            batches_processed += 1;
            on_batch(i, batch_loss, self);
        }

        // KAIZEN-047: Print profiler summary at end of epoch
        if self.profiler.is_enabled() && self.profiler.step_count() > 0 {
            self.profiler.print_report();
        }

        total_loss / batches_processed.max(1) as f32
    }

    // --- DDP (data-parallel) support methods ---

    /// Ensure the per-block gradient accumulator exists.
    ///
    /// For DDP, we always need accumulation buffers (even with accumulation_steps=1)
    /// because gradients must be downloaded to CPU for AllReduce before optimizer step.
    pub(crate) fn ensure_grad_accum(&mut self) {
        if self.grad_accum.is_some() {
            return;
        }
        let mc = &self.config.model_config;
        let hidden_size = mc.hidden_size;
        let kv_hidden = mc.num_kv_heads * mc.head_dim();
        let block_sizes = super::grad_accumulator::PerBlockGradientAccumulator::compute_block_sizes(
            hidden_size,
            kv_hidden,
            mc.intermediate_size,
        );
        self.grad_accum = Some(super::grad_accumulator::PerBlockGradientAccumulator::new(
            self.cuda_blocks.len(),
            block_sizes,
            mc.vocab_size,
            hidden_size,
        ));
    }

    /// Forward + backward for one batch, always accumulating (no optimizer step).
    ///
    /// Used by `DistributedCudaTrainer` to compute local gradients before AllReduce.
    /// Returns average loss for the batch.
    pub(crate) fn forward_backward_batch(&mut self, batch: &LMBatch) -> f32 {
        if batch.batch_size == 0 {
            return 0.0;
        }

        if self.accumulated_batches == 0 {
            self.embed_optimizer.zero_grad_refs(&mut vec![&mut self.model.embed_tokens.weight]);
        }

        let mut total_loss = 0.0;
        let mut valid_count = 0;

        for i in 0..batch.batch_size {
            let Some(input_ids) = batch.get_input(i) else { continue };
            let Some(target_ids) = batch.get_target(i) else { continue };

            // Always accumulate_only=true: gradients go to CPU accum buffers
            if let Some(loss) = self.train_step_single(input_ids, target_ids, true) {
                total_loss += loss;
                valid_count += 1;
                if let Some(accum) = &mut self.grad_accum {
                    accum.accumulated_count += 1;
                }
            }
        }

        if valid_count > 0 {
            total_loss / valid_count as f32
        } else {
            0.0
        }
    }

    /// Apply DDP-averaged gradients: upload to GPU and run optimizer step.
    ///
    /// Called after AllReduce has written averaged gradients into the grad_accum.
    /// Runs gpu_optimizer_from_accum() for blocks + LM head + final norm,
    /// then optimizer_step() for embedding.
    pub(crate) fn apply_ddp_gradients(&mut self) {
        self.accumulated_loss = 0.0;
        self.accumulated_batches = 0;
        self.gpu_optimizer_from_accum();
        self.optimizer_step();
    }

    /// Get a reference to the gradient accumulator (for DDP AllReduce).
    pub(crate) fn grad_accum_ref(
        &self,
    ) -> Option<&super::grad_accumulator::PerBlockGradientAccumulator> {
        self.grad_accum.as_ref()
    }

    /// Get a mutable reference to the gradient accumulator (for DDP AllReduce).
    pub(crate) fn grad_accum_mut(
        &mut self,
    ) -> Option<&mut super::grad_accumulator::PerBlockGradientAccumulator> {
        self.grad_accum.as_mut()
    }

    /// Get the training config.
    pub(crate) fn config(&self) -> &TransformerTrainConfig {
        &self.config
    }

    /// Get CPU embedding gradient as flat Vec for AllReduce.
    pub(crate) fn embed_grad_vec(&self) -> Option<Vec<f32>> {
        self.model.embed_tokens.weight.grad().map(|g| g.to_vec())
    }

    /// Set CPU embedding gradient from AllReduced flat Vec.
    pub(crate) fn set_embed_grad(&mut self, grad: Vec<f32>) {
        self.model.embed_tokens.weight.set_grad(ndarray::Array1::from(grad));
    }

    /// Returns true if max_steps has been reached.
    pub fn reached_max_steps(&self) -> bool {
        self.config.max_steps.map_or(false, |max| self.step >= max)
    }

    /// Get current step count.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Get current learning rate (warmup + cosine decay).
    ///
    /// ALB-079: Phase 1 = linear warmup (0 → lr_max), Phase 2 = cosine decay
    /// (lr_max → 0) over remaining steps. Requires `max_steps` for decay;
    /// without it, falls back to constant lr after warmup.
    pub fn current_lr(&self) -> f32 {
        let base_lr = self.config.lr;
        if self.step < self.config.warmup_steps {
            // Phase 1: Linear warmup
            base_lr * (self.step as f32 / self.config.warmup_steps.max(1) as f32)
        } else if let Some(max_steps) = self.config.max_steps {
            // Phase 2: Cosine decay from lr_max to 0
            let decay_steps = max_steps.saturating_sub(self.config.warmup_steps);
            if decay_steps == 0 {
                return base_lr;
            }
            let decay_step = self.step - self.config.warmup_steps;
            let progress = (decay_step as f32 / decay_steps as f32).min(1.0);
            0.5 * base_lr * (1.0 + (std::f32::consts::PI * progress).cos())
        } else {
            // No max_steps: constant lr (legacy behavior)
            base_lr
        }
    }

    /// KAIZEN-047: Enable step profiling with a report every `interval` steps.
    ///
    /// When enabled, prints a table of wall-clock timings per training phase
    /// every `interval` training steps. Use interval=0 for manual-only reporting.
    ///
    /// # Contract (C-STEPPROF-001)
    ///
    /// - No additional GPU synchronization points (relies on existing syncs)
    /// - Overhead: ~11 `Instant::now()` calls per step (~1µs total on Linux)
    /// - Timings include async dispatch overhead (not pure kernel time)
    pub fn enable_profiler(&mut self, interval: usize) {
        self.profiler = StepProfiler::new(true, interval);
    }

    /// Print the profiler report (if profiling is enabled).
    pub fn print_profiler_report(&self) {
        self.profiler.print_report();
    }

    /// R-004: Get last observed gradient L2 norm (LM head proxy).
    pub fn last_grad_norm(&self) -> f32 {
        self.last_grad_norm
    }

    /// R-040: Get per-parameter-group gradient norms.
    /// Returns (lm_head_grad_norm, embed_grad_norm).
    pub fn param_grad_norms(&self) -> (f32, f32) {
        (self.last_grad_norm, self.last_embed_grad_norm)
    }

    /// R-012: Get total trainable parameter count for MFU calculation.
    pub fn num_params(&self) -> usize {
        self.model.parameters().iter().map(|t| t.len()).sum()
    }

    /// R-013: Query GPU memory usage (used_mb, total_mb).
    pub fn gpu_memory_mb(&self) -> (u64, u64) {
        match self.cuda_trainer.context().memory_info() {
            Ok((free, total)) => {
                let total_mb = (total / (1024 * 1024)) as u64;
                let used_mb = ((total - free) / (1024 * 1024)) as u64;
                (used_mb, total_mb)
            }
            Err(_) => (0, 0),
        }
    }

    /// Sync all GPU weights back to CPU model.
    ///
    /// # Contract (C-SYNCWT-001)
    ///
    /// Must be called before save or any CPU model access after training.
    pub fn sync_weights_to_cpu(&mut self) {
        for (layer_idx, block) in self.cuda_blocks.iter().enumerate() {
            if let Ok(weights) = block.download_weights() {
                let layer = &mut self.model.layers[layer_idx];

                layer.self_attn.w_q = Tensor::from_vec(weights.w_q, false);
                layer.self_attn.w_k = Tensor::from_vec(weights.w_k, false);
                layer.self_attn.w_v = Tensor::from_vec(weights.w_v, false);
                layer.self_attn.w_o = Tensor::from_vec(weights.w_o, false);

                layer.ffn.w_gate = Tensor::from_vec(weights.w_gate, false);
                layer.ffn.w_up = Tensor::from_vec(weights.w_up, false);
                layer.ffn.w_down = Tensor::from_vec(weights.w_down, false);

                layer.input_norm.weight = Tensor::from_vec(weights.input_norm_weight, false);
                layer.post_attn_norm.weight =
                    Tensor::from_vec(weights.post_attn_norm_weight, false);
            }
        }

        // Sync final norm weight
        if let Ok(norm_data) = self.cuda_trainer.download(&self.gpu_training.final_norm_weight) {
            self.model.norm.weight = Tensor::from_vec(norm_data, false);
        }

        // Sync LM head weight
        if let Ok(lm_data) = self.cuda_trainer.download(&self.lm_head_weight_gpu) {
            if self.model.lm_head.is_some() {
                self.model.lm_head = Some(Tensor::from_vec(lm_data, false));
            }
            // If tied weights, embedding was updated by CPU optimizer — don't overwrite
        }
    }

    /// Get reference to model (syncs weights first).
    pub fn model(&self) -> &Transformer {
        &self.model
    }

    /// Get mutable reference to model.
    pub fn model_mut(&mut self) -> &mut Transformer {
        &mut self.model
    }

    /// Check if using mixed precision.
    pub fn is_mixed_precision(&self) -> bool {
        self.config.precision_config.is_mixed()
    }

    /// Get the gradient scaler (R-002: loss scaling for mixed precision).
    pub fn grad_scaler(&self) -> &GradScaler {
        &self.grad_scaler
    }

    /// Check if using gradient checkpointing.
    pub fn is_checkpointing(&self) -> bool {
        self.config.checkpoint_config.enabled
    }

    /// Save model weights (syncs GPU→CPU first).
    pub fn save(
        &mut self,
        path: impl AsRef<std::path::Path>,
        name: &str,
        architecture: &str,
    ) -> crate::Result<()> {
        self.sync_weights_to_cpu();

        let model_params = self.model.parameters();
        let num_total = model_params.len();
        let num_layers = (num_total - 2) / 9;
        let mut names: Vec<String> = Vec::with_capacity(num_total);

        names.push("model.embed_tokens.weight".to_string());
        names.push("model.norm.weight".to_string());

        for layer in 0..num_layers {
            names.push(format!("model.layers.{layer}.input_layernorm.weight"));
            names.push(format!("model.layers.{layer}.post_attention_layernorm.weight"));
            names.push(format!("model.layers.{layer}.self_attn.q_proj.weight"));
            names.push(format!("model.layers.{layer}.self_attn.k_proj.weight"));
            names.push(format!("model.layers.{layer}.self_attn.v_proj.weight"));
            names.push(format!("model.layers.{layer}.self_attn.o_proj.weight"));
            names.push(format!("model.layers.{layer}.mlp.gate_proj.weight"));
            names.push(format!("model.layers.{layer}.mlp.up_proj.weight"));
            names.push(format!("model.layers.{layer}.mlp.down_proj.weight"));
        }

        if names.len() < num_total {
            names.push("lm_head.weight".to_string());
        }

        let params: Vec<(String, Tensor)> = names
            .into_iter()
            .zip(model_params)
            .map(|(name, tensor)| (name, tensor.clone()))
            .collect();

        let metadata = ModelMetadata::new(name, architecture);
        let model = Model::new(metadata, params);
        let config = SaveConfig::new(ModelFormat::SafeTensors);

        save_model(&model, path, &config)
    }

    /// R-011: Prepare checkpoint data for async save.
    /// Syncs GPU weights to CPU and snapshots tensor data as Send-able Vec<f32>.
    /// Returns a closure that writes the checkpoint file from another thread.
    pub fn prepare_async_save(
        &mut self,
        name: &str,
        architecture: &str,
    ) -> Box<dyn FnOnce(&std::path::Path) -> crate::Result<()> + Send> {
        self.sync_weights_to_cpu();

        let model_params = self.model.parameters();
        let num_total = model_params.len();
        let num_layers = (num_total - 2) / 9;
        let mut names: Vec<String> = Vec::with_capacity(num_total);

        names.push("model.embed_tokens.weight".to_string());
        names.push("model.norm.weight".to_string());
        for layer in 0..num_layers {
            names.push(format!("model.layers.{layer}.input_layernorm.weight"));
            names.push(format!("model.layers.{layer}.post_attention_layernorm.weight"));
            names.push(format!("model.layers.{layer}.self_attn.q_proj.weight"));
            names.push(format!("model.layers.{layer}.self_attn.k_proj.weight"));
            names.push(format!("model.layers.{layer}.self_attn.v_proj.weight"));
            names.push(format!("model.layers.{layer}.self_attn.o_proj.weight"));
            names.push(format!("model.layers.{layer}.mlp.gate_proj.weight"));
            names.push(format!("model.layers.{layer}.mlp.up_proj.weight"));
            names.push(format!("model.layers.{layer}.mlp.down_proj.weight"));
        }
        if names.len() < num_total {
            names.push("lm_head.weight".to_string());
        }

        // Snapshot raw Vec<f32> data — Send-safe (Tensor contains Rc, not Send)
        let param_data: Vec<(String, Vec<f32>)> =
            names.into_iter().zip(model_params).map(|(n, t)| (n, t.data().to_vec())).collect();

        let name = name.to_string();
        let architecture = architecture.to_string();

        Box::new(move |path: &std::path::Path| {
            let params: Vec<(String, Tensor)> =
                param_data.into_iter().map(|(n, d)| (n, Tensor::from_vec(d, false))).collect();
            let metadata = ModelMetadata::new(&name, &architecture);
            let model = Model::new(metadata, params);
            let config = SaveConfig::new(ModelFormat::SafeTensors);
            save_model(&model, path, &config)
        })
    }

    /// GPU device name.
    pub fn gpu_name(&self) -> String {
        self.cuda_trainer.device_name()
    }

    /// R-001: Save CPU embedding optimizer state (m/v buffers + step counter).
    ///
    /// Writes `optimizer_state.json` to the given directory. GPU block optimizer
    /// states remain on-device (D2H for 20 buffers × N blocks is deferred).
    pub fn save_optimizer_state(&self, dir: &std::path::Path) -> crate::Result<()> {
        let path = dir.join("optimizer_state.json");
        let m_data: Vec<Option<Vec<f32>>> = self
            .embed_optimizer
            .first_moments()
            .iter()
            .map(|opt| opt.as_ref().map(|a| a.to_vec()))
            .collect();
        let v_data: Vec<Option<Vec<f32>>> = self
            .embed_optimizer
            .second_moments()
            .iter()
            .map(|opt| opt.as_ref().map(|a| a.to_vec()))
            .collect();
        let state = serde_json::json!({
            "type": "adamw_cpu_embed",
            "step": self.embed_optimizer.step_count(),
            "m": m_data,
            "v": v_data,
        });
        let json_str = serde_json::to_string(&state).map_err(|e| {
            crate::error::Error::ConfigError(format!("serialize optimizer state: {}", e))
        })?;
        std::fs::write(&path, json_str).map_err(|e| {
            crate::error::Error::ConfigError(format!("write optimizer state: {}", e))
        })?;
        Ok(())
    }

    /// R-001: Load CPU embedding optimizer state from `optimizer_state.json`.
    ///
    /// Returns true if state was loaded, false if file doesn't exist.
    pub fn load_optimizer_state(&mut self, dir: &std::path::Path) -> bool {
        let path = dir.join("optimizer_state.json");
        let data = match std::fs::read_to_string(&path) {
            Ok(d) => d,
            Err(_) => return false,
        };
        let state: serde_json::Value = match serde_json::from_str(&data) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if let Some(step) = state["step"].as_u64() {
            self.embed_optimizer.set_step_count(step);
        }
        restore_moment_buffers(&state["m"], |idx, arr| {
            self.embed_optimizer.set_first_moment(idx, arr);
        });
        restore_moment_buffers(&state["v"], |idx, arr| {
            self.embed_optimizer.set_second_moment(idx, arr);
        });
        true
    }
}

/// Parse a JSON array of moment buffers and apply each via callback.
#[cfg(feature = "cuda")]
fn restore_moment_buffers(
    json_arr: &serde_json::Value,
    mut set_fn: impl FnMut(usize, ndarray::Array1<f32>),
) {
    let Some(arr) = json_arr.as_array() else { return };
    for (idx, val) in arr.iter().enumerate() {
        let Some(inner) = val.as_array() else { continue };
        let floats: Vec<f32> = inner.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
        if !floats.is_empty() {
            set_fn(idx, ndarray::Array1::from_vec(floats));
        }
    }
}

// ── Non-CUDA stub ──

#[cfg(not(feature = "cuda"))]
pub struct CudaTransformerTrainer;

#[cfg(not(feature = "cuda"))]
impl CudaTransformerTrainer {
    pub fn new(_config: super::config::TransformerTrainConfig) -> crate::Result<Self> {
        Err(crate::error::Error::ConfigError(
            "CUDA not available (compiled without cuda feature)".into(),
        ))
    }

    pub fn with_model(
        _model: crate::transformer::Transformer,
        _config: super::config::TransformerTrainConfig,
    ) -> crate::Result<Self> {
        Err(crate::error::Error::ConfigError(
            "CUDA not available (compiled without cuda feature)".into(),
        ))
    }

    pub fn gpu_name(&self) -> String {
        unreachable!("CudaTransformerTrainer stub should never be instantiated")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_cuda_trainer_stub_returns_error() {
        use super::super::config::TransformerTrainConfig;
        use crate::transformer::TransformerConfig;

        let mc = TransformerConfig::tiny();
        let config = TransformerTrainConfig::new(mc);
        let result = super::CudaTransformerTrainer::new(config);
        assert!(result.is_err());
    }
}
