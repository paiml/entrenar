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
//! ├── loss_fn: CausalLMLoss              (CPU cross-entropy)
//! └── config: TransformerTrainConfig
//! ```
//!
//! # Transfer budget (C-GPUTRAIN-002)
//!
//! Exactly 3 PCIe transfers per training step:
//! 1. H2D: hidden states after embedding (seq×H×4 bytes)
//! 2. D2H: logits for cross-entropy (seq×V×4 bytes)
//! 3. H2D: grad_logits from cross-entropy (seq×V×4 bytes)

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaStream, GpuBuffer};

#[cfg(feature = "cuda")]
use crate::autograd::cuda_backward::{gemm_backward_a, gemm_backward_b, rms_norm_backward};
#[cfg(feature = "cuda")]
use crate::autograd::cuda_forward::{gemm_forward, pre_warm_forward_kernels, rms_norm_forward};
#[cfg(feature = "cuda")]
use crate::autograd::cuda_optim::{adamw_step_cuda, gradient_clip_cuda};
#[cfg(feature = "cuda")]
use crate::autograd::cuda_training::{cuda_training_available, CudaTrainer};
#[cfg(feature = "cuda")]
use crate::autograd::Tensor;
#[cfg(feature = "cuda")]
use crate::io::{save_model, Model, ModelFormat, ModelMetadata, SaveConfig};
#[cfg(feature = "cuda")]
use crate::optim::{AdamW, Optimizer};
#[cfg(feature = "cuda")]
use crate::train::{CausalLMLoss, LossFn, MetricsTracker};
#[cfg(feature = "cuda")]
use crate::transformer::{
    BlockWeights, CudaGradWorkspace, CudaTransformerBlock, GpuBlockOptimizerState, Transformer,
};

#[cfg(feature = "cuda")]
use super::batch::LMBatch;
#[cfg(feature = "cuda")]
use super::config::TransformerTrainConfig;

/// Compute exact gradient L2 norm of the shared workspace (downloads all buffers to CPU).
///
/// Free function to avoid borrow conflicts with `&mut self`.
/// For 350M model, downloads ~58 MB per block (~1.8ms on PCIe 4.0 x16).
#[cfg(feature = "cuda")]
fn compute_workspace_clip_scale(ws: &CudaGradWorkspace, max_norm: f32) -> (f32, f32) {
    fn exact_sq_norm(buf: &GpuBuffer<f32>) -> f64 {
        let mut host = vec![0.0f32; buf.len()];
        if buf.copy_to_host_at(&mut host, 0).is_err() {
            return 0.0;
        }
        host.iter().map(|&x| (x as f64) * (x as f64)).sum()
    }

    let total_sq = exact_sq_norm(&ws.grad_w_q)
        + exact_sq_norm(&ws.grad_w_k)
        + exact_sq_norm(&ws.grad_w_v)
        + exact_sq_norm(&ws.grad_w_o)
        + exact_sq_norm(&ws.grad_gate)
        + exact_sq_norm(&ws.grad_up)
        + exact_sq_norm(&ws.grad_down)
        + exact_sq_norm(&ws.grad_input_norm)
        + exact_sq_norm(&ws.grad_post_attn_norm);

    let grad_norm = total_sq.sqrt() as f32;
    let scale = if grad_norm > max_norm {
        max_norm / grad_norm
    } else {
        1.0
    };
    (scale, grad_norm)
}

/// Clip all gradient buffers in the shared workspace using per-block L2 norm estimate.
///
/// R-004: Returns pre-clip gradient L2 norm for observability logging.
#[cfg(feature = "cuda")]
fn clip_workspace_gradients(ws: &mut CudaGradWorkspace, max_norm: f32, stream: &CudaStream) -> f32 {
    let (scale, grad_norm) = compute_workspace_clip_scale(ws, max_norm);
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

/// R-004: Compute gradient L2 norm without clipping (for observability only).
///
/// Used when grad_clip is disabled to still report gradient norms.
#[cfg(feature = "cuda")]
fn compute_workspace_grad_norm(ws: &CudaGradWorkspace) -> f32 {
    let (_, norm) = compute_workspace_clip_scale(ws, f32::MAX);
    norm
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
    /// CPU loss function (cross-entropy stays on CPU)
    loss_fn: CausalLMLoss,
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

    /// Create a GPU-resident trainer from an existing model.
    ///
    /// # Errors
    ///
    /// Returns `Err` if CUDA initialization fails.
    pub fn with_model(model: Transformer, config: TransformerTrainConfig) -> crate::Result<Self> {
        if !cuda_training_available() {
            return Err(crate::error::Error::ConfigError(
                "CUDA not available".into(),
            ));
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
        .map_err(|e| {
            crate::error::Error::ConfigError(format!("Kernel pre-warm failed: {e:?}"))
        })?;

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

        let mut layer_inputs = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layer_inputs.push(
                GpuBuffer::new(&ctx, buf_size).map_err(|e| {
                    crate::error::Error::ConfigError(format!("Layer input alloc failed: {e:?}"))
                })?,
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
        let lm_head_data = model
            .lm_head
            .as_ref()
            .unwrap_or(&model.embed_tokens.weight)
            .data();
        let lm_head_slice = lm_head_data.as_slice().expect("contiguous");
        let lm_head_weight_gpu = GpuBuffer::from_host(&ctx, lm_head_slice).map_err(|e| {
            crate::error::Error::ConfigError(format!("LM head upload failed: {e:?}"))
        })?;
        let lm_head_grad_gpu = GpuBuffer::new(&ctx, vocab_size * hidden_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("LM head grad alloc failed: {e:?}"))
        })?;
        // CRITICAL: Must zero-initialize m/v buffers. GpuBuffer::new() does NOT
        // zero memory (cuMemAlloc returns uninitialized VRAM).
        let lm_head_m = GpuBuffer::from_host(&ctx, &vec![0.0f32; vocab_size * hidden_size]).map_err(|e| {
            crate::error::Error::ConfigError(format!("LM head m alloc failed: {e:?}"))
        })?;
        let lm_head_v = GpuBuffer::from_host(&ctx, &vec![0.0f32; vocab_size * hidden_size]).map_err(|e| {
            crate::error::Error::ConfigError(format!("LM head v alloc failed: {e:?}"))
        })?;

        // Final norm optimizer states
        let final_norm_m = GpuBuffer::from_host(&ctx, &vec![0.0f32; hidden_size]).map_err(|e| {
            crate::error::Error::ConfigError(format!("Final norm m alloc failed: {e:?}"))
        })?;
        let final_norm_v = GpuBuffer::from_host(&ctx, &vec![0.0f32; hidden_size]).map_err(|e| {
            crate::error::Error::ConfigError(format!("Final norm v alloc failed: {e:?}"))
        })?;

        // Sync to ensure all uploads completed
        stream.synchronize().map_err(|e| {
            crate::error::Error::ConfigError(format!("Stream sync failed: {e:?}"))
        })?;

        println!(
            "  ✓ GPU training state allocated (LM head: {:.1} MB)",
            (vocab_size * hidden_size * 4) as f64 / 1e6
        );

        let loss_fn = CausalLMLoss::new(vocab_size);
        // C-EMBED-GRAD-001: CPU optimizer must match YAML hyperparams (not defaults)
        let embed_optimizer = AdamW::new(
            config.lr,
            config.beta1,
            config.beta2,
            1e-8,
            config.weight_decay,
        );

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
            loss_fn,
            embed_optimizer,
            config,
            metrics: MetricsTracker::new(),
            step: 0,
            accumulated_loss: 0.0,
            accumulated_batches: 0,
            last_grad_norm: 0.0,
        })
    }

    /// Run one forward+backward+optimizer step for a single sequence.
    ///
    /// # Contract (C-GPUSTEP-001)
    ///
    /// - Precondition: `input_ids.len() == target_ids.len() <= max_seq_len`
    /// - Postcondition: All GPU weights updated, embedding gradients set
    /// - Transfer count: exactly 3 PCIe transfers
    fn train_step_single(
        &mut self,
        input_ids: &[u32],
        target_ids: &[u32],
    ) -> Option<f32> {
        let hidden_size = self.config.model_config.hidden_size;
        let vocab_size = self.config.model_config.vocab_size;

        // Truncate to max_seq_len — GPU buffers are pre-allocated for this size
        let max_sl = self.config.max_seq_len;
        let input_ids = if input_ids.len() > max_sl { &input_ids[..max_sl] } else { input_ids };
        let target_ids = if target_ids.len() > max_sl { &target_ids[..max_sl] } else { target_ids };
        let seq_len = input_ids.len();

        // Steps 1-6: GPU forward pass → returns logits on CPU
        let logits_data = self.gpu_forward(input_ids, seq_len, hidden_size, vocab_size)?;

        // Step 7: Cross-entropy loss + gradient (CPU)
        let (loss_val, grad_logits) =
            self.cpu_loss_and_grad(&logits_data, target_ids, seq_len, vocab_size)?;

        // Steps 8-11: GPU backward pass
        let grad_output_is_a =
            self.gpu_backward(&grad_logits, seq_len, hidden_size, vocab_size)?;

        // Step 12: Embedding backward (CPU)
        self.embed_backward(input_ids, seq_len, hidden_size, vocab_size, grad_output_is_a)?;

        Some(loss_val)
    }

    /// GPU forward pass: embed → blocks → norm → LM head → download logits.
    ///
    /// Transfers: 1 H2D (hidden states), 1 D2H (logits).
    #[allow(unsafe_code)]
    fn gpu_forward(
        &mut self,
        input_ids: &[u32],
        seq_len: usize,
        hidden_size: usize,
        vocab_size: usize,
    ) -> Option<Vec<f32>> {
        let stream = self.cuda_trainer.stream();

        // Embedding lookup (CPU)
        let hidden = self.model.embed_tokens.forward(input_ids);
        let hidden_slice = hidden.data().as_slice()?;

        // Upload hidden states to GPU (Transfer 1: H2D)
        // Pad to max_seq_len so D2D copies to pre-allocated layer_inputs match.
        let max_buf_size = self.config.max_seq_len * hidden_size;
        let mut padded_hidden = vec![0.0f32; max_buf_size];
        padded_hidden[..hidden_slice.len()].copy_from_slice(hidden_slice);
        let mut gpu_input = self.cuda_trainer.upload(&padded_hidden).ok()?;
        let mut gpu_output = self.cuda_trainer.zeros(max_buf_size).ok()?;

        // Forward through CUDA blocks, saving layer inputs
        for (i, block) in self.cuda_blocks.iter_mut().enumerate() {
            // SAFETY: Both buffers are valid GPU allocations with matching max_seq_len size.
            // Copy completes before block.forward() reads from gpu_input (same stream ordering).
            unsafe {
                self.gpu_training.layer_inputs[i]
                    .copy_from_buffer_async(&gpu_input, stream)
                    .ok()?;
            }
            block.forward(&gpu_input, &mut gpu_output, seq_len, stream).ok()?;
            std::mem::swap(&mut gpu_input, &mut gpu_output);
        }

        // Save blocks output for final norm backward
        // SAFETY: Disjoint GPU buffers with matching max_seq_len sizes.
        unsafe {
            self.gpu_training
                .blocks_output
                .copy_from_buffer_async(&gpu_input, stream)
                .ok()?;
        }

        // Final RMSNorm forward (GPU)
        rms_norm_forward(
            &gpu_input,
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

        // Download logits (Transfer 2: D2H)
        // logits_buf is max_seq_len*vocab_size; only download seq_len*vocab_size.
        stream.synchronize().ok()?;
        let mut buf = vec![0.0f32; seq_len * vocab_size];
        self.gpu_training.logits_buf.copy_to_host_at(&mut buf, 0).ok()?;

        // NaN guard: bail if logits contain non-finite values
        if buf.iter().any(|v| !v.is_finite()) {
            return None;
        }

        Some(buf)
    }

    /// Compute cross-entropy loss and gradient on CPU.
    ///
    /// Returns (loss_value, grad_logits).
    fn cpu_loss_and_grad(
        &self,
        logits_data: &[f32],
        target_ids: &[u32],
        seq_len: usize,
        vocab_size: usize,
    ) -> Option<(f32, Vec<f32>)> {
        let logits_tensor = Tensor::from_vec(logits_data.to_vec(), false);
        let targets_tensor =
            Tensor::from_vec(target_ids.iter().map(|&id| id as f32).collect(), false);
        let loss = self.loss_fn.forward(&logits_tensor, &targets_tensor);
        let loss_val = loss.data()[0];

        // Softmax backward: grad = (softmax(logits) - one_hot(target)) / seq_len
        let mut grad_logits = vec![0.0f32; seq_len * vocab_size];
        let scale = 1.0 / seq_len as f32;

        for pos in 0..seq_len {
            let start = pos * vocab_size;
            let logits_pos = &logits_data[start..start + vocab_size];

            let max = logits_pos.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_vals: Vec<f32> = logits_pos.iter().map(|&x| (x - max).exp()).collect();
            let sum: f32 = exp_vals.iter().sum();
            let target_idx = target_ids[pos] as usize;

            if target_idx < vocab_size {
                for (i, &e) in exp_vals.iter().enumerate() {
                    let prob = e / sum;
                    grad_logits[start + i] =
                        if i == target_idx { (prob - 1.0) * scale } else { prob * scale };
                }
            }
        }

        // Scale by 1/accumulation_steps for gradient accumulation
        if self.config.accumulation_steps > 1 {
            let accum_scale = 1.0 / self.config.accumulation_steps as f32;
            for g in &mut grad_logits {
                *g *= accum_scale;
            }
        }

        Some((loss_val, grad_logits))
    }

    /// GPU backward pass with interleaved per-block optimizer step.
    ///
    /// Each block's backward writes weight gradients to the shared `CudaGradWorkspace`.
    /// Since `gemm_backward_b` overwrites (not accumulates), we must run each block's
    /// optimizer step immediately after its backward, before the next block overwrites
    /// the workspace. This also enables per-block gradient clipping.
    ///
    /// Returns `grad_output_is_a` flag for embedding backward.
    /// Transfer: 1 H2D (grad_logits).
    #[allow(unsafe_code)]
    fn gpu_backward(
        &mut self,
        grad_logits: &[f32],
        seq_len: usize,
        hidden_size: usize,
        vocab_size: usize,
    ) -> Option<bool> {
        let stream = self.cuda_trainer.stream();
        let max_grad_norm = self.config.base.max_grad_norm;
        let lr = self.current_lr();
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let weight_decay = self.config.weight_decay;

        // Upload grad_logits (Transfer 3: H2D)
        let grad_logits_gpu = self.cuda_trainer.upload(grad_logits).ok()?;

        // LM head GEMM backward
        gemm_backward_a(
            &grad_logits_gpu,
            &self.lm_head_weight_gpu,
            &mut self.gpu_training.lm_head_grad_hidden,
            seq_len as u32, hidden_size as u32, vocab_size as u32,
            stream,
        ).ok()?;

        gemm_backward_b(
            &self.gpu_training.norm_output,
            &grad_logits_gpu,
            &mut self.lm_head_grad_gpu,
            seq_len as u32, hidden_size as u32, vocab_size as u32,
            stream,
        ).ok()?;

        // Clip LM head weight gradient
        // SYNC: cuMemcpyDtoH doesn't synchronize with CU_STREAM_NON_BLOCKING streams.
        // Without this, compute_clip_scale reads stale GPU buffers → garbage clip scale.
        // Root cause of ALB-065 silent crash.
        if let Some(max_norm) = max_grad_norm {
            stream.synchronize().ok()?;
            let (scale, norm) = Self::compute_clip_scale_with_norm(&self.lm_head_grad_gpu, max_norm);
            self.last_grad_norm = norm; // R-004: capture for observability
            let n = self.lm_head_grad_gpu.len() as u32;
            let _ = gradient_clip_cuda(&mut self.lm_head_grad_gpu, scale, n, stream);
        }

        // Final RMSNorm backward
        rms_norm_backward(
            &self.gpu_training.blocks_output,
            &self.gpu_training.final_norm_weight,
            &self.gpu_training.lm_head_grad_hidden,
            &mut self.gpu_training.grad_buf_a,
            &mut self.gpu_training.grad_final_norm_weight,
            seq_len as u32, hidden_size as u32, 1e-5_f32, stream,
        ).ok()?;

        // Clip final norm weight gradient (sync for same reason as LM head clip)
        if let Some(max_norm) = max_grad_norm {
            stream.synchronize().ok()?;
            let (scale, _) = Self::compute_clip_scale_with_norm(&self.gpu_training.grad_final_norm_weight, max_norm);
            let n = self.gpu_training.grad_final_norm_weight.len() as u32;
            let _ = gradient_clip_cuda(&mut self.gpu_training.grad_final_norm_weight, scale, n, stream);
        }

        // Increment optimizer step counter (once per optimizer step, not per block)
        self.gpu_training.step += 1;
        let step = self.gpu_training.step;

        // LM head optimizer step (before blocks overwrite anything)
        let n_lm = self.lm_head_weight_gpu.len() as u32;
        let _ = adamw_step_cuda(
            &mut self.lm_head_weight_gpu,
            &self.lm_head_grad_gpu,
            &mut self.lm_head_m,
            &mut self.lm_head_v,
            lr, beta1, beta2, 1e-8, weight_decay, step, n_lm, stream,
        );

        // Final norm optimizer step
        let n_norm = self.gpu_training.final_norm_weight.len() as u32;
        let _ = adamw_step_cuda(
            &mut self.gpu_training.final_norm_weight,
            &self.gpu_training.grad_final_norm_weight,
            &mut self.final_norm_m,
            &mut self.final_norm_v,
            lr, beta1, beta2, 1e-8, weight_decay, step, n_norm, stream,
        );

        // Backward through blocks in reverse, with interleaved clip + optimizer.
        //
        // Each block's backward writes weight gradients to the shared CudaGradWorkspace.
        // We immediately clip and optimize before the next block overwrites the workspace.
        //
        // SAFETY: grad_buf_a and grad_buf_b are disjoint fields. Raw pointers
        // allow alternating read/write without violating aliasing rules.
        let grad_a_ptr: *mut GpuBuffer<f32> = &mut self.gpu_training.grad_buf_a;
        let grad_b_ptr: *mut GpuBuffer<f32> = &mut self.gpu_training.grad_buf_b;
        let mut grad_output_is_a = true;

        for layer_idx in (0..self.cuda_blocks.len()).rev() {
            let (grad_output, grad_input) = unsafe {
                if grad_output_is_a {
                    (&*grad_a_ptr, &mut *grad_b_ptr)
                } else {
                    (&*grad_b_ptr, &mut *grad_a_ptr)
                }
            };
            self.cuda_blocks[layer_idx].backward(
                &self.gpu_training.layer_inputs[layer_idx],
                grad_output, grad_input, seq_len, stream,
                &mut self.cuda_grad_workspace,
            ).ok()?;

            // Per-block gradient clipping DISABLED (ALB-067: CPU-side L2 norm bottleneck).
            //
            // compute_workspace_clip_scale() downloads 9 gradient buffers per block to
            // CPU for L2 norm computation. For 24 blocks × 4 sequences/batch: 864 D2H
            // transfers (~1.4 GB/step) + heap allocation + f64 summation. This pegs CPU
            // at 100% and GPU at 7%, making each step take minutes instead of seconds.
            //
            // SAFETY: Per-block weight gradients are bounded by block activations and
            // regularized by AdamW second moments. LM head + final norm weight clipping
            // is preserved (lines 653-676). Activation gradient clipping (C-EMBED-GRAD-001)
            // in embed_backward() is preserved — this is the critical safety net.
            //
            // TODO(ALB-067): Re-enable when GPU-side squared norm reduction is implemented
            // in trueno. The kernel needs: per-thread x[i]^2, warp shuffle reduction,
            // shared memory block reduction, single f32 output per buffer.
            //
            // No stream.synchronize() needed: backward and optimizer_step are both
            // GPU-side operations on the same stream — CUDA stream ordering guarantees
            // the backward completes before optimizer_step reads the workspace.

            // Per-block optimizer step: consume workspace gradients before next block overwrites
            let _ = self.cuda_blocks[layer_idx].optimizer_step(
                &mut self.gpu_training.optimizer_states[layer_idx],
                step, lr, beta1, beta2, 1e-8, weight_decay, stream,
                &self.cuda_grad_workspace,
            );

            grad_output_is_a = !grad_output_is_a;
        }

        stream.synchronize().ok()?;

        Some(grad_output_is_a)
    }

    /// Compute exact gradient L2 norm by downloading the full buffer to CPU.
    ///
    /// R-004: Returns `(clip_scale, grad_norm)` for observability.
    fn compute_clip_scale_with_norm(buf: &GpuBuffer<f32>, max_norm: f32) -> (f32, f32) {
        let mut host = vec![0.0f32; buf.len()];
        if buf.copy_to_host_at(&mut host, 0).is_err() {
            return (1.0, 0.0);
        }
        let sq_sum: f64 = host.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let grad_norm = sq_sum.sqrt() as f32;
        let scale = if grad_norm > max_norm {
            max_norm / grad_norm
        } else {
            1.0
        };
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
            if grad_output_is_a { &*grad_a_ptr } else { &*grad_b_ptr }
        };
        let mut embed_grad_data = self.cuda_trainer.download(embed_grad_buf).ok()?;

        // C-EMBED-GRAD-001: Clip activation gradient before scatter-add.
        // Without this, 24-layer random-init backward amplifies gradients to ~1e35,
        // which overflows the CPU AdamW's second moment buffer.
        if let Some(max_norm) = self.config.base.max_grad_norm {
            let sq_sum: f64 = embed_grad_data.iter().map(|&x| (x as f64) * (x as f64)).sum();
            let grad_norm = sq_sum.sqrt() as f32;
            if grad_norm > max_norm {
                let scale = max_norm / grad_norm;
                for g in &mut embed_grad_data {
                    *g *= scale;
                }
            }
        }

        // Scatter-add into embedding weight gradient
        let embed_weight = &mut self.model.embed_tokens.weight;
        if embed_weight.grad().is_none() {
            let zeros = ndarray::Array1::zeros(embed_weight.len());
            embed_weight.set_grad(zeros);
        }
        if let Some(existing_grad) = embed_weight.grad() {
            let mut new_grad = existing_grad.clone();
            for (pos, &token_id) in input_ids.iter().enumerate() {
                let tid = token_id as usize;
                if tid < vocab_size {
                    let src = pos * hidden_size;
                    let dst = tid * hidden_size;
                    for h in 0..hidden_size {
                        new_grad[dst + h] += embed_grad_data[src + h];
                    }
                }
            }
            embed_weight.set_grad(new_grad);
        }
        Some(())
    }

    /// Apply optimizer step to CPU embedding and update metrics.
    ///
    /// GPU block optimizer steps now run interleaved with backward in `gpu_backward()`.
    /// LM head and final norm optimizer steps also run in `gpu_backward()`.
    /// This method handles only CPU embedding and bookkeeping.
    fn optimizer_step(&mut self) {
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
    /// Returns average loss for the batch.
    pub fn train_batch(&mut self, batch: &LMBatch) -> f32 {
        if batch.batch_size == 0 {
            return 0.0;
        }

        if self.accumulated_batches == 0 {
            // Zero embedding gradients at start of accumulation window
            self.embed_optimizer
                .zero_grad_refs(&mut vec![&mut self.model.embed_tokens.weight]);
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

            if let Some(loss) = self.train_step_single(input_ids, target_ids) {
                total_loss += loss;
                valid_count += 1;
            }
        }

        let avg_loss = if valid_count > 0 {
            total_loss / valid_count as f32
        } else {
            0.0
        };

        self.accumulated_loss += avg_loss / self.config.accumulation_steps as f32;
        self.accumulated_batches += 1;

        if self.accumulated_batches >= self.config.accumulation_steps {
            self.optimizer_step();
        }

        avg_loss
    }

    /// R-005: Evaluate a batch without backward pass or weight updates.
    /// Returns average cross-entropy loss, or 0.0 if no valid items.
    pub fn eval_batch(&mut self, batch: &LMBatch) -> f32 {
        let hidden_size = self.config.model_config.hidden_size;
        let vocab_size = self.config.model_config.vocab_size;
        let mut total_loss = 0.0;
        let mut valid_count = 0;
        for i in 0..batch.batch_size {
            let Some(input_ids) = batch.get_input(i) else { continue };
            let Some(target_ids) = batch.get_target(i) else { continue };
            let seq_len = input_ids.len();
            let Some(logits) = self.gpu_forward(input_ids, seq_len, hidden_size, vocab_size) else { continue };
            let Some((loss, _)) = self.cpu_loss_and_grad(&logits, target_ids, seq_len, vocab_size) else { continue };
            total_loss += loss;
            valid_count += 1;
        }
        if valid_count > 0 { total_loss / valid_count as f32 } else { 0.0 }
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

        total_loss / batches_processed.max(1) as f32
    }

    /// Returns true if max_steps has been reached.
    pub fn reached_max_steps(&self) -> bool {
        self.config.max_steps.map_or(false, |max| self.step >= max)
    }

    /// Get current step count.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Get current learning rate (with warmup).
    pub fn current_lr(&self) -> f32 {
        let base_lr = self.config.lr;
        if self.step < self.config.warmup_steps {
            base_lr * (self.step as f32 / self.config.warmup_steps.max(1) as f32)
        } else {
            base_lr
        }
    }

    /// R-004: Get last observed gradient L2 norm (LM head proxy).
    pub fn last_grad_norm(&self) -> f32 {
        self.last_grad_norm
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

                layer.input_norm.weight =
                    Tensor::from_vec(weights.input_norm_weight, false);
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
            names.push(format!(
                "model.layers.{layer}.post_attention_layernorm.weight"
            ));
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
        let param_data: Vec<(String, Vec<f32>)> = names
            .into_iter()
            .zip(model_params)
            .map(|(n, t)| (n, t.data().to_vec()))
            .collect();

        let name = name.to_string();
        let architecture = architecture.to_string();

        Box::new(move |path: &std::path::Path| {
            let params: Vec<(String, Tensor)> = param_data
                .into_iter()
                .map(|(n, d)| (n, Tensor::from_vec(d, false)))
                .collect();
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
        let m_data: Vec<Option<Vec<f32>>> = self.embed_optimizer.first_moments()
            .iter()
            .map(|opt| opt.as_ref().map(|a| a.to_vec()))
            .collect();
        let v_data: Vec<Option<Vec<f32>>> = self.embed_optimizer.second_moments()
            .iter()
            .map(|opt| opt.as_ref().map(|a| a.to_vec()))
            .collect();
        let state = serde_json::json!({
            "type": "adamw_cpu_embed",
            "step": self.embed_optimizer.step_count(),
            "m": m_data,
            "v": v_data,
        });
        let json_str = serde_json::to_string(&state)
            .map_err(|e| crate::error::Error::ConfigError(format!("serialize optimizer state: {}", e)))?;
        std::fs::write(&path, json_str)
            .map_err(|e| crate::error::Error::ConfigError(format!("write optimizer state: {}", e)))?;
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
        let floats: Vec<f32> = inner.iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();
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
    pub fn new(
        _config: super::config::TransformerTrainConfig,
    ) -> crate::Result<Self> {
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
