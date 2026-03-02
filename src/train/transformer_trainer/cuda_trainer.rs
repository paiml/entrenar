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
use crate::autograd::cuda_optim::adamw_step_cuda;
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
        let lm_head_m = GpuBuffer::new(&ctx, vocab_size * hidden_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("LM head m alloc failed: {e:?}"))
        })?;
        let lm_head_v = GpuBuffer::new(&ctx, vocab_size * hidden_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("LM head v alloc failed: {e:?}"))
        })?;

        // Final norm optimizer states
        let final_norm_m = GpuBuffer::new(&ctx, hidden_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("Final norm m alloc failed: {e:?}"))
        })?;
        let final_norm_v = GpuBuffer::new(&ctx, hidden_size).map_err(|e| {
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
        let embed_optimizer = AdamW::default_params(config.lr);

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
        let logits_data = match self.gpu_forward(input_ids, seq_len, hidden_size, vocab_size) {
            Some(d) => d,
            None => {
                eprintln!("[CUDA] gpu_forward failed (seq_len={seq_len}, H={hidden_size}, V={vocab_size})");
                return None;
            }
        };

        // Step 7: Cross-entropy loss + gradient (CPU)
        let (loss_val, grad_logits) = match self.cpu_loss_and_grad(
            &logits_data, target_ids, seq_len, vocab_size,
        ) {
            Some(v) => v,
            None => {
                eprintln!("[CUDA] cpu_loss_and_grad failed");
                return None;
            }
        };

        // Steps 8-11: GPU backward pass
        let grad_output_is_a = match self.gpu_backward(&grad_logits, seq_len, hidden_size, vocab_size) {
            Some(v) => v,
            None => {
                eprintln!("[CUDA] gpu_backward failed");
                return None;
            }
        };

        // Step 12: Embedding backward (CPU)
        if self.embed_backward(input_ids, seq_len, hidden_size, vocab_size, grad_output_is_a).is_none() {
            eprintln!("[CUDA] embed_backward failed");
            return None;
        }

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
        let actual_buf_size = seq_len * hidden_size;
        let mut gpu_input = self.cuda_trainer.upload(hidden_slice).ok()?;
        let mut gpu_output = self.cuda_trainer.zeros(actual_buf_size).ok()?;

        // Forward through CUDA blocks, saving layer inputs
        for (i, block) in self.cuda_blocks.iter_mut().enumerate() {
            // SAFETY: Both buffers are valid GPU allocations. Copy completes
            // before block.forward() reads from gpu_input (same stream ordering).
            unsafe {
                self.gpu_training.layer_inputs[i]
                    .copy_from_buffer_async(&gpu_input, stream)
                    .ok()?;
            }
            block.forward(&gpu_input, &mut gpu_output, seq_len, stream).ok()?;
            std::mem::swap(&mut gpu_input, &mut gpu_output);
        }

        // Save blocks output for final norm backward
        // SAFETY: Disjoint GPU buffers, stream-ordered.
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
        stream.synchronize().ok()?;
        let mut buf = vec![0.0f32; seq_len * vocab_size];
        self.gpu_training.logits_buf.copy_to_host(&mut buf).ok()?;

        // NaN guard
        if buf.iter().any(|v| !v.is_finite()) {
            eprintln!("[CUDA] NaN in logits — falling back");
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

    /// GPU backward pass: upload grad → LM head backward → norm backward → block backward.
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

        // Upload grad_logits (Transfer 3: H2D)
        let grad_logits_gpu = self.cuda_trainer.upload(grad_logits).map_err(|e| {
            eprintln!("[CUDA backward] upload grad_logits failed: {e:?}");
        }).ok()?;

        // LM head GEMM backward
        gemm_backward_a(
            &grad_logits_gpu,
            &self.lm_head_weight_gpu,
            &mut self.gpu_training.lm_head_grad_hidden,
            seq_len as u32, hidden_size as u32, vocab_size as u32,
            stream,
        ).map_err(|e| {
            eprintln!("[CUDA backward] gemm_backward_a failed: {e:?}");
        }).ok()?;
        gemm_backward_b(
            &self.gpu_training.norm_output,
            &grad_logits_gpu,
            &mut self.lm_head_grad_gpu,
            seq_len as u32, hidden_size as u32, vocab_size as u32,
            stream,
        ).map_err(|e| {
            eprintln!("[CUDA backward] gemm_backward_b failed: {e:?}");
        }).ok()?;

        // Final RMSNorm backward
        rms_norm_backward(
            &self.gpu_training.blocks_output,
            &self.gpu_training.final_norm_weight,
            &self.gpu_training.lm_head_grad_hidden,
            &mut self.gpu_training.grad_buf_a,
            &mut self.gpu_training.grad_final_norm_weight,
            seq_len as u32, hidden_size as u32, 1e-5_f32, stream,
        ).map_err(|e| {
            eprintln!("[CUDA backward] rms_norm_backward failed: {e:?}");
        }).ok()?;

        // Backward through blocks in reverse
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
            ).map_err(|e| {
                eprintln!("[CUDA backward] block[{layer_idx}].backward failed: {e:?}");
            }).ok()?;
            grad_output_is_a = !grad_output_is_a;
        }

        stream.synchronize().ok()?;
        Some(grad_output_is_a)
    }

    /// Download embedding gradient from GPU and scatter-add into CPU weight.
    #[allow(unsafe_code)]
    fn embed_backward(
        &mut self,
        input_ids: &[u32],
        seq_len: usize,
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
        let embed_grad_data = self.cuda_trainer.download(embed_grad_buf).ok()?;

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

    /// Apply optimizer step to all GPU weights and CPU embedding.
    ///
    /// Called after accumulation_steps batches.
    fn optimizer_step(&mut self) {
        let lr = self.current_lr();
        let stream = self.cuda_trainer.stream();

        self.gpu_training.step += 1;
        let step = self.gpu_training.step;

        // GPU optimizer step for transformer blocks
        for (block, opt_state) in self.cuda_blocks.iter_mut().zip(
            self.gpu_training.optimizer_states.iter_mut(),
        ) {
            let _ = block.optimizer_step(
                opt_state,
                step,
                lr,
                0.9,   // beta1
                0.999, // beta2
                1e-8,  // eps
                0.01,  // weight_decay
                stream,
                &self.cuda_grad_workspace,
            );
        }

        // GPU optimizer step for LM head weight
        let n_lm = self.lm_head_weight_gpu.len() as u32;
        let _ = adamw_step_cuda(
            &mut self.lm_head_weight_gpu,
            &self.lm_head_grad_gpu,
            &mut self.lm_head_m,
            &mut self.lm_head_v,
            lr,
            0.9,
            0.999,
            1e-8,
            0.01,
            step,
            n_lm,
            stream,
        );

        // GPU optimizer step for final norm weight
        let n_norm = self.gpu_training.final_norm_weight.len() as u32;
        let _ = adamw_step_cuda(
            &mut self.gpu_training.final_norm_weight,
            &self.gpu_training.grad_final_norm_weight,
            &mut self.final_norm_m,
            &mut self.final_norm_v,
            lr,
            0.9,
            0.999,
            1e-8,
            0.01,
            step,
            n_norm,
            stream,
        );

        let _ = stream.synchronize();

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

    /// GPU device name.
    pub fn gpu_name(&self) -> String {
        self.cuda_trainer.device_name()
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
