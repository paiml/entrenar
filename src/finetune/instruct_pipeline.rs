//! Instruction-following fine-tuning pipeline (GH-371)
//!
//! Wires Transformer + LoRA for causal language model fine-tuning on
//! instruction-response pairs.
//!
//! # Architecture
//!
//! ```text
//! [prompt_ids ++ response_ids] -> Transformer.forward() -> logits [seq_len, vocab_size]
//!   -> causal_lm_loss(logits[prompt_len..], response_ids) -> scalar loss
//! ```
//!
//! # Contract
//!
//! - F-INST-002: Loss computed only on response tokens (prompt tokens masked)
//! - F-INST-003: Perplexity = exp(avg_loss) reported per epoch
//! - F-INST-004: LoRA adapters saved in APR format

use crate::lora::LoRALayer;
use crate::optim::{clip_grad_norm_refs, AdamW, Optimizer};
use crate::tokenizer::HfTokenizer;
use crate::transformer::{Transformer, TransformerConfig};
use crate::Tensor;
use std::path::{Path, PathBuf};

#[cfg(feature = "cuda")]
use crate::autograd::cuda_backward::pre_warm_lora_backward_kernels as pre_warm_backward_cache_kernels;
#[cfg(feature = "cuda")]
use crate::autograd::cuda_forward::{
    gemm_forward, pre_warm_forward_kernels, pre_warm_lora_backward_kernels,
};
#[cfg(feature = "cuda")]
use crate::autograd::cuda_optim::{fused_causal_cross_entropy_cuda, pre_warm_lora_adamw_kernels};
#[cfg(feature = "cuda")]
use crate::autograd::cuda_training::{cuda_training_available, CudaTrainer};
#[cfg(feature = "cuda")]
use crate::gpu::guard::VramGuard;
#[cfg(feature = "cuda")]
use crate::transformer::{
    CudaBlock, CudaBlockScratch, CudaLoraGradWorkspace, CudaTransformerBlock, GpuLoraOptimizerState,
};
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use trueno_gpu::driver::GpuBuffer;

/// Configuration for instruction fine-tuning.
#[derive(Debug, Clone)]
pub struct InstructConfig {
    /// LoRA rank
    pub lora_rank: usize,
    /// LoRA alpha
    pub lora_alpha: f32,
    /// Learning rate
    pub learning_rate: f32,
    /// Number of training epochs
    pub epochs: usize,
    /// Maximum sequence length (prompt + response)
    pub max_seq_len: usize,
    /// Maximum gradient norm for clipping
    pub gradient_clip_norm: Option<f32>,
    /// Quantize frozen weights to NF4 (4-bit) for QLoRA training (default: false).
    ///
    /// When enabled, uses `CudaNf4TransformerBlock` (~8x VRAM compression) instead
    /// of `CudaTransformerBlock`. GPU backward pass updates only LoRA adapters.
    pub quantize_nf4: bool,
}

impl Default for InstructConfig {
    fn default() -> Self {
        Self {
            lora_rank: 16,
            lora_alpha: 32.0,
            learning_rate: 2e-4,
            epochs: 3,
            max_seq_len: 512,
            gradient_clip_norm: Some(1.0),
            quantize_nf4: false,
        }
    }
}

/// Result of processing one instruction-response pair.
#[derive(Debug, Clone)]
pub struct InstructStepResult {
    /// Cross-entropy loss on response tokens
    pub loss: f32,
    /// Number of response tokens
    pub num_response_tokens: usize,
    /// Perplexity = exp(loss)
    pub perplexity: f32,
}

/// Result of processing a mini-batch of instruction samples.
#[derive(Debug, Clone)]
pub struct InstructBatchResult {
    /// Average cross-entropy loss across the batch (response tokens only)
    pub avg_loss: f32,
    /// Total response tokens in batch
    pub total_response_tokens: usize,
    /// Perplexity = exp(avg_loss)
    pub perplexity: f32,
    /// Gradient norm before clipping
    pub grad_norm: f32,
}

/// Instruction fine-tuning pipeline.
///
/// Owns the transformer and LoRA adapters. Uses `Transformer::forward()`
/// for causal LM logits and computes loss on response tokens only.
/// GPU-resident training state for NF4 QLoRA backward pass.
///
/// Holds per-layer activation snapshots and scratch buffers needed for
/// activation checkpointing during NF4 backward. Mirrors the layer_input
/// snapshotting from ClassifyPipeline's GpuTrainingState.
#[cfg(feature = "cuda")]
struct InstructGpuTrainingState {
    /// Saved input to each block during forward [num_layers][max_seq_len * hidden_size]
    layer_inputs: Vec<GpuBuffer<f32>>,
    /// Final RMSNorm weight uploaded to GPU [hidden_size]
    final_norm_weight: GpuBuffer<f32>,
    /// Blocks output saved on GPU for final norm backward [max_seq_len * hidden_size]
    blocks_output: GpuBuffer<f32>,
    /// Gradient scratch buffer A [max_seq_len * hidden_size]
    grad_buf_a: GpuBuffer<f32>,
    /// Gradient scratch buffer B [max_seq_len * hidden_size]
    grad_buf_b: GpuBuffer<f32>,
    /// Gradient for final RMSNorm weight [hidden_size]
    grad_final_norm_weight: GpuBuffer<f32>,
    /// Transposed embedding weights on GPU [hidden_size * vocab_size] for lm_head forward GEMM
    embed_transposed: GpuBuffer<f32>,
    /// KAIZEN-068: Non-transposed embedding weights on GPU [vocab_size * hidden_size] for
    /// lm_head backward GEMM. Eliminates ~1.45GB H2D upload per training step.
    embed_original: GpuBuffer<f32>,
    /// GPU scratch for logits [max_seq_len * vocab_size]
    logits_buf: GpuBuffer<f32>,
    /// GPU scratch for grad_hidden [max_seq_len * hidden_size]
    grad_hidden_buf: GpuBuffer<f32>,
    /// KAIZEN-045: Pre-allocated scratch buffer for activation checkpointing in backward
    /// [max_seq_len * hidden_size]. Eliminates per-backward cuMemAlloc/cuMemFree.
    output_scratch: GpuBuffer<f32>,
    /// KAIZEN-045: Pre-allocated upload buffer for gradient H2D transfer in backward
    /// [max_seq_len * hidden_size]. Eliminates per-backward cuMemAlloc/cuMemFree.
    grad_upload_buf: GpuBuffer<f32>,
    /// KAIZEN-062: Pre-allocated forward ping-pong buffer A [max_seq_len * hidden_size].
    /// Eliminates per-forward cuMemAlloc/cuMemFree in forward_cuda_training.
    fwd_scratch_a: GpuBuffer<f32>,
    /// KAIZEN-062: Pre-allocated forward ping-pong buffer B [max_seq_len * hidden_size].
    fwd_scratch_b: GpuBuffer<f32>,
    /// KAIZEN-062: Pre-allocated lm_head hidden input buffer [max_seq_len * hidden_size].
    /// Eliminates per-forward cuMemAlloc/cuMemFree in forward_logits_gpu.
    lm_head_hidden_buf: GpuBuffer<f32>,
}

pub struct InstructPipeline {
    /// Base transformer model
    pub model: Transformer,
    /// LoRA adapters applied to Q/V attention projections
    pub lora_layers: Vec<LoRALayer>,
    /// Pipeline configuration
    pub config: InstructConfig,
    /// AdamW optimizer for trainable parameters
    optimizer: AdamW,
    /// Optional BPE tokenizer
    tokenizer: Option<HfTokenizer>,
    /// Path to base model (for checkpoint provenance)
    model_dir: Option<PathBuf>,
    /// CUDA trainer for GPU memory management
    #[cfg(feature = "cuda")]
    cuda_trainer: Option<CudaTrainer>,
    /// CUDA-accelerated transformer blocks — one per layer
    #[cfg(feature = "cuda")]
    cuda_blocks: Option<Vec<CudaBlock>>,
    /// Shared scratch buffers for NF4 forward pass
    #[cfg(feature = "cuda")]
    shared_scratch: Option<CudaBlockScratch>,
    /// Count of GPU forward passes that produced NaN/Inf
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    cuda_nan_count: usize,
    /// GPU training state for NF4 QLoRA backward pass
    #[cfg(feature = "cuda")]
    gpu_training: Option<InstructGpuTrainingState>,
    /// Shared LoRA gradient workspace for NF4 QLoRA backward
    #[cfg(feature = "cuda")]
    cuda_lora_grad_workspace: Option<CudaLoraGradWorkspace>,
    /// Per-layer LoRA optimizer states for NF4 QLoRA training
    #[cfg(feature = "cuda")]
    cuda_lora_optimizer_states: Option<Vec<GpuLoraOptimizerState>>,
    /// NF4 LoRA optimizer step counter
    #[cfg(feature = "cuda")]
    nf4_lora_step: u32,
    /// VRAM reservation guard (GPU-SHARE-002). Releases ledger entry on Drop.
    /// Held for RAII — released when pipeline is dropped.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    vram_guard: Option<VramGuard>,
}

impl InstructPipeline {
    /// Create a new pipeline with random weights.
    pub fn new(model_config: &TransformerConfig, instruct_config: InstructConfig) -> Self {
        let model = Transformer::new(model_config);
        let mut lora_layers = Self::build_lora_layers(&model, model_config, &instruct_config);

        for lora in &mut lora_layers {
            for param in lora.trainable_params() {
                param.set_requires_grad(true);
            }
        }

        let optimizer = AdamW::default_params(instruct_config.learning_rate);

        #[allow(unused_mut)]
        let mut pipeline = Self {
            model,
            lora_layers,
            config: instruct_config,
            optimizer,
            tokenizer: None,
            model_dir: None,
            #[cfg(feature = "cuda")]
            cuda_trainer: None,
            #[cfg(feature = "cuda")]
            cuda_blocks: None,
            #[cfg(feature = "cuda")]
            shared_scratch: None,
            #[cfg(feature = "cuda")]
            cuda_nan_count: 0,
            #[cfg(feature = "cuda")]
            gpu_training: None,
            #[cfg(feature = "cuda")]
            cuda_lora_grad_workspace: None,
            #[cfg(feature = "cuda")]
            cuda_lora_optimizer_states: None,
            #[cfg(feature = "cuda")]
            nf4_lora_step: 0,
            #[cfg(feature = "cuda")]
            vram_guard: None,
        };

        #[cfg(feature = "cuda")]
        if pipeline.config.quantize_nf4 {
            pipeline.init_cuda(model_config);
        }

        pipeline
    }

    /// Create pipeline from pretrained model weights.
    ///
    /// Loads transformer from SafeTensors and optionally a BPE tokenizer.
    ///
    /// # Errors
    /// Returns error if model files cannot be loaded.
    pub fn from_pretrained(
        model_dir: &Path,
        model_config: &TransformerConfig,
        instruct_config: InstructConfig,
    ) -> crate::Result<Self> {
        let model = Transformer::from_safetensors(model_dir, model_config)?;
        let mut lora_layers = Self::build_lora_layers(&model, model_config, &instruct_config);

        for lora in &mut lora_layers {
            for param in lora.trainable_params() {
                param.set_requires_grad(true);
            }
        }

        let optimizer = AdamW::default_params(instruct_config.learning_rate);

        // CONTRACT: Training requires a BPE tokenizer — byte-fallback is not acceptable.
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            Some(HfTokenizer::from_file(&tokenizer_path).map_err(|e| {
                crate::Error::ConfigError(format!(
                    "Failed to load tokenizer from '{}': {e}. \
                     Training requires a BPE tokenizer.",
                    tokenizer_path.display(),
                ))
            })?)
        } else {
            return Err(crate::Error::ConfigError(format!(
                "No tokenizer.json found in '{}'. Training requires a BPE tokenizer.",
                model_dir.display(),
            )));
        };

        #[allow(unused_mut)]
        let mut pipeline = Self {
            model,
            lora_layers,
            config: instruct_config,
            optimizer,
            tokenizer,
            model_dir: Some(model_dir.to_path_buf()),
            #[cfg(feature = "cuda")]
            cuda_trainer: None,
            #[cfg(feature = "cuda")]
            cuda_blocks: None,
            #[cfg(feature = "cuda")]
            shared_scratch: None,
            #[cfg(feature = "cuda")]
            cuda_nan_count: 0,
            #[cfg(feature = "cuda")]
            gpu_training: None,
            #[cfg(feature = "cuda")]
            cuda_lora_grad_workspace: None,
            #[cfg(feature = "cuda")]
            cuda_lora_optimizer_states: None,
            #[cfg(feature = "cuda")]
            nf4_lora_step: 0,
            #[cfg(feature = "cuda")]
            vram_guard: None,
        };

        #[cfg(feature = "cuda")]
        if pipeline.config.quantize_nf4 {
            pipeline.init_cuda(model_config);
        }

        Ok(pipeline)
    }

    /// Create pipeline from APR model file (.apr format).
    ///
    /// Loads transformer weights from the APR binary, dequantizing from any
    /// stored dtype (F16, Q4K, etc.) to F32. Loads sibling tokenizer if present
    /// (e.g., `model.tokenizer.json` next to `model.apr`).
    ///
    /// # Errors
    /// Returns error if APR file cannot be loaded or weights are invalid.
    pub fn from_apr(
        apr_path: &Path,
        model_config: &TransformerConfig,
        instruct_config: InstructConfig,
    ) -> crate::Result<Self> {
        let model = Transformer::from_apr(apr_path, model_config)?;
        let mut lora_layers = Self::build_lora_layers(&model, model_config, &instruct_config);

        for lora in &mut lora_layers {
            for param in lora.trainable_params() {
                param.set_requires_grad(true);
            }
        }

        let optimizer = AdamW::default_params(instruct_config.learning_rate);

        // Sibling tokenizer: {stem}.tokenizer.json next to the .apr file
        // CONTRACT: Training requires a BPE tokenizer — byte-fallback is not acceptable.
        let tokenizer = {
            let sibling = apr_path.file_stem().and_then(|stem| {
                apr_path
                    .parent()
                    .map(|p| p.join(format!("{}.tokenizer.json", stem.to_str().unwrap_or(""))))
            });

            match sibling {
                Some(ref path) if path.exists() => {
                    let tok = HfTokenizer::from_file(path).map_err(|e| {
                        crate::Error::ConfigError(format!(
                            "Failed to load tokenizer from '{}': {e}. \
                             Training requires a BPE tokenizer — byte-level \
                             fallback is not supported.",
                            path.display(),
                        ))
                    })?;
                    eprintln!(
                        "[tokenizer] Loaded BPE tokenizer from {} (vocab_size={})",
                        path.display(),
                        tok.vocab_size(),
                    );
                    Some(tok)
                }
                _ => {
                    return Err(crate::Error::ConfigError(format!(
                        "No sibling tokenizer found for '{}'. Expected \
                         '{}.tokenizer.json' next to the .apr file. Training \
                         requires a BPE tokenizer.",
                        apr_path.display(),
                        apr_path.file_stem().unwrap_or_default().to_str().unwrap_or(""),
                    )));
                }
            }
        };

        #[allow(unused_mut)]
        let mut pipeline = Self {
            model,
            lora_layers,
            config: instruct_config,
            optimizer,
            tokenizer,
            model_dir: Some(apr_path.to_path_buf()),
            #[cfg(feature = "cuda")]
            cuda_trainer: None,
            #[cfg(feature = "cuda")]
            cuda_blocks: None,
            #[cfg(feature = "cuda")]
            shared_scratch: None,
            #[cfg(feature = "cuda")]
            cuda_nan_count: 0,
            #[cfg(feature = "cuda")]
            gpu_training: None,
            #[cfg(feature = "cuda")]
            cuda_lora_grad_workspace: None,
            #[cfg(feature = "cuda")]
            cuda_lora_optimizer_states: None,
            #[cfg(feature = "cuda")]
            nf4_lora_step: 0,
            #[cfg(feature = "cuda")]
            vram_guard: None,
        };

        #[cfg(feature = "cuda")]
        if pipeline.config.quantize_nf4 {
            pipeline.init_cuda(model_config);
        }

        Ok(pipeline)
    }

    /// Build LoRA layers for Q and V projections (same pattern as ClassifyPipeline).
    /// Build LoRA layers for Q and V projections of each transformer layer.
    pub fn build_lora_layers(
        model: &Transformer,
        model_config: &TransformerConfig,
        config: &InstructConfig,
    ) -> Vec<LoRALayer> {
        let hidden = model_config.hidden_size;
        let head_dim = hidden / model_config.num_attention_heads;

        let mut lora_layers = Vec::new();

        for layer in &model.layers {
            let attn = &layer.self_attn;

            // Q projection LoRA
            let q_dim = model_config.num_attention_heads * head_dim;
            let q_weight = Tensor::from_vec(
                attn.w_q.data().as_slice().expect("contiguous w_q").to_vec(),
                false,
            );
            lora_layers.push(LoRALayer::new(
                q_weight,
                q_dim,
                hidden,
                config.lora_rank,
                config.lora_alpha,
            ));

            // V projection LoRA
            let v_dim = model_config.num_kv_heads * head_dim;
            let v_weight = Tensor::from_vec(
                attn.w_v.data().as_slice().expect("contiguous w_v").to_vec(),
                false,
            );
            lora_layers.push(LoRALayer::new(
                v_weight,
                v_dim,
                hidden,
                config.lora_rank,
                config.lora_alpha,
            ));
        }

        lora_layers
    }

    /// Tokenize text without truncation.
    ///
    /// Returns the full token sequence. Callers (e.g., `train_step`) are
    /// responsible for budget allocation and truncation of the concatenated
    /// prompt+response sequence.
    ///
    /// Falls back to byte-level encoding (each UTF-8 byte as a u32 token ID)
    /// when no BPE tokenizer is loaded.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        match self.tokenizer.as_ref() {
            Some(tok) => tok.encode(text),
            None => {
                // Byte-level fallback when no BPE tokenizer is loaded
                text.bytes().map(u32::from).collect()
            }
        }
    }

    /// Returns `true` if a BPE tokenizer is loaded.
    #[must_use]
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    /// Compute causal LM loss on a single instruction-response pair.
    ///
    /// # Contract (F-INST-002)
    /// Loss is computed only on response tokens. Prompt tokens are masked.
    ///
    /// When CUDA NF4 blocks are available, dispatches to GPU forward pass
    /// with CPU loss computation and GPU backward/optimizer.
    pub fn train_step(&mut self, prompt_ids: &[u32], response_ids: &[u32]) -> InstructStepResult {
        let full_ids: Vec<u32> = prompt_ids.iter().chain(response_ids.iter()).copied().collect();

        let prompt_len = prompt_ids.len();
        let response_len = response_ids.len();

        if response_len == 0 || full_ids.len() < 2 {
            return InstructStepResult { loss: 0.0, num_response_tokens: 0, perplexity: 1.0 };
        }

        let full_ids = if full_ids.len() > self.config.max_seq_len {
            full_ids[..self.config.max_seq_len].to_vec()
        } else {
            full_ids
        };
        let seq_len = full_ids.len();
        let vocab_size = self.model.config().vocab_size;

        // Cap prompt_len at truncated sequence length. If the prompt alone
        // exceeds max_seq_len, all response tokens were truncated away.
        let prompt_len = prompt_len.min(seq_len);

        // ── GPU path (NF4 QLoRA) ──────────────────────────────────────
        #[cfg(feature = "cuda")]
        if self.cuda_blocks.is_some() {
            return self.cuda_train_step(&full_ids, prompt_len, seq_len, vocab_size);
        }

        // ── CPU path ──────────────────────────────────────────────────

        // 1. Zero gradients
        for lora in &mut self.lora_layers {
            for param in lora.trainable_params() {
                param.zero_grad();
            }
        }

        // 2. Forward pass → logits [seq_len, vocab_size]
        let logits = self.model.forward(&full_ids);
        let logits_data = logits.data().as_slice().expect("contiguous logits").to_vec();

        // 3. Causal LM loss on response tokens only
        let loss_start = prompt_len.saturating_sub(1);
        let loss_end = seq_len - 1;
        let num_loss_tokens = loss_end.saturating_sub(loss_start);

        if num_loss_tokens == 0 {
            return InstructStepResult { loss: 0.0, num_response_tokens: 0, perplexity: 1.0 };
        }

        let (avg_loss, grad_logits) =
            Self::compute_causal_lm_loss(&logits_data, &full_ids, loss_start, loss_end, vocab_size);

        // 4. Backward through autograd
        logits.set_grad(ndarray::Array1::from(grad_logits));
        if let Some(op) = logits.backward_op() {
            op.backward();
        }

        // 5. Optimizer step on LoRA parameters
        let mut params: Vec<&mut Tensor> = Vec::new();
        for lora in &mut self.lora_layers {
            params.extend(lora.trainable_params());
        }

        if let Some(max_norm) = self.config.gradient_clip_norm {
            clip_grad_norm_refs(&mut params, max_norm);
        }

        self.optimizer.step_refs(&mut params);

        InstructStepResult {
            loss: avg_loss,
            num_response_tokens: num_loss_tokens,
            perplexity: avg_loss.exp().min(1e6),
        }
    }

    /// GPU-accelerated training step for NF4 QLoRA.
    ///
    /// 1. GPU forward through NF4 transformer blocks → normed hidden states
    /// 2. CPU lm_head matmul → logits
    /// 3. CPU causal LM loss on response tokens only
    /// 4. CPU gradient of loss w.r.t. hidden states (through lm_head)
    /// 5. GPU backward through NF4 blocks → LoRA gradient + optimizer step
    #[cfg(feature = "cuda")]
    fn cuda_train_step(
        &mut self,
        full_ids: &[u32],
        prompt_len: usize,
        seq_len: usize,
        vocab_size: usize,
    ) -> InstructStepResult {
        let loss_start = prompt_len.saturating_sub(1);
        let loss_end = seq_len - 1;
        let num_loss_tokens = loss_end.saturating_sub(loss_start);

        if num_loss_tokens == 0 {
            return InstructStepResult { loss: 0.0, num_response_tokens: 0, perplexity: 1.0 };
        }

        // 1. GPU forward → logits stay GPU-resident in training.logits_buf (KAIZEN-064)
        //    Eliminates ~296MB logits D2H download per step.
        if !self.forward_logits_gpu_resident(full_ids) {
            // Fallback: GPU forward failed, use CPU path with D2H
            eprintln!("[CUDA] GPU forward failed, falling back to CPU for this step");
            return self.cuda_train_step_cpu_loss(
                full_ids,
                loss_start,
                loss_end,
                num_loss_tokens,
                seq_len,
                vocab_size,
            );
        }

        // 2. Fused GPU causal cross-entropy loss + softmax backward (KAIZEN-064)
        //    Gradient computed in-place in logits_buf. Eliminates:
        //    - CPU softmax computation
        //    - ~296MB grad_logits H2D upload
        //    Prepare shifted targets: position pos predicts full_ids[pos + 1]
        let targets: Vec<u32> = (0..seq_len)
            .map(|pos| if pos + 1 < full_ids.len() { full_ids[pos + 1] } else { 0 })
            .collect();

        let scale = 1.0 / num_loss_tokens as f32;

        let avg_loss = (|| -> Option<f32> {
            let trainer = self.cuda_trainer.as_ref()?;
            let stream = trainer.stream();
            let training = self.gpu_training.as_mut()?;
            fused_causal_cross_entropy_cuda(
                &mut training.logits_buf,
                &targets,
                seq_len as u32,
                vocab_size as u32,
                loss_start as u32,
                loss_end as u32,
                scale,
                stream,
            )
            .ok()
        })();

        let avg_loss = match avg_loss {
            Some(l) if l.is_finite() => l,
            Some(_) => {
                eprintln!("[CUDA] NaN/Inf loss detected — skipping backward pass");
                return InstructStepResult {
                    loss: 100.0,
                    num_response_tokens: num_loss_tokens,
                    perplexity: 1e6,
                };
            }
            None => {
                eprintln!("[CUDA] fused causal cross-entropy failed — falling back to CPU");
                return self.cuda_train_step_cpu_loss(
                    full_ids,
                    loss_start,
                    loss_end,
                    num_loss_tokens,
                    seq_len,
                    vocab_size,
                );
            }
        };

        // 3. GPU GEMM backward: grad_hidden = grad_logits @ embed
        //    KAIZEN-064: grad_logits already in logits_buf (in-place from fused kernel).
        //    No upload needed — saves ~296MB H2D per step.
        //    KAIZEN-065: grad_hidden stays GPU-resident in grad_hidden_buf — no D2H download.
        //    KAIZEN-068: embed_original is GPU-resident — no per-step H2D upload (~1.45GB saved).
        let hidden_size = self.model.config().hidden_size;

        let gemm_ok = (|| -> Option<()> {
            let trainer = self.cuda_trainer.as_ref()?;
            let stream = trainer.stream();
            let training = self.gpu_training.as_mut()?;
            gemm_forward(
                &training.logits_buf,
                &training.embed_original,
                &mut training.grad_hidden_buf,
                seq_len as u32,
                vocab_size as u32,
                hidden_size as u32,
                stream,
            )
            .map_err(|e| {
                eprintln!("[CUDA] lm_head backward GEMM failed: {e}");
            })
            .ok()?;
            // KAIZEN-065: No stream.synchronize() needed — GEMM and rms_norm_backward
            // are on the same CUDA stream, so ordering is guaranteed.
            // No trainer.download() needed — grad_hidden_buf is read directly by
            // backward_nf4_gpu_blocks_gpu_resident.
            Some(())
        })();

        if gemm_ok.is_none() {
            eprintln!("[CUDA] lm_head backward failed — returning loss without weight update");
            return InstructStepResult {
                loss: avg_loss,
                num_response_tokens: num_loss_tokens,
                perplexity: avg_loss.exp().min(1e6),
            };
        }

        // 4. GPU backward through NF4 blocks + LoRA optimizer (KAIZEN-065: GPU-resident path)
        //    Gradient flows directly from grad_hidden_buf → rms_norm_backward → block loop.
        //    Eliminates ~5MB D2H + ~5MB H2D + sync + Vec alloc per step.
        if self.config.quantize_nf4 {
            self.backward_nf4_gpu_blocks_gpu_resident(seq_len);
        }

        InstructStepResult {
            loss: avg_loss,
            num_response_tokens: num_loss_tokens,
            perplexity: avg_loss.exp().min(1e6),
        }
    }

    /// CPU fallback for causal LM loss when GPU fused kernel is unavailable.
    /// Used when forward_logits_gpu_resident or fused_causal_cross_entropy_cuda fails.
    #[cfg(feature = "cuda")]
    fn cuda_train_step_cpu_loss(
        &mut self,
        full_ids: &[u32],
        loss_start: usize,
        loss_end: usize,
        num_loss_tokens: usize,
        seq_len: usize,
        vocab_size: usize,
    ) -> InstructStepResult {
        let logits_data = match self.forward_logits_gpu(full_ids) {
            Some(data) => data,
            None => {
                let logits = self.model.forward(full_ids);
                logits.data().as_slice().expect("contiguous logits").to_vec()
            }
        };

        let (avg_loss, grad_logits) =
            Self::compute_causal_lm_loss(&logits_data, full_ids, loss_start, loss_end, vocab_size);

        if !avg_loss.is_finite() {
            return InstructStepResult {
                loss: 100.0,
                num_response_tokens: num_loss_tokens,
                perplexity: 1e6,
            };
        }

        let hidden_size = self.model.config().hidden_size;

        let grad_hidden = (|| -> Option<Vec<f32>> {
            let trainer = self.cuda_trainer.as_ref()?;
            let stream = trainer.stream();
            let training = self.gpu_training.as_mut()?;
            training
                .logits_buf
                .copy_from_host_at(&grad_logits, 0)
                .map_err(|e| {
                    eprintln!("[CUDA] lm_head backward: grad_logits upload failed: {e}");
                })
                .ok()?;
            // KAIZEN-068: embed_original is GPU-resident — no per-step H2D upload.
            gemm_forward(
                &training.logits_buf,
                &training.embed_original,
                &mut training.grad_hidden_buf,
                seq_len as u32,
                vocab_size as u32,
                hidden_size as u32,
                stream,
            )
            .map_err(|e| {
                eprintln!("[CUDA] lm_head backward GEMM failed: {e}");
            })
            .ok()?;
            stream.synchronize().ok()?;
            let full_grad = trainer.download(&training.grad_hidden_buf).ok()?;
            Some(full_grad[..seq_len * hidden_size].to_vec())
        })();

        let grad_hidden = match grad_hidden {
            Some(g) => g,
            None => {
                return InstructStepResult {
                    loss: avg_loss,
                    num_response_tokens: num_loss_tokens,
                    perplexity: avg_loss.exp().min(1e6),
                };
            }
        };

        if self.config.quantize_nf4 {
            self.backward_nf4_gpu_blocks(&grad_hidden, seq_len);
        }

        InstructStepResult {
            loss: avg_loss,
            num_response_tokens: num_loss_tokens,
            perplexity: avg_loss.exp().min(1e6),
        }
    }

    /// Evaluate loss and perplexity on a set of samples without updating weights.
    pub fn evaluate(
        &self,
        prompt_ids_batch: &[Vec<u32>],
        response_ids_batch: &[Vec<u32>],
    ) -> InstructBatchResult {
        let mut total_loss = 0.0f32;
        let mut total_response_tokens = 0usize;

        for (prompt_ids, response_ids) in prompt_ids_batch.iter().zip(response_ids_batch.iter()) {
            let full_ids: Vec<u32> =
                prompt_ids.iter().chain(response_ids.iter()).copied().collect();

            let prompt_len = prompt_ids.len();
            if response_ids.is_empty() || full_ids.len() < 2 {
                continue;
            }

            let full_ids = if full_ids.len() > self.config.max_seq_len {
                full_ids[..self.config.max_seq_len].to_vec()
            } else {
                full_ids
            };
            let seq_len = full_ids.len();
            let vocab_size = self.model.config().vocab_size;
            let prompt_len = prompt_len.min(seq_len);

            let logits = self.model.forward(&full_ids);
            let logits_data = logits.data().as_slice().expect("contiguous logits").to_vec();

            let loss_start = prompt_len.saturating_sub(1);
            let loss_end = seq_len - 1;
            let num_loss_tokens = loss_end.saturating_sub(loss_start);

            let (sample_loss, _) = Self::compute_causal_lm_loss(
                &logits_data,
                &full_ids,
                loss_start,
                loss_end,
                vocab_size,
            );

            total_loss += sample_loss * num_loss_tokens as f32;
            total_response_tokens += num_loss_tokens;
        }

        let avg_loss =
            if total_response_tokens > 0 { total_loss / total_response_tokens as f32 } else { 0.0 };

        InstructBatchResult {
            avg_loss,
            total_response_tokens,
            perplexity: avg_loss.exp().min(1e6),
            grad_norm: 0.0,
        }
    }

    /// Compute causal LM loss and gradients for the given position range.
    ///
    /// Returns (average_loss, gradient_logits).
    fn compute_causal_lm_loss(
        logits_data: &[f32],
        full_ids: &[u32],
        loss_start: usize,
        loss_end: usize,
        vocab_size: usize,
    ) -> (f32, Vec<f32>) {
        let seq_len = full_ids.len();
        let num_loss_tokens = loss_end.saturating_sub(loss_start);
        let mut total_loss = 0.0f32;
        let mut grad_logits = vec![0.0f32; seq_len * vocab_size];

        for pos in loss_start..loss_end {
            let target = full_ids[pos + 1] as usize;
            if target >= vocab_size {
                continue;
            }

            let logit_start = pos * vocab_size;
            let row = &logits_data[logit_start..logit_start + vocab_size];

            // Numerically stable log-softmax + gradient in one pass.
            // Writes exp values directly into grad_logits row instead of
            // Uses a per-position Vec<f32> written in-place (~608KB for vocab=151936).
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let grad_row = &mut grad_logits[logit_start..logit_start + vocab_size];
            let mut sum_exp = 0.0f32;
            for j in 0..vocab_size {
                let exp_v = (row[j] - max_val).exp();
                grad_row[j] = exp_v;
                sum_exp += exp_v;
            }

            let log_sum_exp = sum_exp.ln() + max_val;
            let loss_i = -(row[target] - log_sum_exp);
            total_loss += if loss_i.is_finite() { loss_i } else { 100.0 };

            // Convert exp values to softmax gradient in-place: (exp/sum) / num_loss_tokens
            let inv_n = 1.0 / num_loss_tokens as f32;
            let scale = inv_n / sum_exp;
            for j in 0..vocab_size {
                grad_row[j] *= scale;
            }
            grad_row[target] -= inv_n;
        }

        let avg_loss = if num_loss_tokens > 0 { total_loss / num_loss_tokens as f32 } else { 0.0 };

        (avg_loss, grad_logits)
    }

    /// Number of trainable LoRA parameters.
    #[must_use]
    pub fn num_trainable_parameters(&self) -> usize {
        // LoRA layers store weight + lora_a + lora_b; we count lora_a + lora_b
        self.lora_layers.len()
            * 2
            * self.config.lora_rank
            * (self.lora_layers.first().map_or(0, |_| {
                // Approximate: each LoRA pair has rank * (rows + cols) params
                // This is a rough estimate since layers may differ in size
                1
            }))
    }

    /// Update learning rate (for LR scheduling).
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.optimizer.set_lr(lr);
    }

    /// Get current learning rate.
    #[must_use]
    pub fn learning_rate(&self) -> f32 {
        self.optimizer.lr()
    }

    /// Set model path for checkpoint provenance.
    pub fn set_model_path(&mut self, path: &Path) {
        self.model_dir = Some(path.to_path_buf());
    }

    /// Synchronize GPU LoRA weights back to CPU LoRA layers (NF4 QLoRA).
    ///
    /// Required for checkpointing after NF4 QLoRA training. Downloads A_q, B_q,
    /// A_v, B_v from each NF4 block and updates the corresponding CPU LoRA layers.
    ///
    /// # Contract (C-QLORA-CKPT-001)
    ///
    /// - **Precondition**: NF4 QLoRA training completed (optimizer steps applied)
    /// - **Postcondition**: CPU LoRA layers match GPU-trained LoRA weights
    #[cfg(feature = "cuda")]
    pub fn sync_lora_to_cpu(&mut self) {
        let blocks = match self.cuda_blocks.as_ref() {
            Some(b) => b,
            None => return,
        };

        let lora_scale = self.config.lora_alpha / self.config.lora_rank.max(1) as f32;
        let inv_scale = if lora_scale.abs() > 1e-10 { 1.0 / lora_scale } else { 1.0 };

        for (layer_idx, block) in blocks.iter().enumerate() {
            if let Ok((a_q, b_q, a_v, b_v)) = block.download_lora_weights() {
                let q_lora_idx = layer_idx * 2;
                let v_lora_idx = layer_idx * 2 + 1;

                // Un-scale B matrices (GPU stores B * lora_scale)
                let b_q_unscaled: Vec<f32> = b_q.iter().map(|&v| v * inv_scale).collect();
                let b_v_unscaled: Vec<f32> = b_v.iter().map(|&v| v * inv_scale).collect();

                if q_lora_idx < self.lora_layers.len() {
                    *self.lora_layers[q_lora_idx].lora_a_mut() = crate::Tensor::from_vec(a_q, true);
                    *self.lora_layers[q_lora_idx].lora_b_mut() =
                        crate::Tensor::from_vec(b_q_unscaled, true);
                }
                if v_lora_idx < self.lora_layers.len() {
                    *self.lora_layers[v_lora_idx].lora_a_mut() = crate::Tensor::from_vec(a_v, true);
                    *self.lora_layers[v_lora_idx].lora_b_mut() =
                        crate::Tensor::from_vec(b_v_unscaled, true);
                }
            }
        }
    }

    // ── CUDA GPU acceleration ────────────────────────────────────────────

    /// Initialize CUDA acceleration: create trainer, upload blocks, init LoRA training.
    ///
    /// GPU-SHARE-002: Acquires a VRAM guard from the ledger before allocating GPU
    /// memory. If the ledger denies the reservation (insufficient VRAM), falls back
    /// to CPU training. The guard is held for the lifetime of the pipeline and
    /// released on Drop.
    #[cfg(feature = "cuda")]
    fn init_cuda(&mut self, model_config: &TransformerConfig) {
        // GPU-SHARE-002: Acquire VRAM reservation before allocating
        let budget_mb = Self::estimate_vram_mb(model_config, &self.config);
        let task_label = if self.config.quantize_nf4 { "instruct-qlora" } else { "instruct-lora" };
        match VramGuard::acquire(budget_mb, task_label) {
            Ok(guard) => {
                eprintln!(
                    "[GPU-SHARE] VRAM reserved: {budget_mb} MB for {task_label} (gpu: {})",
                    guard.gpu_uuid()
                );
                self.vram_guard = Some(guard);
            }
            Err(e) => {
                eprintln!("[GPU-SHARE] VRAM guard denied: {e} — falling back to CPU");
                return;
            }
        }

        let (trainer, blocks, scratch) =
            Self::try_init_cuda(&self.model, model_config, &self.config, &self.lora_layers);

        if trainer.is_none() {
            // CUDA init failed — release the guard
            self.vram_guard = None;
            return;
        }

        self.cuda_trainer = trainer;
        self.cuda_blocks = blocks;
        self.shared_scratch = scratch;

        // GPU training state (layer input snapshots for backward)
        self.gpu_training = Self::try_init_gpu_training(
            &self.model,
            model_config,
            self.config.max_seq_len,
            self.cuda_trainer.as_ref(),
            self.cuda_blocks.as_ref(),
        );

        // NF4 LoRA training state
        if self.config.quantize_nf4 {
            let (grad_ws, opt_states) = Self::try_init_nf4_lora_training(
                self.cuda_trainer.as_ref(),
                self.cuda_blocks.as_ref(),
                model_config,
                &self.config,
            );
            self.cuda_lora_grad_workspace = grad_ws;
            self.cuda_lora_optimizer_states = opt_states;
        }

        // GPU-SHARE-002: Update actual VRAM usage after all allocations
        if let Some(ref mut guard) = self.vram_guard {
            let _ = guard.update_actual(budget_mb);
        }
    }

    /// Estimate VRAM usage (MB) for GPU training based on model architecture.
    ///
    /// Used by GPU-SHARE-002 to reserve VRAM via the ledger before allocation.
    #[cfg(feature = "cuda")]
    fn estimate_vram_mb(model_config: &TransformerConfig, config: &InstructConfig) -> usize {
        if config.quantize_nf4 {
            // NF4 QLoRA: weights at 4-bit (~0.5 bytes/param) + FP32 scratch + overhead
            let weight_elements =
                model_config.per_layer_weight_elements() * model_config.num_hidden_layers;
            // NF4 weights: 0.5 bytes/param, LoRA adapters in FP16: ~2% of weights
            let weight_mb = weight_elements / (2 * 1024 * 1024);
            // Scratch buffers scale with seq_len * hidden_size
            let scratch_mb =
                (config.max_seq_len * model_config.hidden_size * 4 * 10) / (1024 * 1024);
            // CUDA context + kernel cache + misc overhead
            let overhead_mb = 512;
            weight_mb + scratch_mb + overhead_mb
        } else {
            // FP32: use TransformerConfig's exact VRAM calculator + overhead
            model_config.total_training_vram_bytes_shared(config.max_seq_len) / (1024 * 1024) + 256
        }
    }

    /// Attempt to initialize CUDA acceleration.
    ///
    /// Creates `CudaTrainer` and uploads all transformer layer weights to GPU.
    /// Returns `(None, None, None)` if CUDA is unavailable or any step fails.
    #[cfg(feature = "cuda")]
    fn try_init_cuda(
        model: &Transformer,
        model_config: &TransformerConfig,
        config: &InstructConfig,
        lora_layers: &[LoRALayer],
    ) -> (Option<CudaTrainer>, Option<Vec<CudaBlock>>, Option<CudaBlockScratch>) {
        if !cuda_training_available() {
            eprintln!("[CUDA] No CUDA runtime detected — using CPU");
            return (None, None, None);
        }

        let trainer = match CudaTrainer::new() {
            Ok(t) => {
                eprintln!(
                    "[CUDA] Initialized: {} ({:.1} GB)",
                    t.device_name(),
                    t.total_memory() as f64 / 1e9
                );
                t
            }
            Err(e) => {
                eprintln!("[CUDA] Failed to create trainer: {e} — using CPU");
                return (None, None, None);
            }
        };

        let ctx = Arc::clone(trainer.context());
        let max_seq_len = config.max_seq_len;

        // C-PREWARM-001: JIT-compile forward kernels before block upload
        if let Err(e) = pre_warm_forward_kernels(
            model_config.hidden_size,
            model_config.intermediate_size,
            model_config.num_attention_heads,
            model_config.num_kv_heads,
            model_config.head_dim(),
            max_seq_len,
        ) {
            eprintln!("[CUDA] Failed to pre-warm forward kernels: {e} — using CPU");
            return (None, None, None);
        }

        let quantize_nf4 = config.quantize_nf4;
        if quantize_nf4 {
            eprintln!(
                "[CUDA] NF4 quantization enabled — frozen weights will be 4-bit (~8x compression)"
            );
        }

        // C-PREWARM-001: Pre-warm ALL training kernels BEFORE block upload.
        // All LoRA training needs backward and optimizer kernels, not just NF4.
        let head_dim = model_config.head_dim();
        if let Err(e) = pre_warm_lora_backward_kernels(
            model_config.hidden_size,
            model_config.num_attention_heads * head_dim,
            model_config.num_kv_heads * head_dim,
            max_seq_len,
            config.lora_rank,
        ) {
            eprintln!("[CUDA] Failed to pre-warm LoRA backward kernels: {e} — using CPU");
            return (None, None, None);
        }

        if let Err(e) = pre_warm_backward_cache_kernels(
            model_config.hidden_size,
            model_config.num_attention_heads * head_dim,
            model_config.num_kv_heads * head_dim,
            max_seq_len,
            config.lora_rank,
            model_config.intermediate_size,
            model_config.num_attention_heads,
            quantize_nf4,
        ) {
            eprintln!("[CUDA] Failed to pre-warm backward cache kernels: {e} — using CPU");
            return (None, None, None);
        }

        if let Err(e) = pre_warm_lora_adamw_kernels(
            model_config.hidden_size,
            model_config.num_attention_heads * head_dim,
            model_config.num_kv_heads * head_dim,
            config.lora_rank,
            0, // instruct has no classifier head
            model_config.intermediate_size,
            quantize_nf4,
        ) {
            eprintln!("[CUDA] Failed to pre-warm AdamW kernels: {e} — using CPU");
            return (None, None, None);
        }

        let mut blocks = Vec::with_capacity(model.config.num_hidden_layers);

        for (i, layer) in model.layers.iter().enumerate() {
            let input_norm = layer.input_norm.weight.data();
            let input_norm = input_norm.as_slice().expect("contiguous input_norm");
            let post_attn_norm = layer.post_attn_norm.weight.data();
            let post_attn_norm = post_attn_norm.as_slice().expect("contiguous post_attn_norm");
            let w_q = layer.self_attn.w_q.data();
            let w_q = w_q.as_slice().expect("contiguous w_q");
            let w_k = layer.self_attn.w_k.data();
            let w_k = w_k.as_slice().expect("contiguous w_k");
            let w_v = layer.self_attn.w_v.data();
            let w_v = w_v.as_slice().expect("contiguous w_v");
            let w_o = layer.self_attn.w_o.data();
            let w_o = w_o.as_slice().expect("contiguous w_o");
            let w_gate = layer.ffn.w_gate.data();
            let w_gate = w_gate.as_slice().expect("contiguous w_gate");
            let w_up = layer.ffn.w_up.data();
            let w_up = w_up.as_slice().expect("contiguous w_up");
            let w_down = layer.ffn.w_down.data();
            let w_down = w_down.as_slice().expect("contiguous w_down");

            let result = if quantize_nf4 {
                let lora_scale = config.lora_alpha / config.lora_rank as f32;
                let lora_rank = config.lora_rank;
                let q_lora_idx = i * 2;
                let v_lora_idx = i * 2 + 1;

                // Q LoRA
                let q_a_data;
                let q_b_data;
                let q_lora = if q_lora_idx < lora_layers.len() {
                    q_a_data = lora_layers[q_lora_idx].lora_a().data();
                    q_b_data = lora_layers[q_lora_idx].lora_b().data();
                    Some((
                        q_a_data.as_slice().expect("contiguous lora_a_q"),
                        q_b_data.as_slice().expect("contiguous lora_b_q"),
                    ))
                } else {
                    None
                };

                // V LoRA
                let v_a_data;
                let v_b_data;
                let v_lora = if v_lora_idx < lora_layers.len() {
                    v_a_data = lora_layers[v_lora_idx].lora_a().data();
                    v_b_data = lora_layers[v_lora_idx].lora_b().data();
                    Some((
                        v_a_data.as_slice().expect("contiguous lora_a_v"),
                        v_b_data.as_slice().expect("contiguous lora_b_v"),
                    ))
                } else {
                    None
                };

                crate::transformer::CudaNf4TransformerBlock::new(
                    model_config,
                    i,
                    Arc::clone(&ctx),
                    input_norm,
                    post_attn_norm,
                    w_q,
                    w_k,
                    w_v,
                    w_o,
                    w_gate,
                    w_up,
                    w_down,
                    max_seq_len,
                    q_lora,
                    v_lora,
                    lora_scale,
                    lora_rank,
                )
                .map(CudaBlock::Nf4)
            } else {
                CudaTransformerBlock::new(
                    model_config,
                    i,
                    Arc::clone(&ctx),
                    input_norm,
                    post_attn_norm,
                    w_q,
                    w_k,
                    w_v,
                    w_o,
                    w_gate,
                    w_up,
                    w_down,
                    max_seq_len,
                )
                .map(CudaBlock::Fp32)
            };

            match result {
                Ok(block) => blocks.push(block),
                Err(e) => {
                    eprintln!(
                        "[CUDA] Failed to upload layer {i} to GPU: {e} — falling back to CPU"
                    );
                    return (None, None, None);
                }
            }
        }

        eprintln!(
            "[CUDA] Uploaded {} transformer layers to GPU (max_seq_len={})",
            blocks.len(),
            max_seq_len
        );

        assert_eq!(blocks.len(), model.config.num_hidden_layers);

        // C-SCRATCH-001: Shared scratch for NF4
        let shared_scratch = if quantize_nf4 {
            match CudaBlockScratch::new(model_config, max_seq_len, &ctx, config.lora_rank) {
                Ok(s) => Some(s),
                Err(e) => {
                    eprintln!("[CUDA] Failed to allocate shared scratch: {e} — using CPU");
                    return (None, None, None);
                }
            }
        } else {
            None
        };

        (Some(trainer), Some(blocks), shared_scratch)
    }

    /// Initialize GPU training state for NF4 QLoRA backward pass.
    #[cfg(feature = "cuda")]
    fn try_init_gpu_training(
        model: &Transformer,
        model_config: &TransformerConfig,
        max_seq_len: usize,
        cuda_trainer: Option<&CudaTrainer>,
        cuda_blocks: Option<&Vec<CudaBlock>>,
    ) -> Option<InstructGpuTrainingState> {
        let trainer = cuda_trainer?;
        let blocks = cuda_blocks?;

        let hidden_size = model_config.hidden_size;
        let buf_size = max_seq_len * hidden_size;
        let num_layers = blocks.len();

        // Allocate layer-input snapshot buffers
        let mut layer_inputs = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            match trainer.zeros(buf_size) {
                Ok(buf) => layer_inputs.push(buf),
                Err(e) => {
                    eprintln!("[CUDA] GPU training init failed (layer input alloc): {e}");
                    return None;
                }
            }
        }

        // Upload final RMSNorm weight
        let norm_data = model.norm.weight.data();
        let norm_slice = norm_data.as_slice().expect("contiguous final norm weight");
        let final_norm_weight = match trainer.upload(norm_slice) {
            Ok(buf) => buf,
            Err(e) => {
                eprintln!("[CUDA] GPU training init failed (final norm upload): {e}");
                return None;
            }
        };

        // Allocate gradient scratch buffers
        let blocks_output = trainer.zeros(buf_size).ok()?;
        let grad_buf_a = trainer.zeros(buf_size).ok()?;
        let grad_buf_b = trainer.zeros(buf_size).ok()?;
        let grad_final_norm_weight = trainer.zeros(hidden_size).ok()?;

        // Upload embedding weights in both layouts:
        // - embed_transposed [hidden, vocab]: for lm_head forward GEMM (logits = hidden @ embed_T)
        // - embed_original [vocab, hidden]: for lm_head backward GEMM (grad_hidden = grad_logits @ embed)
        // KAIZEN-068: Both layouts stored permanently on GPU. Eliminates ~1.45GB H2D upload per step.
        // VRAM cost: ~1.45GB additional (Qwen3-4B: 151936 × 2560 × 4). Fits in 24GB RTX 4090.
        let vocab_size = model_config.vocab_size;
        let embed_data = model.embed_tokens.weight.data();
        let embed_slice = embed_data.as_slice().expect("contiguous embed");

        let embed_original = trainer
            .upload(embed_slice)
            .map_err(|e| {
                eprintln!("[CUDA] embed_original upload failed: {e}");
            })
            .ok()?;

        let mut embed_t = vec![0.0f32; hidden_size * vocab_size];
        for v in 0..vocab_size {
            for k in 0..hidden_size {
                embed_t[k * vocab_size + v] = embed_slice[v * hidden_size + k];
            }
        }
        let embed_transposed = trainer
            .upload(&embed_t)
            .map_err(|e| {
                eprintln!("[CUDA] embed_transposed upload failed: {e}");
            })
            .ok()?;

        // Logits scratch: [max_seq_len, vocab_size]
        let logits_buf = trainer
            .zeros(max_seq_len * vocab_size)
            .map_err(|e| {
                eprintln!("[CUDA] logits_buf alloc failed: {e}");
            })
            .ok()?;

        // Grad-hidden scratch: [max_seq_len, hidden_size]
        let grad_hidden_buf = trainer.zeros(buf_size).ok()?;

        eprintln!(
            "[CUDA] GPU training state initialized: {num_layers} layers, \
             {buf_size} buf_size, embed=[{vocab_size}x{hidden_size}]+embed_T=[{hidden_size}x{vocab_size}] on GPU (NF4 QLoRA mode)"
        );

        // KAIZEN-045: Pre-allocate backward scratch buffers to eliminate per-backward
        // cuMemAlloc/cuMemFree. Each cuMemAlloc costs ~10-100µs.
        let output_scratch = trainer.zeros(buf_size).ok()?;
        let grad_upload_buf = trainer.zeros(buf_size).ok()?;

        // KAIZEN-062: Pre-allocate forward ping-pong buffers + lm_head hidden input.
        // Eliminates 3× cuMemAlloc/cuMemFree per training step.
        let fwd_scratch_a = trainer.zeros(buf_size).ok()?;
        let fwd_scratch_b = trainer.zeros(buf_size).ok()?;
        let lm_head_hidden_buf = trainer.zeros(buf_size).ok()?;

        Some(InstructGpuTrainingState {
            layer_inputs,
            final_norm_weight,
            blocks_output,
            grad_buf_a,
            grad_buf_b,
            grad_final_norm_weight,
            embed_transposed,
            embed_original,
            logits_buf,
            grad_hidden_buf,
            output_scratch,
            grad_upload_buf,
            fwd_scratch_a,
            fwd_scratch_b,
            lm_head_hidden_buf,
        })
    }

    /// Initialize NF4 LoRA training state: gradient workspace + per-layer optimizer states.
    #[cfg(feature = "cuda")]
    fn try_init_nf4_lora_training(
        cuda_trainer: Option<&CudaTrainer>,
        cuda_blocks: Option<&Vec<CudaBlock>>,
        model_config: &TransformerConfig,
        config: &InstructConfig,
    ) -> (Option<CudaLoraGradWorkspace>, Option<Vec<GpuLoraOptimizerState>>) {
        let trainer = match cuda_trainer {
            Some(t) => t,
            None => return (None, None),
        };
        let blocks = match cuda_blocks {
            Some(b) => b,
            None => return (None, None),
        };

        let grad_ws =
            match CudaLoraGradWorkspace::new(trainer.context(), model_config, config.lora_rank) {
                Ok(ws) => ws,
                Err(e) => {
                    eprintln!("[CUDA] NF4 LoRA grad workspace alloc failed: {e}");
                    return (None, None);
                }
            };

        let mut opt_states = Vec::with_capacity(blocks.len());
        for (i, block) in blocks.iter().enumerate() {
            match block.init_lora_optimizer_state() {
                Ok(state) => opt_states.push(state),
                Err(e) => {
                    eprintln!("[CUDA] NF4 LoRA optimizer init failed (layer {i}): {e}");
                    return (None, None);
                }
            }
        }

        eprintln!(
            "[CUDA] NF4 QLoRA training initialized: {} layers, rank={}, scale={:.2}",
            blocks.len(),
            config.lora_rank,
            config.lora_alpha / config.lora_rank as f32,
        );

        (Some(grad_ws), Some(opt_states))
    }

    /// GPU-accelerated forward pass saving layer inputs for backward.
    #[cfg(feature = "cuda")]
    #[allow(unsafe_code)]
    fn forward_cuda_training(
        model: &Transformer,
        token_ids: &[u32],
        trainer: &CudaTrainer,
        cuda_blocks: &mut [CudaBlock],
        training_state: &mut InstructGpuTrainingState,
        shared_scratch: &mut Option<CudaBlockScratch>,
    ) -> Option<()> {
        let seq_len = token_ids.len();
        let hidden_size = model.config.hidden_size;
        let max_seq_len = model
            .config
            .max_position_embeddings
            .min(training_state.layer_inputs.first().map_or(seq_len, |b| b.len() / hidden_size));

        if seq_len > max_seq_len {
            eprintln!("[CUDA] seq_len ({seq_len}) > max_seq_len ({max_seq_len}), truncating");
            return None;
        }

        // Embed on CPU, then zero-pad to max_seq_len for GPU buffer compatibility.
        // Training state buffers are allocated at max_seq_len * hidden_size;
        // copy_from_buffer_async requires exact size match.
        let hidden = model.embed_tokens.forward(token_ids);
        let hidden_data = hidden.data();
        let hidden_slice = hidden_data.as_slice().expect("contiguous hidden");

        // KAIZEN-062: Upload hidden states into pre-allocated GPU buffer.
        // Partial write via copy_from_host_at — padded positions are irrelevant
        // (block.forward uses seq_len for attention masking and GEMM dimensions).
        training_state
            .fwd_scratch_a
            .copy_from_host_at(hidden_slice, 0)
            .map_err(|e| {
                eprintln!("[CUDA] upload failed: {e}");
            })
            .ok()?;

        // KAIZEN-062: Ping-pong with pre-allocated forward scratch buffers.
        // Eliminates 2× cuMemAlloc + 2× cuMemFree per forward pass.
        let scratch_a_ptr: *mut GpuBuffer<f32> =
            std::ptr::from_mut(&mut training_state.fwd_scratch_a);
        let scratch_b_ptr: *mut GpuBuffer<f32> =
            std::ptr::from_mut(&mut training_state.fwd_scratch_b);
        let mut input_is_a = true;

        // Run through CUDA transformer blocks, saving inputs
        let stream = trainer.stream();
        for (i, block) in cuda_blocks.iter_mut().enumerate() {
            // SAFETY: scratch_a_ptr and scratch_b_ptr point to disjoint struct fields.
            // Only one is written (output) while the other is read (input) per iteration.
            let (gpu_input, gpu_output) = unsafe {
                if input_is_a {
                    (&*scratch_a_ptr, &mut *scratch_b_ptr)
                } else {
                    (&*scratch_b_ptr, &mut *scratch_a_ptr)
                }
            };

            // Save input for backward pass
            // SAFETY: layer_inputs[i] is a disjoint field from fwd_scratch_a/b.
            unsafe {
                if let Err(e) =
                    training_state.layer_inputs[i].copy_from_buffer_async(gpu_input, stream)
                {
                    eprintln!(
                        "[CUDA] Layer {i} input save failed: {e} (src={}, dst={})",
                        gpu_input.len(),
                        training_state.layer_inputs[i].len()
                    );
                    return None;
                }
            }

            // Forward uses actual seq_len for attention masking; padded positions
            // produce zeros that don't affect loss (loss only covers actual tokens).
            if let Err(e) =
                block.forward(gpu_input, gpu_output, seq_len, stream, shared_scratch.as_mut())
            {
                eprintln!("[CUDA] Layer {i} forward failed: {e}");
                return None;
            }
            input_is_a = !input_is_a;
        }

        // After ping-pong: result is in the buffer that would be "input" for the next iteration
        let final_output = unsafe {
            if input_is_a {
                &*scratch_a_ptr
            } else {
                &*scratch_b_ptr
            }
        };

        // Save blocks output for RMSNorm backward
        // SAFETY: blocks_output is a disjoint field from fwd_scratch_a/b.
        unsafe {
            if let Err(e) =
                training_state.blocks_output.copy_from_buffer_async(final_output, stream)
            {
                eprintln!("[CUDA] blocks_output save failed: {e}");
                return None;
            }
        }

        // KAIZEN-066: GPU RMSNorm forward — output goes directly to lm_head_hidden_buf.
        // Eliminates ~5MB D2H download + CPU RMSNorm + ~5MB H2D upload.
        // Same CUDA stream guarantees blocks_output copy and RMSNorm are ordered.
        crate::autograd::cuda_backward::rms_norm_forward(
            final_output,
            &training_state.final_norm_weight,
            &mut training_state.lm_head_hidden_buf,
            seq_len as u32,
            hidden_size as u32,
            stream,
        )
        .map_err(|e| {
            eprintln!("[CUDA] GPU RMSNorm forward failed: {e}");
        })
        .ok()?;

        Some(())
    }

    /// NF4 QLoRA backward pass through all GPU transformer blocks.
    ///
    /// Computes gradient flow through frozen NF4 weights and updates LoRA
    /// adapters. After each block backward, immediately runs the LoRA optimizer
    /// step (grad workspace is shared across layers).
    #[cfg(feature = "cuda")]
    #[allow(unsafe_code)]
    fn backward_nf4_gpu_blocks(&mut self, grad_final_hidden: &[f32], seq_len: usize) -> Option<()> {
        let hidden_size = self.model.config.hidden_size;

        // Upload gradient and run RMSNorm backward in a scope to release borrows
        // before calling the shared block-loop.
        {
            let trainer = self.cuda_trainer.as_ref()?;
            let training_state = self.gpu_training.as_mut()?;
            let stream = trainer.stream();

            // KAIZEN-062: Upload gradient directly to pre-allocated GPU buffer via partial write.
            training_state.grad_upload_buf.copy_from_host_at(grad_final_hidden, 0).ok()?;

            // RMSNorm backward on GPU
            crate::autograd::cuda_backward::rms_norm_backward(
                &training_state.blocks_output,
                &training_state.final_norm_weight,
                &training_state.grad_upload_buf,
                &mut training_state.grad_buf_a,
                &mut training_state.grad_final_norm_weight,
                seq_len as u32,
                hidden_size as u32,
                1e-5_f32,
                stream,
            )
            .ok()?;
        }

        self.backward_nf4_gpu_blocks_loop(seq_len)
    }

    /// GPU-resident backward: gradient already in grad_hidden_buf from GEMM (KAIZEN-065).
    ///
    /// Same as backward_nf4_gpu_blocks but reads gradient directly from
    /// grad_hidden_buf instead of uploading from CPU. Eliminates:
    /// - ~5MB D2H download (grad_hidden_buf → CPU)
    /// - ~5MB H2D upload (CPU → grad_upload_buf)
    /// - 1× stream.synchronize() GPU drain point
    /// - 1× Vec<f32> heap allocation (~5MB)
    #[cfg(feature = "cuda")]
    #[allow(unsafe_code)]
    fn backward_nf4_gpu_blocks_gpu_resident(&mut self, seq_len: usize) -> Option<()> {
        let hidden_size = self.model.config.hidden_size;

        // KAIZEN-065: grad_hidden_buf already contains the gradient from lm_head backward GEMM.
        // No D2H download or H2D upload needed — both are on the same CUDA stream,
        // so the GEMM output is guaranteed complete before rms_norm_backward reads it.
        {
            let trainer = self.cuda_trainer.as_ref()?;
            let training_state = self.gpu_training.as_mut()?;
            let stream = trainer.stream();

            // RMSNorm backward on GPU — read gradient directly from grad_hidden_buf
            crate::autograd::cuda_backward::rms_norm_backward(
                &training_state.blocks_output,
                &training_state.final_norm_weight,
                &training_state.grad_hidden_buf,
                &mut training_state.grad_buf_a,
                &mut training_state.grad_final_norm_weight,
                seq_len as u32,
                hidden_size as u32,
                1e-5_f32,
                stream,
            )
            .ok()?;
        }

        self.backward_nf4_gpu_blocks_loop(seq_len)
    }

    /// Shared backward loop for NF4 blocks — called by both CPU-upload and
    /// GPU-resident backward paths after RMSNorm backward completes.
    #[cfg(feature = "cuda")]
    #[allow(unsafe_code)]
    fn backward_nf4_gpu_blocks_loop(&mut self, seq_len: usize) -> Option<()> {
        let trainer = self.cuda_trainer.as_ref()?;
        let lr = self.optimizer.lr();
        let stream = trainer.stream();

        let training_state = self.gpu_training.as_mut()?;
        let blocks = self.cuda_blocks.as_mut()?;
        let shared_scratch = self.shared_scratch.as_mut()?;
        let grad_lora = self.cuda_lora_grad_workspace.as_mut()?;
        let opt_states = self.cuda_lora_optimizer_states.as_mut()?;

        // Backward through blocks in reverse, interleaved with optimizer
        let num_layers = blocks.len();

        let grad_a_ptr: *mut GpuBuffer<f32> = std::ptr::from_mut(&mut training_state.grad_buf_a);
        let grad_b_ptr: *mut GpuBuffer<f32> = std::ptr::from_mut(&mut training_state.grad_buf_b);
        let mut grad_output_is_a = true;

        self.nf4_lora_step += 1;
        let step = self.nf4_lora_step;

        // KAIZEN-045: Use pre-allocated output_scratch from training state
        let output_scratch_ptr: *mut GpuBuffer<f32> =
            std::ptr::from_mut(&mut training_state.output_scratch);

        for layer_idx in (0..num_layers).rev() {
            // SAFETY: grad_a_ptr and grad_b_ptr point to disjoint fields.
            let (grad_output, grad_input) = unsafe {
                if grad_output_is_a {
                    (&*grad_a_ptr, &mut *grad_b_ptr)
                } else {
                    (&*grad_b_ptr, &mut *grad_a_ptr)
                }
            };

            // SAFETY: output_scratch_ptr points to a disjoint field of training_state.
            blocks[layer_idx]
                .backward_nf4(
                    &training_state.layer_inputs[layer_idx],
                    grad_output,
                    grad_input,
                    unsafe { &mut *output_scratch_ptr },
                    seq_len,
                    stream,
                    shared_scratch,
                    grad_lora,
                )
                .ok()?;

            // Immediately apply LoRA optimizer step
            blocks[layer_idx]
                .lora_optimizer_step(
                    &mut opt_states[layer_idx],
                    step,
                    lr,
                    0.9,   // beta1
                    0.999, // beta2
                    1e-8,  // eps
                    0.01,  // weight_decay
                    stream,
                    grad_lora,
                )
                .ok()?;

            grad_output_is_a = !grad_output_is_a;
        }

        stream.synchronize().ok()?;

        Some(())
    }

    /// GPU-accelerated forward pass (inference-only, no layer input saving).
    #[cfg(feature = "cuda")]
    fn forward_cuda_inference(
        model: &Transformer,
        token_ids: &[u32],
        trainer: &CudaTrainer,
        cuda_blocks: &mut [CudaBlock],
        shared_scratch: &mut Option<CudaBlockScratch>,
    ) -> Option<Vec<f32>> {
        let seq_len = token_ids.len();
        let hidden_size = model.config.hidden_size;

        let hidden = model.embed_tokens.forward(token_ids);
        let hidden_data = hidden.data();
        let hidden_slice = hidden_data.as_slice().expect("contiguous hidden");

        let mut gpu_input = trainer.upload(hidden_slice).ok()?;
        let mut gpu_output = trainer.zeros(seq_len * hidden_size).ok()?;

        let stream = trainer.stream();
        for (i, block) in cuda_blocks.iter_mut().enumerate() {
            if let Err(e) =
                block.forward(&gpu_input, &mut gpu_output, seq_len, stream, shared_scratch.as_mut())
            {
                eprintln!("[CUDA] Layer {i} forward failed: {e}");
                return None;
            }
            std::mem::swap(&mut gpu_input, &mut gpu_output);
        }

        if let Err(e) = stream.synchronize() {
            eprintln!("[CUDA] Stream sync failed: {e}");
            return None;
        }

        let result_data = trainer.download(&gpu_input).ok()?;

        if result_data.iter().any(|v| !v.is_finite()) {
            return None;
        }

        let result_tensor = crate::Tensor::from_vec(result_data, false);
        let normed = model.norm.forward_batched(&result_tensor, seq_len, hidden_size);
        let normed_data = normed.data();
        let normed_slice = normed_data.as_slice().expect("contiguous normed");
        Some(normed_slice.to_vec())
    }

    /// Forward pass dispatching to GPU when available.
    /// Returns logits as flat Vec<f32> of shape [seq_len, vocab_size].
    ///
    /// lm_head GEMM runs on GPU: hidden[seq, hidden] @ embed_T[hidden, vocab] → logits[seq, vocab]
    #[cfg(feature = "cuda")]
    fn forward_logits_gpu(&mut self, token_ids: &[u32]) -> Option<Vec<f32>> {
        let seq_len = token_ids.len();
        let vocab_size = self.model.config().vocab_size;
        let hidden_size = self.model.config().hidden_size;

        // Get normed hidden states. Training path writes directly to lm_head_hidden_buf (KAIZEN-066).
        // Inference path returns CPU Vec that needs upload.
        if self.gpu_training.is_some() {
            let (trainer, blocks) = match (&self.cuda_trainer, &mut self.cuda_blocks) {
                (Some(ref t), Some(ref mut b)) => (t, b),
                _ => return None,
            };
            let mut training = self.gpu_training.take();
            let result = Self::forward_cuda_training(
                &self.model,
                token_ids,
                trainer,
                blocks,
                training.as_mut().expect("gpu_training was Some"),
                &mut self.shared_scratch,
            );
            self.gpu_training = training;
            result?; // lm_head_hidden_buf is ready
        } else {
            let (trainer, blocks) = match (&self.cuda_trainer, &mut self.cuda_blocks) {
                (Some(ref t), Some(ref mut b)) => (t, b),
                _ => return None,
            };
            let normed_hidden = Self::forward_cuda_inference(
                &self.model,
                token_ids,
                trainer,
                blocks,
                &mut self.shared_scratch,
            )?;
            // Inference path: upload CPU normed hidden to GPU
            let training = self.gpu_training.as_mut()?;
            training
                .lm_head_hidden_buf
                .copy_from_host_at(&normed_hidden, 0)
                .map_err(|e| {
                    eprintln!("[CUDA] lm_head forward: hidden upload failed: {e}");
                })
                .ok()?;
        }

        // GPU GEMM: logits[seq, vocab] = hidden[seq, hidden] @ embed_T[hidden, vocab]
        let trainer = self.cuda_trainer.as_ref()?;
        let training = self.gpu_training.as_mut()?;
        let stream = trainer.stream();

        if let Err(e) = gemm_forward(
            &training.lm_head_hidden_buf,
            &training.embed_transposed,
            &mut training.logits_buf,
            seq_len as u32,
            hidden_size as u32,
            vocab_size as u32,
            stream,
        ) {
            eprintln!("[CUDA] lm_head forward GEMM failed: {e}");
            return None;
        }

        if let Err(e) = stream.synchronize() {
            eprintln!("[CUDA] lm_head forward sync failed: {e}");
            return None;
        }

        // Download only the logits we need (seq_len * vocab_size), not the full max_seq_len buffer
        let full_logits = trainer
            .download(&training.logits_buf)
            .map_err(|e| {
                eprintln!("[CUDA] lm_head forward: logits download failed: {e}");
            })
            .ok()?;

        Some(full_logits[..seq_len * vocab_size].to_vec())
    }

    /// GPU forward pass with logits staying GPU-resident (KAIZEN-064).
    ///
    /// Same as `forward_logits_gpu` but skips the logits D2H download (~296MB for Qwen3-4B).
    /// After this call, `training.logits_buf` contains logits[seq_len, vocab_size] on GPU.
    /// Returns true on success, false on failure.
    #[cfg(feature = "cuda")]
    fn forward_logits_gpu_resident(&mut self, token_ids: &[u32]) -> bool {
        let seq_len = token_ids.len();
        let vocab_size = self.model.config().vocab_size;
        let hidden_size = self.model.config().hidden_size;

        // KAIZEN-066: Training path writes normed hidden directly to lm_head_hidden_buf (GPU-resident).
        // Inference path returns CPU Vec that needs upload.
        if self.gpu_training.is_some() {
            let (trainer, blocks) = match (&self.cuda_trainer, &mut self.cuda_blocks) {
                (Some(ref t), Some(ref mut b)) => (t, b),
                _ => return false,
            };
            let mut training = self.gpu_training.take();
            let result = Self::forward_cuda_training(
                &self.model,
                token_ids,
                trainer,
                blocks,
                training.as_mut().expect("gpu_training was Some"),
                &mut self.shared_scratch,
            );
            self.gpu_training = training;
            if result.is_none() {
                return false;
            }
            // lm_head_hidden_buf is ready
        } else {
            let (trainer, blocks) = match (&self.cuda_trainer, &mut self.cuda_blocks) {
                (Some(ref t), Some(ref mut b)) => (t, b),
                _ => return false,
            };
            let normed_hidden = match Self::forward_cuda_inference(
                &self.model,
                token_ids,
                trainer,
                blocks,
                &mut self.shared_scratch,
            ) {
                Some(h) => h,
                None => return false,
            };
            // Inference path: upload CPU normed hidden to GPU
            let training = match self.gpu_training.as_mut() {
                Some(t) => t,
                None => return false,
            };
            if training.lm_head_hidden_buf.copy_from_host_at(&normed_hidden, 0).is_err() {
                eprintln!("[CUDA] lm_head forward: hidden upload failed");
                return false;
            }
        }

        // GPU GEMM: logits[seq, vocab] = hidden[seq, hidden] @ embed_T[hidden, vocab]
        let (trainer, training) = match (&self.cuda_trainer, &mut self.gpu_training) {
            (Some(ref t), Some(ref mut tr)) => (t, tr),
            _ => return false,
        };
        let stream = trainer.stream();

        if gemm_forward(
            &training.lm_head_hidden_buf,
            &training.embed_transposed,
            &mut training.logits_buf,
            seq_len as u32,
            hidden_size as u32,
            vocab_size as u32,
            stream,
        )
        .is_err()
        {
            eprintln!("[CUDA] lm_head forward GEMM failed");
            return false;
        }

        // KAIZEN-064: No logits download — logits stay in training.logits_buf for fused kernel.
        // KAIZEN-067: No stream.synchronize() needed — fused_causal_cross_entropy_cuda runs
        // on the same CUDA stream, so GEMM output is guaranteed complete before kernel reads.
        true
    }

    /// Check if this pipeline is using CUDA acceleration.
    #[must_use]
    pub fn is_cuda(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.cuda_blocks.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Get GPU device name, or `None` if not using CUDA.
    #[must_use]
    pub fn gpu_name(&self) -> Option<String> {
        #[cfg(feature = "cuda")]
        {
            self.cuda_trainer.as_ref().map(CudaTrainer::device_name)
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    /// Get total GPU memory in bytes, or `None` if not using CUDA.
    #[must_use]
    pub fn gpu_total_memory(&self) -> Option<usize> {
        #[cfg(feature = "cuda")]
        {
            self.cuda_trainer.as_ref().map(CudaTrainer::total_memory)
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    /// Summary of pipeline configuration.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "InstructPipeline: {} LoRA layers, rank={}, alpha={:.1}{}",
            self.lora_layers.len(),
            self.config.lora_rank,
            self.config.lora_alpha,
            if self.config.quantize_nf4 { ", NF4 QLoRA" } else { "" },
        )
    }

    /// Get a reference to the tokenizer, if loaded.
    #[must_use]
    pub fn tokenizer(&self) -> Option<&HfTokenizer> {
        self.tokenizer.as_ref()
    }

    /// Autoregressive text generation with LoRA adapters (entrenar#246).
    ///
    /// Generates tokens one at a time using the transformer + LoRA forward pass.
    /// Supports greedy decoding (temperature=0) and temperature-scaled sampling
    /// with optional top-k filtering.
    ///
    /// # Arguments
    /// * `prompt` - Input text to continue from
    /// * `config` - Generation parameters (max tokens, temperature, top-k)
    ///
    /// # Returns
    /// Generated text (excluding the input prompt)
    ///
    /// # Errors
    /// Returns error if no tokenizer is loaded.
    pub fn generate(&self, prompt: &str, config: &GenerateConfig) -> crate::Result<String> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            crate::Error::ConfigError("No tokenizer loaded — cannot generate text".into())
        })?;

        let mut token_ids = tokenizer.encode(prompt);
        let prompt_len = token_ids.len();
        let eos_token = tokenizer.eos_id().unwrap_or(151643); // Qwen2 default EOS

        let vocab_size = self.model.config().vocab_size;

        for _ in 0..config.max_new_tokens {
            // Truncate to max_seq_len if needed
            if token_ids.len() >= self.config.max_seq_len {
                break;
            }

            // Forward pass with LoRA
            let hidden = self.model.forward_hidden_with_lora(&token_ids, &self.lora_layers);
            let seq_len = token_ids.len();
            let hidden_size = self.model.config().hidden_size;

            // Apply lm_head to get logits
            let lm_weight = self.model.lm_head_weight();
            let logits =
                crate::autograd::matmul(&hidden, lm_weight, seq_len, hidden_size, vocab_size);

            // Extract logits for last position
            let logits_data = logits.data();
            let logits_slice = logits_data.as_slice().unwrap_or(&[]);
            let last_pos_start = (seq_len - 1) * vocab_size;
            let last_pos_logits = &logits_slice[last_pos_start..last_pos_start + vocab_size];

            // Sample next token
            let next_token = sample_token(last_pos_logits, config.temperature, config.top_k);

            if next_token == eos_token {
                break;
            }

            // Check stop tokens
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            token_ids.push(next_token);
        }

        // Decode only the generated part (not the prompt)
        let generated_ids = &token_ids[prompt_len..];
        Ok(tokenizer.decode(generated_ids))
    }

    /// Generate a chat response using ChatML format (entrenar#246).
    ///
    /// Formats messages as ChatML (`<|im_start|>` / `<|im_end|>`) and generates
    /// the assistant's response.
    ///
    /// # Arguments
    /// * `system` - System prompt
    /// * `user_message` - User's input message
    /// * `config` - Generation parameters
    ///
    /// # Returns
    /// The assistant's generated response text.
    ///
    /// # Errors
    /// Returns error if no tokenizer is loaded.
    pub fn generate_chat(
        &self,
        system: &str,
        user_message: &str,
        config: &GenerateConfig,
    ) -> crate::Result<String> {
        let prompt = format!(
            "<|im_start|>system\n{system}<|im_end|>\n\
             <|im_start|>user\n{user_message}<|im_end|>\n\
             <|im_start|>assistant\n"
        );

        let mut response = self.generate(&prompt, config)?;

        // Strip trailing <|im_end|> if present
        if let Some(stripped) = response.strip_suffix("<|im_end|>") {
            response = stripped.to_string();
        }

        Ok(response)
    }
}

/// Configuration for autoregressive text generation.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Maximum number of new tokens to generate (default: 256)
    pub max_new_tokens: usize,
    /// Sampling temperature (0.0 = greedy/argmax, >0 = stochastic)
    pub temperature: f32,
    /// Top-k filtering (0 = disabled, >0 = keep only top-k logits)
    pub top_k: usize,
    /// Additional stop token IDs (generation stops on EOS or any of these)
    pub stop_tokens: Vec<u32>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self { max_new_tokens: 256, temperature: 0.7, top_k: 50, stop_tokens: Vec::new() }
    }
}

impl GenerateConfig {
    /// Create a greedy decoding config (deterministic, always picks highest probability token).
    #[must_use]
    pub fn greedy(max_new_tokens: usize) -> Self {
        Self { max_new_tokens, temperature: 0.0, top_k: 0, stop_tokens: Vec::new() }
    }
}

/// Sample a token from logits with temperature and top-k filtering.
fn sample_token(logits: &[f32], temperature: f32, top_k: usize) -> u32 {
    if temperature <= 0.0 || top_k == 1 {
        // Greedy: argmax
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx as u32);
    }

    // Temperature scaling
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();

    // Top-k filtering
    let mut indices_and_logits: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
    indices_and_logits
        .sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let k = if top_k > 0 && top_k < indices_and_logits.len() {
        top_k
    } else {
        indices_and_logits.len()
    };
    let top = &indices_and_logits[..k];

    // Softmax over top-k
    let max_logit = top[0].1;
    let exps: Vec<f32> = top.iter().map(|(_, l)| (l - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

    // Sample from distribution (simple linear scan)
    let r: f32 = simple_random();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return top[i].0 as u32;
        }
    }

    // Fallback to top-1
    top[0].0 as u32
}

/// Simple pseudo-random float in [0, 1) using thread-local state.
/// Not cryptographically secure but sufficient for sampling.
fn simple_random() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = Cell::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42)
        );
    }
    STATE.with(|s| {
        // xorshift64
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x >> 40) as f32 / (1u64 << 24) as f32
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruct_pipeline_new() {
        let model_config = TransformerConfig::tiny();
        let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
        let pipeline = InstructPipeline::new(&model_config, instruct_config);
        // 2 layers * 2 (Q+V) = 4 LoRA layers for tiny config
        assert_eq!(pipeline.lora_layers.len(), 4);
    }

    #[test]
    fn test_instruct_train_step() {
        let model_config = TransformerConfig::tiny();
        let instruct_config =
            InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
        let mut pipeline = InstructPipeline::new(&model_config, instruct_config);

        let prompt_ids: Vec<u32> = (0..10).collect();
        let response_ids: Vec<u32> = (10..20).collect();

        let result = pipeline.train_step(&prompt_ids, &response_ids);
        assert!(result.loss >= 0.0);
        assert_eq!(result.num_response_tokens, 10);
        assert!(result.perplexity >= 1.0);
    }

    #[test]
    fn test_instruct_evaluate() {
        let model_config = TransformerConfig::tiny();
        let instruct_config =
            InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
        let pipeline = InstructPipeline::new(&model_config, instruct_config);

        let prompts = vec![vec![0u32, 1, 2, 3, 4]];
        let responses = vec![vec![5u32, 6, 7, 8, 9]];

        let result = pipeline.evaluate(&prompts, &responses);
        assert!(result.avg_loss >= 0.0);
        assert_eq!(result.total_response_tokens, 5);
    }

    #[test]
    fn test_empty_response_noop() {
        let model_config = TransformerConfig::tiny();
        let instruct_config =
            InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
        let mut pipeline = InstructPipeline::new(&model_config, instruct_config);

        let result = pipeline.train_step(&[0, 1, 2], &[]);
        assert_eq!(result.loss, 0.0);
        assert_eq!(result.num_response_tokens, 0);
    }

    #[test]
    fn test_generate_config_default() {
        let config = GenerateConfig::default();
        assert_eq!(config.max_new_tokens, 256);
        assert!((config.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 50);
        assert!(config.stop_tokens.is_empty());
    }

    #[test]
    fn test_generate_config_greedy() {
        let config = GenerateConfig::greedy(128);
        assert_eq!(config.max_new_tokens, 128);
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_k, 0);
    }

    #[test]
    fn test_sample_token_greedy() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let token = sample_token(&logits, 0.0, 0);
        assert_eq!(token, 3); // index of 0.9 (highest)
    }

    #[test]
    fn test_sample_token_greedy_top_k_1() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let token = sample_token(&logits, 1.0, 1);
        assert_eq!(token, 3); // top-1 is always argmax
    }

    #[test]
    fn test_sample_token_temperature_sampling() {
        // With very high temperature, distribution flattens — token 3 shouldn't always win
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let mut seen = std::collections::HashSet::new();
        for _ in 0..100 {
            let token = sample_token(&logits, 10.0, 0);
            seen.insert(token);
        }
        // With temp=10.0, we should see multiple different tokens
        assert!(seen.len() > 1, "Expected diversity with high temperature");
    }

    #[test]
    fn test_generate_no_tokenizer_errors() {
        let model_config = TransformerConfig::tiny();
        let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
        let pipeline = InstructPipeline::new(&model_config, instruct_config);
        // Pipeline created with new() has no tokenizer
        let result = pipeline.generate("hello", &GenerateConfig::greedy(10));
        assert!(result.is_err());
    }

    #[test]
    fn test_simple_random_range() {
        for _ in 0..1000 {
            let r = simple_random();
            assert!((0.0..1.0).contains(&r), "Random value {r} out of [0, 1) range");
        }
    }

    #[test]
    fn test_instruct_config_default() {
        let config = InstructConfig::default();
        assert_eq!(config.lora_rank, 16);
        assert!((config.lora_alpha - 32.0).abs() < f32::EPSILON);
        assert!((config.learning_rate - 2e-4).abs() < f32::EPSILON);
        assert_eq!(config.epochs, 3);
        assert_eq!(config.max_seq_len, 512);
        assert_eq!(config.gradient_clip_norm, Some(1.0));
        assert!(!config.quantize_nf4);
    }

    #[test]
    fn test_instruct_step_result_fields() {
        let result = InstructStepResult { loss: 2.5, num_response_tokens: 10, perplexity: 12.18 };
        assert!((result.loss - 2.5).abs() < f32::EPSILON);
        assert_eq!(result.num_response_tokens, 10);
        assert!((result.perplexity - 12.18).abs() < 0.01);
    }

    #[test]
    fn test_instruct_batch_result_fields() {
        let result = InstructBatchResult {
            avg_loss: 1.5,
            total_response_tokens: 100,
            perplexity: 4.48,
            grad_norm: 0.5,
        };
        assert!((result.avg_loss - 1.5).abs() < f32::EPSILON);
        assert_eq!(result.total_response_tokens, 100);
        assert!((result.perplexity - 4.48).abs() < 0.01);
        assert!((result.grad_norm - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tokenize_byte_fallback_no_tokenizer() {
        let model_config = TransformerConfig::tiny();
        let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
        let pipeline = InstructPipeline::new(&model_config, instruct_config);
        assert!(!pipeline.has_tokenizer());
        let tokens = pipeline.tokenize("AB");
        // Byte-level fallback: 'A' = 65, 'B' = 66
        assert_eq!(tokens, vec![65, 66]);
    }

    #[test]
    fn test_tokenize_byte_fallback_utf8() {
        let model_config = TransformerConfig::tiny();
        let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
        let pipeline = InstructPipeline::new(&model_config, instruct_config);
        let tokens = pipeline.tokenize("\u{00e9}"); // 'é' = 2 UTF-8 bytes: 0xC3, 0xA9
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], 0xC3);
        assert_eq!(tokens[1], 0xA9);
    }

    #[test]
    fn test_num_trainable_parameters() {
        let model_config = TransformerConfig::tiny();
        let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
        let pipeline = InstructPipeline::new(&model_config, instruct_config);
        let count = pipeline.num_trainable_parameters();
        // 4 LoRA layers * 2 * rank(4) * 1 = 32
        assert_eq!(count, 32);
    }

    #[test]
    fn test_set_and_get_learning_rate() {
        let model_config = TransformerConfig::tiny();
        let instruct_config =
            InstructConfig { lora_rank: 4, learning_rate: 1e-3, ..InstructConfig::default() };
        let mut pipeline = InstructPipeline::new(&model_config, instruct_config);
        assert!((pipeline.learning_rate() - 1e-3).abs() < 1e-6);
        pipeline.set_learning_rate(5e-4);
        assert!((pipeline.learning_rate() - 5e-4).abs() < 1e-6);
    }

    #[test]
    fn test_set_model_path() {
        let model_config = TransformerConfig::tiny();
        let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
        let mut pipeline = InstructPipeline::new(&model_config, instruct_config);
        assert!(pipeline.model_dir.is_none());
        pipeline.set_model_path(std::path::Path::new("/tmp/test-model"));
        assert_eq!(
            pipeline.model_dir.as_ref().map(|p| p.to_str().unwrap_or("")),
            Some("/tmp/test-model")
        );
    }

    #[test]
    fn test_is_cuda_false_without_gpu() {
        let model_config = TransformerConfig::tiny();
        let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
        let pipeline = InstructPipeline::new(&model_config, instruct_config);
        assert!(!pipeline.is_cuda());
    }

    #[test]
    fn test_gpu_name_none_without_gpu() {
        let model_config = TransformerConfig::tiny();
        let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
        let pipeline = InstructPipeline::new(&model_config, instruct_config);
        assert!(pipeline.gpu_name().is_none());
    }

    #[test]
    fn test_gpu_total_memory_none_without_gpu() {
        let model_config = TransformerConfig::tiny();
        let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
        let pipeline = InstructPipeline::new(&model_config, instruct_config);
        assert!(pipeline.gpu_total_memory().is_none());
    }

    #[test]
    fn test_summary_format() {
        let model_config = TransformerConfig::tiny();
        let instruct_config =
            InstructConfig { lora_rank: 8, lora_alpha: 16.0, ..InstructConfig::default() };
        let pipeline = InstructPipeline::new(&model_config, instruct_config);
        let summary = pipeline.summary();
        assert!(summary.contains("InstructPipeline"));
        assert!(summary.contains("rank=8"));
        assert!(summary.contains("alpha=16.0"));
        assert!(!summary.contains("NF4 QLoRA"));
    }

    #[test]
    fn test_summary_nf4_label() {
        let model_config = TransformerConfig::tiny();
        let instruct_config =
            InstructConfig { lora_rank: 8, quantize_nf4: true, ..InstructConfig::default() };
        let pipeline = InstructPipeline::new(&model_config, instruct_config);
        let summary = pipeline.summary();
        assert!(summary.contains("NF4 QLoRA"));
    }

    #[test]
    fn test_tokenizer_none_for_new_pipeline() {
        let model_config = TransformerConfig::tiny();
        let instruct_config = InstructConfig::default();
        let pipeline = InstructPipeline::new(&model_config, instruct_config);
        assert!(pipeline.tokenizer().is_none());
    }

    #[test]
    fn test_train_step_truncation() {
        // Test that sequences exceeding max_seq_len are truncated
        let model_config = TransformerConfig::tiny();
        let instruct_config =
            InstructConfig { lora_rank: 4, max_seq_len: 16, ..InstructConfig::default() };
        let mut pipeline = InstructPipeline::new(&model_config, instruct_config);

        let prompt_ids: Vec<u32> = (0..10).collect();
        let response_ids: Vec<u32> = (10..30).collect(); // total = 30 > 16

        let result = pipeline.train_step(&prompt_ids, &response_ids);
        // Should still compute loss on the truncated sequence
        assert!(result.loss >= 0.0);
    }

    #[test]
    fn test_train_step_short_sequence() {
        let model_config = TransformerConfig::tiny();
        let instruct_config =
            InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
        let mut pipeline = InstructPipeline::new(&model_config, instruct_config);

        // Single prompt + single response token
        let result = pipeline.train_step(&[0], &[1]);
        assert!(result.loss >= 0.0);
        assert_eq!(result.num_response_tokens, 1);
    }

    #[test]
    fn test_evaluate_empty_batch() {
        let model_config = TransformerConfig::tiny();
        let instruct_config =
            InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
        let pipeline = InstructPipeline::new(&model_config, instruct_config);

        let result = pipeline.evaluate(&[], &[]);
        assert_eq!(result.avg_loss, 0.0);
        assert_eq!(result.total_response_tokens, 0);
        assert_eq!(result.grad_norm, 0.0);
    }

    #[test]
    fn test_evaluate_skips_empty_responses() {
        let model_config = TransformerConfig::tiny();
        let instruct_config =
            InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
        let pipeline = InstructPipeline::new(&model_config, instruct_config);

        // Mix of empty and non-empty responses
        let prompts = vec![vec![0u32, 1, 2], vec![3u32, 4, 5]];
        let responses = vec![vec![], vec![6u32, 7, 8]];

        let result = pipeline.evaluate(&prompts, &responses);
        // Only second sample should contribute
        assert_eq!(result.total_response_tokens, 3);
    }

    #[test]
    fn test_evaluate_truncation() {
        let model_config = TransformerConfig::tiny();
        let instruct_config =
            InstructConfig { lora_rank: 4, max_seq_len: 10, ..InstructConfig::default() };
        let pipeline = InstructPipeline::new(&model_config, instruct_config);

        let prompts = vec![vec![0u32; 8]];
        let responses = vec![vec![1u32; 8]]; // total 16, will be truncated to 10

        let result = pipeline.evaluate(&prompts, &responses);
        assert!(result.avg_loss >= 0.0);
    }

    #[test]
    fn test_compute_causal_lm_loss_basic() {
        // Construct simple logits: 3 positions, vocab_size = 5
        let vocab_size = 5;
        let seq_len = 3;
        let mut logits = vec![0.0f32; seq_len * vocab_size];
        // Set position 0 logits to favor token 1
        logits[1] = 10.0; // position 0, token 1
                          // Set position 1 logits to favor token 2
        logits[vocab_size + 2] = 10.0; // position 1, token 2

        let full_ids = vec![0u32, 1, 2]; // targets are shifted: pos0→1, pos1→2
        let (loss, grad) = InstructPipeline::compute_causal_lm_loss(
            &logits, &full_ids, 0, // loss_start
            2, // loss_end
            vocab_size,
        );
        assert!(loss >= 0.0);
        assert!(loss < 1.0, "Loss should be low when logits match targets, got {loss}");
        assert_eq!(grad.len(), seq_len * vocab_size);
    }

    #[test]
    fn test_compute_causal_lm_loss_empty_range() {
        let vocab_size = 5;
        let logits = vec![0.0f32; 15];
        let full_ids = vec![0u32, 1, 2];
        let (loss, grad) =
            InstructPipeline::compute_causal_lm_loss(&logits, &full_ids, 2, 2, vocab_size);
        assert_eq!(loss, 0.0);
        // Gradient should be all zeros (no loss tokens)
        assert!(grad.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_compute_causal_lm_loss_target_out_of_vocab() {
        // Test graceful handling when target >= vocab_size
        let vocab_size = 5;
        let logits = vec![0.0f32; 10]; // 2 positions
        let full_ids = vec![0u32, 100]; // target 100 >= vocab_size 5
        let (loss, _grad) =
            InstructPipeline::compute_causal_lm_loss(&logits, &full_ids, 0, 1, vocab_size);
        // Should not panic; loss should be 0 (skipped)
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_build_lora_layers_count() {
        let model_config = TransformerConfig::tiny();
        let model = crate::transformer::Transformer::new(&model_config);
        let instruct_config = InstructConfig { lora_rank: 8, ..InstructConfig::default() };
        let layers = InstructPipeline::build_lora_layers(&model, &model_config, &instruct_config);
        // 2 layers (tiny) * 2 (Q+V) = 4
        assert_eq!(layers.len(), model_config.num_hidden_layers * 2);
    }

    #[test]
    fn test_sample_token_with_top_k() {
        // With top_k=2, only top-2 logits should be candidates
        let logits = vec![0.0, 0.0, 0.0, 100.0, 99.0]; // tokens 3 and 4 dominate
        for _ in 0..50 {
            let token = sample_token(&logits, 1.0, 2);
            assert!(token == 3 || token == 4, "Expected token 3 or 4 with top_k=2, got {token}");
        }
    }

    #[test]
    fn test_sample_token_empty_logits() {
        let logits: Vec<f32> = vec![];
        let token = sample_token(&logits, 0.0, 0);
        assert_eq!(token, 0); // map_or default
    }

    #[test]
    fn test_sample_token_single_element() {
        let logits = vec![1.0];
        let token = sample_token(&logits, 0.0, 0);
        assert_eq!(token, 0);
    }

    #[test]
    fn test_train_step_with_no_gradient_clip() {
        let model_config = TransformerConfig::tiny();
        let instruct_config = InstructConfig {
            lora_rank: 4,
            max_seq_len: 64,
            gradient_clip_norm: None,
            ..InstructConfig::default()
        };
        let mut pipeline = InstructPipeline::new(&model_config, instruct_config);

        let prompt_ids: Vec<u32> = (0..5).collect();
        let response_ids: Vec<u32> = (5..10).collect();

        let result = pipeline.train_step(&prompt_ids, &response_ids);
        assert!(result.loss >= 0.0);
    }

    #[test]
    fn test_generate_chat_no_tokenizer_errors() {
        let model_config = TransformerConfig::tiny();
        let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
        let pipeline = InstructPipeline::new(&model_config, instruct_config);
        let result = pipeline.generate_chat("system", "hello", &GenerateConfig::greedy(5));
        assert!(result.is_err());
    }

    #[test]
    fn test_perplexity_clamped() {
        // When loss is extremely high, perplexity should be clamped to 1e6
        let model_config = TransformerConfig::tiny();
        let instruct_config =
            InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
        let mut pipeline = InstructPipeline::new(&model_config, instruct_config);

        let result = pipeline.train_step(&[0, 1, 2], &[3, 4, 5]);
        // Perplexity should be clamped
        assert!(result.perplexity <= 1e6);
    }

    #[test]
    fn test_evaluate_multiple_samples() {
        let model_config = TransformerConfig::tiny();
        let instruct_config =
            InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
        let pipeline = InstructPipeline::new(&model_config, instruct_config);

        let prompts = vec![vec![0u32, 1, 2], vec![3u32, 4, 5], vec![6u32, 7, 8]];
        let responses = vec![vec![10u32, 11], vec![12u32, 13, 14], vec![15u32]];

        let result = pipeline.evaluate(&prompts, &responses);
        // Total response tokens: 2 + 3 + 1 = 6
        assert_eq!(result.total_response_tokens, 6);
        assert!(result.avg_loss >= 0.0);
        assert!(result.perplexity >= 1.0);
    }

    #[test]
    fn test_instruct_config_clone() {
        let config = InstructConfig {
            lora_rank: 32,
            lora_alpha: 64.0,
            learning_rate: 1e-5,
            epochs: 5,
            max_seq_len: 1024,
            gradient_clip_norm: Some(2.0),
            quantize_nf4: true,
        };
        let cloned = config.clone();
        assert_eq!(cloned.lora_rank, 32);
        assert!((cloned.lora_alpha - 64.0).abs() < f32::EPSILON);
        assert_eq!(cloned.epochs, 5);
        assert_eq!(cloned.max_seq_len, 1024);
        assert_eq!(cloned.gradient_clip_norm, Some(2.0));
        assert!(cloned.quantize_nf4);
    }

    #[test]
    fn test_step_result_clone_and_debug() {
        let result = InstructStepResult { loss: 1.5, num_response_tokens: 42, perplexity: 4.48 };
        let cloned = result.clone();
        assert_eq!(cloned.num_response_tokens, 42);
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("InstructStepResult"));
    }

    #[test]
    fn test_batch_result_clone_and_debug() {
        let result = InstructBatchResult {
            avg_loss: 0.5,
            total_response_tokens: 200,
            perplexity: 1.65,
            grad_norm: 0.8,
        };
        let cloned = result.clone();
        assert_eq!(cloned.total_response_tokens, 200);
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("InstructBatchResult"));
    }
}
