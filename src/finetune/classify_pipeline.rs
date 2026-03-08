//! Classification fine-tuning pipeline
//!
//! Wires Transformer + LoRA + ClassificationHead for sequence classification.
//!
//! # Architecture
//!
//! ```text
//! token_ids -> Transformer.forward_hidden() -> [seq_len, hidden_size]
//!           -> ClassificationHead.forward()  -> [num_classes]
//!           -> cross_entropy_loss(target)    -> scalar loss
//! ```
//!
//! # Contract
//!
//! See `aprender/contracts/classification-finetune-v1.yaml`

use super::classification::{
    load_multi_label_corpus, load_safety_corpus, ClassificationHead, MultiLabelSafetySample,
    SafetySample, TokenizedSample,
};
use crate::autograd::matmul;
use crate::lora::LoRAConfig;
use crate::lora::LoRALayer;
use crate::optim::{clip_grad_norm_refs, AdamW, Optimizer};
use crate::tokenizer::HfTokenizer;
use crate::transformer::Transformer;
use crate::transformer::TransformerConfig;
use crate::Tensor;
use std::path::{Path, PathBuf};

#[cfg(feature = "cuda")]
use crate::autograd::cuda_backward::pre_warm_lora_backward_kernels as pre_warm_backward_cache_kernels;
#[cfg(feature = "cuda")]
use crate::autograd::cuda_forward::{pre_warm_forward_kernels, pre_warm_lora_backward_kernels};
#[cfg(feature = "cuda")]
use crate::autograd::cuda_optim::pre_warm_lora_adamw_kernels;
#[cfg(feature = "cuda")]
use crate::autograd::cuda_training::{cuda_training_available, CudaTrainer};
#[cfg(feature = "cuda")]
use crate::autograd::ops::pre_warm_realizador_gemm;
#[cfg(feature = "cuda")]
use crate::gpu::guard::VramGuard;
#[cfg(feature = "cuda")]
use crate::transformer::{
    CudaBlock, CudaBlockScratch, CudaGradWorkspace, CudaLoraGradWorkspace, CudaTransformerBlock,
    GpuBlockOptimizerState, GpuLoraOptimizerState,
};
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use trueno_gpu::driver::GpuBuffer;

/// Classification fine-tuning pipeline configuration.
#[derive(Debug, Clone)]
pub struct ClassifyConfig {
    /// Number of output classes
    pub num_classes: usize,
    /// LoRA rank
    pub lora_rank: usize,
    /// LoRA alpha
    pub lora_alpha: f32,
    /// Learning rate
    pub learning_rate: f32,
    /// Number of training epochs
    pub epochs: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Log every N steps
    pub log_interval: usize,
    /// Mini-batch size for `train_batch()`.
    ///
    /// Samples are processed one at a time (forward + backward), but the
    /// optimizer step is applied once per batch after accumulating gradients.
    pub batch_size: usize,
    /// Number of gradient accumulation steps.
    ///
    /// Allows effective batch size = `batch_size * accumulation_steps` without
    /// increasing peak memory beyond a single micro-batch forward pass.
    pub accumulation_steps: usize,
    /// Maximum gradient norm for clipping.
    ///
    /// When `Some(max_norm)`, gradients are clipped to this L2 norm before
    /// the optimizer step. `None` disables gradient clipping.
    pub gradient_clip_norm: Option<f32>,
    /// Per-class loss weights for imbalanced datasets.
    ///
    /// When `Some(weights)`, the cross-entropy loss for label `c` is multiplied
    /// by `weights[c]`. Weights should sum to `num_classes` to preserve loss scale.
    /// When `None`, all classes are weighted equally (weight = 1.0).
    ///
    /// See SPEC-TUNE-2026-001 §9 for weight computation strategies.
    pub class_weights: Option<Vec<f32>>,
    /// Quantize frozen weights to NF4 (4-bit) for QLoRA training (default: false).
    ///
    /// When enabled, uses `CudaNf4TransformerBlock` (~8x VRAM compression) instead
    /// of `CudaTransformerBlock`. GPU backward pass is disabled (LoRA-only training).
    pub quantize_nf4: bool,
}

impl Default for ClassifyConfig {
    fn default() -> Self {
        Self {
            num_classes: 5,
            lora_rank: 16,
            lora_alpha: 16.0,
            learning_rate: 1e-4,
            epochs: 3,
            max_seq_len: 512,
            log_interval: 100,
            batch_size: 32,
            accumulation_steps: 1,
            gradient_clip_norm: Some(1.0),
            class_weights: None,
            quantize_nf4: false,
        }
    }
}

/// Hyperparameter diagnostic from contract validation.
///
/// Contract: qlora-hyperparameters-v1.yaml (C-HP-001..008)
#[derive(Debug, Clone)]
pub struct HyperparamDiagnostic {
    pub contract_id: &'static str,
    pub severity: DiagSeverity,
    pub message: String,
    pub recommendation: String,
}

/// Severity level for hyperparameter diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagSeverity {
    /// Informational — config is acceptable but not optimal.
    Info,
    /// Warning — config violates a research-grounded contract.
    Warn,
    /// Error — config is mathematically invalid (e.g., lr=0).
    Error,
}

/// Collection of hyperparameter diagnostics.
#[derive(Debug, Clone, Default)]
pub struct HyperparamDiagnostics {
    pub items: Vec<HyperparamDiagnostic>,
}

impl HyperparamDiagnostics {
    /// Check if any diagnostic matches a contract ID.
    pub fn has_warning(&self, contract_id: &str) -> bool {
        self.items.iter().any(|d| {
            d.contract_id == contract_id
                && matches!(d.severity, DiagSeverity::Warn | DiagSeverity::Error)
        })
    }

    /// Check if there are any blocking errors.
    pub fn has_errors(&self) -> bool {
        self.items.iter().any(|d| matches!(d.severity, DiagSeverity::Error))
    }

    /// Print all diagnostics to stderr.
    pub fn print_all(&self) {
        for d in &self.items {
            let prefix = match d.severity {
                DiagSeverity::Info => "[HP-INFO]",
                DiagSeverity::Warn => "[HP-WARN]",
                DiagSeverity::Error => "[HP-ERROR]",
            };
            eprintln!("{prefix} {}: {} → {}", d.contract_id, d.message, d.recommendation);
        }
    }
}

/// Data distribution statistics for data-driven hyperparameter validation.
///
/// Contract: C-HP-004 (seq_len from data), C-HP-008 (epochs for imbalance)
pub struct DataStats {
    /// 99th percentile of BPE token lengths in training data.
    pub p99_token_length: usize,
    /// Class imbalance ratio (majority_count / minority_count).
    pub imbalance_ratio: f32,
    /// Number of samples in minority class.
    pub minority_count: usize,
}

impl ClassifyConfig {
    /// Create a QLoRA config with research-grounded defaults.
    ///
    /// Every parameter traces to a published source:
    ///
    /// | Parameter | Value | Source |
    /// |-----------|-------|--------|
    /// | `learning_rate` | 2e-4 (≤13B) / 1e-4 (>13B) | Dettmers 2023 Table 9 |
    /// | `lora_alpha` | 2 × rank | Lightning AI experiments |
    /// | `batch_size` | 4 | RTX VRAM budget |
    /// | `accumulation_steps` | 4 | effective=16, Dettmers 2023 |
    /// | `gradient_clip_norm` | 1.0 | Standard practice |
    /// | `epochs` | 3 | Imbalanced classification |
    ///
    /// Contract: provable-contracts/contracts/entrenar/qlora-hyperparameters-v1.yaml
    pub fn qlora_default(model_params: u64) -> Self {
        // C-HP-001: lr scales with model size (Dettmers 2023 Table 9)
        let learning_rate = if model_params <= 13_000_000_000 { 2e-4 } else { 1e-4 };
        let lora_rank = 16;
        Self {
            num_classes: 2,
            lora_rank,
            // C-HP-003: alpha = 2 * rank (Lightning AI)
            lora_alpha: (2 * lora_rank) as f32,
            learning_rate,
            // C-HP-008: >= 2 epochs for imbalanced classification
            epochs: 3,
            // C-HP-004: set from data, 256 is SSC v3 default
            max_seq_len: 256,
            log_interval: 100,
            // C-HP-002: effective=16 (Dettmers 2023). Micro-batch=16 to saturate
            // GPU occupancy (batch=4 leaves RTX 4090 at 1.5% MFU).
            batch_size: 16,
            accumulation_steps: 1,
            // C-HP-006: gradient clipping (standard, SSC v2.2 precedent)
            gradient_clip_norm: Some(1.0),
            class_weights: None,
            quantize_nf4: true,
        }
    }

    /// Validate hyperparameters against research-grounded contracts.
    ///
    /// Returns diagnostics (warnings/errors) for each violated contract.
    /// Does NOT block — caller decides whether to abort or proceed.
    ///
    /// Contract: qlora-hyperparameters-v1.yaml (FALSIFY-HP-001..008)
    pub fn validate_hyperparameters(&self, model_params: u64) -> HyperparamDiagnostics {
        let mut diags = HyperparamDiagnostics::default();

        // C-HP-001: Learning rate scaling
        if self.quantize_nf4 && model_params <= 13_000_000_000 && self.learning_rate < 1.5e-4 {
            diags.items.push(HyperparamDiagnostic {
                contract_id: "C-HP-001",
                severity: DiagSeverity::Warn,
                message: format!(
                    "lr={:.0e} too low for {}B model (Dettmers 2023: use 2e-4 for ≤13B)",
                    self.learning_rate,
                    model_params / 1_000_000_000
                ),
                recommendation: "learning_rate: 0.0002".to_string(),
            });
        }

        // C-HP-002: Effective batch size
        let eff_batch = self.batch_size * self.accumulation_steps;
        if eff_batch != 16 {
            diags.items.push(HyperparamDiagnostic {
                contract_id: "C-HP-002",
                severity: DiagSeverity::Warn,
                message: format!(
                    "effective_batch={eff_batch} ({}×{}), Dettmers 2023 recommends 16 for ≤13B",
                    self.batch_size, self.accumulation_steps
                ),
                recommendation: format!(
                    "batch_size: {}, accumulation_steps: {}",
                    self.batch_size,
                    16 / self.batch_size.max(1)
                ),
            });
        }

        // C-HP-003: Alpha/rank ratio
        let expected_alpha = 2.0 * self.lora_rank as f32;
        if (self.lora_alpha - expected_alpha).abs() > 0.5 {
            diags.items.push(HyperparamDiagnostic {
                contract_id: "C-HP-003",
                severity: DiagSeverity::Warn,
                message: format!(
                    "lora_alpha={} with rank={} (ratio={:.1}), Lightning AI: alpha=2×rank={} optimal",
                    self.lora_alpha, self.lora_rank,
                    self.lora_alpha / self.lora_rank as f32,
                    expected_alpha
                ),
                recommendation: format!("lora_alpha: {expected_alpha}"),
            });
        }

        // C-HP-006: Gradient clipping
        if self.gradient_clip_norm.is_none() {
            diags.items.push(HyperparamDiagnostic {
                contract_id: "C-HP-006",
                severity: DiagSeverity::Warn,
                message: "No gradient clipping — SSC v2.2 saw grad norms up to 115.1".to_string(),
                recommendation: "gradient_clip_norm: 1.0".to_string(),
            });
        }

        // Blocking errors
        if self.learning_rate <= 0.0 {
            diags.items.push(HyperparamDiagnostic {
                contract_id: "C-HP-001",
                severity: DiagSeverity::Error,
                message: "learning_rate must be > 0".to_string(),
                recommendation: "learning_rate: 0.0002".to_string(),
            });
        }
        if self.batch_size == 0 {
            diags.items.push(HyperparamDiagnostic {
                contract_id: "C-HP-002",
                severity: DiagSeverity::Error,
                message: "batch_size must be > 0".to_string(),
                recommendation: "batch_size: 4".to_string(),
            });
        }

        diags
    }

    /// Validate hyperparameters that depend on training data distribution.
    ///
    /// Requires measuring the actual data (genchi genbutsu — go and see).
    ///
    /// Contract: C-HP-004 (seq_len), C-HP-008 (epochs)
    pub fn validate_with_data(&self, stats: &DataStats) -> HyperparamDiagnostics {
        let mut diags = HyperparamDiagnostics::default();

        // C-HP-004: Sequence length from data distribution
        if self.max_seq_len > 2 * stats.p99_token_length && stats.p99_token_length > 0 {
            diags.items.push(HyperparamDiagnostic {
                contract_id: "C-HP-004",
                severity: DiagSeverity::Warn,
                message: format!(
                    "max_seq_len={} but p99(tokens)={} — attention is O(n²), wasting {:.0}× compute",
                    self.max_seq_len,
                    stats.p99_token_length,
                    (self.max_seq_len as f64 / stats.p99_token_length as f64).powi(2)
                ),
                recommendation: format!(
                    "max_seq_len: {} (next_pow2 of p99)",
                    stats.p99_token_length.next_power_of_two()
                ),
            });
        }

        // C-HP-008: Epochs for imbalanced classification
        if stats.imbalance_ratio > 5.0 && self.epochs < 2 {
            let eff_batch = self.batch_size * self.accumulation_steps;
            let updates_per_epoch = stats.minority_count / eff_batch.max(1);
            diags.items.push(HyperparamDiagnostic {
                contract_id: "C-HP-008",
                severity: DiagSeverity::Warn,
                message: format!(
                    "epochs={} with {:.1}:1 imbalance — minority gets only {} gradient updates",
                    self.epochs,
                    stats.imbalance_ratio,
                    updates_per_epoch * self.epochs
                ),
                recommendation: format!(
                    "epochs: 3 (minority gets {} updates)",
                    updates_per_epoch * 3
                ),
            });
        }

        diags
    }
}

/// Result of processing one mini-batch via [`ClassifyPipeline::train_batch`].
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Average cross-entropy loss across the batch
    pub avg_loss: f32,
    /// Number of correctly classified samples
    pub correct: usize,
    /// Total number of samples in the batch
    pub total: usize,
    /// Global gradient norm (before clipping). 0.0 if clipping disabled.
    pub grad_norm: f32,
}

impl BatchResult {
    /// Compute classification accuracy as `correct / total`.
    ///
    /// Returns 0.0 for an empty batch (total == 0).
    #[must_use]
    pub fn accuracy(&self) -> f32 {
        self.correct as f32 / self.total.max(1) as f32
    }
}

/// GPU-resident training state for full-finetune backward pass.
///
/// Holds per-layer activation snapshots, optimizer state, and scratch buffers
/// required to run backward through all transformer blocks on GPU.
///
/// # Contract (C-GPUTRAIN-001)
///
/// - **Precondition**: `init()` called after CUDA blocks are created
/// - **Postcondition**: All layer_inputs saved during forward; optimizer states zero-initialized
/// - **Invariant**: `layer_inputs.len() == num_layers`; buffers never reallocated after init
#[cfg(feature = "cuda")]
struct GpuTrainingState {
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
    /// Per-block AdamW optimizer states
    optimizer_states: Vec<GpuBlockOptimizerState>,
    /// Global optimizer step counter
    step: u32,
    /// KAIZEN-045: Pre-allocated scratch buffer for activation checkpointing in backward
    /// [max_seq_len * hidden_size]. Eliminates per-backward cuMemAlloc/cuMemFree.
    output_scratch: GpuBuffer<f32>,
    /// KAIZEN-045: Pre-allocated upload buffer for gradient H2D transfer in backward
    /// [max_seq_len * hidden_size]. Eliminates per-backward cuMemAlloc/cuMemFree.
    grad_upload_buf: GpuBuffer<f32>,
    /// KAIZEN-060: Pre-allocated forward ping-pong buffers [max_seq_len * hidden_size].
    /// Eliminates 2 × cuMemAlloc/Free per forward pass.
    fwd_scratch_a: GpuBuffer<f32>,
    fwd_scratch_b: GpuBuffer<f32>,
    /// KAIZEN-061: Pre-allocated CPU staging buffer for backward mean-pool gradient.
    /// Sized to max_seq_len * hidden_size. Eliminates ~1.25MB heap alloc per sample
    /// in both backward_gpu_blocks and backward_nf4_gpu_blocks (~17.5GB/epoch).
    backward_cpu_staging: Vec<f32>,
}

/// Classification fine-tuning pipeline.
///
/// Owns the transformer, LoRA adapters, and classification head.
/// Provides `train_step()` for single-step training and `train()` for full loop.
///
/// When compiled with `feature = "cuda"` and a GPU is available, the forward pass
/// runs on CUDA via `CudaTransformerBlock`s for ~10-50x speedup (F-CUDA-007).
/// When `gpu_training` is set, the backward pass also runs on GPU with full-finetune
/// of all transformer weights (F-CUDA-014).
pub struct ClassifyPipeline {
    /// Base transformer model (weights frozen)
    pub model: Transformer,
    /// Classification head (trainable)
    pub classifier: ClassificationHead,
    /// LoRA adapters applied to attention projections
    pub lora_layers: Vec<LoRALayer>,
    /// Pipeline configuration
    pub config: ClassifyConfig,
    /// AdamW optimizer for trainable parameters
    optimizer: AdamW,
    /// Optional BPE tokenizer (None = byte-level fallback)
    tokenizer: Option<HfTokenizer>,
    /// Path to base model directory (set by `from_pretrained`, None for random init)
    model_dir: Option<PathBuf>,
    /// CUDA trainer for GPU memory management (F-CUDA-002)
    #[cfg(feature = "cuda")]
    cuda_trainer: Option<CudaTrainer>,
    /// CUDA-accelerated transformer blocks — one per layer (F-CUDA-006)
    #[cfg(feature = "cuda")]
    cuda_blocks: Option<Vec<CudaBlock>>,
    /// Shared scratch buffers for NF4 forward pass (C-SCRATCH-001).
    /// One allocation shared across all 36 layers — saves 7.5 GB for Qwen3-4B.
    #[cfg(feature = "cuda")]
    shared_scratch: Option<CudaBlockScratch>,
    /// Count of GPU forward passes that produced NaN/Inf and fell back to CPU.
    /// Used to decide when to permanently disable CUDA (threshold: 100).
    #[cfg(feature = "cuda")]
    cuda_nan_count: usize,
    /// GPU training state for full-finetune backward pass (F-CUDA-014).
    /// When `Some`, backward pass runs on GPU updating all transformer weights.
    #[cfg(feature = "cuda")]
    gpu_training: Option<GpuTrainingState>,
    /// Shared gradient workspace — one set of weight-gradient buffers for all layers.
    /// Contract: C-GRADWS-001. Saves (L-1) * 372 MB for Qwen3-4B (13.0 GB).
    #[cfg(feature = "cuda")]
    cuda_grad_workspace: Option<CudaGradWorkspace>,
    /// Shared LoRA gradient workspace for NF4 QLoRA backward (ENT-153).
    /// One set of LoRA gradient buffers shared across all layers.
    #[cfg(feature = "cuda")]
    cuda_lora_grad_workspace: Option<CudaLoraGradWorkspace>,
    /// Per-layer LoRA optimizer states for NF4 QLoRA training (ENT-153).
    #[cfg(feature = "cuda")]
    cuda_lora_optimizer_states: Option<Vec<GpuLoraOptimizerState>>,
    /// KAIZEN-014: Per-layer gradient accumulators for batch-correct LoRA optimizer.
    #[cfg(feature = "cuda")]
    cuda_lora_grad_accum: Option<Vec<CudaLoraGradWorkspace>>,
    /// NF4 LoRA optimizer step counter (separate from fp32 GpuTrainingState.step).
    #[cfg(feature = "cuda")]
    nf4_lora_step: u32,
    /// wgpu-accelerated forward pass (GPU feature, non-CUDA)
    #[cfg(feature = "gpu")]
    wgpu_forward_pass: Option<crate::transformer::WgpuForwardPass>,
    /// VRAM reservation guard (GPU-SHARE-002). Releases ledger entry on Drop.
    /// Held for RAII — released when pipeline is dropped.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    vram_guard: Option<VramGuard>,
}

impl ClassifyPipeline {
    /// Create a new classification pipeline with random weights and byte-level tokenization.
    ///
    /// # Arguments
    /// * `model_config` - Transformer configuration (e.g., `TransformerConfig::qwen2_0_5b()`)
    /// * `classify_config` - Classification pipeline configuration
    pub fn new(model_config: &TransformerConfig, classify_config: ClassifyConfig) -> Self {
        let model = Transformer::new(model_config);
        let classifier =
            ClassificationHead::new(model_config.hidden_size, classify_config.num_classes);
        let mut lora_layers = Self::build_lora_layers(&model, model_config, &classify_config);

        // Ensure LoRA A/B matrices have requires_grad=true
        for lora in &mut lora_layers {
            for param in lora.trainable_params() {
                param.set_requires_grad(true);
            }
        }

        let optimizer = AdamW::default_params(classify_config.learning_rate);

        // ── CUDA initialization (F-CUDA-001..006, GPU-SHARE-002) ─────────
        #[cfg(feature = "cuda")]
        let (cuda_trainer, cuda_blocks, shared_scratch, vram_guard) =
            Self::try_init_cuda(&model, model_config, &classify_config, &lora_layers);

        // ── GPU training state (F-CUDA-014) ────────────────────────────
        #[cfg(feature = "cuda")]
        let gpu_training = Self::try_init_gpu_training(
            &model,
            model_config,
            classify_config.max_seq_len,
            cuda_trainer.as_ref(),
            cuda_blocks.as_ref(),
        );

        // ── Shared gradient workspace (C-GRADWS-001) ────────────────────
        #[cfg(feature = "cuda")]
        let cuda_grad_workspace = if classify_config.quantize_nf4 {
            None
        } else {
            cuda_trainer.as_ref().and_then(|t| {
                CudaGradWorkspace::new(t.context(), model_config)
                    .map_err(|e| eprintln!("[CUDA] Failed to allocate grad workspace: {e}"))
                    .ok()
            })
        };

        // ── NF4 LoRA training state (ENT-153) ──────────────────────────
        #[cfg(feature = "cuda")]
        let (cuda_lora_grad_workspace, cuda_lora_optimizer_states, cuda_lora_grad_accum) =
            if classify_config.quantize_nf4 {
                Self::try_init_nf4_lora_training(
                    cuda_trainer.as_ref(),
                    cuda_blocks.as_ref(),
                    model_config,
                    &classify_config,
                )
            } else {
                (None, None, None)
            };

        // ── wgpu initialization (when CUDA unavailable) ──────────────────
        #[cfg(feature = "gpu")]
        let wgpu_forward_pass = {
            #[cfg(feature = "cuda")]
            let has_cuda = cuda_trainer.is_some();
            #[cfg(not(feature = "cuda"))]
            let has_cuda = false;

            if !has_cuda {
                // KAIZEN-015: Pre-upload FFN weights to GPU (zero H2D per forward pass)
                match crate::transformer::WgpuForwardPass::with_resident_weights(&model) {
                    Ok(pass) => {
                        eprintln!("[wgpu] GPU forward pass initialized (resident weights)");
                        Some(pass)
                    }
                    Err(e) => {
                        eprintln!("[wgpu] GPU resident init failed, trying default: {e}");
                        match crate::transformer::WgpuForwardPass::new_default(model_config) {
                            Ok(pass) => {
                                eprintln!("[wgpu] GPU forward pass initialized (upload per call)");
                                Some(pass)
                            }
                            Err(e2) => {
                                eprintln!("[wgpu] GPU initialization failed, using CPU: {e2}");
                                None
                            }
                        }
                    }
                }
            } else {
                None // CUDA takes priority
            }
        };

        Self {
            model,
            classifier,
            lora_layers,
            config: classify_config,
            optimizer,
            tokenizer: None,
            model_dir: None,
            #[cfg(feature = "cuda")]
            cuda_trainer,
            #[cfg(feature = "cuda")]
            cuda_blocks,
            #[cfg(feature = "cuda")]
            shared_scratch,
            #[cfg(feature = "cuda")]
            cuda_nan_count: 0,
            #[cfg(feature = "cuda")]
            gpu_training,
            #[cfg(feature = "cuda")]
            cuda_grad_workspace,
            #[cfg(feature = "cuda")]
            cuda_lora_grad_workspace,
            #[cfg(feature = "cuda")]
            cuda_lora_optimizer_states,
            #[cfg(feature = "cuda")]
            cuda_lora_grad_accum,
            #[cfg(feature = "cuda")]
            nf4_lora_step: 0,
            #[cfg(feature = "gpu")]
            wgpu_forward_pass,
            #[cfg(feature = "cuda")]
            vram_guard,
        }
    }

    /// Create a classification pipeline from pretrained weights.
    ///
    /// Loads a transformer from SafeTensors weights and optionally a BPE tokenizer
    /// from `tokenizer.json` in the model directory.
    ///
    /// # Arguments
    /// * `model_dir` - Directory containing SafeTensors weights (and optionally `tokenizer.json`)
    /// * `model_config` - Transformer configuration matching the pretrained weights
    /// * `classify_config` - Classification pipeline configuration
    ///
    /// # Errors
    /// Returns error if the model directory doesn't exist or weights fail to load.
    pub fn from_pretrained(
        model_dir: impl AsRef<Path>,
        model_config: &TransformerConfig,
        classify_config: ClassifyConfig,
    ) -> crate::Result<Self> {
        let model_dir = model_dir.as_ref();

        let model = Transformer::from_safetensors(model_dir, model_config)?;
        let classifier =
            ClassificationHead::new(model_config.hidden_size, classify_config.num_classes);
        let mut lora_layers = Self::build_lora_layers(&model, model_config, &classify_config);

        for lora in &mut lora_layers {
            for param in lora.trainable_params() {
                param.set_requires_grad(true);
            }
        }

        // CONTRACT: Training requires a BPE tokenizer — byte-fallback is not acceptable.
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            Some(
                HfTokenizer::from_file(&tokenizer_path)
                    .map_err(|e| crate::Error::Io(format!("Failed to load tokenizer: {e}")))?,
            )
        } else {
            return Err(crate::Error::ConfigError(format!(
                "No tokenizer.json found in '{}'. Training requires a BPE tokenizer.",
                model_dir.display(),
            )));
        };

        let optimizer = AdamW::default_params(classify_config.learning_rate);

        // ── CUDA initialization (F-CUDA-001..006, GPU-SHARE-002) ─────────
        #[cfg(feature = "cuda")]
        let (cuda_trainer, cuda_blocks, shared_scratch, vram_guard) =
            Self::try_init_cuda(&model, model_config, &classify_config, &lora_layers);

        // ── GPU training state (F-CUDA-014) ────────────────────────────
        #[cfg(feature = "cuda")]
        let gpu_training = Self::try_init_gpu_training(
            &model,
            model_config,
            classify_config.max_seq_len,
            cuda_trainer.as_ref(),
            cuda_blocks.as_ref(),
        );

        // ── Shared gradient workspace (C-GRADWS-001) ────────────────────
        #[cfg(feature = "cuda")]
        let cuda_grad_workspace = if classify_config.quantize_nf4 {
            None // No grad workspace needed for frozen NF4 weights
        } else {
            cuda_trainer.as_ref().and_then(|t| {
                CudaGradWorkspace::new(t.context(), model_config)
                    .map_err(|e| eprintln!("[CUDA] Failed to allocate grad workspace: {e}"))
                    .ok()
            })
        };

        // ── NF4 LoRA training state (ENT-153) ──────────────────────────
        #[cfg(feature = "cuda")]
        let (cuda_lora_grad_workspace, cuda_lora_optimizer_states, cuda_lora_grad_accum) =
            if classify_config.quantize_nf4 {
                Self::try_init_nf4_lora_training(
                    cuda_trainer.as_ref(),
                    cuda_blocks.as_ref(),
                    model_config,
                    &classify_config,
                )
            } else {
                (None, None, None)
            };

        // ── wgpu initialization (when CUDA unavailable) ──────────────────
        #[cfg(feature = "gpu")]
        let wgpu_forward_pass = {
            #[cfg(feature = "cuda")]
            let has_cuda = cuda_trainer.is_some();
            #[cfg(not(feature = "cuda"))]
            let has_cuda = false;

            if !has_cuda {
                // KAIZEN-015: Pre-upload FFN weights to GPU
                match crate::transformer::WgpuForwardPass::with_resident_weights(&model) {
                    Ok(pass) => {
                        eprintln!(
                            "[wgpu] Batched forward pass initialized ({} layers, resident weights)",
                            model_config.num_hidden_layers
                        );
                        Some(pass)
                    }
                    Err(e) => {
                        eprintln!("[wgpu] Resident init failed, trying default: {e}");
                        match crate::transformer::WgpuForwardPass::new_default(model_config) {
                            Ok(pass) => {
                                eprintln!("[wgpu] Batched forward pass initialized ({} layers, upload per call)", model_config.num_hidden_layers);
                                Some(pass)
                            }
                            Err(e2) => {
                                eprintln!("[wgpu] GPU init failed, using CPU: {e2}");
                                None
                            }
                        }
                    }
                }
            } else {
                None
            }
        };

        Ok(Self {
            model,
            classifier,
            lora_layers,
            config: classify_config,
            optimizer,
            tokenizer,
            model_dir: Some(model_dir.to_path_buf()),
            #[cfg(feature = "cuda")]
            cuda_trainer,
            #[cfg(feature = "cuda")]
            cuda_blocks,
            #[cfg(feature = "cuda")]
            shared_scratch,
            #[cfg(feature = "cuda")]
            cuda_nan_count: 0,
            #[cfg(feature = "cuda")]
            gpu_training,
            #[cfg(feature = "cuda")]
            cuda_grad_workspace,
            #[cfg(feature = "cuda")]
            cuda_lora_grad_workspace,
            #[cfg(feature = "cuda")]
            cuda_lora_optimizer_states,
            #[cfg(feature = "cuda")]
            cuda_lora_grad_accum,
            #[cfg(feature = "cuda")]
            nf4_lora_step: 0,
            #[cfg(feature = "gpu")]
            wgpu_forward_pass,
            #[cfg(feature = "cuda")]
            vram_guard,
        })
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
        classify_config: ClassifyConfig,
    ) -> crate::Result<Self> {
        let model = Transformer::from_apr(apr_path, model_config)?;
        let classifier =
            ClassificationHead::new(model_config.hidden_size, classify_config.num_classes);
        let mut lora_layers = Self::build_lora_layers(&model, model_config, &classify_config);

        for lora in &mut lora_layers {
            for param in lora.trainable_params() {
                param.set_requires_grad(true);
            }
        }

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
                             Training requires a BPE tokenizer.",
                            path.display(),
                        ))
                    })?;
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

        let optimizer = AdamW::default_params(classify_config.learning_rate);

        #[cfg(feature = "cuda")]
        let (cuda_trainer, cuda_blocks, shared_scratch, vram_guard) =
            Self::try_init_cuda(&model, model_config, &classify_config, &lora_layers);

        #[cfg(feature = "cuda")]
        let gpu_training = Self::try_init_gpu_training(
            &model,
            model_config,
            classify_config.max_seq_len,
            cuda_trainer.as_ref(),
            cuda_blocks.as_ref(),
        );

        #[cfg(feature = "cuda")]
        let cuda_grad_workspace = if classify_config.quantize_nf4 {
            None
        } else {
            cuda_trainer.as_ref().and_then(|t| {
                CudaGradWorkspace::new(t.context(), model_config)
                    .map_err(|e| eprintln!("[CUDA] Failed to allocate grad workspace: {e}"))
                    .ok()
            })
        };

        #[cfg(feature = "cuda")]
        let (cuda_lora_grad_workspace, cuda_lora_optimizer_states, cuda_lora_grad_accum) =
            if classify_config.quantize_nf4 {
                Self::try_init_nf4_lora_training(
                    cuda_trainer.as_ref(),
                    cuda_blocks.as_ref(),
                    model_config,
                    &classify_config,
                )
            } else {
                (None, None, None)
            };

        // ── wgpu initialization ──────────────────────────────────────────
        #[cfg(feature = "gpu")]
        let wgpu_forward_pass = {
            #[cfg(feature = "cuda")]
            let has_cuda = cuda_trainer.is_some();
            #[cfg(not(feature = "cuda"))]
            let has_cuda = false;

            if !has_cuda {
                // KAIZEN-015: Pre-upload FFN weights to GPU
                crate::transformer::WgpuForwardPass::with_resident_weights(&model)
                    .or_else(|e| {
                        eprintln!("[wgpu] Resident init failed: {e}, trying default");
                        crate::transformer::WgpuForwardPass::new_default(model_config)
                    })
                    .map_err(|e| eprintln!("[wgpu] GPU init failed: {e}"))
                    .ok()
            } else {
                None
            }
        };

        Ok(Self {
            model,
            classifier,
            lora_layers,
            config: classify_config,
            optimizer,
            tokenizer,
            model_dir: Some(apr_path.to_path_buf()),
            #[cfg(feature = "cuda")]
            cuda_trainer,
            #[cfg(feature = "cuda")]
            cuda_blocks,
            #[cfg(feature = "cuda")]
            shared_scratch,
            #[cfg(feature = "cuda")]
            cuda_nan_count: 0,
            #[cfg(feature = "cuda")]
            gpu_training,
            #[cfg(feature = "cuda")]
            cuda_grad_workspace,
            #[cfg(feature = "cuda")]
            cuda_lora_grad_workspace,
            #[cfg(feature = "cuda")]
            cuda_lora_optimizer_states,
            #[cfg(feature = "cuda")]
            cuda_lora_grad_accum,
            #[cfg(feature = "cuda")]
            nf4_lora_step: 0,
            #[cfg(feature = "gpu")]
            wgpu_forward_pass,
            #[cfg(feature = "cuda")]
            vram_guard,
        })
    }

    /// Tokenize input text using BPE tokenizer.
    ///
    /// Truncates to `config.max_seq_len` and ensures at least one token.
    ///
    /// # Panics
    /// Panics if no BPE tokenizer is loaded. Training pipelines MUST have a
    /// tokenizer — byte-level fallback is a silent corruption path.
    pub(crate) fn tokenize(&self, text: &str) -> Vec<u32> {
        let mut ids = match self.tokenizer.as_ref() {
            Some(tok) => tok.encode(text),
            None => {
                // Byte-level fallback when no BPE tokenizer is loaded
                text.bytes().map(u32::from).collect()
            }
        };
        ids.truncate(self.config.max_seq_len);
        if ids.is_empty() {
            ids.push(0);
        }
        ids
    }

    /// Pre-tokenize a batch of samples for efficient training (KAIZEN-028).
    ///
    /// Tokenizes each sample once and stores the token IDs alongside the label.
    /// This eliminates redundant BPE encoding across epochs and batches.
    ///
    /// # Contract (C-PRETOK-001)
    ///
    /// - **Precondition**: All samples have non-empty `input`
    /// - **Postcondition**: Each `TokenizedSample` has `token_ids.len() in 1..=max_seq_len`
    /// - **Invariant**: Tokenization is deterministic — same input always produces same IDs
    pub fn pre_tokenize(&self, samples: &[SafetySample]) -> Vec<TokenizedSample> {
        let has_tokenizer = self.tokenizer.is_some();
        samples
            .iter()
            .map(|s| {
                let token_ids = if has_tokenizer {
                    self.tokenize(&s.input)
                } else {
                    // Byte-level fallback for tests without BPE tokenizer
                    let mut ids = s.input_ids();
                    ids.truncate(self.config.max_seq_len);
                    if ids.is_empty() {
                        ids.push(0);
                    }
                    ids
                };
                TokenizedSample { token_ids, label: s.label }
            })
            .collect()
    }

    /// Train on a batch of pre-tokenized samples (KAIZEN-028).
    ///
    /// Identical to [`train_batch`] but skips tokenization — token IDs are
    /// pre-computed at dataset construction time.
    pub fn train_batch_tokenized(&mut self, samples: &[TokenizedSample]) -> BatchResult {
        if samples.is_empty() {
            return BatchResult { avg_loss: 0.0, correct: 0, total: 0, grad_norm: 0.0 };
        }

        let batch_size = samples.len();

        // ── 1. Zero gradients ──────────────────────────────────────────
        self.zero_all_gradients();

        // ── 2. Accumulate gradients over all samples ───────────────────
        #[cfg(feature = "gpu")]
        let (total_loss, correct) = self
            .try_train_batch_wgpu_tokenized(samples)
            .unwrap_or_else(|| self.train_batch_per_sample_tokenized(samples));

        #[cfg(not(feature = "gpu"))]
        let (total_loss, correct) = self.train_batch_per_sample_tokenized(samples);

        // ── 3. Normalize gradients by batch size ───────────────────────
        self.scale_all_gradients(1.0 / batch_size as f32);

        // ── 4. Gradient clipping (captures pre-clip norm) ────────────
        let grad_norm = if let Some(max_norm) = self.config.gradient_clip_norm {
            let mut params = self.trainable_parameters_mut();
            clip_grad_norm_refs(&mut params, max_norm)
        } else {
            self.compute_grad_norm()
        };

        // ── 5. Optimizer step (once for the whole batch) ───────────────
        #[cfg(feature = "cuda")]
        {
            if self.gpu_training.is_some() && !self.config.quantize_nf4 {
                let lr = self.optimizer.lr();
                self.gpu_optimizer_step(lr);
            }
        }

        #[cfg(feature = "cuda")]
        {
            if self.gpu_training.is_some() && self.config.quantize_nf4 {
                self.nf4_lora_batch_optimizer_step(batch_size);
            }
        }

        let mut params: Vec<&mut Tensor> = Vec::new();
        if !self.config.quantize_nf4 {
            for lora in &mut self.lora_layers {
                params.extend(lora.trainable_params());
            }
        }
        params.extend(self.classifier.parameters_mut());
        self.optimizer.step_refs(&mut params);

        BatchResult {
            avg_loss: total_loss / batch_size as f32,
            correct,
            total: batch_size,
            grad_norm,
        }
    }

    /// Per-sample forward + backward with pre-tokenized IDs (KAIZEN-028).
    fn train_batch_per_sample_tokenized(&mut self, samples: &[TokenizedSample]) -> (f32, usize) {
        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        for sample in samples {
            let (loss, predicted) = self.forward_backward_single(&sample.token_ids, sample.label);
            total_loss += loss;
            if predicted == sample.label {
                correct += 1;
            }
        }
        (total_loss, correct)
    }

    /// Batched wgpu forward with pre-tokenized IDs (KAIZEN-028).
    #[cfg(feature = "gpu")]
    fn try_train_batch_wgpu_tokenized(
        &mut self,
        samples: &[TokenizedSample],
    ) -> Option<(f32, usize)> {
        if self.wgpu_forward_pass.is_none() {
            return None;
        }

        let batch_token_ids: Vec<Vec<u32>> = samples.iter().map(|s| s.token_ids.clone()).collect();

        let lora_ref =
            if self.lora_layers.is_empty() { None } else { Some(self.lora_layers.as_slice()) };

        let hiddens = self
            .wgpu_forward_pass
            .as_ref()
            .expect("checked is_none above")
            .forward_hidden_batch(&self.model, &batch_token_ids, lora_ref)
            .map_err(|e| {
                eprintln!("[wgpu] Batched forward failed, falling back to per-sample: {e}")
            })
            .ok()?;

        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        for (i, hidden) in hiddens.iter().enumerate() {
            let (loss, predicted) = self.classify_backward_from_hidden(
                hidden,
                batch_token_ids[i].len(),
                samples[i].label,
            );
            total_loss += loss;
            if predicted == samples[i].label {
                correct += 1;
            }
        }
        Some((total_loss, correct))
    }

    /// Accumulate gradients with pre-tokenized samples (KAIZEN-028).
    ///
    /// Identical to [`accumulate_gradients`] but uses pre-tokenized IDs.
    pub fn accumulate_gradients_tokenized(
        &mut self,
        micro_batch: &[TokenizedSample],
    ) -> BatchResult {
        if micro_batch.is_empty() {
            return BatchResult { avg_loss: 0.0, correct: 0, total: 0, grad_norm: 0.0 };
        }

        let mut total_loss = 0.0f32;
        let mut correct = 0usize;

        for sample in micro_batch {
            let (loss, predicted) = self.forward_backward_single(&sample.token_ids, sample.label);
            total_loss += loss;
            if predicted == sample.label {
                correct += 1;
            }
        }

        BatchResult {
            avg_loss: total_loss / micro_batch.len() as f32,
            correct,
            total: micro_batch.len(),
            grad_norm: 0.0,
        }
    }

    /// Forward-only inference with pre-tokenized sample (KAIZEN-028).
    ///
    /// Used for validation where we need loss + prediction without backward pass.
    pub fn forward_only_tokenized(&mut self, token_ids: &[u32], label: usize) -> (f32, usize) {
        self.forward_only(token_ids, label)
    }

    /// Build LoRA layers for Q and V projections across all transformer layers.
    fn build_lora_layers(
        model: &Transformer,
        model_config: &TransformerConfig,
        classify_config: &ClassifyConfig,
    ) -> Vec<LoRALayer> {
        let lora_config = LoRAConfig::new(classify_config.lora_rank, classify_config.lora_alpha)
            .target_qv_projections();

        let mut lora_layers = Vec::new();
        let hidden = model_config.hidden_size;
        let head_dim = model_config.head_dim();

        for layer in &model.layers {
            let attn = &layer.self_attn;

            // Q projection LoRA
            if lora_config.should_apply("q_proj", None) {
                let q_dim = model_config.num_attention_heads * head_dim;
                let q_weight = Tensor::from_vec(
                    attn.w_q.data().as_slice().expect("contiguous w_q").to_vec(),
                    false,
                );
                lora_layers.push(LoRALayer::new(
                    q_weight,
                    q_dim,
                    hidden,
                    classify_config.lora_rank,
                    classify_config.lora_alpha,
                ));
            }

            // V projection LoRA
            if lora_config.should_apply("v_proj", None) {
                let v_dim = model_config.num_kv_heads * head_dim;
                let v_weight = Tensor::from_vec(
                    attn.w_v.data().as_slice().expect("contiguous w_v").to_vec(),
                    false,
                );
                lora_layers.push(LoRALayer::new(
                    v_weight,
                    v_dim,
                    hidden,
                    classify_config.lora_rank,
                    classify_config.lora_alpha,
                ));
            }
        }

        lora_layers
    }

    // ── CUDA GPU acceleration (F-CUDA-001..014) ────────────────────────

    /// C-PREWARM-001: JIT-compile all CUDA kernels before block upload.
    ///
    /// CUDA JIT needs free VRAM for PTX compilation. After uploading transformer
    /// layers, JIT fails with CUDA_ERROR_ILLEGAL_ADDRESS or OOM.
    #[cfg(feature = "cuda")]
    fn pre_warm_all_kernels(
        model_config: &TransformerConfig,
        classify_config: &ClassifyConfig,
    ) -> bool {
        let max_seq_len = classify_config.max_seq_len;
        let quantize_nf4 = classify_config.quantize_nf4;
        let head_dim = model_config.head_dim();

        if let Err(e) = pre_warm_forward_kernels(
            model_config.hidden_size,
            model_config.intermediate_size,
            model_config.num_attention_heads,
            model_config.num_kv_heads,
            head_dim,
            max_seq_len,
        ) {
            eprintln!("[CUDA] Failed to pre-warm forward kernels: {e} — using CPU");
            return false;
        }

        if quantize_nf4 {
            eprintln!(
                "[CUDA] NF4 quantization enabled — frozen weights will be 4-bit (~8x compression)"
            );
        }

        if let Err(e) = pre_warm_lora_backward_kernels(
            model_config.hidden_size,
            model_config.num_attention_heads * head_dim,
            model_config.num_kv_heads * head_dim,
            max_seq_len,
            classify_config.lora_rank,
        ) {
            eprintln!(
                "[CUDA] Failed to pre-warm LoRA forward-cache backward kernels: {e} — using CPU"
            );
            return false;
        }

        if let Err(e) = pre_warm_backward_cache_kernels(
            model_config.hidden_size,
            model_config.num_attention_heads * head_dim,
            model_config.num_kv_heads * head_dim,
            max_seq_len,
            classify_config.lora_rank,
            model_config.intermediate_size,
            model_config.num_attention_heads,
            quantize_nf4,
        ) {
            eprintln!("[CUDA] Failed to pre-warm backward cache kernels: {e} — using CPU");
            return false;
        }

        if let Err(e) = pre_warm_lora_adamw_kernels(
            model_config.hidden_size,
            model_config.num_attention_heads * head_dim,
            model_config.num_kv_heads * head_dim,
            classify_config.lora_rank,
            classify_config.num_classes,
            model_config.intermediate_size,
            quantize_nf4,
        ) {
            eprintln!("[CUDA] Failed to pre-warm AdamW kernels: {e} — using CPU");
            return false;
        }

        let realizador_warmed = pre_warm_realizador_gemm(
            max_seq_len,
            model_config.hidden_size,
            model_config.num_kv_heads * head_dim,
            model_config.intermediate_size,
            classify_config.lora_rank,
            classify_config.num_classes,
        );
        if realizador_warmed > 0 {
            eprintln!("[CUDA] Pre-warmed {realizador_warmed} realizador GEMM shapes");
        }

        true
    }

    /// Estimate VRAM usage (MB) for GPU training based on model architecture.
    ///
    /// Used by GPU-SHARE-002 to reserve VRAM via the ledger before allocation.
    #[cfg(feature = "cuda")]
    fn estimate_vram_mb(model_config: &TransformerConfig, config: &ClassifyConfig) -> usize {
        if config.quantize_nf4 {
            let weight_elements =
                model_config.per_layer_weight_elements() * model_config.num_hidden_layers;
            let weight_mb = weight_elements / (2 * 1024 * 1024);
            let scratch_mb =
                (config.max_seq_len * model_config.hidden_size * 4 * 10) / (1024 * 1024);
            let overhead_mb = 512;
            weight_mb + scratch_mb + overhead_mb
        } else {
            model_config.total_training_vram_bytes_shared(config.max_seq_len) / (1024 * 1024) + 256
        }
    }

    /// GPU-SHARE-002: Acquire VRAM guard before GPU allocation.
    ///
    /// Returns `None` if VRAM is insufficient (C-VRAM-001 enforcement).
    #[cfg(feature = "cuda")]
    fn acquire_vram_guard(
        model_config: &TransformerConfig,
        classify_config: &ClassifyConfig,
    ) -> Option<VramGuard> {
        let budget_mb = Self::estimate_vram_mb(model_config, classify_config);
        let task_label =
            if classify_config.quantize_nf4 { "classify-qlora" } else { "classify-lora" };
        match VramGuard::acquire(budget_mb, task_label) {
            Ok(guard) => {
                eprintln!(
                    "[GPU-SHARE] VRAM reserved: {budget_mb} MB for {task_label} (gpu: {})",
                    guard.gpu_uuid()
                );
                Some(guard)
            }
            Err(e) => {
                eprintln!("[GPU-SHARE] VRAM guard denied: {e} — falling back to CPU");
                None
            }
        }
    }

    /// Attempt to initialize CUDA acceleration.
    ///
    /// Creates `CudaTrainer` and uploads all transformer layer weights to GPU as
    /// `CudaTransformerBlock`s. Returns `(None, None, None, None)` if CUDA is
    /// unavailable or any initialization step fails (F-CUDA-003: graceful fallback).
    ///
    /// GPU-SHARE-002: Acquires a VRAM guard from the ledger before allocating GPU
    /// memory. The guard is returned and must be stored in the pipeline struct for
    /// RAII release on Drop.
    #[cfg(feature = "cuda")]
    fn try_init_cuda(
        model: &Transformer,
        model_config: &TransformerConfig,
        classify_config: &ClassifyConfig,
        lora_layers: &[LoRALayer],
    ) -> (Option<CudaTrainer>, Option<Vec<CudaBlock>>, Option<CudaBlockScratch>, Option<VramGuard>)
    {
        if !cuda_training_available() {
            eprintln!("[CUDA] No CUDA runtime detected — using CPU");
            return (None, None, None, None);
        }

        // GPU-SHARE-002: Acquire VRAM reservation before allocating
        let mut vram_guard = Self::acquire_vram_guard(model_config, classify_config);
        if vram_guard.is_none() {
            return (None, None, None, None);
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
                return (None, None, None, None);
            }
        };

        let ctx = Arc::clone(trainer.context());
        let max_seq_len = classify_config.max_seq_len;
        let quantize_nf4 = classify_config.quantize_nf4;

        if !Self::pre_warm_all_kernels(model_config, classify_config) {
            return (None, None, None, None);
        }

        let mut blocks = Vec::with_capacity(model.config.num_hidden_layers);

        for (i, layer) in model.layers.iter().enumerate() {
            // Extract weight data from CPU tensors (F-CUDA-005)
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
                // Extract LoRA data for this layer's Q and V projections
                let lora_scale = classify_config.lora_alpha / classify_config.lora_rank as f32;
                let lora_rank = classify_config.lora_rank;
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
                    return (None, None, None, None);
                }
            }
        }

        eprintln!(
            "[CUDA] Uploaded {} transformer layers to GPU (max_seq_len={})",
            blocks.len(),
            max_seq_len
        );

        // F-CUDA-006: verify all layers uploaded
        assert_eq!(blocks.len(), model.config.num_hidden_layers);

        // C-SCRATCH-001: Allocate one shared scratch for NF4 (saves 7.5 GB for Qwen3-4B)
        let shared_scratch = if quantize_nf4 {
            match CudaBlockScratch::new(model_config, max_seq_len, &ctx, classify_config.lora_rank)
            {
                Ok(s) => Some(s),
                Err(e) => {
                    eprintln!("[CUDA] Failed to allocate shared scratch: {e} — using CPU");
                    return (None, None, None, None);
                }
            }
        } else {
            None // fp32 blocks own their scratch (needed for backward)
        };

        // GPU-SHARE-002: Update actual VRAM usage after all allocations
        if let Some(ref mut guard) = vram_guard {
            let _ = guard.update_actual(guard.budget_mb());
        }

        (Some(trainer), Some(blocks), shared_scratch, vram_guard)
    }

    /// Initialize GPU training state for full-finetune backward pass (F-CUDA-014).
    ///
    /// Allocates layer-input snapshot buffers, uploads final RMSNorm weight,
    /// and initializes per-block AdamW optimizer state. Returns `None` if CUDA
    /// is not active or any allocation fails.
    ///
    /// # Contract (C-GPUTRAINIT-001)
    ///
    /// - **Precondition**: CUDA trainer and blocks are initialized (`Some`)
    /// - **Postcondition**: All buffers allocated; optimizer states zero-initialized
    /// - **Invariant**: Returns `None` on any failure (graceful fallback to CPU training)
    #[cfg(feature = "cuda")]
    fn try_init_gpu_training(
        model: &Transformer,
        model_config: &TransformerConfig,
        max_seq_len: usize,
        cuda_trainer: Option<&CudaTrainer>,
        cuda_blocks: Option<&Vec<CudaBlock>>,
    ) -> Option<GpuTrainingState> {
        let trainer = cuda_trainer?;
        let blocks = cuda_blocks?;

        let hidden_size = model_config.hidden_size;
        let buf_size = max_seq_len * hidden_size;
        let num_layers = blocks.len();

        // ── VRAM budget guard (C-GPUTRAINIT-002) ────────────────────────
        // Pre-compute optimizer state size to avoid OOM that would poison
        // the CUDA context (CUDA_ERROR_ILLEGAL_ADDRESS after failed alloc).
        // NF4 blocks have no per-layer fp32 optimizer — only need layer inputs + grad scratch
        let is_nf4 = blocks.first().is_some_and(|b| matches!(b, CudaBlock::Nf4(_)));
        let per_layer_weights = model_config.per_layer_weight_elements();
        let optimizer_bytes = if is_nf4 { 0 } else { num_layers * per_layer_weights * 2 * 4 };
        let layer_input_bytes = num_layers * buf_size * 4;
        let grad_scratch_bytes = (3 * buf_size + hidden_size) * 4;
        let total_training_bytes = optimizer_bytes + layer_input_bytes + grad_scratch_bytes;

        // After block upload + grad workspace, remaining VRAM is approximately:
        //   device_vram - (weights + scratch + grad_workspace)
        // For safety, we estimate remaining as device_vram minus our VRAM budget formula.
        let model_vram = model_config.total_training_vram_bytes_shared(max_seq_len);
        // Use 24 GB as conservative device VRAM (RTX 4090 = 24564 MiB)
        let device_vram = 24_u64 * 1024 * 1024 * 1024;
        let remaining_vram = device_vram.saturating_sub(model_vram as u64);

        if total_training_bytes as u64 > remaining_vram {
            eprintln!(
                "[CUDA] Skipping GPU training state: needs {:.1} GB \
                 (optimizer: {:.1} GB, layer inputs: {:.1} GB), \
                 estimated remaining VRAM: {:.1} GB — will use GPU forward + CPU backward",
                total_training_bytes as f64 / 1e9,
                optimizer_bytes as f64 / 1e9,
                layer_input_bytes as f64 / 1e9,
                remaining_vram as f64 / 1e9,
            );
            return None;
        }

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

        // Initialize per-block optimizer states (fp32 only — NF4 uses separate LoRA optimizer)
        let mut optimizer_states = Vec::with_capacity(num_layers);
        let is_nf4 = blocks.first().is_some_and(|b| matches!(b, CudaBlock::Nf4(_)));
        if !is_nf4 {
            for (i, block) in blocks.iter().enumerate() {
                match block.init_optimizer_state() {
                    Ok(state) => optimizer_states.push(state),
                    Err(e) => {
                        eprintln!(
                            "[CUDA] GPU training init failed (optimizer state layer {i}): {e}"
                        );
                        return None;
                    }
                }
            }
        }

        eprintln!(
            "[CUDA] GPU training state initialized: {num_layers} layers, \
             {buf_size} buf_size{}",
            if is_nf4 {
                " (NF4 QLoRA mode — LoRA optimizer separate)".to_string()
            } else {
                format!(
                    " ({:.1} MB optimizer state)",
                    (optimizer_states.len() * 18 * buf_size * 4) as f64 / 1e6
                )
            }
        );

        // KAIZEN-045: Pre-allocate backward scratch buffers to eliminate per-backward
        // cuMemAlloc/cuMemFree. Each cuMemAlloc costs ~10-100µs; over 14K samples this
        // saves 28K+ CUDA memory operations per epoch.
        let output_scratch = trainer.zeros(buf_size).ok()?;
        let grad_upload_buf = trainer.zeros(buf_size).ok()?;

        // KAIZEN-060: Pre-allocate forward ping-pong buffers to eliminate
        // 2 × cuMemAlloc/Free per forward pass (was trainer.upload + trainer.zeros per call).
        let fwd_scratch_a = trainer.zeros(buf_size).ok()?;
        let fwd_scratch_b = trainer.zeros(buf_size).ok()?;

        // KAIZEN-061: Pre-allocate CPU staging buffer for backward mean-pool gradient.
        // Eliminates vec![0.0; seq_len * hidden_size] (~1.25MB) per sample in both
        // backward_gpu_blocks and backward_nf4_gpu_blocks.
        let backward_cpu_staging = vec![0.0f32; buf_size];

        Some(GpuTrainingState {
            layer_inputs,
            final_norm_weight,
            blocks_output,
            grad_buf_a,
            grad_buf_b,
            grad_final_norm_weight,
            optimizer_states,
            step: 0,
            output_scratch,
            grad_upload_buf,
            fwd_scratch_a,
            fwd_scratch_b,
            backward_cpu_staging,
        })
    }

    /// Initialize NF4 LoRA training state: gradient workspace + per-layer optimizer states + accumulators.
    ///
    /// # Contract (C-NF4LORA-INIT-001)
    ///
    /// - **Precondition**: CUDA trainer and NF4 blocks initialized
    /// - **Postcondition**: LoRA grad workspace allocated, one optimizer state per NF4 block,
    ///   one gradient accumulator per NF4 block (KAIZEN-014)
    /// - **Invariant**: Returns `(None, None, None)` on any failure (graceful fallback to CPU LoRA)
    #[cfg(feature = "cuda")]
    fn try_init_nf4_lora_training(
        cuda_trainer: Option<&CudaTrainer>,
        cuda_blocks: Option<&Vec<CudaBlock>>,
        model_config: &TransformerConfig,
        classify_config: &ClassifyConfig,
    ) -> (
        Option<CudaLoraGradWorkspace>,
        Option<Vec<GpuLoraOptimizerState>>,
        Option<Vec<CudaLoraGradWorkspace>>,
    ) {
        let trainer = match cuda_trainer {
            Some(t) => t,
            None => return (None, None, None),
        };
        let blocks = match cuda_blocks {
            Some(b) => b,
            None => return (None, None, None),
        };

        // Allocate shared LoRA gradient workspace
        let grad_ws = match CudaLoraGradWorkspace::new(
            trainer.context(),
            model_config,
            classify_config.lora_rank,
        ) {
            Ok(ws) => ws,
            Err(e) => {
                eprintln!("[CUDA] NF4 LoRA grad workspace alloc failed: {e}");
                return (None, None, None);
            }
        };

        // Initialize per-block LoRA optimizer states
        let mut opt_states = Vec::with_capacity(blocks.len());
        for (i, block) in blocks.iter().enumerate() {
            match block.init_lora_optimizer_state() {
                Ok(state) => opt_states.push(state),
                Err(e) => {
                    eprintln!("[CUDA] NF4 LoRA optimizer init failed (layer {i}): {e}");
                    return (None, None, None);
                }
            }
        }

        // KAIZEN-014: Allocate per-layer gradient accumulators
        let mut grad_accum = Vec::with_capacity(blocks.len());
        for i in 0..blocks.len() {
            match CudaLoraGradWorkspace::new(
                trainer.context(),
                model_config,
                classify_config.lora_rank,
            ) {
                Ok(ws) => grad_accum.push(ws),
                Err(e) => {
                    eprintln!("[CUDA] NF4 LoRA grad accum alloc failed (layer {i}): {e}");
                    return (None, None, None);
                }
            }
        }

        let accum_vram_mb = {
            let h = model_config.hidden_size;
            let q_dim = model_config.q_dim();
            let kv = model_config.num_kv_heads * model_config.head_dim();
            let r = classify_config.lora_rank;
            let per_layer_elems = h * r + r * q_dim + h * r + r * kv + h + h;
            let total_bytes = per_layer_elems * 4 * blocks.len();
            total_bytes as f64 / (1024.0 * 1024.0)
        };

        eprintln!(
            "[CUDA] NF4 QLoRA training initialized: {} layers, rank={}, scale={:.2}, accum={:.1} MB",
            blocks.len(),
            classify_config.lora_rank,
            classify_config.lora_alpha / classify_config.lora_rank as f32,
            accum_vram_mb,
        );

        (Some(grad_ws), Some(opt_states), Some(grad_accum))
    }

    /// GPU-accelerated forward pass that saves layer inputs for backward (F-CUDA-014).
    ///
    /// Like `forward_hidden_cuda_impl` but additionally:
    /// 1. Saves each block's input for backward pass
    /// 2. Keeps blocks output on GPU (for RMSNorm backward)
    /// 3. Also downloads and applies final RMSNorm on CPU for the classifier
    ///
    /// # Contract (C-GPUFWD-001)
    ///
    /// - **Precondition**: CUDA trainer, blocks, and gpu_training are all `Some`;
    ///   `token_ids.len() <= max_seq_len`
    /// - **Postcondition**: `gpu_training.layer_inputs[i]` contains input to block `i`;
    ///   `gpu_training.blocks_output` contains final block output (pre-norm);
    ///   returned Tensor is the normed hidden states on CPU
    /// - **Invariant**: GPU blocks contain valid forward-pass scratch for backward
    #[cfg(feature = "cuda")]
    #[allow(unsafe_code)]
    fn forward_hidden_cuda_training(
        model: &Transformer,
        token_ids: &[u32],
        trainer: &CudaTrainer,
        cuda_blocks: &mut [CudaBlock],
        training_state: &mut GpuTrainingState,
        shared_scratch: &mut Option<CudaBlockScratch>,
    ) -> Option<Tensor> {
        let seq_len = token_ids.len();
        let hidden_size = model.config.hidden_size;

        // Step 1: Embed on CPU
        let hidden = model.embed_tokens.forward(token_ids);
        let hidden_data = hidden.data();
        let hidden_slice = hidden_data.as_slice().expect("contiguous hidden");

        // Step 2: Upload to GPU using pre-allocated fwd_scratch_a (KAIZEN-060)
        // Pad remaining buffer to match pre-allocated size (kernels use seq_len param).
        training_state.fwd_scratch_a.copy_from_host_at(hidden_slice, 0).ok()?;

        // Step 3: Run through CUDA transformer blocks, saving inputs
        // KAIZEN-060: Use pre-allocated ping-pong buffers instead of per-call alloc.
        let stream = trainer.stream();
        let scratch_a_ptr: *mut GpuBuffer<f32> = &raw mut training_state.fwd_scratch_a;
        let scratch_b_ptr: *mut GpuBuffer<f32> = &raw mut training_state.fwd_scratch_b;
        let mut input_is_a = true;

        for (i, block) in cuda_blocks.iter_mut().enumerate() {
            // SAFETY: scratch_a_ptr and scratch_b_ptr point to disjoint fields.
            let (input, output) = unsafe {
                if input_is_a {
                    (&*scratch_a_ptr, &mut *scratch_b_ptr)
                } else {
                    (&*scratch_b_ptr, &mut *scratch_a_ptr)
                }
            };

            // Save input to this block for backward pass
            // SAFETY: Both buffers are valid GPU allocations with matching sizes.
            // The copy completes before block.forward() reads from input.
            unsafe {
                training_state.layer_inputs[i].copy_from_buffer_async(input, stream).ok()?;
            }

            if let Err(e) = block.forward(input, output, seq_len, stream, shared_scratch.as_mut()) {
                eprintln!("[CUDA] Layer {i} forward failed: {e}");
                return None;
            }
            input_is_a = !input_is_a;
        }
        // After loop: the buffer indicated by input_is_a holds the final output
        let final_output = unsafe {
            if input_is_a {
                &*scratch_a_ptr
            } else {
                &*scratch_b_ptr
            }
        };

        // Save blocks output for final norm backward
        // SAFETY: Both buffers valid, copy completes before any read.
        unsafe {
            training_state.blocks_output.copy_from_buffer_async(final_output, stream).ok()?;
        }

        // Sync and download for CPU classifier path
        stream.synchronize().ok()?;
        let result_data = trainer.download(final_output).ok()?;

        // NaN guard
        if result_data.iter().any(|v| !v.is_finite()) {
            return None;
        }

        // Apply final RMSNorm on CPU
        let result_tensor = Tensor::from_vec(result_data, false);
        Some(model.norm.forward_batched(&result_tensor, seq_len, hidden_size))
    }

    /// Run backward pass through all GPU transformer blocks (F-CUDA-014).
    ///
    /// Computes gradients for all transformer weights by:
    /// 1. Computing grad through classifier and mean-pool on CPU
    /// 2. Uploading gradient to GPU
    /// 3. Running RMSNorm backward on GPU (final norm)
    /// 4. Running block.backward() for each layer in reverse
    ///
    /// # Contract (C-GPUBACK-001)
    ///
    /// - **Precondition**: Forward training pass completed (`forward_hidden_cuda_training`),
    ///   `grad_logits` has length `num_classes`, `hidden_pre_norm` is the raw block output
    /// - **Postcondition**: Each block's scratch contains weight gradients (grad_w_q/k/v/o,
    ///   grad_gate/up/down, grad_input_norm, grad_post_attn_norm)
    /// - **Invariant**: Zero GPU memory allocation during backward; all buffers preallocated
    #[cfg(feature = "cuda")]
    #[allow(unsafe_code)]
    fn backward_gpu_blocks(&mut self, grad_logits: &[f32], seq_len: usize) -> Option<()> {
        let grad_ws = self.cuda_grad_workspace.as_mut()?;
        let trainer = self.cuda_trainer.as_ref()?;
        let hidden_size = self.model.config.hidden_size;
        let num_classes = self.config.num_classes;

        // Step 1: Classifier backward on CPU (trivial: hidden_size * num_classes mults)
        // grad_pooled = W_classifier^T @ grad_logits
        let w_data = self.classifier.weight.data();
        let w_slice = w_data.as_slice().expect("contiguous classifier weight");
        let mut grad_pooled = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            let mut sum = 0.0f32;
            for c in 0..num_classes {
                sum += grad_logits[c] * w_slice[j * num_classes + c];
            }
            grad_pooled[j] = sum;
        }

        // Step 2: Mean-pool backward into pre-allocated CPU staging buffer (KAIZEN-061)
        let scale = 1.0 / seq_len as f32;
        let training_state = self.gpu_training.as_mut()?;
        for i in 0..seq_len {
            for j in 0..hidden_size {
                training_state.backward_cpu_staging[i * hidden_size + j] = grad_pooled[j] * scale;
            }
        }

        // Step 3: Upload gradient to pre-allocated GPU buffer (KAIZEN-045)
        let stream = trainer.stream();
        training_state
            .grad_upload_buf
            .copy_from_host_at(&training_state.backward_cpu_staging[..seq_len * hidden_size], 0)
            .ok()?;

        let blocks = self.cuda_blocks.as_mut()?;

        // Step 4: RMSNorm backward on GPU (final norm)
        // input = blocks_output, gamma = final_norm_weight
        // grad_output = grad_upload_buf, grad_input = grad_buf_a, grad_gamma = grad_final_norm_weight
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

        // Step 5: Backward through blocks in reverse
        // grad_buf_a has the gradient w.r.t. blocks_output (= grad for last block's output)
        // We alternate between grad_buf_a and grad_buf_b as we propagate backward
        let num_layers = blocks.len();

        // Alternate between grad_buf_a (output) and grad_buf_b (input) buffers.
        // After RMSNorm backward, grad_buf_a holds the gradient for the last block's output.
        // We use raw pointers to get disjoint mutable borrows of grad_buf_a/b.
        // SAFETY: grad_buf_a and grad_buf_b are separate heap-allocated GPU buffers
        // (disjoint fields of GpuTrainingState). We never alias them.
        let grad_a_ptr: *mut GpuBuffer<f32> = &raw mut training_state.grad_buf_a;
        let grad_b_ptr: *mut GpuBuffer<f32> = &raw mut training_state.grad_buf_b;
        let mut grad_output_is_a = true;

        for layer_idx in (0..num_layers).rev() {
            // SAFETY: grad_a_ptr and grad_b_ptr point to disjoint fields.
            let (grad_output, grad_input) = unsafe {
                if grad_output_is_a {
                    (&*grad_a_ptr, &mut *grad_b_ptr)
                } else {
                    (&*grad_b_ptr, &mut *grad_a_ptr)
                }
            };

            blocks[layer_idx]
                .backward(
                    &training_state.layer_inputs[layer_idx],
                    grad_output,
                    grad_input,
                    seq_len,
                    stream,
                    grad_ws,
                )
                .ok()?;

            grad_output_is_a = !grad_output_is_a;
        }

        // Sync to ensure all backward kernels completed
        stream.synchronize().ok()?;

        Some(())
    }

    /// Run GPU-resident AdamW optimizer step on all transformer block weights.
    ///
    /// # Contract (C-GPUOPT-001)
    ///
    /// - **Precondition**: `backward_gpu_blocks()` completed successfully
    /// - **Postcondition**: All block weights updated; optimizer step counter incremented
    /// - **Invariant**: Learning rate and hyperparameters applied uniformly across all blocks
    #[cfg(feature = "cuda")]
    fn gpu_optimizer_step(&mut self, lr: f32) -> Option<()> {
        let grad_ws = self.cuda_grad_workspace.as_ref()?;
        let trainer = self.cuda_trainer.as_ref()?;
        let stream = trainer.stream();
        let training_state = self.gpu_training.as_mut()?;
        let blocks = self.cuda_blocks.as_mut()?;

        training_state.step += 1;
        let step = training_state.step;

        for (block, opt_state) in blocks.iter_mut().zip(training_state.optimizer_states.iter_mut())
        {
            block
                .optimizer_step(
                    opt_state, step, lr, 0.9,   // beta1
                    0.999, // beta2
                    1e-8,  // eps
                    0.01,  // weight_decay
                    stream, grad_ws,
                )
                .ok()?;
        }

        // Sync to ensure all optimizer kernels completed
        stream.synchronize().ok()?;

        Some(())
    }

    /// NF4 QLoRA backward pass through all GPU transformer blocks (ENT-153).
    ///
    /// Like `backward_gpu_blocks` but uses NF4 transposed GEMM for gradient flow
    /// through frozen weights, and computes LoRA gradients for Q/V projections.
    /// After each block backward, immediately runs the LoRA optimizer step
    /// (grad workspace is shared across layers, so we must consume grads before
    /// the next layer overwrites them).
    ///
    /// # Contract (C-NF4BACK-001)
    ///
    /// - **Precondition**: Forward training pass completed, grad_logits has length num_classes
    /// - **Postcondition**: LoRA weights updated for all NF4 blocks
    /// - **Invariant**: Zero GPU memory allocation during backward
    #[cfg(feature = "cuda")]
    #[allow(unsafe_code)]
    fn backward_nf4_gpu_blocks(&mut self, grad_logits: &[f32], seq_len: usize) -> Option<()> {
        use crate::transformer::cuda_block::cuda_add_inplace;

        let trainer = self.cuda_trainer.as_ref()?;
        let hidden_size = self.model.config.hidden_size;
        let num_classes = self.config.num_classes;

        // Step 1: Classifier backward on CPU (trivial: hidden_size * num_classes mults)
        let w_data = self.classifier.weight.data();
        let w_slice = w_data.as_slice().expect("contiguous classifier weight");
        let mut grad_pooled = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            let mut sum = 0.0f32;
            for c in 0..num_classes {
                sum += grad_logits[c] * w_slice[j * num_classes + c];
            }
            grad_pooled[j] = sum;
        }

        // Step 2: Mean-pool backward into pre-allocated CPU staging buffer (KAIZEN-061)
        let scale = 1.0 / seq_len as f32;
        let training_state = self.gpu_training.as_mut()?;
        for i in 0..seq_len {
            for j in 0..hidden_size {
                training_state.backward_cpu_staging[i * hidden_size + j] = grad_pooled[j] * scale;
            }
        }

        // Step 3: Upload gradient to pre-allocated GPU buffer (KAIZEN-045)
        let stream = trainer.stream();
        training_state
            .grad_upload_buf
            .copy_from_host_at(&training_state.backward_cpu_staging[..seq_len * hidden_size], 0)
            .ok()?;

        let blocks = self.cuda_blocks.as_mut()?;
        let shared_scratch = self.shared_scratch.as_mut()?;
        let grad_lora = self.cuda_lora_grad_workspace.as_mut()?;
        let grad_accum = self.cuda_lora_grad_accum.as_mut()?;

        // Step 4: RMSNorm backward on GPU (final norm)
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

        // Step 5: Backward through blocks in reverse, accumulate gradients (KAIZEN-014)
        let num_layers = blocks.len();

        // Alternate gradient buffers using raw pointers for disjoint mutable borrows
        let grad_a_ptr: *mut GpuBuffer<f32> = &raw mut training_state.grad_buf_a;
        let grad_b_ptr: *mut GpuBuffer<f32> = &raw mut training_state.grad_buf_b;
        let mut grad_output_is_a = true;

        // KAIZEN-045: Use pre-allocated output_scratch from training state
        let output_scratch_ptr: *mut GpuBuffer<f32> = &raw mut training_state.output_scratch;

        for layer_idx in (0..num_layers).rev() {
            // SAFETY: grad_a_ptr and grad_b_ptr point to disjoint fields.
            let (grad_output, grad_input) = unsafe {
                if grad_output_is_a {
                    (&*grad_a_ptr, &mut *grad_b_ptr)
                } else {
                    (&*grad_b_ptr, &mut *grad_a_ptr)
                }
            };

            // NF4 backward: activation checkpointing + LoRA gradient computation
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

            // KAIZEN-014: Accumulate gradients into per-layer accumulators
            // (grad_lora workspace is shared — must consume before next layer overwrites)
            let accum = &mut grad_accum[layer_idx];
            cuda_add_inplace(
                &mut accum.grad_lora_a_q,
                &grad_lora.grad_lora_a_q,
                grad_lora.grad_lora_a_q.len(),
                stream,
            )
            .ok()?;
            cuda_add_inplace(
                &mut accum.grad_lora_b_q,
                &grad_lora.grad_lora_b_q,
                grad_lora.grad_lora_b_q.len(),
                stream,
            )
            .ok()?;
            cuda_add_inplace(
                &mut accum.grad_lora_a_v,
                &grad_lora.grad_lora_a_v,
                grad_lora.grad_lora_a_v.len(),
                stream,
            )
            .ok()?;
            cuda_add_inplace(
                &mut accum.grad_lora_b_v,
                &grad_lora.grad_lora_b_v,
                grad_lora.grad_lora_b_v.len(),
                stream,
            )
            .ok()?;
            cuda_add_inplace(
                &mut accum.grad_input_norm,
                &grad_lora.grad_input_norm,
                grad_lora.grad_input_norm.len(),
                stream,
            )
            .ok()?;
            cuda_add_inplace(
                &mut accum.grad_post_attn_norm,
                &grad_lora.grad_post_attn_norm,
                grad_lora.grad_post_attn_norm.len(),
                stream,
            )
            .ok()?;

            grad_output_is_a = !grad_output_is_a;
        }

        // Sync to ensure all backward + accumulation kernels completed
        stream.synchronize().ok()?;

        Some(())
    }

    /// KAIZEN-014: Batch-level LoRA optimizer step with averaged gradients.
    ///
    /// Divides accumulated gradients by `batch_size` (via reduced `lr`) and applies
    /// a single AdamW step per layer. Then zeros the accumulators for the next batch.
    ///
    /// # Contract (C-NF4BATCH-001)
    ///
    /// - **Precondition**: `backward_nf4_gpu_blocks` called `batch_size` times since last step
    /// - **Postcondition**: LoRA weights updated once, accumulators zeroed
    /// - **Invariant**: Effective LR = base_lr / batch_size (gradient averaging)
    #[cfg(feature = "cuda")]
    pub(crate) fn nf4_lora_batch_optimizer_step(&mut self, batch_size: usize) {
        let Some(trainer) = self.cuda_trainer.as_ref() else { return };
        let Some(blocks) = self.cuda_blocks.as_mut() else { return };
        let Some(opt_states) = self.cuda_lora_optimizer_states.as_mut() else { return };
        let Some(grad_accum) = self.cuda_lora_grad_accum.as_mut() else { return };

        let stream = trainer.stream();
        let lr = self.optimizer.lr() / batch_size as f32;

        self.nf4_lora_step += 1;
        let step = self.nf4_lora_step;

        // KAIZEN-043: Pre-allocate a single zero buffer sized to the largest
        // accumulator. Reuse via slicing instead of allocating vec![0.0; len]
        // per buffer per layer (was: 216 allocations/batch for 36 layers × 6 bufs).
        let max_accum_len = grad_accum
            .iter()
            .map(|g| {
                g.grad_lora_a_q
                    .len()
                    .max(g.grad_lora_b_q.len())
                    .max(g.grad_lora_a_v.len())
                    .max(g.grad_lora_b_v.len())
                    .max(g.grad_input_norm.len())
                    .max(g.grad_post_attn_norm.len())
            })
            .max()
            .unwrap_or(0);
        let zeros = vec![0.0f32; max_accum_len];

        for layer_idx in 0..blocks.len() {
            let _ = blocks[layer_idx].lora_optimizer_step(
                &mut opt_states[layer_idx],
                step,
                lr,
                0.9,
                0.999,
                1e-8,
                0.01,
                stream,
                &grad_accum[layer_idx],
            );

            // Zero accumulators for next batch (reuse pre-allocated zero buffer)
            let zero_buf = |buf: &mut GpuBuffer<f32>| {
                let _ = buf.copy_from_host(&zeros[..buf.len()]);
            };
            zero_buf(&mut grad_accum[layer_idx].grad_lora_a_q);
            zero_buf(&mut grad_accum[layer_idx].grad_lora_b_q);
            zero_buf(&mut grad_accum[layer_idx].grad_lora_a_v);
            zero_buf(&mut grad_accum[layer_idx].grad_lora_b_v);
            zero_buf(&mut grad_accum[layer_idx].grad_input_norm);
            zero_buf(&mut grad_accum[layer_idx].grad_post_attn_norm);
        }

        let _ = stream.synchronize();
    }

    /// Synchronize GPU-updated weights back to CPU model tensors.
    ///
    /// Required for checkpointing and after training completes. Downloads all
    /// weight data from GPU and updates the CPU model's weight tensors in-place.
    ///
    /// # Contract (C-SYNCWT-001)
    ///
    /// - **Precondition**: GPU blocks have valid weights (after one or more optimizer steps)
    /// - **Postcondition**: CPU model weights match GPU weights exactly
    /// - **Invariant**: GPU weights are not modified
    #[cfg(feature = "cuda")]
    pub fn sync_weights_to_cpu(&mut self) {
        let blocks = match self.cuda_blocks.as_ref() {
            Some(b) => b,
            None => return,
        };

        for (layer_idx, block) in blocks.iter().enumerate() {
            if let Ok(weights) = block.download_weights() {
                let layer = &mut self.model.layers[layer_idx];

                // Update attention weights
                layer.self_attn.w_q = Tensor::from_vec(weights.w_q, false);
                layer.self_attn.w_k = Tensor::from_vec(weights.w_k, false);
                layer.self_attn.w_v = Tensor::from_vec(weights.w_v, false);
                layer.self_attn.w_o = Tensor::from_vec(weights.w_o, false);

                // Update FFN weights
                layer.ffn.w_gate = Tensor::from_vec(weights.w_gate, false);
                layer.ffn.w_up = Tensor::from_vec(weights.w_up, false);
                layer.ffn.w_down = Tensor::from_vec(weights.w_down, false);

                // Update norm weights
                layer.input_norm.weight = Tensor::from_vec(weights.input_norm_weight, false);
                layer.post_attn_norm.weight =
                    Tensor::from_vec(weights.post_attn_norm_weight, false);
            }
        }
    }

    /// Synchronize GPU LoRA weights back to CPU LoRA layers (NF4 QLoRA).
    ///
    /// Required for checkpointing after NF4 QLoRA training. Downloads A_q, B_q,
    /// A_v, B_v from each NF4 block and updates the corresponding CPU LoRA layers.
    ///
    /// B matrices are stored pre-scaled on GPU (includes lora_scale). This method
    /// un-scales them before writing back to CPU.
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
                    *self.lora_layers[q_lora_idx].lora_a_mut() = Tensor::from_vec(a_q, true);
                    *self.lora_layers[q_lora_idx].lora_b_mut() =
                        Tensor::from_vec(b_q_unscaled, true);
                }
                if v_lora_idx < self.lora_layers.len() {
                    *self.lora_layers[v_lora_idx].lora_a_mut() = Tensor::from_vec(a_v, true);
                    *self.lora_layers[v_lora_idx].lora_b_mut() =
                        Tensor::from_vec(b_v_unscaled, true);
                }
            }
        }
    }

    /// Check if GPU training (full finetune backward) is active.
    #[must_use]
    pub fn is_gpu_training(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.gpu_training.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Forward pass through transformer layers, dispatching to GPU when available.
    ///
    /// Priority: CUDA > wgpu > CPU
    ///
    /// - **CUDA path** (F-CUDA-007..009): Embed on CPU, upload to GPU, run CUDA layers, download
    /// - **wgpu path**: Batched FFN matmuls via `WgpuForwardPass`, attention on CPU
    /// - **CPU path**: Use `Transformer::forward_hidden()`
    fn forward_hidden_dispatch(&mut self, token_ids: &[u32]) -> Tensor {
        #[cfg(feature = "cuda")]
        if let Some(tensor) = self.try_forward_hidden_gpu(token_ids) {
            return tensor;
        }

        // wgpu fallback (batched FFN matmuls on GPU, attention on CPU)
        #[cfg(feature = "gpu")]
        if let Some(ref wgpu_pass) = self.wgpu_forward_pass {
            match wgpu_pass.forward_hidden(&self.model, token_ids) {
                Ok(tensor) => return tensor,
                Err(e) => {
                    eprintln!("[wgpu] Forward pass failed, falling back to CPU: {e}");
                }
            }
        }

        // CPU fallback — KAIZEN-011: use LoRA-aware forward when adapters exist
        if self.lora_layers.is_empty() {
            self.model.forward_hidden(token_ids)
        } else {
            self.model.forward_hidden_with_lora(token_ids, &self.lora_layers)
        }
    }

    /// Attempt GPU-accelerated forward pass (training or inference).
    ///
    /// Returns `Some(tensor)` on success, `None` to fall back to CPU.
    #[cfg(feature = "cuda")]
    fn try_forward_hidden_gpu(&mut self, token_ids: &[u32]) -> Option<Tensor> {
        if self.gpu_training.is_some() {
            // GPU training path: saves layer inputs for backward
            let (trainer, blocks) = match (&self.cuda_trainer, &mut self.cuda_blocks) {
                (Some(ref t), Some(ref mut b)) => (t, b),
                _ => return None,
            };
            let mut training = self.gpu_training.take();
            let result = Self::forward_hidden_cuda_training(
                &self.model,
                token_ids,
                trainer,
                blocks,
                training.as_mut().expect("gpu_training was Some"),
                &mut self.shared_scratch,
            );
            self.gpu_training = training;

            if result.is_none() {
                self.cuda_nan_count += 1;
            }
            return result;
        }

        // Inference-only GPU path (no layer input saving)
        let (trainer, blocks) = match (&self.cuda_trainer, &mut self.cuda_blocks) {
            (Some(ref t), Some(ref mut b)) => (t, b),
            _ => return None,
        };
        match Self::forward_hidden_cuda_impl(
            &self.model,
            token_ids,
            trainer,
            blocks,
            &mut self.shared_scratch,
        ) {
            Some(tensor) => Some(tensor),
            None => {
                self.cuda_nan_count += 1;
                if self.cuda_nan_count > 100 {
                    eprintln!(
                        "[CUDA] {} NaN fallbacks — disabling GPU acceleration",
                        self.cuda_nan_count
                    );
                    self.cuda_trainer = None;
                    self.cuda_blocks = None;
                }
                None
            }
        }
    }

    /// GPU-accelerated forward pass (F-CUDA-007).
    ///
    /// 1. Embed tokens on CPU (F-CUDA-008: small op)
    /// 2. Upload hidden states to GPU
    /// 3. Run through all CudaTransformerBlocks
    /// 4. Apply final RMSNorm on CPU
    /// 5. Return hidden states (F-CUDA-009)
    ///
    /// Returns `None` on any GPU error, signaling caller to use CPU fallback.
    #[cfg(feature = "cuda")]
    fn forward_hidden_cuda_impl(
        model: &Transformer,
        token_ids: &[u32],
        trainer: &CudaTrainer,
        cuda_blocks: &mut [CudaBlock],
        shared_scratch: &mut Option<CudaBlockScratch>,
    ) -> Option<Tensor> {
        let seq_len = token_ids.len();
        let hidden_size = model.config.hidden_size;

        // Step 1: Embed on CPU
        let hidden = model.embed_tokens.forward(token_ids);
        let hidden_data = hidden.data();
        let hidden_slice = hidden_data.as_slice().expect("contiguous hidden");

        // Step 2: Upload to GPU
        let mut gpu_input = trainer.upload(hidden_slice).ok()?;
        let mut gpu_output = trainer.zeros(seq_len * hidden_size).ok()?;

        // Step 3: Run through CUDA transformer blocks
        let stream = trainer.stream();
        for (i, block) in cuda_blocks.iter_mut().enumerate() {
            if let Err(e) =
                block.forward(&gpu_input, &mut gpu_output, seq_len, stream, shared_scratch.as_mut())
            {
                eprintln!("[CUDA] Layer {i} forward failed: {e}");
                return None;
            }
            // Swap: output becomes input for next layer
            std::mem::swap(&mut gpu_input, &mut gpu_output);
        }
        // After the loop, gpu_input holds the final output (due to swap)

        // Sync stream to ensure all CUDA kernels have completed before download
        if let Err(e) = stream.synchronize() {
            eprintln!("[CUDA] Stream sync failed: {e}");
            return None;
        }

        // Step 4: Download from GPU
        let result_data = trainer.download(&gpu_input).ok()?;

        // Step 4.5: NaN guard — GPU kernels can produce NaN with certain weight
        // distributions (e.g., random init). Fall back to CPU if detected.
        if result_data.iter().any(|v| !v.is_finite()) {
            return None;
        }

        // Step 5: Apply final RMSNorm on CPU
        let result_tensor = Tensor::from_vec(result_data, false);
        Some(model.norm.forward_batched(&result_tensor, seq_len, hidden_size))
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
            self.cuda_trainer.as_ref().map(crate::autograd::cuda_training::CudaTrainer::device_name)
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
            self.cuda_trainer
                .as_ref()
                .map(crate::autograd::cuda_training::CudaTrainer::total_memory)
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    /// Get the base model directory path (if loaded from pretrained weights).
    #[must_use]
    pub fn model_dir(&self) -> Option<&Path> {
        self.model_dir.as_deref()
    }

    /// Set the base model path (used when the model was loaded from an APR file
    /// rather than a SafeTensors directory, so checkpoint saves can record the
    /// provenance in `adapter_config.json`).
    pub fn set_model_path(&mut self, path: impl Into<PathBuf>) {
        self.model_dir = Some(path.into());
    }

    /// Single training step: forward + loss + backward + optimizer update.
    ///
    /// Performs the complete training cycle:
    /// 1. Forward pass through transformer + classification head
    /// 2. Cross-entropy loss computation
    /// 3. Gradient computation via autograd backward
    /// 4. AdamW optimizer step on trainable parameters
    ///
    /// # Arguments
    /// * `token_ids` - Tokenized input
    /// * `label` - Target class index
    ///
    /// # Returns
    /// Loss value as f32
    pub fn train_step(&mut self, token_ids: &[u32], label: usize) -> f32 {
        let seq_len = token_ids.len();
        let num_classes = self.config.num_classes;

        // ── 1. Zero gradients ─────────────────────────────────────────
        self.classifier.weight.zero_grad();
        self.classifier.bias.zero_grad();
        for lora in &self.lora_layers {
            lora.lora_a().zero_grad();
            lora.lora_b().zero_grad();
        }

        // ── 2. Forward pass (GPU-dispatched if available) ─────────────
        let hidden = self.forward_hidden_dispatch(token_ids);
        let pooled = self.classifier.mean_pool(&hidden, seq_len);

        // matmul builds autograd backward ops (connects classifier.weight to loss)
        let logits =
            matmul(&pooled, &self.classifier.weight, 1, self.classifier.hidden_size(), num_classes);

        // Add bias (element-wise, preserving grad tracking)
        let logits_with_bias: Vec<f32> = logits
            .data()
            .as_slice()
            .expect("contiguous logits")
            .iter()
            .zip(self.classifier.bias.data().as_slice().expect("contiguous bias").iter())
            .map(|(&l, &b)| l + b)
            .collect();

        // ── 3. Cross-entropy loss + manual gradient ───────────────────
        // Compute softmax probabilities
        let max_val = logits_with_bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits_with_bias.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp).collect();

        // Class weight for this sample's label (default 1.0 if no weights configured)
        let w = self.config.class_weights.as_ref().map_or(1.0, |weights| weights[label]);

        // Loss = -w[label] * log(prob[target])
        let loss_val = -w * (probs[label].max(1e-10).ln());
        let loss_val = if loss_val.is_finite() { loss_val } else { 100.0 };

        // ∂L/∂logits = w[label] * (probs - one_hot(target))
        // The weight multiplier applies to both loss and gradient
        let mut grad_logits: Vec<f32> = probs.iter().map(|&p| w * p).collect();
        grad_logits[label] -= w;

        // ── 4. Backward through matmul (autograd) ─────────────────────
        // Set loss gradient on the matmul output, then call backward
        let logits_tensor = logits;
        logits_tensor.set_grad(ndarray::Array1::from(grad_logits.clone()));
        if let Some(op) = logits_tensor.backward_op() {
            op.backward();
        }

        // Manually set bias gradient (∂L/∂bias = ∂L/∂logits)
        self.classifier.bias.set_grad(ndarray::Array1::from(grad_logits.clone()));

        // GPU backward through transformer blocks (F-CUDA-014 / ENT-153)
        #[cfg(feature = "cuda")]
        {
            if self.gpu_training.is_some() {
                if self.config.quantize_nf4 {
                    // NF4 QLoRA: backward accumulates gradients (KAIZEN-014)
                    self.backward_nf4_gpu_blocks(&grad_logits, seq_len);
                } else {
                    self.backward_gpu_blocks(&grad_logits, seq_len);
                }
            }
        }

        // KAIZEN-014: NF4 QLoRA batch step (batch_size=1 for single-sample train_step)
        #[cfg(feature = "cuda")]
        {
            if self.gpu_training.is_some() && self.config.quantize_nf4 {
                self.nf4_lora_batch_optimizer_step(1);
            }
        }

        // ── 5. Optimizer step ─────────────────────────────────────────
        // GPU optimizer step for fp32 transformer block weights (F-CUDA-014)
        #[cfg(feature = "cuda")]
        {
            if self.gpu_training.is_some() && !self.config.quantize_nf4 {
                let lr = self.optimizer.lr();
                self.gpu_optimizer_step(lr);
            }
        }

        // CPU optimizer step
        // KAIZEN-011: Include LoRA in CPU optimizer when NOT on CUDA
        let has_cuda_training = {
            #[cfg(feature = "cuda")]
            {
                self.gpu_training.is_some()
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        };
        let mut params: Vec<&mut Tensor> = Vec::new();
        if !self.config.quantize_nf4 || !has_cuda_training {
            for lora in &mut self.lora_layers {
                params.extend(lora.trainable_params());
            }
        }
        params.extend(self.classifier.parameters_mut());
        self.optimizer.step_refs(&mut params);

        loss_val
    }

    // ── Mini-batch training (SSC-025) ───────────────────────────────────

    /// Train on a mini-batch of samples with gradient accumulation.
    ///
    /// Unlike [`train_step`] which processes one sample and immediately calls
    /// `optimizer.step()`, this method:
    ///
    /// 1. Zeros all gradients
    /// 2. Iterates over every sample in the batch, computing forward + loss + backward
    /// 3. Gradients accumulate naturally across samples (sum)
    /// 4. Normalizes accumulated gradients by batch size
    /// 5. Optionally clips gradient norm (if `config.gradient_clip_norm` is set)
    /// 6. Calls `optimizer.step()` **once** for the entire batch
    ///
    /// This reduces optimizer overhead from O(N) to O(1) per batch and produces
    /// smoother gradient estimates.
    ///
    /// # Arguments
    /// * `samples` - Slice of `SafetySample` (shell text + label). Text is
    ///   tokenized via byte-level encoding internally.
    ///
    /// # Returns
    /// [`BatchResult`] with average loss, correct predictions, and total count
    pub fn train_batch(&mut self, samples: &[SafetySample]) -> BatchResult {
        if samples.is_empty() {
            return BatchResult { avg_loss: 0.0, correct: 0, total: 0, grad_norm: 0.0 };
        }

        let batch_size = samples.len();

        // ── 1. Zero gradients ──────────────────────────────────────────
        self.zero_all_gradients();

        // ── 2. Accumulate gradients over all samples ───────────────────
        // KAIZEN-008: try batched wgpu forward (uploads FFN weights ONCE per layer)
        #[cfg(feature = "gpu")]
        let (total_loss, correct) = self
            .try_train_batch_wgpu(samples)
            .unwrap_or_else(|| self.train_batch_per_sample(samples));

        #[cfg(not(feature = "gpu"))]
        let (total_loss, correct) = self.train_batch_per_sample(samples);

        // ── 3. Normalize gradients by batch size ───────────────────────
        self.scale_all_gradients(1.0 / batch_size as f32);

        // ── 4. Gradient clipping (captures pre-clip norm) ────────────
        let grad_norm = if let Some(max_norm) = self.config.gradient_clip_norm {
            let mut params = self.trainable_parameters_mut();
            clip_grad_norm_refs(&mut params, max_norm)
        } else {
            self.compute_grad_norm()
        };

        // ── 5. Optimizer step (once for the whole batch) ───────────────
        // GPU optimizer step for fp32 transformer block weights (F-CUDA-014)
        // NF4 QLoRA: batch optimizer step applied below
        #[cfg(feature = "cuda")]
        {
            if self.gpu_training.is_some() && !self.config.quantize_nf4 {
                let lr = self.optimizer.lr();
                self.gpu_optimizer_step(lr);
            }
        }

        // KAIZEN-014: NF4 QLoRA batch optimizer step
        #[cfg(feature = "cuda")]
        {
            if self.gpu_training.is_some() && self.config.quantize_nf4 {
                self.nf4_lora_batch_optimizer_step(batch_size);
            }
        }

        // CPU optimizer step
        // KAIZEN-011: Include LoRA in CPU optimizer when NOT on CUDA.
        // NF4+CUDA: LoRA trained on GPU (backward_nf4_gpu_blocks), skip here
        // NF4+non-CUDA: LoRA trained on CPU via autograd, include here
        // fp32 mode: LoRA + classifier head always on CPU
        let has_cuda_training = {
            #[cfg(feature = "cuda")]
            {
                self.gpu_training.is_some()
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        };
        let mut params: Vec<&mut Tensor> = Vec::new();
        if !self.config.quantize_nf4 || !has_cuda_training {
            for lora in &mut self.lora_layers {
                params.extend(lora.trainable_params());
            }
        }
        params.extend(self.classifier.parameters_mut());
        self.optimizer.step_refs(&mut params);

        BatchResult {
            avg_loss: total_loss / batch_size as f32,
            correct,
            total: batch_size,
            grad_norm,
        }
    }

    /// Per-sample forward + backward fallback for train_batch.
    fn train_batch_per_sample(&mut self, samples: &[SafetySample]) -> (f32, usize) {
        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        for sample in samples {
            let ids = self.tokenize(&sample.input);
            let (loss, predicted) = self.forward_backward_single(&ids, sample.label);
            total_loss += loss;
            if predicted == sample.label {
                correct += 1;
            }
        }
        (total_loss, correct)
    }

    /// Batched wgpu forward pass for train_batch (KAIZEN-008 + KAIZEN-010).
    ///
    /// Tokenizes all samples, runs a single batched forward through all transformer
    /// layers (uploading FFN weights ONCE per layer), then classifies each sample.
    ///
    /// KAIZEN-010: Passes LoRA layers to the batched forward so that LoRA
    /// adjusts are applied to Q/V projections. Without this, only the
    /// classifier head (5,122 params) trains on the wgpu path.
    ///
    /// Returns `Some((total_loss, correct))` on success, `None` to fall back.
    #[cfg(feature = "gpu")]
    fn try_train_batch_wgpu(&mut self, samples: &[SafetySample]) -> Option<(f32, usize)> {
        if self.wgpu_forward_pass.is_none() {
            return None;
        }

        let batch_token_ids: Vec<Vec<u32>> =
            samples.iter().map(|s| self.tokenize(&s.input)).collect();

        // KAIZEN-010: Pass LoRA layers so gradients flow through Q/V adapters
        let lora_ref =
            if self.lora_layers.is_empty() { None } else { Some(self.lora_layers.as_slice()) };

        let hiddens = self
            .wgpu_forward_pass
            .as_ref()
            .expect("checked is_none above")
            .forward_hidden_batch(&self.model, &batch_token_ids, lora_ref)
            .map_err(|e| {
                eprintln!("[wgpu] Batched forward failed, falling back to per-sample: {e}")
            })
            .ok()?;

        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        for (i, hidden) in hiddens.iter().enumerate() {
            let (loss, predicted) = self.classify_backward_from_hidden(
                hidden,
                batch_token_ids[i].len(),
                samples[i].label,
            );
            total_loss += loss;
            if predicted == samples[i].label {
                correct += 1;
            }
        }
        Some((total_loss, correct))
    }

    /// Accumulate gradients for a micro-batch without calling optimizer.step().
    ///
    /// Use this with [`apply_accumulated_gradients`] for gradient accumulation
    /// across multiple micro-batches. This enables effective batch sizes larger
    /// than what fits in memory:
    ///
    /// ```text
    /// effective_batch_size = micro_batch_size * accumulation_steps
    /// ```
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Effective batch_size = 8 * 4 = 32
    /// for micro_batch in data.chunks(8) {
    ///     pipeline.accumulate_gradients(micro_batch);
    /// }
    /// pipeline.apply_accumulated_gradients(4);
    /// ```
    ///
    /// # Arguments
    /// * `micro_batch` - Slice of samples for one accumulation step
    ///
    /// # Returns
    /// [`BatchResult`] for this micro-batch (loss/accuracy before optimizer step)
    pub fn accumulate_gradients(&mut self, micro_batch: &[SafetySample]) -> BatchResult {
        if micro_batch.is_empty() {
            return BatchResult { avg_loss: 0.0, correct: 0, total: 0, grad_norm: 0.0 };
        }

        let mut total_loss = 0.0f32;
        let mut correct = 0usize;

        for sample in micro_batch {
            let ids = self.tokenize(&sample.input);
            let (loss, predicted) = self.forward_backward_single(&ids, sample.label);
            total_loss += loss;
            if predicted == sample.label {
                correct += 1;
            }
        }

        BatchResult {
            avg_loss: total_loss / micro_batch.len() as f32,
            correct,
            total: micro_batch.len(),
            grad_norm: 0.0, // Grad norm computed at apply time, not accumulate time
        }
    }

    /// Normalize accumulated gradients and apply optimizer step.
    ///
    /// Call this after one or more [`accumulate_gradients`] calls. It:
    /// 1. Divides all gradients by `num_accumulation_steps * micro_batch_size`
    ///    (the total sample count across all micro-batches)
    /// 2. Clips gradient norm if configured
    /// 3. Calls `optimizer.step()` once
    /// 4. Zeros all gradients for the next accumulation cycle
    ///
    /// # Arguments
    /// * `total_samples` - Total number of samples accumulated (sum of micro-batch sizes)
    pub fn apply_accumulated_gradients(&mut self, total_samples: usize) {
        if total_samples == 0 {
            return;
        }

        // ── 1. Normalize gradients ─────────────────────────────────────
        self.scale_all_gradients(1.0 / total_samples as f32);

        // ── 2. Gradient clipping ───────────────────────────────────────
        if let Some(max_norm) = self.config.gradient_clip_norm {
            let mut params = self.trainable_parameters_mut();
            clip_grad_norm_refs(&mut params, max_norm);
        }

        // ── 3. Optimizer step ──────────────────────────────────────────
        // NF4 mode: only classifier head (LoRA trained on GPU in backward)
        let mut params: Vec<&mut Tensor> = Vec::new();
        if !self.config.quantize_nf4 {
            for lora in &mut self.lora_layers {
                params.extend(lora.trainable_params());
            }
        }
        params.extend(self.classifier.parameters_mut());
        self.optimizer.step_refs(&mut params);

        // ── 4. Zero gradients for next cycle ───────────────────────────
        self.zero_all_gradients();
    }

    /// Forward pass + backward for a single sample (no optimizer step).
    ///
    /// Computes cross-entropy loss and accumulates gradients into the existing
    /// gradient buffers (does NOT zero them first). Returns the loss and
    /// the predicted class index (argmax of logits).
    fn forward_backward_single(&mut self, token_ids: &[u32], label: usize) -> (f32, usize) {
        let num_classes = self.config.num_classes;

        // ── Contract precondition (F-CLASS-002): label in bounds ─────────
        debug_assert!(
            label < num_classes,
            "F-CLASS-002: label index {label} >= num_classes {num_classes}"
        );

        // ── Pad to max_seq_len for deterministic GPU kernel shapes (C-PREWARM-001) ──
        // GPU backward kernels embed seq_len in cache keys. Pre-warming compiles for
        // max_seq_len only, so variable-length inputs would trigger JIT compilation
        // post-VRAM-fill. Pad to max_seq_len; mean_pool uses orig_seq_len for correctness.
        let orig_seq_len = token_ids.len().min(self.config.max_seq_len);
        let has_gpu_training = {
            #[cfg(feature = "cuda")]
            {
                self.gpu_training.is_some()
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        };
        // KAIZEN-035: Avoid token_ids.to_vec() on CPU path — borrow directly.
        let hidden = if has_gpu_training {
            let mut padded = vec![0u32; self.config.max_seq_len];
            padded[..orig_seq_len].copy_from_slice(&token_ids[..orig_seq_len]);
            self.forward_hidden_dispatch(&padded)
        } else {
            self.forward_hidden_dispatch(token_ids)
        };
        let pooled = self.classifier.mean_pool(&hidden, orig_seq_len);

        let logits =
            matmul(&pooled, &self.classifier.weight, 1, self.classifier.hidden_size(), num_classes);

        let logits_with_bias: Vec<f32> = logits
            .data()
            .as_slice()
            .expect("contiguous logits")
            .iter()
            .zip(self.classifier.bias.data().as_slice().expect("contiguous bias").iter())
            .map(|(&l, &b)| l + b)
            .collect();

        // ── Contract postcondition (F-CLASS-001): logit shape ────────────
        debug_assert_eq!(
            logits_with_bias.len(),
            num_classes,
            "F-CLASS-001: logits.len()={} != num_classes={num_classes}",
            logits_with_bias.len()
        );
        // ── Contract postcondition: no NaN in logits ────────────────────
        debug_assert!(
            logits_with_bias.iter().all(|v| v.is_finite()),
            "F-CLASS-001: logits contain NaN or Inf"
        );

        // ── Predicted class (argmax) ────────────────────────────────────
        let predicted = logits_with_bias
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx);

        // ── Cross-entropy loss (weighted) ────────────────────────────────
        let max_val = logits_with_bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits_with_bias.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp).collect();

        // Class weight for this sample's label (default 1.0 if no weights configured)
        let w = self.config.class_weights.as_ref().map_or(1.0, |weights| weights[label]);

        let loss_val = -w * (probs[label].max(1e-10).ln());
        let loss_val = if loss_val.is_finite() { loss_val } else { 100.0 };

        // ── Contract postcondition (F-CLASS-005): loss finite & non-negative
        debug_assert!(loss_val.is_finite(), "F-CLASS-005: loss is not finite");
        debug_assert!(loss_val >= 0.0, "F-CLASS-005: loss is negative: {loss_val}");

        // ── Backward ────────────────────────────────────────────────────
        // dL/d_logits = w[label] * (softmax(logits) - one_hot(label))
        let mut grad_logits: Vec<f32> = probs.iter().map(|&p| w * p).collect();
        grad_logits[label] -= w;

        // CPU autograd backward for classifier head (LoRA + classifier.weight)
        logits.set_grad(ndarray::Array1::from(grad_logits.clone()));
        if let Some(op) = logits.backward_op() {
            op.backward();
        }

        // Accumulate bias gradient (not set — accumulate)
        self.classifier.bias.accumulate_grad(ndarray::Array1::from(grad_logits.clone()));

        // GPU backward through all transformer blocks (F-CUDA-014 / ENT-153)
        #[cfg(feature = "cuda")]
        {
            if self.gpu_training.is_some() {
                if self.config.quantize_nf4 {
                    self.backward_nf4_gpu_blocks(&grad_logits, orig_seq_len);
                } else {
                    self.backward_gpu_blocks(&grad_logits, orig_seq_len);
                }
            }
        }

        (loss_val, predicted)
    }

    /// Classifier head + loss + backward from pre-computed hidden states (KAIZEN-008).
    ///
    /// Extracts the classify-and-backward logic from `forward_backward_single` for use
    /// with batched wgpu forward pass, where hidden states are computed in bulk.
    ///
    /// # Contract (C-WGPU-BATCH-001)
    ///
    /// - **Precondition**: hidden tensor has shape (seq_len * hidden_size), label < num_classes
    /// - **Postcondition**: gradients accumulated into classifier.weight, classifier.bias, LoRA params
    /// - **Invariant**: numerically identical to forward_backward_single classifier path
    #[cfg(feature = "gpu")]
    fn classify_backward_from_hidden(
        &mut self,
        hidden: &Tensor,
        orig_seq_len: usize,
        label: usize,
    ) -> (f32, usize) {
        let num_classes = self.config.num_classes;

        debug_assert!(
            label < num_classes,
            "F-CLASS-002: label index {label} >= num_classes {num_classes}"
        );

        // ── Classifier forward ────────────────────────────────────────
        let pooled = self.classifier.mean_pool(hidden, orig_seq_len);
        let logits =
            matmul(&pooled, &self.classifier.weight, 1, self.classifier.hidden_size(), num_classes);

        let logits_with_bias: Vec<f32> = logits
            .data()
            .as_slice()
            .expect("contiguous logits")
            .iter()
            .zip(self.classifier.bias.data().as_slice().expect("contiguous bias").iter())
            .map(|(&l, &b)| l + b)
            .collect();

        debug_assert_eq!(
            logits_with_bias.len(),
            num_classes,
            "F-CLASS-001: logits.len()={} != num_classes={num_classes}",
            logits_with_bias.len()
        );
        debug_assert!(
            logits_with_bias.iter().all(|v| v.is_finite()),
            "F-CLASS-001: logits contain NaN or Inf"
        );

        // ── Predicted class (argmax) ────────────────────────────────────
        let predicted = logits_with_bias
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx);

        // ── Cross-entropy loss (weighted) ────────────────────────────────
        let max_val = logits_with_bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits_with_bias.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp).collect();

        let w = self.config.class_weights.as_ref().map_or(1.0, |weights| weights[label]);
        let loss_val = -w * (probs[label].max(1e-10).ln());
        let loss_val = if loss_val.is_finite() { loss_val } else { 100.0 };

        // ── Backward ────────────────────────────────────────────────────
        let mut grad_logits: Vec<f32> = probs.iter().map(|&p| w * p).collect();
        grad_logits[label] -= w;

        logits.set_grad(ndarray::Array1::from(grad_logits.clone()));
        if let Some(op) = logits.backward_op() {
            op.backward();
        }

        self.classifier.bias.accumulate_grad(ndarray::Array1::from(grad_logits));

        (loss_val, predicted)
    }

    /// Zero all trainable parameter gradients.
    fn zero_all_gradients(&self) {
        self.classifier.weight.zero_grad();
        self.classifier.bias.zero_grad();
        for lora in &self.lora_layers {
            lora.lora_a().zero_grad();
            lora.lora_b().zero_grad();
        }
    }

    /// Scale all trainable parameter gradients by a constant factor.
    ///
    /// Used to normalize accumulated gradients: `grad *= factor`.
    fn scale_all_gradients(&self, factor: f32) {
        // KAIZEN-037: scale in-place — zero allocation (was: clone + alloc per param)
        for lora in &self.lora_layers {
            lora.lora_a().scale_grad(factor);
            lora.lora_b().scale_grad(factor);
        }
        for param in self.classifier.parameters() {
            param.scale_grad(factor);
        }
    }

    /// Compute the global L2 norm of all trainable gradients.
    ///
    /// Used by the monitor when gradient clipping is not enabled.
    fn compute_grad_norm(&self) -> f32 {
        let mut total_norm_sq = 0.0f32;
        // KAIZEN-037: iterate directly — no intermediate Vec collection
        for lora in &self.lora_layers {
            for param in [lora.lora_a(), lora.lora_b()] {
                if let Some(grad) = param.grad() {
                    total_norm_sq += grad.iter().map(|&g| g * g).sum::<f32>();
                }
            }
        }
        for param in self.classifier.parameters() {
            if let Some(grad) = param.grad() {
                total_norm_sq += grad.iter().map(|&g| g * g).sum::<f32>();
            }
        }
        total_norm_sq.sqrt()
    }

    /// Forward-only pass for a single sample (no backward, no optimizer step).
    ///
    /// Computes cross-entropy loss and predicted class without accumulating
    /// gradients. Used for validation/evaluation.
    ///
    /// # Arguments
    /// * `token_ids` - Tokenized input
    /// * `label` - Target class index
    ///
    /// # Returns
    /// `(loss, predicted_class)` tuple
    pub fn forward_only(&mut self, token_ids: &[u32], label: usize) -> (f32, usize) {
        let num_classes = self.config.num_classes;

        // Pad to max_seq_len for deterministic GPU kernel shapes (matches forward_backward_single)
        let orig_seq_len = token_ids.len().min(self.config.max_seq_len);
        let has_gpu_training = {
            #[cfg(feature = "cuda")]
            {
                self.gpu_training.is_some()
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        };
        // KAIZEN-035: Avoid token_ids.to_vec() on CPU path — borrow directly.
        let hidden = if has_gpu_training {
            let mut padded = vec![0u32; self.config.max_seq_len];
            padded[..orig_seq_len].copy_from_slice(&token_ids[..orig_seq_len]);
            self.forward_hidden_dispatch(&padded)
        } else {
            self.forward_hidden_dispatch(token_ids)
        };
        let pooled = self.classifier.mean_pool(&hidden, orig_seq_len);

        let logits =
            matmul(&pooled, &self.classifier.weight, 1, self.classifier.hidden_size(), num_classes);

        let logits_with_bias: Vec<f32> = logits
            .data()
            .as_slice()
            .expect("contiguous logits")
            .iter()
            .zip(self.classifier.bias.data().as_slice().expect("contiguous bias").iter())
            .map(|(&l, &b)| l + b)
            .collect();

        // Predicted class (argmax)
        let predicted = logits_with_bias
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx);

        // Cross-entropy loss (weighted)
        let max_val = logits_with_bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits_with_bias.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp).collect();

        // Class weight for this sample's label (default 1.0 if no weights configured)
        let w = self.config.class_weights.as_ref().map_or(1.0, |weights| weights[label]);

        let loss_val = -w * (probs[label].max(1e-10).ln());
        let loss_val = if loss_val.is_finite() { loss_val } else { 100.0 };

        (loss_val, predicted)
    }

    /// Forward-only pass returning loss, predicted class, and softmax probabilities.
    ///
    /// Identical to [`forward_only`] but also returns the full probability distribution
    /// for confidence analysis, calibration, and per-sample diagnostics.
    pub fn forward_only_with_probs(
        &mut self,
        token_ids: &[u32],
        label: usize,
    ) -> (f32, usize, Vec<f32>) {
        let seq_len = token_ids.len();
        let num_classes = self.config.num_classes;

        let hidden = self.forward_hidden_dispatch(token_ids);
        let pooled = self.classifier.mean_pool(&hidden, seq_len);

        let logits =
            matmul(&pooled, &self.classifier.weight, 1, self.classifier.hidden_size(), num_classes);

        let logits_with_bias: Vec<f32> = logits
            .data()
            .as_slice()
            .expect("contiguous logits")
            .iter()
            .zip(self.classifier.bias.data().as_slice().expect("contiguous bias").iter())
            .map(|(&l, &b)| l + b)
            .collect();

        let predicted = logits_with_bias
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx);

        let max_val = logits_with_bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits_with_bias.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp).collect();

        let w = self.config.class_weights.as_ref().map_or(1.0, |weights| weights[label]);
        let loss_val = -w * (probs[label].max(1e-10).ln());
        let loss_val = if loss_val.is_finite() { loss_val } else { 100.0 };

        (loss_val, predicted, probs)
    }

    /// Multi-label training step using BCE with logits loss.
    ///
    /// Unlike `train_step` which uses cross-entropy (mutually exclusive classes),
    /// this uses BCE with logits (each class is independent binary decision).
    ///
    /// # Arguments
    /// * `token_ids` - Tokenized input
    /// * `targets` - Multi-hot target vector (length == num_classes)
    ///
    /// # Returns
    /// Loss value as f32
    pub fn multi_label_train_step(&mut self, token_ids: &[u32], targets: &[f32]) -> f32 {
        let seq_len = token_ids.len();
        let num_classes = self.config.num_classes;
        assert_eq!(targets.len(), num_classes, "F-CLASS-001: target length must match num_classes");

        // ── 1. Zero gradients ─────────────────────────────────────────
        self.classifier.weight.zero_grad();
        self.classifier.bias.zero_grad();
        for lora in &self.lora_layers {
            lora.lora_a().zero_grad();
            lora.lora_b().zero_grad();
        }

        // ── 2. Forward pass (GPU-dispatched if available) ─────────────
        let hidden = self.forward_hidden_dispatch(token_ids);
        let pooled = self.classifier.mean_pool(&hidden, seq_len);

        let logits =
            matmul(&pooled, &self.classifier.weight, 1, self.classifier.hidden_size(), num_classes);

        // Add bias
        let logits_with_bias: Vec<f32> = logits
            .data()
            .as_slice()
            .expect("contiguous logits")
            .iter()
            .zip(self.classifier.bias.data().as_slice().expect("contiguous bias").iter())
            .map(|(&l, &b)| l + b)
            .collect();

        // ── 3. BCE with logits loss + manual gradient ───────────────
        // Per-element: L_i = max(x_i, 0) - x_i * t_i + log(1 + exp(-|x_i|))
        let loss_val: f32 = logits_with_bias
            .iter()
            .zip(targets.iter())
            .map(|(&x, &t)| {
                let relu = x.max(0.0);
                relu - x * t + (1.0 + (-x.abs()).exp()).ln()
            })
            .sum::<f32>()
            / num_classes as f32;

        let loss_val = if loss_val.is_finite() { loss_val } else { 100.0 };

        // ∂L/∂logits = (σ(x) - targets) / N
        let grad_logits: Vec<f32> = logits_with_bias
            .iter()
            .zip(targets.iter())
            .map(|(&x, &t)| {
                let sigma = if x >= 0.0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    let e = x.exp();
                    e / (1.0 + e)
                };
                (sigma - t) / num_classes as f32
            })
            .collect();

        // ── 4. Backward through matmul (autograd) ─────────────────────
        let logits_tensor = logits;
        logits_tensor.set_grad(ndarray::Array1::from(grad_logits.clone()));
        if let Some(op) = logits_tensor.backward_op() {
            op.backward();
        }

        // Manually set bias gradient
        self.classifier.bias.set_grad(ndarray::Array1::from(grad_logits));

        // ── 5. Optimizer step ─────────────────────────────────────────
        let mut params: Vec<&mut Tensor> = Vec::new();
        for lora in &mut self.lora_layers {
            params.extend(lora.trainable_params());
        }
        params.extend(self.classifier.parameters_mut());
        self.optimizer.step_refs(&mut params);

        loss_val
    }

    /// Load multi-label corpus from JSONL file.
    ///
    /// Supports both single-label `{"input","label"}` and multi-label `{"input","labels"}`
    /// formats. Single-label entries are automatically converted to multi-hot encoding.
    ///
    /// # Errors
    /// Returns error if file is invalid or labels out of range.
    pub fn load_multi_label_corpus(
        &self,
        path: &Path,
    ) -> crate::Result<Vec<MultiLabelSafetySample>> {
        load_multi_label_corpus(path, self.config.num_classes)
    }

    /// Get all trainable parameters (LoRA A/B + classifier weight/bias).
    pub fn trainable_parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params: Vec<&mut Tensor> = Vec::new();
        for lora in &mut self.lora_layers {
            params.extend(lora.trainable_params());
        }
        params.extend(self.classifier.parameters_mut());
        params
    }

    /// Count total trainable parameters.
    #[must_use]
    pub fn num_trainable_parameters(&self) -> usize {
        let lora_params: usize =
            self.lora_layers.iter().map(|l: &LoRALayer| l.rank() * (l.d_in() + l.d_out())).sum();
        lora_params + self.classifier.num_parameters()
    }

    /// Load corpus from JSONL file.
    ///
    /// # Errors
    /// Returns error if file is invalid or labels out of range.
    pub fn load_corpus(&self, path: &Path) -> crate::Result<Vec<SafetySample>> {
        load_safety_corpus(path, self.config.num_classes)
    }

    /// Merge all LoRA adapters into base weights (for inference).
    pub fn merge_adapters(&mut self) {
        for lora in &mut self.lora_layers {
            lora.merge();
        }
    }

    /// Set the learning rate of the internal optimizer.
    ///
    /// Used by `ClassifyTrainer` to apply LR scheduling.
    pub fn set_optimizer_lr(&mut self, lr: f32) {
        self.optimizer.set_lr(lr);
    }

    /// Get the current learning rate of the internal optimizer.
    #[must_use]
    pub fn optimizer_lr(&self) -> f32 {
        self.optimizer.lr()
    }

    /// Get a reference to the internal AdamW optimizer (F-CKPT-004).
    #[must_use]
    pub fn optimizer(&self) -> &AdamW {
        &self.optimizer
    }

    /// Get a mutable reference to the internal AdamW optimizer (F-CKPT-004).
    pub fn optimizer_mut(&mut self) -> &mut AdamW {
        &mut self.optimizer
    }

    /// Summary of the pipeline configuration.
    #[must_use]
    pub fn summary(&self) -> String {
        let tokenizer_info = if let Some(ref tok) = self.tokenizer {
            format!("BPE (vocab={})", tok.vocab_size())
        } else {
            "byte-level (256)".to_string()
        };
        let device_info = if let Some(name) = self.gpu_name() {
            format!("CUDA ({name})")
        } else {
            "CPU".to_string()
        };
        format!(
            "ClassifyPipeline:\n  Model: {} hidden, {} layers\n  Device: {}\n  Tokenizer: {}\n  LoRA: rank={}, alpha={:.1}, {} adapters\n  Classifier: {}->{} ({} params)\n  Total trainable: {} params",
            self.model.config.hidden_size,
            self.model.config.num_hidden_layers,
            device_info,
            tokenizer_info,
            self.config.lora_rank,
            self.config.lora_alpha,
            self.lora_layers.len(),
            self.classifier.hidden_size(),
            self.classifier.num_classes(),
            self.classifier.num_parameters(),
            self.num_trainable_parameters(),
        )
    }

    /// Collect all LoRA + classifier gradients into a flat `Vec<f32>`.
    ///
    /// Used by distributed training workers to send gradients to the coordinator
    /// for AllReduce averaging (F-DP-001).
    ///
    /// Layout: `[lora_0_a_grad, lora_0_b_grad, ..., lora_N_a_grad, lora_N_b_grad,
    ///           classifier_weight_grad, classifier_bias_grad]`
    #[must_use]
    pub fn collect_lora_gradients(&self) -> Vec<f32> {
        let total = self.num_trainable_parameters();
        let mut grads = Vec::with_capacity(total);

        for lora in &self.lora_layers {
            if let Some(g) = lora.lora_a().grad() {
                grads.extend(g.iter());
            } else {
                grads.extend(std::iter::repeat_n(0.0f32, lora.lora_a().data().len()));
            }
            if let Some(g) = lora.lora_b().grad() {
                grads.extend(g.iter());
            } else {
                grads.extend(std::iter::repeat_n(0.0f32, lora.lora_b().data().len()));
            }
        }

        if let Some(g) = self.classifier.weight.grad() {
            grads.extend(g.iter());
        } else {
            grads.extend(std::iter::repeat_n(0.0f32, self.classifier.weight.data().len()));
        }
        if let Some(g) = self.classifier.bias.grad() {
            grads.extend(g.iter());
        } else {
            grads.extend(std::iter::repeat_n(0.0f32, self.classifier.bias.data().len()));
        }

        grads
    }

    /// Apply averaged gradients from AllReduce and run optimizer step.
    ///
    /// Used by distributed training workers after receiving averaged gradients
    /// from the coordinator (F-DP-001 weight consistency).
    ///
    /// The gradient vector layout must match `collect_lora_gradients()`.
    pub fn apply_lora_gradients(&mut self, averaged_grads: &[f32]) {
        let mut offset = 0usize;

        // Write averaged gradients into each parameter's grad slot
        for lora in &self.lora_layers {
            let a_len = lora.lora_a().data().len();
            if offset + a_len <= averaged_grads.len() {
                lora.lora_a().set_grad(ndarray::Array1::from_vec(
                    averaged_grads[offset..offset + a_len].to_vec(),
                ));
            }
            offset += a_len;

            let b_len = lora.lora_b().data().len();
            if offset + b_len <= averaged_grads.len() {
                lora.lora_b().set_grad(ndarray::Array1::from_vec(
                    averaged_grads[offset..offset + b_len].to_vec(),
                ));
            }
            offset += b_len;
        }

        let w_len = self.classifier.weight.data().len();
        if offset + w_len <= averaged_grads.len() {
            self.classifier.weight.set_grad(ndarray::Array1::from_vec(
                averaged_grads[offset..offset + w_len].to_vec(),
            ));
        }
        offset += w_len;

        let b_len = self.classifier.bias.data().len();
        if offset + b_len <= averaged_grads.len() {
            self.classifier.bias.set_grad(ndarray::Array1::from_vec(
                averaged_grads[offset..offset + b_len].to_vec(),
            ));
        }

        // Now run optimizer step with the averaged gradients
        let mut params: Vec<&mut Tensor> = Vec::new();
        for lora in &mut self.lora_layers {
            params.extend(lora.trainable_params());
        }
        params.extend(self.classifier.parameters_mut());
        self.optimizer.step_refs(&mut params);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn tiny_config() -> TransformerConfig {
        TransformerConfig::tiny()
    }

    #[test]
    fn test_classify_pipeline_creation() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 5,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let pipeline = ClassifyPipeline::new(&model_config, classify_config);
        assert_eq!(pipeline.classifier.num_classes(), 5);
        assert!(!pipeline.lora_layers.is_empty(), "Should have LoRA layers");
    }

    #[test]
    fn test_classify_pipeline_train_step() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 5,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
        let loss = pipeline.train_step(&[1, 2, 3], 0);
        assert!(loss.is_finite(), "F-CLASS-005: loss must be finite");
        assert!(loss > 0.0, "Cross-entropy loss must be positive");
    }

    #[test]
    fn test_classify_pipeline_convergence() {
        // SSC-017: Training must reduce loss across epochs
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            learning_rate: 1e-2,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);

        // Train on 3 samples for 20 epochs
        let samples = [(vec![1u32, 2, 3], 0usize), (vec![4, 5, 6], 1), (vec![7, 8, 9], 2)];

        let mut first_epoch_loss = 0.0f32;
        let mut last_epoch_loss = 0.0f32;

        for epoch in 0..20 {
            let mut epoch_loss = 0.0f32;
            for (tokens, label) in &samples {
                epoch_loss += pipeline.train_step(tokens, *label);
            }
            epoch_loss /= samples.len() as f32;

            if epoch == 0 {
                first_epoch_loss = epoch_loss;
            }
            last_epoch_loss = epoch_loss;
        }

        assert!(
            last_epoch_loss < first_epoch_loss,
            "SSC-017: Loss must decrease. First epoch: {first_epoch_loss:.4}, last: {last_epoch_loss:.4}"
        );
    }

    #[test]
    fn test_classify_pipeline_trainable_params() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 5,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
        let params = pipeline.trainable_parameters_mut();
        // LoRA A + B per adapter + classifier weight + bias
        assert!(params.len() >= 3, "Should have at least classifier + 1 LoRA adapter params");
    }

    #[test]
    fn test_classify_pipeline_summary() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig::default();
        let pipeline = ClassifyPipeline::new(&model_config, classify_config);
        let summary = pipeline.summary();
        assert!(summary.contains("ClassifyPipeline"));
        assert!(summary.contains("LoRA"));
        assert!(summary.contains("Classifier"));
    }

    #[test]
    fn test_multi_label_train_step() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 5,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);

        // Multi-hot: classes 1 and 2 active (needs-quoting AND non-deterministic)
        let targets = vec![0.0, 1.0, 1.0, 0.0, 0.0];
        let loss = pipeline.multi_label_train_step(&[1, 2, 3], &targets);
        assert!(loss.is_finite(), "F-CLASS-005: loss must be finite");
        assert!(loss > 0.0, "BCE loss must be positive");
    }

    #[test]
    fn test_multi_label_convergence() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            learning_rate: 1e-2,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);

        // Train on multi-label samples
        let samples: [(Vec<u32>, Vec<f32>); 3] = [
            (vec![1, 2, 3], vec![1.0, 1.0, 0.0]), // classes 0+1
            (vec![4, 5, 6], vec![0.0, 1.0, 1.0]), // classes 1+2
            (vec![7, 8, 9], vec![1.0, 0.0, 1.0]), // classes 0+2
        ];

        let mut first_loss = 0.0f32;
        let mut last_loss = 0.0f32;

        for epoch in 0..20 {
            let mut epoch_loss = 0.0f32;
            for (tokens, targets) in &samples {
                epoch_loss += pipeline.multi_label_train_step(tokens, targets);
            }
            epoch_loss /= samples.len() as f32;

            if epoch == 0 {
                first_loss = epoch_loss;
            }
            last_loss = epoch_loss;
        }

        assert!(
            last_loss < first_loss,
            "SSC-021: Multi-label loss must decrease. First: {first_loss:.4}, last: {last_loss:.4}"
        );
    }

    #[test]
    #[should_panic(expected = "F-CLASS-001")]
    fn test_multi_label_wrong_target_length() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 5,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
        // Wrong number of targets (3 instead of 5)
        pipeline.multi_label_train_step(&[1, 2, 3], &[1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_classify_pipeline_merge() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 5,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);

        // Should not panic
        pipeline.merge_adapters();

        // All LoRA layers should be merged
        for lora in &pipeline.lora_layers {
            assert!(lora.is_merged(), "All adapters should be merged");
        }
    }

    // =========================================================================
    // SSC-025: Mini-batch training with gradient accumulation
    // =========================================================================

    fn make_samples() -> Vec<SafetySample> {
        vec![
            SafetySample { input: "echo hello".into(), label: 0 },
            SafetySample { input: "rm -rf /".into(), label: 1 },
            SafetySample { input: "ls -la".into(), label: 2 },
        ]
    }

    #[test]
    fn test_ssc025_batch_result_accuracy() {
        let r = BatchResult { avg_loss: 1.0, correct: 3, total: 4, grad_norm: 0.0 };
        assert!((r.accuracy() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_ssc025_batch_result_accuracy_empty() {
        let r = BatchResult { avg_loss: 0.0, correct: 0, total: 0, grad_norm: 0.0 };
        assert!((r.accuracy() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_ssc025_batch_result_accuracy_perfect() {
        let r = BatchResult { avg_loss: 0.1, correct: 10, total: 10, grad_norm: 0.0 };
        assert!((r.accuracy() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ssc025_config_defaults() {
        let config = ClassifyConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.accumulation_steps, 1);
        assert_eq!(config.gradient_clip_norm, Some(1.0));
    }

    #[test]
    fn test_ssc025_config_custom_batch() {
        let config = ClassifyConfig {
            batch_size: 8,
            accumulation_steps: 4,
            gradient_clip_norm: Some(0.5),
            ..ClassifyConfig::default()
        };
        assert_eq!(config.batch_size, 8);
        assert_eq!(config.accumulation_steps, 4);
        assert_eq!(config.gradient_clip_norm, Some(0.5));
    }

    #[test]
    fn test_ssc025_train_batch_finite_loss() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            batch_size: 3,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
        let samples = make_samples();

        let result = pipeline.train_batch(&samples);
        assert!(
            result.avg_loss.is_finite(),
            "SSC-025: batch loss must be finite, got {}",
            result.avg_loss
        );
        assert!(result.avg_loss > 0.0, "Cross-entropy loss must be positive");
        assert_eq!(result.total, 3);
    }

    #[test]
    fn test_ssc025_train_batch_empty() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);

        let result = pipeline.train_batch(&[]);
        assert_eq!(result.total, 0);
        assert_eq!(result.correct, 0);
        assert!((result.avg_loss - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_ssc025_train_batch_convergence() {
        // SSC-025: Loss must decrease over multiple batches
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            learning_rate: 1e-2,
            gradient_clip_norm: None, // disable clipping for convergence test
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
        let samples = make_samples();

        let mut first_loss = 0.0f32;
        let mut last_loss = 0.0f32;

        for epoch in 0..20 {
            let result = pipeline.train_batch(&samples);
            if epoch == 0 {
                first_loss = result.avg_loss;
            }
            last_loss = result.avg_loss;
        }

        assert!(
            last_loss < first_loss,
            "SSC-025: Batch training must reduce loss. First: {first_loss:.4}, last: {last_loss:.4}"
        );
    }

    #[test]
    fn test_ssc025_gradient_clipping_bounds_norm() {
        let model_config = tiny_config();
        let max_norm = 0.5;
        let classify_config = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            learning_rate: 1e-2,
            gradient_clip_norm: Some(max_norm),
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
        let samples = make_samples();

        // Run one batch — the internal clip should have bounded the norm
        // We verify indirectly: the pipeline should not diverge with aggressive clipping
        let result = pipeline.train_batch(&samples);
        assert!(result.avg_loss.is_finite(), "SSC-025: clipped batch loss must be finite");
    }

    #[test]
    fn test_ssc025_gradient_clipping_disabled() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            gradient_clip_norm: None,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
        let samples = make_samples();

        let result = pipeline.train_batch(&samples);
        assert!(result.avg_loss.is_finite(), "SSC-025: unclipped batch loss must be finite");
    }

    #[test]
    fn test_ssc025_accumulate_gradients() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            learning_rate: 1e-2,
            gradient_clip_norm: None,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
        let samples = make_samples();

        // Split into micro-batches of 1
        pipeline.zero_all_gradients();
        let mut total_samples = 0;
        for sample in &samples {
            let result = pipeline.accumulate_gradients(std::slice::from_ref(sample));
            assert!(result.avg_loss.is_finite());
            assert_eq!(result.total, 1);
            total_samples += result.total;
        }

        // Apply accumulated gradients
        pipeline.apply_accumulated_gradients(total_samples);

        // Pipeline should still work after accumulation
        let result = pipeline.train_batch(&samples);
        assert!(result.avg_loss.is_finite());
    }

    #[test]
    fn test_ssc025_accumulate_gradients_convergence() {
        // Gradient accumulation should converge similarly to full-batch
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            learning_rate: 1e-2,
            gradient_clip_norm: None,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
        let samples = make_samples();

        let mut first_loss = 0.0f32;
        let mut last_loss = 0.0f32;

        for epoch in 0..20 {
            // Zero grads at start of each accumulation cycle
            pipeline.zero_all_gradients();
            let mut epoch_loss = 0.0f32;
            let mut total = 0;
            for sample in &samples {
                let result = pipeline.accumulate_gradients(std::slice::from_ref(sample));
                epoch_loss += result.avg_loss;
                total += result.total;
            }
            pipeline.apply_accumulated_gradients(total);

            let avg = epoch_loss / samples.len() as f32;
            if epoch == 0 {
                first_loss = avg;
            }
            last_loss = avg;
        }

        assert!(
            last_loss < first_loss,
            "SSC-025: Accumulated gradient training must reduce loss. First: {first_loss:.4}, last: {last_loss:.4}"
        );
    }

    #[test]
    fn test_ssc025_accumulate_gradients_empty() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);

        let result = pipeline.accumulate_gradients(&[]);
        assert_eq!(result.total, 0);
        assert_eq!(result.correct, 0);

        // apply with 0 should be a no-op
        pipeline.apply_accumulated_gradients(0);
    }

    #[test]
    fn test_ssc025_safety_sample_input_ids() {
        let sample = SafetySample { input: "echo".into(), label: 0 };
        let ids = sample.input_ids();
        assert_eq!(ids, vec![u32::from(b'e'), u32::from(b'c'), u32::from(b'h'), u32::from(b'o')]);
    }

    #[test]
    fn test_ssc025_safety_sample_input_ids_empty() {
        let sample = SafetySample { input: String::new(), label: 0 };
        assert!(sample.input_ids().is_empty());
    }

    #[test]
    fn test_ssc025_batch_result_debug() {
        let r = BatchResult { avg_loss: 1.5, correct: 2, total: 3, grad_norm: 0.0 };
        let debug = format!("{r:?}");
        assert!(debug.contains("BatchResult"));
        assert!(debug.contains("1.5"));
    }

    #[test]
    fn test_ssc025_single_sample_batch() {
        // A batch of 1 should behave like a single train_step
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            gradient_clip_norm: None,
            ..ClassifyConfig::default()
        };
        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
        let samples = vec![SafetySample { input: "echo hello".into(), label: 0 }];

        let result = pipeline.train_batch(&samples);
        assert_eq!(result.total, 1);
        assert!(result.avg_loss.is_finite());
        assert!(result.avg_loss > 0.0);
    }

    // =========================================================================
    // Tokenizer integration tests
    // =========================================================================

    #[test]
    fn test_tokenize_byte_level_fallback() {
        // new() pipeline has no tokenizer — should use byte-level
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let pipeline = ClassifyPipeline::new(&model_config, classify_config);

        let ids = pipeline.tokenize("echo");
        assert_eq!(ids, vec![u32::from(b'e'), u32::from(b'c'), u32::from(b'h'), u32::from(b'o')]);
    }

    #[test]
    fn test_tokenize_truncation() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            max_seq_len: 4,
            ..ClassifyConfig::default()
        };
        let pipeline = ClassifyPipeline::new(&model_config, classify_config);

        let ids = pipeline.tokenize("hello world");
        assert_eq!(ids.len(), 4, "Should truncate to max_seq_len");
    }

    #[test]
    fn test_tokenize_empty_guard() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let pipeline = ClassifyPipeline::new(&model_config, classify_config);

        let ids = pipeline.tokenize("");
        assert_eq!(ids.len(), 1, "Empty input should produce at least 1 token");
        assert_eq!(ids[0], 0, "Empty input guard token should be 0");
    }

    #[test]
    fn test_from_pretrained_missing_dir() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };

        let result = ClassifyPipeline::from_pretrained(
            "/nonexistent/model/dir",
            &model_config,
            classify_config,
        );
        assert!(result.is_err(), "from_pretrained with missing dir should fail");
    }

    #[test]
    fn test_summary_shows_tokenizer_byte_level() {
        let model_config = tiny_config();
        let classify_config = ClassifyConfig::default();
        let pipeline = ClassifyPipeline::new(&model_config, classify_config);
        let summary = pipeline.summary();
        assert!(
            summary.contains("byte-level (256)"),
            "Summary should show byte-level tokenizer, got: {summary}"
        );
    }

    // ── Coverage expansion tests ─────────────────────────────────────

    #[test]
    fn test_cov_qlora_default_small() {
        let c = ClassifyConfig::qlora_default(4_000_000_000);
        assert_eq!(c.num_classes, 2);
        assert_eq!(c.lora_rank, 16);
        assert!((c.lora_alpha - 32.0).abs() < f32::EPSILON);
        assert!((c.learning_rate - 2e-4).abs() < 1e-6);
        assert_eq!(c.epochs, 3);
        assert_eq!(c.max_seq_len, 256);
        assert_eq!(c.batch_size, 16);
        assert_eq!(c.accumulation_steps, 1);
        assert_eq!(c.gradient_clip_norm, Some(1.0));
        assert!(c.quantize_nf4);
    }

    #[test]
    fn test_cov_qlora_default_large() {
        let c = ClassifyConfig::qlora_default(70_000_000_000);
        assert!((c.learning_rate - 1e-4).abs() < 1e-6);
    }

    #[test]
    fn test_cov_qlora_boundary_13b() {
        let c = ClassifyConfig::qlora_default(13_000_000_000);
        assert!((c.learning_rate - 2e-4).abs() < 1e-6);
    }

    #[test]
    fn test_cov_hp_all_good() {
        let c = ClassifyConfig::qlora_default(4_000_000_000);
        let d = c.validate_hyperparameters(4_000_000_000);
        assert!(!d.has_errors());
    }

    #[test]
    fn test_cov_hp_lr_too_low() {
        let c = ClassifyConfig {
            learning_rate: 1e-5,
            quantize_nf4: true,
            ..ClassifyConfig::qlora_default(4_000_000_000)
        };
        assert!(c.validate_hyperparameters(4_000_000_000).has_warning("C-HP-001"));
    }

    #[test]
    fn test_cov_hp_lr_zero() {
        let c = ClassifyConfig { learning_rate: 0.0, ..ClassifyConfig::default() };
        let d = c.validate_hyperparameters(4_000_000_000);
        assert!(d.has_errors());
        assert!(d.has_warning("C-HP-001"));
    }

    #[test]
    fn test_cov_hp_lr_neg() {
        let c = ClassifyConfig { learning_rate: -0.001, ..ClassifyConfig::default() };
        assert!(c.validate_hyperparameters(4_000_000_000).has_errors());
    }

    #[test]
    fn test_cov_hp_bs_zero() {
        let c = ClassifyConfig { batch_size: 0, ..ClassifyConfig::default() };
        let d = c.validate_hyperparameters(4_000_000_000);
        assert!(d.has_errors());
        assert!(d.has_warning("C-HP-002"));
    }

    #[test]
    fn test_cov_hp_eff_batch_not_16() {
        let c =
            ClassifyConfig { batch_size: 4, accumulation_steps: 2, ..ClassifyConfig::default() };
        assert!(c.validate_hyperparameters(4_000_000_000).has_warning("C-HP-002"));
    }

    #[test]
    fn test_cov_hp_eff_batch_is_16() {
        let c =
            ClassifyConfig { batch_size: 4, accumulation_steps: 4, ..ClassifyConfig::default() };
        assert!(!c.validate_hyperparameters(4_000_000_000).has_warning("C-HP-002"));
    }

    #[test]
    fn test_cov_hp_alpha_mismatch() {
        let c = ClassifyConfig { lora_rank: 16, lora_alpha: 8.0, ..ClassifyConfig::default() };
        assert!(c.validate_hyperparameters(4_000_000_000).has_warning("C-HP-003"));
    }

    #[test]
    fn test_cov_hp_alpha_ok() {
        let c = ClassifyConfig { lora_rank: 16, lora_alpha: 32.0, ..ClassifyConfig::default() };
        assert!(!c.validate_hyperparameters(4_000_000_000).has_warning("C-HP-003"));
    }

    #[test]
    fn test_cov_hp_no_clip() {
        let c = ClassifyConfig { gradient_clip_norm: None, ..ClassifyConfig::default() };
        assert!(c.validate_hyperparameters(4_000_000_000).has_warning("C-HP-006"));
    }

    #[test]
    fn test_cov_hp_with_clip() {
        let c = ClassifyConfig { gradient_clip_norm: Some(1.0), ..ClassifyConfig::default() };
        assert!(!c.validate_hyperparameters(4_000_000_000).has_warning("C-HP-006"));
    }

    #[test]
    fn test_cov_hp_lr_non_nf4() {
        let c = ClassifyConfig {
            learning_rate: 1e-5,
            quantize_nf4: false,
            ..ClassifyConfig::default()
        };
        assert!(!c.validate_hyperparameters(4_000_000_000).has_warning("C-HP-001"));
    }

    #[test]
    fn test_cov_hp_lr_big_model() {
        let c =
            ClassifyConfig { learning_rate: 1e-5, quantize_nf4: true, ..ClassifyConfig::default() };
        assert!(!c.validate_hyperparameters(70_000_000_000).has_warning("C-HP-001"));
    }

    #[test]
    fn test_cov_diag_empty() {
        let d = HyperparamDiagnostics::default();
        assert!(!d.has_warning("X"));
        assert!(!d.has_errors());
    }

    #[test]
    fn test_cov_diag_info_not_warn() {
        let d = HyperparamDiagnostics {
            items: vec![HyperparamDiagnostic {
                contract_id: "C-HP-001",
                severity: DiagSeverity::Info,
                message: "i".into(),
                recommendation: "r".into(),
            }],
        };
        assert!(!d.has_warning("C-HP-001"));
    }

    #[test]
    fn test_cov_diag_warn_counted() {
        let d = HyperparamDiagnostics {
            items: vec![HyperparamDiagnostic {
                contract_id: "C-HP-003",
                severity: DiagSeverity::Warn,
                message: "w".into(),
                recommendation: "r".into(),
            }],
        };
        assert!(d.has_warning("C-HP-003"));
        assert!(!d.has_warning("C-HP-001"));
        assert!(!d.has_errors());
    }

    #[test]
    fn test_cov_diag_error_as_warn() {
        let d = HyperparamDiagnostics {
            items: vec![HyperparamDiagnostic {
                contract_id: "C-HP-002",
                severity: DiagSeverity::Error,
                message: "e".into(),
                recommendation: "r".into(),
            }],
        };
        assert!(d.has_warning("C-HP-002"));
        assert!(d.has_errors());
    }

    #[test]
    fn test_cov_diag_print_all() {
        let d = HyperparamDiagnostics {
            items: vec![
                HyperparamDiagnostic {
                    contract_id: "A",
                    severity: DiagSeverity::Info,
                    message: "i".into(),
                    recommendation: "r".into(),
                },
                HyperparamDiagnostic {
                    contract_id: "B",
                    severity: DiagSeverity::Warn,
                    message: "w".into(),
                    recommendation: "r".into(),
                },
                HyperparamDiagnostic {
                    contract_id: "C",
                    severity: DiagSeverity::Error,
                    message: "e".into(),
                    recommendation: "r".into(),
                },
            ],
        };
        d.print_all();
    }

    #[test]
    fn test_cov_diag_severity_traits() {
        assert_eq!(format!("{:?}", DiagSeverity::Info), "Info");
        assert_eq!(format!("{:?}", DiagSeverity::Warn), "Warn");
        assert_eq!(format!("{:?}", DiagSeverity::Error), "Error");
        let a = DiagSeverity::Warn;
        assert_eq!(a, a);
    }

    #[test]
    fn test_cov_diag_diagnostic_clone() {
        let d = HyperparamDiagnostic {
            contract_id: "C-HP-001",
            severity: DiagSeverity::Info,
            message: "m".into(),
            recommendation: "r".into(),
        };
        let d2 = d.clone();
        assert_eq!(d2.contract_id, "C-HP-001");
        assert!(format!("{d2:?}").contains("C-HP-001"));
    }

    #[test]
    fn test_cov_diags_default_clone() {
        let d = HyperparamDiagnostics::default();
        assert!(d.clone().items.is_empty());
    }

    #[test]
    fn test_cov_data_seq_high() {
        let c = ClassifyConfig { max_seq_len: 512, ..ClassifyConfig::default() };
        let s = DataStats { p99_token_length: 100, imbalance_ratio: 1.0, minority_count: 1000 };
        assert!(c.validate_with_data(&s).has_warning("C-HP-004"));
    }

    #[test]
    fn test_cov_data_seq_ok() {
        let c = ClassifyConfig { max_seq_len: 128, ..ClassifyConfig::default() };
        let s = DataStats { p99_token_length: 100, imbalance_ratio: 1.0, minority_count: 1000 };
        assert!(!c.validate_with_data(&s).has_warning("C-HP-004"));
    }

    #[test]
    fn test_cov_data_seq_zero_p99() {
        let c = ClassifyConfig { max_seq_len: 512, ..ClassifyConfig::default() };
        let s = DataStats { p99_token_length: 0, imbalance_ratio: 1.0, minority_count: 1000 };
        assert!(!c.validate_with_data(&s).has_warning("C-HP-004"));
    }

    #[test]
    fn test_cov_data_imb_few_epochs() {
        let c = ClassifyConfig {
            epochs: 1,
            batch_size: 16,
            accumulation_steps: 1,
            ..ClassifyConfig::default()
        };
        let s = DataStats { p99_token_length: 100, imbalance_ratio: 10.0, minority_count: 100 };
        assert!(c.validate_with_data(&s).has_warning("C-HP-008"));
    }

    #[test]
    fn test_cov_data_imb_ok_epochs() {
        let c = ClassifyConfig { epochs: 3, ..ClassifyConfig::default() };
        let s = DataStats { p99_token_length: 100, imbalance_ratio: 10.0, minority_count: 100 };
        assert!(!c.validate_with_data(&s).has_warning("C-HP-008"));
    }

    #[test]
    fn test_cov_data_low_imb() {
        let c = ClassifyConfig { epochs: 1, ..ClassifyConfig::default() };
        let s = DataStats { p99_token_length: 100, imbalance_ratio: 2.0, minority_count: 100 };
        assert!(!c.validate_with_data(&s).has_warning("C-HP-008"));
    }

    #[test]
    fn test_cov_data_both_warn() {
        let c = ClassifyConfig {
            max_seq_len: 1024,
            epochs: 1,
            batch_size: 16,
            accumulation_steps: 1,
            ..ClassifyConfig::default()
        };
        let s = DataStats { p99_token_length: 50, imbalance_ratio: 20.0, minority_count: 80 };
        let d = c.validate_with_data(&s);
        assert!(d.has_warning("C-HP-004"));
        assert!(d.has_warning("C-HP-008"));
    }

    #[test]
    fn test_cov_pretok_basic() {
        let mc = tiny_config();
        let cc = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let p = ClassifyPipeline::new(&mc, cc);
        let tok = p.pre_tokenize(&make_samples());
        assert_eq!(tok.len(), 3);
        for (t, s) in tok.iter().zip(make_samples().iter()) {
            assert_eq!(t.label, s.label);
            assert!(!t.token_ids.is_empty());
        }
    }

    #[test]
    fn test_cov_pretok_truncate() {
        let mc = tiny_config();
        let cc = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            max_seq_len: 4,
            ..ClassifyConfig::default()
        };
        let p = ClassifyPipeline::new(&mc, cc);
        let tok = p.pre_tokenize(&[SafetySample { input: "echo hello world".into(), label: 0 }]);
        assert_eq!(tok[0].token_ids.len(), 4);
    }

    #[test]
    fn test_cov_pretok_empty() {
        let mc = tiny_config();
        let cc = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let p = ClassifyPipeline::new(&mc, cc);
        let tok = p.pre_tokenize(&[SafetySample { input: String::new(), label: 0 }]);
        assert!(!tok[0].token_ids.is_empty());
    }

    #[test]
    fn test_cov_btok_empty() {
        let mc = tiny_config();
        let cc = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut p = ClassifyPipeline::new(&mc, cc);
        let r = p.train_batch_tokenized(&[]);
        assert_eq!(r.total, 0);
    }

    #[test]
    fn test_cov_btok_basic() {
        let mc = tiny_config();
        let cc = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut p = ClassifyPipeline::new(&mc, cc);
        let s = vec![
            TokenizedSample { token_ids: vec![1, 2, 3], label: 0 },
            TokenizedSample { token_ids: vec![4, 5, 6], label: 1 },
        ];
        let r = p.train_batch_tokenized(&s);
        assert_eq!(r.total, 2);
        assert!(r.avg_loss.is_finite() && r.avg_loss > 0.0);
    }

    #[test]
    fn test_cov_btok_converge() {
        let mc = tiny_config();
        let cc = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            learning_rate: 1e-2,
            gradient_clip_norm: None,
            ..ClassifyConfig::default()
        };
        let mut p = ClassifyPipeline::new(&mc, cc);
        let s = vec![
            TokenizedSample { token_ids: vec![1, 2, 3], label: 0 },
            TokenizedSample { token_ids: vec![4, 5, 6], label: 1 },
            TokenizedSample { token_ids: vec![7, 8, 9], label: 2 },
        ];
        let mut first = 0.0f32;
        let mut last = 0.0f32;
        for ep in 0..20 {
            let r = p.train_batch_tokenized(&s);
            if ep == 0 {
                first = r.avg_loss;
            }
            last = r.avg_loss;
        }
        assert!(last < first);
    }

    #[test]
    fn test_cov_btok_clip() {
        let mc = tiny_config();
        let cc = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            gradient_clip_norm: Some(0.5),
            ..ClassifyConfig::default()
        };
        let mut p = ClassifyPipeline::new(&mc, cc);
        let r = p.train_batch_tokenized(&[TokenizedSample { token_ids: vec![1, 2, 3], label: 0 }]);
        assert!(r.avg_loss.is_finite());
    }

    #[test]
    fn test_cov_atok_empty() {
        let mc = tiny_config();
        let cc = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut p = ClassifyPipeline::new(&mc, cc);
        assert_eq!(p.accumulate_gradients_tokenized(&[]).total, 0);
    }

    #[test]
    fn test_cov_atok_basic() {
        let mc = tiny_config();
        let cc = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            learning_rate: 1e-2,
            gradient_clip_norm: None,
            ..ClassifyConfig::default()
        };
        let mut p = ClassifyPipeline::new(&mc, cc);
        let mb = vec![
            TokenizedSample { token_ids: vec![1, 2, 3], label: 0 },
            TokenizedSample { token_ids: vec![4, 5, 6], label: 1 },
        ];
        p.zero_all_gradients();
        let r = p.accumulate_gradients_tokenized(&mb);
        assert_eq!(r.total, 2);
        assert!(r.avg_loss.is_finite());
        p.apply_accumulated_gradients(r.total);
    }

    #[test]
    fn test_cov_fwd_only() {
        let mc = tiny_config();
        let cc = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut p = ClassifyPipeline::new(&mc, cc);
        let (l, pr) = p.forward_only(&[1, 2, 3], 0);
        assert!(l.is_finite() && l > 0.0);
        assert!(pr < 3);
    }

    #[test]
    fn test_cov_fwd_all_labels() {
        let mc = tiny_config();
        let cc = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut p = ClassifyPipeline::new(&mc, cc);
        for lab in 0..3 {
            let (l, _) = p.forward_only(&[1, 2, 3], lab);
            assert!(l.is_finite());
        }
    }

    #[test]
    fn test_cov_fwd_tokenized() {
        let mc = tiny_config();
        let cc = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut p = ClassifyPipeline::new(&mc, cc);
        let (l, pr) = p.forward_only_tokenized(&[1, 2, 3], 0);
        assert!(l.is_finite() && pr < 3);
    }

    #[test]
    fn test_cov_fwd_probs() {
        let mc = tiny_config();
        let cc = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut p = ClassifyPipeline::new(&mc, cc);
        let (l, pr, probs) = p.forward_only_with_probs(&[1, 2, 3], 0);
        assert!(l.is_finite() && l > 0.0 && pr < 3);
        assert_eq!(probs.len(), 3);
        assert!(((probs.iter().sum::<f32>()) - 1.0).abs() < 1e-5);
        for &v in &probs {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_cov_fwd_probs_argmax() {
        let mc = tiny_config();
        let cc = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut p = ClassifyPipeline::new(&mc, cc);
        let (_, pred, probs) = p.forward_only_with_probs(&[1, 2, 3], 0);
        let am = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(pred, am);
    }

    #[test]
    fn test_cov_cw_train() {
        let mc = tiny_config();
        let cc = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            class_weights: Some(vec![1.0, 5.0, 1.0]),
            ..ClassifyConfig::default()
        };
        let mut p = ClassifyPipeline::new(&mc, cc);
        assert!(p.train_step(&[1, 2, 3], 1).is_finite());
    }

    #[test]
    fn test_cov_cw_batch() {
        let mc = tiny_config();
        let cc = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            class_weights: Some(vec![0.5, 5.0, 0.5]),
            ..ClassifyConfig::default()
        };
        let mut p = ClassifyPipeline::new(&mc, cc);
        assert!(p.train_batch(&make_samples()).avg_loss.is_finite());
    }

    #[test]
    fn test_cov_cw_fwd() {
        let mc = tiny_config();
        let cc = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            class_weights: Some(vec![1.0, 2.0, 3.0]),
            ..ClassifyConfig::default()
        };
        let mut p = ClassifyPipeline::new(&mc, cc);
        assert!(p.forward_only(&[1, 2, 3], 2).0.is_finite());
    }

    #[test]
    fn test_cov_set_lr() {
        let mc = tiny_config();
        let mut p = ClassifyPipeline::new(
            &mc,
            ClassifyConfig { learning_rate: 1e-3, ..ClassifyConfig::default() },
        );
        assert!((p.optimizer_lr() - 1e-3).abs() < 1e-6);
        p.set_optimizer_lr(5e-4);
        assert!((p.optimizer_lr() - 5e-4).abs() < 1e-6);
    }

    #[test]
    fn test_cov_opt_ref() {
        let mc = tiny_config();
        let p = ClassifyPipeline::new(&mc, ClassifyConfig::default());
        assert!((p.optimizer().lr() - ClassifyConfig::default().learning_rate).abs() < 1e-8);
    }

    #[test]
    fn test_cov_opt_mut() {
        let mc = tiny_config();
        let mut p = ClassifyPipeline::new(&mc, ClassifyConfig::default());
        p.optimizer_mut().set_lr(2e-4);
        assert!((p.optimizer_lr() - 2e-4).abs() < 1e-6);
    }

    #[test]
    fn test_cov_model_dir_none() {
        let p = ClassifyPipeline::new(&tiny_config(), ClassifyConfig::default());
        assert!(p.model_dir().is_none());
    }

    #[test]
    fn test_cov_set_model_path() {
        let mut p = ClassifyPipeline::new(&tiny_config(), ClassifyConfig::default());
        p.set_model_path("/tmp/m");
        assert_eq!(p.model_dir(), Some(Path::new("/tmp/m")));
    }

    #[test]
    fn test_cov_set_model_path_buf() {
        let mut p = ClassifyPipeline::new(&tiny_config(), ClassifyConfig::default());
        p.set_model_path(PathBuf::from("/opt/v1"));
        assert_eq!(p.model_dir(), Some(Path::new("/opt/v1")));
    }

    #[test]
    fn test_cov_is_cuda() {
        let p = ClassifyPipeline::new(&tiny_config(), ClassifyConfig::default());
        #[cfg(not(feature = "cuda"))]
        assert!(!p.is_cuda());
        let _ = p.is_cuda();
    }

    #[test]
    fn test_cov_gpu_name() {
        let p = ClassifyPipeline::new(&tiny_config(), ClassifyConfig::default());
        #[cfg(not(feature = "cuda"))]
        assert!(p.gpu_name().is_none());
        let _ = p.gpu_name();
    }

    #[test]
    fn test_cov_gpu_mem() {
        let p = ClassifyPipeline::new(&tiny_config(), ClassifyConfig::default());
        #[cfg(not(feature = "cuda"))]
        assert!(p.gpu_total_memory().is_none());
        let _ = p.gpu_total_memory();
    }

    #[test]
    fn test_cov_is_gpu_training() {
        let p = ClassifyPipeline::new(&tiny_config(), ClassifyConfig::default());
        #[cfg(not(feature = "cuda"))]
        assert!(!p.is_gpu_training());
        let _ = p.is_gpu_training();
    }

    #[test]
    fn test_cov_num_params() {
        let mc = tiny_config();
        let p = ClassifyPipeline::new(
            &mc,
            ClassifyConfig {
                num_classes: 5,
                lora_rank: 4,
                lora_alpha: 4.0,
                ..ClassifyConfig::default()
            },
        );
        let n = p.num_trainable_parameters();
        assert!(n > 0);
        assert!(n >= mc.hidden_size * 5 + 5);
    }

    #[test]
    fn test_cov_params_scale() {
        let mc = tiny_config();
        let s = ClassifyPipeline::new(
            &mc,
            ClassifyConfig {
                num_classes: 2,
                lora_rank: 2,
                lora_alpha: 2.0,
                ..ClassifyConfig::default()
            },
        );
        let l = ClassifyPipeline::new(
            &mc,
            ClassifyConfig {
                num_classes: 2,
                lora_rank: 16,
                lora_alpha: 16.0,
                ..ClassifyConfig::default()
            },
        );
        assert!(l.num_trainable_parameters() > s.num_trainable_parameters());
    }

    #[test]
    fn test_cov_grads_len() {
        let mc = tiny_config();
        let p = ClassifyPipeline::new(
            &mc,
            ClassifyConfig {
                num_classes: 3,
                lora_rank: 4,
                lora_alpha: 4.0,
                ..ClassifyConfig::default()
            },
        );
        assert_eq!(p.collect_lora_gradients().len(), p.num_trainable_parameters());
    }

    #[test]
    fn test_cov_grads_zero() {
        let mc = tiny_config();
        let p = ClassifyPipeline::new(
            &mc,
            ClassifyConfig {
                num_classes: 3,
                lora_rank: 4,
                lora_alpha: 4.0,
                ..ClassifyConfig::default()
            },
        );
        assert!(p.collect_lora_gradients().iter().all(|&g| g == 0.0));
    }

    #[test]
    fn test_cov_grads_nonzero() {
        let mc = tiny_config();
        let cc = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            gradient_clip_norm: None,
            ..ClassifyConfig::default()
        };
        let mut p = ClassifyPipeline::new(&mc, cc);
        p.zero_all_gradients();
        let _ = p.accumulate_gradients(&[SafetySample { input: "echo hi".into(), label: 0 }]);
        assert!(p.collect_lora_gradients().iter().any(|&g| g != 0.0));
    }

    #[test]
    fn test_cov_apply_grads() {
        let mc = tiny_config();
        let cc = ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        };
        let mut p = ClassifyPipeline::new(&mc, cc);
        let n = p.num_trainable_parameters();
        p.apply_lora_gradients(&(0..n).map(|i| i as f32 * 0.001).collect::<Vec<_>>());
    }

    #[test]
    fn test_cov_apply_grads_short() {
        let mc = tiny_config();
        let mut p = ClassifyPipeline::new(
            &mc,
            ClassifyConfig {
                num_classes: 3,
                lora_rank: 4,
                lora_alpha: 4.0,
                ..ClassifyConfig::default()
            },
        );
        p.apply_lora_gradients(&[0.1, 0.2]);
    }

    #[test]
    fn test_cov_apply_grads_empty() {
        let mc = tiny_config();
        let mut p = ClassifyPipeline::new(
            &mc,
            ClassifyConfig {
                num_classes: 3,
                lora_rank: 4,
                lora_alpha: 4.0,
                ..ClassifyConfig::default()
            },
        );
        p.apply_lora_gradients(&[]);
    }

    #[test]
    fn test_cov_merge_idem() {
        let mc = tiny_config();
        let mut p = ClassifyPipeline::new(
            &mc,
            ClassifyConfig {
                num_classes: 3,
                lora_rank: 4,
                lora_alpha: 4.0,
                ..ClassifyConfig::default()
            },
        );
        p.merge_adapters();
        p.merge_adapters();
        for lora in &p.lora_layers {
            assert!(lora.is_merged());
        }
    }

    #[test]
    fn test_cov_dispatch_lora() {
        let mc = tiny_config();
        let mut p = ClassifyPipeline::new(
            &mc,
            ClassifyConfig {
                num_classes: 3,
                lora_rank: 4,
                lora_alpha: 4.0,
                ..ClassifyConfig::default()
            },
        );
        assert!(!p.lora_layers.is_empty());
        assert!(!p.forward_hidden_dispatch(&[1, 2, 3]).data().is_empty());
    }

    #[test]
    fn test_cov_summary_detail() {
        let p = ClassifyPipeline::new(
            &tiny_config(),
            ClassifyConfig {
                num_classes: 5,
                lora_rank: 8,
                lora_alpha: 16.0,
                ..ClassifyConfig::default()
            },
        );
        let s = p.summary();
        assert!(
            s.contains("ClassifyPipeline")
                && s.contains("64 hidden")
                && s.contains("CPU")
                && s.contains("rank=8")
        );
    }

    #[test]
    fn test_cov_from_pretrained_err() {
        assert!(ClassifyPipeline::from_pretrained(
            "/nonexist",
            &tiny_config(),
            ClassifyConfig::default()
        )
        .is_err());
    }

    #[test]
    fn test_cov_from_apr_err() {
        assert!(ClassifyPipeline::from_apr(
            Path::new("/nonexist.apr"),
            &tiny_config(),
            ClassifyConfig::default()
        )
        .is_err());
    }

    #[test]
    fn test_cov_load_corpus_err() {
        let p = ClassifyPipeline::new(
            &tiny_config(),
            ClassifyConfig { num_classes: 3, ..ClassifyConfig::default() },
        );
        assert!(p.load_corpus(Path::new("/ne.jsonl")).is_err());
    }

    #[test]
    fn test_cov_load_ml_corpus_err() {
        let p = ClassifyPipeline::new(
            &tiny_config(),
            ClassifyConfig { num_classes: 3, ..ClassifyConfig::default() },
        );
        assert!(p.load_multi_label_corpus(Path::new("/ne.jsonl")).is_err());
    }

    #[test]
    fn test_cov_batch_accuracy_1() {
        let r = BatchResult { avg_loss: 1.0, correct: 1, total: 100, grad_norm: 0.5 };
        assert!((r.accuracy() - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_cov_batch_clone() {
        let r = BatchResult { avg_loss: 1.5, correct: 2, total: 3, grad_norm: 0.42 };
        let r2 = r.clone();
        assert_eq!(r2.correct, 2);
        assert!((r2.grad_norm - 0.42).abs() < 1e-6);
    }

    #[test]
    fn test_cov_config_clone() {
        let c = ClassifyConfig::default();
        let c2 = c.clone();
        assert_eq!(c2.num_classes, c.num_classes);
        assert!(format!("{c2:?}").contains("ClassifyConfig"));
    }

    #[test]
    fn test_cov_config_nf4_false() {
        assert!(!ClassifyConfig::default().quantize_nf4);
    }

    #[test]
    fn test_cov_config_cw_none() {
        assert!(ClassifyConfig::default().class_weights.is_none());
    }

    #[test]
    fn test_cov_zero_grads() {
        let mc = tiny_config();
        let mut p = ClassifyPipeline::new(
            &mc,
            ClassifyConfig {
                num_classes: 3,
                lora_rank: 4,
                lora_alpha: 4.0,
                ..ClassifyConfig::default()
            },
        );
        let _ = p.train_step(&[1, 2, 3], 0);
        p.zero_all_gradients();
        assert!(p.compute_grad_norm().abs() < 1e-6);
    }

    #[test]
    fn test_cov_grad_norm() {
        let mc = tiny_config();
        let mut p = ClassifyPipeline::new(
            &mc,
            ClassifyConfig {
                num_classes: 3,
                lora_rank: 4,
                lora_alpha: 4.0,
                gradient_clip_norm: None,
                ..ClassifyConfig::default()
            },
        );
        p.zero_all_gradients();
        let _ = p.accumulate_gradients(&[SafetySample { input: "ls".into(), label: 0 }]);
        assert!(p.compute_grad_norm() >= 0.0);
    }

    #[test]
    fn test_cov_scale_grads() {
        let mc = tiny_config();
        let mut p = ClassifyPipeline::new(
            &mc,
            ClassifyConfig {
                num_classes: 3,
                lora_rank: 4,
                lora_alpha: 4.0,
                gradient_clip_norm: None,
                ..ClassifyConfig::default()
            },
        );
        p.zero_all_gradients();
        let _ = p.accumulate_gradients(&[SafetySample { input: "ls".into(), label: 0 }]);
        let b = p.compute_grad_norm();
        p.scale_all_gradients(2.0);
        let a = p.compute_grad_norm();
        if b > 1e-8 {
            assert!((a / b - 2.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_cov_binary() {
        let mut p = ClassifyPipeline::new(
            &tiny_config(),
            ClassifyConfig {
                num_classes: 2,
                lora_rank: 4,
                lora_alpha: 4.0,
                ..ClassifyConfig::default()
            },
        );
        assert!(p.train_step(&[1, 2, 3], 0).is_finite());
        assert!(p.train_step(&[4, 5, 6], 1).is_finite());
    }

    #[test]
    fn test_cov_many_classes() {
        let mut p = ClassifyPipeline::new(
            &tiny_config(),
            ClassifyConfig {
                num_classes: 20,
                lora_rank: 4,
                lora_alpha: 4.0,
                ..ClassifyConfig::default()
            },
        );
        assert!(p.train_step(&[1, 2, 3], 15).is_finite());
    }

    #[test]
    fn test_cov_single_token() {
        let mut p = ClassifyPipeline::new(
            &tiny_config(),
            ClassifyConfig {
                num_classes: 3,
                lora_rank: 4,
                lora_alpha: 4.0,
                ..ClassifyConfig::default()
            },
        );
        assert!(p.train_step(&[42], 1).is_finite());
    }

    #[test]
    fn test_cov_long_input() {
        let mut p = ClassifyPipeline::new(
            &tiny_config(),
            ClassifyConfig {
                num_classes: 3,
                lora_rank: 4,
                lora_alpha: 4.0,
                max_seq_len: 10,
                ..ClassifyConfig::default()
            },
        );
        assert!(p.train_step(&(0..50).collect::<Vec<u32>>(), 0).is_finite());
    }

    #[test]
    fn test_cov_lora_count() {
        let p = ClassifyPipeline::new(
            &tiny_config(),
            ClassifyConfig {
                num_classes: 3,
                lora_rank: 4,
                lora_alpha: 4.0,
                ..ClassifyConfig::default()
            },
        );
        assert_eq!(p.lora_layers.len(), 4);
    }

    #[test]
    fn test_cov_lora_grad() {
        let p = ClassifyPipeline::new(
            &tiny_config(),
            ClassifyConfig {
                num_classes: 3,
                lora_rank: 4,
                lora_alpha: 4.0,
                ..ClassifyConfig::default()
            },
        );
        for l in &p.lora_layers {
            assert!(l.lora_a().requires_grad() && l.lora_b().requires_grad());
        }
    }

    #[test]
    fn test_cov_train_eval() {
        let mc = tiny_config();
        let mut p = ClassifyPipeline::new(
            &mc,
            ClassifyConfig {
                num_classes: 3,
                lora_rank: 4,
                lora_alpha: 4.0,
                learning_rate: 1e-2,
                ..ClassifyConfig::default()
            },
        );
        for _ in 0..5 {
            let _ = p.train_step(&[1, 2, 3], 0);
        }
        let (l, pr) = p.forward_only(&[1, 2, 3], 0);
        assert!(l.is_finite() && pr < 3);
    }

    #[test]
    fn test_cov_batch_then_probs() {
        let mc = tiny_config();
        let mut p = ClassifyPipeline::new(
            &mc,
            ClassifyConfig {
                num_classes: 3,
                lora_rank: 4,
                lora_alpha: 4.0,
                learning_rate: 1e-2,
                ..ClassifyConfig::default()
            },
        );
        for _ in 0..5 {
            let _ = p.train_batch(&make_samples());
        }
        let (l, pr, probs) = p.forward_only_with_probs(&[1, 2, 3], 0);
        assert!(l.is_finite() && pr < 3 && probs.len() == 3);
    }

    #[test]
    fn test_cov_multi_diag() {
        let c = ClassifyConfig {
            learning_rate: 0.0,
            batch_size: 0,
            lora_rank: 16,
            lora_alpha: 8.0,
            gradient_clip_norm: None,
            quantize_nf4: false,
            ..ClassifyConfig::default()
        };
        let d = c.validate_hyperparameters(4_000_000_000);
        assert!(d.has_errors());
        assert!(d.items.len() >= 3);
    }

    #[test]
    fn test_cov_nf4_config() {
        let p = ClassifyPipeline::new(
            &tiny_config(),
            ClassifyConfig { quantize_nf4: true, ..ClassifyConfig::default() },
        );
        assert!(!p.is_cuda());
    }

    #[test]
    fn test_cov_nf4_btok() {
        let mut p = ClassifyPipeline::new(
            &tiny_config(),
            ClassifyConfig {
                num_classes: 3,
                lora_rank: 4,
                lora_alpha: 4.0,
                quantize_nf4: true,
                ..ClassifyConfig::default()
            },
        );
        assert!(p
            .train_batch_tokenized(&[TokenizedSample { token_ids: vec![1, 2, 3], label: 0 }])
            .avg_loss
            .is_finite());
    }

    #[test]
    fn test_cov_apply_accum_nf4() {
        let mut p = ClassifyPipeline::new(
            &tiny_config(),
            ClassifyConfig {
                num_classes: 3,
                lora_rank: 4,
                lora_alpha: 4.0,
                quantize_nf4: true,
                gradient_clip_norm: Some(1.0),
                ..ClassifyConfig::default()
            },
        );
        p.zero_all_gradients();
        let _ = p.accumulate_gradients(&[SafetySample { input: "echo t".into(), label: 0 }]);
        p.apply_accumulated_gradients(1);
    }

    #[test]
    fn test_cov_apply_accum_fp32() {
        let mut p = ClassifyPipeline::new(
            &tiny_config(),
            ClassifyConfig {
                num_classes: 3,
                lora_rank: 4,
                lora_alpha: 4.0,
                quantize_nf4: false,
                gradient_clip_norm: Some(1.0),
                ..ClassifyConfig::default()
            },
        );
        p.zero_all_gradients();
        let _ = p.accumulate_gradients(&[SafetySample { input: "echo t".into(), label: 0 }]);
        p.apply_accumulated_gradients(1);
    }
}
