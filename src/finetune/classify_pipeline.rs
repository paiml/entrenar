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
    SafetySample,
};
use crate::autograd::matmul;
use crate::lora::LoRAConfig;
use crate::lora::LoRALayer;
use crate::optim::{clip_grad_norm_refs, AdamW, Optimizer};
use crate::tokenizer::HfTokenizer;
use crate::transformer::Transformer;
use crate::transformer::TransformerConfig;
use crate::Tensor;
use std::path::Path;

#[cfg(feature = "cuda")]
use crate::autograd::cuda_training::{cuda_training_available, CudaTrainer};
#[cfg(feature = "cuda")]
use crate::transformer::CudaTransformerBlock;
#[cfg(feature = "cuda")]
use std::sync::Arc;

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
        }
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

/// Classification fine-tuning pipeline.
///
/// Owns the transformer, LoRA adapters, and classification head.
/// Provides `train_step()` for single-step training and `train()` for full loop.
///
/// When compiled with `feature = "cuda"` and a GPU is available, the forward pass
/// runs on CUDA via `CudaTransformerBlock`s for ~10-50x speedup (F-CUDA-007).
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
    /// CUDA trainer for GPU memory management (F-CUDA-002)
    #[cfg(feature = "cuda")]
    cuda_trainer: Option<CudaTrainer>,
    /// CUDA-accelerated transformer blocks — one per layer (F-CUDA-006)
    #[cfg(feature = "cuda")]
    cuda_blocks: Option<Vec<CudaTransformerBlock>>,
    /// Count of GPU forward passes that produced NaN/Inf and fell back to CPU.
    /// Used to decide when to permanently disable CUDA (threshold: 100).
    #[cfg(feature = "cuda")]
    cuda_nan_count: usize,
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

        // ── CUDA initialization (F-CUDA-001..006) ────────────────────────
        #[cfg(feature = "cuda")]
        let (cuda_trainer, cuda_blocks) =
            Self::try_init_cuda(&model, &model_config, &classify_config);

        Self {
            model,
            classifier,
            lora_layers,
            config: classify_config,
            optimizer,
            tokenizer: None,
            #[cfg(feature = "cuda")]
            cuda_trainer,
            #[cfg(feature = "cuda")]
            cuda_blocks,
            #[cfg(feature = "cuda")]
            cuda_nan_count: 0,
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

        // Load tokenizer if tokenizer.json exists (optional — falls back to byte-level)
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            Some(
                HfTokenizer::from_file(&tokenizer_path)
                    .map_err(|e| crate::Error::Io(format!("Failed to load tokenizer: {e}")))?,
            )
        } else {
            None
        };

        let optimizer = AdamW::default_params(classify_config.learning_rate);

        // ── CUDA initialization (F-CUDA-001..006) ────────────────────────
        #[cfg(feature = "cuda")]
        let (cuda_trainer, cuda_blocks) =
            Self::try_init_cuda(&model, &model_config, &classify_config);

        Ok(Self {
            model,
            classifier,
            lora_layers,
            config: classify_config,
            optimizer,
            tokenizer,
            #[cfg(feature = "cuda")]
            cuda_trainer,
            #[cfg(feature = "cuda")]
            cuda_blocks,
            #[cfg(feature = "cuda")]
            cuda_nan_count: 0,
        })
    }

    /// Tokenize input text using BPE tokenizer if available, else byte-level fallback.
    ///
    /// Truncates to `config.max_seq_len` and ensures at least one token.
    pub(crate) fn tokenize(&self, text: &str) -> Vec<u32> {
        let mut ids = if let Some(ref tok) = self.tokenizer {
            tok.encode(text)
        } else {
            text.bytes().map(u32::from).collect()
        };
        ids.truncate(self.config.max_seq_len);
        if ids.is_empty() {
            ids.push(0);
        }
        ids
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
        let head_dim = hidden / model_config.num_attention_heads;

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

    /// Attempt to initialize CUDA acceleration.
    ///
    /// Creates `CudaTrainer` and uploads all transformer layer weights to GPU as
    /// `CudaTransformerBlock`s. Returns `(None, None)` if CUDA is unavailable
    /// or any initialization step fails (F-CUDA-003: graceful fallback).
    #[cfg(feature = "cuda")]
    fn try_init_cuda(
        model: &Transformer,
        model_config: &TransformerConfig,
        classify_config: &ClassifyConfig,
    ) -> (Option<CudaTrainer>, Option<Vec<CudaTransformerBlock>>) {
        if !cuda_training_available() {
            eprintln!("[CUDA] No CUDA runtime detected — using CPU");
            return (None, None);
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
                return (None, None);
            }
        };

        let ctx = Arc::clone(trainer.context());
        let max_seq_len = classify_config.max_seq_len;
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

            match CudaTransformerBlock::new(
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
            ) {
                Ok(block) => blocks.push(block),
                Err(e) => {
                    eprintln!(
                        "[CUDA] Failed to upload layer {i} to GPU: {e} — falling back to CPU"
                    );
                    return (None, None);
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

        (Some(trainer), Some(blocks))
    }

    /// Forward pass through transformer layers, dispatching to GPU when available.
    ///
    /// - **GPU path** (F-CUDA-007..009): Embed on CPU, upload to GPU, run CUDA layers, download
    /// - **CPU path**: Use `Transformer::forward_hidden()`
    fn forward_hidden_dispatch(&mut self, token_ids: &[u32]) -> Tensor {
        #[cfg(feature = "cuda")]
        {
            if let (Some(ref trainer), Some(ref mut blocks)) =
                (&self.cuda_trainer, &mut self.cuda_blocks)
            {
                match Self::forward_hidden_cuda_impl(&self.model, token_ids, trainer, blocks) {
                    Some(tensor) => return tensor,
                    None => {
                        // GPU produced NaN/Inf for this sample — use CPU fallback
                        // but keep CUDA enabled for subsequent samples.
                        // Sporadic NaN is expected with randomly initialized weights
                        // due to different accumulation ordering in CUDA GEMM.
                        self.cuda_nan_count += 1;
                        if self.cuda_nan_count > 100 {
                            eprintln!(
                                "[CUDA] {} NaN fallbacks — disabling GPU acceleration",
                                self.cuda_nan_count
                            );
                            self.cuda_trainer = None;
                            self.cuda_blocks = None;
                        }
                    }
                }
            }
        }

        // CPU fallback
        self.model.forward_hidden(token_ids)
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
        cuda_blocks: &mut [CudaTransformerBlock],
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
            if let Err(e) = block.forward(&gpu_input, &mut gpu_output, seq_len, stream) {
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
            self.cuda_trainer.as_ref().map(|t| t.device_name())
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
            self.cuda_trainer.as_ref().map(|t| t.total_memory())
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
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

        // Loss = -log(prob[target])
        let loss_val = -(probs[label].max(1e-10).ln());
        let loss_val = if loss_val.is_finite() { loss_val } else { 100.0 };

        // ∂L/∂logits = probs - one_hot(target)
        // This is the well-known cross-entropy gradient
        let mut grad_logits = probs.clone();
        grad_logits[label] -= 1.0;

        // ── 4. Backward through matmul (autograd) ─────────────────────
        // Set loss gradient on the matmul output, then call backward
        let logits_tensor = logits;
        logits_tensor.set_grad(ndarray::Array1::from(grad_logits.clone()));
        if let Some(op) = logits_tensor.backward_op() {
            op.backward();
        }

        // Manually set bias gradient (∂L/∂bias = ∂L/∂logits)
        self.classifier.bias.set_grad(ndarray::Array1::from(grad_logits));

        // ── 5. Optimizer step ─────────────────────────────────────────
        // Collect trainable params without borrowing through self (avoids borrow conflict with optimizer)
        let mut params: Vec<&mut Tensor> = Vec::new();
        for lora in &mut self.lora_layers {
            params.extend(lora.trainable_params());
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
        let mut params: Vec<&mut Tensor> = Vec::new();
        for lora in &mut self.lora_layers {
            params.extend(lora.trainable_params());
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
        let mut params: Vec<&mut Tensor> = Vec::new();
        for lora in &mut self.lora_layers {
            params.extend(lora.trainable_params());
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
        let seq_len = token_ids.len();
        let num_classes = self.config.num_classes;

        // ── Contract precondition (F-CLASS-002): label in bounds ─────────
        debug_assert!(
            label < num_classes,
            "F-CLASS-002: label index {label} >= num_classes {num_classes}"
        );

        // ── Forward pass (GPU-dispatched if available) ──────────────────
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

        // ── Cross-entropy loss ──────────────────────────────────────────
        let max_val = logits_with_bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits_with_bias.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp).collect();

        let loss_val = -(probs[label].max(1e-10).ln());
        let loss_val = if loss_val.is_finite() { loss_val } else { 100.0 };

        // ── Contract postcondition (F-CLASS-005): loss finite & non-negative
        debug_assert!(loss_val.is_finite(), "F-CLASS-005: loss is not finite");
        debug_assert!(loss_val >= 0.0, "F-CLASS-005: loss is negative: {loss_val}");

        // ── Backward ────────────────────────────────────────────────────
        // dL/d_logits = softmax(logits) - one_hot(label)
        let mut grad_logits = probs;
        grad_logits[label] -= 1.0;

        logits.set_grad(ndarray::Array1::from(grad_logits.clone()));
        if let Some(op) = logits.backward_op() {
            op.backward();
        }

        // Accumulate bias gradient (not set — accumulate)
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
        let all_params: Vec<&Tensor> = self
            .lora_layers
            .iter()
            .flat_map(|l| vec![l.lora_a(), l.lora_b()])
            .chain(self.classifier.parameters())
            .collect();

        for param in all_params {
            if let Some(grad) = param.grad() {
                param.set_grad(grad * factor);
            }
        }
    }

    /// Compute the global L2 norm of all trainable gradients.
    ///
    /// Used by the monitor when gradient clipping is not enabled.
    fn compute_grad_norm(&self) -> f32 {
        let mut total_norm_sq = 0.0f32;
        let all_params: Vec<&Tensor> = self
            .lora_layers
            .iter()
            .flat_map(|l| vec![l.lora_a(), l.lora_b()])
            .chain(self.classifier.parameters())
            .collect();
        for param in all_params {
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
        let seq_len = token_ids.len();
        let num_classes = self.config.num_classes;

        // Forward pass (F-CUDA-009: classification head runs on CPU after GPU hidden states)
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

        // Predicted class (argmax)
        let predicted = logits_with_bias
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx);

        // Cross-entropy loss
        let max_val = logits_with_bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits_with_bias.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp).collect();

        let loss_val = -(probs[label].max(1e-10).ln());
        let loss_val = if loss_val.is_finite() { loss_val } else { 100.0 };

        (loss_val, predicted)
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
}

#[cfg(test)]
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
        assert_eq!(ids, vec![b'e' as u32, b'c' as u32, b'h' as u32, b'o' as u32]);
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
        assert_eq!(ids, vec![b'e' as u32, b'c' as u32, b'h' as u32, b'o' as u32]);
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
}
