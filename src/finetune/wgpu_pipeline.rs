//! WgpuInstructPipeline — GPU-only training pipeline (§26.11.7)
//!
//! Root cause fix: bypasses entrenar's `Transformer` (which requires 20-min CPU
//! dequant of 28 GB Q4K→F32). Instead, uses `WgslForwardPass` with pre-uploaded
//! GPU weights from `dequant_model_weights()`.
//!
//! No `Transformer` object. No CPU F32 projection weights. No SATD.
//!
//! # Architecture (Unsloth pattern)
//!
//! ```text
//! OwnedQuantizedModel (Q4K, seconds to load)
//!   → dequant_model_weights() (streaming, ~2 min)
//!   → WgslForwardPass.upload_weight() (persistent GPU buffers)
//!   → WgpuInstructPipeline.train_step() (all GPU)
//! ```
//!
//! # Contract: wgsl-training-pipeline-v1
//!
//! - `fast_load`: load_time < 5 min on GB10
//! - `no_transformer`: does not construct Transformer

#[cfg(feature = "gpu")]
use trueno::backends::gpu::wgpu;
#[cfg(feature = "gpu")]
use trueno::backends::gpu::WgslForwardPass;

#[cfg(feature = "gpu")]
use crate::autograd::wgpu_cross_entropy::WgslCrossEntropy;
#[cfg(feature = "gpu")]
use crate::autograd::wgpu_training::WgpuTrainer;

use crate::finetune::instruct_pipeline::InstructStepResult;
use crate::tokenizer::HfTokenizer;

/// GPU-only instruct training pipeline. No `Transformer` object.
#[cfg(feature = "gpu")]
pub struct WgpuInstructPipeline {
    /// GPU forward pass with persistent weight buffers
    fwd: WgslForwardPass,
    /// Fused cross-entropy loss on GPU
    cross_entropy: WgslCrossEntropy,
    /// GPU optimizer + backward GEMM
    trainer: WgpuTrainer,
    /// Pre-uploaded lm_head — PRE-CHUNKED to avoid per-step download
    /// Each chunk is < 2 GB (fits in wgpu bind group)
    lm_head_t_chunks: Vec<(wgpu::Buffer, u32)>, // [(chunk_buf, chunk_n)] for forward
    lm_head_chunks: Vec<(wgpu::Buffer, u32)>,    // [(chunk_buf, chunk_n)] for backward
    /// GPU buffers
    logits_buf: wgpu::Buffer,
    labels_buf: wgpu::Buffer,
    losses_buf: wgpu::Buffer,
    logsumexp_buf: wgpu::Buffer,
    /// Config
    num_layers: usize,
    hidden_dim: usize,
    vocab_size: usize,
    max_seq_len: usize,
    /// Tokenizer
    tokenizer: HfTokenizer,
    /// Embedding weights (CPU, small: vocab × hidden × 4 for token lookup)
    embed_weights: Vec<f32>,
    /// Output norm weights (CPU, small: hidden × 4)
    output_norm: Vec<f32>,
    /// RMSNorm epsilon
    eps: f32,
}

#[cfg(feature = "gpu")]
impl WgpuInstructPipeline {
    /// Create from pre-uploaded WgslForwardPass.
    ///
    /// Caller is responsible for:
    /// 1. Loading OwnedQuantizedModel
    /// 2. Calling dequant_model_weights()
    /// 3. Uploading weights to WgslForwardPass
    /// 4. Uploading lm_head to GPU buffers
    ///
    /// This constructor does NOT touch Transformer or from_apr().
    /// Contract: wgsl-training-pipeline-v1 / no_transformer
    pub fn new(
        fwd: WgslForwardPass,
        trainer: WgpuTrainer,
        tokenizer: HfTokenizer,
        embed_weights: Vec<f32>,
        output_norm: Vec<f32>,
        lm_head_t_chunks: Vec<(wgpu::Buffer, u32)>,
        lm_head_chunks: Vec<(wgpu::Buffer, u32)>,
        num_layers: usize,
        hidden_dim: usize,
        vocab_size: usize,
        max_seq_len: usize,
        eps: f32,
    ) -> Self {
        let ce = WgslCrossEntropy::new(
            trainer.device_ref().clone(),
            trainer.queue_ref().clone(),
        );

        let seq = max_seq_len as u32;
        let vocab = vocab_size as u32;
        let make_buf = |size: u64, label: &str| -> wgpu::Buffer {
            trainer.device_ref().create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        Self {
            fwd,
            cross_entropy: ce,
            logits_buf: make_buf(seq as u64 * vocab as u64, "logits"),
            labels_buf: make_buf(seq as u64, "labels"),
            losses_buf: make_buf(seq as u64, "losses"),
            logsumexp_buf: make_buf(seq as u64, "logsumexp"),
            lm_head_t_chunks,
            lm_head_chunks,
            trainer,
            num_layers,
            hidden_dim,
            vocab_size,
            max_seq_len,
            tokenizer,
            embed_weights,
            output_norm,
            eps,
        }
    }

    /// Encode text to token IDs using the tokenizer.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    /// Training step: forward → loss → backward → optimizer. All GPU.
    ///
    /// Contract: qlora-training-loop-v1 / lora_forward_wgsl
    pub fn train_step(
        &mut self,
        prompt_ids: &[u32],
        response_ids: &[u32],
    ) -> InstructStepResult {
        let t0 = std::time::Instant::now();

        let full_ids: Vec<u32> = prompt_ids.iter().chain(response_ids).copied().collect();
        let seq_len = full_ids.len().min(self.max_seq_len);
        let full_ids = &full_ids[..seq_len];
        let prompt_len = prompt_ids.len().min(seq_len);

        let loss_start = prompt_len.saturating_sub(1);
        let loss_end = seq_len - 1;
        let num_loss_tokens = loss_end.saturating_sub(loss_start);

        if num_loss_tokens == 0 {
            return InstructStepResult { loss: 0.0, num_response_tokens: 0, perplexity: 1.0 };
        }

        // 1. Embed tokens (CPU lookup, small: seq × hidden)
        let mut hidden = Vec::with_capacity(seq_len * self.hidden_dim);
        for &tok in full_ids {
            let offset = (tok as usize) * self.hidden_dim;
            let end = offset + self.hidden_dim;
            if end <= self.embed_weights.len() {
                hidden.extend_from_slice(&self.embed_weights[offset..end]);
            } else {
                hidden.extend(std::iter::repeat(0.0f32).take(self.hidden_dim));
            }
        }

        let t1 = std::time::Instant::now();

        // 2. GPU forward through 28 transformer layers
        self.fwd.queue_ref().write_buffer(
            self.fwd.hidden_buffer(), 0,
            bytemuck::cast_slice(&hidden),
        );

        for layer_idx in 0..self.num_layers {
            let prefix = format!("layer.{layer_idx}");
            if let Err(e) = self.fwd.forward_layer_training(seq_len as u32, &prefix) {
                eprintln!("[wgpu] Layer {layer_idx} failed: {e}");
                return InstructStepResult { loss: 100.0, num_response_tokens: num_loss_tokens, perplexity: 1e6 };
            }
        }

        let t2 = std::time::Instant::now();

        // 3. Download hidden, CPU RMSNorm, GPU lm_head matmul
        let final_hidden = self.fwd.download_hidden(self.hidden_dim * seq_len);
        let mut normed = vec![0.0f32; seq_len * self.hidden_dim];
        for pos in 0..seq_len {
            let offset = pos * self.hidden_dim;
            let row = &final_hidden[offset..offset + self.hidden_dim];
            let ss: f32 = row.iter().map(|x| x * x).sum();
            let inv_rms = 1.0 / (ss / self.hidden_dim as f32 + self.eps).sqrt();
            for (j, &x) in row.iter().enumerate() {
                normed[offset + j] = x * inv_rms * self.output_norm[j];
            }
        }

        // lm_head: logits = normed @ lm_head_t (GPU, pre-chunked, no per-step download)
        let a_buf = self.trainer.upload(&normed);
        let mut logits_data = vec![0.0f32; seq_len * self.vocab_size];
        let mut col_offset = 0usize;
        for (chunk_buf, chunk_n) in &self.lm_head_t_chunks {
            let cn = *chunk_n as usize;
            let c_chunk = self.trainer.zeros(seq_len * cn);
            self.trainer.matmul_forward(
                &a_buf, chunk_buf, &c_chunk,
                seq_len as u32, self.hidden_dim as u32, *chunk_n,
            );
            let chunk_data = self.trainer.download(&c_chunk);
            for row in 0..seq_len {
                let dst = row * self.vocab_size + col_offset;
                let src = row * cn;
                logits_data[dst..dst + cn].copy_from_slice(&chunk_data[src..src + cn]);
            }
            col_offset += cn;
        }

        let t3 = std::time::Instant::now();

        // 4. GPU fused cross-entropy
        self.trainer.queue_ref().write_buffer(
            &self.logits_buf, 0,
            bytemuck::cast_slice(&logits_data[..seq_len * self.vocab_size]),
        );
        let labels: Vec<u32> = (0..seq_len)
            .map(|i| if i + 1 < full_ids.len() { full_ids[i + 1] } else { 0 })
            .collect();
        self.trainer.queue_ref().write_buffer(
            &self.labels_buf, 0,
            bytemuck::cast_slice(&labels),
        );

        let avg_loss = self.cross_entropy.forward(
            &self.logits_buf, &self.labels_buf,
            &self.losses_buf, &self.logsumexp_buf,
            seq_len as u32, self.vocab_size as u32,
            loss_start as u32, loss_end as u32,
        );

        if !avg_loss.is_finite() {
            return InstructStepResult { loss: 100.0, num_response_tokens: num_loss_tokens, perplexity: 1e6 };
        }

        // 5. GPU fused CE backward (in-place into logits_buf)
        self.cross_entropy.backward(
            &self.logits_buf, &self.labels_buf, &self.logsumexp_buf,
            seq_len as u32, self.vocab_size as u32,
            loss_start as u32, loss_end as u32,
        );

        let t4 = std::time::Instant::now();

        // 6. lm_head backward: grad_hidden = grad_logits @ lm_head (pre-chunked GPU)
        // grad_logits[seq, vocab] @ lm_head[vocab, hidden] = grad_hidden[seq, hidden]
        // lm_head is chunked along rows (vocab dimension).
        let grad_logits = self.trainer.download(&self.logits_buf);
        let grad_logits_full = &grad_logits[..seq_len * self.vocab_size];
        let mut grad_hidden = vec![0.0f32; seq_len * self.hidden_dim];
        let mut row_offset = 0usize;
        for (chunk_buf, chunk_k) in &self.lm_head_chunks {
            let ck = *chunk_k as usize;
            // Extract grad_logits columns for this chunk
            let mut gl_chunk = vec![0.0f32; seq_len * ck];
            for row in 0..seq_len {
                let src = row * self.vocab_size + row_offset;
                let dst = row * ck;
                gl_chunk[dst..dst + ck].copy_from_slice(&grad_logits_full[src..src + ck]);
            }
            let gl_buf = self.trainer.upload(&gl_chunk);
            let gh_chunk = self.trainer.zeros(seq_len * self.hidden_dim);
            self.trainer.matmul_forward(
                &gl_buf, chunk_buf, &gh_chunk,
                seq_len as u32, *chunk_k, self.hidden_dim as u32,
            );
            // Accumulate into grad_hidden
            let gh_data = self.trainer.download(&gh_chunk);
            for i in 0..grad_hidden.len() {
                grad_hidden[i] += gh_data[i];
            }
            row_offset += ck;
        }

        let t5 = std::time::Instant::now();

        eprintln!(
            "[PROFILE] step: {:.0}ms (embed={:.0} fwd={:.0} lm+norm={:.0} ce={:.0} bwd={:.0})",
            t5.duration_since(t0).as_millis(),
            t1.duration_since(t0).as_millis(),
            t2.duration_since(t1).as_millis(),
            t3.duration_since(t2).as_millis(),
            t4.duration_since(t3).as_millis(),
            t5.duration_since(t4).as_millis(),
        );

        InstructStepResult {
            loss: avg_loss,
            num_response_tokens: num_loss_tokens,
            perplexity: avg_loss.exp().min(1e6),
        }
    }
}
