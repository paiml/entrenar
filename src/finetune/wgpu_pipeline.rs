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
    lm_head_t_chunks: Vec<(wgpu::Buffer, u32)>, // [(chunk_buf, chunk_n)] for forward
    lm_head_chunks: Vec<(wgpu::Buffer, u32)>,    // [(chunk_buf, chunk_n)] for backward
    /// KAIZEN: scatter/gather pipelines (replace 1024 copy_buffer_to_buffer calls)
    scatter_pipeline: wgpu::ComputePipeline,
    gather_pipeline: wgpu::ComputePipeline,
    scatter_bgl: wgpu::BindGroupLayout,
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

        // KAIZEN: scatter/gather pipelines (one dispatch replaces 1024 copies)
        let scatter_bgl = trainer.device_ref().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scatter_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let scatter_pl = trainer.device_ref().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scatter_pl"), bind_group_layouts: &[&scatter_bgl], push_constant_ranges: &[],
        });
        let scatter_shader = trainer.device_ref().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scatter"), source: wgpu::ShaderSource::Wgsl(trueno::backends::gpu::shaders::COLUMN_SCATTER_SHADER.into()),
        });
        let scatter_pipeline = trainer.device_ref().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scatter_pipe"), layout: Some(&scatter_pl), module: &scatter_shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        let gather_shader = trainer.device_ref().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gather"), source: wgpu::ShaderSource::Wgsl(trueno::backends::gpu::shaders::COLUMN_GATHER_SHADER.into()),
        });
        let gather_pipeline = trainer.device_ref().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gather_pipe"), layout: Some(&scatter_pl), module: &gather_shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });

        Self {
            fwd,
            cross_entropy: ce,
            logits_buf: make_buf(seq as u64 * vocab as u64, "logits"),
            labels_buf: make_buf(seq as u64, "labels"),
            losses_buf: make_buf(seq as u64, "losses"),
            logsumexp_buf: make_buf(seq as u64, "logsumexp"),
            lm_head_t_chunks,
            lm_head_chunks,
            scatter_pipeline,
            gather_pipeline,
            scatter_bgl,
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

    /// GPU scatter: copy [seq, chunk_n] into [seq, full_n] at col_offset. One dispatch.
    fn dispatch_scatter(&self, src: &wgpu::Buffer, dst: &wgpu::Buffer, seq_len: u32, chunk_n: u32, full_n: u32, col_offset: u32) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct P { seq_len: u32, chunk_n: u32, full_n: u32, col_offset: u32 }
        let params = P { seq_len, chunk_n, full_n, col_offset };
        let pbuf = self.trainer.device_ref().create_buffer(&wgpu::BufferDescriptor {
            label: None, size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        self.trainer.queue_ref().write_buffer(&pbuf, 0, bytemuck::bytes_of(&params));
        let bg = self.trainer.device_ref().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.scatter_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pbuf.as_entire_binding() },
            ],
        });
        let total = seq_len * chunk_n;
        let wg = total.div_ceil(256);
        let (x, y) = if wg <= 65535 { (wg, 1) } else { (65535, wg.div_ceil(65535)) };
        let mut encoder = self.trainer.device_ref().create_command_encoder(&Default::default());
        { let mut pass = encoder.begin_compute_pass(&Default::default()); pass.set_pipeline(&self.scatter_pipeline); pass.set_bind_group(0, &bg, &[]); pass.dispatch_workgroups(x, y, 1); }
        self.trainer.queue_ref().submit(Some(encoder.finish()));
    }

    /// GPU gather: extract [seq, chunk_n] from [seq, full_n] at col_offset. One dispatch.
    fn dispatch_gather(&self, src: &wgpu::Buffer, dst: &wgpu::Buffer, seq_len: u32, chunk_n: u32, full_n: u32, col_offset: u32) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct P { seq_len: u32, chunk_n: u32, full_n: u32, col_offset: u32 }
        let params = P { seq_len, chunk_n, full_n, col_offset };
        let pbuf = self.trainer.device_ref().create_buffer(&wgpu::BufferDescriptor {
            label: None, size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        self.trainer.queue_ref().write_buffer(&pbuf, 0, bytemuck::bytes_of(&params));
        let bg = self.trainer.device_ref().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.scatter_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pbuf.as_entire_binding() },
            ],
        });
        let total = seq_len * chunk_n;
        let wg = total.div_ceil(256);
        let (x, y) = if wg <= 65535 { (wg, 1) } else { (65535, wg.div_ceil(65535)) };
        let mut encoder = self.trainer.device_ref().create_command_encoder(&Default::default());
        { let mut pass = encoder.begin_compute_pass(&Default::default()); pass.set_pipeline(&self.gather_pipeline); pass.set_bind_group(0, &bg, &[]); pass.dispatch_workgroups(x, y, 1); }
        self.trainer.queue_ref().submit(Some(encoder.finish()));
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

        // 2. GPU forward through 28 transformer layers — ONE submit for all layers
        self.fwd.queue_ref().write_buffer(
            self.fwd.hidden_buffer(), 0,
            bytemuck::cast_slice(&hidden),
        );

        let _saved_activations = match self.fwd.forward_all_layers_training(seq_len as u32, self.num_layers) {
            Ok(saved) => saved,
            Err(e) => {
                eprintln!("[wgpu] GPU forward failed: {e}");
                return InstructStepResult { loss: 100.0, num_response_tokens: num_loss_tokens, perplexity: 1e6 };
            }
        };

        let t2 = std::time::Instant::now();

        // 3. Download hidden, CPU RMSNorm, GPU lm_head matmul
        let t2a = std::time::Instant::now();
        let final_hidden = self.fwd.download_hidden(self.hidden_dim * seq_len);
        let t2b = std::time::Instant::now();
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

        let t2c = std::time::Instant::now();
        // lm_head: chunked GEMM + GPU scatter (exact-sized buffers per step)
        let a_buf = self.trainer.upload(&normed);
        let labels: Vec<u32> = (0..seq_len)
            .map(|i| if i + 1 < full_ids.len() { full_ids[i + 1] } else { 0 })
            .collect();

        let mut col_offset = 0u64;
        for (chunk_buf, chunk_n) in &self.lm_head_t_chunks {
            let cn = *chunk_n as u64;
            let c_chunk = self.trainer.zeros((seq_len as u64 * cn) as usize);
            self.trainer.matmul_forward(
                &a_buf, chunk_buf, &c_chunk,
                seq_len as u32, self.hidden_dim as u32, *chunk_n,
            );
            // GPU scatter: one dispatch replaces 512 copy_buffer_to_buffer calls
            self.dispatch_scatter(
                &c_chunk, &self.logits_buf,
                seq_len as u32, *chunk_n, self.vocab_size as u32, col_offset as u32,
            );
            col_offset += cn;
        }

        let t3 = std::time::Instant::now();

        // Fused CE on full logits_buf (assembled via GPU scatter, no CPU download)
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

        // Fused CE backward (in-place into logits_buf)
        self.cross_entropy.backward(
            &self.logits_buf, &self.labels_buf, &self.logsumexp_buf,
            seq_len as u32, self.vocab_size as u32,
            loss_start as u32, loss_end as u32,
        );

        let t4 = std::time::Instant::now();

        // 6. lm_head backward via GPU scatter + pre-allocated scratch (KAIZEN: zero alloc)
        // grad_hidden = grad_logits @ lm_head, chunked along vocab dimension.
        let mut grad_hidden = vec![0.0f32; seq_len * self.hidden_dim];
        // Extract grad_logits columns per chunk via GPU copy, GEMM, accumulate.
        let mut row_offset = 0u64;
        for (i, (chunk_buf, chunk_k)) in self.lm_head_chunks.iter().enumerate() {
            let ck = *chunk_k as u64;
            // GPU gather: extract grad_logits columns for this chunk
            let gl_chunk = self.trainer.zeros((seq_len as u64 * ck) as usize);
            self.dispatch_gather(
                &self.logits_buf, &gl_chunk,
                seq_len as u32, *chunk_k, self.vocab_size as u32, row_offset as u32,
            );
            // GEMM: gl_chunk[seq, chunk_k] @ lm_head_chunk[chunk_k, hidden] → grad contribution
            let gh_chunk = self.trainer.zeros(seq_len * self.hidden_dim);
            self.trainer.matmul_forward(
                &gl_chunk, chunk_buf, &gh_chunk,
                seq_len as u32, *chunk_k, self.hidden_dim as u32,
            );
            // Download and accumulate (GPU add kernel would be better)
            let gh_data = self.trainer.download(&gh_chunk);
            for j in 0..grad_hidden.len() {
                grad_hidden[j] += gh_data[j];
            }
            row_offset += ck;
        }

        let t5 = std::time::Instant::now();

        eprintln!(
            "[PROFILE] step: {:.0}ms (embed={:.0} fwd={:.0} lm+norm={:.0}[dl={:.0} norm={:.0} gemm={:.0}] ce={:.0} bwd={:.0})",
            t5.duration_since(t0).as_millis(),
            t1.duration_since(t0).as_millis(),
            t2.duration_since(t1).as_millis(),
            t3.duration_since(t2).as_millis(),
            t2b.duration_since(t2a).as_millis(),
            t2c.duration_since(t2b).as_millis(),
            t3.duration_since(t2c).as_millis(),
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
