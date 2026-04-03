//! NF4 QLoRA backward pass: `backward_nf4_gpu_blocks`,
//! `backward_nf4_gpu_blocks_gpu_resident`, `backward_nf4_gpu_blocks_loop`.

#[allow(clippy::wildcard_imports)]
use super::*;

#[cfg(feature = "cuda")]
use trueno_gpu::driver::GpuBuffer;

impl InstructPipeline {
    /// NF4 QLoRA backward pass through all GPU transformer blocks.
    ///
    /// Computes gradient flow through frozen NF4 weights and updates LoRA
    /// adapters. After each block backward, immediately runs the LoRA optimizer
    /// step (grad workspace is shared across layers).
    #[cfg(feature = "cuda")]
    #[allow(unsafe_code)]
    pub(super) fn backward_nf4_gpu_blocks(
        &mut self,
        grad_final_hidden: &[f32],
        seq_len: usize,
    ) -> Option<()> {
        let hidden_size = self.model.config.hidden_size;

        // Upload gradient and run RMSNorm backward in a scope to release borrows
        // before calling the shared block-loop.
        {
            let trainer = self.cuda_trainer.as_ref()?;
            let training_state = self.gpu_training.as_mut()?;
            let stream = trainer.stream();

            // PMAT-420: Use trainer.upload (fresh alloc) instead of copy_from_host_at
            training_state.grad_upload_buf = trainer.upload(grad_final_hidden).ok()?;

            // PMAT-420: Re-allocate grad buffers at seq_len if forward re-sized layer_inputs
            let expected_len = seq_len * hidden_size;
            if training_state.grad_buf_a.len() != expected_len {
                training_state.grad_buf_a = trainer.zeros(expected_len).ok()?;
                training_state.grad_buf_b = trainer.zeros(expected_len).ok()?;
                training_state.output_scratch = trainer.zeros(expected_len).ok()?;
                training_state.grad_upload_buf = trainer.upload(grad_final_hidden).ok()?;
            }

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
    /// - ~5MB D2H download (grad_hidden_buf -> CPU)
    /// - ~5MB H2D upload (CPU -> grad_upload_buf)
    /// - 1x stream.synchronize() GPU drain point
    /// - 1x Vec<f32> heap allocation (~5MB)
    #[cfg(feature = "cuda")]
    #[allow(unsafe_code)]
    pub(super) fn backward_nf4_gpu_blocks_gpu_resident(&mut self, seq_len: usize) -> Option<()> {
        let hidden_size = self.model.config.hidden_size;

        // KAIZEN-065: grad_hidden_buf already contains the gradient from lm_head backward GEMM.
        {
            let trainer = self.cuda_trainer.as_ref()?;
            let training_state = self.gpu_training.as_mut()?;
            let stream = trainer.stream();

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

        let result = self.backward_nf4_gpu_blocks_loop(seq_len);
        // entrenar#318: invalidate shared scratch causal mask after backward
        // to prevent gradient contamination on next forward (backward writes to shared scratch)
        if let Some(ref mut scratch) = self.shared_scratch {
            scratch.causal_mask_cached_seq_len = 0;
        }
        result
    }

    /// Shared backward loop for NF4 blocks -- called by both CPU-upload and
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

            // ENT-265: Clip NF4 LoRA gradients before optimizer step.
            // PMAT-477: Fused clip (zero D2H sync) when available; sync fallback otherwise.
            if let Some(max_norm) = self.config.gradient_clip_norm {
                if let Some(ref clip_state) = self.lora_fused_clip {
                    super::super::fused_lora_clip::clip_lora_gradients_fused(
                        grad_lora, max_norm, clip_state, stream,
                    );
                } else {
                    grad_lora.clip_gradients(max_norm, stream);
                }
            }

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
}
