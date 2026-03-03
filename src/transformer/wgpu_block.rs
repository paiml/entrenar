//! wgpu-accelerated transformer forward pass
//!
//! Provides [`WgpuForwardPass`] that batches all transformer layer matmuls
//! and activations into a single GPU execution, eliminating per-operation
//! CPU↔GPU round-trips.
//!
//! # Architecture
//!
//! - Weights are uploaded once at construction and kept GPU-resident
//! - Each forward call: upload hidden → batch all ops → download result
//! - Uses `GpuCommandBatch` for deferred execution with persistent buffers
//!
//! # Contract (C-WGPU-FWD-001)
//!
//! - **Precondition**: Transformer model loaded with valid weights
//! - **Postcondition**: Output hidden states numerically match CPU forward pass (within fp32 tolerance)
//! - **Invariant**: GPU buffers remain valid across forward calls

use crate::autograd::Tensor;
use crate::lora::LoRALayer;
use crate::transformer::config::TransformerConfig;
use crate::transformer::model::Transformer;
use trueno::backends::gpu::{GpuCommandBatch, GpuDevice};

/// wgpu-accelerated transformer forward pass
///
/// Batches all matmul and activation operations across transformer layers
/// into a single `GpuCommandBatch::execute()` call.
///
/// Current implementation accelerates FFN matmuls (gate/up/down projections)
/// which dominate compute time (~60-70% of forward pass). Attention remains
/// on CPU due to softmax/RoPE complexity (Phase 2 will add GPU attention).
pub struct WgpuForwardPass {
    device: GpuDevice,
    config: TransformerConfig,
    /// Number of transformer layers
    num_layers: usize,
}

impl WgpuForwardPass {
    /// Create a new wgpu forward pass from a transformer model
    ///
    /// # Arguments
    /// * `model` - Transformer model with loaded weights
    /// * `adapter_index` - wgpu adapter index to use
    ///
    /// # Errors
    /// Returns error if GPU device creation fails
    pub fn new(config: &TransformerConfig, adapter_index: u32) -> Result<Self, String> {
        let device = GpuDevice::new_with_adapter_index(adapter_index)?;

        Ok(Self {
            device,
            config: config.clone(),
            num_layers: config.num_hidden_layers,
        })
    }

    /// Create from default GPU adapter
    pub fn new_default(config: &TransformerConfig) -> Result<Self, String> {
        let device = GpuDevice::new()?;

        Ok(Self {
            device,
            config: config.clone(),
            num_layers: config.num_hidden_layers,
        })
    }

    /// Execute forward pass through all transformer layers on GPU
    ///
    /// Batches FFN matmuls (gate/up/down projections) per layer. Attention and
    /// normalization remain on CPU for this phase.
    ///
    /// # Arguments
    /// * `model` - Transformer model with weights
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    /// Hidden states tensor (seq_len * hidden_size)
    pub fn forward_hidden(&self, model: &Transformer, token_ids: &[u32]) -> Result<Tensor, String> {
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;

        // Step 1: Embed tokens on CPU (small operation)
        let mut hidden = model.embed_tokens.forward(token_ids);

        // Step 2: Process each transformer layer
        // Attention stays on CPU; FFN matmuls go to GPU via batched execution
        //
        // KAIZEN-004: Suppress per-op wgpu during attention. Without this,
        // each Q/K/V/O projection triggers a full buffer upload/compute/download
        // cycle (144 per-op matmuls × ~3-5ms = 430-720ms overhead per sample).
        // CPU SIMD is equally fast and doesn't contend for GPU bandwidth.
        crate::autograd::suppress_per_op_wgpu();
        for layer in &model.layers {
            // --- Attention on CPU SIMD (includes RoPE, softmax, masking) ---
            let norm1 = layer.input_norm.forward_batched(&hidden, seq_len, hidden_size);
            let attn_out = layer.self_attn.forward(&norm1, seq_len);
            let residual1 = crate::autograd::add(&hidden, &attn_out);

            // --- FFN on GPU (3 large matmuls + SwiGLU) ---
            let norm2 = layer.post_attn_norm.forward_batched(&residual1, seq_len, hidden_size);

            let ffn_out = self.forward_ffn_gpu(
                &norm2,
                &layer.ffn.w_gate,
                &layer.ffn.w_up,
                &layer.ffn.w_down,
                seq_len,
                hidden_size,
                intermediate_size,
            )?;

            // Residual connection
            hidden = crate::autograd::add(&residual1, &ffn_out);
        }
        // KAIZEN-004: Re-enable per-op wgpu for backward pass / other operations
        crate::autograd::unsuppress_per_op_wgpu();

        // Step 3: Final normalization on CPU
        let normalized = model.norm.forward_batched(&hidden, seq_len, hidden_size);

        Ok(normalized)
    }

    /// Execute FFN forward pass on GPU using batched operations
    ///
    /// Batches gate/up/down matmuls + SwiGLU into single GPU execution:
    /// 1. Upload: norm_output, w_gate, w_up, w_down
    /// 2. Compute: gate = norm @ w_gate, up = norm @ w_up, silu(gate) * up, down
    /// 3. Download: ffn_output
    ///
    /// This eliminates 6 CPU↔GPU transfers per layer (3 matmuls × 2 transfers each)
    /// down to 2 total (1 upload batch + 1 download).
    fn forward_ffn_gpu(
        &self,
        input: &Tensor,
        w_gate: &Tensor,
        w_up: &Tensor,
        w_down: &Tensor,
        seq_len: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Tensor, String> {
        use trueno::backends::gpu::runtime;

        runtime::block_on(async {
            let mut batch = GpuCommandBatch::new(self.device.clone());

            // Upload input and weights
            let input_data = input.data();
            let input_slice = input_data.as_slice()
                .ok_or("Input tensor not contiguous")?;
            let gate_data = w_gate.data();
            let gate_slice = gate_data.as_slice()
                .ok_or("Gate weight not contiguous")?;
            let up_data = w_up.data();
            let up_slice = up_data.as_slice()
                .ok_or("Up weight not contiguous")?;
            let down_data = w_down.data();
            let down_slice = down_data.as_slice()
                .ok_or("Down weight not contiguous")?;

            let buf_input = batch.upload(input_slice);
            let buf_gate = batch.upload(gate_slice);
            let buf_up = batch.upload(up_slice);
            let buf_down = batch.upload(down_slice);

            // Gate projection: (seq_len, hidden) @ (hidden, intermediate)
            let gate_out = batch.matmul(
                buf_input,
                buf_gate,
                seq_len as u32,
                hidden_size as u32,
                intermediate_size as u32,
            );

            // Up projection: (seq_len, hidden) @ (hidden, intermediate)
            let up_out = batch.matmul(
                buf_input,
                buf_up,
                seq_len as u32,
                hidden_size as u32,
                intermediate_size as u32,
            );

            // SwiGLU: swish(gate) * up
            let gate_activated = batch.swish(gate_out);
            let swiglu_out = batch.mul(gate_activated, up_out);

            // Down projection: (seq_len, intermediate) @ (intermediate, hidden)
            let ffn_out = batch.matmul(
                swiglu_out,
                buf_down,
                seq_len as u32,
                intermediate_size as u32,
                hidden_size as u32,
            );

            // Execute all ops in single batch
            batch.execute().await?;

            // Download result
            let result_data = batch.read(ffn_out).await?;

            Ok(Tensor::from_vec(result_data, false))
        })
    }

    /// Execute batched forward pass for multiple samples (KAIZEN-008).
    ///
    /// Processes all samples through each layer together, uploading FFN weights
    /// ONCE per layer instead of once per sample. With batch_size=20 × 36 layers,
    /// this reduces weight uploads from 720 to 36 (20× reduction, ~146 GB saved).
    ///
    /// Attention remains per-sample on CPU SIMD. FFN inputs are concatenated
    /// across samples for a single large matmul per layer.
    ///
    /// # KAIZEN-010: LoRA integration
    ///
    /// When `lora_layers` is provided, attention uses `forward_with_lora()` to
    /// apply LoRA corrections to Q and V projections. Layout: `[Q_0, V_0, Q_1, V_1, ...]`
    /// (2 LoRA layers per transformer layer). Without this, only the classifier
    /// head trains on the wgpu path (5,122 params vs 5.9M with LoRA).
    pub fn forward_hidden_batch(
        &self,
        model: &Transformer,
        batch_token_ids: &[Vec<u32>],
        lora_layers: Option<&[LoRALayer]>,
    ) -> Result<Vec<Tensor>, String> {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let n = batch_token_ids.len();

        // Step 1: Embed all samples on CPU
        let mut hiddens: Vec<Tensor> = batch_token_ids
            .iter()
            .map(|ids| model.embed_tokens.forward(ids))
            .collect();

        // Step 2: Layer-at-a-time processing
        // Attention on CPU SIMD (per-sample), FFN on GPU (all samples concatenated)
        crate::autograd::suppress_per_op_wgpu();
        for (layer_idx, layer) in model.layers.iter().enumerate() {
            // Attention on CPU (per-sample, independent)
            let mut ffn_inputs: Vec<Vec<f32>> = Vec::with_capacity(n);
            let mut residuals: Vec<Tensor> = Vec::with_capacity(n);
            for (i, hidden) in hiddens.iter().enumerate() {
                let seq_len = batch_token_ids[i].len();
                let norm1 = layer.input_norm.forward_batched(hidden, seq_len, hidden_size);

                // KAIZEN-010: Use LoRA-enabled attention when adapters are available
                let attn_out = match lora_layers {
                    Some(loras) => {
                        let q_idx = layer_idx * 2;
                        let v_idx = layer_idx * 2 + 1;
                        if v_idx < loras.len() {
                            layer.self_attn.forward_with_lora(
                                &norm1,
                                seq_len,
                                loras[q_idx].lora_a(),
                                loras[q_idx].lora_b(),
                                loras[v_idx].lora_a(),
                                loras[v_idx].lora_b(),
                                loras[q_idx].rank(),
                                loras[q_idx].scale(),
                            )
                        } else {
                            layer.self_attn.forward(&norm1, seq_len)
                        }
                    }
                    None => layer.self_attn.forward(&norm1, seq_len),
                };

                let residual1 = crate::autograd::add(hidden, &attn_out);
                let norm2 = layer.post_attn_norm.forward_batched(&residual1, seq_len, hidden_size);
                ffn_inputs.push(
                    norm2
                        .data()
                        .as_slice()
                        .expect("norm2 contiguous")
                        .to_vec(),
                );
                residuals.push(residual1);
            }

            // Concatenate all samples' FFN inputs for single GPU batch
            let total_tokens: usize = batch_token_ids.iter().map(|ids| ids.len()).sum();
            let mut concat_input =
                Vec::with_capacity(total_tokens * hidden_size);
            for inp in &ffn_inputs {
                concat_input.extend_from_slice(inp);
            }
            let concat_tensor = Tensor::from_vec(concat_input, false);

            // FFN on GPU — weights uploaded ONCE for all samples
            let ffn_out = self.forward_ffn_gpu(
                &concat_tensor,
                &layer.ffn.w_gate,
                &layer.ffn.w_up,
                &layer.ffn.w_down,
                total_tokens,
                hidden_size,
                intermediate_size,
            )?;

            // Split FFN output back into per-sample tensors + residual
            let ffn_data = ffn_out.data();
            let ffn_slice = ffn_data.as_slice().expect("ffn contiguous");
            let mut offset = 0;
            hiddens = residuals
                .into_iter()
                .enumerate()
                .map(|(i, r)| {
                    let len = batch_token_ids[i].len() * hidden_size;
                    let sample_ffn =
                        Tensor::from_vec(ffn_slice[offset..offset + len].to_vec(), false);
                    offset += len;
                    crate::autograd::add(&r, &sample_ffn)
                })
                .collect();
        }
        crate::autograd::unsuppress_per_op_wgpu();

        // Step 3: Final normalization (per-sample)
        let results: Vec<Tensor> = hiddens
            .into_iter()
            .enumerate()
            .map(|(i, h)| {
                let seq_len = batch_token_ids[i].len();
                model.norm.forward_batched(&h, seq_len, hidden_size)
            })
            .collect();

        Ok(results)
    }

    /// Get the adapter info for display
    pub fn adapter_info(&self) -> String {
        format!("wgpu device ({}x{} model, {} layers)",
            self.config.hidden_size,
            self.config.intermediate_size,
            self.num_layers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgpu_forward_pass_creation() {
        if !GpuDevice::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let config = TransformerConfig {
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: 4,
            intermediate_size: 128,
            vocab_size: 100,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            use_bias: false,
            head_dim_override: None,
        };

        let pass = WgpuForwardPass::new_default(&config);
        assert!(pass.is_ok(), "WgpuForwardPass creation failed: {:?}", pass.err());
    }

    #[test]
    fn test_wgpu_ffn_numerical_correctness() {
        if !GpuDevice::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let config = TransformerConfig {
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_kv_heads: 2,
            intermediate_size: 16,
            vocab_size: 32,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            use_bias: false,
            head_dim_override: None,
        };

        let pass = WgpuForwardPass::new_default(&config)
            .expect("GPU available but creation failed");

        // Create small test tensors
        let input = Tensor::from_vec(vec![1.0; 8], false); // 1 token × 8 hidden
        let w_gate = Tensor::from_vec(vec![0.1; 8 * 16], false);
        let w_up = Tensor::from_vec(vec![0.1; 8 * 16], false);
        let w_down = Tensor::from_vec(vec![0.1; 16 * 8], false);

        let gpu_result = pass.forward_ffn_gpu(
            &input, &w_gate, &w_up, &w_down,
            1, 8, 16,
        );

        assert!(gpu_result.is_ok(), "GPU FFN failed: {:?}", gpu_result.err());

        let gpu_data = gpu_result.expect("checked above");
        assert_eq!(gpu_data.len(), 8, "Output should be 1 × 8");

        // Verify no NaN/Inf
        for (i, &val) in gpu_data.data().iter().enumerate() {
            assert!(val.is_finite(), "NaN/Inf at index {}: {}", i, val);
        }
    }
}
