//! NF4 weight management for WGPU training
//!
//! Loads NF4-quantized weights and dequantizes on GPU per-layer.
//! This keeps total VRAM under 16GB for Qwen3-4B (36 layers).
//!
//! # Contract: wgpu-transformer-trainer-v1.yaml (C-WGPU-TRAIN-001)
//!
//! - NF4 dequant on GPU matches CPU within ε < 1e-6 (FALSIFY-WGPU-003)
//! - Per-layer dequant avoids storing all fp32 weights simultaneously

#[cfg(feature = "gpu")]
use trueno::backends::gpu::GpuDevice;

/// NF4 quantized layer weights (compact representation)
///
/// Stores packed 4-bit nibbles + per-block scales.
/// Total size per projection: n_params / 2 bytes (packed) + n_params / block_size * 4 bytes (scales)
#[cfg(feature = "gpu")]
pub struct Nf4LayerWeights {
    /// Gate projection [intermediate, hidden] packed NF4
    pub gate_packed: Vec<u32>,
    pub gate_scales: Vec<f32>,
    /// Up projection [intermediate, hidden] packed NF4
    pub up_packed: Vec<u32>,
    pub up_scales: Vec<f32>,
    /// Down projection [hidden, intermediate] packed NF4
    pub down_packed: Vec<u32>,
    pub down_scales: Vec<f32>,
    /// Q projection [num_heads * head_dim, hidden] packed NF4
    pub q_packed: Vec<u32>,
    pub q_scales: Vec<f32>,
    /// K projection [num_kv_heads * head_dim, hidden] packed NF4
    pub k_packed: Vec<u32>,
    pub k_scales: Vec<f32>,
    /// V projection [num_kv_heads * head_dim, hidden] packed NF4
    pub v_packed: Vec<u32>,
    pub v_scales: Vec<f32>,
    /// O projection [hidden, num_heads * head_dim] packed NF4
    pub o_packed: Vec<u32>,
    pub o_scales: Vec<f32>,
    /// Number of elements per projection
    pub gate_n: u32,
    pub up_n: u32,
    pub down_n: u32,
    pub q_n: u32,
    pub k_n: u32,
    pub v_n: u32,
    pub o_n: u32,
    /// NF4 block size (typically 64)
    pub block_size: u32,
}

#[cfg(feature = "gpu")]
impl Nf4LayerWeights {
    /// Dequantize gate projection to fp32 on GPU
    pub fn dequant_gate(&self, device: &GpuDevice) -> Result<Vec<f32>, String> {
        let mut output = vec![0.0f32; self.gate_n as usize];
        device.nf4_dequant(
            &self.gate_packed,
            &self.gate_scales,
            &mut output,
            self.gate_n,
            self.block_size,
        )?;
        Ok(output)
    }

    /// Dequantize up projection to fp32 on GPU
    pub fn dequant_up(&self, device: &GpuDevice) -> Result<Vec<f32>, String> {
        let mut output = vec![0.0f32; self.up_n as usize];
        device.nf4_dequant(
            &self.up_packed,
            &self.up_scales,
            &mut output,
            self.up_n,
            self.block_size,
        )?;
        Ok(output)
    }

    /// Dequantize down projection to fp32 on GPU
    pub fn dequant_down(&self, device: &GpuDevice) -> Result<Vec<f32>, String> {
        let mut output = vec![0.0f32; self.down_n as usize];
        device.nf4_dequant(
            &self.down_packed,
            &self.down_scales,
            &mut output,
            self.down_n,
            self.block_size,
        )?;
        Ok(output)
    }

    /// Memory usage in bytes (NF4 packed + scales)
    pub fn memory_bytes(&self) -> usize {
        let packed_bytes = (self.gate_packed.len() + self.up_packed.len() + self.down_packed.len()
            + self.q_packed.len() + self.k_packed.len() + self.v_packed.len()
            + self.o_packed.len()) * 4;
        let scale_bytes = (self.gate_scales.len() + self.up_scales.len() + self.down_scales.len()
            + self.q_scales.len() + self.k_scales.len() + self.v_scales.len()
            + self.o_scales.len()) * 4;
        packed_bytes + scale_bytes
    }
}

/// LoRA adapter pair for a single projection (rank-r)
///
/// Forward: y = x @ W^T + x @ B^T @ A^T (where A is [rank, in_dim], B is [out_dim, rank])
/// Backward: gradients flow through B and A, frozen base W is not updated
#[cfg(feature = "gpu")]
pub struct LoraAdapter {
    /// A matrix [rank, in_dim] — fp32, trainable
    pub a: Vec<f32>,
    /// B matrix [out_dim, rank] — fp32, trainable
    pub b: Vec<f32>,
    /// AdamW first moment for A
    pub m_a: Vec<f32>,
    /// AdamW second moment for A
    pub v_a: Vec<f32>,
    /// AdamW first moment for B
    pub m_b: Vec<f32>,
    /// AdamW second moment for B
    pub v_b: Vec<f32>,
    /// Dimensions
    pub rank: u32,
    pub in_dim: u32,
    pub out_dim: u32,
}

#[cfg(feature = "gpu")]
impl LoraAdapter {
    /// Create a new LoRA adapter with Kaiming-uniform A and zero B
    pub fn new(rank: u32, in_dim: u32, out_dim: u32) -> Self {
        let a_len = (rank * in_dim) as usize;
        let b_len = (out_dim * rank) as usize;

        // Kaiming-uniform initialization for A
        let scale = (2.0 / in_dim as f64).sqrt() as f32;
        let mut a = vec![0.0f32; a_len];
        // Simple deterministic pseudo-random init
        for (i, val) in a.iter_mut().enumerate() {
            let hash = ((i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)) as f32;
            *val = (hash / u64::MAX as f32 * 2.0 - 1.0) * scale;
        }

        Self {
            a,
            b: vec![0.0f32; b_len], // B initialized to zero → LoRA starts as identity
            m_a: vec![0.0f32; a_len],
            v_a: vec![0.0f32; a_len],
            m_b: vec![0.0f32; b_len],
            v_b: vec![0.0f32; b_len],
            rank,
            in_dim,
            out_dim,
        }
    }

    /// Total trainable parameters
    pub fn num_params(&self) -> usize {
        self.a.len() + self.b.len()
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    #[test]
    fn test_lora_adapter_creation() {
        let adapter = LoraAdapter::new(16, 2560, 4096);
        assert_eq!(adapter.a.len(), 16 * 2560);
        assert_eq!(adapter.b.len(), 4096 * 16);
        assert_eq!(adapter.num_params(), 16 * 2560 + 4096 * 16);
        // B should be zero (identity initialization)
        assert!(adapter.b.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_nf4_layer_memory() {
        // Simulate Qwen3-4B FFN layer
        let h: u32 = 2560;
        let i: u32 = 9728;
        let bs: u32 = 64;

        let layer = Nf4LayerWeights {
            gate_packed: vec![0u32; (h * i / 8) as usize], // 4 bits per param, 8 params per u32
            gate_scales: vec![0.0f32; (h * i / bs) as usize],
            up_packed: vec![0u32; (h * i / 8) as usize],
            up_scales: vec![0.0f32; (h * i / bs) as usize],
            down_packed: vec![0u32; (i * h / 8) as usize],
            down_scales: vec![0.0f32; (i * h / bs) as usize],
            q_packed: vec![0u32; (h * 4096 / 8) as usize],
            q_scales: vec![0.0f32; (h * 4096 / bs) as usize],
            k_packed: vec![0u32; (h * 1024 / 8) as usize],
            k_scales: vec![0.0f32; (h * 1024 / bs) as usize],
            v_packed: vec![0u32; (h * 1024 / 8) as usize],
            v_scales: vec![0.0f32; (h * 1024 / bs) as usize],
            o_packed: vec![0u32; (4096 * h / 8) as usize],
            o_scales: vec![0.0f32; (4096 * h / bs) as usize],
            gate_n: h * i,
            up_n: h * i,
            down_n: i * h,
            q_n: h * 4096,
            k_n: h * 1024,
            v_n: h * 1024,
            o_n: 4096 * h,
            block_size: bs,
        };

        let mb = layer.memory_bytes() as f64 / 1024.0 / 1024.0;
        eprintln!("Qwen3-4B NF4 layer: {mb:.1} MB");
        assert!(mb < 100.0, "NF4 layer should be < 100MB, got {mb:.1}");
    }
}
