//! GGUF-compatible quantization formats (Q4_0, Q8_0)
//!
//! Implements quantization formats compatible with llama.cpp and GGUF:
//! - Q4_0: 4-bit quantization with per-block f16 scale (32 elements/block)
//! - Q8_0: 8-bit quantization with per-block f16 scale (32 elements/block)
//!
//! Block structure:
//! - Q4_0: 2 bytes scale (f16) + 16 bytes data (32 × 4-bit) = 18 bytes/block
//! - Q8_0: 2 bytes scale (f16) + 32 bytes data (32 × 8-bit) = 34 bytes/block

use serde::{Deserialize, Serialize};

/// GGUF block size (standard for llama.cpp)
pub const GGUF_BLOCK_SIZE: usize = 32;

/// Q4_0 quantized tensor (GGUF format)
///
/// 4-bit quantization with per-block f16 scale factors.
/// Each block: 32 values → 18 bytes (2 scale + 16 data)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Q4_0 {
    /// Per-block scale factors (stored as f32, converted to f16 on export)
    pub scales: Vec<f32>,
    /// Packed 4-bit data (2 values per byte, 16 bytes per block)
    pub data: Vec<u8>,
    /// Original number of elements
    pub len: usize,
}

impl Q4_0 {
    /// Quantize f32 values to Q4_0 format
    pub fn quantize(values: &[f32]) -> Self {
        let len = values.len();
        let num_blocks = len.div_ceil(GGUF_BLOCK_SIZE);

        let mut scales = Vec::with_capacity(num_blocks);
        let mut data = Vec::with_capacity(num_blocks * 16); // 16 bytes per block

        for block_idx in 0..num_blocks {
            let start = block_idx * GGUF_BLOCK_SIZE;
            let end = (start + GGUF_BLOCK_SIZE).min(len);
            let block = &values[start..end];

            // Compute scale: max absolute value / 7 (4-bit signed: -8 to 7)
            let max_abs = block
                .iter()
                .map(|v| v.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);

            let scale = if max_abs < 1e-10 {
                1e-10
            } else {
                max_abs / 7.0
            };
            scales.push(scale);

            // Quantize block (pad with zeros if incomplete)
            let mut block_data = [0u8; 16];
            for i in 0..GGUF_BLOCK_SIZE {
                let val = if start + i < end { block[i] } else { 0.0 };

                // Quantize to [-8, 7] range
                let q = ((val / scale).round().clamp(-8.0, 7.0) as i8) & 0x0F;

                // Pack 2 values per byte
                if i % 2 == 0 {
                    block_data[i / 2] = (q as u8) & 0x0F;
                } else {
                    block_data[i / 2] |= ((q as u8) & 0x0F) << 4;
                }
            }
            data.extend_from_slice(&block_data);
        }

        Self { scales, data, len }
    }

    /// Dequantize Q4_0 back to f32
    pub fn dequantize(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.len);
        let num_blocks = self.scales.len();

        for block_idx in 0..num_blocks {
            let scale = self.scales[block_idx];
            let start = block_idx * GGUF_BLOCK_SIZE;
            let block_len = (self.len - start).min(GGUF_BLOCK_SIZE);

            for i in 0..block_len {
                let byte_idx = block_idx * 16 + i / 2;
                let byte = self.data[byte_idx];

                // Extract 4-bit value
                let nibble = if i % 2 == 0 {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                };

                // Sign extend from 4-bit
                let q = if nibble & 0x08 != 0 {
                    (nibble | 0xF0) as i8
                } else {
                    nibble as i8
                };

                result.push(q as f32 * scale);
            }
        }

        result
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.scales.len() * 4 + self.data.len() // scales as f32 for now
    }

    /// Get GGUF-format memory (with f16 scales)
    pub fn gguf_bytes(&self) -> usize {
        self.scales.len() * 2 + self.data.len() // 2 bytes per f16 scale
    }

    /// Get compression ratio vs f32
    pub fn compression_ratio(&self) -> f32 {
        let original = self.len * 4;
        original as f32 / self.gguf_bytes() as f32
    }

    /// Number of blocks
    pub fn num_blocks(&self) -> usize {
        self.scales.len()
    }
}

/// Q8_0 quantized tensor (GGUF format)
///
/// 8-bit quantization with per-block f16 scale factors.
/// Each block: 32 values → 34 bytes (2 scale + 32 data)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Q8_0 {
    /// Per-block scale factors (stored as f32, converted to f16 on export)
    pub scales: Vec<f32>,
    /// 8-bit quantized data (1 byte per value)
    pub data: Vec<i8>,
    /// Original number of elements
    pub len: usize,
}

impl Q8_0 {
    /// Quantize f32 values to Q8_0 format
    pub fn quantize(values: &[f32]) -> Self {
        let len = values.len();
        let num_blocks = len.div_ceil(GGUF_BLOCK_SIZE);

        let mut scales = Vec::with_capacity(num_blocks);
        let mut data = Vec::with_capacity(len);

        for block_idx in 0..num_blocks {
            let start = block_idx * GGUF_BLOCK_SIZE;
            let end = (start + GGUF_BLOCK_SIZE).min(len);
            let block = &values[start..end];

            // Compute scale: max absolute value / 127 (8-bit signed: -128 to 127)
            let max_abs = block
                .iter()
                .map(|v| v.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);

            let scale = if max_abs < 1e-10 {
                1e-10
            } else {
                max_abs / 127.0
            };
            scales.push(scale);

            // Quantize block
            for &val in block {
                let q = (val / scale).round().clamp(-128.0, 127.0) as i8;
                data.push(q);
            }

            // Pad incomplete blocks with zeros
            let padding = GGUF_BLOCK_SIZE - block.len();
            data.extend(std::iter::repeat_n(0i8, padding));
        }

        // Trim padding from last block
        data.truncate(len);

        Self { scales, data, len }
    }

    /// Dequantize Q8_0 back to f32
    pub fn dequantize(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.len);

        for (i, &q) in self.data.iter().enumerate() {
            let block_idx = i / GGUF_BLOCK_SIZE;
            let scale = self.scales[block_idx];
            result.push(q as f32 * scale);
        }

        result
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.scales.len() * 4 + self.data.len()
    }

    /// Get GGUF-format memory (with f16 scales)
    pub fn gguf_bytes(&self) -> usize {
        self.scales.len() * 2 + self.data.len()
    }

    /// Get compression ratio vs f32
    pub fn compression_ratio(&self) -> f32 {
        let original = self.len * 4;
        original as f32 / self.gguf_bytes() as f32
    }

    /// Number of blocks
    pub fn num_blocks(&self) -> usize {
        self.scales.len()
    }
}

/// Quantization type enum for GGUF export
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GGUFQuantType {
    /// 4-bit quantization
    Q4_0,
    /// 8-bit quantization
    Q8_0,
}

impl GGUFQuantType {
    /// Get bytes per block for this quantization type
    pub fn bytes_per_block(&self) -> usize {
        match self {
            GGUFQuantType::Q4_0 => 18, // 2 (scale) + 16 (data)
            GGUFQuantType::Q8_0 => 34, // 2 (scale) + 32 (data)
        }
    }

    /// Get bits per value
    pub fn bits(&self) -> usize {
        match self {
            GGUFQuantType::Q4_0 => 4,
            GGUFQuantType::Q8_0 => 8,
        }
    }

    /// Get theoretical compression ratio vs f32
    pub fn theoretical_compression(&self) -> f32 {
        32.0 / self.bits() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    // ========================================================================
    // PROPERTY TESTS - Bit packing correctness
    // ========================================================================

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(200))]

        /// Q4_0 round-trip should preserve values within quantization error
        #[test]
        fn prop_q4_0_round_trip(
            values in prop::collection::vec(-10.0f32..10.0, 32..128),
        ) {
            let quantized = Q4_0::quantize(&values);
            let dequantized = quantized.dequantize();

            prop_assert_eq!(dequantized.len(), values.len());

            // Quantization error should be bounded
            for (i, (&orig, &deq)) in values.iter().zip(dequantized.iter()).enumerate() {
                let error = (orig - deq).abs();
                let max_error = quantized.scales[i / GGUF_BLOCK_SIZE] * 1.5;
                prop_assert!(
                    error <= max_error,
                    "Q4_0 error {} > {} at index {}",
                    error, max_error, i
                );
            }
        }

        /// Q8_0 round-trip should preserve values within quantization error
        #[test]
        fn prop_q8_0_round_trip(
            values in prop::collection::vec(-10.0f32..10.0, 32..128),
        ) {
            let quantized = Q8_0::quantize(&values);
            let dequantized = quantized.dequantize();

            prop_assert_eq!(dequantized.len(), values.len());

            // Q8_0 should have smaller error than Q4_0
            for (i, (&orig, &deq)) in values.iter().zip(dequantized.iter()).enumerate() {
                let error = (orig - deq).abs();
                let max_error = quantized.scales[i / GGUF_BLOCK_SIZE] * 1.1;
                prop_assert!(
                    error <= max_error,
                    "Q8_0 error {} > {} at index {}",
                    error, max_error, i
                );
            }
        }

        /// Q4_0 should use correct number of blocks
        #[test]
        fn prop_q4_0_block_count(len in 1usize..256) {
            let values = vec![1.0f32; len];
            let quantized = Q4_0::quantize(&values);

            let expected_blocks = len.div_ceil(GGUF_BLOCK_SIZE);
            prop_assert_eq!(quantized.num_blocks(), expected_blocks);
            prop_assert_eq!(quantized.scales.len(), expected_blocks);
            prop_assert_eq!(quantized.data.len(), expected_blocks * 16);
        }

        /// Q8_0 should use correct number of blocks
        #[test]
        fn prop_q8_0_block_count(len in 1usize..256) {
            let values = vec![1.0f32; len];
            let quantized = Q8_0::quantize(&values);

            let expected_blocks = len.div_ceil(GGUF_BLOCK_SIZE);
            prop_assert_eq!(quantized.num_blocks(), expected_blocks);
            prop_assert_eq!(quantized.scales.len(), expected_blocks);
        }

        /// Q4_0 compression ratio should be close to 8x for large tensors
        #[test]
        fn prop_q4_0_compression(len in 256usize..1024) {
            let values = vec![1.0f32; len];
            let quantized = Q4_0::quantize(&values);

            let ratio = quantized.compression_ratio();
            // Q4_0: 4 bits/value + scale overhead → ~7x compression
            prop_assert!(ratio > 5.0, "Q4_0 compression {} should be > 5x", ratio);
            prop_assert!(ratio < 10.0, "Q4_0 compression {} should be < 10x", ratio);
        }

        /// Q8_0 compression ratio should be close to 4x for large tensors
        #[test]
        fn prop_q8_0_compression(len in 256usize..1024) {
            let values = vec![1.0f32; len];
            let quantized = Q8_0::quantize(&values);

            let ratio = quantized.compression_ratio();
            // Q8_0: 8 bits/value + scale overhead → ~3.5x compression
            prop_assert!(ratio > 3.0, "Q8_0 compression {} should be > 3x", ratio);
            prop_assert!(ratio < 5.0, "Q8_0 compression {} should be < 5x", ratio);
        }
    }

    // ========================================================================
    // UNIT TESTS
    // ========================================================================

    #[test]
    fn test_q4_0_basic() {
        let values = vec![0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1.5];
        let quantized = Q4_0::quantize(&values);
        let dequantized = quantized.dequantize();

        assert_eq!(dequantized.len(), values.len());

        // Check approximate reconstruction
        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            assert!(error < 1.0, "Error {} too large", error);
        }
    }

    #[test]
    fn test_q8_0_basic() {
        let values = vec![0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1.5];
        let quantized = Q8_0::quantize(&values);
        let dequantized = quantized.dequantize();

        assert_eq!(dequantized.len(), values.len());

        // Q8_0 should have better precision than Q4_0
        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            assert!(error < 0.1, "Error {} too large for Q8_0", error);
        }
    }

    #[test]
    fn test_q4_0_block_size() {
        // Exactly one block
        let values = vec![1.0; GGUF_BLOCK_SIZE];
        let quantized = Q4_0::quantize(&values);
        assert_eq!(quantized.num_blocks(), 1);
        assert_eq!(quantized.data.len(), 16);

        // Two blocks
        let values = vec![1.0; GGUF_BLOCK_SIZE + 1];
        let quantized = Q4_0::quantize(&values);
        assert_eq!(quantized.num_blocks(), 2);
        assert_eq!(quantized.data.len(), 32);
    }

    #[test]
    fn test_q8_0_block_size() {
        // Exactly one block
        let values = vec![1.0; GGUF_BLOCK_SIZE];
        let quantized = Q8_0::quantize(&values);
        assert_eq!(quantized.num_blocks(), 1);

        // Two blocks
        let values = vec![1.0; GGUF_BLOCK_SIZE + 1];
        let quantized = Q8_0::quantize(&values);
        assert_eq!(quantized.num_blocks(), 2);
    }

    #[test]
    fn test_q4_0_zeros() {
        let values = vec![0.0; 64];
        let quantized = Q4_0::quantize(&values);
        let dequantized = quantized.dequantize();

        for val in dequantized {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_q8_0_zeros() {
        let values = vec![0.0; 64];
        let quantized = Q8_0::quantize(&values);
        let dequantized = quantized.dequantize();

        for val in dequantized {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_gguf_quant_type() {
        assert_eq!(GGUFQuantType::Q4_0.bits(), 4);
        assert_eq!(GGUFQuantType::Q8_0.bits(), 8);

        assert_eq!(GGUFQuantType::Q4_0.bytes_per_block(), 18);
        assert_eq!(GGUFQuantType::Q8_0.bytes_per_block(), 34);

        assert_abs_diff_eq!(
            GGUFQuantType::Q4_0.theoretical_compression(),
            8.0,
            epsilon = 0.1
        );
        assert_abs_diff_eq!(
            GGUFQuantType::Q8_0.theoretical_compression(),
            4.0,
            epsilon = 0.1
        );
    }

    #[test]
    fn test_q4_0_memory_bytes() {
        let values = vec![1.0; 1024];
        let quantized = Q4_0::quantize(&values);

        // 1024 values = 32 blocks
        // GGUF bytes: 32 * 18 = 576 bytes
        assert_eq!(quantized.num_blocks(), 32);
        assert_eq!(quantized.gguf_bytes(), 32 * 18);

        let ratio = quantized.compression_ratio();
        assert!(ratio > 7.0, "Compression ratio {} should be > 7x", ratio);
    }

    #[test]
    fn test_q8_0_memory_bytes() {
        let values = vec![1.0; 1024];
        let quantized = Q8_0::quantize(&values);

        // 1024 values = 32 blocks
        // GGUF bytes: 32 * 34 = 1088 bytes
        assert_eq!(quantized.num_blocks(), 32);
        assert_eq!(quantized.gguf_bytes(), 32 * 34);

        let ratio = quantized.compression_ratio();
        assert!(ratio > 3.5, "Compression ratio {} should be > 3.5x", ratio);
    }

    #[test]
    fn test_q8_0_better_than_q4_0() {
        // Q8_0 should have lower error than Q4_0
        let values: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();

        let q4 = Q4_0::quantize(&values);
        let q8 = Q8_0::quantize(&values);

        let deq4 = q4.dequantize();
        let deq8 = q8.dequantize();

        let error4: f32 = values
            .iter()
            .zip(deq4.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let error8: f32 = values
            .iter()
            .zip(deq8.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(
            error8 < error4,
            "Q8_0 error {} should be < Q4_0 error {}",
            error8,
            error4
        );
    }

    #[test]
    fn test_q4_0_negative_values() {
        let values = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
        let quantized = Q4_0::quantize(&values);
        let dequantized = quantized.dequantize();

        for (&orig, &deq) in values.iter().zip(dequantized.iter()) {
            // Both should be negative
            assert!(deq < 0.0, "Expected negative, got {}", deq);
            // Error should be reasonable
            let error = (orig - deq).abs();
            assert!(error < 2.0, "Error {} too large", error);
        }
    }
}
