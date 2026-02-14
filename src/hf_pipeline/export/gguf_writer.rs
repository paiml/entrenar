//! GGUF v3 binary writer
//!
//! Implements the GGUF format specification for writing model files compatible
//! with llama.cpp and other GGUF consumers.
//!
//! Format: magic(4) + version(4) + tensor_count(8) + metadata_count(8) +
//!         metadata_kv[] + tensor_info[] + padding + tensor_data[]

use crate::quant::{GGUF_BLOCK_SIZE, Q4_0, Q8_0};

/// GGUF file magic bytes
const GGUF_MAGIC: &[u8; 4] = b"GGUF";
/// GGUF format version 3
const GGUF_VERSION: u32 = 3;
/// Alignment for tensor data blocks
const GGUF_ALIGNMENT: usize = 32;

/// GGUF metadata value types
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum GgufMetadataValue {
    String(String),
    U32(u32),
    U64(u64),
    F32(f32),
}

/// GGUF tensor data types (matches ggml_type enum)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum GgufDtype {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q8_0 = 8,
}

/// GGUF quantization mode for export
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufQuantization {
    /// No quantization — store as F32
    None,
    /// Quantize to Q4_0 (4-bit, 32-element blocks)
    Q4_0,
    /// Quantize to Q8_0 (8-bit, 32-element blocks)
    Q8_0,
}

/// Tensor info recorded before data is written
struct TensorEntry {
    name: String,
    shape: Vec<usize>,
    dtype: GgufDtype,
    /// Data bytes (already quantized/converted)
    data: Vec<u8>,
}

/// GGUF v3 binary writer
///
/// Accumulates metadata KVs and tensors, then serializes to a complete GGUF file.
pub struct GgufWriter {
    metadata: Vec<(String, GgufMetadataValue)>,
    tensors: Vec<TensorEntry>,
}

impl GgufWriter {
    /// Create a new empty GGUF writer
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
        }
    }

    /// Add a string metadata key-value pair
    pub fn write_string(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata
            .push((key.into(), GgufMetadataValue::String(value.into())));
    }

    /// Add a metadata key-value pair
    pub fn write_metadata_kv(&mut self, key: impl Into<String>, value: GgufMetadataValue) {
        self.metadata.push((key.into(), value));
    }

    /// Add an F32 tensor
    pub fn write_tensor_data_f32(
        &mut self,
        name: impl Into<String>,
        data: &[f32],
        shape: Vec<usize>,
    ) {
        let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
        self.tensors.push(TensorEntry {
            name: name.into(),
            shape,
            dtype: GgufDtype::F32,
            data: bytes,
        });
    }

    /// Add a Q4_0 quantized tensor from f32 data
    pub fn write_tensor_data_q4_0(
        &mut self,
        name: impl Into<String>,
        data: &[f32],
        shape: Vec<usize>,
    ) {
        let quantized = Q4_0::quantize(data);
        let bytes = encode_q4_0_blocks(&quantized);
        self.tensors.push(TensorEntry {
            name: name.into(),
            shape,
            dtype: GgufDtype::Q4_0,
            data: bytes,
        });
    }

    /// Add a Q8_0 quantized tensor from f32 data
    pub fn write_tensor_data_q8_0(
        &mut self,
        name: impl Into<String>,
        data: &[f32],
        shape: Vec<usize>,
    ) {
        let quantized = Q8_0::quantize(data);
        let bytes = encode_q8_0_blocks(&quantized);
        self.tensors.push(TensorEntry {
            name: name.into(),
            shape,
            dtype: GgufDtype::Q8_0,
            data: bytes,
        });
    }

    /// Finalize and return the complete GGUF file bytes
    pub fn finalize(self) -> Vec<u8> {
        let mut buf = Vec::new();

        // --- Header ---
        buf.extend_from_slice(GGUF_MAGIC);
        buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());

        // --- Metadata KVs ---
        for (key, value) in &self.metadata {
            write_gguf_string(&mut buf, key);
            match value {
                GgufMetadataValue::String(s) => {
                    buf.extend_from_slice(&8u32.to_le_bytes()); // GGUF_TYPE_STRING
                    write_gguf_string(&mut buf, s);
                }
                GgufMetadataValue::U32(v) => {
                    buf.extend_from_slice(&4u32.to_le_bytes()); // GGUF_TYPE_UINT32
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                GgufMetadataValue::U64(v) => {
                    buf.extend_from_slice(&10u32.to_le_bytes()); // GGUF_TYPE_UINT64
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                GgufMetadataValue::F32(v) => {
                    buf.extend_from_slice(&6u32.to_le_bytes()); // GGUF_TYPE_FLOAT32
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
        }

        // --- Tensor Info ---
        // First compute where tensor data will start (after header + metadata + tensor_info + alignment)
        let tensor_info_size: usize = self
            .tensors
            .iter()
            .map(|t| {
                8 + t.name.len() // gguf_string (len + bytes)
                    + 4          // n_dimensions (u32)
                    + t.shape.len() * 8  // dimensions (u64 each)
                    + 4          // dtype (u32)
                    + 8 // offset (u64)
            })
            .sum();

        let header_plus_meta_size = buf.len() + tensor_info_size;
        let data_start = align_to(header_plus_meta_size, GGUF_ALIGNMENT);

        // Write tensor info entries with computed offsets
        let mut current_offset: u64 = 0;
        let mut tensor_offsets = Vec::with_capacity(self.tensors.len());

        for tensor in &self.tensors {
            write_gguf_string(&mut buf, &tensor.name);
            buf.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
            for &dim in &tensor.shape {
                buf.extend_from_slice(&(dim as u64).to_le_bytes());
            }
            buf.extend_from_slice(&(tensor.dtype as u32).to_le_bytes());
            buf.extend_from_slice(&current_offset.to_le_bytes());

            tensor_offsets.push(current_offset);
            let aligned_size = align_to(tensor.data.len(), GGUF_ALIGNMENT);
            current_offset += aligned_size as u64;
        }

        // --- Alignment padding before tensor data ---
        let padding = data_start - buf.len();
        buf.extend(std::iter::repeat_n(0u8, padding));

        // --- Tensor data blocks (32-byte aligned) ---
        for tensor in &self.tensors {
            buf.extend_from_slice(&tensor.data);
            let pad = align_to(tensor.data.len(), GGUF_ALIGNMENT) - tensor.data.len();
            buf.extend(std::iter::repeat_n(0u8, pad));
        }

        buf
    }
}

/// Round `n` up to the next multiple of `alignment`
fn align_to(n: usize, alignment: usize) -> usize {
    n.div_ceil(alignment) * alignment
}

/// Write a GGUF-format string: u64 length + raw bytes (no null terminator)
fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// Encode Q4_0 quantized data into GGUF binary block format
/// Each block: f16 scale (2 bytes) + 16 bytes packed 4-bit data = 18 bytes
fn encode_q4_0_blocks(q: &Q4_0) -> Vec<u8> {
    let num_blocks = q.num_blocks();
    let mut bytes = Vec::with_capacity(num_blocks * 18);

    for block_idx in 0..num_blocks {
        // Scale as f16
        let scale_f16 = half::f16::from_f32(q.scales[block_idx]);
        bytes.extend_from_slice(&scale_f16.to_le_bytes());

        // 16 bytes of packed 4-bit data
        let data_start = block_idx * 16;
        let data_end = (data_start + 16).min(q.data.len());
        bytes.extend_from_slice(&q.data[data_start..data_end]);

        // Pad if the last block is short
        let pad = 16 - (data_end - data_start);
        bytes.extend(std::iter::repeat_n(0u8, pad));
    }

    bytes
}

/// Encode Q8_0 quantized data into GGUF binary block format
/// Each block: f16 scale (2 bytes) + 32 bytes i8 data = 34 bytes
fn encode_q8_0_blocks(q: &Q8_0) -> Vec<u8> {
    let num_blocks = q.num_blocks();
    let mut bytes = Vec::with_capacity(num_blocks * 34);

    for block_idx in 0..num_blocks {
        // Scale as f16
        let scale_f16 = half::f16::from_f32(q.scales[block_idx]);
        bytes.extend_from_slice(&scale_f16.to_le_bytes());

        // 32 bytes of i8 data
        let data_start = block_idx * GGUF_BLOCK_SIZE;
        let data_end = (data_start + GGUF_BLOCK_SIZE).min(q.data.len());
        let block_data = &q.data[data_start..data_end];

        // Cast i8 slice to u8 slice for extending
        for &val in block_data {
            bytes.push(val as u8);
        }

        // Pad incomplete blocks
        let pad = GGUF_BLOCK_SIZE - block_data.len();
        bytes.extend(std::iter::repeat_n(0u8, pad));
    }

    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_empty_gguf() {
        let writer = GgufWriter::new();
        let data = writer.finalize();

        // Magic + version + tensor_count + metadata_count = 4+4+8+8 = 24
        assert!(data.len() >= 24);
        assert_eq!(&data[0..4], b"GGUF");
        assert_eq!(u32::from_le_bytes(data[4..8].try_into().unwrap()), 3);
        assert_eq!(u64::from_le_bytes(data[8..16].try_into().unwrap()), 0); // tensor count
        assert_eq!(u64::from_le_bytes(data[16..24].try_into().unwrap()), 0); // metadata count
    }

    #[test]
    fn test_metadata_string() {
        let mut writer = GgufWriter::new();
        writer.write_string("general.name", "test-model");
        let data = writer.finalize();

        // metadata count should be 1
        assert_eq!(u64::from_le_bytes(data[16..24].try_into().unwrap()), 1);
        // The file should contain our string
        let haystack = String::from_utf8_lossy(&data);
        assert!(haystack.contains("general.name"));
        assert!(haystack.contains("test-model"));
    }

    #[test]
    fn test_f32_tensor() {
        let mut writer = GgufWriter::new();
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        writer.write_tensor_data_f32("weights.0", &values, vec![2, 2]);
        let data = writer.finalize();

        // tensor count = 1
        assert_eq!(u64::from_le_bytes(data[8..16].try_into().unwrap()), 1);
        // file should contain the tensor data bytes
        assert!(data.len() > 24 + 16); // header + tensor data
    }

    #[test]
    fn test_q4_0_tensor() {
        let mut writer = GgufWriter::new();
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        writer.write_tensor_data_q4_0("quant_weights", &values, vec![64]);
        let data = writer.finalize();

        assert_eq!(u64::from_le_bytes(data[8..16].try_into().unwrap()), 1);
        // Q4_0: 64 elements = 2 blocks * 18 bytes = 36 bytes of quant data
        assert!(data.len() >= 36);
    }

    #[test]
    fn test_q8_0_tensor() {
        let mut writer = GgufWriter::new();
        let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        writer.write_tensor_data_q8_0("q8_weights", &values, vec![32]);
        let data = writer.finalize();

        assert_eq!(u64::from_le_bytes(data[8..16].try_into().unwrap()), 1);
    }

    #[test]
    fn test_multiple_tensors() {
        let mut writer = GgufWriter::new();
        writer.write_string("general.name", "multi");
        writer.write_tensor_data_f32("a", &[1.0, 2.0], vec![2]);
        writer.write_tensor_data_f32("b", &[3.0, 4.0, 5.0], vec![3]);
        let data = writer.finalize();

        assert_eq!(u64::from_le_bytes(data[8..16].try_into().unwrap()), 2);
        assert_eq!(u64::from_le_bytes(data[16..24].try_into().unwrap()), 1);
    }

    #[test]
    fn test_alignment() {
        // All tensor data offsets should be 32-byte aligned
        let mut writer = GgufWriter::new();
        writer.write_tensor_data_f32("small", &[1.0], vec![1]);
        writer.write_tensor_data_f32("another", &[2.0, 3.0], vec![2]);
        let data = writer.finalize();

        // File size should be 32-byte aligned at each tensor boundary
        assert_eq!(data.len() % GGUF_ALIGNMENT, 0);
    }

    #[test]
    fn test_metadata_u32() {
        let mut writer = GgufWriter::new();
        writer.write_metadata_kv("test.layers", GgufMetadataValue::U32(12));
        let data = writer.finalize();
        assert_eq!(u64::from_le_bytes(data[16..24].try_into().unwrap()), 1);
    }

    #[test]
    fn test_metadata_f32() {
        let mut writer = GgufWriter::new();
        writer.write_metadata_kv("test.loss", GgufMetadataValue::F32(0.5));
        let data = writer.finalize();
        assert_eq!(u64::from_le_bytes(data[16..24].try_into().unwrap()), 1);
    }

    #[test]
    fn test_metadata_u64() {
        let mut writer = GgufWriter::new();
        writer.write_metadata_kv("test.params", GgufMetadataValue::U64(7_000_000_000));
        let data = writer.finalize();
        assert_eq!(u64::from_le_bytes(data[16..24].try_into().unwrap()), 1);
    }

    #[test]
    fn test_encode_q4_0_block_size() {
        let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let q = Q4_0::quantize(&values);
        let bytes = encode_q4_0_blocks(&q);
        // 1 block * 18 bytes
        assert_eq!(bytes.len(), 18);
    }

    #[test]
    fn test_encode_q8_0_block_size() {
        let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let q = Q8_0::quantize(&values);
        let bytes = encode_q8_0_blocks(&q);
        // 1 block * 34 bytes
        assert_eq!(bytes.len(), 34);
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(50))]

        #[test]
        fn prop_gguf_always_starts_with_magic(
            n_tensors in 0usize..4,
            n_elements in 1usize..64,
        ) {
            let mut writer = GgufWriter::new();
            for i in 0..n_tensors {
                let data: Vec<f32> = (0..n_elements).map(|j| (i * n_elements + j) as f32).collect();
                writer.write_tensor_data_f32(&format!("t.{i}"), &data, vec![n_elements]);
            }
            let output = writer.finalize();
            prop_assert_eq!(&output[0..4], b"GGUF");
            prop_assert_eq!(u32::from_le_bytes(output[4..8].try_into().unwrap()), 3);
            prop_assert_eq!(
                u64::from_le_bytes(output[8..16].try_into().unwrap()),
                n_tensors as u64
            );
        }

        #[test]
        fn prop_gguf_file_size_32_aligned(
            n_elements in 1usize..128,
        ) {
            let mut writer = GgufWriter::new();
            let data: Vec<f32> = vec![0.5; n_elements];
            writer.write_tensor_data_f32("w", &data, vec![n_elements]);
            let output = writer.finalize();
            prop_assert_eq!(output.len() % GGUF_ALIGNMENT, 0);
        }

        #[test]
        fn prop_q4_0_encode_correct_block_count(
            n_elements in 1usize..256,
        ) {
            let data: Vec<f32> = vec![1.0; n_elements];
            let q = Q4_0::quantize(&data);
            let bytes = encode_q4_0_blocks(&q);
            let expected_blocks = n_elements.div_ceil(GGUF_BLOCK_SIZE);
            prop_assert_eq!(bytes.len(), expected_blocks * 18);
        }

        #[test]
        fn prop_q8_0_encode_correct_block_count(
            n_elements in 1usize..256,
        ) {
            let data: Vec<f32> = vec![1.0; n_elements];
            let q = Q8_0::quantize(&data);
            let bytes = encode_q8_0_blocks(&q);
            let expected_blocks = n_elements.div_ceil(GGUF_BLOCK_SIZE);
            prop_assert_eq!(bytes.len(), expected_blocks * 34);
        }
    }
}
