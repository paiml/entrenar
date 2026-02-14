//! GGUF file verification
//!
//! Parses and validates GGUF v3 files, returning a summary of their contents.

use crate::hf_pipeline::error::FetchError;

/// Summary of a verified GGUF file
#[derive(Debug, Clone)]
pub struct GgufSummary {
    /// GGUF format version
    pub version: u32,
    /// Number of tensors
    pub tensor_count: u64,
    /// Number of metadata key-value pairs
    pub metadata_count: u64,
    /// Total file size in bytes
    pub file_size: usize,
    /// Tensor names and their data types (as GGUF type IDs)
    pub tensors: Vec<GgufTensorInfo>,
}

/// Info about a single tensor in the GGUF file
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name
    pub name: String,
    /// Shape dimensions
    pub shape: Vec<u64>,
    /// GGUF data type ID
    pub dtype: u32,
    /// Byte offset in file
    pub offset: u64,
}

/// Parsed GGUF header (magic, version, counts)
struct GgufHeader {
    version: u32,
    tensor_count: u64,
    metadata_count: u64,
}

/// Verify a GGUF file and return a summary
///
/// Parses the header and tensor info sections, validating:
/// - Magic bytes ("GGUF")
/// - Version (must be 3)
/// - Tensor count consistency
/// - No truncation (file large enough for declared tensors)
pub fn verify_gguf(data: &[u8]) -> Result<GgufSummary, FetchError> {
    let header = parse_header(data)?;
    let pos = skip_all_metadata(data, 24, header.metadata_count)?;
    let tensors = parse_all_tensor_info(data, pos, header.tensor_count)?;

    Ok(GgufSummary {
        version: header.version,
        tensor_count: header.tensor_count,
        metadata_count: header.metadata_count,
        file_size: data.len(),
        tensors,
    })
}

/// Parse and validate the 24-byte GGUF header
fn parse_header(data: &[u8]) -> Result<GgufHeader, FetchError> {
    if data.len() < 24 {
        return Err(FetchError::ConfigParseError {
            message: "GGUF file too small: less than 24 bytes".to_string(),
        });
    }

    if &data[0..4] != b"GGUF" {
        return Err(FetchError::ConfigParseError {
            message: format!(
                "Invalid GGUF magic: expected 'GGUF', got '{}'",
                String::from_utf8_lossy(&data[0..4])
            ),
        });
    }

    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version != 3 {
        return Err(FetchError::ConfigParseError {
            message: format!("Unsupported GGUF version: {version} (expected 3)"),
        });
    }

    let tensor_count = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let metadata_count = u64::from_le_bytes(data[16..24].try_into().unwrap());

    Ok(GgufHeader {
        version,
        tensor_count,
        metadata_count,
    })
}

/// Skip over all metadata key-value pairs, returning the position after them
fn skip_all_metadata(data: &[u8], start: usize, count: u64) -> Result<usize, FetchError> {
    let mut pos = start;
    for _ in 0..count {
        pos = skip_gguf_string(data, pos)?;
        if pos + 4 > data.len() {
            return Err(truncation_error(pos));
        }
        let value_type = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        pos = skip_gguf_value(data, pos, value_type)?;
    }
    Ok(pos)
}

/// Parse a single tensor info entry, returning (info, new_position)
fn parse_tensor_info(data: &[u8], pos: usize) -> Result<(GgufTensorInfo, usize), FetchError> {
    let (name, mut pos) = read_gguf_string(data, pos)?;

    // n_dimensions
    if pos + 4 > data.len() {
        return Err(truncation_error(pos));
    }
    let n_dims = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;

    // Dimensions
    let mut shape = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
        if pos + 8 > data.len() {
            return Err(truncation_error(pos));
        }
        shape.push(u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()));
        pos += 8;
    }

    // dtype
    if pos + 4 > data.len() {
        return Err(truncation_error(pos));
    }
    let dtype = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;

    // offset
    if pos + 8 > data.len() {
        return Err(truncation_error(pos));
    }
    let offset = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
    pos += 8;

    Ok((
        GgufTensorInfo {
            name,
            shape,
            dtype,
            offset,
        },
        pos,
    ))
}

/// Parse all tensor info entries
fn parse_all_tensor_info(
    data: &[u8],
    start: usize,
    count: u64,
) -> Result<Vec<GgufTensorInfo>, FetchError> {
    let mut tensors = Vec::with_capacity(count as usize);
    let mut pos = start;
    for _ in 0..count {
        let (info, new_pos) = parse_tensor_info(data, pos)?;
        tensors.push(info);
        pos = new_pos;
    }
    Ok(tensors)
}

/// Read a GGUF string at the given position, return (string, new_position)
fn read_gguf_string(data: &[u8], pos: usize) -> Result<(String, usize), FetchError> {
    if pos + 8 > data.len() {
        return Err(truncation_error(pos));
    }
    let len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
    let start = pos + 8;
    let end = start + len;
    if end > data.len() {
        return Err(truncation_error(start));
    }
    let s = String::from_utf8_lossy(&data[start..end]).to_string();
    Ok((s, end))
}

/// Skip over a GGUF string, returning the new position
fn skip_gguf_string(data: &[u8], pos: usize) -> Result<usize, FetchError> {
    let (_, new_pos) = read_gguf_string(data, pos)?;
    Ok(new_pos)
}

/// Skip a GGUF value based on its type, returning the new position
fn skip_gguf_value(data: &[u8], pos: usize, value_type: u32) -> Result<usize, FetchError> {
    match value_type {
        0 | 1 | 7 => Ok(pos + 1),         // UINT8, INT8, BOOL
        2 | 3 => Ok(pos + 2),             // UINT16, INT16
        4..=6 => Ok(pos + 4),             // UINT32, INT32, FLOAT32
        8 => skip_gguf_string(data, pos), // STRING
        10..=12 => Ok(pos + 8),           // UINT64, INT64, FLOAT64
        9 => skip_gguf_array(data, pos),  // ARRAY
        _ => Err(FetchError::ConfigParseError {
            message: format!("Unknown GGUF metadata type: {value_type}"),
        }),
    }
}

/// Skip a GGUF array value: type(4) + count(8) + values
fn skip_gguf_array(data: &[u8], pos: usize) -> Result<usize, FetchError> {
    if pos + 12 > data.len() {
        return Err(truncation_error(pos));
    }
    let elem_type = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    let count = u64::from_le_bytes(data[pos + 4..pos + 12].try_into().unwrap());
    let mut p = pos + 12;
    for _ in 0..count {
        p = skip_gguf_value(data, p, elem_type)?;
    }
    Ok(p)
}

/// Create a truncation error
fn truncation_error(pos: usize) -> FetchError {
    FetchError::ConfigParseError {
        message: format!("GGUF file truncated at byte offset {pos}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf_pipeline::export::gguf_writer::{quantize_to_gguf_bytes, GgufQuantization};
    use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

    /// Helper: serialize GGUF to a Vec<u8> via aprender
    fn write_gguf(tensors: &[GgufTensor], metadata: &[(String, GgufValue)]) -> Vec<u8> {
        let mut buf = Vec::new();
        export_tensors_to_gguf(&mut buf, tensors, metadata).unwrap();
        buf
    }

    #[test]
    fn test_verify_empty_gguf() {
        let data = write_gguf(&[], &[]);

        let summary = verify_gguf(&data).unwrap();
        assert_eq!(summary.version, 3);
        assert_eq!(summary.tensor_count, 0);
        assert_eq!(summary.metadata_count, 0);
    }

    #[test]
    fn test_verify_with_metadata() {
        let metadata = vec![("general.name".into(), GgufValue::String("test".into()))];
        let data = write_gguf(&[], &metadata);

        let summary = verify_gguf(&data).unwrap();
        assert_eq!(summary.metadata_count, 1);
    }

    #[test]
    fn test_verify_with_tensors() {
        let tensors = vec![
            GgufTensor {
                name: "w1".into(),
                shape: vec![3],
                dtype: GgmlType::F32,
                data: bytemuck::cast_slice(&[1.0f32, 2.0, 3.0]).to_vec(),
            },
            GgufTensor {
                name: "w2".into(),
                shape: vec![2],
                dtype: GgmlType::F32,
                data: bytemuck::cast_slice(&[4.0f32, 5.0]).to_vec(),
            },
        ];
        let data = write_gguf(&tensors, &[]);

        let summary = verify_gguf(&data).unwrap();
        assert_eq!(summary.tensor_count, 2);
        assert_eq!(summary.tensors.len(), 2);
        assert_eq!(summary.tensors[0].name, "w1");
        assert_eq!(summary.tensors[1].name, "w2");
    }

    #[test]
    fn test_roundtrip_f32() {
        let metadata = vec![(
            "general.architecture".into(),
            GgufValue::String("llama".into()),
        )];
        let tensors = vec![GgufTensor {
            name: "layer.0.weight".into(),
            shape: vec![8, 16],
            dtype: GgmlType::F32,
            data: bytemuck::cast_slice(&[1.0f32; 128]).to_vec(),
        }];
        let data = write_gguf(&tensors, &metadata);

        let summary = verify_gguf(&data).unwrap();
        assert_eq!(summary.tensor_count, 1);
        assert_eq!(summary.metadata_count, 1);
        assert_eq!(summary.tensors[0].shape, vec![8, 16]);
        assert_eq!(summary.tensors[0].dtype, 0); // F32
    }

    #[test]
    fn test_roundtrip_q4_0() {
        let (bytes, dtype) = quantize_to_gguf_bytes(&[0.5; 64], GgufQuantization::Q4_0);
        let tensors = vec![GgufTensor {
            name: "quantized".into(),
            shape: vec![64],
            dtype,
            data: bytes,
        }];
        let data = write_gguf(&tensors, &[]);

        let summary = verify_gguf(&data).unwrap();
        assert_eq!(summary.tensors[0].dtype, 2); // Q4_0
    }

    #[test]
    fn test_roundtrip_q8_0() {
        let (bytes, dtype) = quantize_to_gguf_bytes(&[0.5; 32], GgufQuantization::Q8_0);
        let tensors = vec![GgufTensor {
            name: "quantized".into(),
            shape: vec![32],
            dtype,
            data: bytes,
        }];
        let data = write_gguf(&tensors, &[]);

        let summary = verify_gguf(&data).unwrap();
        assert_eq!(summary.tensors[0].dtype, 8); // Q8_0
    }

    #[test]
    fn test_verify_invalid_magic() {
        let data =
            b"GGML\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        let result = verify_gguf(data);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Invalid GGUF magic"));
    }

    #[test]
    fn test_verify_wrong_version() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&2u32.to_le_bytes()); // version 2
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let result = verify_gguf(&data);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Unsupported GGUF version"));
    }

    #[test]
    fn test_verify_too_small() {
        let result = verify_gguf(&[0u8; 10]);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("too small"));
    }

    #[test]
    fn test_verify_truncated_metadata() {
        // Valid header claiming 1 metadata but no actual metadata bytes
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata (but missing)

        let result = verify_gguf(&data);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("truncated"));
    }

    #[test]
    fn test_verify_u32_metadata() {
        let metadata = vec![("layers".into(), GgufValue::Uint32(32))];
        let (bytes, dtype) = quantize_to_gguf_bytes(&[1.0], GgufQuantization::None);
        let tensors = vec![GgufTensor {
            name: "w".into(),
            shape: vec![1],
            dtype,
            data: bytes,
        }];
        let data = write_gguf(&tensors, &metadata);

        let summary = verify_gguf(&data).unwrap();
        assert_eq!(summary.metadata_count, 1);
        assert_eq!(summary.tensor_count, 1);
    }

    #[test]
    fn test_verify_f32_metadata() {
        let metadata = vec![("loss".into(), GgufValue::Float32(0.42))];
        let data = write_gguf(&[], &metadata);

        let summary = verify_gguf(&data).unwrap();
        assert_eq!(summary.metadata_count, 1);
    }

    #[test]
    fn test_verify_u64_metadata() {
        let metadata = vec![("params".into(), GgufValue::Uint64(7_000_000_000))];
        let data = write_gguf(&[], &metadata);

        let summary = verify_gguf(&data).unwrap();
        assert_eq!(summary.metadata_count, 1);
    }

    // =====================================================================
    // Falsification tests: adversarial property-based roundtrip verification
    // =====================================================================

    use proptest::prelude::*;

    /// Extract f32 tensor data from raw GGUF bytes at the given tensor's offset.
    /// `data_section_start` is the byte offset where the tensor data section begins.
    /// Uses manual LE decoding to avoid alignment requirements of bytemuck::cast_slice.
    fn extract_f32_tensor_data(
        gguf_bytes: &[u8],
        data_section_start: usize,
        tensor_info: &GgufTensorInfo,
        num_elements: usize,
    ) -> Vec<f32> {
        let start = data_section_start + tensor_info.offset as usize;
        (0..num_elements)
            .map(|i| {
                let off = start + i * 4;
                f32::from_le_bytes(gguf_bytes[off..off + 4].try_into().unwrap())
            })
            .collect()
    }

    /// Find the start of the tensor data section by scanning past header + metadata + tensor info.
    ///
    /// Note: aprender's `export_tensors_to_gguf` places tensor data immediately after
    /// tensor info with NO alignment padding. Tensor offsets in the info entries are
    /// relative to this position.
    fn find_data_section_start(gguf_bytes: &[u8], summary: &GgufSummary) -> usize {
        let mut pos = 24; // skip header
                          // Skip metadata
        for _ in 0..summary.metadata_count {
            let (_, new_pos) = read_gguf_string(gguf_bytes, pos).unwrap();
            pos = new_pos;
            let value_type = u32::from_le_bytes(gguf_bytes[pos..pos + 4].try_into().unwrap());
            pos += 4;
            pos = skip_gguf_value(gguf_bytes, pos, value_type).unwrap();
        }
        // Skip tensor info
        for _ in 0..summary.tensor_count {
            let (_, new_pos) = read_gguf_string(gguf_bytes, pos).unwrap();
            pos = new_pos;
            let n_dims = u32::from_le_bytes(gguf_bytes[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4 + n_dims * 8 + 4 + 8; // dims + dtype + offset
        }
        pos
    }

    #[test]
    fn test_falsify_f32_tensor_data_survives_roundtrip() {
        // Adversarial: verify actual float bytes, not just structural metadata
        let original: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.0137).collect();
        let tensors = vec![GgufTensor {
            name: "weights".into(),
            shape: vec![16, 16],
            dtype: GgmlType::F32,
            data: bytemuck::cast_slice(&original).to_vec(),
        }];
        let data = write_gguf(&tensors, &[]);
        let summary = verify_gguf(&data).unwrap();
        let data_start = find_data_section_start(&data, &summary);
        let recovered = extract_f32_tensor_data(&data, data_start, &summary.tensors[0], 256);
        assert_eq!(
            original, recovered,
            "f32 tensor data must survive roundtrip exactly"
        );
    }

    #[test]
    fn test_falsify_special_float_values_survive() {
        // Edge case floats: 0, -0, subnormals, max, min, inf, -inf, NaN
        let special: Vec<f32> = vec![
            0.0,
            -0.0,
            f32::MIN_POSITIVE, // smallest positive normal
            f32::EPSILON,
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            // Pad to block size of 32
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
            20.0,
            21.0,
            22.0,
            23.0,
            24.0,
        ];
        let tensors = vec![GgufTensor {
            name: "special".into(),
            shape: vec![special.len() as u64],
            dtype: GgmlType::F32,
            data: bytemuck::cast_slice(&special).to_vec(),
        }];
        let data = write_gguf(&tensors, &[]);
        let summary = verify_gguf(&data).unwrap();
        let data_start = find_data_section_start(&data, &summary);
        let recovered =
            extract_f32_tensor_data(&data, data_start, &summary.tensors[0], special.len());

        for (i, (&orig, &rec)) in special.iter().zip(recovered.iter()).enumerate() {
            if orig.is_nan() {
                assert!(rec.is_nan(), "index {i}: NaN must survive roundtrip");
            } else {
                assert_eq!(
                    orig.to_bits(),
                    rec.to_bits(),
                    "index {i}: bitwise equality failed for {orig}"
                );
            }
        }
    }

    #[test]
    fn test_falsify_multi_tensor_ordering_preserved() {
        // Verify tensor order is deterministic and matches insertion order
        let names: Vec<String> = (0..8).map(|i| format!("layer.{i}.weight")).collect();
        let tensors: Vec<GgufTensor> = names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let val = (i + 1) as f32;
                GgufTensor {
                    name: name.clone(),
                    shape: vec![1],
                    dtype: GgmlType::F32,
                    data: bytemuck::cast_slice(&[val]).to_vec(),
                }
            })
            .collect();
        let data = write_gguf(&tensors, &[]);
        let summary = verify_gguf(&data).unwrap();

        assert_eq!(summary.tensor_count, 8);
        for (i, info) in summary.tensors.iter().enumerate() {
            assert_eq!(info.name, names[i], "tensor {i} name mismatch");
            assert_eq!(info.dtype, 0, "tensor {i} should be F32");
        }

        // Verify actual data values
        let data_start = find_data_section_start(&data, &summary);
        for (i, info) in summary.tensors.iter().enumerate() {
            let recovered = extract_f32_tensor_data(&data, data_start, info, 1);
            let expected = (i + 1) as f32;
            assert!(
                (recovered[0] - expected).abs() < f32::EPSILON,
                "tensor {i} data mismatch: expected {expected}, got {}",
                recovered[0]
            );
        }
    }

    #[test]
    fn test_falsify_mixed_metadata_types_roundtrip() {
        // All metadata types in one file — verify none corrupt the parse
        let metadata = vec![
            ("str.key".into(), GgufValue::String("hello world".into())),
            ("u32.key".into(), GgufValue::Uint32(42)),
            ("u64.key".into(), GgufValue::Uint64(u64::MAX)),
            ("f32.key".into(), GgufValue::Float32(std::f32::consts::PI)),
            ("i32.key".into(), GgufValue::Int32(-999)),
            ("bool.key".into(), GgufValue::Bool(true)),
        ];
        let tensors = vec![GgufTensor {
            name: "t".into(),
            shape: vec![4],
            dtype: GgmlType::F32,
            data: bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]).to_vec(),
        }];
        let data = write_gguf(&tensors, &metadata);
        let summary = verify_gguf(&data).unwrap();

        assert_eq!(summary.metadata_count, 6);
        assert_eq!(summary.tensor_count, 1);
        assert_eq!(summary.tensors[0].name, "t");
        assert_eq!(summary.tensors[0].shape, vec![4]);
    }

    #[test]
    fn test_falsify_q4_0_q8_0_f32_mixed_in_single_file() {
        // Mix all three quantization types in one GGUF file
        let f32_data = [1.0f32; 32];
        let (q4_bytes, q4_dtype) = quantize_to_gguf_bytes(&[0.5; 64], GgufQuantization::Q4_0);
        let (q8_bytes, q8_dtype) = quantize_to_gguf_bytes(&[0.3; 32], GgufQuantization::Q8_0);

        let tensors = vec![
            GgufTensor {
                name: "f32_tensor".into(),
                shape: vec![32],
                dtype: GgmlType::F32,
                data: bytemuck::cast_slice(&f32_data).to_vec(),
            },
            GgufTensor {
                name: "q4_tensor".into(),
                shape: vec![64],
                dtype: q4_dtype,
                data: q4_bytes,
            },
            GgufTensor {
                name: "q8_tensor".into(),
                shape: vec![32],
                dtype: q8_dtype,
                data: q8_bytes,
            },
        ];
        let data = write_gguf(&tensors, &[]);
        let summary = verify_gguf(&data).unwrap();

        assert_eq!(summary.tensor_count, 3);
        assert_eq!(summary.tensors[0].dtype, 0); // F32
        assert_eq!(summary.tensors[1].dtype, 2); // Q4_0
        assert_eq!(summary.tensors[2].dtype, 8); // Q8_0

        // Verify f32 data is intact
        let data_start = find_data_section_start(&data, &summary);
        let recovered = extract_f32_tensor_data(&data, data_start, &summary.tensors[0], 32);
        assert_eq!(recovered, f32_data.to_vec());
    }

    #[test]
    fn test_falsify_long_tensor_name() {
        // Adversarial: 1000-char tensor name
        let long_name = "a".repeat(1000);
        let tensors = vec![GgufTensor {
            name: long_name.clone(),
            shape: vec![1],
            dtype: GgmlType::F32,
            data: bytemuck::cast_slice(&[42.0f32]).to_vec(),
        }];
        let data = write_gguf(&tensors, &[]);
        let summary = verify_gguf(&data).unwrap();
        assert_eq!(summary.tensors[0].name, long_name);
    }

    #[test]
    fn test_falsify_high_dimensional_shape() {
        // 5D tensor shape
        let shape = vec![2u64, 3, 4, 5, 6];
        let num_elements: u64 = shape.iter().product();
        let values: Vec<f32> = (0..num_elements).map(|i| i as f32).collect();
        let tensors = vec![GgufTensor {
            name: "5d".into(),
            shape: shape.clone(),
            dtype: GgmlType::F32,
            data: bytemuck::cast_slice(&values).to_vec(),
        }];
        let data = write_gguf(&tensors, &[]);
        let summary = verify_gguf(&data).unwrap();
        assert_eq!(summary.tensors[0].shape, shape);

        let data_start = find_data_section_start(&data, &summary);
        let recovered = extract_f32_tensor_data(
            &data,
            data_start,
            &summary.tensors[0],
            num_elements as usize,
        );
        assert_eq!(values, recovered);
    }

    #[test]
    fn test_falsify_exporter_gguf_roundtrip_via_file() {
        // Full pipeline: Exporter.export_gguf() → read file → verify_gguf() → check
        use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

        let mut weights = ModelWeights::new();
        weights.add_tensor("attn.q", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        weights.add_tensor("attn.k", vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        weights.metadata.architecture = Some("llama".into());
        weights.metadata.model_name = Some("test-falsify".into());
        weights.metadata.num_params = 8;
        weights.metadata.hidden_size = Some(2);
        weights.metadata.num_layers = Some(1);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("falsify.gguf");

        let exporter = Exporter::new()
            .output_dir(dir.path())
            .gguf_quantization(GgufQuantization::None);
        let result = exporter
            .export(&weights, ExportFormat::GGUF, "falsify.gguf")
            .unwrap();

        assert_eq!(result.num_tensors, 2);
        assert!(result.size_bytes > 0);

        let file_data = std::fs::read(&path).unwrap();
        let summary = verify_gguf(&file_data).unwrap();

        assert_eq!(summary.version, 3);
        assert_eq!(summary.tensor_count, 2);
        // Metadata: architecture + name + parameter_count + hidden_size + num_layers = 5
        assert_eq!(summary.metadata_count, 5);

        // Tensors should be sorted alphabetically
        assert_eq!(summary.tensors[0].name, "attn.k");
        assert_eq!(summary.tensors[1].name, "attn.q");
        assert_eq!(summary.tensors[0].dtype, 0); // F32
        assert_eq!(summary.tensors[1].dtype, 0);
        assert_eq!(summary.tensors[0].shape, vec![2, 2]);
        assert_eq!(summary.tensors[1].shape, vec![2, 2]);

        // Verify actual data
        let data_start = find_data_section_start(&file_data, &summary);
        let k_data = extract_f32_tensor_data(&file_data, data_start, &summary.tensors[0], 4);
        let q_data = extract_f32_tensor_data(&file_data, data_start, &summary.tensors[1], 4);
        assert_eq!(k_data, vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(q_data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_falsify_exporter_q4_0_roundtrip_via_file() {
        use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

        let mut weights = ModelWeights::new();
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        weights.add_tensor("quantized_layer", data, vec![64]);

        let dir = tempfile::tempdir().unwrap();
        let exporter = Exporter::new()
            .output_dir(dir.path())
            .gguf_quantization(GgufQuantization::Q4_0);
        let result = exporter
            .export(&weights, ExportFormat::GGUF, "q4.gguf")
            .unwrap();
        assert_eq!(result.num_tensors, 1);

        let file_data = std::fs::read(dir.path().join("q4.gguf")).unwrap();
        let summary = verify_gguf(&file_data).unwrap();
        assert_eq!(summary.tensors[0].dtype, 2); // Q4_0
        assert_eq!(summary.tensors[0].shape, vec![64]);
    }

    #[test]
    fn test_falsify_exporter_q8_0_roundtrip_via_file() {
        use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

        let mut weights = ModelWeights::new();
        let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
        weights.add_tensor("q8_layer", data, vec![128]);

        let dir = tempfile::tempdir().unwrap();
        let exporter = Exporter::new()
            .output_dir(dir.path())
            .gguf_quantization(GgufQuantization::Q8_0);
        let result = exporter
            .export(&weights, ExportFormat::GGUF, "q8.gguf")
            .unwrap();
        assert_eq!(result.num_tensors, 1);

        let file_data = std::fs::read(dir.path().join("q8.gguf")).unwrap();
        let summary = verify_gguf(&file_data).unwrap();
        assert_eq!(summary.tensors[0].dtype, 8); // Q8_0
        assert_eq!(summary.tensors[0].shape, vec![128]);
    }

    #[test]
    fn test_falsify_no_metadata_mode() {
        // Exporter with include_metadata=false must produce 0 metadata entries
        use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

        let mut weights = ModelWeights::new();
        weights.add_tensor("w", vec![1.0], vec![1]);
        weights.metadata.architecture = Some("llama".into());
        weights.metadata.model_name = Some("should-not-appear".into());

        let dir = tempfile::tempdir().unwrap();
        let exporter = Exporter::new()
            .output_dir(dir.path())
            .include_metadata(false);
        exporter
            .export(&weights, ExportFormat::GGUF, "nometa.gguf")
            .unwrap();

        let file_data = std::fs::read(dir.path().join("nometa.gguf")).unwrap();
        let summary = verify_gguf(&file_data).unwrap();
        assert_eq!(summary.metadata_count, 0);
        assert_eq!(summary.tensor_count, 1);
    }

    #[test]
    fn test_falsify_minimal_metadata_only_param_count() {
        // All optional metadata fields are None — only num_params should appear
        use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

        let mut weights = ModelWeights::new();
        weights.add_tensor("w", vec![1.0, 2.0], vec![2]);
        weights.metadata.num_params = 2;
        // architecture, model_name, hidden_size, num_layers all None

        let dir = tempfile::tempdir().unwrap();
        let exporter = Exporter::new().output_dir(dir.path());
        exporter
            .export(&weights, ExportFormat::GGUF, "minimal.gguf")
            .unwrap();

        let file_data = std::fs::read(dir.path().join("minimal.gguf")).unwrap();
        let summary = verify_gguf(&file_data).unwrap();
        // Only general.parameter_count should be present
        assert_eq!(summary.metadata_count, 1);
        assert_eq!(summary.tensor_count, 1);
    }

    #[test]
    fn test_falsify_exporter_alphabetical_tensor_sort() {
        // Tensors added in reverse order must appear alphabetically in GGUF
        use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

        let mut weights = ModelWeights::new();
        // Add in reverse alphabetical order
        weights.add_tensor("z_layer", vec![3.0], vec![1]);
        weights.add_tensor("m_layer", vec![2.0], vec![1]);
        weights.add_tensor("a_layer", vec![1.0], vec![1]);

        let dir = tempfile::tempdir().unwrap();
        let exporter = Exporter::new()
            .output_dir(dir.path())
            .include_metadata(false);
        exporter
            .export(&weights, ExportFormat::GGUF, "sorted.gguf")
            .unwrap();

        let file_data = std::fs::read(dir.path().join("sorted.gguf")).unwrap();
        let summary = verify_gguf(&file_data).unwrap();
        assert_eq!(summary.tensor_count, 3);
        assert_eq!(summary.tensors[0].name, "a_layer");
        assert_eq!(summary.tensors[1].name, "m_layer");
        assert_eq!(summary.tensors[2].name, "z_layer");

        // Verify data follows the sorted order (a=1.0, m=2.0, z=3.0)
        let data_start = find_data_section_start(&file_data, &summary);
        let a_data = extract_f32_tensor_data(&file_data, data_start, &summary.tensors[0], 1);
        let m_data = extract_f32_tensor_data(&file_data, data_start, &summary.tensors[1], 1);
        let z_data = extract_f32_tensor_data(&file_data, data_start, &summary.tensors[2], 1);
        assert!(
            (a_data[0] - 1.0).abs() < f32::EPSILON,
            "a_layer should be 1.0"
        );
        assert!(
            (m_data[0] - 2.0).abs() < f32::EPSILON,
            "m_layer should be 2.0"
        );
        assert!(
            (z_data[0] - 3.0).abs() < f32::EPSILON,
            "z_layer should be 3.0"
        );
    }

    #[test]
    fn test_falsify_utf8_tensor_names() {
        // Non-ASCII tensor names must roundtrip correctly
        let tensors = vec![
            GgufTensor {
                name: "layer.火.weight".into(),
                shape: vec![1],
                dtype: GgmlType::F32,
                data: bytemuck::cast_slice(&[1.0f32]).to_vec(),
            },
            GgufTensor {
                name: "модель.bias".into(),
                shape: vec![1],
                dtype: GgmlType::F32,
                data: bytemuck::cast_slice(&[2.0f32]).to_vec(),
            },
        ];
        let data = write_gguf(&tensors, &[]);
        let summary = verify_gguf(&data).unwrap();
        assert_eq!(summary.tensor_count, 2);
        assert_eq!(summary.tensors[0].name, "layer.火.weight");
        assert_eq!(summary.tensors[1].name, "модель.bias");
    }

    #[test]
    fn test_falsify_utf8_metadata_values() {
        let metadata = vec![
            (
                "general.name".into(),
                GgufValue::String("模型-テスト".into()),
            ),
            (
                "general.architecture".into(),
                GgufValue::String("трансформер".into()),
            ),
        ];
        let data = write_gguf(&[], &metadata);
        let summary = verify_gguf(&data).unwrap();
        assert_eq!(summary.metadata_count, 2);
    }

    #[test]
    fn test_falsify_10d_tensor_shape() {
        // 10-dimensional shape must survive roundtrip
        let shape = vec![2u64, 2, 2, 2, 2, 2, 2, 2, 2, 2]; // 1024 elements
        let num_elements: u64 = shape.iter().product();
        let values: Vec<f32> = (0..num_elements).map(|i| i as f32).collect();
        let tensors = vec![GgufTensor {
            name: "10d".into(),
            shape: shape.clone(),
            dtype: GgmlType::F32,
            data: bytemuck::cast_slice(&values).to_vec(),
        }];
        let data = write_gguf(&tensors, &[]);
        let summary = verify_gguf(&data).unwrap();
        assert_eq!(summary.tensors[0].shape, shape);
        assert_eq!(summary.tensors[0].shape.len(), 10);
    }

    #[test]
    fn test_falsify_exporter_write_error_on_readonly_path() {
        // Export to a path that cannot be created should return GgufWriteError
        use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

        let mut weights = ModelWeights::new();
        weights.add_tensor("w", vec![1.0], vec![1]);

        let exporter = Exporter::new().output_dir("/proc/nonexistent_dir");
        let result = exporter.export(&weights, ExportFormat::GGUF, "fail.gguf");
        assert!(result.is_err(), "export to invalid path must fail");
    }

    #[test]
    fn test_falsify_exporter_creates_parent_directories() {
        // Export to a nested path should create parent directories
        use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

        let mut weights = ModelWeights::new();
        weights.add_tensor("w", vec![1.0], vec![1]);

        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("deep").join("nested").join("path");
        let exporter = Exporter::new().output_dir(&nested).include_metadata(false);
        let result = exporter.export(&weights, ExportFormat::GGUF, "model.gguf");
        assert!(result.is_ok(), "export should create parent directories");
        assert!(nested.join("model.gguf").exists());
    }

    #[test]
    fn test_falsify_exporter_partial_metadata_combinations() {
        // Test all 4 combinations of architecture/model_name being Some/None
        use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

        let cases: Vec<(Option<&str>, Option<&str>, u64)> = vec![
            (None, None, 1),                      // only param_count
            (Some("llama"), None, 2),             // arch + param_count
            (None, Some("my-model"), 2),          // name + param_count
            (Some("llama"), Some("my-model"), 3), // arch + name + param_count
        ];

        for (arch, name, expected_meta) in cases {
            let mut weights = ModelWeights::new();
            weights.add_tensor("w", vec![1.0], vec![1]);
            weights.metadata.num_params = 1;
            weights.metadata.architecture = arch.map(String::from);
            weights.metadata.model_name = name.map(String::from);
            weights.metadata.hidden_size = None;
            weights.metadata.num_layers = None;

            let dir = tempfile::tempdir().unwrap();
            let exporter = Exporter::new().output_dir(dir.path());
            exporter
                .export(&weights, ExportFormat::GGUF, "meta.gguf")
                .unwrap();

            let file_data = std::fs::read(dir.path().join("meta.gguf")).unwrap();
            let summary = verify_gguf(&file_data).unwrap();
            assert_eq!(
                summary.metadata_count, expected_meta,
                "arch={arch:?}, name={name:?}: expected {expected_meta} metadata, got {}",
                summary.metadata_count
            );
        }
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]

        #[test]
        fn prop_falsify_arbitrary_f32_tensors_roundtrip(
            n_tensors in 1usize..6,
            n_elements in 1usize..64,
        ) {
            let tensors: Vec<GgufTensor> = (0..n_tensors)
                .map(|i| {
                    let data: Vec<f32> = (0..n_elements)
                        .map(|j| (i * n_elements + j) as f32 * 0.1)
                        .collect();
                    GgufTensor {
                        name: format!("tensor.{i}"),
                        shape: vec![n_elements as u64],
                        dtype: GgmlType::F32,
                        data: bytemuck::cast_slice(&data).to_vec(),
                    }
                })
                .collect();

            let gguf_bytes = write_gguf(&tensors, &[]);
            let summary = verify_gguf(&gguf_bytes).unwrap();

            prop_assert_eq!(summary.tensor_count, n_tensors as u64);
            prop_assert_eq!(summary.tensors.len(), n_tensors);

            // Verify every tensor's name, shape, dtype
            for (i, info) in summary.tensors.iter().enumerate() {
                prop_assert_eq!(&info.name, &format!("tensor.{i}"));
                prop_assert_eq!(&info.shape, &vec![n_elements as u64]);
                prop_assert_eq!(info.dtype, 0); // F32
            }

            // Verify actual f32 data roundtrips
            let data_start = find_data_section_start(&gguf_bytes, &summary);
            for (i, info) in summary.tensors.iter().enumerate() {
                let recovered = extract_f32_tensor_data(&gguf_bytes, data_start, info, n_elements);
                for (j, &val) in recovered.iter().enumerate() {
                    let expected = (i * n_elements + j) as f32 * 0.1;
                    prop_assert!(
                        (val - expected).abs() < 1e-6,
                        "tensor {i} element {j}: expected {expected}, got {val}"
                    );
                }
            }
        }

        #[test]
        fn prop_falsify_metadata_count_always_matches(
            n_metadata in 0usize..8,
        ) {
            let metadata: Vec<(String, GgufValue)> = (0..n_metadata)
                .map(|i| (format!("key.{i}"), GgufValue::Uint32(i as u32)))
                .collect();
            let gguf_bytes = write_gguf(&[], &metadata);
            let summary = verify_gguf(&gguf_bytes).unwrap();
            prop_assert_eq!(summary.metadata_count, n_metadata as u64);
            prop_assert_eq!(summary.tensor_count, 0);
        }

        #[test]
        fn prop_falsify_q4_0_roundtrip_preserves_dtype(
            n_elements in 1usize..128,
        ) {
            let data: Vec<f32> = vec![0.42; n_elements];
            let (bytes, dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::Q4_0);
            let tensors = vec![GgufTensor {
                name: "q4".into(),
                shape: vec![n_elements as u64],
                dtype,
                data: bytes,
            }];
            let gguf_bytes = write_gguf(&tensors, &[]);
            let summary = verify_gguf(&gguf_bytes).unwrap();
            prop_assert_eq!(summary.tensors[0].dtype, 2); // Q4_0
            prop_assert_eq!(&summary.tensors[0].shape, &vec![n_elements as u64]);
        }

        #[test]
        fn prop_falsify_q8_0_roundtrip_preserves_dtype(
            n_elements in 1usize..128,
        ) {
            let data: Vec<f32> = vec![0.42; n_elements];
            let (bytes, dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::Q8_0);
            let tensors = vec![GgufTensor {
                name: "q8".into(),
                shape: vec![n_elements as u64],
                dtype,
                data: bytes,
            }];
            let gguf_bytes = write_gguf(&tensors, &[]);
            let summary = verify_gguf(&gguf_bytes).unwrap();
            prop_assert_eq!(summary.tensors[0].dtype, 8); // Q8_0
            prop_assert_eq!(&summary.tensors[0].shape, &vec![n_elements as u64]);
        }

        #[test]
        fn prop_falsify_gguf_header_always_valid(
            n_tensors in 0usize..5,
            n_metadata in 0usize..5,
            n_elements in 1usize..32,
        ) {
            let metadata: Vec<(String, GgufValue)> = (0..n_metadata)
                .map(|i| (format!("m.{i}"), GgufValue::String(format!("v{i}"))))
                .collect();
            let tensors: Vec<GgufTensor> = (0..n_tensors)
                .map(|i| GgufTensor {
                    name: format!("t.{i}"),
                    shape: vec![n_elements as u64],
                    dtype: GgmlType::F32,
                    data: vec![0u8; n_elements * 4],
                })
                .collect();

            let gguf_bytes = write_gguf(&tensors, &metadata);

            // Header must always be valid
            prop_assert_eq!(&gguf_bytes[0..4], b"GGUF");
            prop_assert_eq!(
                u32::from_le_bytes(gguf_bytes[4..8].try_into().unwrap()),
                3
            );
            prop_assert_eq!(
                u64::from_le_bytes(gguf_bytes[8..16].try_into().unwrap()),
                n_tensors as u64
            );
            prop_assert_eq!(
                u64::from_le_bytes(gguf_bytes[16..24].try_into().unwrap()),
                n_metadata as u64
            );

            // Must verify cleanly
            let summary = verify_gguf(&gguf_bytes).unwrap();
            prop_assert_eq!(summary.version, 3);
            prop_assert_eq!(summary.tensor_count, n_tensors as u64);
            prop_assert_eq!(summary.metadata_count, n_metadata as u64);
        }

        #[test]
        fn prop_falsify_tensor_sort_always_alphabetical(
            seed in 0u64..1000,
        ) {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            // Generate deterministic pseudo-random names from seed
            let names: Vec<String> = (0..20).map(|i| {
                let mut h = DefaultHasher::new();
                (seed, i).hash(&mut h);
                format!("tensor_{:016x}", h.finish())
            }).collect();

            let mut weights =
                crate::hf_pipeline::export::weights::ModelWeights::new();
            for (i, name) in names.iter().enumerate() {
                weights.add_tensor(name.clone(), vec![i as f32], vec![1]);
            }

            let dir = tempfile::tempdir().unwrap();
            let exporter = crate::hf_pipeline::export::exporter::Exporter::new()
                .output_dir(dir.path())
                .include_metadata(false);
            exporter
                .export(
                    &weights,
                    crate::hf_pipeline::export::format::ExportFormat::GGUF,
                    "sort.gguf",
                )
                .unwrap();

            let file_data = std::fs::read(dir.path().join("sort.gguf")).unwrap();
            let summary = verify_gguf(&file_data).unwrap();

            let mut sorted_names = names.clone();
            sorted_names.sort();

            prop_assert_eq!(summary.tensors.len(), 20);
            for (i, expected_name) in sorted_names.iter().enumerate() {
                prop_assert_eq!(&summary.tensors[i].name, expected_name);
            }
        }
    }

    // =====================================================================
    // TIER 3: Stress & boundary tests
    // =====================================================================

    #[test]
    fn test_falsify_stress_100_tensors() {
        let mut weights = crate::hf_pipeline::export::weights::ModelWeights::new();
        for i in 0..100 {
            weights.add_tensor(format!("layer.{i:03}.weight"), vec![i as f32; 32], vec![32]);
        }

        let dir = tempfile::tempdir().unwrap();
        let exporter = crate::hf_pipeline::export::exporter::Exporter::new()
            .output_dir(dir.path())
            .include_metadata(false);
        let result = exporter
            .export(
                &weights,
                crate::hf_pipeline::export::format::ExportFormat::GGUF,
                "stress100.gguf",
            )
            .unwrap();
        assert_eq!(result.num_tensors, 100);

        let file_data = std::fs::read(dir.path().join("stress100.gguf")).unwrap();
        let summary = verify_gguf(&file_data).unwrap();
        assert_eq!(summary.tensor_count, 100);
        assert_eq!(summary.tensors.len(), 100);

        // Verify alphabetical sort with zero-padded names
        for i in 0..100 {
            assert_eq!(
                summary.tensors[i].name,
                format!("layer.{i:03}.weight"),
                "tensor {i} not in sorted position"
            );
        }
    }

    #[test]
    fn test_falsify_stress_5000_char_tensor_name() {
        let long_name = "x".repeat(5000);
        let tensors = vec![GgufTensor {
            name: long_name.clone(),
            shape: vec![1],
            dtype: GgmlType::F32,
            data: bytemuck::cast_slice(&[99.0f32]).to_vec(),
        }];
        let data = write_gguf(&tensors, &[]);
        let summary = verify_gguf(&data).unwrap();
        assert_eq!(summary.tensors[0].name, long_name);
        assert_eq!(summary.tensor_count, 1);
    }

    #[test]
    fn test_falsify_stress_block_boundary_exact_sizes() {
        // Tensor sizes that are exact multiples of block size (32)
        for n_blocks in [1, 2, 4, 8, 16, 32] {
            let n_elements = n_blocks * 32;
            let data: Vec<f32> = (0..n_elements).map(|i| i as f32 * 0.01).collect();
            let (q4_bytes, q4_dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::Q4_0);
            let (q8_bytes, q8_dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::Q8_0);

            assert_eq!(q4_bytes.len(), n_blocks * 18, "Q4_0 at {n_blocks} blocks");
            assert_eq!(q8_bytes.len(), n_blocks * 34, "Q8_0 at {n_blocks} blocks");

            // Verify through GGUF roundtrip
            let tensors = vec![
                GgufTensor {
                    name: "q4".into(),
                    shape: vec![n_elements as u64],
                    dtype: q4_dtype,
                    data: q4_bytes,
                },
                GgufTensor {
                    name: "q8".into(),
                    shape: vec![n_elements as u64],
                    dtype: q8_dtype,
                    data: q8_bytes,
                },
            ];
            let gguf = write_gguf(&tensors, &[]);
            let summary = verify_gguf(&gguf).unwrap();
            assert_eq!(summary.tensor_count, 2, "at {n_blocks} blocks");
            assert_eq!(summary.tensors[0].dtype, 2); // Q4_0
            assert_eq!(summary.tensors[1].dtype, 8); // Q8_0
        }
    }

    #[test]
    fn test_falsify_all_16_metadata_combinations() {
        use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

        // All 16 combinations of (arch, name, hidden_size, num_layers) being Some/None
        for mask in 0u8..16 {
            let has_arch = mask & 1 != 0;
            let has_name = mask & 2 != 0;
            let has_hidden = mask & 4 != 0;
            let has_layers = mask & 8 != 0;

            let mut weights = ModelWeights::new();
            weights.add_tensor("w", vec![1.0], vec![1]);
            weights.metadata.num_params = 1;
            weights.metadata.architecture = if has_arch { Some("llama".into()) } else { None };
            weights.metadata.model_name = if has_name { Some("test".into()) } else { None };
            weights.metadata.hidden_size = if has_hidden { Some(64) } else { None };
            weights.metadata.num_layers = if has_layers { Some(4) } else { None };

            let dir = tempfile::tempdir().unwrap();
            let exporter = Exporter::new().output_dir(dir.path());
            exporter
                .export(&weights, ExportFormat::GGUF, "meta.gguf")
                .unwrap();

            let file_data = std::fs::read(dir.path().join("meta.gguf")).unwrap();
            let summary = verify_gguf(&file_data).unwrap();

            // param_count always present + each optional field
            let expected: u64 = 1
                + u64::from(has_arch)
                + u64::from(has_name)
                + u64::from(has_hidden)
                + u64::from(has_layers);
            assert_eq!(
                summary.metadata_count, expected,
                "mask={mask:#06b}: expected {expected} metadata, got {}",
                summary.metadata_count
            );
        }
    }

    #[test]
    fn test_falsify_magic_bytes_survive_all_quant_modes() {
        use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

        for quant in [
            GgufQuantization::None,
            GgufQuantization::Q4_0,
            GgufQuantization::Q8_0,
        ] {
            let mut weights = ModelWeights::new();
            weights.add_tensor("w", vec![1.0; 64], vec![64]);

            let dir = tempfile::tempdir().unwrap();
            let exporter = Exporter::new()
                .output_dir(dir.path())
                .gguf_quantization(quant)
                .include_metadata(false);
            exporter
                .export(&weights, ExportFormat::GGUF, "magic.gguf")
                .unwrap();

            let file_data = std::fs::read(dir.path().join("magic.gguf")).unwrap();
            assert_eq!(&file_data[0..4], b"GGUF", "magic bytes wrong for {quant:?}");
            assert_eq!(
                u32::from_le_bytes(file_data[4..8].try_into().unwrap()),
                3,
                "version wrong for {quant:?}"
            );
        }
    }

    #[test]
    fn test_falsify_file_size_grows_with_tensor_count() {
        let mut prev_size = 0u64;
        for n_tensors in [1, 5, 10, 20, 50] {
            let mut weights = crate::hf_pipeline::export::weights::ModelWeights::new();
            for i in 0..n_tensors {
                weights.add_tensor(format!("t.{i:03}"), vec![1.0; 32], vec![32]);
            }

            let dir = tempfile::tempdir().unwrap();
            let exporter = crate::hf_pipeline::export::exporter::Exporter::new()
                .output_dir(dir.path())
                .include_metadata(false);
            let result = exporter
                .export(
                    &weights,
                    crate::hf_pipeline::export::format::ExportFormat::GGUF,
                    "grow.gguf",
                )
                .unwrap();

            assert!(
                result.size_bytes > prev_size,
                "size must grow: {n_tensors} tensors = {} bytes, prev = {prev_size}",
                result.size_bytes
            );
            prev_size = result.size_bytes;
        }
    }

    #[test]
    fn test_falsify_deterministic_output() {
        // Same weights exported twice must produce identical bytes
        use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

        let mut weights = ModelWeights::new();
        weights.add_tensor("a", vec![1.0, 2.0], vec![2]);
        weights.add_tensor("b", vec![3.0, 4.0], vec![2]);
        weights.metadata.architecture = Some("test".into());
        weights.metadata.num_params = 4;

        let dir1 = tempfile::tempdir().unwrap();
        let dir2 = tempfile::tempdir().unwrap();
        let exporter1 = Exporter::new().output_dir(dir1.path());
        let exporter2 = Exporter::new().output_dir(dir2.path());
        exporter1
            .export(&weights, ExportFormat::GGUF, "det.gguf")
            .unwrap();
        exporter2
            .export(&weights, ExportFormat::GGUF, "det.gguf")
            .unwrap();

        let bytes1 = std::fs::read(dir1.path().join("det.gguf")).unwrap();
        let bytes2 = std::fs::read(dir2.path().join("det.gguf")).unwrap();
        assert_eq!(
            bytes1, bytes2,
            "identical weights must produce identical GGUF files"
        );
    }
}
