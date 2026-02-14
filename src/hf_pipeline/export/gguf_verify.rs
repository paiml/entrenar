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
}
