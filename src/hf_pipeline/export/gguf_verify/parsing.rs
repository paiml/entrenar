//! GGUF binary parsing helpers

use crate::hf_pipeline::error::FetchError;
use super::types::GgufTensorInfo;

/// Read a little-endian u32 from a byte slice at the given offset.
/// Caller must ensure `pos + 4 <= data.len()`.
pub(super) fn read_u32_le(data: &[u8], pos: usize) -> Result<u32, FetchError> {
    let bytes: [u8; 4] = data
        .get(pos..pos + 4)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| truncation_error(pos))?;
    Ok(u32::from_le_bytes(bytes))
}

/// Read a little-endian u64 from a byte slice at the given offset.
/// Caller must ensure `pos + 8 <= data.len()`.
pub(super) fn read_u64_le(data: &[u8], pos: usize) -> Result<u64, FetchError> {
    let bytes: [u8; 8] = data
        .get(pos..pos + 8)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| truncation_error(pos))?;
    Ok(u64::from_le_bytes(bytes))
}

/// Parsed GGUF header (magic, version, counts)
pub(super) struct GgufHeader {
    pub(super) version: u32,
    pub(super) tensor_count: u64,
    pub(super) metadata_count: u64,
}

/// Parse and validate the 24-byte GGUF header
pub(super) fn parse_header(data: &[u8]) -> Result<GgufHeader, FetchError> {
    if data.len() < 24 {
        return Err(FetchError::ConfigParseError {
            message: "GGUF file too small: less than 24 bytes".to_string(),
        });
    }

    let magic = data.get(0..4).unwrap_or_default();
    if magic != b"GGUF" {
        return Err(FetchError::ConfigParseError {
            message: format!(
                "Invalid GGUF magic: expected 'GGUF', got '{}'",
                String::from_utf8_lossy(magic)
            ),
        });
    }

    let version = read_u32_le(data, 4)?;
    if version != 3 {
        return Err(FetchError::ConfigParseError {
            message: format!("Unsupported GGUF version: {version} (expected 3)"),
        });
    }

    let tensor_count = read_u64_le(data, 8)?;
    let metadata_count = read_u64_le(data, 16)?;

    Ok(GgufHeader {
        version,
        tensor_count,
        metadata_count,
    })
}

/// Skip over all metadata key-value pairs, returning the position after them
pub(super) fn skip_all_metadata(
    data: &[u8],
    start: usize,
    count: u64,
) -> Result<usize, FetchError> {
    let mut pos = start;
    for _ in 0..count {
        pos = skip_gguf_string(data, pos)?;
        let value_type = read_u32_le(data, pos)?;
        pos += 4;
        pos = skip_gguf_value(data, pos, value_type)?;
    }
    Ok(pos)
}

/// Parse a single tensor info entry, returning (info, new_position)
pub(super) fn parse_tensor_info(
    data: &[u8],
    pos: usize,
) -> Result<(GgufTensorInfo, usize), FetchError> {
    let (name, mut pos) = read_gguf_string(data, pos)?;

    // n_dimensions
    let n_dims = read_u32_le(data, pos)?;
    pos += 4;

    // Dimensions
    let mut shape = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
        shape.push(read_u64_le(data, pos)?);
        pos += 8;
    }

    // dtype
    let dtype = read_u32_le(data, pos)?;
    pos += 4;

    // offset
    let offset = read_u64_le(data, pos)?;
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
pub(super) fn parse_all_tensor_info(
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
pub(super) fn read_gguf_string(
    data: &[u8],
    pos: usize,
) -> Result<(String, usize), FetchError> {
    let len = read_u64_le(data, pos)? as usize;
    let start = pos + 8;
    let end = start + len;
    if end > data.len() {
        return Err(truncation_error(start));
    }
    let s = String::from_utf8_lossy(&data[start..end]).to_string();
    Ok((s, end))
}

/// Skip over a GGUF string, returning the new position
pub(super) fn skip_gguf_string(data: &[u8], pos: usize) -> Result<usize, FetchError> {
    let (_, new_pos) = read_gguf_string(data, pos)?;
    Ok(new_pos)
}

/// Skip a GGUF value based on its type, returning the new position
pub(super) fn skip_gguf_value(
    data: &[u8],
    pos: usize,
    value_type: u32,
) -> Result<usize, FetchError> {
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
pub(super) fn skip_gguf_array(data: &[u8], pos: usize) -> Result<usize, FetchError> {
    let elem_type = read_u32_le(data, pos)?;
    let count = read_u64_le(data, pos + 4)?;
    let mut p = pos + 12;
    for _ in 0..count {
        p = skip_gguf_value(data, p, elem_type)?;
    }
    Ok(p)
}

/// Create a truncation error
pub(super) fn truncation_error(pos: usize) -> FetchError {
    FetchError::ConfigParseError {
        message: format!("GGUF file truncated at byte offset {pos}"),
    }
}
