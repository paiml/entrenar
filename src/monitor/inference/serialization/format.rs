//! Trace format types and constants.

use serde::{Deserialize, Serialize};

/// Trace format for serialization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceFormat {
    /// APRT binary format (compact, fast)
    ///
    /// Header: 0x41505254 ("APRT"), version byte, path type
    Binary,

    /// Pretty-printed JSON
    Json,

    /// JSON Lines (one JSON per line)
    JsonLines,
}

/// Magic bytes for APRT binary format
pub const APRT_MAGIC: [u8; 4] = [0x41, 0x50, 0x52, 0x54]; // "APRT"

/// Current format version
pub const APRT_VERSION: u8 = 1;

/// Path type identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PathType {
    Linear = 0,
    Tree = 1,
    Forest = 2,
    KNN = 3,
    Neural = 4,
    Custom = 255,
}

impl From<u8> for PathType {
    fn from(value: u8) -> Self {
        match value {
            0 => PathType::Linear,
            1 => PathType::Tree,
            2 => PathType::Forest,
            3 => PathType::KNN,
            4 => PathType::Neural,
            _ => PathType::Custom,
        }
    }
}
