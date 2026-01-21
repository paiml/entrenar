//! Decision Path Trait and Error Types

use serde::{Deserialize, Serialize};

/// Common interface for all decision paths
pub trait DecisionPath: Clone + Send + Sync + 'static {
    /// Human-readable explanation
    fn explain(&self) -> String;

    /// Feature importance scores (contribution of each feature)
    fn feature_contributions(&self) -> &[f32];

    /// Confidence in this decision (0.0 - 1.0)
    fn confidence(&self) -> f32;

    /// Compact binary representation
    fn to_bytes(&self) -> Vec<u8>;

    /// Reconstruct from binary representation
    fn from_bytes(bytes: &[u8]) -> Result<Self, PathError>
    where
        Self: Sized;
}

/// Error type for path operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PathError {
    /// Invalid binary format
    InvalidFormat(String),
    /// Insufficient data
    InsufficientData { expected: usize, actual: usize },
    /// Version mismatch
    VersionMismatch { expected: u8, actual: u8 },
}

impl std::fmt::Display for PathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PathError::InvalidFormat(msg) => write!(f, "Invalid format: {msg}"),
            PathError::InsufficientData { expected, actual } => {
                write!(f, "Insufficient data: expected {expected}, got {actual}")
            }
            PathError::VersionMismatch { expected, actual } => {
                write!(f, "Version mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for PathError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_error_display_invalid_format() {
        let err = PathError::InvalidFormat("test error".to_string());
        assert_eq!(format!("{err}"), "Invalid format: test error");
    }

    #[test]
    fn test_path_error_display_insufficient_data() {
        let err = PathError::InsufficientData {
            expected: 100,
            actual: 50,
        };
        assert_eq!(format!("{err}"), "Insufficient data: expected 100, got 50");
    }

    #[test]
    fn test_path_error_display_version_mismatch() {
        let err = PathError::VersionMismatch {
            expected: 1,
            actual: 2,
        };
        assert_eq!(format!("{err}"), "Version mismatch: expected 1, got 2");
    }

    #[test]
    fn test_path_error_clone() {
        let err = PathError::InvalidFormat("test".to_string());
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_path_error_partial_eq() {
        let err1 = PathError::InsufficientData {
            expected: 10,
            actual: 5,
        };
        let err2 = PathError::InsufficientData {
            expected: 10,
            actual: 5,
        };
        let err3 = PathError::InsufficientData {
            expected: 10,
            actual: 6,
        };
        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }
}
