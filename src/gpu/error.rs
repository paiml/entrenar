//! GPU sharing error types (GPU-SHARE-001/002/003).

use std::fmt;

/// Errors from GPU resource management.
#[derive(Debug)]
pub enum GpuError {
    /// Not enough VRAM for requested budget.
    InsufficientMemory {
        budget_mb: usize,
        available_mb: usize,
        reserved_mb: usize,
        total_mb: usize,
    },
    /// Wait-for-VRAM timed out.
    Timeout {
        budget_mb: usize,
        timeout_secs: u64,
    },
    /// Ledger file is corrupt or unreadable.
    LedgerCorrupt(String),
    /// I/O error accessing ledger.
    Io(std::io::Error),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientMemory { budget_mb, available_mb, reserved_mb, total_mb } => {
                write!(
                    f,
                    "insufficient VRAM: need {budget_mb} MB, available {available_mb} MB \
                     (reserved {reserved_mb} MB / total {total_mb} MB)"
                )
            }
            Self::Timeout { budget_mb, timeout_secs } => {
                write!(f, "VRAM wait timed out after {timeout_secs}s (need {budget_mb} MB)")
            }
            Self::LedgerCorrupt(msg) => write!(f, "ledger corrupt: {msg}"),
            Self::Io(e) => write!(f, "GPU ledger I/O: {e}"),
        }
    }
}

impl std::error::Error for GpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for GpuError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
