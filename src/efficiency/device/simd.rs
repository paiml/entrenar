//! SIMD capability detection and abstraction.

use serde::{Deserialize, Serialize};

/// SIMD capability levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SimdCapability {
    /// No SIMD support
    #[default]
    None,
    /// SSE4.1/4.2 (128-bit)
    Sse4,
    /// AVX2 (256-bit)
    Avx2,
    /// AVX-512 (512-bit)
    Avx512,
    /// ARM NEON (128-bit)
    Neon,
}

impl SimdCapability {
    /// Returns the vector width in bits
    pub fn vector_width_bits(&self) -> u32 {
        match self {
            Self::None => 0,
            Self::Sse4 => 128,
            Self::Avx2 => 256,
            Self::Avx512 => 512,
            Self::Neon => 128,
        }
    }

    /// Detect SIMD capability of current CPU
    #[cfg(target_arch = "x86_64")]
    pub fn detect() -> Self {
        if is_x86_feature_detected!("avx512f") {
            Self::Avx512
        } else if is_x86_feature_detected!("avx2") {
            Self::Avx2
        } else if is_x86_feature_detected!("sse4.1") {
            Self::Sse4
        } else {
            Self::None
        }
    }

    /// Detect SIMD capability of current CPU (ARM)
    #[cfg(target_arch = "aarch64")]
    pub fn detect() -> Self {
        // NEON is mandatory on aarch64
        Self::Neon
    }

    /// Detect SIMD capability (fallback for other architectures)
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn detect() -> Self {
        Self::None
    }
}

impl std::fmt::Display for SimdCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Sse4 => write!(f, "SSE4"),
            Self::Avx2 => write!(f, "AVX2"),
            Self::Avx512 => write!(f, "AVX-512"),
            Self::Neon => write!(f, "NEON"),
        }
    }
}
