//! Fine-tuning methods for model adaptation.

use serde::{Deserialize, Serialize};

/// Fine-tuning methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FineTuneMethod {
    /// Low-Rank Adaptation
    LoRA {
        /// Rank of the low-rank matrices
        rank: u32,
        /// Scaling factor (alpha)
        alpha: f32,
    },
    /// Quantized LoRA (4-bit base weights)
    QLoRA {
        /// Rank of the low-rank matrices
        rank: u32,
        /// Quantization bits (typically 4)
        bits: u8,
    },
    /// Adapter layers
    Adapter,
    /// Prefix tuning
    Prefix,
    /// IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)
    IA3,
    /// Full fine-tuning (all parameters)
    Full,
}

impl FineTuneMethod {
    /// Create LoRA with default alpha = rank
    pub fn lora(rank: u32) -> Self {
        Self::LoRA {
            rank,
            alpha: rank as f32,
        }
    }

    /// Create QLoRA with 4-bit quantization
    pub fn qlora(rank: u32) -> Self {
        Self::QLoRA { rank, bits: 4 }
    }

    /// Get the memory reduction factor compared to full fine-tuning
    pub fn memory_reduction_factor(&self) -> f64 {
        match self {
            Self::LoRA { rank, .. } => {
                // LoRA typically uses ~0.1-1% of parameters
                // Higher rank = more parameters
                100.0 / f64::from(*rank).max(1.0)
            }
            Self::QLoRA { rank, bits } => {
                // QLoRA: quantized base + low-rank adapters
                // 4-bit = 8x compression on base, plus LoRA overhead
                let base_compression = 32.0 / f64::from(*bits);
                let lora_overhead = f64::from(*rank) * 0.01;
                base_compression / (1.0 + lora_overhead)
            }
            Self::Adapter => 10.0, // ~10% of full
            Self::Prefix => 20.0,  // ~5% of full
            Self::IA3 => 50.0,     // ~2% of full
            Self::Full => 1.0,     // No reduction
        }
    }

    /// Get typical trainable parameter percentage
    pub fn trainable_params_percent(&self) -> f64 {
        match self {
            Self::LoRA { rank, .. } => 0.1 * (f64::from(*rank) / 8.0).min(2.0),
            Self::QLoRA { rank, .. } => 0.1 * (f64::from(*rank) / 8.0).min(2.0),
            Self::Adapter => 5.0,
            Self::Prefix => 1.0,
            Self::IA3 => 0.01,
            Self::Full => 100.0,
        }
    }
}

impl std::fmt::Display for FineTuneMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LoRA { rank, alpha } => write!(f, "LoRA(r={rank}, α={alpha})"),
            Self::QLoRA { rank, bits } => write!(f, "QLoRA(r={rank}, {bits}-bit)"),
            Self::Adapter => write!(f, "Adapter"),
            Self::Prefix => write!(f, "Prefix"),
            Self::IA3 => write!(f, "IA³"),
            Self::Full => write!(f, "Full"),
        }
    }
}
