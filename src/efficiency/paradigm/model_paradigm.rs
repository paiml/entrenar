//! Model training/inference paradigm definitions.

use serde::{Deserialize, Serialize};

use super::FineTuneMethod;

/// Model training/inference paradigm
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum ModelParadigm {
    /// Traditional ML (sklearn-style): logistic regression, random forests, etc.
    TraditionalMl,
    /// Deep learning: neural networks trained from scratch
    #[default]
    DeepLearning,
    /// Fine-tuning a pretrained model
    FineTuning(FineTuneMethod),
    /// Knowledge distillation (teacher → student)
    Distillation,
    /// Mixture of Experts
    MoE,
    /// Ensemble of multiple models
    Ensemble,
}

impl ModelParadigm {
    /// Create a LoRA fine-tuning paradigm
    pub fn lora(rank: u32, alpha: f32) -> Self {
        Self::FineTuning(FineTuneMethod::LoRA { rank, alpha })
    }

    /// Create a QLoRA fine-tuning paradigm
    pub fn qlora(rank: u32, bits: u8) -> Self {
        Self::FineTuning(FineTuneMethod::QLoRA { rank, bits })
    }

    /// Get typical memory multiplier relative to model size
    ///
    /// Returns the factor by which memory usage increases during training
    /// compared to inference-only memory.
    pub fn typical_memory_multiplier(&self) -> f64 {
        match self {
            Self::TraditionalMl => 1.5, // Minimal overhead
            Self::DeepLearning => 4.0,  // Gradients + optimizer state + activations
            Self::FineTuning(method) => {
                match method {
                    FineTuneMethod::Full => 4.0,         // Same as deep learning
                    FineTuneMethod::LoRA { .. } => 1.2,  // Base frozen + small adapters
                    FineTuneMethod::QLoRA { .. } => 1.1, // Quantized base + adapters
                    FineTuneMethod::Adapter => 1.5,
                    FineTuneMethod::Prefix => 1.3,
                    FineTuneMethod::IA3 => 1.1,
                }
            }
            Self::Distillation => 5.0, // Teacher + student models
            Self::MoE => 2.0,          // Active experts only
            Self::Ensemble => 3.0,     // Multiple models, typically smaller
        }
    }

    /// Get typical training speedup relative to full training
    ///
    /// Returns the factor by which training is faster compared to
    /// training from scratch.
    pub fn typical_training_speedup(&self) -> f64 {
        match self {
            Self::TraditionalMl => 10.0, // Very fast training
            Self::DeepLearning => 1.0,   // Baseline
            Self::FineTuning(method) => {
                match method {
                    FineTuneMethod::Full => 2.0, // Pretrained initialization helps
                    FineTuneMethod::LoRA { rank, .. } => {
                        // Higher rank = slower but still faster than full
                        5.0 / (1.0 + (f64::from(*rank) / 64.0))
                    }
                    FineTuneMethod::QLoRA { .. } => 6.0,
                    FineTuneMethod::Adapter => 4.0,
                    FineTuneMethod::Prefix => 5.0,
                    FineTuneMethod::IA3 => 8.0,
                }
            }
            Self::Distillation => 1.5, // Training student is faster than teacher
            Self::MoE => 0.8,          // Routing overhead
            Self::Ensemble => 0.5,     // Train multiple models
        }
    }

    /// Get typical quality retention compared to full training
    ///
    /// Returns expected quality as a fraction of full fine-tuning quality.
    pub fn typical_quality_retention(&self) -> f64 {
        match self {
            Self::TraditionalMl => 0.7, // Depends heavily on task
            Self::DeepLearning => 1.0,  // Baseline
            Self::FineTuning(method) => match method {
                FineTuneMethod::Full => 1.0,
                FineTuneMethod::LoRA { rank, .. } => 0.95 + (f64::from(*rank) / 256.0).min(0.05),
                FineTuneMethod::QLoRA { .. } => 0.93,
                FineTuneMethod::Adapter => 0.92,
                FineTuneMethod::Prefix => 0.88,
                FineTuneMethod::IA3 => 0.90,
            },
            Self::Distillation => 0.85, // Student typically slightly worse
            Self::MoE => 1.05,          // Can exceed with specialization
            Self::Ensemble => 1.02,     // Slight improvement from diversity
        }
    }

    /// Check if this paradigm requires a pretrained model
    pub fn requires_pretrained(&self) -> bool {
        matches!(self, Self::FineTuning(_) | Self::Distillation)
    }

    /// Check if this paradigm is parameter-efficient
    pub fn is_parameter_efficient(&self) -> bool {
        matches!(
            self,
            Self::FineTuning(
                FineTuneMethod::LoRA { .. }
                    | FineTuneMethod::QLoRA { .. }
                    | FineTuneMethod::Adapter
                    | FineTuneMethod::Prefix
                    | FineTuneMethod::IA3
            )
        )
    }

    /// Get recommended batch size multiplier
    ///
    /// Parameter-efficient methods allow larger batch sizes due to lower memory.
    pub fn batch_size_multiplier(&self) -> f64 {
        match self {
            Self::TraditionalMl => 10.0,
            Self::DeepLearning => 1.0,
            Self::FineTuning(method) => match method {
                FineTuneMethod::Full => 1.0,
                FineTuneMethod::LoRA { .. } => 2.0,
                FineTuneMethod::QLoRA { .. } => 4.0,
                FineTuneMethod::Adapter => 1.5,
                FineTuneMethod::Prefix => 1.8,
                FineTuneMethod::IA3 => 3.0,
            },
            Self::Distillation => 0.5, // Need memory for both models
            Self::MoE => 1.2,
            Self::Ensemble => 0.3,
        }
    }
}

impl std::fmt::Display for ModelParadigm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TraditionalMl => write!(f, "Traditional ML"),
            Self::DeepLearning => write!(f, "Deep Learning"),
            Self::FineTuning(method) => write!(f, "Fine-tuning ({method})"),
            Self::Distillation => write!(f, "Knowledge Distillation"),
            Self::MoE => write!(f, "Mixture of Experts"),
            Self::Ensemble => write!(f, "Ensemble"),
        }
    }
}
