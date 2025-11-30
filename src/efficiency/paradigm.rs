//! Model Paradigm Classification (ENT-010)
//!
//! Provides classification of ML model paradigms with associated
//! memory and performance characteristics.

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
            Self::Adapter => 10.0,  // ~10% of full
            Self::Prefix => 20.0,   // ~5% of full
            Self::IA3 => 50.0,      // ~2% of full
            Self::Full => 1.0,      // No reduction
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
            Self::TraditionalMl => 1.5,  // Minimal overhead
            Self::DeepLearning => 4.0,   // Gradients + optimizer state + activations
            Self::FineTuning(method) => {
                match method {
                    FineTuneMethod::Full => 4.0, // Same as deep learning
                    FineTuneMethod::LoRA { .. } => 1.2, // Base frozen + small adapters
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
            Self::FineTuning(method) => {
                match method {
                    FineTuneMethod::Full => 1.0,
                    FineTuneMethod::LoRA { rank, .. } => 0.95 + (f64::from(*rank) / 256.0).min(0.05),
                    FineTuneMethod::QLoRA { .. } => 0.93,
                    FineTuneMethod::Adapter => 0.92,
                    FineTuneMethod::Prefix => 0.88,
                    FineTuneMethod::IA3 => 0.90,
                }
            }
            Self::Distillation => 0.85, // Student typically slightly worse
            Self::MoE => 1.05,          // Can exceed with specialization
            Self::Ensemble => 1.02,     // Slight improvement from diversity
        }
    }

    /// Check if this paradigm requires a pretrained model
    pub fn requires_pretrained(&self) -> bool {
        matches!(
            self,
            Self::FineTuning(_) | Self::Distillation
        )
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
            Self::FineTuning(method) => {
                match method {
                    FineTuneMethod::Full => 1.0,
                    FineTuneMethod::LoRA { .. } => 2.0,
                    FineTuneMethod::QLoRA { .. } => 4.0,
                    FineTuneMethod::Adapter => 1.5,
                    FineTuneMethod::Prefix => 1.8,
                    FineTuneMethod::IA3 => 3.0,
                }
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fine_tune_method_lora() {
        let lora = FineTuneMethod::lora(64);
        match lora {
            FineTuneMethod::LoRA { rank, alpha } => {
                assert_eq!(rank, 64);
                assert!((alpha - 64.0).abs() < f32::EPSILON);
            }
            _ => panic!("Expected LoRA"),
        }
    }

    #[test]
    fn test_fine_tune_method_qlora() {
        let qlora = FineTuneMethod::qlora(32);
        match qlora {
            FineTuneMethod::QLoRA { rank, bits } => {
                assert_eq!(rank, 32);
                assert_eq!(bits, 4);
            }
            _ => panic!("Expected QLoRA"),
        }
    }

    #[test]
    fn test_fine_tune_method_memory_reduction() {
        let full = FineTuneMethod::Full;
        let lora = FineTuneMethod::lora(64);
        let qlora = FineTuneMethod::qlora(32);

        assert!((full.memory_reduction_factor() - 1.0).abs() < 0.01);
        assert!(lora.memory_reduction_factor() > 1.0);
        assert!(qlora.memory_reduction_factor() > lora.memory_reduction_factor());
    }

    #[test]
    fn test_fine_tune_method_trainable_params() {
        let full = FineTuneMethod::Full;
        let lora = FineTuneMethod::lora(8);
        let ia3 = FineTuneMethod::IA3;

        assert!((full.trainable_params_percent() - 100.0).abs() < 0.01);
        assert!(lora.trainable_params_percent() < 1.0);
        assert!(ia3.trainable_params_percent() < lora.trainable_params_percent());
    }

    #[test]
    fn test_fine_tune_method_display() {
        assert_eq!(format!("{}", FineTuneMethod::lora(64)), "LoRA(r=64, α=64)");
        assert_eq!(format!("{}", FineTuneMethod::qlora(32)), "QLoRA(r=32, 4-bit)");
        assert_eq!(format!("{}", FineTuneMethod::IA3), "IA³");
    }

    #[test]
    fn test_model_paradigm_lora() {
        let paradigm = ModelParadigm::lora(64, 16.0);
        match paradigm {
            ModelParadigm::FineTuning(FineTuneMethod::LoRA { rank, alpha }) => {
                assert_eq!(rank, 64);
                assert!((alpha - 16.0).abs() < f32::EPSILON);
            }
            _ => panic!("Expected LoRA fine-tuning"),
        }
    }

    #[test]
    fn test_model_paradigm_memory_multiplier() {
        let traditional = ModelParadigm::TraditionalMl;
        let deep = ModelParadigm::DeepLearning;
        let lora = ModelParadigm::lora(64, 64.0);
        let distill = ModelParadigm::Distillation;

        assert!(traditional.typical_memory_multiplier() < deep.typical_memory_multiplier());
        assert!(lora.typical_memory_multiplier() < deep.typical_memory_multiplier());
        assert!(distill.typical_memory_multiplier() > deep.typical_memory_multiplier());
    }

    #[test]
    fn test_model_paradigm_training_speedup() {
        let deep = ModelParadigm::DeepLearning;
        let lora = ModelParadigm::lora(64, 64.0);
        let traditional = ModelParadigm::TraditionalMl;

        assert!((deep.typical_training_speedup() - 1.0).abs() < 0.01);
        assert!(lora.typical_training_speedup() > deep.typical_training_speedup());
        assert!(traditional.typical_training_speedup() > lora.typical_training_speedup());
    }

    #[test]
    fn test_model_paradigm_quality_retention() {
        let deep = ModelParadigm::DeepLearning;
        let lora = ModelParadigm::lora(64, 64.0);
        let distill = ModelParadigm::Distillation;
        let ensemble = ModelParadigm::Ensemble;

        assert!((deep.typical_quality_retention() - 1.0).abs() < 0.01);
        assert!(lora.typical_quality_retention() > 0.9);
        assert!(distill.typical_quality_retention() < 1.0);
        assert!(ensemble.typical_quality_retention() > 1.0);
    }

    #[test]
    fn test_model_paradigm_requires_pretrained() {
        assert!(!ModelParadigm::TraditionalMl.requires_pretrained());
        assert!(!ModelParadigm::DeepLearning.requires_pretrained());
        assert!(ModelParadigm::lora(64, 64.0).requires_pretrained());
        assert!(ModelParadigm::Distillation.requires_pretrained());
    }

    #[test]
    fn test_model_paradigm_is_parameter_efficient() {
        assert!(!ModelParadigm::DeepLearning.is_parameter_efficient());
        assert!(!ModelParadigm::FineTuning(FineTuneMethod::Full).is_parameter_efficient());
        assert!(ModelParadigm::lora(64, 64.0).is_parameter_efficient());
        assert!(ModelParadigm::qlora(32, 4).is_parameter_efficient());
    }

    #[test]
    fn test_model_paradigm_batch_size_multiplier() {
        let deep = ModelParadigm::DeepLearning;
        let qlora = ModelParadigm::qlora(32, 4);
        let distill = ModelParadigm::Distillation;

        assert!((deep.batch_size_multiplier() - 1.0).abs() < 0.01);
        assert!(qlora.batch_size_multiplier() > deep.batch_size_multiplier());
        assert!(distill.batch_size_multiplier() < deep.batch_size_multiplier());
    }

    #[test]
    fn test_model_paradigm_default() {
        let default = ModelParadigm::default();
        assert!(matches!(default, ModelParadigm::DeepLearning));
    }

    #[test]
    fn test_model_paradigm_display() {
        assert_eq!(format!("{}", ModelParadigm::TraditionalMl), "Traditional ML");
        assert_eq!(format!("{}", ModelParadigm::DeepLearning), "Deep Learning");
        assert_eq!(format!("{}", ModelParadigm::MoE), "Mixture of Experts");
        assert!(format!("{}", ModelParadigm::lora(64, 64.0)).contains("LoRA"));
    }

    #[test]
    fn test_model_paradigm_serialization() {
        let paradigm = ModelParadigm::lora(32, 16.0);
        let json = serde_json::to_string(&paradigm).unwrap();
        let parsed: ModelParadigm = serde_json::from_str(&json).unwrap();

        assert!(parsed.is_parameter_efficient());
    }
}
