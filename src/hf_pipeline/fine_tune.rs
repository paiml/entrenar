//! Fine-tuning configuration for HuggingFace models
//!
//! Bridges the HF pipeline with LoRA/QLoRA adapters for efficient fine-tuning.
//!
//! # References
//!
//! [1] Hu, E., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language
//!     Models." arXiv:2106.09685
//!
//! [2] Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized
//!     LLMs." arXiv:2305.14314

use crate::hf_pipeline::error::Result;
use crate::hf_pipeline::FetchError;
use crate::lora::LoRAConfig;
use std::path::PathBuf;

/// Fine-tuning method selection
#[derive(Debug, Clone)]
pub enum FineTuneMethod {
    /// Full fine-tuning (all parameters trainable)
    Full,
    /// LoRA: Low-rank adaptation (Hu et al. 2021)
    LoRA(LoRAConfig),
    /// QLoRA: Quantized LoRA (Dettmers et al. 2023)
    QLoRA {
        /// LoRA configuration
        lora_config: LoRAConfig,
        /// Quantization bits (4 or 8)
        bits: u8,
    },
    /// Prefix tuning (Li & Liang 2021)
    PrefixTuning {
        /// Number of prefix tokens
        prefix_length: usize,
    },
}

impl Default for FineTuneMethod {
    fn default() -> Self {
        Self::LoRA(LoRAConfig::default())
    }
}

/// Fine-tuning configuration
#[derive(Debug, Clone)]
pub struct FineTuneConfig {
    /// Base model repository ID
    pub model_id: String,
    /// Fine-tuning method
    pub method: FineTuneMethod,
    /// Output directory for checkpoints
    pub output_dir: PathBuf,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Weight decay
    pub weight_decay: f64,
    /// Warmup ratio (fraction of total steps)
    pub warmup_ratio: f32,
    /// Save checkpoints every N steps
    pub save_steps: usize,
    /// Evaluate every N steps
    pub eval_steps: usize,
    /// Use gradient checkpointing (memory optimization)
    pub gradient_checkpointing: bool,
    /// Use mixed precision (fp16/bf16)
    pub mixed_precision: Option<MixedPrecision>,
}

/// Mixed precision training options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedPrecision {
    /// FP16 mixed precision
    Fp16,
    /// BF16 mixed precision (better for training)
    Bf16,
}

impl Default for FineTuneConfig {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            method: FineTuneMethod::default(),
            output_dir: PathBuf::from("./output"),
            learning_rate: 2e-4, // Recommended for LoRA
            epochs: 3,
            batch_size: 8,
            max_seq_length: 512,
            gradient_accumulation_steps: 4,
            weight_decay: 0.01,
            warmup_ratio: 0.03,
            save_steps: 500,
            eval_steps: 100,
            gradient_checkpointing: true,
            mixed_precision: Some(MixedPrecision::Bf16),
        }
    }
}

impl FineTuneConfig {
    /// Create new fine-tuning config for a model
    #[must_use]
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            ..Default::default()
        }
    }

    /// Use LoRA fine-tuning
    #[must_use]
    pub fn with_lora(mut self, config: LoRAConfig) -> Self {
        self.method = FineTuneMethod::LoRA(config);
        self
    }

    /// Use QLoRA fine-tuning
    #[must_use]
    pub fn with_qlora(mut self, lora_config: LoRAConfig, bits: u8) -> Self {
        self.method = FineTuneMethod::QLoRA { lora_config, bits };
        self
    }

    /// Use full fine-tuning
    #[must_use]
    pub fn full_fine_tune(mut self) -> Self {
        self.method = FineTuneMethod::Full;
        self
    }

    /// Set learning rate
    #[must_use]
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set number of epochs
    #[must_use]
    pub fn epochs(mut self, n: usize) -> Self {
        self.epochs = n;
        self
    }

    /// Set batch size
    #[must_use]
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set output directory
    #[must_use]
    pub fn output_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.output_dir = path.into();
        self
    }

    /// Enable gradient checkpointing
    #[must_use]
    pub fn gradient_checkpointing(mut self, enabled: bool) -> Self {
        self.gradient_checkpointing = enabled;
        self
    }

    /// Set mixed precision mode
    #[must_use]
    pub fn mixed_precision(mut self, mode: Option<MixedPrecision>) -> Self {
        self.mixed_precision = mode;
        self
    }

    /// Estimate trainable parameters based on fine-tuning method
    #[must_use]
    pub fn estimate_trainable_params(&self, total_params: u64) -> u64 {
        match &self.method {
            FineTuneMethod::Full => total_params,
            FineTuneMethod::LoRA(config) => {
                // LoRA params = 2 * rank * d * num_modules
                // Rough estimate assuming attention projections
                let d = 4096; // Typical hidden size
                let num_modules = config.num_target_modules().max(4);
                let num_layers = 32; // Typical
                2 * (config.rank as u64) * d * (num_modules as u64) * num_layers
            }
            FineTuneMethod::QLoRA { lora_config, .. } => {
                let d = 4096;
                let num_modules = lora_config.num_target_modules().max(4);
                let num_layers = 32;
                2 * (lora_config.rank as u64) * d * (num_modules as u64) * num_layers
            }
            FineTuneMethod::PrefixTuning { prefix_length } => {
                // Prefix params = prefix_length * hidden_size * 2 * num_layers
                let hidden_size = 4096u64;
                let num_layers = 32u64;
                (*prefix_length as u64) * hidden_size * 2 * num_layers
            }
        }
    }

    /// Estimate memory requirements in bytes
    #[must_use]
    pub fn estimate_memory(&self, total_params: u64) -> MemoryRequirement {
        let trainable = self.estimate_trainable_params(total_params);

        // Model memory
        let model_bytes = match &self.method {
            FineTuneMethod::Full => total_params * 4,    // FP32
            FineTuneMethod::LoRA(_) => total_params * 2, // FP16 base + LoRA
            FineTuneMethod::QLoRA { bits, .. } => {
                // Quantized base + FP16 LoRA
                let base = match bits {
                    4 => total_params / 2,
                    8 => total_params,
                    _ => total_params,
                };
                base + trainable * 2
            }
            FineTuneMethod::PrefixTuning { .. } => total_params * 2 + trainable * 4,
        };

        // Optimizer states (Adam: 2x for momentum + variance)
        let optimizer_bytes = trainable * 4 * 2;

        // Gradients
        let gradient_bytes = trainable * 4;

        // Activations (rough estimate based on batch size and seq len)
        let activation_bytes = (self.batch_size * self.max_seq_length * 4096 * 4) as u64
            * if self.gradient_checkpointing { 1 } else { 4 };

        MemoryRequirement {
            model: model_bytes,
            optimizer: optimizer_bytes,
            gradients: gradient_bytes,
            activations: activation_bytes,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.model_id.is_empty() {
            return Err(FetchError::InvalidRepoId {
                repo_id: String::new(),
            });
        }

        if self.learning_rate <= 0.0 {
            return Err(FetchError::ConfigParseError {
                message: "Learning rate must be positive".into(),
            });
        }

        if self.batch_size == 0 {
            return Err(FetchError::ConfigParseError {
                message: "Batch size must be greater than 0".into(),
            });
        }

        if let FineTuneMethod::QLoRA { bits, .. } = &self.method {
            if *bits != 4 && *bits != 8 {
                return Err(FetchError::ConfigParseError {
                    message: format!("QLoRA bits must be 4 or 8, got {bits}"),
                });
            }
        }

        Ok(())
    }
}

/// Memory requirements breakdown
#[derive(Debug, Clone, Copy)]
pub struct MemoryRequirement {
    /// Model weights
    pub model: u64,
    /// Optimizer states
    pub optimizer: u64,
    /// Gradients
    pub gradients: u64,
    /// Activations
    pub activations: u64,
}

impl MemoryRequirement {
    /// Total memory required
    #[must_use]
    pub fn total(&self) -> u64 {
        self.model + self.optimizer + self.gradients + self.activations
    }

    /// Check if fits in available memory
    #[must_use]
    pub fn fits_in(&self, available: u64) -> bool {
        self.total() <= available
    }

    /// Memory savings compared to full fine-tuning
    #[must_use]
    pub fn savings_vs_full(&self, full_params: u64) -> f32 {
        let full_memory = full_params * 4 + full_params * 4 * 2 + full_params * 4;
        1.0 - (self.total() as f32 / full_memory as f32)
    }

    /// Format as human-readable string
    #[must_use]
    pub fn format_human(&self) -> String {
        format!(
            "Model: {:.1}GB, Optimizer: {:.1}GB, Gradients: {:.1}GB, Activations: {:.1}GB, Total: {:.1}GB",
            self.model as f64 / 1e9,
            self.optimizer as f64 / 1e9,
            self.gradients as f64 / 1e9,
            self.activations as f64 / 1e9,
            self.total() as f64 / 1e9
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // FineTuneMethod Tests
    // =========================================================================

    #[test]
    fn test_fine_tune_method_default() {
        let method = FineTuneMethod::default();
        assert!(matches!(method, FineTuneMethod::LoRA(_)));
    }

    #[test]
    fn test_fine_tune_method_qlora() {
        let method = FineTuneMethod::QLoRA {
            lora_config: LoRAConfig::default(),
            bits: 4,
        };
        if let FineTuneMethod::QLoRA { bits, .. } = method {
            assert_eq!(bits, 4);
        } else {
            panic!("Expected QLoRA");
        }
    }

    // =========================================================================
    // FineTuneConfig Tests
    // =========================================================================

    #[test]
    fn test_fine_tune_config_default() {
        let config = FineTuneConfig::default();
        assert!(config.model_id.is_empty());
        assert_eq!(config.epochs, 3);
        assert_eq!(config.batch_size, 8);
    }

    #[test]
    fn test_fine_tune_config_builder() {
        let config = FineTuneConfig::new("microsoft/codebert-base")
            .learning_rate(1e-4)
            .epochs(5)
            .batch_size(16)
            .output_dir("/tmp/output");

        assert_eq!(config.model_id, "microsoft/codebert-base");
        assert_eq!(config.learning_rate, 1e-4);
        assert_eq!(config.epochs, 5);
        assert_eq!(config.batch_size, 16);
    }

    #[test]
    fn test_fine_tune_config_with_lora() {
        let lora = LoRAConfig::new(16, 16.0).target_attention_projections();
        let config = FineTuneConfig::new("model").with_lora(lora.clone());

        if let FineTuneMethod::LoRA(c) = &config.method {
            assert_eq!(c.rank, 16);
        } else {
            panic!("Expected LoRA method");
        }
    }

    #[test]
    fn test_fine_tune_config_with_qlora() {
        let lora = LoRAConfig::new(8, 8.0);
        let config = FineTuneConfig::new("model").with_qlora(lora, 4);

        if let FineTuneMethod::QLoRA { bits, .. } = &config.method {
            assert_eq!(*bits, 4);
        } else {
            panic!("Expected QLoRA method");
        }
    }

    #[test]
    fn test_fine_tune_config_full() {
        let config = FineTuneConfig::new("model").full_fine_tune();
        assert!(matches!(config.method, FineTuneMethod::Full));
    }

    // =========================================================================
    // Validation Tests
    // =========================================================================

    #[test]
    fn test_validate_empty_model_id() {
        let config = FineTuneConfig::default();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_valid_config() {
        let config = FineTuneConfig::new("valid/model");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_invalid_learning_rate() {
        let config = FineTuneConfig::new("model").learning_rate(0.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_qlora_bits() {
        let config = FineTuneConfig::new("model").with_qlora(LoRAConfig::default(), 3);
        assert!(config.validate().is_err());
    }

    // =========================================================================
    // Memory Estimation Tests
    // =========================================================================

    #[test]
    fn test_estimate_trainable_params_full() {
        let config = FineTuneConfig::new("model").full_fine_tune();
        assert_eq!(config.estimate_trainable_params(1_000_000), 1_000_000);
    }

    #[test]
    fn test_estimate_trainable_params_lora() {
        let lora = LoRAConfig::new(8, 8.0).target_attention_projections();
        let config = FineTuneConfig::new("model").with_lora(lora);
        let trainable = config.estimate_trainable_params(7_000_000_000);
        // LoRA should have far fewer trainable params
        assert!(trainable < 100_000_000);
    }

    #[test]
    fn test_estimate_memory_full_vs_lora() {
        let full_config = FineTuneConfig::new("model").full_fine_tune();
        let lora_config = FineTuneConfig::new("model").with_lora(LoRAConfig::default());

        let params = 7_000_000_000u64;
        let full_mem = full_config.estimate_memory(params);
        let lora_mem = lora_config.estimate_memory(params);

        // LoRA should use significantly less memory
        assert!(lora_mem.total() < full_mem.total());
    }

    #[test]
    fn test_estimate_memory_qlora_vs_lora() {
        let lora_config = FineTuneConfig::new("model").with_lora(LoRAConfig::default());
        let qlora_config = FineTuneConfig::new("model").with_qlora(LoRAConfig::default(), 4);

        let params = 7_000_000_000u64;
        let lora_mem = lora_config.estimate_memory(params);
        let qlora_mem = qlora_config.estimate_memory(params);

        // QLoRA should use less memory than LoRA (quantized base)
        assert!(qlora_mem.model < lora_mem.model);
    }

    // =========================================================================
    // MemoryRequirement Tests
    // =========================================================================

    #[test]
    fn test_memory_requirement_total() {
        let mem = MemoryRequirement {
            model: 1000,
            optimizer: 500,
            gradients: 250,
            activations: 100,
        };
        assert_eq!(mem.total(), 1850);
    }

    #[test]
    fn test_memory_requirement_fits_in() {
        let mem = MemoryRequirement {
            model: 1000,
            optimizer: 500,
            gradients: 250,
            activations: 100,
        };
        assert!(mem.fits_in(2000));
        assert!(!mem.fits_in(1000));
    }

    #[test]
    fn test_memory_requirement_savings() {
        let mem = MemoryRequirement {
            model: 500,
            optimizer: 100,
            gradients: 50,
            activations: 50,
        };
        // Full memory for 1000 params = 1000*4 + 1000*8 + 1000*4 = 16000
        let savings = mem.savings_vs_full(1000);
        assert!(savings > 0.0);
        assert!(savings < 1.0);
    }

    #[test]
    fn test_memory_format_human() {
        let mem = MemoryRequirement {
            model: 14_000_000_000,
            optimizer: 2_000_000_000,
            gradients: 1_000_000_000,
            activations: 500_000_000,
        };
        let formatted = mem.format_human();
        assert!(formatted.contains("14.0GB"));
        assert!(formatted.contains("Total:"));
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_gradient_checkpointing_builder() {
        let config = FineTuneConfig::new("model").gradient_checkpointing(false);
        assert!(!config.gradient_checkpointing);

        let config2 = FineTuneConfig::new("model").gradient_checkpointing(true);
        assert!(config2.gradient_checkpointing);
    }

    #[test]
    fn test_mixed_precision_builder() {
        let config = FineTuneConfig::new("model").mixed_precision(Some(MixedPrecision::Fp16));
        assert_eq!(config.mixed_precision, Some(MixedPrecision::Fp16));

        let config2 = FineTuneConfig::new("model").mixed_precision(None);
        assert!(config2.mixed_precision.is_none());
    }

    #[test]
    fn test_mixed_precision_variants() {
        assert_ne!(MixedPrecision::Fp16, MixedPrecision::Bf16);
        assert_eq!(MixedPrecision::Fp16, MixedPrecision::Fp16);
    }

    #[test]
    fn test_estimate_trainable_params_qlora() {
        let lora = LoRAConfig::new(8, 8.0);
        let config = FineTuneConfig::new("model").with_qlora(lora, 4);
        let params = config.estimate_trainable_params(1_000_000_000);
        // QLoRA should return much fewer params
        assert!(params < 1_000_000_000);
        assert!(params > 0);
    }

    #[test]
    fn test_prefix_tuning_method() {
        let method = FineTuneMethod::PrefixTuning { prefix_length: 10 };
        if let FineTuneMethod::PrefixTuning { prefix_length } = method {
            assert_eq!(prefix_length, 10);
        } else {
            panic!("Expected PrefixTuning");
        }
    }

    #[test]
    fn test_estimate_trainable_params_prefix() {
        // Test prefix tuning branch if it exists
        let config = FineTuneConfig {
            method: FineTuneMethod::PrefixTuning { prefix_length: 20 },
            ..FineTuneConfig::new("model")
        };
        let params = config.estimate_trainable_params(1_000_000);
        // Should return some estimate
        assert!(params > 0);
    }

    #[test]
    fn test_fine_tune_config_clone() {
        let config = FineTuneConfig::new("model").epochs(10).batch_size(32);
        let cloned = config.clone();
        assert_eq!(config.epochs, cloned.epochs);
        assert_eq!(config.batch_size, cloned.batch_size);
    }

    #[test]
    fn test_fine_tune_method_clone() {
        let method = FineTuneMethod::QLoRA {
            lora_config: LoRAConfig::default(),
            bits: 4,
        };
        let cloned = method.clone();
        if let FineTuneMethod::QLoRA { bits, .. } = cloned {
            assert_eq!(bits, 4);
        } else {
            panic!("Expected QLoRA after clone");
        }
    }
}
