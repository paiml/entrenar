//! LoRA/QLoRA configuration optimizer and memory planner.
//!
//! This crate provides tools for:
//! - Optimal LoRA configuration based on memory constraints
//! - Memory planning for different fine-tuning methods
//! - Adapter merging and inspection
//!
//! # Toyota Way Principles
//!
//! - **Heijunka**: Memory planner levels resource allocation
//! - **Kaizen**: Iterative configuration refinement
//! - **Muda Elimination**: Optimal rank selection avoids wasted parameters

pub mod memory;
pub mod merge;
pub mod optimizer;

pub use memory::{MemoryPlanner, MemoryRequirement};
pub use merge::MergeEngine;
pub use optimizer::{LoraOptimizer, OptimalConfig};

use entrenar_common::Result;

/// Plan an optimal LoRA configuration for given constraints.
pub fn plan(model_params: u64, available_vram_gb: f64, method: Method) -> Result<OptimalConfig> {
    LoraOptimizer::new(model_params, available_vram_gb).optimize(method)
}

/// Fine-tuning method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Full fine-tuning (all parameters)
    Full,
    /// LoRA (Low-Rank Adaptation)
    LoRA,
    /// QLoRA (Quantized LoRA)
    QLoRA,
    /// Automatically select based on constraints
    Auto,
}

impl std::str::FromStr for Method {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "full" => Ok(Self::Full),
            "lora" => Ok(Self::LoRA),
            "qlora" => Ok(Self::QLoRA),
            "auto" => Ok(Self::Auto),
            _ => Err(format!(
                "Unknown method: {}. Use: full, lora, qlora, auto",
                s
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_method_parsing() {
        assert_eq!("lora".parse::<Method>().unwrap(), Method::LoRA);
        assert_eq!("QLoRA".parse::<Method>().unwrap(), Method::QLoRA);
        assert_eq!("AUTO".parse::<Method>().unwrap(), Method::Auto);
    }

    #[test]
    fn test_plan_returns_config() {
        let config = plan(7_000_000_000, 16.0, Method::Auto);
        assert!(config.is_ok());
    }
}
