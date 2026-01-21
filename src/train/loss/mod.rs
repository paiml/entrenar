//! Loss functions for training
//!
//! This module provides various loss functions for neural network training:
//!
//! - [`MSELoss`] - Mean Squared Error for regression
//! - [`L1Loss`] - Mean Absolute Error (more robust to outliers)
//! - [`HuberLoss`] / [`SmoothL1Loss`] - Smooth combination of MSE and L1
//! - [`CrossEntropyLoss`] - For classification tasks
//! - [`CausalLMLoss`] - For autoregressive language modeling
//! - [`WeightedLoss`] - Scalar weighting wrapper
//! - [`SampleWeightedLoss`] - Per-sample weighting for curriculum learning

mod causal_lm;
mod cross_entropy;
mod mse;
mod traits;
mod weighted;

pub use causal_lm::CausalLMLoss;
pub use cross_entropy::CrossEntropyLoss;
pub use mse::{HuberLoss, L1Loss, MSELoss, SmoothL1Loss};
pub use traits::LossFn;
pub use weighted::{SampleWeightedLoss, WeightedLoss};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_names() {
        assert_eq!(MSELoss.name(), "MSE");
        assert_eq!(CrossEntropyLoss.name(), "CrossEntropy");
        assert_eq!(HuberLoss::new(1.0).name(), "Huber");
        assert_eq!(L1Loss.name(), "L1");
        assert_eq!(WeightedLoss::new(Box::new(MSELoss), 1.0).name(), "Weighted");
        assert_eq!(
            SampleWeightedLoss::new(Box::new(MSELoss)).name(),
            "SampleWeighted"
        );
        assert_eq!(CausalLMLoss::new(10).name(), "CausalLM");
    }
}
