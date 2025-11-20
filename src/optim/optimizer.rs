//! Optimizer trait

use crate::Tensor;

/// Trait for optimization algorithms
pub trait Optimizer {
    /// Perform a single optimization step
    fn step(&mut self, params: &mut [Tensor]);

    /// Zero out all gradients
    fn zero_grad(&mut self, params: &mut [Tensor]) {
        for param in params {
            param.zero_grad();
        }
    }

    /// Get learning rate
    fn lr(&self) -> f32;

    /// Set learning rate
    fn set_lr(&mut self, lr: f32);
}
