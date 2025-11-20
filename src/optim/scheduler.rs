//! Learning rate schedulers

use super::Optimizer;
use std::f32::consts::PI;

/// Learning rate scheduler trait
pub trait LRScheduler {
    /// Get the current learning rate
    fn get_lr(&self) -> f32;

    /// Step the scheduler (typically called after each epoch or batch)
    fn step(&mut self);
}

/// Cosine Annealing Learning Rate Scheduler
///
/// Decreases the learning rate following a cosine curve from lr_max to lr_min.
///
/// Formula: lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
///
/// Where:
/// - t is the current step
/// - T is the total number of steps
/// - lr_max is the initial learning rate
/// - lr_min is the minimum learning rate (default 0)
pub struct CosineAnnealingLR {
    lr_max: f32,
    lr_min: f32,
    t_max: usize,
    current_step: usize,
}

impl CosineAnnealingLR {
    /// Create a new cosine annealing scheduler
    ///
    /// # Arguments
    /// * `lr_max` - Initial (maximum) learning rate
    /// * `t_max` - Total number of steps for the schedule
    /// * `lr_min` - Minimum learning rate (default 0)
    pub fn new(lr_max: f32, t_max: usize, lr_min: f32) -> Self {
        Self {
            lr_max,
            lr_min,
            t_max,
            current_step: 0,
        }
    }

    /// Create scheduler with lr_min = 0
    pub fn default_min(lr_max: f32, t_max: usize) -> Self {
        Self::new(lr_max, t_max, 0.0)
    }

    /// Apply the current learning rate to an optimizer
    pub fn apply<O: Optimizer>(&self, optimizer: &mut O) {
        optimizer.set_lr(self.get_lr());
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self) -> f32 {
        if self.current_step >= self.t_max {
            return self.lr_min;
        }

        let progress = self.current_step as f32 / self.t_max as f32;
        let cosine_decay = 0.5 * (1.0 + (PI * progress).cos());
        self.lr_min + (self.lr_max - self.lr_min) * cosine_decay
    }

    fn step(&mut self) {
        self.current_step += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_cosine_annealing_initial_lr() {
        let scheduler = CosineAnnealingLR::new(1.0, 100, 0.0);
        // At step 0, should return lr_max
        assert_abs_diff_eq!(scheduler.get_lr(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_annealing_final_lr() {
        let mut scheduler = CosineAnnealingLR::new(1.0, 100, 0.0);

        // Step to the end
        for _ in 0..100 {
            scheduler.step();
        }

        // At step t_max, should return lr_min
        assert_abs_diff_eq!(scheduler.get_lr(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_annealing_midpoint() {
        let mut scheduler = CosineAnnealingLR::new(1.0, 100, 0.0);

        // Step to midpoint
        for _ in 0..50 {
            scheduler.step();
        }

        // At midpoint (t = T/2), cos(π/2) = 0, so lr = lr_max / 2
        assert_abs_diff_eq!(scheduler.get_lr(), 0.5, epsilon = 1e-4);
    }

    #[test]
    fn test_cosine_annealing_with_min() {
        let mut scheduler = CosineAnnealingLR::new(1.0, 100, 0.1);

        // At start
        assert_abs_diff_eq!(scheduler.get_lr(), 1.0, epsilon = 1e-6);

        // Step to end
        for _ in 0..100 {
            scheduler.step();
        }

        // At end, should be lr_min = 0.1
        assert_abs_diff_eq!(scheduler.get_lr(), 0.1, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_annealing_decreases_monotonically() {
        let mut scheduler = CosineAnnealingLR::new(1.0, 100, 0.0);
        let mut prev_lr = scheduler.get_lr();

        for _ in 0..100 {
            scheduler.step();
            let current_lr = scheduler.get_lr();
            assert!(
                current_lr <= prev_lr,
                "Learning rate should decrease monotonically: prev={}, current={}",
                prev_lr,
                current_lr
            );
            prev_lr = current_lr;
        }
    }

    #[test]
    fn test_cosine_annealing_with_optimizer() {
        use crate::optim::SGD;

        let mut optimizer = SGD::new(1.0, 0.0);
        let mut scheduler = CosineAnnealingLR::default_min(1.0, 10);

        // Initial learning rate
        assert_abs_diff_eq!(optimizer.lr(), 1.0, epsilon = 1e-6);

        // Apply scheduler
        scheduler.apply(&mut optimizer);
        assert_abs_diff_eq!(optimizer.lr(), 1.0, epsilon = 1e-6);

        // Step and apply
        scheduler.step();
        scheduler.apply(&mut optimizer);

        // Learning rate should have decreased
        assert!(optimizer.lr() < 1.0);
    }

    #[test]
    fn test_cosine_annealing_past_t_max() {
        let mut scheduler = CosineAnnealingLR::new(1.0, 10, 0.0);

        // Step past t_max
        for _ in 0..20 {
            scheduler.step();
        }

        // Should stay at lr_min
        assert_abs_diff_eq!(scheduler.get_lr(), 0.0, epsilon = 1e-6);
    }
}
