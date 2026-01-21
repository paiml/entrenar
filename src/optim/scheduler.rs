//! Learning rate schedulers
//!
//! Provides learning rate scheduling strategies for training:
//! - `CosineAnnealingLR` - Smooth cosine decay
//! - `LinearWarmupLR` - Linear warmup from 0 to target
//! - `StepDecayLR` - Step decay by factor every N epochs
//! - `WarmupCosineDecayLR` - Combined warmup + cosine decay

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

/// Linear Warmup Learning Rate Scheduler
///
/// Linearly increases learning rate from 0 to target over warmup_steps.
/// After warmup, maintains target learning rate.
///
/// Formula: lr_t = lr_target * min(1, t / warmup_steps)
pub struct LinearWarmupLR {
    lr_target: f32,
    warmup_steps: usize,
    current_step: usize,
}

impl LinearWarmupLR {
    /// Create a new linear warmup scheduler
    ///
    /// # Arguments
    /// * `lr_target` - Target learning rate after warmup
    /// * `warmup_steps` - Number of steps for warmup
    pub fn new(lr_target: f32, warmup_steps: usize) -> Self {
        Self {
            lr_target,
            warmup_steps,
            current_step: 0,
        }
    }

    /// Apply the current learning rate to an optimizer
    pub fn apply<O: Optimizer>(&self, optimizer: &mut O) {
        optimizer.set_lr(self.get_lr());
    }
}

impl LRScheduler for LinearWarmupLR {
    fn get_lr(&self) -> f32 {
        if self.warmup_steps == 0 {
            return self.lr_target;
        }

        let progress = (self.current_step as f32 / self.warmup_steps as f32).min(1.0);
        self.lr_target * progress
    }

    fn step(&mut self) {
        self.current_step += 1;
    }
}

/// Step Decay Learning Rate Scheduler
///
/// Multiplies learning rate by gamma every step_size epochs.
///
/// Formula: lr_t = lr_initial * gamma^(floor(epoch / step_size))
pub struct StepDecayLR {
    lr_initial: f32,
    gamma: f32,
    step_size: usize,
    current_epoch: usize,
}

impl StepDecayLR {
    /// Create a new step decay scheduler
    ///
    /// # Arguments
    /// * `lr_initial` - Initial learning rate
    /// * `step_size` - Decay LR every step_size epochs
    /// * `gamma` - Multiplicative factor (e.g., 0.1 for 10x reduction)
    pub fn new(lr_initial: f32, step_size: usize, gamma: f32) -> Self {
        Self {
            lr_initial,
            gamma,
            step_size,
            current_epoch: 0,
        }
    }

    /// Apply the current learning rate to an optimizer
    pub fn apply<O: Optimizer>(&self, optimizer: &mut O) {
        optimizer.set_lr(self.get_lr());
    }
}

impl LRScheduler for StepDecayLR {
    fn get_lr(&self) -> f32 {
        if self.step_size == 0 {
            return self.lr_initial;
        }
        let num_decays = self.current_epoch / self.step_size;
        self.lr_initial * self.gamma.powi(num_decays as i32)
    }

    fn step(&mut self) {
        self.current_epoch += 1;
    }
}

/// Warmup + Cosine Decay Learning Rate Scheduler
///
/// Combines linear warmup with cosine annealing decay.
/// - Phase 1 (warmup): Linear increase from 0 to lr_max
/// - Phase 2 (decay): Cosine decay from lr_max to lr_min
pub struct WarmupCosineDecayLR {
    lr_max: f32,
    lr_min: f32,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl WarmupCosineDecayLR {
    /// Create a new warmup + cosine decay scheduler
    ///
    /// # Arguments
    /// * `lr_max` - Maximum learning rate (after warmup)
    /// * `lr_min` - Minimum learning rate (at end)
    /// * `warmup_steps` - Number of warmup steps
    /// * `total_steps` - Total training steps (including warmup)
    pub fn new(lr_max: f32, lr_min: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            lr_max,
            lr_min,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }

    /// Apply the current learning rate to an optimizer
    pub fn apply<O: Optimizer>(&self, optimizer: &mut O) {
        optimizer.set_lr(self.get_lr());
    }
}

impl LRScheduler for WarmupCosineDecayLR {
    fn get_lr(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            // Warmup phase: linear increase
            if self.warmup_steps == 0 {
                return self.lr_max;
            }
            let progress = self.current_step as f32 / self.warmup_steps as f32;
            return self.lr_max * progress;
        }

        // Cosine decay phase
        let decay_steps = self.total_steps.saturating_sub(self.warmup_steps);
        if decay_steps == 0 {
            return self.lr_min;
        }

        let decay_step = self.current_step - self.warmup_steps;
        if decay_step >= decay_steps {
            return self.lr_min;
        }

        let progress = decay_step as f32 / decay_steps as f32;
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
                "Learning rate should decrease monotonically: prev={prev_lr}, current={current_lr}"
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

    // =========================================================================
    // LinearWarmupLR tests
    // =========================================================================

    #[test]
    fn test_linear_warmup_initial() {
        let scheduler = LinearWarmupLR::new(0.001, 100);
        // At step 0, LR should be 0
        assert_abs_diff_eq!(scheduler.get_lr(), 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_linear_warmup_midpoint() {
        let mut scheduler = LinearWarmupLR::new(0.001, 100);
        for _ in 0..50 {
            scheduler.step();
        }
        // At midpoint, should be half of target
        assert_abs_diff_eq!(scheduler.get_lr(), 0.0005, epsilon = 1e-7);
    }

    #[test]
    fn test_linear_warmup_complete() {
        let mut scheduler = LinearWarmupLR::new(0.001, 100);
        for _ in 0..100 {
            scheduler.step();
        }
        // After warmup, should be at target
        assert_abs_diff_eq!(scheduler.get_lr(), 0.001, epsilon = 1e-7);
    }

    #[test]
    fn test_linear_warmup_past_warmup() {
        let mut scheduler = LinearWarmupLR::new(0.001, 100);
        for _ in 0..200 {
            scheduler.step();
        }
        // Should stay at target after warmup
        assert_abs_diff_eq!(scheduler.get_lr(), 0.001, epsilon = 1e-7);
    }

    #[test]
    fn test_linear_warmup_increases_monotonically() {
        let mut scheduler = LinearWarmupLR::new(0.001, 100);
        let mut prev_lr = scheduler.get_lr();

        for _ in 0..100 {
            scheduler.step();
            let current_lr = scheduler.get_lr();
            assert!(
                current_lr >= prev_lr,
                "LR should increase during warmup: prev={prev_lr}, current={current_lr}"
            );
            prev_lr = current_lr;
        }
    }

    // =========================================================================
    // StepDecayLR tests
    // =========================================================================

    #[test]
    fn test_step_decay_initial() {
        let scheduler = StepDecayLR::new(0.1, 10, 0.1);
        assert_abs_diff_eq!(scheduler.get_lr(), 0.1, epsilon = 1e-7);
    }

    #[test]
    fn test_step_decay_first_decay() {
        let mut scheduler = StepDecayLR::new(0.1, 10, 0.1);
        for _ in 0..10 {
            scheduler.step();
        }
        // After 10 epochs, should decay by gamma
        assert_abs_diff_eq!(scheduler.get_lr(), 0.01, epsilon = 1e-7);
    }

    #[test]
    fn test_step_decay_second_decay() {
        let mut scheduler = StepDecayLR::new(0.1, 10, 0.1);
        for _ in 0..20 {
            scheduler.step();
        }
        // After 20 epochs, should decay twice
        assert_abs_diff_eq!(scheduler.get_lr(), 0.001, epsilon = 1e-8);
    }

    #[test]
    fn test_step_decay_between_steps() {
        let mut scheduler = StepDecayLR::new(0.1, 10, 0.1);
        for _ in 0..5 {
            scheduler.step();
        }
        // Between decay steps, should stay at initial
        assert_abs_diff_eq!(scheduler.get_lr(), 0.1, epsilon = 1e-7);
    }

    // =========================================================================
    // WarmupCosineDecayLR tests
    // =========================================================================

    #[test]
    fn test_warmup_cosine_initial() {
        let scheduler = WarmupCosineDecayLR::new(0.001, 0.0, 10, 100);
        // At step 0, should be 0 (warmup phase)
        assert_abs_diff_eq!(scheduler.get_lr(), 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_warmup_cosine_warmup_midpoint() {
        let mut scheduler = WarmupCosineDecayLR::new(0.001, 0.0, 10, 100);
        for _ in 0..5 {
            scheduler.step();
        }
        // Midpoint of warmup: half of lr_max
        assert_abs_diff_eq!(scheduler.get_lr(), 0.0005, epsilon = 1e-7);
    }

    #[test]
    fn test_warmup_cosine_warmup_complete() {
        let mut scheduler = WarmupCosineDecayLR::new(0.001, 0.0, 10, 100);
        for _ in 0..10 {
            scheduler.step();
        }
        // At end of warmup, should be at lr_max
        assert_abs_diff_eq!(scheduler.get_lr(), 0.001, epsilon = 1e-7);
    }

    #[test]
    fn test_warmup_cosine_decay_complete() {
        let mut scheduler = WarmupCosineDecayLR::new(0.001, 0.0, 10, 100);
        for _ in 0..100 {
            scheduler.step();
        }
        // At end, should be at lr_min
        assert_abs_diff_eq!(scheduler.get_lr(), 0.0, epsilon = 1e-7);
    }

    #[test]
    fn test_warmup_cosine_warmup_increases_then_decreases() {
        let mut scheduler = WarmupCosineDecayLR::new(0.001, 0.0, 10, 100);
        let mut prev_lr = scheduler.get_lr();

        // Warmup phase: should increase
        for _ in 0..10 {
            scheduler.step();
            let current_lr = scheduler.get_lr();
            assert!(
                current_lr >= prev_lr,
                "LR should increase during warmup: prev={prev_lr}, current={current_lr}"
            );
            prev_lr = current_lr;
        }

        // Decay phase: should decrease
        for _ in 10..100 {
            scheduler.step();
            let current_lr = scheduler.get_lr();
            assert!(
                current_lr <= prev_lr,
                "LR should decrease during decay: prev={prev_lr}, current={current_lr}"
            );
            prev_lr = current_lr;
        }
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_linear_warmup_apply() {
        use crate::optim::SGD;
        let mut optimizer = SGD::new(0.0, 0.0);
        let mut scheduler = LinearWarmupLR::new(0.01, 10);

        scheduler.step();
        scheduler.apply(&mut optimizer);
        assert!(optimizer.lr() > 0.0);
    }

    #[test]
    fn test_linear_warmup_zero_steps() {
        let scheduler = LinearWarmupLR::new(0.01, 0);
        // With warmup_steps = 0, should immediately return target
        assert_abs_diff_eq!(scheduler.get_lr(), 0.01, epsilon = 1e-8);
    }

    #[test]
    fn test_step_decay_apply() {
        use crate::optim::SGD;
        let mut optimizer = SGD::new(0.0, 0.0);
        let scheduler = StepDecayLR::new(0.1, 10, 0.1);

        scheduler.apply(&mut optimizer);
        assert_abs_diff_eq!(optimizer.lr(), 0.1, epsilon = 1e-8);
    }

    #[test]
    fn test_step_decay_zero_step_size() {
        let scheduler = StepDecayLR::new(0.1, 0, 0.1);
        // With step_size = 0, should always return initial
        assert_abs_diff_eq!(scheduler.get_lr(), 0.1, epsilon = 1e-8);
    }

    #[test]
    fn test_warmup_cosine_apply() {
        use crate::optim::SGD;
        let mut optimizer = SGD::new(0.0, 0.0);
        let mut scheduler = WarmupCosineDecayLR::new(0.01, 0.0, 10, 100);

        for _ in 0..10 {
            scheduler.step();
        }
        scheduler.apply(&mut optimizer);
        assert_abs_diff_eq!(optimizer.lr(), 0.01, epsilon = 1e-8);
    }

    #[test]
    fn test_warmup_cosine_zero_warmup_steps() {
        let scheduler = WarmupCosineDecayLR::new(0.01, 0.0, 0, 100);
        // With warmup_steps = 0, should start at lr_max
        assert_abs_diff_eq!(scheduler.get_lr(), 0.01, epsilon = 1e-8);
    }

    #[test]
    fn test_warmup_cosine_zero_total_steps() {
        let scheduler = WarmupCosineDecayLR::new(0.01, 0.001, 0, 0);
        // With total_steps = 0 and warmup_steps = 0, decay_steps = 0
        assert_abs_diff_eq!(scheduler.get_lr(), 0.001, epsilon = 1e-8);
    }

    #[test]
    fn test_warmup_cosine_past_total() {
        let mut scheduler = WarmupCosineDecayLR::new(0.01, 0.001, 10, 50);
        for _ in 0..100 {
            scheduler.step();
        }
        // Past total steps, should return lr_min
        assert_abs_diff_eq!(scheduler.get_lr(), 0.001, epsilon = 1e-8);
    }
}
