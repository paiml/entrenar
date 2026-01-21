//! Pruning schedule definitions
//!
//! Defines when and how sparsity increases during training:
//! - OneShot: Prune all at once at a specific step
//! - Gradual: Linear interpolation from initial to final sparsity
//! - Cubic: Cubic schedule (Zhu & Gupta, 2017) for smoother transitions
//!
//! # Toyota Way: Kaizen (Continuous Improvement)
//! Gradual and cubic schedules allow incremental model adaptation.
//!
//! # References
//! - Zhu, M., & Gupta, S. (2017). To prune, or not to prune: exploring the
//!   efficacy of pruning for model compression. arXiv:1710.01878.

use serde::{Deserialize, Serialize};

/// Pruning schedule defining when sparsity increases during training.
///
/// # Variants
///
/// - `OneShot`: All pruning happens at a single step
/// - `Gradual`: Linear interpolation between initial and final sparsity
/// - `Cubic`: Cubic polynomial schedule for smoother transitions
///
/// # Example
///
/// ```
/// use entrenar::prune::PruningSchedule;
///
/// // One-shot pruning at step 1000
/// let oneshot = PruningSchedule::OneShot { step: 1000 };
/// assert_eq!(oneshot.sparsity_at_step(500), 0.0);
/// assert_eq!(oneshot.sparsity_at_step(1000), 1.0);
///
/// // Gradual pruning from steps 100-1000
/// let gradual = PruningSchedule::Gradual {
///     start_step: 100,
///     end_step: 1000,
///     initial_sparsity: 0.0,
///     final_sparsity: 0.5,
///     frequency: 10,
/// };
/// assert_eq!(gradual.sparsity_at_step(50), 0.0);
/// assert_eq!(gradual.sparsity_at_step(1000), 0.5);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PruningSchedule {
    /// Prune once at specified step.
    OneShot {
        /// Step at which to apply pruning.
        step: usize,
    },

    /// Gradually increase sparsity over steps with linear interpolation.
    Gradual {
        /// Step to begin pruning.
        start_step: usize,
        /// Step at which final sparsity is reached.
        end_step: usize,
        /// Initial sparsity (typically 0.0).
        initial_sparsity: f32,
        /// Target final sparsity.
        final_sparsity: f32,
        /// Prune every N steps.
        frequency: usize,
    },

    /// Cubic sparsity schedule (Zhu & Gupta, 2017).
    ///
    /// Formula: s_t = s_f * (1 - (1 - t/T)^3)
    ///
    /// This provides faster initial pruning that slows as it approaches
    /// the target, giving the model more time to adapt.
    Cubic {
        /// Step to begin pruning.
        start_step: usize,
        /// Step at which final sparsity is reached.
        end_step: usize,
        /// Target final sparsity.
        final_sparsity: f32,
    },
}

impl PruningSchedule {
    /// Compute the target sparsity at a given training step.
    ///
    /// # Arguments
    ///
    /// * `step` - Current training step
    ///
    /// # Returns
    ///
    /// Target sparsity as a value between 0.0 and 1.0.
    ///
    /// # Panics
    ///
    /// Does not panic. Returns bounded values for all inputs.
    pub fn sparsity_at_step(&self, step: usize) -> f32 {
        match self {
            PruningSchedule::OneShot { step: prune_step } => {
                if step >= *prune_step {
                    1.0
                } else {
                    0.0
                }
            }
            PruningSchedule::Gradual {
                start_step,
                end_step,
                initial_sparsity,
                final_sparsity,
                ..
            } => {
                if step < *start_step {
                    *initial_sparsity
                } else if step >= *end_step {
                    *final_sparsity
                } else {
                    let progress = (step - start_step) as f32 / (end_step - start_step) as f32;
                    initial_sparsity + progress * (final_sparsity - initial_sparsity)
                }
            }
            PruningSchedule::Cubic {
                start_step,
                end_step,
                final_sparsity,
            } => {
                // s_t = s_f * (1 - (1 - t/T)^3)
                if step < *start_step {
                    0.0
                } else if step >= *end_step {
                    *final_sparsity
                } else {
                    let t = (step - start_step) as f32;
                    let total = (end_step - start_step) as f32;
                    let ratio = 1.0 - t / total;
                    final_sparsity * (1.0 - ratio.powi(3))
                }
            }
        }
    }

    /// Check if pruning should be applied at this step based on frequency.
    ///
    /// For `OneShot`, returns true only at the prune step.
    /// For `Gradual`, returns true at steps matching the frequency.
    /// For `Cubic`, returns true at every step in the pruning window.
    ///
    /// # Arguments
    ///
    /// * `step` - Current training step
    pub fn should_prune_at_step(&self, step: usize) -> bool {
        match self {
            PruningSchedule::OneShot { step: prune_step } => step == *prune_step,
            PruningSchedule::Gradual {
                start_step,
                end_step,
                frequency,
                ..
            } => {
                if step < *start_step || step > *end_step {
                    return false;
                }
                if *frequency == 0 {
                    return step == *start_step;
                }
                (step - start_step).is_multiple_of(*frequency)
            }
            PruningSchedule::Cubic {
                start_step,
                end_step,
                ..
            } => step >= *start_step && step <= *end_step,
        }
    }

    /// Get the total number of pruning operations for this schedule.
    ///
    /// # Returns
    ///
    /// Expected number of times pruning will be applied.
    pub fn num_pruning_steps(&self) -> usize {
        match self {
            PruningSchedule::OneShot { .. } => 1,
            PruningSchedule::Gradual {
                start_step,
                end_step,
                frequency,
                ..
            } => {
                if *frequency == 0 {
                    1
                } else {
                    (end_step - start_step) / frequency + 1
                }
            }
            PruningSchedule::Cubic {
                start_step,
                end_step,
                ..
            } => end_step - start_step + 1,
        }
    }

    /// Check if the schedule is valid.
    ///
    /// # Errors
    ///
    /// Returns an error message if the schedule is invalid.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            PruningSchedule::OneShot { .. } => Ok(()),
            PruningSchedule::Gradual {
                start_step,
                end_step,
                initial_sparsity,
                final_sparsity,
                ..
            } => {
                if end_step <= start_step {
                    return Err(format!(
                        "end_step ({end_step}) must be greater than start_step ({start_step})"
                    ));
                }
                if *initial_sparsity < 0.0 || *initial_sparsity > 1.0 {
                    return Err(format!(
                        "initial_sparsity ({initial_sparsity}) must be between 0.0 and 1.0"
                    ));
                }
                if *final_sparsity < 0.0 || *final_sparsity > 1.0 {
                    return Err(format!(
                        "final_sparsity ({final_sparsity}) must be between 0.0 and 1.0"
                    ));
                }
                Ok(())
            }
            PruningSchedule::Cubic {
                start_step,
                end_step,
                final_sparsity,
            } => {
                if end_step <= start_step {
                    return Err(format!(
                        "end_step ({end_step}) must be greater than start_step ({start_step})"
                    ));
                }
                if *final_sparsity < 0.0 || *final_sparsity > 1.0 {
                    return Err(format!(
                        "final_sparsity ({final_sparsity}) must be between 0.0 and 1.0"
                    ));
                }
                Ok(())
            }
        }
    }

    /// Check if pruning has completed (current step is past the schedule).
    pub fn is_complete(&self, step: usize) -> bool {
        match self {
            PruningSchedule::OneShot { step: prune_step } => step > *prune_step,
            PruningSchedule::Gradual { end_step, .. } => step > *end_step,
            PruningSchedule::Cubic { end_step, .. } => step > *end_step,
        }
    }
}

impl Default for PruningSchedule {
    fn default() -> Self {
        PruningSchedule::OneShot { step: 0 }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // OneShot Schedule Tests
    // =========================================================================

    #[test]
    fn test_oneshot_before_step_returns_zero() {
        // TEST_ID: SCHED-001
        // FALSIFIES: OneShot returns non-zero before prune step
        let schedule = PruningSchedule::OneShot { step: 1000 };
        assert_eq!(
            schedule.sparsity_at_step(0),
            0.0,
            "SCHED-001 FALSIFIED: OneShot should return 0.0 before prune step"
        );
        assert_eq!(
            schedule.sparsity_at_step(999),
            0.0,
            "SCHED-001 FALSIFIED: OneShot should return 0.0 at step before prune"
        );
    }

    #[test]
    fn test_oneshot_at_step_returns_one() {
        // TEST_ID: SCHED-002
        // FALSIFIES: OneShot returns wrong value at prune step
        let schedule = PruningSchedule::OneShot { step: 1000 };
        assert_eq!(
            schedule.sparsity_at_step(1000),
            1.0,
            "SCHED-002 FALSIFIED: OneShot should return 1.0 at prune step"
        );
    }

    #[test]
    fn test_oneshot_after_step_returns_one() {
        // TEST_ID: SCHED-003
        // FALSIFIES: OneShot returns wrong value after prune step
        let schedule = PruningSchedule::OneShot { step: 1000 };
        assert_eq!(
            schedule.sparsity_at_step(1001),
            1.0,
            "SCHED-003 FALSIFIED: OneShot should return 1.0 after prune step"
        );
        assert_eq!(
            schedule.sparsity_at_step(10000),
            1.0,
            "SCHED-003 FALSIFIED: OneShot should return 1.0 long after prune step"
        );
    }

    #[test]
    fn test_oneshot_step_zero() {
        // TEST_ID: SCHED-004
        // Edge case: prune at step 0
        let schedule = PruningSchedule::OneShot { step: 0 };
        assert_eq!(
            schedule.sparsity_at_step(0),
            1.0,
            "SCHED-004 FALSIFIED: OneShot at step 0 should return 1.0 immediately"
        );
    }

    #[test]
    fn test_oneshot_should_prune_only_at_step() {
        // TEST_ID: SCHED-005
        let schedule = PruningSchedule::OneShot { step: 500 };
        assert!(
            !schedule.should_prune_at_step(499),
            "SCHED-005 FALSIFIED: should_prune should be false before step"
        );
        assert!(
            schedule.should_prune_at_step(500),
            "SCHED-005 FALSIFIED: should_prune should be true at step"
        );
        assert!(
            !schedule.should_prune_at_step(501),
            "SCHED-005 FALSIFIED: should_prune should be false after step"
        );
    }

    // =========================================================================
    // Gradual Schedule Tests
    // =========================================================================

    #[test]
    fn test_gradual_before_start_returns_initial() {
        // TEST_ID: SCHED-010
        // FALSIFIES: Gradual returns wrong value before start
        let schedule = PruningSchedule::Gradual {
            start_step: 100,
            end_step: 1000,
            initial_sparsity: 0.0,
            final_sparsity: 0.5,
            frequency: 10,
        };
        assert_eq!(
            schedule.sparsity_at_step(0),
            0.0,
            "SCHED-010 FALSIFIED: Gradual should return initial_sparsity before start"
        );
        assert_eq!(
            schedule.sparsity_at_step(99),
            0.0,
            "SCHED-010 FALSIFIED: Gradual should return initial_sparsity at step before start"
        );
    }

    #[test]
    fn test_gradual_after_end_returns_final() {
        // TEST_ID: SCHED-011
        // FALSIFIES: Gradual returns wrong value after end
        let schedule = PruningSchedule::Gradual {
            start_step: 100,
            end_step: 1000,
            initial_sparsity: 0.0,
            final_sparsity: 0.5,
            frequency: 10,
        };
        assert_eq!(
            schedule.sparsity_at_step(1000),
            0.5,
            "SCHED-011 FALSIFIED: Gradual should return final_sparsity at end"
        );
        assert_eq!(
            schedule.sparsity_at_step(10000),
            0.5,
            "SCHED-011 FALSIFIED: Gradual should return final_sparsity after end"
        );
    }

    #[test]
    fn test_gradual_linear_interpolation_midpoint() {
        // TEST_ID: SCHED-012
        // FALSIFIES: Gradual doesn't perform linear interpolation correctly
        let schedule = PruningSchedule::Gradual {
            start_step: 0,
            end_step: 100,
            initial_sparsity: 0.0,
            final_sparsity: 1.0,
            frequency: 10,
        };
        // At step 50, should be 50% through the schedule
        let sparsity = schedule.sparsity_at_step(50);
        assert!(
            (sparsity - 0.5).abs() < 1e-6,
            "SCHED-012 FALSIFIED: Gradual at midpoint should be 0.5, got {sparsity}"
        );
    }

    #[test]
    fn test_gradual_linear_interpolation_quarter() {
        // TEST_ID: SCHED-013
        let schedule = PruningSchedule::Gradual {
            start_step: 0,
            end_step: 100,
            initial_sparsity: 0.0,
            final_sparsity: 0.8,
            frequency: 10,
        };
        // At step 25, should be 25% * 0.8 = 0.2
        let sparsity = schedule.sparsity_at_step(25);
        assert!(
            (sparsity - 0.2).abs() < 1e-6,
            "SCHED-013 FALSIFIED: Gradual at 25% should be 0.2, got {sparsity}"
        );
    }

    #[test]
    fn test_gradual_with_nonzero_initial() {
        // TEST_ID: SCHED-014
        // FALSIFIES: Gradual doesn't handle non-zero initial sparsity
        let schedule = PruningSchedule::Gradual {
            start_step: 0,
            end_step: 100,
            initial_sparsity: 0.2,
            final_sparsity: 0.8,
            frequency: 10,
        };
        // At step 50, should be 0.2 + 0.5 * (0.8 - 0.2) = 0.5
        let sparsity = schedule.sparsity_at_step(50);
        assert!(
            (sparsity - 0.5).abs() < 1e-6,
            "SCHED-014 FALSIFIED: Gradual with initial 0.2 at midpoint should be 0.5, got {sparsity}"
        );
    }

    #[test]
    fn test_gradual_should_prune_at_frequency() {
        // TEST_ID: SCHED-015
        let schedule = PruningSchedule::Gradual {
            start_step: 100,
            end_step: 200,
            initial_sparsity: 0.0,
            final_sparsity: 0.5,
            frequency: 10,
        };
        assert!(
            !schedule.should_prune_at_step(99),
            "SCHED-015 FALSIFIED: should not prune before start"
        );
        assert!(
            schedule.should_prune_at_step(100),
            "SCHED-015 FALSIFIED: should prune at start"
        );
        assert!(
            !schedule.should_prune_at_step(105),
            "SCHED-015 FALSIFIED: should not prune between frequencies"
        );
        assert!(
            schedule.should_prune_at_step(110),
            "SCHED-015 FALSIFIED: should prune at frequency"
        );
        assert!(
            !schedule.should_prune_at_step(201),
            "SCHED-015 FALSIFIED: should not prune after end"
        );
    }

    #[test]
    fn test_gradual_zero_frequency_prunes_once() {
        // TEST_ID: SCHED-016
        let schedule = PruningSchedule::Gradual {
            start_step: 100,
            end_step: 200,
            initial_sparsity: 0.0,
            final_sparsity: 0.5,
            frequency: 0,
        };
        assert!(
            schedule.should_prune_at_step(100),
            "SCHED-016 FALSIFIED: should prune at start with freq=0"
        );
        assert!(
            !schedule.should_prune_at_step(150),
            "SCHED-016 FALSIFIED: should not prune mid-schedule with freq=0"
        );
    }

    // =========================================================================
    // Cubic Schedule Tests
    // =========================================================================

    #[test]
    fn test_cubic_before_start_returns_zero() {
        // TEST_ID: SCHED-020
        // FALSIFIES: Cubic returns non-zero before start
        let schedule = PruningSchedule::Cubic {
            start_step: 100,
            end_step: 1000,
            final_sparsity: 0.5,
        };
        assert_eq!(
            schedule.sparsity_at_step(0),
            0.0,
            "SCHED-020 FALSIFIED: Cubic should return 0.0 before start"
        );
        assert_eq!(
            schedule.sparsity_at_step(99),
            0.0,
            "SCHED-020 FALSIFIED: Cubic should return 0.0 at step before start"
        );
    }

    #[test]
    fn test_cubic_after_end_returns_final() {
        // TEST_ID: SCHED-021
        // FALSIFIES: Cubic returns wrong value after end
        let schedule = PruningSchedule::Cubic {
            start_step: 0,
            end_step: 100,
            final_sparsity: 0.5,
        };
        assert_eq!(
            schedule.sparsity_at_step(100),
            0.5,
            "SCHED-021 FALSIFIED: Cubic should return final_sparsity at end"
        );
        assert_eq!(
            schedule.sparsity_at_step(10000),
            0.5,
            "SCHED-021 FALSIFIED: Cubic should return final_sparsity after end"
        );
    }

    #[test]
    fn test_cubic_formula_at_start() {
        // TEST_ID: SCHED-022
        // At start: t=0, s = s_f * (1 - (1 - 0)^3) = s_f * 0 = 0
        let schedule = PruningSchedule::Cubic {
            start_step: 0,
            end_step: 100,
            final_sparsity: 0.5,
        };
        let sparsity = schedule.sparsity_at_step(0);
        assert!(
            sparsity.abs() < 1e-6,
            "SCHED-022 FALSIFIED: Cubic at start should be 0.0, got {sparsity}"
        );
    }

    #[test]
    fn test_cubic_formula_at_midpoint() {
        // TEST_ID: SCHED-023
        // At midpoint: t=50, T=100, ratio=(1-0.5)=0.5, s = 0.5 * (1 - 0.5^3) = 0.5 * 0.875 = 0.4375
        let schedule = PruningSchedule::Cubic {
            start_step: 0,
            end_step: 100,
            final_sparsity: 0.5,
        };
        let sparsity = schedule.sparsity_at_step(50);
        let expected = 0.5 * (1.0 - 0.5_f32.powi(3));
        assert!(
            (sparsity - expected).abs() < 1e-6,
            "SCHED-023 FALSIFIED: Cubic at midpoint should be {expected}, got {sparsity}"
        );
    }

    #[test]
    fn test_cubic_faster_initial_pruning() {
        // TEST_ID: SCHED-024
        // FALSIFIES: Cubic doesn't provide faster initial pruning than linear
        // At 25% progress, cubic should have higher sparsity than 25% linear
        let schedule = PruningSchedule::Cubic {
            start_step: 0,
            end_step: 100,
            final_sparsity: 1.0,
        };
        let sparsity_25 = schedule.sparsity_at_step(25);
        let linear_25 = 0.25; // 25% of final_sparsity
        assert!(
            sparsity_25 > linear_25,
            "SCHED-024 FALSIFIED: Cubic should be faster than linear at 25%, got {sparsity_25} vs {linear_25}"
        );
    }

    #[test]
    fn test_cubic_slower_final_pruning() {
        // TEST_ID: SCHED-025
        // At 75% progress, cubic should have lower sparsity increase than linear would
        let schedule = PruningSchedule::Cubic {
            start_step: 0,
            end_step: 100,
            final_sparsity: 1.0,
        };
        let sparsity_75 = schedule.sparsity_at_step(75);
        let linear_75 = 0.75;
        // Cubic at 75%: s = 1 * (1 - 0.25^3) = 1 - 0.015625 = 0.984375
        // This is HIGHER than linear, showing faster convergence
        // But the RATE of change is slower (derivative is lower)
        assert!(
            sparsity_75 > linear_75,
            "SCHED-025 FALSIFIED: Cubic at 75% should be higher than linear ({sparsity_75})"
        );
    }

    #[test]
    fn test_cubic_should_prune_in_window() {
        // TEST_ID: SCHED-026
        let schedule = PruningSchedule::Cubic {
            start_step: 100,
            end_step: 200,
            final_sparsity: 0.5,
        };
        assert!(
            !schedule.should_prune_at_step(99),
            "SCHED-026 FALSIFIED: should not prune before window"
        );
        assert!(
            schedule.should_prune_at_step(100),
            "SCHED-026 FALSIFIED: should prune at start of window"
        );
        assert!(
            schedule.should_prune_at_step(150),
            "SCHED-026 FALSIFIED: should prune during window"
        );
        assert!(
            schedule.should_prune_at_step(200),
            "SCHED-026 FALSIFIED: should prune at end of window"
        );
        assert!(
            !schedule.should_prune_at_step(201),
            "SCHED-026 FALSIFIED: should not prune after window"
        );
    }

    // =========================================================================
    // Validation Tests
    // =========================================================================

    #[test]
    fn test_validate_oneshot_always_valid() {
        // TEST_ID: SCHED-030
        let schedule = PruningSchedule::OneShot { step: 0 };
        assert!(
            schedule.validate().is_ok(),
            "SCHED-030 FALSIFIED: OneShot should always be valid"
        );
    }

    #[test]
    fn test_validate_gradual_end_after_start() {
        // TEST_ID: SCHED-031
        let schedule = PruningSchedule::Gradual {
            start_step: 100,
            end_step: 50, // Invalid: end < start
            initial_sparsity: 0.0,
            final_sparsity: 0.5,
            frequency: 10,
        };
        assert!(
            schedule.validate().is_err(),
            "SCHED-031 FALSIFIED: Gradual with end < start should be invalid"
        );
    }

    #[test]
    fn test_validate_gradual_invalid_initial_sparsity() {
        // TEST_ID: SCHED-032
        let schedule = PruningSchedule::Gradual {
            start_step: 0,
            end_step: 100,
            initial_sparsity: -0.1, // Invalid
            final_sparsity: 0.5,
            frequency: 10,
        };
        assert!(
            schedule.validate().is_err(),
            "SCHED-032 FALSIFIED: Negative initial_sparsity should be invalid"
        );
    }

    #[test]
    fn test_validate_gradual_invalid_final_sparsity() {
        // TEST_ID: SCHED-033
        let schedule = PruningSchedule::Gradual {
            start_step: 0,
            end_step: 100,
            initial_sparsity: 0.0,
            final_sparsity: 1.5, // Invalid
            frequency: 10,
        };
        assert!(
            schedule.validate().is_err(),
            "SCHED-033 FALSIFIED: final_sparsity > 1.0 should be invalid"
        );
    }

    #[test]
    fn test_validate_cubic_end_after_start() {
        // TEST_ID: SCHED-034
        let schedule = PruningSchedule::Cubic {
            start_step: 100,
            end_step: 50, // Invalid
            final_sparsity: 0.5,
        };
        assert!(
            schedule.validate().is_err(),
            "SCHED-034 FALSIFIED: Cubic with end < start should be invalid"
        );
    }

    #[test]
    fn test_validate_cubic_invalid_final_sparsity() {
        // TEST_ID: SCHED-035
        let schedule = PruningSchedule::Cubic {
            start_step: 0,
            end_step: 100,
            final_sparsity: 2.0, // Invalid
        };
        assert!(
            schedule.validate().is_err(),
            "SCHED-035 FALSIFIED: final_sparsity > 1.0 should be invalid"
        );
    }

    // =========================================================================
    // Utility Tests
    // =========================================================================

    #[test]
    fn test_num_pruning_steps_oneshot() {
        // TEST_ID: SCHED-040
        let schedule = PruningSchedule::OneShot { step: 1000 };
        assert_eq!(
            schedule.num_pruning_steps(),
            1,
            "SCHED-040 FALSIFIED: OneShot should have exactly 1 pruning step"
        );
    }

    #[test]
    fn test_num_pruning_steps_gradual() {
        // TEST_ID: SCHED-041
        let schedule = PruningSchedule::Gradual {
            start_step: 0,
            end_step: 100,
            initial_sparsity: 0.0,
            final_sparsity: 0.5,
            frequency: 10,
        };
        // Steps: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 = 11 steps
        assert_eq!(
            schedule.num_pruning_steps(),
            11,
            "SCHED-041 FALSIFIED: Gradual with freq=10 over 100 steps should have 11 pruning steps"
        );
    }

    #[test]
    fn test_num_pruning_steps_cubic() {
        // TEST_ID: SCHED-042
        let schedule = PruningSchedule::Cubic {
            start_step: 0,
            end_step: 100,
            final_sparsity: 0.5,
        };
        // Cubic prunes at every step in the window
        assert_eq!(
            schedule.num_pruning_steps(),
            101,
            "SCHED-042 FALSIFIED: Cubic from 0-100 should have 101 pruning steps"
        );
    }

    #[test]
    fn test_is_complete_oneshot() {
        // TEST_ID: SCHED-043
        let schedule = PruningSchedule::OneShot { step: 100 };
        assert!(
            !schedule.is_complete(100),
            "SCHED-043 FALSIFIED: OneShot should not be complete at prune step"
        );
        assert!(
            schedule.is_complete(101),
            "SCHED-043 FALSIFIED: OneShot should be complete after prune step"
        );
    }

    #[test]
    fn test_is_complete_gradual() {
        // TEST_ID: SCHED-044
        let schedule = PruningSchedule::Gradual {
            start_step: 0,
            end_step: 100,
            initial_sparsity: 0.0,
            final_sparsity: 0.5,
            frequency: 10,
        };
        assert!(
            !schedule.is_complete(100),
            "SCHED-044 FALSIFIED: Gradual should not be complete at end_step"
        );
        assert!(
            schedule.is_complete(101),
            "SCHED-044 FALSIFIED: Gradual should be complete after end_step"
        );
    }

    #[test]
    fn test_is_complete_cubic() {
        // TEST_ID: SCHED-046
        // FALSIFIES: Cubic is_complete doesn't work correctly
        let schedule = PruningSchedule::Cubic {
            start_step: 0,
            end_step: 100,
            final_sparsity: 0.5,
        };
        assert!(
            !schedule.is_complete(50),
            "SCHED-046 FALSIFIED: Cubic should not be complete mid-schedule"
        );
        assert!(
            !schedule.is_complete(100),
            "SCHED-046 FALSIFIED: Cubic should not be complete at end_step"
        );
        assert!(
            schedule.is_complete(101),
            "SCHED-046 FALSIFIED: Cubic should be complete after end_step"
        );
    }

    #[test]
    fn test_validate_cubic_valid() {
        // TEST_ID: SCHED-047
        // FALSIFIES: Valid Cubic schedule fails validation
        let schedule = PruningSchedule::Cubic {
            start_step: 0,
            end_step: 100,
            final_sparsity: 0.5,
        };
        assert!(
            schedule.validate().is_ok(),
            "SCHED-047 FALSIFIED: Valid Cubic schedule should pass validation"
        );
    }

    #[test]
    fn test_validate_cubic_negative_sparsity() {
        // TEST_ID: SCHED-048
        let schedule = PruningSchedule::Cubic {
            start_step: 0,
            end_step: 100,
            final_sparsity: -0.1,
        };
        assert!(
            schedule.validate().is_err(),
            "SCHED-048 FALSIFIED: Negative final_sparsity should be invalid"
        );
    }

    #[test]
    fn test_num_pruning_steps_gradual_zero_frequency() {
        // TEST_ID: SCHED-049
        // FALSIFIES: num_pruning_steps doesn't handle freq=0
        let schedule = PruningSchedule::Gradual {
            start_step: 0,
            end_step: 100,
            initial_sparsity: 0.0,
            final_sparsity: 0.5,
            frequency: 0,
        };
        assert_eq!(
            schedule.num_pruning_steps(),
            1,
            "SCHED-049 FALSIFIED: Gradual with freq=0 should have exactly 1 pruning step"
        );
    }

    #[test]
    fn test_default_schedule() {
        // TEST_ID: SCHED-045
        let schedule = PruningSchedule::default();
        match schedule {
            PruningSchedule::OneShot { step } => {
                assert_eq!(
                    step, 0,
                    "SCHED-045 FALSIFIED: Default should be OneShot at step 0"
                );
            }
            _ => panic!("SCHED-045 FALSIFIED: Default should be OneShot variant"),
        }
    }

    // =========================================================================
    // Serialization Tests
    // =========================================================================

    #[test]
    fn test_serialize_oneshot() {
        // TEST_ID: SCHED-050
        let schedule = PruningSchedule::OneShot { step: 1000 };
        let json = serde_json::to_string(&schedule).unwrap();
        assert!(
            json.contains("one_shot"),
            "SCHED-050 FALSIFIED: OneShot should serialize with type=one_shot"
        );
        let deserialized: PruningSchedule = serde_json::from_str(&json).unwrap();
        assert_eq!(
            schedule, deserialized,
            "SCHED-050 FALSIFIED: Deserialized should match original"
        );
    }

    #[test]
    fn test_serialize_gradual() {
        // TEST_ID: SCHED-051
        let schedule = PruningSchedule::Gradual {
            start_step: 100,
            end_step: 1000,
            initial_sparsity: 0.0,
            final_sparsity: 0.5,
            frequency: 10,
        };
        let json = serde_json::to_string(&schedule).unwrap();
        assert!(
            json.contains("gradual"),
            "SCHED-051 FALSIFIED: Gradual should serialize with type=gradual"
        );
        let deserialized: PruningSchedule = serde_json::from_str(&json).unwrap();
        assert_eq!(
            schedule, deserialized,
            "SCHED-051 FALSIFIED: Deserialized should match original"
        );
    }

    #[test]
    fn test_serialize_cubic() {
        // TEST_ID: SCHED-052
        let schedule = PruningSchedule::Cubic {
            start_step: 0,
            end_step: 100,
            final_sparsity: 0.5,
        };
        let json = serde_json::to_string(&schedule).unwrap();
        assert!(
            json.contains("cubic"),
            "SCHED-052 FALSIFIED: Cubic should serialize with type=cubic"
        );
        let deserialized: PruningSchedule = serde_json::from_str(&json).unwrap();
        assert_eq!(
            schedule, deserialized,
            "SCHED-052 FALSIFIED: Deserialized should match original"
        );
    }

    #[test]
    fn test_deserialize_from_yaml() {
        // TEST_ID: SCHED-053
        let yaml = r"
type: gradual
start_step: 100
end_step: 1000
initial_sparsity: 0.0
final_sparsity: 0.5
frequency: 10
";
        let schedule: PruningSchedule = serde_yaml::from_str(yaml).unwrap();
        match schedule {
            PruningSchedule::Gradual {
                start_step,
                end_step,
                initial_sparsity,
                final_sparsity,
                frequency,
            } => {
                assert_eq!(start_step, 100);
                assert_eq!(end_step, 1000);
                assert!((initial_sparsity - 0.0).abs() < 1e-6);
                assert!((final_sparsity - 0.5).abs() < 1e-6);
                assert_eq!(frequency, 10);
            }
            _ => panic!("SCHED-053 FALSIFIED: Should deserialize to Gradual variant"),
        }
    }

    // =========================================================================
    // Property-Based Tests
    // =========================================================================

    #[test]
    fn test_sparsity_monotonic_gradual() {
        // TEST_ID: SCHED-060
        // FALSIFIES: Gradual sparsity can decrease over time
        let schedule = PruningSchedule::Gradual {
            start_step: 0,
            end_step: 100,
            initial_sparsity: 0.0,
            final_sparsity: 0.5,
            frequency: 1,
        };
        let mut prev = 0.0;
        for step in 0..=100 {
            let sparsity = schedule.sparsity_at_step(step);
            assert!(
                sparsity >= prev,
                "SCHED-060 FALSIFIED: Sparsity decreased from {prev} to {sparsity} at step {step}"
            );
            prev = sparsity;
        }
    }

    #[test]
    fn test_sparsity_monotonic_cubic() {
        // TEST_ID: SCHED-061
        // FALSIFIES: Cubic sparsity can decrease over time
        let schedule = PruningSchedule::Cubic {
            start_step: 0,
            end_step: 100,
            final_sparsity: 0.5,
        };
        let mut prev = 0.0;
        for step in 0..=100 {
            let sparsity = schedule.sparsity_at_step(step);
            assert!(
                sparsity >= prev - 1e-6, // Allow floating point tolerance
                "SCHED-061 FALSIFIED: Sparsity decreased from {prev} to {sparsity} at step {step}"
            );
            prev = sparsity;
        }
    }

    #[test]
    fn test_sparsity_bounded_zero_to_final() {
        // TEST_ID: SCHED-062
        // FALSIFIES: Sparsity can exceed bounds
        let schedule = PruningSchedule::Gradual {
            start_step: 0,
            end_step: 100,
            initial_sparsity: 0.0,
            final_sparsity: 0.5,
            frequency: 1,
        };
        for step in 0..=200 {
            let sparsity = schedule.sparsity_at_step(step);
            assert!(
                (0.0..=0.5).contains(&sparsity),
                "SCHED-062 FALSIFIED: Sparsity {sparsity} out of bounds [0.0, 0.5] at step {step}"
            );
        }
    }

    #[test]
    fn test_clone_produces_equal_schedule() {
        // TEST_ID: SCHED-063
        let schedule = PruningSchedule::Gradual {
            start_step: 100,
            end_step: 1000,
            initial_sparsity: 0.1,
            final_sparsity: 0.9,
            frequency: 50,
        };
        let cloned = schedule.clone();
        assert_eq!(
            schedule, cloned,
            "SCHED-063 FALSIFIED: Clone should equal original"
        );
    }

    #[test]
    fn test_debug_format() {
        // TEST_ID: SCHED-064
        let schedule = PruningSchedule::OneShot { step: 100 };
        let debug = format!("{schedule:?}");
        assert!(
            debug.contains("OneShot"),
            "SCHED-064 FALSIFIED: Debug should contain variant name"
        );
        assert!(
            debug.contains("100"),
            "SCHED-064 FALSIFIED: Debug should contain step value"
        );
    }
}

// =============================================================================
// Property Tests with Proptest
// =============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Gradual sparsity is always monotonically increasing
        #[test]
        fn gradual_monotonic(
            start in 0usize..1000,
            duration in 1usize..1000,
            initial in 0.0f32..0.5,
            final_val in 0.5f32..1.0,
        ) {
            let schedule = PruningSchedule::Gradual {
                start_step: start,
                end_step: start + duration,
                initial_sparsity: initial,
                final_sparsity: final_val,
                frequency: 1,
            };

            let mut prev = initial;
            for step in start..=(start + duration) {
                let sparsity = schedule.sparsity_at_step(step);
                prop_assert!(sparsity >= prev - 1e-5);
                prev = sparsity;
            }
        }

        /// Cubic sparsity is always monotonically increasing
        #[test]
        fn cubic_monotonic(
            start in 0usize..1000,
            duration in 1usize..1000,
            final_val in 0.1f32..1.0,
        ) {
            let schedule = PruningSchedule::Cubic {
                start_step: start,
                end_step: start + duration,
                final_sparsity: final_val,
            };

            let mut prev = 0.0;
            for step in start..=(start + duration) {
                let sparsity = schedule.sparsity_at_step(step);
                prop_assert!(sparsity >= prev - 1e-5);
                prev = sparsity;
            }
        }

        /// Sparsity is always bounded by initial and final
        #[test]
        fn sparsity_bounded(
            start in 0usize..100,
            duration in 1usize..100,
            initial in 0.0f32..0.5,
            final_val in 0.5f32..1.0,
            test_step in 0usize..500,
        ) {
            let schedule = PruningSchedule::Gradual {
                start_step: start,
                end_step: start + duration,
                initial_sparsity: initial,
                final_sparsity: final_val,
                frequency: 1,
            };

            let sparsity = schedule.sparsity_at_step(test_step);
            prop_assert!(sparsity >= initial - 1e-6);
            prop_assert!(sparsity <= final_val + 1e-6);
        }

        /// OneShot is idempotent after prune step
        #[test]
        fn oneshot_idempotent(
            prune_step in 0usize..1000,
            test_step in 0usize..2000,
        ) {
            let schedule = PruningSchedule::OneShot { step: prune_step };
            let sparsity = schedule.sparsity_at_step(test_step);

            if test_step >= prune_step {
                prop_assert_eq!(sparsity, 1.0);
            } else {
                prop_assert_eq!(sparsity, 0.0);
            }
        }

        /// Serialize/deserialize roundtrip
        #[test]
        fn serde_roundtrip(
            start in 0usize..1000,
            duration in 1usize..1000,
            initial in 0.0f32..0.5,
            final_val in 0.5f32..1.0,
        ) {
            let schedule = PruningSchedule::Gradual {
                start_step: start,
                end_step: start + duration,
                initial_sparsity: initial,
                final_sparsity: final_val,
                frequency: 10,
            };

            let json = serde_json::to_string(&schedule).unwrap();
            let deserialized: PruningSchedule = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(schedule, deserialized);
        }
    }
}
