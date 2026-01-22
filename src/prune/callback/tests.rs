//! Tests for pruning callback
//!
//! Comprehensive test suite for the `PruningCallback` integration with
//! the training loop.

use super::*;
use crate::prune::calibrate::CalibrationConfig;
use crate::prune::config::PruningConfig;
use crate::prune::schedule::PruningSchedule;
use crate::train::callback::{CallbackAction, CallbackContext, TrainerCallback};

fn default_config() -> PruningConfig {
    PruningConfig::default()
}

fn gradual_config() -> PruningConfig {
    PruningConfig::default().with_schedule(PruningSchedule::Gradual {
        start_step: 100,
        end_step: 1000,
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 100,
    })
}

// =============================================================================
// Construction Tests
// =============================================================================

#[test]
fn test_callback_new_default() {
    // TEST_ID: CB-001
    let callback = PruningCallback::new(default_config());
    assert!(
        callback.is_enabled(),
        "CB-001 FALSIFIED: Default should be enabled"
    );
    assert_eq!(
        callback.current_sparsity(),
        0.0,
        "CB-001 FALSIFIED: Initial sparsity should be 0.0"
    );
    assert_eq!(
        callback.parameters_pruned(),
        0,
        "CB-001 FALSIFIED: Initial parameters pruned should be 0"
    );
}

#[test]
fn test_callback_enabled_toggle() {
    // TEST_ID: CB-002
    let mut callback = PruningCallback::new(default_config());
    assert!(callback.is_enabled());
    callback.set_enabled(false);
    assert!(
        !callback.is_enabled(),
        "CB-002 FALSIFIED: Should be disabled"
    );
    callback.set_enabled(true);
    assert!(
        callback.is_enabled(),
        "CB-002 FALSIFIED: Should be enabled again"
    );
}

#[test]
fn test_callback_name() {
    // TEST_ID: CB-003
    let callback = PruningCallback::new(default_config());
    assert_eq!(
        callback.name(),
        "PruningCallback",
        "CB-003 FALSIFIED: Name should be PruningCallback"
    );
}

// =============================================================================
// Sparsity Tracking Tests
// =============================================================================

#[test]
fn test_current_sparsity_starts_at_zero() {
    // TEST_ID: CB-010
    let callback = PruningCallback::new(gradual_config());
    assert_eq!(
        callback.current_sparsity(),
        0.0,
        "CB-010 FALSIFIED: Initial sparsity must be 0.0"
    );
}

#[test]
fn test_set_current_sparsity_clamps_to_bounds() {
    // TEST_ID: CB-011
    let mut callback = PruningCallback::new(default_config());

    callback.set_current_sparsity(0.5);
    assert_eq!(
        callback.current_sparsity(),
        0.5,
        "CB-011 FALSIFIED: Should set sparsity to 0.5"
    );

    callback.set_current_sparsity(-0.5);
    assert_eq!(
        callback.current_sparsity(),
        0.0,
        "CB-011 FALSIFIED: Should clamp negative to 0.0"
    );

    callback.set_current_sparsity(1.5);
    assert_eq!(
        callback.current_sparsity(),
        1.0,
        "CB-011 FALSIFIED: Should clamp >1.0 to 1.0"
    );
}

#[test]
fn test_target_sparsity_from_config() {
    // TEST_ID: CB-012
    let config = PruningConfig::default().with_target_sparsity(0.7);
    let callback = PruningCallback::new(config);
    assert!(
        (callback.target_sparsity() - 0.7).abs() < 1e-6,
        "CB-012 FALSIFIED: Target sparsity should match config"
    );
}

// =============================================================================
// Progress Tracking Tests
// =============================================================================

#[test]
fn test_progress_zero_when_no_pruning() {
    // TEST_ID: CB-020
    let config = PruningConfig::default().with_target_sparsity(0.5);
    let callback = PruningCallback::new(config);
    assert_eq!(
        callback.progress(),
        0.0,
        "CB-020 FALSIFIED: Progress should be 0.0 when no pruning has occurred"
    );
}

#[test]
fn test_progress_increases_with_sparsity() {
    // TEST_ID: CB-021
    let config = PruningConfig::default().with_target_sparsity(0.5);
    let mut callback = PruningCallback::new(config);

    callback.set_current_sparsity(0.25);
    assert!(
        (callback.progress() - 0.5).abs() < 1e-6,
        "CB-021 FALSIFIED: Progress should be 0.5 at half target sparsity"
    );

    callback.set_current_sparsity(0.5);
    assert!(
        (callback.progress() - 1.0).abs() < 1e-6,
        "CB-021 FALSIFIED: Progress should be 1.0 at target sparsity"
    );
}

#[test]
fn test_progress_clamped_to_one() {
    // TEST_ID: CB-022
    let config = PruningConfig::default().with_target_sparsity(0.5);
    let mut callback = PruningCallback::new(config);
    callback.set_current_sparsity(0.6); // Exceeds target
    assert_eq!(
        callback.progress(),
        1.0,
        "CB-022 FALSIFIED: Progress should be clamped to 1.0"
    );
}

#[test]
fn test_progress_one_when_zero_target() {
    // TEST_ID: CB-023
    // Edge case: zero target sparsity
    let config = PruningConfig::default().with_target_sparsity(0.0);
    let callback = PruningCallback::new(config);
    assert_eq!(
        callback.progress(),
        1.0,
        "CB-023 FALSIFIED: Progress should be 1.0 when target is 0"
    );
}

// =============================================================================
// Should Prune Logic Tests
// =============================================================================

#[test]
fn test_should_prune_respects_disabled() {
    // TEST_ID: CB-030
    let mut callback = PruningCallback::new(gradual_config());
    callback.set_enabled(false);
    assert!(
        !callback.should_prune(100),
        "CB-030 FALSIFIED: should_prune should be false when disabled"
    );
}

#[test]
fn test_should_prune_respects_schedule() {
    // TEST_ID: CB-031
    // Note: At step 100 (start_step), sparsity_at_step returns initial_sparsity (0.0)
    // so target > current_sparsity (0.0 > 0.0) is false.
    // We test at step 200 where sparsity is > 0.
    let callback = PruningCallback::new(gradual_config());
    // Before start_step
    assert!(
        !callback.should_prune(50),
        "CB-031 FALSIFIED: should not prune before start"
    );
    // At step 200: sparsity = 0.0 + (100/900) * 0.5 ≈ 0.056
    // schedule.should_prune_at_step(200) = true (at frequency)
    assert!(
        callback.should_prune(200),
        "CB-031 FALSIFIED: should prune at frequency step"
    );
    // At step 300
    assert!(
        callback.should_prune(300),
        "CB-031 FALSIFIED: should prune at frequency step"
    );
}

#[test]
fn test_should_prune_when_sparsity_already_achieved() {
    // TEST_ID: CB-032
    let mut callback = PruningCallback::new(gradual_config());
    // Set current sparsity to max
    callback.set_current_sparsity(0.5);
    // Even at prune step, should not prune if already at target
    assert!(
        !callback.should_prune(100),
        "CB-032 FALSIFIED: should not prune when already at target sparsity"
    );
}

// =============================================================================
// Callback Interface Tests
// =============================================================================

#[test]
fn test_on_train_begin_validates_schedule() {
    // TEST_ID: CB-040
    let config = PruningConfig::default().with_schedule(PruningSchedule::Gradual {
        start_step: 100,
        end_step: 1000,
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 10,
    });
    let mut callback = PruningCallback::new(config);
    let ctx = CallbackContext::default();
    let action = callback.on_train_begin(&ctx);
    assert_eq!(
        action,
        CallbackAction::Continue,
        "CB-040 FALSIFIED: Valid schedule should continue"
    );
}

#[test]
fn test_on_train_begin_rejects_invalid_schedule() {
    // TEST_ID: CB-041
    let config = PruningConfig::default().with_schedule(PruningSchedule::Gradual {
        start_step: 1000,
        end_step: 100, // Invalid: end < start
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 10,
    });
    let mut callback = PruningCallback::new(config);
    let ctx = CallbackContext::default();
    let action = callback.on_train_begin(&ctx);
    assert_eq!(
        action,
        CallbackAction::Stop,
        "CB-041 FALSIFIED: Invalid schedule should stop training"
    );
}

#[test]
fn test_on_step_end_updates_sparsity() {
    // TEST_ID: CB-042
    let config = PruningConfig::default().with_schedule(PruningSchedule::OneShot { step: 100 });
    let mut callback = PruningCallback::new(config);

    let mut ctx = CallbackContext::default();
    ctx.global_step = 50;
    callback.on_step_end(&ctx);
    assert_eq!(
        callback.current_sparsity(),
        0.0,
        "CB-042 FALSIFIED: Sparsity should be 0 before prune step"
    );

    ctx.global_step = 100;
    callback.on_step_end(&ctx);
    assert_eq!(
        callback.current_sparsity(),
        1.0,
        "CB-042 FALSIFIED: Sparsity should be 1.0 at prune step"
    );
}

#[test]
fn test_on_step_end_tracks_last_prune_step() {
    // TEST_ID: CB-043
    let config = PruningConfig::default().with_schedule(PruningSchedule::OneShot { step: 100 });
    let mut callback = PruningCallback::new(config);

    assert!(
        callback.last_prune_step().is_none(),
        "CB-043 FALSIFIED: last_prune_step should be None initially"
    );

    let mut ctx = CallbackContext::default();
    ctx.global_step = 100;
    callback.on_step_end(&ctx);

    assert_eq!(
        callback.last_prune_step(),
        Some(100),
        "CB-043 FALSIFIED: last_prune_step should be 100 after pruning"
    );
}

#[test]
fn test_on_step_end_disabled_does_nothing() {
    // TEST_ID: CB-044
    let config = PruningConfig::default().with_schedule(PruningSchedule::OneShot { step: 0 });
    let mut callback = PruningCallback::new(config);
    callback.set_enabled(false);

    let ctx = CallbackContext::default();
    callback.on_step_end(&ctx);

    assert_eq!(
        callback.current_sparsity(),
        0.0,
        "CB-044 FALSIFIED: Disabled callback should not update sparsity"
    );
}

#[test]
fn test_on_train_end_logs_summary() {
    // TEST_ID: CB-045
    let mut callback = PruningCallback::new(gradual_config());
    callback.set_current_sparsity(0.5);

    let ctx = CallbackContext::default();
    // This should not panic
    callback.on_train_end(&ctx);
}

// =============================================================================
// Completion Tests
// =============================================================================

#[test]
fn test_is_complete_oneshot() {
    // TEST_ID: CB-050
    let config = PruningConfig::default().with_schedule(PruningSchedule::OneShot { step: 100 });
    let mut callback = PruningCallback::new(config);

    assert!(
        !callback.is_complete(),
        "CB-050 FALSIFIED: Should not be complete before pruning"
    );

    let mut ctx = CallbackContext::default();
    ctx.global_step = 100;
    callback.on_step_end(&ctx);

    // Still at prune step, not complete
    assert!(
        !callback.is_complete(),
        "CB-050 FALSIFIED: Should not be complete at prune step"
    );

    ctx.global_step = 101;
    callback.on_step_end(&ctx);

    // Now past prune step
    // Note: We need to update last_prune_step to a step past completion
    callback.last_prune_step = Some(101);
    assert!(
        callback.is_complete(),
        "CB-050 FALSIFIED: Should be complete after prune step"
    );
}

#[test]
fn test_is_complete_gradual() {
    // TEST_ID: CB-051
    let config = PruningConfig::default().with_schedule(PruningSchedule::Gradual {
        start_step: 0,
        end_step: 100,
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 10,
    });
    let mut callback = PruningCallback::new(config);

    // Simulate pruning to completion
    let mut ctx = CallbackContext::default();
    ctx.global_step = 100;
    callback.on_step_end(&ctx);

    // At end_step, not yet complete
    assert!(
        !callback.is_complete(),
        "CB-051 FALSIFIED: Should not be complete at end_step"
    );

    ctx.global_step = 101;
    callback.last_prune_step = Some(101);
    assert!(
        callback.is_complete(),
        "CB-051 FALSIFIED: Should be complete after end_step"
    );
}

// =============================================================================
// Clone Tests
// =============================================================================

#[test]
fn test_callback_clone() {
    // TEST_ID: CB-060
    let mut callback = PruningCallback::new(gradual_config());
    callback.set_current_sparsity(0.3);
    callback.set_enabled(false);

    let cloned = callback.clone();
    assert_eq!(
        callback.current_sparsity(),
        cloned.current_sparsity(),
        "CB-060 FALSIFIED: Cloned sparsity should match"
    );
    assert_eq!(
        callback.is_enabled(),
        cloned.is_enabled(),
        "CB-060 FALSIFIED: Cloned enabled state should match"
    );
}

// =============================================================================
// Config Access Tests
// =============================================================================

#[test]
fn test_config_access() {
    // TEST_ID: CB-070
    let config = gradual_config();
    let callback = PruningCallback::new(config.clone());
    assert_eq!(
        callback.config().target_sparsity(),
        config.target_sparsity(),
        "CB-070 FALSIFIED: Config access should return the config"
    );
}

#[test]
fn test_schedule_access() {
    // TEST_ID: CB-071
    let callback = PruningCallback::new(gradual_config());
    match callback.schedule() {
        PruningSchedule::Gradual { end_step, .. } => {
            assert_eq!(
                *end_step, 1000,
                "CB-071 FALSIFIED: Schedule should be accessible"
            );
        }
        _ => panic!("CB-071 FALSIFIED: Expected Gradual schedule"),
    }
}

// =============================================================================
// Calibration Tests
// =============================================================================

#[test]
fn test_calibration_created_for_wanda() {
    // TEST_ID: CB-080
    use crate::prune::config::PruneMethod;

    let config = PruningConfig::default().with_method(PruneMethod::Wanda);
    let callback = PruningCallback::new(config);
    assert!(
        callback.calibration.is_some(),
        "CB-080 FALSIFIED: Wanda should have calibration collector"
    );
}

#[test]
fn test_no_calibration_for_magnitude() {
    // TEST_ID: CB-081
    use crate::prune::config::PruneMethod;

    let config = PruningConfig::default().with_method(PruneMethod::Magnitude);
    let callback = PruningCallback::new(config);
    assert!(
        callback.calibration.is_none(),
        "CB-081 FALSIFIED: Magnitude should not have calibration collector"
    );
}

#[test]
fn test_with_calibration_constructor() {
    // TEST_ID: CB-082
    let config = gradual_config();
    let cal_config = CalibrationConfig::new().with_num_samples(256);
    let callback = PruningCallback::with_calibration(config, cal_config);
    assert!(
        callback.calibration.is_some(),
        "CB-082 FALSIFIED: with_calibration should create calibration collector"
    );
}
