//! Callback system for training events
//!
//! Provides extensible hooks for training loop events:
//! - `on_train_begin` / `on_train_end`
//! - `on_epoch_begin` / `on_epoch_end`
//! - `on_step_begin` / `on_step_end`
//!
//! # Example
//!
//! ```rust
//! use entrenar::train::callback::{TrainerCallback, CallbackContext, CallbackAction};
//!
//! struct PrintCallback;
//!
//! impl TrainerCallback for PrintCallback {
//!     fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
//!         println!("Epoch {} finished with loss {:.4}", ctx.epoch, ctx.loss);
//!         CallbackAction::Continue
//!     }
//! }
//! ```

use std::path::PathBuf;

/// Context passed to callbacks with current training state
#[derive(Clone, Debug)]
pub struct CallbackContext {
    /// Current epoch (0-indexed)
    pub epoch: usize,
    /// Total epochs planned
    pub max_epochs: usize,
    /// Current step within epoch
    pub step: usize,
    /// Total steps in epoch
    pub steps_per_epoch: usize,
    /// Global step count
    pub global_step: usize,
    /// Current loss value
    pub loss: f32,
    /// Current learning rate
    pub lr: f32,
    /// Best loss seen so far
    pub best_loss: Option<f32>,
    /// Validation loss (if available)
    pub val_loss: Option<f32>,
    /// Training duration in seconds
    pub elapsed_secs: f64,
}

impl Default for CallbackContext {
    fn default() -> Self {
        Self {
            epoch: 0,
            max_epochs: 0,
            step: 0,
            steps_per_epoch: 0,
            global_step: 0,
            loss: 0.0,
            lr: 0.0,
            best_loss: None,
            val_loss: None,
            elapsed_secs: 0.0,
        }
    }
}

/// Action to take after a callback
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallbackAction {
    /// Continue training normally
    Continue,
    /// Stop training (early stopping)
    Stop,
    /// Skip rest of current epoch
    SkipEpoch,
}

/// Trait for training callbacks
///
/// Implement this trait to hook into training events. All methods have
/// default no-op implementations, so you only need to implement the
/// events you care about.
pub trait TrainerCallback: Send {
    /// Called before training starts
    fn on_train_begin(&mut self, _ctx: &CallbackContext) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Called after training ends
    fn on_train_end(&mut self, _ctx: &CallbackContext) {}

    /// Called before each epoch
    fn on_epoch_begin(&mut self, _ctx: &CallbackContext) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Called after each epoch
    fn on_epoch_end(&mut self, _ctx: &CallbackContext) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Called before each training step
    fn on_step_begin(&mut self, _ctx: &CallbackContext) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Called after each training step
    fn on_step_end(&mut self, _ctx: &CallbackContext) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Called when validation is performed
    fn on_validation(&mut self, _ctx: &CallbackContext) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Get callback name for logging
    fn name(&self) -> &str {
        "TrainerCallback"
    }
}

// =============================================================================
// Early Stopping Callback
// =============================================================================

/// Early stopping callback to halt training when loss plateaus
///
/// Monitors a metric and stops training if no improvement is seen
/// for `patience` epochs.
///
/// # Example
///
/// ```rust
/// use entrenar::train::callback::EarlyStopping;
///
/// // Stop if no improvement for 5 epochs, min improvement 0.001
/// let early_stop = EarlyStopping::new(5, 0.001);
/// ```
#[derive(Clone, Debug)]
pub struct EarlyStopping {
    /// Number of epochs to wait for improvement
    patience: usize,
    /// Minimum improvement to reset patience
    min_delta: f32,
    /// Best loss seen so far
    best_loss: f32,
    /// Epochs without improvement
    epochs_without_improvement: usize,
    /// Whether to restore best weights (placeholder)
    restore_best: bool,
    /// Monitor validation loss instead of training loss
    monitor_val: bool,
}

impl EarlyStopping {
    /// Create new early stopping callback
    pub fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: f32::INFINITY,
            epochs_without_improvement: 0,
            restore_best: false,
            monitor_val: false,
        }
    }

    /// Configure to restore best weights on stop
    pub fn with_restore_best(mut self) -> Self {
        self.restore_best = true;
        self
    }

    /// Configure to monitor validation loss (requires validation data)
    ///
    /// When enabled, early stopping will only consider validation loss.
    /// If validation loss is not available, training loss is used as fallback.
    pub fn monitor_validation(mut self) -> Self {
        self.monitor_val = true;
        self
    }

    /// Reset internal state
    pub fn reset(&mut self) {
        self.best_loss = f32::INFINITY;
        self.epochs_without_improvement = 0;
    }

    /// Check if loss improved
    fn check_improvement(&mut self, loss: f32) -> bool {
        if loss < self.best_loss - self.min_delta {
            self.best_loss = loss;
            self.epochs_without_improvement = 0;
            true
        } else {
            self.epochs_without_improvement += 1;
            false
        }
    }
}

impl TrainerCallback for EarlyStopping {
    fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        // Use val_loss if monitoring validation (with fallback), otherwise use training loss
        let loss = if self.monitor_val {
            ctx.val_loss.unwrap_or(ctx.loss)
        } else {
            ctx.loss
        };
        self.check_improvement(loss);

        if self.epochs_without_improvement >= self.patience {
            eprintln!(
                "Early stopping: no improvement for {} epochs (best loss: {:.4})",
                self.patience, self.best_loss
            );
            CallbackAction::Stop
        } else {
            CallbackAction::Continue
        }
    }

    fn name(&self) -> &str {
        "EarlyStopping"
    }
}

// =============================================================================
// Checkpoint Callback
// =============================================================================

/// Checkpoint callback to save model state periodically
///
/// Saves model state every N epochs or when a new best loss is achieved.
#[derive(Clone, Debug)]
pub struct CheckpointCallback {
    /// Directory to save checkpoints
    checkpoint_dir: PathBuf,
    /// Save every N epochs (None = only save best)
    save_every: Option<usize>,
    /// Save on best loss
    save_best: bool,
    /// Best loss seen
    best_loss: f32,
    /// Last saved epoch
    last_saved_epoch: Option<usize>,
}

impl CheckpointCallback {
    /// Create checkpoint callback saving to directory
    pub fn new(checkpoint_dir: impl Into<PathBuf>) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.into(),
            save_every: None,
            save_best: true,
            best_loss: f32::INFINITY,
            last_saved_epoch: None,
        }
    }

    /// Configure to save every N epochs
    pub fn save_every(mut self, epochs: usize) -> Self {
        self.save_every = Some(epochs);
        self
    }

    /// Configure to save on best loss
    pub fn save_best(mut self, save: bool) -> Self {
        self.save_best = save;
        self
    }

    /// Get checkpoint path for epoch
    pub fn checkpoint_path(&self, epoch: usize) -> PathBuf {
        self.checkpoint_dir
            .join(format!("checkpoint_epoch_{}.json", epoch))
    }

    /// Get best checkpoint path
    pub fn best_checkpoint_path(&self) -> PathBuf {
        self.checkpoint_dir.join("checkpoint_best.json")
    }

    /// Save checkpoint (placeholder - actual implementation needs model access)
    fn save_checkpoint(&mut self, epoch: usize, is_best: bool) {
        // Ensure directory exists
        std::fs::create_dir_all(&self.checkpoint_dir).ok();

        // Placeholder: In real implementation, would serialize model state
        let path = if is_best {
            self.best_checkpoint_path()
        } else {
            self.checkpoint_path(epoch)
        };

        // Write a marker file (real implementation would save model weights)
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let info = format!(
            r#"{{"epoch": {}, "is_best": {}, "timestamp": {}}}"#,
            epoch, is_best, timestamp
        );
        std::fs::write(&path, info).ok();

        self.last_saved_epoch = Some(epoch);
    }
}

impl TrainerCallback for CheckpointCallback {
    fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        let mut should_save = false;
        let mut is_best = false;

        // Check if we should save periodically
        if let Some(interval) = self.save_every {
            if (ctx.epoch + 1).is_multiple_of(interval) {
                should_save = true;
            }
        }

        // Check if this is the best model
        let loss = ctx.val_loss.unwrap_or(ctx.loss);
        if self.save_best && loss < self.best_loss {
            self.best_loss = loss;
            should_save = true;
            is_best = true;
        }

        if should_save {
            self.save_checkpoint(ctx.epoch, is_best);
        }

        CallbackAction::Continue
    }

    fn on_train_end(&mut self, ctx: &CallbackContext) {
        // Save final checkpoint
        self.save_checkpoint(ctx.epoch, false);
    }

    fn name(&self) -> &str {
        "CheckpointCallback"
    }
}

// =============================================================================
// Progress Callback
// =============================================================================

/// Progress callback for logging training progress
#[derive(Clone, Debug)]
pub struct ProgressCallback {
    /// Log every N steps
    log_interval: usize,
}

impl ProgressCallback {
    /// Create progress callback
    pub fn new(log_interval: usize) -> Self {
        Self { log_interval }
    }
}

impl Default for ProgressCallback {
    fn default() -> Self {
        Self { log_interval: 10 }
    }
}

impl TrainerCallback for ProgressCallback {
    fn on_epoch_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
        println!(
            "Epoch {}/{} starting (lr: {:.2e})",
            ctx.epoch + 1,
            ctx.max_epochs,
            ctx.lr
        );
        CallbackAction::Continue
    }

    fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        let val_str = ctx
            .val_loss
            .map(|v| format!(", val_loss: {:.4}", v))
            .unwrap_or_default();

        println!(
            "Epoch {}/{}: loss: {:.4}{} ({:.1}s)",
            ctx.epoch + 1,
            ctx.max_epochs,
            ctx.loss,
            val_str,
            ctx.elapsed_secs
        );
        CallbackAction::Continue
    }

    fn on_step_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        if ctx.step > 0 && ctx.step.is_multiple_of(self.log_interval) {
            println!(
                "  Step {}/{}: loss: {:.4}",
                ctx.step, ctx.steps_per_epoch, ctx.loss
            );
        }
        CallbackAction::Continue
    }

    fn name(&self) -> &str {
        "ProgressCallback"
    }
}

// =============================================================================
// Monitor Callback (integration with entrenar::monitor)
// =============================================================================

/// Callback that integrates with entrenar's monitoring system
#[derive(Debug)]
pub struct MonitorCallback {
    collector: crate::monitor::MetricsCollector,
    andon: crate::monitor::AndonSystem,
}

impl MonitorCallback {
    /// Create a new monitor callback
    pub fn new() -> Self {
        Self {
            collector: crate::monitor::MetricsCollector::new(),
            andon: crate::monitor::AndonSystem::new(),
        }
    }

    /// Get the metrics collector
    pub fn collector(&self) -> &crate::monitor::MetricsCollector {
        &self.collector
    }

    /// Get summary as JSON
    pub fn summary_json(&self) -> Result<String, serde_json::Error> {
        // Convert summary to string keys for JSON
        let summary: std::collections::HashMap<String, _> = self
            .collector
            .summary()
            .into_iter()
            .map(|(k, v)| (k.as_str().to_string(), v))
            .collect();
        serde_json::to_string_pretty(&summary)
    }
}

impl Default for MonitorCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainerCallback for MonitorCallback {
    fn on_step_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        // Record loss at each step
        self.collector
            .record(crate::monitor::Metric::Loss, ctx.loss as f64);
        self.collector
            .record(crate::monitor::Metric::LearningRate, ctx.lr as f64);
        CallbackAction::Continue
    }

    fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        self.collector
            .record(crate::monitor::Metric::Epoch, ctx.epoch as f64);

        // Check for NaN/Inf loss
        if ctx.loss.is_nan() {
            self.andon.critical("NaN loss detected");
        } else if ctx.loss.is_infinite() {
            self.andon.critical("Infinite loss detected");
        }

        // Check if andon suggests stopping
        if self.andon.should_stop() {
            CallbackAction::Stop
        } else {
            CallbackAction::Continue
        }
    }

    fn name(&self) -> &str {
        "MonitorCallback"
    }
}

// =============================================================================
// LR Scheduler Callback
// =============================================================================

use crate::optim::LRScheduler;

/// Callback that applies a learning rate scheduler during training
///
/// Can schedule per-step or per-epoch updates.
///
/// # Example
///
/// ```rust,ignore
/// use entrenar::train::LRSchedulerCallback;
/// use entrenar::optim::CosineAnnealingLR;
///
/// let scheduler = CosineAnnealingLR::new(0.001, 100, 0.0);
/// let callback = LRSchedulerCallback::per_epoch(scheduler);
/// trainer.add_callback(callback);
/// ```
pub struct LRSchedulerCallback<S: LRScheduler + Send> {
    scheduler: S,
    per_step: bool,
    initial_lr: Option<f32>,
}

impl<S: LRScheduler + Send> LRSchedulerCallback<S> {
    /// Create callback that steps scheduler per epoch
    pub fn per_epoch(scheduler: S) -> Self {
        Self {
            scheduler,
            per_step: false,
            initial_lr: None,
        }
    }

    /// Create callback that steps scheduler per step
    pub fn per_step(scheduler: S) -> Self {
        Self {
            scheduler,
            per_step: true,
            initial_lr: None,
        }
    }

    /// Get current learning rate from scheduler
    pub fn current_lr(&self) -> f32 {
        self.scheduler.get_lr()
    }
}

impl<S: LRScheduler + Send> TrainerCallback for LRSchedulerCallback<S> {
    fn on_train_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
        self.initial_lr = Some(ctx.lr);
        CallbackAction::Continue
    }

    fn on_epoch_end(&mut self, _ctx: &CallbackContext) -> CallbackAction {
        if !self.per_step {
            self.scheduler.step();
        }
        CallbackAction::Continue
    }

    fn on_step_end(&mut self, _ctx: &CallbackContext) -> CallbackAction {
        if self.per_step {
            self.scheduler.step();
        }
        CallbackAction::Continue
    }

    fn name(&self) -> &str {
        "LRSchedulerCallback"
    }
}

// =============================================================================
// Callback Manager
// =============================================================================

/// Manages multiple callbacks and dispatches events
pub struct CallbackManager {
    callbacks: Vec<Box<dyn TrainerCallback>>,
}

impl CallbackManager {
    /// Create new callback manager
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    }

    /// Add a callback
    pub fn add<C: TrainerCallback + 'static>(&mut self, callback: C) {
        self.callbacks.push(Box::new(callback));
    }

    /// Check if no callbacks are registered
    pub fn is_empty(&self) -> bool {
        self.callbacks.is_empty()
    }

    /// Get number of callbacks
    pub fn len(&self) -> usize {
        self.callbacks.len()
    }

    /// Fire train begin event
    pub fn on_train_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
        for cb in &mut self.callbacks {
            if cb.on_train_begin(ctx) == CallbackAction::Stop {
                return CallbackAction::Stop;
            }
        }
        CallbackAction::Continue
    }

    /// Fire train end event
    pub fn on_train_end(&mut self, ctx: &CallbackContext) {
        for cb in &mut self.callbacks {
            cb.on_train_end(ctx);
        }
    }

    /// Fire epoch begin event
    pub fn on_epoch_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
        for cb in &mut self.callbacks {
            match cb.on_epoch_begin(ctx) {
                CallbackAction::Stop => return CallbackAction::Stop,
                CallbackAction::SkipEpoch => return CallbackAction::SkipEpoch,
                _ => {}
            }
        }
        CallbackAction::Continue
    }

    /// Fire epoch end event
    pub fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        for cb in &mut self.callbacks {
            if cb.on_epoch_end(ctx) == CallbackAction::Stop {
                return CallbackAction::Stop;
            }
        }
        CallbackAction::Continue
    }

    /// Fire step begin event
    pub fn on_step_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
        for cb in &mut self.callbacks {
            if cb.on_step_begin(ctx) == CallbackAction::Stop {
                return CallbackAction::Stop;
            }
        }
        CallbackAction::Continue
    }

    /// Fire step end event
    pub fn on_step_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        for cb in &mut self.callbacks {
            if cb.on_step_end(ctx) == CallbackAction::Stop {
                return CallbackAction::Stop;
            }
        }
        CallbackAction::Continue
    }
}

impl Default for CallbackManager {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_callback_context_default() {
        let ctx = CallbackContext::default();
        assert_eq!(ctx.epoch, 0);
        assert_eq!(ctx.loss, 0.0);
        assert!(ctx.best_loss.is_none());
    }

    #[test]
    fn test_early_stopping_patience() {
        let mut es = EarlyStopping::new(3, 0.001);
        let mut ctx = CallbackContext::default();

        // First epoch - establishes baseline
        ctx.loss = 1.0;
        assert_eq!(es.on_epoch_end(&ctx), CallbackAction::Continue);

        // Improvement
        ctx.loss = 0.9;
        ctx.epoch = 1;
        assert_eq!(es.on_epoch_end(&ctx), CallbackAction::Continue);

        // No improvement (within delta)
        ctx.loss = 0.899;
        ctx.epoch = 2;
        assert_eq!(es.on_epoch_end(&ctx), CallbackAction::Continue);

        // Still no improvement
        ctx.loss = 0.899;
        ctx.epoch = 3;
        assert_eq!(es.on_epoch_end(&ctx), CallbackAction::Continue);

        // Still no improvement - should stop (patience=3)
        ctx.loss = 0.899;
        ctx.epoch = 4;
        assert_eq!(es.on_epoch_end(&ctx), CallbackAction::Stop);
    }

    #[test]
    fn test_early_stopping_improvement_resets() {
        let mut es = EarlyStopping::new(2, 0.01);
        let mut ctx = CallbackContext::default();

        ctx.loss = 1.0;
        es.on_epoch_end(&ctx);

        ctx.loss = 1.0;
        ctx.epoch = 1;
        es.on_epoch_end(&ctx);

        // Improvement resets counter
        ctx.loss = 0.5;
        ctx.epoch = 2;
        assert_eq!(es.on_epoch_end(&ctx), CallbackAction::Continue);
        assert_eq!(es.epochs_without_improvement, 0);
    }

    #[test]
    fn test_checkpoint_callback_paths() {
        let cb = CheckpointCallback::new("/tmp/checkpoints");
        assert_eq!(
            cb.checkpoint_path(5),
            PathBuf::from("/tmp/checkpoints/checkpoint_epoch_5.json")
        );
        assert_eq!(
            cb.best_checkpoint_path(),
            PathBuf::from("/tmp/checkpoints/checkpoint_best.json")
        );
    }

    #[test]
    fn test_callback_manager_dispatch() {
        let mut manager = CallbackManager::new();

        // Add early stopping that triggers after 1 epoch without improvement
        let es = EarlyStopping::new(1, 0.001);
        manager.add(es);

        let mut ctx = CallbackContext::default();
        ctx.loss = 1.0;

        // First epoch
        assert_eq!(manager.on_epoch_end(&ctx), CallbackAction::Continue);

        // Second epoch - no improvement, should stop
        ctx.epoch = 1;
        assert_eq!(manager.on_epoch_end(&ctx), CallbackAction::Stop);
    }

    #[test]
    fn test_progress_callback() {
        let mut progress = ProgressCallback::new(5);
        let ctx = CallbackContext {
            epoch: 0,
            max_epochs: 10,
            step: 5,
            steps_per_epoch: 100,
            loss: 0.5,
            lr: 0.001,
            ..Default::default()
        };

        // Should not panic
        assert_eq!(progress.on_epoch_begin(&ctx), CallbackAction::Continue);
        assert_eq!(progress.on_step_end(&ctx), CallbackAction::Continue);
        assert_eq!(progress.on_epoch_end(&ctx), CallbackAction::Continue);
    }

    #[test]
    fn test_monitor_callback() {
        let mut monitor = MonitorCallback::new();
        let ctx = CallbackContext {
            epoch: 0,
            step: 0,
            loss: 0.5,
            lr: 0.001,
            ..Default::default()
        };

        assert_eq!(monitor.on_step_end(&ctx), CallbackAction::Continue);
        assert_eq!(monitor.on_epoch_end(&ctx), CallbackAction::Continue);

        // Verify metrics were recorded
        let summary = monitor.collector().summary();
        assert!(summary.contains_key(&crate::monitor::Metric::Loss));
    }

    #[test]
    fn test_monitor_callback_nan_detection() {
        let mut monitor = MonitorCallback::new();
        let ctx = CallbackContext {
            loss: f32::NAN,
            ..Default::default()
        };

        // NaN should trigger stop via andon
        assert_eq!(monitor.on_epoch_end(&ctx), CallbackAction::Stop);
    }
}

// =============================================================================
// Property Tests
// =============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Early stopping should always stop after patience epochs without improvement
        #[test]
        fn early_stopping_respects_patience(
            patience in 1usize..10,
            min_delta in 0.0001f32..0.1,
            initial_loss in 0.1f32..10.0,
        ) {
            let mut es = EarlyStopping::new(patience, min_delta);
            let mut ctx = CallbackContext::default();

            // First epoch establishes baseline
            ctx.loss = initial_loss;
            es.on_epoch_end(&ctx);

            // Run for patience + 1 epochs without improvement
            for epoch in 1..=patience {
                ctx.epoch = epoch;
                ctx.loss = initial_loss; // No improvement
                let action = es.on_epoch_end(&ctx);

                if epoch < patience {
                    prop_assert_eq!(action, CallbackAction::Continue);
                } else {
                    prop_assert_eq!(action, CallbackAction::Stop);
                }
            }
        }

        /// Early stopping counter should reset on improvement
        #[test]
        fn early_stopping_resets_on_improvement(
            patience in 2usize..10,
            min_delta in 0.001f32..0.1,
            initial_loss in 1.0f32..10.0,
            improvement in 0.2f32..0.5,
        ) {
            let mut es = EarlyStopping::new(patience, min_delta);
            let mut ctx = CallbackContext::default();

            // Establish baseline
            ctx.loss = initial_loss;
            es.on_epoch_end(&ctx);

            // One epoch without improvement
            ctx.epoch = 1;
            es.on_epoch_end(&ctx);
            prop_assert!(es.epochs_without_improvement >= 1);

            // Improvement resets counter
            ctx.epoch = 2;
            ctx.loss = initial_loss - improvement;
            es.on_epoch_end(&ctx);
            prop_assert_eq!(es.epochs_without_improvement, 0);
        }

        /// Checkpoint paths should be consistent
        #[test]
        fn checkpoint_paths_are_consistent(
            epoch in 0usize..1000,
        ) {
            let cb = CheckpointCallback::new("/tmp/test");

            // Should generate predictable paths
            let path = cb.checkpoint_path(epoch);
            let expected = format!("/tmp/test/checkpoint_epoch_{}.json", epoch);
            prop_assert_eq!(path, PathBuf::from(&expected));

            // Best path should be constant
            let best = cb.best_checkpoint_path();
            prop_assert_eq!(best, PathBuf::from("/tmp/test/checkpoint_best.json"));
        }

        /// Callback manager should propagate stop action
        #[test]
        fn callback_manager_propagates_stop(
            patience in 1usize..5,
        ) {
            let mut manager = CallbackManager::new();
            manager.add(EarlyStopping::new(patience, 0.001));

            let mut ctx = CallbackContext::default();
            ctx.loss = 1.0;

            // Should continue until patience exhausted
            for epoch in 0..patience {
                ctx.epoch = epoch;
                let action = manager.on_epoch_end(&ctx);
                if epoch < patience - 1 {
                    prop_assert_eq!(action, CallbackAction::Continue);
                }
            }

            // Final epoch should stop
            ctx.epoch = patience;
            prop_assert_eq!(manager.on_epoch_end(&ctx), CallbackAction::Stop);
        }

        /// Progress callback should always continue
        #[test]
        fn progress_callback_never_stops(
            epoch in 0usize..100,
            step in 0usize..1000,
            loss in -100.0f32..100.0,
        ) {
            let mut progress = ProgressCallback::new(10);
            let ctx = CallbackContext {
                epoch,
                max_epochs: 100,
                step,
                steps_per_epoch: 100,
                loss,
                lr: 0.001,
                ..Default::default()
            };

            prop_assert_eq!(progress.on_train_begin(&ctx), CallbackAction::Continue);
            prop_assert_eq!(progress.on_epoch_begin(&ctx), CallbackAction::Continue);
            prop_assert_eq!(progress.on_step_begin(&ctx), CallbackAction::Continue);
            prop_assert_eq!(progress.on_step_end(&ctx), CallbackAction::Continue);
            prop_assert_eq!(progress.on_epoch_end(&ctx), CallbackAction::Continue);
        }

        /// Monitor callback should detect NaN/Inf
        #[test]
        fn monitor_callback_detects_nan_inf(
            normal_loss in -100.0f32..100.0,
        ) {
            // Normal loss should continue
            let mut monitor = MonitorCallback::new();
            let ctx = CallbackContext {
                loss: normal_loss,
                ..Default::default()
            };
            prop_assert_eq!(monitor.on_epoch_end(&ctx), CallbackAction::Continue);

            // NaN should stop
            let mut monitor_nan = MonitorCallback::new();
            let ctx_nan = CallbackContext {
                loss: f32::NAN,
                ..Default::default()
            };
            prop_assert_eq!(monitor_nan.on_epoch_end(&ctx_nan), CallbackAction::Stop);

            // Inf should stop
            let mut monitor_inf = MonitorCallback::new();
            let ctx_inf = CallbackContext {
                loss: f32::INFINITY,
                ..Default::default()
            };
            prop_assert_eq!(monitor_inf.on_epoch_end(&ctx_inf), CallbackAction::Stop);
        }

        /// Multiple callbacks should all fire
        #[test]
        fn multiple_callbacks_fire(
            num_callbacks in 1usize..5,
        ) {
            use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

            struct CounterCallback {
                counter: Arc<AtomicUsize>,
            }

            impl TrainerCallback for CounterCallback {
                fn on_train_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                    self.counter.fetch_add(1, Ordering::SeqCst);
                    CallbackAction::Continue
                }
                fn on_train_end(&mut self, _: &CallbackContext) {}
                fn on_epoch_begin(&mut self, _: &CallbackContext) -> CallbackAction { CallbackAction::Continue }
                fn on_epoch_end(&mut self, _: &CallbackContext) -> CallbackAction { CallbackAction::Continue }
                fn on_step_begin(&mut self, _: &CallbackContext) -> CallbackAction { CallbackAction::Continue }
                fn on_step_end(&mut self, _: &CallbackContext) -> CallbackAction { CallbackAction::Continue }
                fn name(&self) -> &str { "CounterCallback" }
            }

            let counter = Arc::new(AtomicUsize::new(0));
            let mut manager = CallbackManager::new();

            for _ in 0..num_callbacks {
                manager.add(CounterCallback { counter: counter.clone() });
            }

            let ctx = CallbackContext::default();
            manager.on_train_begin(&ctx);

            prop_assert_eq!(counter.load(Ordering::SeqCst), num_callbacks);
        }
    }
}
