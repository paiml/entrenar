//! Trainer integration for pruning
//!
//! Provides utilities for integrating pruning with the training pipeline,
//! including fine-tuning after pruning.
//!
//! # Toyota Way: Kaizen (Continuous Improvement)
//! Fine-tuning allows the model to recover from pruning-induced accuracy loss.

use crate::prune::calibrate::CalibrationCollector;
use crate::prune::config::PruningConfig;
use crate::prune::data_loader::{CalibrationDataConfig, CalibrationDataLoader};
use crate::prune::pipeline::{PruneFinetunePipeline, PruningMetrics, PruningStage};
use serde::{Deserialize, Serialize};

/// Configuration for the prune-finetune trainer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruneTrainerConfig {
    /// Pruning configuration.
    pub pruning: PruningConfig,
    /// Calibration data configuration.
    pub calibration: CalibrationDataConfig,
    /// Number of fine-tuning epochs after pruning.
    pub finetune_epochs: usize,
    /// Learning rate for fine-tuning.
    pub finetune_lr: f32,
    /// Whether to evaluate before and after pruning.
    pub evaluate_pre_post: bool,
    /// Checkpoint directory.
    pub checkpoint_dir: Option<String>,
    /// Whether to save intermediate checkpoints.
    pub save_checkpoints: bool,
}

impl Default for PruneTrainerConfig {
    fn default() -> Self {
        Self {
            pruning: PruningConfig::default(),
            calibration: CalibrationDataConfig::default(),
            finetune_epochs: 1,
            finetune_lr: 1e-5,
            evaluate_pre_post: true,
            checkpoint_dir: None,
            save_checkpoints: false,
        }
    }
}

impl PruneTrainerConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the pruning configuration.
    pub fn with_pruning(mut self, config: PruningConfig) -> Self {
        self.pruning = config;
        self
    }

    /// Set the calibration configuration.
    pub fn with_calibration(mut self, config: CalibrationDataConfig) -> Self {
        self.calibration = config;
        self
    }

    /// Set the number of fine-tuning epochs.
    pub fn with_finetune_epochs(mut self, epochs: usize) -> Self {
        self.finetune_epochs = epochs;
        self
    }

    /// Set the fine-tuning learning rate.
    pub fn with_finetune_lr(mut self, lr: f32) -> Self {
        self.finetune_lr = lr;
        self
    }

    /// Enable or disable pre/post evaluation.
    pub fn with_evaluate(mut self, enabled: bool) -> Self {
        self.evaluate_pre_post = enabled;
        self
    }

    /// Set the checkpoint directory.
    pub fn with_checkpoint_dir(mut self, dir: impl Into<String>) -> Self {
        self.checkpoint_dir = Some(dir.into());
        self
    }

    /// Enable or disable checkpoint saving.
    pub fn with_save_checkpoints(mut self, enabled: bool) -> Self {
        self.save_checkpoints = enabled;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        self.pruning.validate()?;

        if self.finetune_lr <= 0.0 {
            return Err("finetune_lr must be positive".to_string());
        }

        Ok(())
    }
}

/// Prune-finetune trainer that orchestrates the full pipeline.
///
/// # Example
///
/// ```ignore
/// use entrenar::prune::{PruneTrainer, PruneTrainerConfig, PruningConfig};
///
/// let config = PruneTrainerConfig::new()
///     .with_pruning(PruningConfig::default().with_target_sparsity(0.5))
///     .with_finetune_epochs(3);
///
/// let mut trainer = PruneTrainer::new(config);
/// trainer.run().unwrap();
/// ```
#[derive(Debug)]
pub struct PruneTrainer {
    /// Configuration.
    config: PruneTrainerConfig,
    /// Pipeline state.
    pipeline: PruneFinetunePipeline,
    /// Calibration data loader.
    data_loader: CalibrationDataLoader,
    /// Calibration collector.
    calibration: Option<CalibrationCollector>,
    /// Current epoch in fine-tuning.
    current_epoch: usize,
}

impl PruneTrainer {
    /// Create a new prune trainer.
    pub fn new(config: PruneTrainerConfig) -> Self {
        let pipeline = PruneFinetunePipeline::new(config.pruning.clone());
        let data_loader = CalibrationDataLoader::new(config.calibration.clone());

        Self {
            config,
            pipeline,
            data_loader,
            calibration: None,
            current_epoch: 0,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &PruneTrainerConfig {
        &self.config
    }

    /// Get the current pipeline state.
    pub fn pipeline(&self) -> &PruneFinetunePipeline {
        &self.pipeline
    }

    /// Get mutable access to the pipeline.
    pub fn pipeline_mut(&mut self) -> &mut PruneFinetunePipeline {
        &mut self.pipeline
    }

    /// Get the current stage.
    pub fn stage(&self) -> PruningStage {
        self.pipeline.stage()
    }

    /// Get the metrics.
    pub fn metrics(&self) -> &PruningMetrics {
        self.pipeline.metrics()
    }

    /// Get the current fine-tuning epoch.
    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }

    /// Check if the trainer is complete.
    pub fn is_complete(&self) -> bool {
        self.pipeline.is_complete()
    }

    /// Check if training succeeded.
    pub fn succeeded(&self) -> bool {
        self.pipeline.succeeded()
    }

    /// Get the error message if failed.
    pub fn error(&self) -> Option<&str> {
        self.pipeline.error()
    }

    /// Initialize the trainer and load calibration data.
    pub fn initialize(&mut self) -> Result<(), String> {
        // Validate configuration
        self.config.validate()?;

        // Load calibration data
        self.data_loader.load()?;

        // Initialize calibration collector if needed
        if self.config.pruning.requires_calibration() {
            self.calibration = Some(CalibrationCollector::new(
                crate::prune::calibrate::CalibrationConfig::new()
                    .with_num_samples(self.config.calibration.num_samples()),
            ));
        }

        Ok(())
    }

    /// Run calibration phase.
    ///
    /// In a real implementation, this would:
    /// 1. Run forward passes through the model
    /// 2. Collect activation statistics for each layer
    /// 3. Store statistics in the calibration collector
    pub fn calibrate(&mut self) -> Result<(), String> {
        if self.pipeline.stage() != PruningStage::Idle
            && self.pipeline.stage() != PruningStage::Calibrating
        {
            return Err("Cannot calibrate in current stage".to_string());
        }

        // Initialize calibration if not done
        if self.calibration.is_none() && self.config.pruning.requires_calibration() {
            self.calibration = Some(CalibrationCollector::new(
                crate::prune::calibrate::CalibrationConfig::new()
                    .with_num_samples(self.config.calibration.num_samples()),
            ));
        }

        // Start calibration stage if at Idle
        if self.pipeline.stage() == PruningStage::Idle {
            if let Some(cal) = self.calibration.take() {
                self.pipeline.start_calibration(cal);
            } else {
                // No calibration needed, advance from Idle to Calibrating
                self.pipeline.advance();
            }
        }

        // Process calibration batches
        for _batch in &self.data_loader {
            // In real implementation:
            // 1. Forward pass through model
            // 2. Extract activations at each layer
            // 3. Update calibration statistics
        }

        // Advance from Calibrating to ComputingImportance
        if self.pipeline.stage() == PruningStage::Calibrating {
            self.pipeline.advance();
        }

        Ok(())
    }

    /// Run the pruning phase.
    ///
    /// In a real implementation, this would:
    /// 1. Compute importance scores using calibration data
    /// 2. Generate sparsity masks
    /// 3. Apply masks to model weights
    pub fn prune(&mut self) -> Result<(), String> {
        // Advance through importance computation
        while self.pipeline.stage() == PruningStage::ComputingImportance {
            // Compute importance scores
            self.pipeline.advance();
        }

        if self.pipeline.stage() != PruningStage::Pruning {
            return Err(format!("Cannot prune in stage {:?}", self.pipeline.stage()));
        }

        // In real implementation:
        // 1. Generate masks based on importance and target sparsity
        // 2. Apply masks to model weights
        // 3. Record metrics

        let target_sparsity = self.config.pruning.target_sparsity();
        self.pipeline.metrics_mut().target_sparsity = target_sparsity;
        self.pipeline.metrics_mut().achieved_sparsity = target_sparsity;

        self.pipeline.advance();
        Ok(())
    }

    /// Run the fine-tuning phase.
    ///
    /// In a real implementation, this would:
    /// 1. Set up optimizer with fine-tuning learning rate
    /// 2. Run training epochs
    /// 3. Track loss and metrics
    pub fn finetune(&mut self) -> Result<(), String> {
        if self.pipeline.stage() != PruningStage::FineTuning {
            return Err(format!(
                "Cannot finetune in stage {:?}",
                self.pipeline.stage()
            ));
        }

        for epoch in 0..self.config.finetune_epochs {
            self.current_epoch = epoch;

            // In real implementation:
            // 1. Run training epoch
            // 2. Track loss
            // 3. Optionally save checkpoint

            // Simulate loss decrease
            let loss = 1.0 / (epoch + 1) as f32;
            self.pipeline.metrics_mut().record_finetune_loss(loss);
        }

        self.pipeline.advance();
        Ok(())
    }

    /// Run evaluation phase.
    pub fn evaluate(&mut self) -> Result<(), String> {
        if self.pipeline.stage() != PruningStage::Evaluating {
            return Err(format!(
                "Cannot evaluate in stage {:?}",
                self.pipeline.stage()
            ));
        }

        // In real implementation:
        // 1. Run evaluation on validation set
        // 2. Compute perplexity/accuracy
        // 3. Record metrics

        self.pipeline.advance();
        Ok(())
    }

    /// Run export phase.
    pub fn export(&mut self) -> Result<(), String> {
        if self.pipeline.stage() != PruningStage::Exporting {
            return Err(format!(
                "Cannot export in stage {:?}",
                self.pipeline.stage()
            ));
        }

        // In real implementation:
        // 1. Save pruned model
        // 2. Export to desired format (SafeTensors, GGUF, etc.)

        self.pipeline.advance();
        Ok(())
    }

    /// Run the full prune-finetune pipeline.
    pub fn run(&mut self) -> Result<PruningMetrics, String> {
        self.initialize()?;
        self.calibrate()?;
        self.prune()?;

        if self.config.pruning.fine_tune_after_pruning() {
            self.finetune()?;
        }

        if self.config.evaluate_pre_post {
            self.evaluate()?;
        }

        self.export()?;

        Ok(self.metrics().clone())
    }

    /// Reset the trainer to initial state.
    pub fn reset(&mut self) {
        self.pipeline.reset();
        self.calibration = None;
        self.current_epoch = 0;
        self.data_loader.reset();
    }
}

impl Clone for PruneTrainer {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            pipeline: self.pipeline.clone(),
            data_loader: self.data_loader.clone(),
            calibration: self.calibration.clone(),
            current_epoch: self.current_epoch,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prune::config::PruneMethod;

    fn default_config() -> PruneTrainerConfig {
        PruneTrainerConfig::new()
            .with_pruning(PruningConfig::default().with_target_sparsity(0.5))
            .with_calibration(CalibrationDataConfig::new().with_num_samples(5))
    }

    // =========================================================================
    // PruneTrainerConfig Tests
    // =========================================================================

    #[test]
    fn test_config_default() {
        // TEST_ID: TI-001
        let config = PruneTrainerConfig::default();
        assert_eq!(config.finetune_epochs, 1);
        assert!((config.finetune_lr - 1e-5).abs() < 1e-10);
        assert!(config.evaluate_pre_post);
        assert!(!config.save_checkpoints);
    }

    #[test]
    fn test_config_builder() {
        // TEST_ID: TI-002
        let config = PruneTrainerConfig::new()
            .with_finetune_epochs(5)
            .with_finetune_lr(1e-4)
            .with_evaluate(false)
            .with_checkpoint_dir("/tmp/checkpoints")
            .with_save_checkpoints(true);

        assert_eq!(config.finetune_epochs, 5);
        assert!((config.finetune_lr - 1e-4).abs() < 1e-10);
        assert!(!config.evaluate_pre_post);
        assert_eq!(config.checkpoint_dir, Some("/tmp/checkpoints".to_string()));
        assert!(config.save_checkpoints);
    }

    #[test]
    fn test_config_validate_valid() {
        // TEST_ID: TI-003
        let config = default_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_invalid_lr() {
        // TEST_ID: TI-004
        let config = PruneTrainerConfig::new().with_finetune_lr(0.0);
        assert!(
            config.validate().is_err(),
            "TI-004 FALSIFIED: Zero LR should be invalid"
        );
    }

    #[test]
    fn test_config_serialize() {
        // TEST_ID: TI-005
        let config = default_config();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: PruneTrainerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.finetune_epochs, deserialized.finetune_epochs);
    }

    // =========================================================================
    // PruneTrainer Tests
    // =========================================================================

    #[test]
    fn test_trainer_new() {
        // TEST_ID: TI-010
        let config = default_config();
        let trainer = PruneTrainer::new(config);

        assert_eq!(trainer.stage(), PruningStage::Idle);
        assert!(!trainer.is_complete());
        assert_eq!(trainer.current_epoch(), 0);
    }

    #[test]
    fn test_trainer_initialize() {
        // TEST_ID: TI-011
        let config = default_config();
        let mut trainer = PruneTrainer::new(config);

        let result = trainer.initialize();
        assert!(
            result.is_ok(),
            "TI-011 FALSIFIED: Initialize should succeed"
        );
    }

    #[test]
    fn test_trainer_calibrate() {
        // TEST_ID: TI-012
        let config = default_config();
        let mut trainer = PruneTrainer::new(config);

        trainer.initialize().unwrap();
        let result = trainer.calibrate();
        assert!(result.is_ok(), "TI-012 FALSIFIED: Calibrate should succeed");
    }

    #[test]
    fn test_trainer_prune() {
        // TEST_ID: TI-013
        let config = default_config();
        let mut trainer = PruneTrainer::new(config);

        trainer.initialize().unwrap();
        trainer.calibrate().unwrap();
        let result = trainer.prune();
        assert!(result.is_ok(), "TI-013 FALSIFIED: Prune should succeed");
    }

    #[test]
    fn test_trainer_finetune() {
        // TEST_ID: TI-014
        let config = default_config().with_finetune_epochs(3);
        let mut trainer = PruneTrainer::new(config);

        trainer.initialize().unwrap();
        trainer.calibrate().unwrap();
        trainer.prune().unwrap();
        let result = trainer.finetune();
        assert!(result.is_ok(), "TI-014 FALSIFIED: Finetune should succeed");

        assert_eq!(
            trainer.metrics().finetune_losses.len(),
            3,
            "TI-014 FALSIFIED: Should have 3 loss entries"
        );
    }

    #[test]
    fn test_trainer_evaluate() {
        // TEST_ID: TI-015
        let config = default_config().with_pruning(
            PruningConfig::default()
                .with_target_sparsity(0.5)
                .with_fine_tune(false),
        );
        let mut trainer = PruneTrainer::new(config);

        trainer.initialize().unwrap();
        trainer.calibrate().unwrap();
        trainer.prune().unwrap();
        let result = trainer.evaluate();
        assert!(result.is_ok(), "TI-015 FALSIFIED: Evaluate should succeed");
    }

    #[test]
    fn test_trainer_full_run() {
        // TEST_ID: TI-016
        let config = default_config().with_finetune_epochs(2);
        let mut trainer = PruneTrainer::new(config);

        let result = trainer.run();
        assert!(result.is_ok(), "TI-016 FALSIFIED: Full run should succeed");
        assert!(trainer.is_complete());
        assert!(trainer.succeeded());

        let metrics = result.unwrap();
        assert!((metrics.target_sparsity - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_trainer_skip_finetune() {
        // TEST_ID: TI-017
        let config = default_config().with_pruning(
            PruningConfig::default()
                .with_target_sparsity(0.5)
                .with_fine_tune(false),
        );
        let mut trainer = PruneTrainer::new(config);

        let result = trainer.run();
        assert!(result.is_ok());

        // Should not have fine-tuning losses
        assert!(
            trainer.metrics().finetune_losses.is_empty(),
            "TI-017 FALSIFIED: Should skip fine-tuning"
        );
    }

    #[test]
    fn test_trainer_reset() {
        // TEST_ID: TI-018
        let config = default_config();
        let mut trainer = PruneTrainer::new(config);

        trainer.run().unwrap();
        assert!(trainer.is_complete());

        trainer.reset();
        assert!(!trainer.is_complete());
        assert_eq!(trainer.stage(), PruningStage::Idle);
        assert_eq!(trainer.current_epoch(), 0);
    }

    #[test]
    fn test_trainer_metrics_access() {
        // TEST_ID: TI-019
        let config = default_config();
        let mut trainer = PruneTrainer::new(config);

        trainer.run().unwrap();
        let metrics = trainer.metrics();
        assert!((metrics.target_sparsity - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_trainer_pipeline_access() {
        // TEST_ID: TI-020
        let config = default_config();
        let mut trainer = PruneTrainer::new(config);

        trainer.run().unwrap();
        assert_eq!(trainer.pipeline().stage(), PruningStage::Complete);
    }

    #[test]
    fn test_trainer_clone() {
        // TEST_ID: TI-021
        let config = default_config();
        let trainer = PruneTrainer::new(config);
        let cloned = trainer.clone();

        assert_eq!(trainer.stage(), cloned.stage());
        assert_eq!(trainer.current_epoch(), cloned.current_epoch());
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn test_trainer_prune_wrong_stage() {
        // TEST_ID: TI-030
        let config = default_config();
        let mut trainer = PruneTrainer::new(config);

        // Try to prune without initialization
        let result = trainer.prune();
        assert!(
            result.is_err(),
            "TI-030 FALSIFIED: Should fail when pruning in wrong stage"
        );
    }

    #[test]
    fn test_trainer_finetune_wrong_stage() {
        // TEST_ID: TI-031
        let config = default_config();
        let mut trainer = PruneTrainer::new(config);

        // Try to finetune without pruning
        let result = trainer.finetune();
        assert!(
            result.is_err(),
            "TI-031 FALSIFIED: Should fail when finetuning in wrong stage"
        );
    }

    #[test]
    fn test_trainer_evaluate_wrong_stage() {
        // TEST_ID: TI-032
        let config = default_config();
        let mut trainer = PruneTrainer::new(config);

        let result = trainer.evaluate();
        assert!(
            result.is_err(),
            "TI-032 FALSIFIED: Should fail when evaluating in wrong stage"
        );
    }

    #[test]
    fn test_trainer_export_wrong_stage() {
        // TEST_ID: TI-033
        let config = default_config();
        let mut trainer = PruneTrainer::new(config);

        let result = trainer.export();
        assert!(
            result.is_err(),
            "TI-033 FALSIFIED: Should fail when exporting in wrong stage"
        );
    }

    // =========================================================================
    // Calibration Tests
    // =========================================================================

    #[test]
    fn test_trainer_calibration_required_for_wanda() {
        // TEST_ID: TI-040
        let config = default_config().with_pruning(
            PruningConfig::default()
                .with_method(PruneMethod::Wanda)
                .with_target_sparsity(0.5),
        );
        let mut trainer = PruneTrainer::new(config);

        trainer.initialize().unwrap();
        assert!(
            trainer.calibration.is_some(),
            "TI-040 FALSIFIED: Wanda should require calibration"
        );
    }

    #[test]
    fn test_trainer_no_calibration_for_magnitude() {
        // TEST_ID: TI-041
        let config = default_config().with_pruning(
            PruningConfig::default()
                .with_method(PruneMethod::Magnitude)
                .with_target_sparsity(0.5),
        );
        let mut trainer = PruneTrainer::new(config);

        trainer.initialize().unwrap();
        // Magnitude doesn't require calibration
        assert!(
            trainer.calibration.is_none(),
            "TI-041 FALSIFIED: Magnitude should not require calibration"
        );
    }
}
