//! Prune-Finetune pipeline orchestrator
//!
//! Manages the full pruning workflow from calibration through export.

use super::metrics::PruningMetrics;
use super::stage::PruningStage;
use crate::prune::calibrate::CalibrationCollector;
use crate::prune::config::PruningConfig;

/// Prune-Finetune pipeline orchestrator.
///
/// Manages the full pruning workflow from calibration through export.
#[derive(Debug)]
pub struct PruneFinetunePipeline {
    /// Configuration.
    config: PruningConfig,
    /// Current stage.
    stage: PruningStage,
    /// Collected metrics.
    metrics: PruningMetrics,
    /// Calibration collector.
    calibration: Option<CalibrationCollector>,
    /// Error message if failed.
    error: Option<String>,
}

impl PruneFinetunePipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: PruningConfig) -> Self {
        let metrics = PruningMetrics::new(config.target_sparsity());
        Self {
            config,
            stage: PruningStage::Idle,
            metrics,
            calibration: None,
            error: None,
        }
    }

    /// Get the current stage.
    pub fn stage(&self) -> PruningStage {
        self.stage
    }

    /// Get the configuration.
    pub fn config(&self) -> &PruningConfig {
        &self.config
    }

    /// Get the collected metrics.
    pub fn metrics(&self) -> &PruningMetrics {
        &self.metrics
    }

    /// Get mutable access to metrics.
    pub fn metrics_mut(&mut self) -> &mut PruningMetrics {
        &mut self.metrics
    }

    /// Get the error message if failed.
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Start the calibration stage.
    pub fn start_calibration(&mut self, calibration: CalibrationCollector) {
        if self.stage != PruningStage::Idle {
            return;
        }
        self.calibration = Some(calibration);
        self.stage = PruningStage::Calibrating;
    }

    /// Advance to the next stage.
    pub fn advance(&mut self) {
        self.stage = match self.stage {
            PruningStage::Idle => PruningStage::Calibrating,
            PruningStage::Calibrating => PruningStage::ComputingImportance,
            PruningStage::ComputingImportance => PruningStage::Pruning,
            PruningStage::Pruning => {
                if self.config.fine_tune_after_pruning() {
                    PruningStage::FineTuning
                } else {
                    PruningStage::Evaluating
                }
            }
            PruningStage::FineTuning => PruningStage::Evaluating,
            PruningStage::Evaluating => PruningStage::Exporting,
            PruningStage::Exporting => PruningStage::Complete,
            // Terminal states don't advance
            PruningStage::Complete | PruningStage::Failed => self.stage,
        };
    }

    /// Mark the pipeline as failed with an error message.
    pub fn fail(&mut self, error: impl Into<String>) {
        self.error = Some(error.into());
        self.stage = PruningStage::Failed;
    }

    /// Reset the pipeline to idle state.
    pub fn reset(&mut self) {
        self.stage = PruningStage::Idle;
        self.metrics = PruningMetrics::new(self.config.target_sparsity());
        self.calibration = None;
        self.error = None;
    }

    /// Check if the pipeline is complete (success or failure).
    pub fn is_complete(&self) -> bool {
        self.stage.is_terminal()
    }

    /// Check if the pipeline succeeded.
    pub fn succeeded(&self) -> bool {
        self.stage == PruningStage::Complete
    }

    /// Check if the pipeline failed.
    pub fn failed(&self) -> bool {
        self.stage == PruningStage::Failed
    }

    /// Get calibration collector if available.
    pub fn calibration(&self) -> Option<&CalibrationCollector> {
        self.calibration.as_ref()
    }

    /// Get calibration progress (0.0 to 1.0).
    pub fn calibration_progress(&self) -> f32 {
        self.calibration
            .as_ref()
            .map_or(0.0, CalibrationCollector::progress)
    }

    /// Get overall pipeline progress (0.0 to 1.0).
    pub fn overall_progress(&self) -> f32 {
        match self.stage {
            PruningStage::Idle => 0.0,
            PruningStage::Calibrating => 0.1 + 0.1 * self.calibration_progress(),
            PruningStage::ComputingImportance => 0.25,
            PruningStage::Pruning => 0.4,
            PruningStage::FineTuning => 0.6,
            PruningStage::Evaluating => 0.8,
            PruningStage::Exporting => 0.95,
            PruningStage::Complete => 1.0,
            PruningStage::Failed => 0.0, // Reset on failure
        }
    }
}

impl Clone for PruneFinetunePipeline {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            stage: self.stage,
            metrics: self.metrics.clone(),
            calibration: self.calibration.clone(),
            error: self.error.clone(),
        }
    }
}
