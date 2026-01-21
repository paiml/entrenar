//! Prune-Finetune-Export pipeline
//!
//! Provides end-to-end workflows for pruning models:
//! 1. Calibrate: Collect activation statistics
//! 2. Prune: Apply pruning with selected method
//! 3. Finetune: Recover accuracy with brief training
//! 4. Export: Save pruned model
//!
//! # Toyota Way Principles
//! - **Heijunka** (Level Loading): Balanced sparsity across layers
//! - **Hansei** (Reflection): Post-pruning evaluation and metrics

use crate::prune::calibrate::CalibrationCollector;
use crate::prune::config::PruningConfig;
use serde::{Deserialize, Serialize};

/// Current stage of the pruning pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PruningStage {
    /// Not started.
    #[default]
    Idle,
    /// Collecting calibration data.
    Calibrating,
    /// Computing importance scores.
    ComputingImportance,
    /// Applying pruning masks.
    Pruning,
    /// Fine-tuning after pruning.
    FineTuning,
    /// Evaluating pruned model.
    Evaluating,
    /// Exporting pruned model.
    Exporting,
    /// Pipeline complete.
    Complete,
    /// Pipeline failed.
    Failed,
}

impl PruningStage {
    /// Check if the pipeline is in an active (non-terminal) state.
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            PruningStage::Calibrating
                | PruningStage::ComputingImportance
                | PruningStage::Pruning
                | PruningStage::FineTuning
                | PruningStage::Evaluating
                | PruningStage::Exporting
        )
    }

    /// Check if the pipeline is complete (success or failure).
    pub fn is_terminal(&self) -> bool {
        matches!(self, PruningStage::Complete | PruningStage::Failed)
    }

    /// Get display name for the stage.
    pub fn display_name(&self) -> &'static str {
        match self {
            PruningStage::Idle => "Idle",
            PruningStage::Calibrating => "Calibrating",
            PruningStage::ComputingImportance => "Computing Importance",
            PruningStage::Pruning => "Pruning",
            PruningStage::FineTuning => "Fine-Tuning",
            PruningStage::Evaluating => "Evaluating",
            PruningStage::Exporting => "Exporting",
            PruningStage::Complete => "Complete",
            PruningStage::Failed => "Failed",
        }
    }
}

/// Metrics collected during pruning.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PruningMetrics {
    /// Achieved sparsity (0.0 to 1.0).
    pub achieved_sparsity: f32,
    /// Target sparsity.
    pub target_sparsity: f32,
    /// Total parameters in model.
    pub total_parameters: usize,
    /// Parameters pruned (set to zero).
    pub parameters_pruned: usize,
    /// Parameters remaining (non-zero).
    pub parameters_remaining: usize,
    /// Per-layer sparsity.
    pub layer_sparsity: Vec<(String, f32)>,
    /// Pre-pruning perplexity (if evaluated).
    pub pre_prune_ppl: Option<f32>,
    /// Post-pruning perplexity (if evaluated).
    pub post_prune_ppl: Option<f32>,
    /// Perplexity increase percentage.
    pub ppl_increase_pct: Option<f32>,
    /// Fine-tuning loss curve.
    pub finetune_losses: Vec<f32>,
    /// Duration of each stage in seconds.
    pub stage_durations: Vec<(PruningStage, f64)>,
}

impl PruningMetrics {
    /// Create new metrics with target sparsity.
    pub fn new(target_sparsity: f32) -> Self {
        Self {
            target_sparsity,
            ..Default::default()
        }
    }

    /// Update achieved sparsity and parameter counts.
    pub fn update_sparsity(&mut self, pruned: usize, total: usize) {
        self.total_parameters = total;
        self.parameters_pruned = pruned;
        self.parameters_remaining = total.saturating_sub(pruned);
        self.achieved_sparsity = if total > 0 {
            pruned as f32 / total as f32
        } else {
            0.0
        };
    }

    /// Add layer sparsity.
    pub fn add_layer_sparsity(&mut self, name: impl Into<String>, sparsity: f32) {
        self.layer_sparsity.push((name.into(), sparsity));
    }

    /// Set pre-pruning perplexity.
    pub fn set_pre_prune_ppl(&mut self, ppl: f32) {
        self.pre_prune_ppl = Some(ppl);
    }

    /// Set post-pruning perplexity and compute increase.
    pub fn set_post_prune_ppl(&mut self, ppl: f32) {
        self.post_prune_ppl = Some(ppl);
        if let Some(pre) = self.pre_prune_ppl {
            if pre > 0.0 {
                self.ppl_increase_pct = Some((ppl - pre) / pre * 100.0);
            }
        }
    }

    /// Record a fine-tuning loss.
    pub fn record_finetune_loss(&mut self, loss: f32) {
        self.finetune_losses.push(loss);
    }

    /// Record stage duration.
    pub fn record_stage_duration(&mut self, stage: PruningStage, duration_secs: f64) {
        self.stage_durations.push((stage, duration_secs));
    }

    /// Get sparsity gap (target - achieved).
    pub fn sparsity_gap(&self) -> f32 {
        self.target_sparsity - self.achieved_sparsity
    }

    /// Check if target sparsity was achieved.
    pub fn target_achieved(&self) -> bool {
        self.achieved_sparsity >= self.target_sparsity - 1e-4
    }

    /// Get mean layer sparsity.
    pub fn mean_layer_sparsity(&self) -> f32 {
        if self.layer_sparsity.is_empty() {
            return self.achieved_sparsity;
        }
        let sum: f32 = self.layer_sparsity.iter().map(|(_, s)| s).sum();
        sum / self.layer_sparsity.len() as f32
    }

    /// Get sparsity variance across layers.
    pub fn layer_sparsity_variance(&self) -> f32 {
        if self.layer_sparsity.is_empty() {
            return 0.0;
        }
        let mean = self.mean_layer_sparsity();
        let variance: f32 = self
            .layer_sparsity
            .iter()
            .map(|(_, s)| (s - mean).powi(2))
            .sum::<f32>()
            / self.layer_sparsity.len() as f32;
        variance
    }

    /// Get total pipeline duration.
    pub fn total_duration_secs(&self) -> f64 {
        self.stage_durations.iter().map(|(_, d)| d).sum()
    }
}

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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prune::calibrate::CalibrationConfig;

    // =========================================================================
    // PruningStage Tests
    // =========================================================================

    #[test]
    fn test_stage_is_active() {
        // TEST_ID: PL-001
        assert!(
            !PruningStage::Idle.is_active(),
            "PL-001 FALSIFIED: Idle should not be active"
        );
        assert!(
            PruningStage::Calibrating.is_active(),
            "PL-001 FALSIFIED: Calibrating should be active"
        );
        assert!(
            PruningStage::Pruning.is_active(),
            "PL-001 FALSIFIED: Pruning should be active"
        );
        assert!(
            !PruningStage::Complete.is_active(),
            "PL-001 FALSIFIED: Complete should not be active"
        );
        assert!(
            !PruningStage::Failed.is_active(),
            "PL-001 FALSIFIED: Failed should not be active"
        );
    }

    #[test]
    fn test_stage_is_terminal() {
        // TEST_ID: PL-002
        assert!(
            !PruningStage::Idle.is_terminal(),
            "PL-002 FALSIFIED: Idle should not be terminal"
        );
        assert!(
            !PruningStage::Pruning.is_terminal(),
            "PL-002 FALSIFIED: Pruning should not be terminal"
        );
        assert!(
            PruningStage::Complete.is_terminal(),
            "PL-002 FALSIFIED: Complete should be terminal"
        );
        assert!(
            PruningStage::Failed.is_terminal(),
            "PL-002 FALSIFIED: Failed should be terminal"
        );
    }

    #[test]
    fn test_stage_display_names() {
        // TEST_ID: PL-003
        assert_eq!(PruningStage::Idle.display_name(), "Idle");
        assert_eq!(PruningStage::Calibrating.display_name(), "Calibrating");
        assert_eq!(PruningStage::Pruning.display_name(), "Pruning");
        assert_eq!(PruningStage::Complete.display_name(), "Complete");
    }

    #[test]
    fn test_stage_default() {
        // TEST_ID: PL-004
        assert_eq!(
            PruningStage::default(),
            PruningStage::Idle,
            "PL-004 FALSIFIED: Default stage should be Idle"
        );
    }

    // =========================================================================
    // PruningMetrics Tests
    // =========================================================================

    #[test]
    fn test_metrics_new() {
        // TEST_ID: PL-010
        let metrics = PruningMetrics::new(0.5);
        assert!(
            (metrics.target_sparsity - 0.5).abs() < 1e-6,
            "PL-010 FALSIFIED: Target sparsity should be 0.5"
        );
        assert_eq!(metrics.achieved_sparsity, 0.0);
        assert_eq!(metrics.total_parameters, 0);
    }

    #[test]
    fn test_metrics_update_sparsity() {
        // TEST_ID: PL-011
        let mut metrics = PruningMetrics::new(0.5);
        metrics.update_sparsity(500, 1000);

        assert_eq!(metrics.total_parameters, 1000);
        assert_eq!(metrics.parameters_pruned, 500);
        assert_eq!(metrics.parameters_remaining, 500);
        assert!(
            (metrics.achieved_sparsity - 0.5).abs() < 1e-6,
            "PL-011 FALSIFIED: Achieved sparsity should be 0.5"
        );
    }

    #[test]
    fn test_metrics_update_sparsity_zero_total() {
        // TEST_ID: PL-012
        let mut metrics = PruningMetrics::new(0.5);
        metrics.update_sparsity(0, 0);
        assert_eq!(metrics.achieved_sparsity, 0.0);
    }

    #[test]
    fn test_metrics_layer_sparsity() {
        // TEST_ID: PL-013
        let mut metrics = PruningMetrics::new(0.5);
        metrics.add_layer_sparsity("layer.0", 0.4);
        metrics.add_layer_sparsity("layer.1", 0.6);

        assert_eq!(metrics.layer_sparsity.len(), 2);
        assert_eq!(metrics.layer_sparsity[0].0, "layer.0");
        assert!((metrics.layer_sparsity[0].1 - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_metrics_perplexity() {
        // TEST_ID: PL-014
        let mut metrics = PruningMetrics::new(0.5);

        metrics.set_pre_prune_ppl(10.0);
        assert_eq!(metrics.pre_prune_ppl, Some(10.0));

        metrics.set_post_prune_ppl(12.0);
        assert_eq!(metrics.post_prune_ppl, Some(12.0));

        // 20% increase
        let ppl_increase = metrics.ppl_increase_pct.unwrap();
        assert!(
            (ppl_increase - 20.0).abs() < 1e-4,
            "PL-014 FALSIFIED: PPL increase should be 20%, got {ppl_increase}"
        );
    }

    #[test]
    fn test_metrics_finetune_losses() {
        // TEST_ID: PL-015
        let mut metrics = PruningMetrics::new(0.5);
        metrics.record_finetune_loss(1.0);
        metrics.record_finetune_loss(0.8);
        metrics.record_finetune_loss(0.6);

        assert_eq!(metrics.finetune_losses.len(), 3);
        assert!((metrics.finetune_losses[2] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_metrics_stage_durations() {
        // TEST_ID: PL-016
        let mut metrics = PruningMetrics::new(0.5);
        metrics.record_stage_duration(PruningStage::Calibrating, 10.0);
        metrics.record_stage_duration(PruningStage::Pruning, 5.0);

        assert_eq!(metrics.stage_durations.len(), 2);
        assert!((metrics.total_duration_secs() - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_metrics_sparsity_gap() {
        // TEST_ID: PL-017
        let mut metrics = PruningMetrics::new(0.5);
        metrics.update_sparsity(300, 1000); // 30% achieved

        let gap = metrics.sparsity_gap();
        assert!(
            (gap - 0.2).abs() < 1e-6,
            "PL-017 FALSIFIED: Gap should be 0.2"
        );
    }

    #[test]
    fn test_metrics_target_achieved() {
        // TEST_ID: PL-018
        let mut metrics = PruningMetrics::new(0.5);
        metrics.update_sparsity(400, 1000);
        assert!(
            !metrics.target_achieved(),
            "PL-018 FALSIFIED: 40% should not achieve 50% target"
        );

        metrics.update_sparsity(500, 1000);
        assert!(
            metrics.target_achieved(),
            "PL-018 FALSIFIED: 50% should achieve 50% target"
        );
    }

    #[test]
    fn test_metrics_mean_layer_sparsity() {
        // TEST_ID: PL-019
        let mut metrics = PruningMetrics::new(0.5);
        metrics.add_layer_sparsity("a", 0.3);
        metrics.add_layer_sparsity("b", 0.5);
        metrics.add_layer_sparsity("c", 0.7);

        let mean = metrics.mean_layer_sparsity();
        assert!(
            (mean - 0.5).abs() < 1e-6,
            "PL-019 FALSIFIED: Mean should be 0.5"
        );
    }

    #[test]
    fn test_metrics_layer_sparsity_variance() {
        // TEST_ID: PL-020
        let mut metrics = PruningMetrics::new(0.5);
        metrics.add_layer_sparsity("a", 0.5);
        metrics.add_layer_sparsity("b", 0.5);

        let variance = metrics.layer_sparsity_variance();
        assert!(
            variance < 1e-6,
            "PL-020 FALSIFIED: Variance should be ~0 for uniform sparsity"
        );
    }

    // =========================================================================
    // PruneFinetunePipeline Tests
    // =========================================================================

    #[test]
    fn test_pipeline_new() {
        // TEST_ID: PL-030
        let config = PruningConfig::default();
        let pipeline = PruneFinetunePipeline::new(config);

        assert_eq!(pipeline.stage(), PruningStage::Idle);
        assert!(!pipeline.is_complete());
        assert!(pipeline.error().is_none());
    }

    #[test]
    fn test_pipeline_advance() {
        // TEST_ID: PL-031
        let config = PruningConfig::default();
        let mut pipeline = PruneFinetunePipeline::new(config);

        assert_eq!(pipeline.stage(), PruningStage::Idle);
        pipeline.advance();
        assert_eq!(pipeline.stage(), PruningStage::Calibrating);
        pipeline.advance();
        assert_eq!(pipeline.stage(), PruningStage::ComputingImportance);
        pipeline.advance();
        assert_eq!(pipeline.stage(), PruningStage::Pruning);
        pipeline.advance();
        // Default config has fine_tune_after_pruning = true
        assert_eq!(pipeline.stage(), PruningStage::FineTuning);
        pipeline.advance();
        assert_eq!(pipeline.stage(), PruningStage::Evaluating);
        pipeline.advance();
        assert_eq!(pipeline.stage(), PruningStage::Exporting);
        pipeline.advance();
        assert_eq!(pipeline.stage(), PruningStage::Complete);
    }

    #[test]
    fn test_pipeline_skip_finetune() {
        // TEST_ID: PL-032
        let config = PruningConfig::default().with_fine_tune(false);
        let mut pipeline = PruneFinetunePipeline::new(config);

        // Advance to Pruning
        pipeline.advance(); // Calibrating
        pipeline.advance(); // ComputingImportance
        pipeline.advance(); // Pruning
        pipeline.advance(); // Should skip FineTuning

        assert_eq!(
            pipeline.stage(),
            PruningStage::Evaluating,
            "PL-032 FALSIFIED: Should skip fine-tuning"
        );
    }

    #[test]
    fn test_pipeline_fail() {
        // TEST_ID: PL-033
        let config = PruningConfig::default();
        let mut pipeline = PruneFinetunePipeline::new(config);

        pipeline.fail("Test error");

        assert_eq!(pipeline.stage(), PruningStage::Failed);
        assert!(pipeline.is_complete());
        assert!(pipeline.failed());
        assert!(!pipeline.succeeded());
        assert_eq!(pipeline.error(), Some("Test error"));
    }

    #[test]
    fn test_pipeline_reset() {
        // TEST_ID: PL-034
        let config = PruningConfig::default();
        let mut pipeline = PruneFinetunePipeline::new(config);

        pipeline.advance();
        pipeline.advance();
        pipeline.fail("Error");

        pipeline.reset();

        assert_eq!(pipeline.stage(), PruningStage::Idle);
        assert!(pipeline.error().is_none());
        assert!(!pipeline.is_complete());
    }

    #[test]
    fn test_pipeline_terminal_no_advance() {
        // TEST_ID: PL-035
        let config = PruningConfig::default();
        let mut pipeline = PruneFinetunePipeline::new(config);

        // Advance to complete
        for _ in 0..10 {
            pipeline.advance();
        }
        assert_eq!(pipeline.stage(), PruningStage::Complete);

        // Should stay at Complete
        pipeline.advance();
        assert_eq!(
            pipeline.stage(),
            PruningStage::Complete,
            "PL-035 FALSIFIED: Terminal state should not advance"
        );
    }

    #[test]
    fn test_pipeline_start_calibration() {
        // TEST_ID: PL-036
        let config = PruningConfig::default();
        let mut pipeline = PruneFinetunePipeline::new(config);

        let cal_config = CalibrationConfig::default();
        let calibration = CalibrationCollector::new(cal_config);
        pipeline.start_calibration(calibration);

        assert_eq!(pipeline.stage(), PruningStage::Calibrating);
        assert!(pipeline.calibration().is_some());
    }

    #[test]
    fn test_pipeline_start_calibration_not_idle() {
        // TEST_ID: PL-037
        let config = PruningConfig::default();
        let mut pipeline = PruneFinetunePipeline::new(config);

        pipeline.advance(); // Now Calibrating

        let cal_config = CalibrationConfig::default();
        let calibration = CalibrationCollector::new(cal_config);
        pipeline.start_calibration(calibration);

        // Should not change state or add calibration since not idle
        assert_eq!(
            pipeline.stage(),
            PruningStage::Calibrating,
            "PL-037 FALSIFIED: Should not restart calibration"
        );
    }

    #[test]
    fn test_pipeline_overall_progress() {
        // TEST_ID: PL-038
        let config = PruningConfig::default();
        let mut pipeline = PruneFinetunePipeline::new(config);

        assert!(
            pipeline.overall_progress().abs() < 1e-6,
            "PL-038 FALSIFIED: Idle progress should be 0"
        );

        pipeline.advance();
        let prog = pipeline.overall_progress();
        assert!(
            prog > 0.0 && prog < 0.5,
            "PL-038 FALSIFIED: Calibrating progress should be between 0 and 0.5"
        );

        // Advance to complete
        for _ in 0..10 {
            pipeline.advance();
        }
        assert!(
            (pipeline.overall_progress() - 1.0).abs() < 1e-6,
            "PL-038 FALSIFIED: Complete progress should be 1.0"
        );
    }

    #[test]
    fn test_pipeline_failed_progress() {
        // TEST_ID: PL-039
        let config = PruningConfig::default();
        let mut pipeline = PruneFinetunePipeline::new(config);

        pipeline.advance();
        pipeline.fail("Error");

        assert!(
            pipeline.overall_progress().abs() < 1e-6,
            "PL-039 FALSIFIED: Failed progress should be 0"
        );
    }

    #[test]
    fn test_pipeline_clone() {
        // TEST_ID: PL-040
        let config = PruningConfig::default();
        let mut pipeline = PruneFinetunePipeline::new(config);
        pipeline.advance();

        let cloned = pipeline.clone();
        assert_eq!(
            pipeline.stage(),
            cloned.stage(),
            "PL-040 FALSIFIED: Cloned stage should match"
        );
    }

    #[test]
    fn test_pipeline_metrics_access() {
        // TEST_ID: PL-041
        let config = PruningConfig::default().with_target_sparsity(0.7);
        let mut pipeline = PruneFinetunePipeline::new(config);

        assert!(
            (pipeline.metrics().target_sparsity - 0.7).abs() < 1e-6,
            "PL-041 FALSIFIED: Metrics target should match config"
        );

        pipeline.metrics_mut().update_sparsity(700, 1000);
        assert_eq!(pipeline.metrics().parameters_pruned, 700);
    }

    // =========================================================================
    // Serialization Tests
    // =========================================================================

    #[test]
    fn test_stage_serialize() {
        // TEST_ID: PL-050
        let stage = PruningStage::Calibrating;
        let json = serde_json::to_string(&stage).unwrap();
        let deserialized: PruningStage = serde_json::from_str(&json).unwrap();
        assert_eq!(stage, deserialized);
    }

    #[test]
    fn test_metrics_serialize() {
        // TEST_ID: PL-051
        let mut metrics = PruningMetrics::new(0.5);
        metrics.update_sparsity(500, 1000);
        metrics.add_layer_sparsity("layer.0", 0.5);

        let json = serde_json::to_string(&metrics).unwrap();
        let deserialized: PruningMetrics = serde_json::from_str(&json).unwrap();

        assert!(
            (deserialized.achieved_sparsity - 0.5).abs() < 1e-6,
            "PL-051 FALSIFIED: Serialization roundtrip failed"
        );
    }
}
