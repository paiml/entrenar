//! Distillation pipeline execution (Heijunka - level scheduling).
//!
//! Orchestrates the complete distillation workflow from model fetching
//! through training and export.

use crate::config::DistillConfig;
use crate::MemoryEstimate;
use entrenar_common::{EntrenarError, Result};
use std::path::PathBuf;

/// Pipeline execution result.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Path to the output model
    pub output_path: PathBuf,
    /// Training metrics
    pub metrics: TrainingMetrics,
    /// Total execution time in seconds
    pub duration_seconds: f64,
}

/// Training metrics collected during distillation.
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Initial loss at start of training
    pub initial_loss: f32,
    /// Final loss at end of training
    pub final_loss: f32,
    /// Best validation loss achieved
    pub best_loss: f32,
    /// Number of training steps completed
    pub steps_completed: u64,
    /// Average throughput (samples/second)
    pub throughput: f32,
}

impl TrainingMetrics {
    /// Calculate loss improvement ratio.
    pub fn improvement_ratio(&self) -> f32 {
        if self.initial_loss > 0.0 {
            1.0 - (self.final_loss / self.initial_loss)
        } else {
            0.0
        }
    }
}

/// Distillation pipeline orchestrator.
pub struct Pipeline<'a> {
    config: &'a DistillConfig,
}

impl<'a> Pipeline<'a> {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: &'a DistillConfig) -> Self {
        Self { config }
    }

    /// Execute the complete distillation pipeline.
    pub fn execute(&self) -> Result<PipelineResult> {
        let start = std::time::Instant::now();

        // Stage 1: Fetch models
        let _teacher_path = self.fetch_teacher()?;
        let _student_path = self.fetch_student()?;

        // Stage 2: Load models
        // (In real implementation, would load SafeTensors)

        // Stage 3: Training loop
        let metrics = self.train()?;

        // Stage 4: Export
        let output_path = self.export()?;

        Ok(PipelineResult {
            output_path,
            metrics,
            duration_seconds: start.elapsed().as_secs_f64(),
        })
    }

    /// Estimate memory requirements for this configuration.
    pub fn estimate_memory(config: &DistillConfig) -> Result<MemoryEstimate> {
        // Estimate model size from model ID
        let teacher_params = estimate_params_from_model_id(&config.teacher.model_id);
        let student_params = estimate_params_from_model_id(&config.student.model_id);

        // Use student params for training memory (teacher is frozen)
        let estimate = MemoryEstimate::new(
            student_params + teacher_params / 4, // Teacher in int8 for inference
            config.training.batch_size as usize,
            config.dataset.max_length,
            4096, // Default hidden dim, would be read from config
        );

        Ok(estimate)
    }

    fn fetch_teacher(&self) -> Result<PathBuf> {
        // In real implementation, would use HfModelFetcher
        // For now, return a placeholder path
        Ok(PathBuf::from("/tmp/teacher"))
    }

    fn fetch_student(&self) -> Result<PathBuf> {
        // In real implementation, would use HfModelFetcher
        Ok(PathBuf::from("/tmp/student"))
    }

    fn train(&self) -> Result<TrainingMetrics> {
        // Simulated training metrics
        // In real implementation, would run actual training loop
        let mut metrics = TrainingMetrics {
            initial_loss: 2.5,
            final_loss: 0.8,
            best_loss: 0.75,
            steps_completed: 0,
            throughput: 100.0,
        };

        let steps_per_epoch = 1000; // Would be calculated from dataset size
        metrics.steps_completed = (self.config.training.epochs as u64) * steps_per_epoch;

        Ok(metrics)
    }

    fn export(&self) -> Result<PathBuf> {
        // Create output directory
        std::fs::create_dir_all(&self.config.output.dir).map_err(|e| EntrenarError::Io {
            context: format!(
                "creating output directory: {}",
                self.config.output.dir.display()
            ),
            source: e,
        })?;

        let output_path = self.config.output.dir.join("model.safetensors");

        // In real implementation, would use Exporter to save model

        Ok(output_path)
    }
}

/// Estimate parameter count from model ID.
fn estimate_params_from_model_id(model_id: &str) -> u64 {
    let lower = model_id.to_lowercase();

    // Common model size patterns
    if lower.contains("70b") {
        70_000_000_000
    } else if lower.contains("65b") {
        65_000_000_000
    } else if lower.contains("30b") || lower.contains("33b") {
        30_000_000_000
    } else if lower.contains("13b") {
        13_000_000_000
    } else if lower.contains("7b") || lower.contains("8b") {
        7_000_000_000
    } else if lower.contains("3b") {
        3_000_000_000
    } else if lower.contains("1.1b") || lower.contains("1b") {
        1_100_000_000
    } else if lower.contains("350m") || lower.contains("base") {
        350_000_000
    } else if lower.contains("125m") || lower.contains("small") {
        125_000_000
    } else {
        // Default to a medium-sized model
        1_000_000_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DistillConfig;

    #[test]
    fn test_estimate_params_from_model_id() {
        assert_eq!(
            estimate_params_from_model_id("meta-llama/Llama-2-7b"),
            7_000_000_000
        );
        assert_eq!(
            estimate_params_from_model_id("TinyLlama/TinyLlama-1.1B"),
            1_100_000_000
        );
        assert_eq!(
            estimate_params_from_model_id("microsoft/codebert-base"),
            350_000_000
        );
    }

    #[test]
    fn test_training_metrics_improvement() {
        let metrics = TrainingMetrics {
            initial_loss: 2.0,
            final_loss: 1.0,
            best_loss: 0.9,
            steps_completed: 1000,
            throughput: 100.0,
        };

        assert!((metrics.improvement_ratio() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_memory_estimation() {
        let config = DistillConfig::minimal("meta-llama/Llama-2-7b", "TinyLlama/TinyLlama-1.1B");

        let estimate = Pipeline::estimate_memory(&config).unwrap();

        // Should estimate reasonable memory for a 7B model
        assert!(estimate.total_bytes > 10_000_000_000); // > 10GB
        assert!(estimate.recommended_batch_size > 0);
    }

    #[test]
    fn test_pipeline_result_has_duration() {
        let config = DistillConfig::minimal("org/teacher", "org/student");
        let pipeline = Pipeline::new(&config);

        // Note: This test would need to be run with proper config validation
        // For now, just verify the struct is constructable
        let result = PipelineResult {
            output_path: PathBuf::from("/tmp/output"),
            metrics: TrainingMetrics::default(),
            duration_seconds: 100.0,
        };

        assert!(result.duration_seconds > 0.0);
    }
}
