//! Distillation Trainer Orchestrator
//!
//! High-level API for knowledge distillation training.
//!
//! # Example
//!
//! ```ignore
//! use entrenar::hf_pipeline::{DistillationTrainer, TrainerConfig};
//!
//! let trainer = DistillationTrainer::new(TrainerConfig {
//!     teacher_model: "microsoft/codebert-base".into(),
//!     student_model: "distilbert-base-uncased".into(),
//!     ..Default::default()
//! });
//!
//! trainer.train(&dataset)?;
//! ```

use crate::hf_pipeline::distillation::{
    AttentionTransfer, DistillationLoss, ProgressiveDistillation,
};
use crate::hf_pipeline::fine_tune::{FineTuneConfig, FineTuneMethod};
use crate::hf_pipeline::loader::TeacherModel;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Distillation training configuration
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// Teacher model ID or path
    pub teacher_model: String,
    /// Student model ID or path
    pub student_model: String,
    /// Output directory for checkpoints and logs
    pub output_dir: PathBuf,
    /// Distillation loss configuration
    pub distillation_loss: DistillationLoss,
    /// Progressive distillation (hidden state matching)
    pub progressive: Option<ProgressiveDistillation>,
    /// Attention transfer
    pub attention_transfer: Option<AttentionTransfer>,
    /// Fine-tuning configuration for student
    pub fine_tune: FineTuneConfig,
    /// Number of training epochs
    pub epochs: usize,
    /// Steps per epoch (0 = auto-detect from dataset)
    pub steps_per_epoch: usize,
    /// Logging frequency (steps)
    pub log_every_n_steps: usize,
    /// Checkpoint frequency (steps)
    pub save_every_n_steps: usize,
    /// Evaluation frequency (steps)
    pub eval_every_n_steps: usize,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f32,
    /// Random seed
    pub seed: u64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            teacher_model: String::new(),
            student_model: String::new(),
            output_dir: PathBuf::from("./distillation_output"),
            distillation_loss: DistillationLoss::default(),
            progressive: None,
            attention_transfer: None,
            fine_tune: FineTuneConfig::default(),
            epochs: 3,
            steps_per_epoch: 0,
            log_every_n_steps: 10,
            save_every_n_steps: 500,
            eval_every_n_steps: 100,
            max_grad_norm: 1.0,
            seed: 42,
        }
    }
}

impl TrainerConfig {
    /// Create new trainer config with teacher and student models
    #[must_use]
    pub fn new(teacher: impl Into<String>, student: impl Into<String>) -> Self {
        Self {
            teacher_model: teacher.into(),
            student_model: student.into(),
            ..Default::default()
        }
    }

    /// Set temperature for distillation
    #[must_use]
    pub fn temperature(mut self, temp: f32) -> Self {
        self.distillation_loss = DistillationLoss::new(temp, self.distillation_loss.alpha);
        self
    }

    /// Set alpha for soft vs hard loss weight
    #[must_use]
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.distillation_loss = DistillationLoss::new(self.distillation_loss.temperature, alpha);
        self
    }

    /// Enable progressive distillation with layer mapping
    #[must_use]
    pub fn with_progressive(mut self, layer_mapping: Vec<(usize, usize)>) -> Self {
        self.progressive = Some(ProgressiveDistillation::new(layer_mapping));
        self
    }

    /// Enable attention transfer
    #[must_use]
    pub fn with_attention_transfer(mut self, weight: f32) -> Self {
        self.attention_transfer = Some(AttentionTransfer::new(weight));
        self
    }

    /// Set output directory
    #[must_use]
    pub fn output_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.output_dir = path.into();
        self
    }

    /// Set number of epochs
    #[must_use]
    pub fn epochs(mut self, n: usize) -> Self {
        self.epochs = n;
        self
    }
}

/// Training state tracking
#[derive(Debug, Clone)]
pub struct TrainingState {
    /// Current epoch (0-indexed)
    pub epoch: usize,
    /// Current global step
    pub global_step: usize,
    /// Steps completed in current epoch
    pub epoch_step: usize,
    /// Best validation loss seen
    pub best_val_loss: f32,
    /// Training start time
    pub start_time: Instant,
    /// Loss history (step, loss)
    pub loss_history: Vec<(usize, f32)>,
    /// Validation loss history (step, loss)
    pub val_loss_history: Vec<(usize, f32)>,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingState {
    /// Create new training state
    #[must_use]
    pub fn new() -> Self {
        Self {
            epoch: 0,
            global_step: 0,
            epoch_step: 0,
            best_val_loss: f32::INFINITY,
            start_time: Instant::now(),
            loss_history: Vec::new(),
            val_loss_history: Vec::new(),
        }
    }

    /// Record training loss
    pub fn record_loss(&mut self, loss: f32) {
        self.loss_history.push((self.global_step, loss));
    }

    /// Record validation loss
    pub fn record_val_loss(&mut self, loss: f32) -> bool {
        self.val_loss_history.push((self.global_step, loss));
        if loss < self.best_val_loss {
            self.best_val_loss = loss;
            true // New best
        } else {
            false
        }
    }

    /// Advance one step
    pub fn step(&mut self) {
        self.global_step += 1;
        self.epoch_step += 1;
    }

    /// Start new epoch
    pub fn new_epoch(&mut self) {
        self.epoch += 1;
        self.epoch_step = 0;
    }

    /// Get elapsed time
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get average loss over last N steps
    #[must_use]
    pub fn avg_loss(&self, n: usize) -> Option<f32> {
        if self.loss_history.is_empty() {
            return None;
        }
        let start = self.loss_history.len().saturating_sub(n);
        let sum: f32 = self.loss_history[start..].iter().map(|(_, l)| l).sum();
        Some(sum / (self.loss_history.len() - start) as f32)
    }

    /// Get steps per second
    #[must_use]
    pub fn steps_per_second(&self) -> f32 {
        let elapsed = self.elapsed().as_secs_f32();
        if elapsed > 0.0 {
            self.global_step as f32 / elapsed
        } else {
            0.0
        }
    }

    /// Get estimated time remaining
    #[must_use]
    pub fn eta(&self, total_steps: usize) -> Duration {
        let sps = self.steps_per_second();
        if sps > 0.0 {
            let remaining = total_steps.saturating_sub(self.global_step);
            Duration::from_secs_f32(remaining as f32 / sps)
        } else {
            Duration::ZERO
        }
    }
}

/// Knowledge Distillation Trainer
///
/// Orchestrates the training loop for distilling knowledge from
/// a teacher model to a student model.
pub struct DistillationTrainer<T: TeacherModel> {
    /// Configuration
    pub config: TrainerConfig,
    /// Teacher model
    teacher: T,
    /// Training state
    state: TrainingState,
}

impl<T: TeacherModel> DistillationTrainer<T> {
    /// Create new trainer with teacher model
    pub fn new(config: TrainerConfig, teacher: T) -> Self {
        Self {
            config,
            teacher,
            state: TrainingState::new(),
        }
    }

    /// Get current training state
    #[must_use]
    pub fn state(&self) -> &TrainingState {
        &self.state
    }

    /// Get teacher model reference
    #[must_use]
    pub fn teacher(&self) -> &T {
        &self.teacher
    }

    /// Compute total loss for a batch
    ///
    /// Combines distillation loss with optional progressive and attention transfer.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn compute_loss(
        &self,
        student_logits: &ndarray::Array2<f32>,
        teacher_logits: &ndarray::Array2<f32>,
        targets: &[usize],
        student_hidden: Option<&[ndarray::Array2<f32>]>,
        teacher_hidden: Option<&[ndarray::Array2<f32>]>,
        student_attention: Option<&[ndarray::Array2<f32>]>,
        teacher_attention: Option<&[ndarray::Array2<f32>]>,
    ) -> f32 {
        // Base distillation loss
        let mut total_loss =
            self.config
                .distillation_loss
                .forward(student_logits, teacher_logits, targets);

        // Progressive distillation (hidden state matching)
        if let (Some(prog), Some(sh), Some(th)) =
            (&self.config.progressive, student_hidden, teacher_hidden)
        {
            total_loss += prog.hidden_state_loss(sh, th);
        }

        // Attention transfer
        if let (Some(at), Some(sa), Some(ta)) = (
            &self.config.attention_transfer,
            student_attention,
            teacher_attention,
        ) {
            total_loss += at.loss(sa, ta);
        }

        total_loss
    }

    /// Check if using LoRA/QLoRA for student fine-tuning
    #[must_use]
    pub fn is_parameter_efficient(&self) -> bool {
        matches!(
            self.config.fine_tune.method,
            FineTuneMethod::LoRA(_)
                | FineTuneMethod::QLoRA { .. }
                | FineTuneMethod::PrefixTuning { .. }
        )
    }

    /// Estimate total memory requirements
    #[must_use]
    pub fn estimate_total_memory(&self) -> u64 {
        let teacher_mem = self.teacher.estimate_memory(
            self.config.fine_tune.batch_size,
            self.config.fine_tune.max_seq_length,
        );
        let student_mem = self
            .config
            .fine_tune
            .estimate_memory(self.teacher.param_count() / 4); // Assume 4x smaller student

        teacher_mem.total() + student_mem.total()
    }

    /// Simulate one training step (for testing)
    pub fn simulate_step(&mut self, loss: f32) {
        self.state.record_loss(loss);
        self.state.step();
    }

    /// Simulate epoch boundary
    pub fn simulate_epoch(&mut self) {
        self.state.new_epoch();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf_pipeline::SafeTensorsTeacher;

    // =========================================================================
    // TrainerConfig Tests
    // =========================================================================

    #[test]
    fn test_trainer_config_default() {
        let config = TrainerConfig::default();
        assert!(config.teacher_model.is_empty());
        assert!(config.student_model.is_empty());
        assert_eq!(config.epochs, 3);
        assert_eq!(config.log_every_n_steps, 10);
    }

    #[test]
    fn test_trainer_config_builder() {
        let config = TrainerConfig::new("teacher/model", "student/model")
            .temperature(6.0)
            .alpha(0.8)
            .epochs(5)
            .output_dir("/tmp/distill");

        assert_eq!(config.teacher_model, "teacher/model");
        assert_eq!(config.student_model, "student/model");
        assert_eq!(config.distillation_loss.temperature, 6.0);
        assert_eq!(config.distillation_loss.alpha, 0.8);
        assert_eq!(config.epochs, 5);
    }

    #[test]
    fn test_trainer_config_progressive() {
        let config = TrainerConfig::new("t", "s").with_progressive(vec![(0, 2), (1, 5)]);

        assert!(config.progressive.is_some());
        let prog = config.progressive.unwrap();
        assert_eq!(prog.layer_mapping.len(), 2);
    }

    #[test]
    fn test_trainer_config_attention_transfer() {
        let config = TrainerConfig::new("t", "s").with_attention_transfer(0.2);

        assert!(config.attention_transfer.is_some());
        let at = config.attention_transfer.unwrap();
        assert_eq!(at.weight, 0.2);
    }

    // =========================================================================
    // TrainingState Tests
    // =========================================================================

    #[test]
    fn test_training_state_new() {
        let state = TrainingState::new();
        assert_eq!(state.epoch, 0);
        assert_eq!(state.global_step, 0);
        assert_eq!(state.best_val_loss, f32::INFINITY);
    }

    #[test]
    fn test_training_state_step() {
        let mut state = TrainingState::new();
        state.step();
        assert_eq!(state.global_step, 1);
        assert_eq!(state.epoch_step, 1);

        state.step();
        assert_eq!(state.global_step, 2);
    }

    #[test]
    fn test_training_state_new_epoch() {
        let mut state = TrainingState::new();
        state.step();
        state.step();
        state.new_epoch();

        assert_eq!(state.epoch, 1);
        assert_eq!(state.epoch_step, 0);
        assert_eq!(state.global_step, 2); // Global step unchanged
    }

    #[test]
    fn test_training_state_record_loss() {
        let mut state = TrainingState::new();
        state.record_loss(1.5);
        state.step();
        state.record_loss(1.2);
        state.step();

        assert_eq!(state.loss_history.len(), 2);
        assert_eq!(state.loss_history[0], (0, 1.5));
        assert_eq!(state.loss_history[1], (1, 1.2));
    }

    #[test]
    fn test_training_state_record_val_loss_improvement() {
        let mut state = TrainingState::new();
        let improved = state.record_val_loss(1.0);
        assert!(improved);
        assert_eq!(state.best_val_loss, 1.0);

        let improved = state.record_val_loss(0.8);
        assert!(improved);
        assert_eq!(state.best_val_loss, 0.8);

        let improved = state.record_val_loss(0.9);
        assert!(!improved);
        assert_eq!(state.best_val_loss, 0.8);
    }

    #[test]
    fn test_training_state_avg_loss() {
        let mut state = TrainingState::new();
        state.record_loss(1.0);
        state.step();
        state.record_loss(2.0);
        state.step();
        state.record_loss(3.0);

        let avg = state.avg_loss(2).unwrap();
        assert!((avg - 2.5).abs() < 0.01);

        let avg_all = state.avg_loss(10).unwrap();
        assert!((avg_all - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_training_state_avg_loss_empty() {
        let state = TrainingState::new();
        assert!(state.avg_loss(10).is_none());
    }

    // =========================================================================
    // DistillationTrainer Tests
    // =========================================================================

    #[test]
    fn test_trainer_creation() {
        let config = TrainerConfig::new("teacher", "student");
        let teacher = SafeTensorsTeacher::mock(12, 768);
        let trainer = DistillationTrainer::new(config, teacher);

        assert_eq!(trainer.config.teacher_model, "teacher");
        assert_eq!(trainer.state().global_step, 0);
    }

    #[test]
    fn test_trainer_is_parameter_efficient() {
        let mut config = TrainerConfig::new("t", "s");
        let teacher = SafeTensorsTeacher::mock(12, 768);

        // Default uses LoRA
        let trainer = DistillationTrainer::new(config.clone(), SafeTensorsTeacher::mock(12, 768));
        assert!(trainer.is_parameter_efficient());

        // Full fine-tuning is not parameter efficient
        config.fine_tune = config.fine_tune.full_fine_tune();
        let trainer = DistillationTrainer::new(config, teacher);
        assert!(!trainer.is_parameter_efficient());
    }

    #[test]
    fn test_trainer_compute_loss() {
        use ndarray::Array2;

        let config = TrainerConfig::new("t", "s");
        let teacher = SafeTensorsTeacher::mock(12, 768);
        let trainer = DistillationTrainer::new(config, teacher);

        let student_logits = Array2::from_shape_vec((2, 10), vec![1.0; 20]).unwrap();
        let teacher_logits = Array2::from_shape_vec((2, 10), vec![1.1; 20]).unwrap();
        let targets = vec![5, 3];

        let loss = trainer.compute_loss(
            &student_logits,
            &teacher_logits,
            &targets,
            None,
            None,
            None,
            None,
        );
        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_trainer_compute_loss_with_progressive() {
        use ndarray::Array2;

        let config = TrainerConfig::new("t", "s").with_progressive(vec![(0, 0)]);
        let teacher = SafeTensorsTeacher::mock(12, 768);
        let trainer = DistillationTrainer::new(config, teacher);

        let student_logits = Array2::from_shape_vec((2, 10), vec![1.0; 20]).unwrap();
        let teacher_logits = Array2::from_shape_vec((2, 10), vec![1.1; 20]).unwrap();
        let targets = vec![5, 3];

        let sh = vec![Array2::<f32>::zeros((2, 768))];
        let th = vec![Array2::<f32>::ones((2, 768))];

        let loss = trainer.compute_loss(
            &student_logits,
            &teacher_logits,
            &targets,
            Some(&sh),
            Some(&th),
            None,
            None,
        );

        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_trainer_simulate_step() {
        let config = TrainerConfig::new("t", "s");
        let teacher = SafeTensorsTeacher::mock(12, 768);
        let mut trainer = DistillationTrainer::new(config, teacher);

        trainer.simulate_step(1.5);
        assert_eq!(trainer.state().global_step, 1);
        assert_eq!(trainer.state().loss_history.len(), 1);

        trainer.simulate_step(1.2);
        assert_eq!(trainer.state().global_step, 2);
    }

    #[test]
    fn test_trainer_simulate_epoch() {
        let config = TrainerConfig::new("t", "s");
        let teacher = SafeTensorsTeacher::mock(12, 768);
        let mut trainer = DistillationTrainer::new(config, teacher);

        trainer.simulate_step(1.0);
        trainer.simulate_step(1.0);
        trainer.simulate_epoch();

        assert_eq!(trainer.state().epoch, 1);
        assert_eq!(trainer.state().epoch_step, 0);
    }

    #[test]
    fn test_trainer_estimate_memory() {
        let config = TrainerConfig::new("t", "s");
        let teacher = SafeTensorsTeacher::mock(12, 768);
        let trainer = DistillationTrainer::new(config, teacher);

        let mem = trainer.estimate_total_memory();
        assert!(mem > 0);
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_training_state_default() {
        let state = TrainingState::default();
        assert_eq!(state.epoch, 0);
        assert_eq!(state.global_step, 0);
    }

    #[test]
    fn test_training_state_elapsed() {
        let state = TrainingState::new();
        std::thread::sleep(Duration::from_millis(5));
        assert!(state.elapsed().as_millis() >= 4);
    }

    #[test]
    fn test_training_state_steps_per_second() {
        let mut state = TrainingState::new();
        state.step();
        state.step();
        std::thread::sleep(Duration::from_millis(10));
        // Should be positive if steps completed
        let sps = state.steps_per_second();
        assert!(sps > 0.0);
    }

    #[test]
    fn test_training_state_steps_per_second_zero_time() {
        // Fresh state with no elapsed time should return 0 or positive
        let state = TrainingState::new();
        let sps = state.steps_per_second();
        assert!(sps >= 0.0);
    }

    #[test]
    fn test_training_state_eta() {
        let mut state = TrainingState::new();
        state.step();
        std::thread::sleep(Duration::from_millis(10));
        let eta = state.eta(100);
        // ETA should be some duration (could be very large or zero)
        assert!(eta.as_secs_f32() >= 0.0);
    }

    #[test]
    fn test_training_state_eta_zero_steps() {
        let state = TrainingState::new();
        let eta = state.eta(100);
        assert_eq!(eta, Duration::ZERO);
    }

    #[test]
    fn test_trainer_teacher_ref() {
        let config = TrainerConfig::new("t", "s");
        let teacher = SafeTensorsTeacher::mock(12, 768);
        let trainer = DistillationTrainer::new(config, teacher);

        // Access teacher reference
        let teacher_ref = trainer.teacher();
        assert!(teacher_ref.param_count() > 0);
    }

    #[test]
    fn test_trainer_compute_loss_with_attention_transfer() {
        use ndarray::Array2;

        let config = TrainerConfig::new("t", "s").with_attention_transfer(0.5);
        let teacher = SafeTensorsTeacher::mock(12, 768);
        let trainer = DistillationTrainer::new(config, teacher);

        let student_logits = Array2::from_shape_vec((2, 10), vec![1.0; 20]).unwrap();
        let teacher_logits = Array2::from_shape_vec((2, 10), vec![1.1; 20]).unwrap();
        let targets = vec![5, 3];

        // Attention maps (simulating attention scores)
        let sa = vec![Array2::<f32>::zeros((2, 12))];
        let ta = vec![Array2::<f32>::ones((2, 12))];

        let loss = trainer.compute_loss(
            &student_logits,
            &teacher_logits,
            &targets,
            None,
            None,
            Some(&sa),
            Some(&ta),
        );

        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_trainer_state_clone() {
        let mut state = TrainingState::new();
        state.step();
        state.record_loss(1.0);

        let cloned = state.clone();
        assert_eq!(state.global_step, cloned.global_step);
        assert_eq!(state.loss_history.len(), cloned.loss_history.len());
    }

    #[test]
    fn test_trainer_config_clone() {
        let config = TrainerConfig::new("t", "s").temperature(4.0).epochs(10);

        let cloned = config.clone();
        assert_eq!(config.epochs, cloned.epochs);
        assert_eq!(
            config.distillation_loss.temperature,
            cloned.distillation_loss.temperature
        );
    }
}
