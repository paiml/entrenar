//! Training configuration and metrics

use std::path::PathBuf;

/// Training configuration
#[derive(Clone, Debug)]
pub struct TrainConfig {
    /// Maximum gradient norm for clipping (None = no clipping)
    pub max_grad_norm: Option<f32>,

    /// Print training progress every N steps
    pub log_interval: usize,

    /// Save checkpoint every N epochs
    pub save_interval: Option<usize>,

    /// Directory to save checkpoints
    pub checkpoint_dir: Option<PathBuf>,

    /// Use mixed precision training
    pub mixed_precision: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            max_grad_norm: Some(1.0),
            log_interval: 10,
            save_interval: None,
            checkpoint_dir: None,
            mixed_precision: false,
        }
    }
}

impl TrainConfig {
    /// Create a new training configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set gradient clipping norm
    pub fn with_grad_clip(mut self, max_norm: f32) -> Self {
        self.max_grad_norm = Some(max_norm);
        self
    }

    /// Disable gradient clipping
    pub fn without_grad_clip(mut self) -> Self {
        self.max_grad_norm = None;
        self
    }

    /// Set logging interval
    pub fn with_log_interval(mut self, interval: usize) -> Self {
        self.log_interval = interval;
        self
    }

    /// Set checkpoint saving
    pub fn with_checkpoints(mut self, interval: usize, dir: PathBuf) -> Self {
        self.save_interval = Some(interval);
        self.checkpoint_dir = Some(dir);
        self
    }
}

/// Tracks training metrics across epochs
#[derive(Clone, Debug)]
pub struct MetricsTracker {
    /// Loss history (one per epoch)
    pub losses: Vec<f32>,

    /// Learning rates (one per epoch)
    pub learning_rates: Vec<f32>,

    /// Training step count
    pub steps: usize,

    /// Current epoch
    pub epoch: usize,
}

impl MetricsTracker {
    /// Create a new metrics tracker
    pub fn new() -> Self {
        Self {
            losses: Vec::new(),
            learning_rates: Vec::new(),
            steps: 0,
            epoch: 0,
        }
    }

    /// Record an epoch's metrics
    pub fn record_epoch(&mut self, loss: f32, lr: f32) {
        self.losses.push(loss);
        self.learning_rates.push(lr);
        self.epoch += 1;
    }

    /// Increment step counter
    pub fn increment_step(&mut self) {
        self.steps += 1;
    }

    /// Get average loss over last N epochs
    pub fn avg_loss(&self, n: usize) -> f32 {
        if self.losses.is_empty() {
            return 0.0;
        }

        let start = self.losses.len().saturating_sub(n);
        let window = &self.losses[start..];
        window.iter().sum::<f32>() / window.len() as f32
    }

    /// Get best (minimum) loss
    pub fn best_loss(&self) -> Option<f32> {
        self.losses
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Check if training is improving (loss decreasing)
    pub fn is_improving(&self, patience: usize) -> bool {
        if self.losses.len() < patience {
            return true;
        }

        let recent = self.losses[self.losses.len() - patience..].to_vec();
        let mut sorted = recent.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Check if losses are generally decreasing
        recent != sorted
    }
}

impl Default for MetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_config_default() {
        let config = TrainConfig::default();
        assert_eq!(config.max_grad_norm, Some(1.0));
        assert_eq!(config.log_interval, 10);
        assert!(config.save_interval.is_none());
    }

    #[test]
    fn test_train_config_builder() {
        let config = TrainConfig::new()
            .with_grad_clip(0.5)
            .with_log_interval(20)
            .without_grad_clip();

        assert_eq!(config.max_grad_norm, None);
        assert_eq!(config.log_interval, 20);
    }

    #[test]
    fn test_metrics_tracker() {
        let mut tracker = MetricsTracker::new();

        tracker.record_epoch(1.0, 0.001);
        tracker.record_epoch(0.8, 0.001);
        tracker.record_epoch(0.6, 0.001);

        assert_eq!(tracker.epoch, 3);
        assert_eq!(tracker.losses.len(), 3);
        assert_eq!(tracker.best_loss(), Some(0.6));
    }

    #[test]
    fn test_metrics_avg_loss() {
        let mut tracker = MetricsTracker::new();

        tracker.record_epoch(1.0, 0.001);
        tracker.record_epoch(0.8, 0.001);
        tracker.record_epoch(0.6, 0.001);

        let avg = tracker.avg_loss(2);
        assert!((avg - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_metrics_is_improving() {
        let mut tracker = MetricsTracker::new();

        // Decreasing losses = improving
        tracker.record_epoch(1.0, 0.001);
        tracker.record_epoch(0.8, 0.001);
        tracker.record_epoch(0.6, 0.001);

        assert!(tracker.is_improving(2));
    }
}
