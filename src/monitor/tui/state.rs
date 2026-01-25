//! Training State for IPC (SPEC-FT-001 Section 10.1)
//!
//! Atomic state updates written by the trainer, read by the TUI monitor.
//! Uses JSON file as the IPC mechanism for simplicity and portability.

use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// GPU telemetry snapshot (NVML-inspired)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuTelemetry {
    /// GPU device name (e.g., "RTX 4090")
    pub device_name: String,
    /// GPU utilization percentage (0-100)
    pub utilization_percent: f32,
    /// VRAM used in GB
    pub vram_used_gb: f32,
    /// VRAM total in GB
    pub vram_total_gb: f32,
    /// Temperature in Celsius
    pub temperature_celsius: f32,
    /// Power draw in watts
    pub power_watts: f32,
    /// Power limit in watts
    pub power_limit_watts: f32,
}

impl GpuTelemetry {
    /// VRAM utilization as percentage
    pub fn vram_percent(&self) -> f32 {
        if self.vram_total_gb > 0.0 {
            (self.vram_used_gb / self.vram_total_gb) * 100.0
        } else {
            0.0
        }
    }

    /// Check if thermal throttling is likely
    pub fn is_thermal_throttling(&self) -> bool {
        self.temperature_celsius > 83.0
    }

    /// Check if power limited
    pub fn is_power_limited(&self) -> bool {
        self.power_watts >= self.power_limit_watts * 0.95
    }
}

/// Sample peek for live decoding visualization
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SamplePeek {
    /// Input function code (truncated for display)
    pub input_preview: String,
    /// Target test code (truncated for display)
    pub target_preview: String,
    /// Generated test code (truncated for display)
    pub generated_preview: String,
    /// Token match percentage (0-100)
    pub token_match_percent: f32,
}

/// Complete training state snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSnapshot {
    /// Unix timestamp in milliseconds
    pub timestamp_ms: u64,
    /// Current epoch (1-indexed)
    pub epoch: usize,
    /// Total epochs
    pub total_epochs: usize,
    /// Current step within epoch
    pub step: usize,
    /// Total steps per epoch
    pub steps_per_epoch: usize,
    /// Current loss value
    pub loss: f32,
    /// Loss history (last N values for sparkline)
    pub loss_history: Vec<f32>,
    /// Current learning rate
    pub learning_rate: f32,
    /// Gradient norm
    pub gradient_norm: f32,
    /// Throughput in tokens per second
    pub tokens_per_second: f32,
    /// Training start timestamp (ms)
    pub start_timestamp_ms: u64,
    /// GPU telemetry (optional)
    pub gpu: Option<GpuTelemetry>,
    /// Sample peek (optional)
    pub sample: Option<SamplePeek>,
    /// Training status
    pub status: TrainingStatus,
    /// Experiment name/ID
    pub experiment_id: String,
    /// Model name
    pub model_name: String,
}

/// Training status enum
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrainingStatus {
    /// Training is initializing
    Initializing,
    /// Training is running
    Running,
    /// Training is paused
    Paused,
    /// Training completed successfully
    Completed,
    /// Training failed with error
    Failed(String),
}

impl Default for TrainingSnapshot {
    fn default() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            timestamp_ms: now,
            epoch: 0,
            total_epochs: 0,
            step: 0,
            steps_per_epoch: 0,
            loss: 0.0,
            loss_history: Vec::new(),
            learning_rate: 0.0,
            gradient_norm: 0.0,
            tokens_per_second: 0.0,
            start_timestamp_ms: now,
            gpu: None,
            sample: None,
            status: TrainingStatus::Initializing,
            experiment_id: String::new(),
            model_name: String::new(),
        }
    }
}

impl TrainingSnapshot {
    /// Calculate elapsed time since training start
    /// Uses the snapshot's timestamp_ms for deterministic/reproducible output (ENT-140)
    pub fn elapsed(&self) -> Duration {
        Duration::from_millis(self.timestamp_ms.saturating_sub(self.start_timestamp_ms))
    }

    /// Calculate estimated remaining time
    pub fn estimated_remaining(&self) -> Option<Duration> {
        if self.tokens_per_second <= 0.0 {
            return None;
        }

        let total_steps = self.total_epochs * self.steps_per_epoch;
        let completed_steps = (self.epoch.saturating_sub(1)) * self.steps_per_epoch + self.step;

        if completed_steps == 0 || total_steps == 0 {
            return None;
        }

        let progress = completed_steps as f64 / total_steps as f64;
        if progress >= 1.0 {
            return Some(Duration::ZERO);
        }

        let elapsed_ms = self.timestamp_ms.saturating_sub(self.start_timestamp_ms);
        let total_estimated_ms = (elapsed_ms as f64 / progress) as u64;
        let remaining_ms = total_estimated_ms.saturating_sub(elapsed_ms);

        Some(Duration::from_millis(remaining_ms))
    }

    /// Global step (epoch * steps_per_epoch + step)
    pub fn global_step(&self) -> usize {
        (self.epoch.saturating_sub(1)) * self.steps_per_epoch + self.step
    }

    /// Progress percentage (0-100)
    pub fn progress_percent(&self) -> f32 {
        let total = self.total_epochs * self.steps_per_epoch;
        if total == 0 {
            return 0.0;
        }
        (self.global_step() as f32 / total as f32) * 100.0
    }

    /// Compute loss trend from recent history
    ///
    /// Returns:
    /// - `LossTrend::Decreasing` if loss is going down (good)
    /// - `LossTrend::Stable` if loss is plateauing
    /// - `LossTrend::Increasing` if loss is going up (bad)
    /// - `LossTrend::Unknown` if not enough data
    pub fn loss_trend(&self) -> LossTrend {
        // Need at least 5 samples to compute trend
        if self.loss_history.len() < 5 {
            return LossTrend::Unknown;
        }

        // Use last 10 samples (or all if less)
        let window = self.loss_history.len().min(10);
        let recent = &self.loss_history[self.loss_history.len() - window..];

        // Compare first half vs second half
        let mid = window / 2;
        let first_half: f32 = recent[..mid].iter().sum::<f32>() / mid as f32;
        let second_half: f32 = recent[mid..].iter().sum::<f32>() / (window - mid) as f32;

        // Calculate relative change
        let change = (second_half - first_half) / first_half.abs().max(1e-6);

        // Threshold: 2% change considered significant
        const THRESHOLD: f32 = 0.02;

        if change < -THRESHOLD {
            LossTrend::Decreasing
        } else if change > THRESHOLD {
            LossTrend::Increasing
        } else {
            LossTrend::Stable
        }
    }
}

/// Loss trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossTrend {
    /// Loss is decreasing (good)
    Decreasing,
    /// Loss is stable/plateauing
    Stable,
    /// Loss is increasing (bad)
    Increasing,
    /// Not enough data to determine
    Unknown,
}

impl LossTrend {
    /// Get Unicode arrow for display
    pub fn arrow(&self) -> &'static str {
        match self {
            LossTrend::Decreasing => "↓",
            LossTrend::Stable => "→",
            LossTrend::Increasing => "↑",
            LossTrend::Unknown => "?",
        }
    }

    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            LossTrend::Decreasing => "decreasing",
            LossTrend::Stable => "stable",
            LossTrend::Increasing => "increasing",
            LossTrend::Unknown => "unknown",
        }
    }
}

/// Training state manager for IPC
///
/// Handles atomic read/write of training snapshots via JSON file.
pub struct TrainingState {
    /// Path to the state file
    state_path: std::path::PathBuf,
    /// Last read snapshot (cached)
    last_snapshot: Option<TrainingSnapshot>,
    /// Last modification time
    last_modified: Option<std::time::SystemTime>,
}

impl TrainingState {
    /// Create a new training state manager
    ///
    /// # Arguments
    ///
    /// * `experiment_dir` - Path to the experiment directory
    pub fn new<P: AsRef<Path>>(experiment_dir: P) -> Self {
        let state_path = experiment_dir.as_ref().join("training_state.json");
        Self {
            state_path,
            last_snapshot: None,
            last_modified: None,
        }
    }

    /// Write a training snapshot atomically
    ///
    /// Uses write-to-temp + rename for atomicity.
    pub fn write(&self, snapshot: &TrainingSnapshot) -> std::io::Result<()> {
        // Ensure parent directory exists
        if let Some(parent) = self.state_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Write to temp file first
        let temp_path = self.state_path.with_extension("json.tmp");
        let file = File::create(&temp_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, snapshot)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Atomic rename
        fs::rename(&temp_path, &self.state_path)?;

        Ok(())
    }

    /// Read the current training snapshot
    ///
    /// Returns `None` if the state file doesn't exist yet.
    pub fn read(&mut self) -> std::io::Result<Option<TrainingSnapshot>> {
        if !self.state_path.exists() {
            return Ok(None);
        }

        // Check if file was modified since last read
        let metadata = fs::metadata(&self.state_path)?;
        let modified = metadata.modified()?;

        if self.last_modified == Some(modified) {
            // Return cached snapshot if file hasn't changed
            return Ok(self.last_snapshot.clone());
        }

        // Read and parse
        let file = File::open(&self.state_path)?;
        let reader = BufReader::new(file);
        let snapshot: TrainingSnapshot = serde_json::from_reader(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Update cache
        self.last_snapshot = Some(snapshot.clone());
        self.last_modified = Some(modified);

        Ok(Some(snapshot))
    }

    /// Check if training state file exists
    pub fn exists(&self) -> bool {
        self.state_path.exists()
    }

    /// Get the state file path
    pub fn path(&self) -> &Path {
        &self.state_path
    }

    /// Wait for the state file to appear (with timeout)
    pub fn wait_for_state(&mut self, timeout: Duration) -> std::io::Result<bool> {
        let start = Instant::now();
        while start.elapsed() < timeout {
            if self.exists() {
                return Ok(true);
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use tempfile::TempDir;

    #[test]
    fn test_training_snapshot_default() {
        let snapshot = TrainingSnapshot::default();
        assert_eq!(snapshot.epoch, 0);
        assert_eq!(snapshot.status, TrainingStatus::Initializing);
    }

    #[test]
    fn test_training_snapshot_progress() {
        let mut snapshot = TrainingSnapshot::default();
        snapshot.epoch = 2;
        snapshot.total_epochs = 10;
        snapshot.step = 50;
        snapshot.steps_per_epoch = 100;

        // Global step = (2-1) * 100 + 50 = 150
        assert_eq!(snapshot.global_step(), 150);

        // Progress = 150 / 1000 = 15%
        assert!((snapshot.progress_percent() - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_gpu_telemetry_vram_percent() {
        let gpu = GpuTelemetry {
            vram_used_gb: 4.0,
            vram_total_gb: 24.0,
            ..Default::default()
        };
        assert!((gpu.vram_percent() - 16.67).abs() < 0.1);
    }

    #[test]
    fn test_training_state_write_read() {
        let temp_dir = TempDir::new().unwrap();
        let mut state = TrainingState::new(temp_dir.path());

        let snapshot = TrainingSnapshot {
            epoch: 5,
            total_epochs: 10,
            loss: 0.42,
            status: TrainingStatus::Running,
            ..Default::default()
        };

        state.write(&snapshot).unwrap();
        assert!(state.exists());

        let read_snapshot = state.read().unwrap().unwrap();
        assert_eq!(read_snapshot.epoch, 5);
        assert!((read_snapshot.loss - 0.42).abs() < 0.001);
    }

    #[test]
    fn test_training_state_caching() {
        let temp_dir = TempDir::new().unwrap();
        let mut state = TrainingState::new(temp_dir.path());

        let snapshot = TrainingSnapshot {
            epoch: 1,
            ..Default::default()
        };

        state.write(&snapshot).unwrap();

        // First read
        let _ = state.read().unwrap();

        // Second read should return cached
        let cached = state.read().unwrap().unwrap();
        assert_eq!(cached.epoch, 1);
    }

    // Property-based tests for monitoring integration (ENT-121)

    proptest! {
        /// TrainingSnapshot JSON serialization round-trip
        #[test]
        fn prop_snapshot_json_roundtrip(
            epoch in 1usize..1000,
            total_epochs in 1usize..100,
            step in 0usize..10000,
            steps_per_epoch in 1usize..10000,
            loss in 0.0f32..100.0,
            learning_rate in 1e-10f32..1.0,
            gradient_norm in 0.0f32..1000.0,
            tokens_per_second in 0.0f32..10000.0,
        ) {
            let snapshot = TrainingSnapshot {
                timestamp_ms: 12345678,
                epoch,
                total_epochs,
                step,
                steps_per_epoch,
                loss,
                loss_history: vec![loss * 1.1, loss * 1.05, loss],
                learning_rate,
                gradient_norm,
                tokens_per_second,
                start_timestamp_ms: 12345000,
                gpu: None,
                sample: None,
                status: TrainingStatus::Running,
                experiment_id: "test".to_string(),
                model_name: "model".to_string(),
            };

            // Serialize
            let json = serde_json::to_string(&snapshot).unwrap();

            // Deserialize
            let restored: TrainingSnapshot = serde_json::from_str(&json).unwrap();

            // Verify all fields preserved
            prop_assert_eq!(restored.epoch, epoch);
            prop_assert_eq!(restored.total_epochs, total_epochs);
            prop_assert_eq!(restored.step, step);
            prop_assert_eq!(restored.steps_per_epoch, steps_per_epoch);
            prop_assert!((restored.loss - loss).abs() < 1e-5);
            prop_assert!((restored.learning_rate - learning_rate).abs() < 1e-10);
            prop_assert!((restored.gradient_norm - gradient_norm).abs() < 1e-5);
        }

        /// Loss trend detection is consistent with loss history direction
        #[test]
        fn prop_loss_trend_consistent(
            base_loss in 1.0f32..10.0,
            trend_factor in -0.1f32..0.1,
        ) {
            // Generate 10 loss values with consistent trend
            // Positive factor = loss going up, Negative factor = loss going down
            let history: Vec<f32> = (0..10)
                .map(|i| base_loss + (i as f32 * trend_factor))
                .collect();

            let snapshot = TrainingSnapshot {
                loss_history: history,
                ..Default::default()
            };

            let trend = snapshot.loss_trend();

            // With 10 samples and consistent trend, we should detect it
            // Positive factor = loss increasing over time = LossTrend::Increasing
            // Negative factor = loss decreasing over time = LossTrend::Decreasing
            if trend_factor > 0.05 {
                prop_assert_eq!(trend, LossTrend::Increasing);
            } else if trend_factor < -0.05 {
                prop_assert_eq!(trend, LossTrend::Decreasing);
            }
            // Stable if |trend_factor| <= 0.05 (within threshold)
        }

        /// GPU telemetry VRAM percentage is always 0-100
        #[test]
        fn prop_gpu_vram_percent_bounded(
            vram_used in 0.0f32..100.0,
            vram_total in 1.0f32..100.0,
        ) {
            let gpu = GpuTelemetry {
                vram_used_gb: vram_used.min(vram_total),
                vram_total_gb: vram_total,
                ..Default::default()
            };

            let percent = gpu.vram_percent();
            prop_assert!(percent >= 0.0);
            prop_assert!(percent <= 100.0);
        }

        /// Progress percent is always 0-100
        #[test]
        fn prop_progress_percent_bounded(
            epoch in 1usize..100,
            total_epochs in 1usize..100,
            step in 0usize..1000,
            steps_per_epoch in 1usize..1000,
        ) {
            let epoch = epoch.min(total_epochs);
            let step = step.min(steps_per_epoch);

            let snapshot = TrainingSnapshot {
                epoch,
                total_epochs,
                step,
                steps_per_epoch,
                ..Default::default()
            };

            let progress = snapshot.progress_percent();
            prop_assert!(progress >= 0.0);
            prop_assert!(progress <= 100.0);
        }

        /// State file write/read preserves all data
        #[test]
        fn prop_state_file_roundtrip(
            epoch in 1usize..100,
            loss in 0.0f32..100.0,
            lr in 1e-6f32..0.1,
        ) {
            let temp_dir = TempDir::new().unwrap();
            let mut state = TrainingState::new(temp_dir.path());

            let snapshot = TrainingSnapshot {
                epoch,
                total_epochs: 10,
                loss,
                learning_rate: lr,
                status: TrainingStatus::Running,
                ..Default::default()
            };

            state.write(&snapshot).unwrap();

            // Clear cache and re-read
            state.last_modified = None;
            let restored = state.read().unwrap().unwrap();

            prop_assert_eq!(restored.epoch, epoch);
            prop_assert!((restored.loss - loss).abs() < 1e-5);
            prop_assert!((restored.learning_rate - lr).abs() < 1e-10);
        }
    }
}
