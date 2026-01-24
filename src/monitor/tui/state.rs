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
    pub fn elapsed(&self) -> Duration {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Duration::from_millis(now.saturating_sub(self.start_timestamp_ms))
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
}
