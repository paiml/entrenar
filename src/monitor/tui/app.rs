//! TUI Monitor Application (SPEC-FT-001 Section 10)
//!
//! Detached observer that reads training state and renders via presentar-terminal.
//! Runs in a separate shell/process from the training loop.
//! ALB-047/048: Terminal resize, Ctrl+C, cursor management via presentar.

use super::state::{TrainingSnapshot, TrainingState, TrainingStatus};
use std::io;
use std::path::Path;
use std::time::Duration;

/// Default TUI refresh interval in milliseconds
const DEFAULT_REFRESH_MS: u64 = 500;

/// Maximum number of loss values retained in the chart history
const LOSS_HISTORY_MAX: usize = 200;

/// TUI Monitor configuration
#[derive(Debug, Clone)]
pub struct TuiMonitorConfig {
    /// Refresh interval in milliseconds
    pub refresh_ms: u64,
    /// Terminal width (0 for auto-detect)
    pub width: usize,
    /// Terminal height (0 for auto-detect)
    pub height: usize,
    /// Show in compact mode
    pub compact: bool,
    /// Exit when training completes
    pub exit_on_complete: bool,
}

impl Default for TuiMonitorConfig {
    fn default() -> Self {
        Self {
            refresh_ms: DEFAULT_REFRESH_MS,
            width: 80,
            height: 24,
            compact: false,
            exit_on_complete: true,
        }
    }
}

/// Detached TUI Monitor (presentar-terminal backend, ALB-047/048)
///
/// Reads training state from the metric store and renders via presentar's
/// TuiApp framework. Gets terminal resize, Ctrl+C, cursor management,
/// and smart diffing for free from the sovereign stack.
pub struct TuiMonitor {
    config: TuiMonitorConfig,
    state: TrainingState,
}

impl TuiMonitor {
    /// Create a new TUI monitor for an experiment
    pub fn new<P: AsRef<Path>>(experiment_dir: P, config: TuiMonitorConfig) -> Self {
        Self {
            config,
            state: TrainingState::new(experiment_dir),
        }
    }

    /// Run the TUI monitor loop using presentar-terminal (ALB-047/048).
    ///
    /// This is a blocking call that runs until training completes, user presses
    /// 'q', or Ctrl+C. Uses presentar's TuiApp for:
    /// - Automatic terminal resize detection
    /// - Ctrl+C / 'q' handling with clean cursor restore
    /// - Smart diffing (only redraws changed cells)
    /// - 60fps rendering with frame budgets
    pub fn run(&mut self) -> io::Result<()> {
        // Wait for state file to appear
        eprintln!("Waiting for training state file at {}...", self.state.path().display());

        if !self.state.wait_for_state(Duration::from_secs(60))? {
            eprintln!("Timeout waiting for training state file.");
            return Ok(());
        }

        eprintln!("Connected to training session. Press 'q' or Ctrl+C to detach.\n");

        // Create presentar dashboard widget
        let experiment_dir = self.state.path().parent().unwrap_or(std::path::Path::new(".")).to_path_buf();
        let dashboard = super::dashboard::TrainingDashboard::new(experiment_dir);

        // Launch presentar TuiApp — handles resize, Ctrl+C, cursor, smart diffing
        let config = presentar_terminal::TuiConfig {
            tick_rate_ms: self.config.refresh_ms,
            ..Default::default()
        };
        let mut app = presentar_terminal::TuiApp::new(dashboard)
            .map_err(|e| io::Error::other(e.to_string()))?
            .with_config(config);

        app.run()
            .map_err(|e| io::Error::other(e.to_string()))?;

        eprintln!("\nDetached from training session. Training continues in background.");
        Ok(())
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_status_match_all_variants() {
        let statuses = [
            TrainingStatus::Initializing,
            TrainingStatus::Running,
            TrainingStatus::Paused,
            TrainingStatus::Completed,
            TrainingStatus::Failed("oom".to_string()),
        ];

        for status in &statuses {
            // Syntactic match covering all arms from render_final_summary
            let description = match status {
                TrainingStatus::Completed => "Completed Successfully".to_string(),
                TrainingStatus::Failed(msg) => format!("FAILED - {msg}"),
                TrainingStatus::Initializing | TrainingStatus::Running | TrainingStatus::Paused => {
                    "In progress".to_string()
                }
            };
            assert!(!description.is_empty());
        }
    }

    #[test]
    fn test_tui_monitor_config_default() {
        let config = TuiMonitorConfig::default();
        assert_eq!(config.refresh_ms, 500);
        assert_eq!(config.width, 80);
        assert_eq!(config.height, 24);
        assert!(!config.compact);
        assert!(config.exit_on_complete);
    }
} // mod tests (variant coverage)

/// Lightweight state writer for the training loop (Producer)
///
/// Used by the trainer to write atomic state updates.
pub struct TrainingStateWriter {
    state: TrainingState,
    snapshot: TrainingSnapshot,
    history_max: usize,
    /// When true, emit formatted progress lines to stdout during training.
    console_progress: bool,
}

impl TrainingStateWriter {
    /// Create a new state writer
    pub fn new<P: AsRef<Path>>(experiment_dir: P, experiment_id: &str, model_name: &str) -> Self {
        let mut snapshot = TrainingSnapshot::default();
        snapshot.experiment_id = experiment_id.to_string();
        snapshot.model_name = model_name.to_string();
        snapshot.status = TrainingStatus::Initializing;

        Self {
            state: TrainingState::new(experiment_dir),
            snapshot,
            history_max: LOSS_HISTORY_MAX,
            console_progress: false,
        }
    }

    /// Enable inline console progress on stdout.
    ///
    /// When enabled, `update_step()` periodically prints a formatted progress
    /// line so users see output without needing a separate `apr monitor` process.
    pub fn with_console_progress(mut self, enabled: bool) -> Self {
        self.console_progress = enabled;
        self
    }

    /// Set total epochs and steps for a training phase
    ///
    /// Call this at the start of each training phase to ensure the TUI
    /// displays correct progress. This resets epoch/step counters.
    pub fn set_epochs(&mut self, total_epochs: usize, steps_per_epoch: usize) {
        self.snapshot.total_epochs = total_epochs;
        self.snapshot.steps_per_epoch = steps_per_epoch;
        // Reset counters for new phase
        self.snapshot.epoch = 0;
        self.snapshot.step = 0;
    }

    /// Set run configuration (optimizer, batch size, paths)
    pub fn set_config(
        &mut self,
        optimizer_name: &str,
        batch_size: usize,
        model_path: &str,
        checkpoint_path: &str,
    ) {
        self.snapshot.optimizer_name = optimizer_name.to_string();
        self.snapshot.batch_size = batch_size;
        self.snapshot.model_path = model_path.to_string();
        self.snapshot.checkpoint_path = checkpoint_path.to_string();
        // Try to get executable path from current process
        if let Ok(exe) = std::env::current_exe() {
            self.snapshot.executable_path = exe.display().to_string();
        }
    }

    /// Set GPU device info for the training snapshot.
    ///
    /// When CUDA is active, populates the `gpu` field with device name and
    /// total VRAM. Updated dynamically by the training loop.
    pub fn set_gpu(&mut self, device_name: &str, vram_total_gb: f32) {
        self.snapshot.gpu = Some(super::state::GpuTelemetry {
            device_name: device_name.to_string(),
            vram_total_gb,
            ..Default::default()
        });
    }

    /// Mark training as started
    pub fn start(&mut self) -> io::Result<()> {
        self.snapshot.status = TrainingStatus::Running;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.snapshot.start_timestamp_ms = now;
        self.snapshot.timestamp_ms = now;
        self.state.write(&self.snapshot)
    }

    /// Update training step
    ///
    /// # Warnings
    /// Logs a warning to stderr if step > steps_per_epoch, which indicates
    /// a configuration mismatch (likely set_epochs() not called at phase start).
    pub fn update_step(
        &mut self,
        epoch: usize,
        step: usize,
        loss: f32,
        learning_rate: f32,
        gradient_norm: f32,
        tokens_per_second: f32,
        accuracy: f32,
    ) -> io::Result<()> {
        // Validate step doesn't exceed configured steps_per_epoch
        if self.snapshot.steps_per_epoch > 0 && step > self.snapshot.steps_per_epoch {
            eprintln!(
                "Warning: step {} exceeds steps_per_epoch {} - call set_epochs() at phase start",
                step, self.snapshot.steps_per_epoch
            );
        }

        // Validate epoch doesn't exceed total_epochs
        if self.snapshot.total_epochs > 0 && epoch > self.snapshot.total_epochs {
            eprintln!(
                "Warning: epoch {} exceeds total_epochs {} - call set_epochs() at phase start",
                epoch, self.snapshot.total_epochs
            );
        }

        self.snapshot.epoch = epoch;
        self.snapshot.step = step;
        self.snapshot.loss = loss;
        self.snapshot.learning_rate = learning_rate;
        self.snapshot.gradient_norm = gradient_norm;
        self.snapshot.tokens_per_second = tokens_per_second;
        self.snapshot.accuracy = accuracy;
        self.snapshot.samples_per_second = tokens_per_second;

        // Update timestamp
        self.snapshot.timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Append to loss history
        self.snapshot.loss_history.push(loss);
        if self.snapshot.loss_history.len() > self.history_max {
            self.snapshot.loss_history.remove(0);
        }

        // Append to LR history for scheduler tracking
        self.snapshot.lr_history.push(learning_rate);
        if self.snapshot.lr_history.len() > self.history_max {
            self.snapshot.lr_history.remove(0);
        }

        // Inline console progress (reuses HeadlessWriter text format)
        if self.console_progress {
            let log_every = (self.snapshot.steps_per_epoch / 10)
                .max(10)
                .min(self.snapshot.steps_per_epoch.max(1));
            if step == 1 || step.is_multiple_of(log_every) || step == self.snapshot.steps_per_epoch {
                self.refresh_gpu_telemetry();
                self.emit_console_progress();
            }
        }

        self.state.write(&self.snapshot)
    }

    /// Refresh GPU telemetry by querying nvidia-smi.
    ///
    /// Only updates if GPU was previously set (i.e., CUDA is active).
    /// Called at the same frequency as console progress (~every 10% of epoch).
    fn refresh_gpu_telemetry(&mut self) {
        let device_name = match &self.snapshot.gpu {
            Some(gpu) => gpu.device_name.clone(),
            None => return,
        };

        let output = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ])
            .output();

        let output = match output {
            Ok(o) if o.status.success() => o,
            _ => return,
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let line = match stdout.lines().next() {
            Some(l) => l.trim(),
            None => return,
        };
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() < 6 {
            return;
        }

        self.snapshot.gpu = Some(super::state::GpuTelemetry {
            device_name,
            utilization_percent: fields[0].parse().unwrap_or(0.0),
            vram_used_gb: fields[1].parse::<f32>().unwrap_or(0.0) / 1024.0,
            vram_total_gb: fields[2].parse::<f32>().unwrap_or(0.0) / 1024.0,
            temperature_celsius: fields[3].parse().unwrap_or(0.0),
            power_watts: fields[4].parse().unwrap_or(0.0),
            power_limit_watts: fields[5].parse().unwrap_or(0.0),
            processes: Vec::new(),
        });
    }

    /// Emit a single console progress line via the headless text formatter.
    fn emit_console_progress(&self) {
        let mut buf = Vec::new();
        let mut writer =
            super::headless::HeadlessWriter::new(&mut buf, super::headless::OutputFormat::Text);
        let _ = writer.write(&self.snapshot);
        if let Ok(s) = String::from_utf8(buf) {
            print!("  {s}");
        }
    }

    /// Emit an epoch-end summary line to the console.
    pub fn emit_epoch_summary(
        &self,
        epoch: usize,
        total_epochs: usize,
        train_loss: f32,
        train_acc: f32,
        val_loss: f32,
        val_acc: f32,
        epoch_secs: f32,
        lr: f32,
        is_best: bool,
    ) {
        if self.console_progress {
            let best = if is_best { " *best*" } else { "" };
            println!(
                "  Epoch {epoch}/{total_epochs} done in {epoch_secs:.0}s — \
                 train_loss: {train_loss:.4}, train_acc: {:.1}%, \
                 val_loss: {val_loss:.4}, val_acc: {:.1}%, LR: {lr:.2e}{best}",
                train_acc * 100.0,
                val_acc * 100.0,
            );
        }
    }

    /// Emit a one-shot informational message through the monitoring framework.
    pub fn emit_info(&self, msg: &str) {
        if self.console_progress {
            println!("  {msg}");
        }
    }

    /// Update GPU telemetry
    pub fn update_gpu(&mut self, gpu: super::state::GpuTelemetry) -> io::Result<()> {
        self.snapshot.gpu = Some(gpu);
        self.state.write(&self.snapshot)
    }

    /// Update sample peek
    pub fn update_sample(&mut self, sample: super::state::SamplePeek) -> io::Result<()> {
        self.snapshot.sample = Some(sample);
        self.state.write(&self.snapshot)
    }

    /// Mark training as completed
    pub fn complete(&mut self) -> io::Result<()> {
        self.snapshot.status = TrainingStatus::Completed;
        self.state.write(&self.snapshot)
    }

    /// Mark training as failed
    pub fn fail(&mut self, error: &str) -> io::Result<()> {
        self.snapshot.status = TrainingStatus::Failed(error.to_string());
        self.state.write(&self.snapshot)
    }

    /// Get the state file path
    pub fn state_path(&self) -> &Path {
        self.state.path()
    }
}

#[cfg(test)]
mod state_writer_tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_training_state_writer() {
        let temp_dir = TempDir::new().expect("temp file creation should succeed");
        let mut writer = TrainingStateWriter::new(temp_dir.path(), "test-001", "test-model");

        writer.set_epochs(10, 100);
        writer.start().expect("file write should succeed");

        writer.update_step(1, 10, 0.5, 0.0002, 1.5, 1200.0, 0.75).expect("file write should succeed");

        // Verify state was written
        let mut state = TrainingState::new(temp_dir.path());
        let snapshot =
            state.read().expect("file read should succeed").expect("file read should succeed");

        assert_eq!(snapshot.epoch, 1);
        assert_eq!(snapshot.step, 10);
        assert!((snapshot.loss - 0.5).abs() < 0.001);
        assert_eq!(snapshot.status, TrainingStatus::Running);
    }

    #[test]
    fn test_training_state_writer_complete() {
        let temp_dir = TempDir::new().expect("temp file creation should succeed");
        let mut writer = TrainingStateWriter::new(temp_dir.path(), "test-001", "test-model");

        writer.start().expect("file write should succeed");
        writer.complete().expect("file write should succeed");

        let mut state = TrainingState::new(temp_dir.path());
        let snapshot =
            state.read().expect("file read should succeed").expect("file read should succeed");

        assert_eq!(snapshot.status, TrainingStatus::Completed);
    }

    #[test]
    fn test_training_state_writer_fail() {
        let temp_dir = TempDir::new().expect("temp file creation should succeed");
        let mut writer = TrainingStateWriter::new(temp_dir.path(), "test-001", "test-model");

        writer.start().expect("file write should succeed");
        writer.fail("OOM error").expect("file write should succeed");

        let mut state = TrainingState::new(temp_dir.path());
        let snapshot =
            state.read().expect("file read should succeed").expect("file read should succeed");

        match snapshot.status {
            TrainingStatus::Failed(msg) => assert!(msg.contains("OOM")),
            _ => panic!("Expected Failed status"),
        }
    }

    #[test]
    fn test_loss_history_truncation() {
        let temp_dir = TempDir::new().expect("temp file creation should succeed");
        let mut writer = TrainingStateWriter::new(temp_dir.path(), "test-001", "test-model");
        writer.history_max = 5; // Small for testing

        writer.start().expect("file write should succeed");

        // Add more than max history
        for i in 0..10 {
            writer
                .update_step(1, i, i as f32 * 0.1, 0.0002, 1.5, 1200.0, 0.0)
                .expect("file write should succeed");
        }

        let mut state = TrainingState::new(temp_dir.path());
        let snapshot =
            state.read().expect("file read should succeed").expect("file read should succeed");

        assert_eq!(snapshot.loss_history.len(), 5);
        // Should have last 5 values (0.5, 0.6, 0.7, 0.8, 0.9)
        assert!((snapshot.loss_history[0] - 0.5).abs() < 0.001);
    }
}
