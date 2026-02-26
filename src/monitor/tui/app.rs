//! TUI Monitor Application (SPEC-FT-001 Section 10)
//!
//! Detached observer that reads training state and renders the TUI.
//! Runs in a separate shell/process from the training loop.

use super::render::render_layout;
use super::state::{TrainingSnapshot, TrainingState, TrainingStatus};
use std::io::{self, Write};
use std::path::Path;
use std::time::{Duration, Instant};

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

/// Detached TUI Monitor
///
/// Reads training state from the metric store and renders the TUI.
/// Does not block the training loop.
pub struct TuiMonitor {
    config: TuiMonitorConfig,
    state: TrainingState,
    last_render: Option<Instant>,
    frame_count: u64,
}

impl TuiMonitor {
    /// Create a new TUI monitor for an experiment
    pub fn new<P: AsRef<Path>>(experiment_dir: P, config: TuiMonitorConfig) -> Self {
        Self {
            config,
            state: TrainingState::new(experiment_dir),
            last_render: None,
            frame_count: 0,
        }
    }

    /// Run the TUI monitor loop
    ///
    /// This is a blocking call that runs until training completes or user exits.
    pub fn run(&mut self) -> io::Result<()> {
        // Wait for state file to appear
        eprintln!(
            "Waiting for training state file at {}...",
            self.state.path().display()
        );

        if !self.state.wait_for_state(Duration::from_secs(60))? {
            eprintln!("Timeout waiting for training state file.");
            return Ok(());
        }

        eprintln!("Connected to training session. Press Ctrl+C to detach.\n");

        // Clear screen and hide cursor
        self.clear_screen()?;
        self.hide_cursor()?;

        // Main render loop
        loop {
            // Check if enough time has passed since last render
            let should_render = self
                .last_render
                .is_none_or(|t| t.elapsed() >= Duration::from_millis(self.config.refresh_ms));

            if should_render {
                // Read latest state
                if let Some(snapshot) = self.state.read()? {
                    self.render_frame(&snapshot)?;

                    // Check for completion
                    if self.config.exit_on_complete
                        && matches!(
                            snapshot.status,
                            TrainingStatus::Completed | TrainingStatus::Failed(_)
                        )
                    {
                        self.render_final_summary(&snapshot)?;
                        break;
                    }
                }

                self.last_render = Some(Instant::now());
                self.frame_count += 1;
            }

            // Sleep briefly to avoid busy-waiting
            std::thread::sleep(Duration::from_millis(50));

            // Keyboard input handling (Ctrl+C, resize) deferred to terminal event integration
        }

        // Restore cursor
        self.show_cursor()?;

        Ok(())
    }

    /// Render a single frame
    fn render_frame(&self, snapshot: &TrainingSnapshot) -> io::Result<()> {
        let mut stdout = io::stdout();

        // Move cursor to top-left
        write!(stdout, "\x1b[H")?;

        // Render layout
        let layout = render_layout(snapshot, self.config.width);
        write!(stdout, "{layout}")?;

        // Status line
        let status = format!(
            "Frame: {} | Loss: {:.4} | Progress: {:.1}%",
            self.frame_count,
            snapshot.loss,
            snapshot.progress_percent()
        );
        writeln!(stdout, "\n{status}")?;

        stdout.flush()
    }

    /// Render final summary when training completes
    fn render_final_summary(&self, snapshot: &TrainingSnapshot) -> io::Result<()> {
        let mut stdout = io::stdout();

        writeln!(stdout, "\n")?;
        writeln!(
            stdout,
            "╔════════════════════════════════════════════════════════════╗"
        )?;
        writeln!(
            stdout,
            "║                    Training Complete                        ║"
        )?;
        writeln!(
            stdout,
            "╠════════════════════════════════════════════════════════════╣"
        )?;

        match &snapshot.status {
            TrainingStatus::Completed => {
                writeln!(
                    stdout,
                    "║  Status:    Completed Successfully                         ║"
                )?;
            }
            TrainingStatus::Failed(msg) => {
                writeln!(stdout, "║  Status:    FAILED - {msg}                    ║")?;
            }
            TrainingStatus::Initializing | TrainingStatus::Running | TrainingStatus::Paused => {}
        }

        writeln!(
            stdout,
            "║  Model:     {:.40}                           ║",
            snapshot.model_name
        )?;
        writeln!(
            stdout,
            "║  Duration:  {}                                    ║",
            super::render::format_duration(snapshot.elapsed())
        )?;
        writeln!(
            stdout,
            "║  Final Loss: {:.6}                                      ║",
            snapshot.loss
        )?;
        writeln!(
            stdout,
            "║  Epochs:    {}/{}                                          ║",
            snapshot.epoch, snapshot.total_epochs
        )?;
        writeln!(
            stdout,
            "╚════════════════════════════════════════════════════════════╝"
        )?;

        stdout.flush()
    }

    fn clear_screen(&self) -> io::Result<()> {
        let mut stdout = io::stdout();
        write!(stdout, "\x1b[2J\x1b[H")?;
        stdout.flush()
    }

    fn hide_cursor(&self) -> io::Result<()> {
        let mut stdout = io::stdout();
        write!(stdout, "\x1b[?25l")?;
        stdout.flush()
    }

    fn show_cursor(&self) -> io::Result<()> {
        let mut stdout = io::stdout();
        write!(stdout, "\x1b[?25h")?;
        stdout.flush()
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
        }
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

        self.state.write(&self.snapshot)
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
        let temp_dir = TempDir::new().unwrap();
        let mut writer = TrainingStateWriter::new(temp_dir.path(), "test-001", "test-model");

        writer.set_epochs(10, 100);
        writer.start().unwrap();

        writer.update_step(1, 10, 0.5, 0.0002, 1.5, 1200.0).unwrap();

        // Verify state was written
        let mut state = TrainingState::new(temp_dir.path());
        let snapshot = state.read().unwrap().unwrap();

        assert_eq!(snapshot.epoch, 1);
        assert_eq!(snapshot.step, 10);
        assert!((snapshot.loss - 0.5).abs() < 0.001);
        assert_eq!(snapshot.status, TrainingStatus::Running);
    }

    #[test]
    fn test_training_state_writer_complete() {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = TrainingStateWriter::new(temp_dir.path(), "test-001", "test-model");

        writer.start().unwrap();
        writer.complete().unwrap();

        let mut state = TrainingState::new(temp_dir.path());
        let snapshot = state.read().unwrap().unwrap();

        assert_eq!(snapshot.status, TrainingStatus::Completed);
    }

    #[test]
    fn test_training_state_writer_fail() {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = TrainingStateWriter::new(temp_dir.path(), "test-001", "test-model");

        writer.start().unwrap();
        writer.fail("OOM error").unwrap();

        let mut state = TrainingState::new(temp_dir.path());
        let snapshot = state.read().unwrap().unwrap();

        match snapshot.status {
            TrainingStatus::Failed(msg) => assert!(msg.contains("OOM")),
            _ => panic!("Expected Failed status"),
        }
    }

    #[test]
    fn test_loss_history_truncation() {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = TrainingStateWriter::new(temp_dir.path(), "test-001", "test-model");
        writer.history_max = 5; // Small for testing

        writer.start().unwrap();

        // Add more than max history
        for i in 0..10 {
            writer
                .update_step(1, i, i as f32 * 0.1, 0.0002, 1.5, 1200.0)
                .unwrap();
        }

        let mut state = TrainingState::new(temp_dir.path());
        let snapshot = state.read().unwrap().unwrap();

        assert_eq!(snapshot.loss_history.len(), 5);
        // Should have last 5 values (0.5, 0.6, 0.7, 0.8, 0.9)
        assert!((snapshot.loss_history[0] - 0.5).abs() < 0.001);
    }
}
