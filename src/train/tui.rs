//! Real-Time Terminal Monitoring and Visualization (ENT-054 through ENT-067)
//!
//! Terminal-based training visualization using trueno-viz exclusively.
//!
//! # Features
//!
//! - `MetricsBuffer`: O(1) ring buffer for streaming metrics (ENT-055)
//! - `Sparkline`: Unicode sparklines for inline metrics (ENT-057)
//! - `ProgressBar`: Progress bar with Kalman-filtered ETA (ENT-058)
//! - `RefreshPolicy`: Adaptive refresh rate control (ENT-060)
//! - `AndonSystem`: Health monitoring with NaN/Inf detection (ENT-066)
//! - `TerminalMonitorCallback`: Unified callback for training loop (ENT-054)
//!
//! # References
//!
//! - Tufte, E. R. (2006). *Beautiful Evidence*. Graphics Press. (Sparklines)
//! - Welch, G., & Bishop, G. (1995). "An Introduction to the Kalman Filter." (ETA)

use std::io::Write;
use std::time::{Duration, Instant};

use crate::train::callback::{CallbackAction, CallbackContext, TrainerCallback};
use trueno_viz::output::{TerminalEncoder, TerminalMode as TruenoTerminalMode};
use trueno_viz::plots::{LossCurve, MetricSeries};
use trueno_viz::prelude::Rgba;

// =============================================================================
// MetricsBuffer - Ring Buffer (ENT-055)
// =============================================================================

/// Fixed-size ring buffer for streaming metrics.
///
/// Provides O(1) push and O(n) iteration for visualization.
#[derive(Debug, Clone)]
pub struct MetricsBuffer {
    data: Vec<f32>,
    capacity: usize,
    write_idx: usize,
    len: usize,
}

impl MetricsBuffer {
    /// Create a new metrics buffer with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            capacity,
            write_idx: 0,
            len: 0,
        }
    }

    /// Push a new value, overwriting oldest if full.
    pub fn push(&mut self, value: f32) {
        self.data[self.write_idx] = value;
        self.write_idx = (self.write_idx + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Get the number of values in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the capacity of the buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the last N values in chronological order.
    pub fn last_n(&self, n: usize) -> Vec<f32> {
        let n = n.min(self.len);
        if n == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(n);
        let start_idx = if self.len == self.capacity {
            (self.write_idx + self.capacity - n) % self.capacity
        } else {
            self.len.saturating_sub(n)
        };

        for i in 0..n {
            let idx = (start_idx + i) % self.capacity;
            result.push(self.data[idx]);
        }
        result
    }

    /// Get all values in chronological order.
    pub fn values(&self) -> Vec<f32> {
        self.last_n(self.len)
    }

    /// Get the most recent value.
    pub fn last(&self) -> Option<f32> {
        if self.len == 0 {
            None
        } else {
            let idx = (self.write_idx + self.capacity - 1) % self.capacity;
            Some(self.data[idx])
        }
    }

    /// Get min value.
    pub fn min(&self) -> Option<f32> {
        if self.len == 0 {
            return None;
        }
        self.values().into_iter().reduce(f32::min)
    }

    /// Get max value.
    pub fn max(&self) -> Option<f32> {
        if self.len == 0 {
            return None;
        }
        self.values().into_iter().reduce(f32::max)
    }

    /// Get mean value.
    pub fn mean(&self) -> Option<f32> {
        if self.len == 0 {
            return None;
        }
        let sum: f32 = self.values().iter().sum();
        Some(sum / self.len as f32)
    }

    /// Clear all values.
    pub fn clear(&mut self) {
        self.write_idx = 0;
        self.len = 0;
    }
}

// =============================================================================
// Sparkline - Unicode Visualization (ENT-057)
// =============================================================================

/// Unicode sparkline characters for inline metric visualization.
pub const SPARK_CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Generate a sparkline string from a slice of values.
///
/// Uses Unicode block elements to create a compact inline chart.
///
/// # Arguments
///
/// * `values` - The values to visualize
/// * `width` - Maximum width (values will be subsampled if needed)
///
/// # Returns
///
/// A string of Unicode block characters representing the values.
pub fn sparkline(values: &[f32], width: usize) -> String {
    if values.is_empty() || width == 0 {
        return String::new();
    }

    // Subsample if needed
    let values: Vec<f32> = if values.len() > width {
        let step = values.len() as f32 / width as f32;
        (0..width)
            .map(|i| {
                let idx = (i as f32 * step) as usize;
                values[idx.min(values.len() - 1)]
            })
            .collect()
    } else {
        values.to_vec()
    };

    // Find extent
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;

    // Handle constant values
    if range < f32::EPSILON {
        return SPARK_CHARS[4].to_string().repeat(values.len());
    }

    // Map to sparkline characters
    values
        .iter()
        .map(|v| {
            let normalized = (v - min) / range;
            let idx = (normalized * 7.0).round() as usize;
            SPARK_CHARS[idx.min(7)]
        })
        .collect()
}

/// Generate a sparkline with custom range.
pub fn sparkline_range(values: &[f32], width: usize, min: f32, max: f32) -> String {
    if values.is_empty() || width == 0 {
        return String::new();
    }

    let range = max - min;
    if range < f32::EPSILON {
        return SPARK_CHARS[4].to_string().repeat(values.len().min(width));
    }

    let values: Vec<f32> = if values.len() > width {
        let step = values.len() as f32 / width as f32;
        (0..width)
            .map(|i| values[(i as f32 * step) as usize])
            .collect()
    } else {
        values.to_vec()
    };

    values
        .iter()
        .map(|v| {
            let clamped = v.clamp(min, max);
            let normalized = (clamped - min) / range;
            let idx = (normalized * 7.0).round() as usize;
            SPARK_CHARS[idx.min(7)]
        })
        .collect()
}

// =============================================================================
// Progress Bar with ETA (ENT-058)
// =============================================================================

/// Kalman filter for ETA estimation.
#[derive(Debug, Clone)]
pub struct KalmanEta {
    /// Estimated step duration (seconds)
    estimate: f64,
    /// Error covariance
    error_cov: f64,
    /// Process noise
    process_noise: f64,
    /// Measurement noise
    measurement_noise: f64,
}

impl Default for KalmanEta {
    fn default() -> Self {
        Self {
            estimate: 1.0,
            error_cov: 1.0,
            process_noise: 0.01,
            measurement_noise: 0.1,
        }
    }
}

impl KalmanEta {
    /// Create a new Kalman filter for ETA estimation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update with a new step duration measurement.
    pub fn update(&mut self, measured_duration: f64) {
        // Prediction step
        let predicted_estimate = self.estimate;
        let predicted_error = self.error_cov + self.process_noise;

        // Update step
        let kalman_gain = predicted_error / (predicted_error + self.measurement_noise);
        self.estimate = predicted_estimate + kalman_gain * (measured_duration - predicted_estimate);
        self.error_cov = (1.0 - kalman_gain) * predicted_error;
    }

    /// Get estimated time remaining for N steps.
    pub fn eta_seconds(&self, remaining_steps: usize) -> f64 {
        self.estimate * remaining_steps as f64
    }

    /// Format ETA as human-readable string.
    pub fn eta_string(&self, remaining_steps: usize) -> String {
        let secs = self.eta_seconds(remaining_steps);
        format_duration(secs)
    }
}

/// Format duration in seconds to human-readable string.
pub fn format_duration(secs: f64) -> String {
    if secs < 60.0 {
        format!("{secs:.0}s")
    } else if secs < 3600.0 {
        let mins = (secs / 60.0).floor();
        let s = (secs % 60.0).floor();
        format!("{mins}m {s:02.0}s")
    } else {
        let hours = (secs / 3600.0).floor();
        let mins = ((secs % 3600.0) / 60.0).floor();
        format!("{hours}h {mins:02.0}m")
    }
}

/// Progress bar renderer.
#[derive(Debug, Clone)]
pub struct ProgressBar {
    /// Total steps
    total: usize,
    /// Current step
    current: usize,
    /// Bar width in characters
    width: usize,
    /// Fill character
    fill_char: char,
    /// Empty character
    empty_char: char,
    /// Kalman filter for ETA
    kalman: KalmanEta,
    /// Last step time
    last_step_time: Option<Instant>,
}

impl ProgressBar {
    /// Create a new progress bar.
    pub fn new(total: usize, width: usize) -> Self {
        Self {
            total,
            current: 0,
            width,
            fill_char: '█',
            empty_char: '░',
            kalman: KalmanEta::new(),
            last_step_time: None,
        }
    }

    /// Update progress.
    pub fn update(&mut self, current: usize) {
        let now = Instant::now();
        if let Some(last_time) = self.last_step_time {
            let elapsed = now.duration_since(last_time).as_secs_f64();
            let steps = current.saturating_sub(self.current);
            if steps > 0 {
                let per_step = elapsed / steps as f64;
                self.kalman.update(per_step);
            }
        }
        self.current = current;
        self.last_step_time = Some(now);
    }

    /// Get progress percentage.
    pub fn percent(&self) -> f32 {
        if self.total == 0 {
            return 100.0;
        }
        (self.current as f32 / self.total as f32) * 100.0
    }

    /// Render progress bar to string.
    pub fn render(&self) -> String {
        let percent = self.percent();
        let filled = ((percent / 100.0) * self.width as f32).round() as usize;
        let empty = self.width.saturating_sub(filled);

        let bar: String = std::iter::repeat_n(self.fill_char, filled)
            .chain(std::iter::repeat_n(self.empty_char, empty))
            .collect();

        let remaining = self.total.saturating_sub(self.current);
        let eta = self.kalman.eta_string(remaining);

        format!("[{bar}] {percent:>5.1}% │ ETA: {eta}")
    }
}

// =============================================================================
// Refresh Policy (ENT-060)
// =============================================================================

/// Adaptive refresh rate policy.
#[derive(Debug, Clone)]
pub struct RefreshPolicy {
    /// Minimum interval between refreshes
    pub min_interval: Duration,
    /// Maximum interval (force refresh)
    pub max_interval: Duration,
    /// Refresh every N steps
    pub step_interval: usize,
    /// Last refresh time
    last_refresh: Instant,
    /// Last refresh step
    last_step: usize,
}

impl Default for RefreshPolicy {
    fn default() -> Self {
        Self {
            min_interval: Duration::from_millis(50),
            max_interval: Duration::from_millis(1000),
            step_interval: 10,
            last_refresh: Instant::now(),
            last_step: 0,
        }
    }
}

impl RefreshPolicy {
    /// Create a new refresh policy.
    pub fn new(min_ms: u64, max_ms: u64, step_interval: usize) -> Self {
        Self {
            min_interval: Duration::from_millis(min_ms),
            max_interval: Duration::from_millis(max_ms),
            step_interval,
            last_refresh: Instant::now(),
            last_step: 0,
        }
    }

    /// Check if a refresh should occur.
    pub fn should_refresh(&mut self, global_step: usize) -> bool {
        let elapsed = self.last_refresh.elapsed();

        // Force refresh after max interval
        if elapsed >= self.max_interval {
            self.last_refresh = Instant::now();
            self.last_step = global_step;
            return true;
        }

        // Rate-limit to min interval
        if elapsed < self.min_interval {
            return false;
        }

        // Step-based refresh
        if global_step.saturating_sub(self.last_step) >= self.step_interval {
            self.last_refresh = Instant::now();
            self.last_step = global_step;
            return true;
        }

        false
    }

    /// Force a refresh (resets timer).
    pub fn force_refresh(&mut self, global_step: usize) {
        self.last_refresh = Instant::now();
        self.last_step = global_step;
    }
}

// =============================================================================
// Andon System - Health Monitoring (ENT-066)
// =============================================================================

/// Alert severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertLevel {
    /// Informational message
    Info,
    /// Warning - training may be suboptimal
    Warning,
    /// Critical - training should stop
    Critical,
}

/// Training health alert.
#[derive(Debug, Clone)]
pub struct Alert {
    /// Alert level
    pub level: AlertLevel,
    /// Alert message
    pub message: String,
    /// Timestamp
    pub timestamp: Instant,
}

/// Andon system for training health monitoring.
///
/// Implements Jidoka (automation with a human touch) principles:
/// - Detects abnormalities automatically
/// - Alerts immediately
/// - Stops training if critical
#[derive(Debug)]
pub struct AndonSystem {
    /// Active alerts
    alerts: Vec<Alert>,
    /// Whether to stop on critical
    stop_on_critical: bool,
    /// Loss history for divergence detection
    loss_history: MetricsBuffer,
    /// EMA of loss for divergence detection
    loss_ema: f32,
    /// EMA alpha
    ema_alpha: f32,
    /// Sigma threshold for divergence
    sigma_threshold: f32,
    /// Steps since last improvement
    stall_counter: usize,
    /// Best loss seen
    best_loss: f32,
    /// Stall threshold (steps)
    stall_threshold: usize,
}

impl Default for AndonSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl AndonSystem {
    /// Create a new Andon system.
    pub fn new() -> Self {
        Self {
            alerts: Vec::new(),
            stop_on_critical: true,
            loss_history: MetricsBuffer::new(100),
            loss_ema: 0.0,
            ema_alpha: 0.1,
            sigma_threshold: 3.0,
            stall_counter: 0,
            best_loss: f32::INFINITY,
            stall_threshold: 1000,
        }
    }

    /// Configure sigma threshold for divergence detection.
    pub fn with_sigma_threshold(mut self, sigma: f32) -> Self {
        self.sigma_threshold = sigma;
        self
    }

    /// Configure stall detection threshold.
    pub fn with_stall_threshold(mut self, steps: usize) -> Self {
        self.stall_threshold = steps;
        self
    }

    /// Configure whether to stop on critical alerts.
    pub fn with_stop_on_critical(mut self, stop: bool) -> Self {
        self.stop_on_critical = stop;
        self
    }

    /// Check loss value for abnormalities.
    ///
    /// Returns `true` if training should stop.
    pub fn check_loss(&mut self, loss: f32) -> bool {
        // Check for NaN/Inf
        if loss.is_nan() {
            self.critical("NaN loss detected - training diverged");
            return self.stop_on_critical;
        }

        if loss.is_infinite() {
            self.critical("Infinite loss detected - training diverged");
            return self.stop_on_critical;
        }

        // Update EMA
        if self.loss_history.is_empty() {
            self.loss_ema = loss;
        } else {
            self.loss_ema = self.ema_alpha * loss + (1.0 - self.ema_alpha) * self.loss_ema;
        }

        // Check for divergence (loss >> EMA)
        if self.loss_history.len() > 10 {
            if let (Some(mean), Some(std)) = (self.loss_history.mean(), self.loss_std()) {
                let z_score = (loss - mean) / std.max(f32::EPSILON);
                if z_score > self.sigma_threshold {
                    self.warning(format!(
                        "Loss spike detected: {loss:.4} ({z_score:.1}σ above mean)"
                    ));
                }
            }
        }

        // Check for stall
        if loss < self.best_loss {
            self.best_loss = loss;
            self.stall_counter = 0;
        } else {
            self.stall_counter += 1;
            if self.stall_counter >= self.stall_threshold {
                self.warning(format!(
                    "Training stalled: no improvement for {} steps",
                    self.stall_counter
                ));
            }
        }

        self.loss_history.push(loss);
        false
    }

    /// Calculate standard deviation of loss history.
    fn loss_std(&self) -> Option<f32> {
        let values = self.loss_history.values();
        if values.len() < 2 {
            return None;
        }
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        Some(variance.sqrt())
    }

    /// Add an info alert.
    pub fn info(&mut self, message: impl Into<String>) {
        self.alerts.push(Alert {
            level: AlertLevel::Info,
            message: message.into(),
            timestamp: Instant::now(),
        });
    }

    /// Add a warning alert.
    pub fn warning(&mut self, message: impl Into<String>) {
        self.alerts.push(Alert {
            level: AlertLevel::Warning,
            message: message.into(),
            timestamp: Instant::now(),
        });
    }

    /// Add a critical alert.
    pub fn critical(&mut self, message: impl Into<String>) {
        self.alerts.push(Alert {
            level: AlertLevel::Critical,
            message: message.into(),
            timestamp: Instant::now(),
        });
    }

    /// Check if there are any critical alerts.
    pub fn has_critical(&self) -> bool {
        self.alerts.iter().any(|a| a.level == AlertLevel::Critical)
    }

    /// Check if training should stop.
    pub fn should_stop(&self) -> bool {
        self.stop_on_critical && self.has_critical()
    }

    /// Get recent alerts.
    pub fn recent_alerts(&self, count: usize) -> &[Alert] {
        let start = self.alerts.len().saturating_sub(count);
        &self.alerts[start..]
    }

    /// Clear all alerts.
    pub fn clear_alerts(&mut self) {
        self.alerts.clear();
    }
}

// =============================================================================
// Terminal Monitor Callback (ENT-054)
// =============================================================================

/// Terminal rendering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TerminalMode {
    /// ASCII only (widest compatibility)
    Ascii,
    /// Unicode characters (modern terminals)
    #[default]
    Unicode,
    /// ANSI color codes
    Ansi,
}

/// Dashboard layout style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DashboardLayout {
    /// Single-line progress only
    Minimal,
    /// Compact 5-line summary
    #[default]
    Compact,
    /// Full dashboard with charts
    Full,
}

/// Real-time terminal monitoring callback.
///
/// Integrates with training loop to provide live visualization.
#[derive(Debug)]
pub struct TerminalMonitorCallback {
    /// Loss buffer
    loss_buffer: MetricsBuffer,
    /// Validation loss buffer
    val_loss_buffer: MetricsBuffer,
    /// Learning rate buffer
    lr_buffer: MetricsBuffer,
    /// Progress bar
    progress: ProgressBar,
    /// Refresh policy
    refresh_policy: RefreshPolicy,
    /// Andon system
    andon: AndonSystem,
    /// Terminal mode
    mode: TerminalMode,
    /// Dashboard layout
    layout: DashboardLayout,
    /// Sparkline width
    sparkline_width: usize,
    /// Start time
    start_time: Instant,
    /// Model name (for display)
    model_name: String,
}

impl Default for TerminalMonitorCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl TerminalMonitorCallback {
    /// Create a new terminal monitor callback.
    pub fn new() -> Self {
        Self {
            loss_buffer: MetricsBuffer::new(100),
            val_loss_buffer: MetricsBuffer::new(100),
            lr_buffer: MetricsBuffer::new(100),
            progress: ProgressBar::new(100, 30),
            refresh_policy: RefreshPolicy::default(),
            andon: AndonSystem::new(),
            mode: TerminalMode::default(),
            layout: DashboardLayout::default(),
            sparkline_width: 20,
            start_time: Instant::now(),
            model_name: "model".to_string(),
        }
    }

    /// Set terminal mode.
    pub fn mode(mut self, mode: TerminalMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set dashboard layout.
    pub fn layout(mut self, layout: DashboardLayout) -> Self {
        self.layout = layout;
        self
    }

    /// Set model name.
    pub fn model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = name.into();
        self
    }

    /// Set sparkline width.
    pub fn sparkline_width(mut self, width: usize) -> Self {
        self.sparkline_width = width;
        self
    }

    /// Set refresh interval.
    pub fn refresh_interval_ms(mut self, ms: u64) -> Self {
        self.refresh_policy.min_interval = Duration::from_millis(ms);
        self
    }

    /// Render the current display.
    fn render(&self, ctx: &CallbackContext) -> String {
        match self.layout {
            DashboardLayout::Minimal => self.render_minimal(ctx),
            DashboardLayout::Compact => self.render_compact(ctx),
            DashboardLayout::Full => self.render_full(ctx),
        }
    }

    /// Render minimal single-line display.
    fn render_minimal(&self, ctx: &CallbackContext) -> String {
        let percent = (ctx.epoch as f32 / ctx.max_epochs as f32) * 100.0;
        format!(
            "\rEpoch {}/{} [{:.1}%] loss={:.4} lr={:.2e}",
            ctx.epoch + 1,
            ctx.max_epochs,
            percent,
            ctx.loss,
            ctx.lr
        )
    }

    /// Render compact 5-line display.
    fn render_compact(&self, ctx: &CallbackContext) -> String {
        let loss_spark = sparkline(&self.loss_buffer.values(), self.sparkline_width);
        let elapsed = self.start_time.elapsed().as_secs_f64();

        let val_info = ctx
            .val_loss
            .map(|v| format!(" val={v:.4}"))
            .unwrap_or_default();

        let best_info = self
            .loss_buffer
            .min()
            .map(|m| format!(" best={m:.4}"))
            .unwrap_or_default();

        format!(
            "\x1b[H\x1b[2J\
             ═══ {} Training ═══\n\
             Epoch {}/{} │ loss={:.4}{}{}\n\
             Loss: {} \n\
             LR: {:.2e} │ {:.1} steps/s\n\
             {}",
            self.model_name,
            ctx.epoch + 1,
            ctx.max_epochs,
            ctx.loss,
            val_info,
            best_info,
            loss_spark,
            ctx.lr,
            ctx.global_step as f64 / elapsed.max(0.001),
            self.progress.render()
        )
    }

    /// Render full dashboard display.
    fn render_full(&self, ctx: &CallbackContext) -> String {
        let loss_spark = sparkline(&self.loss_buffer.values(), self.sparkline_width);
        let lr_spark = sparkline(&self.lr_buffer.values(), self.sparkline_width);
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let steps_per_sec = ctx.global_step as f64 / elapsed.max(0.001);

        let val_spark = if self.val_loss_buffer.is_empty() {
            String::new()
        } else {
            format!(
                "Val Loss: {} {:.4}\n",
                sparkline(&self.val_loss_buffer.values(), self.sparkline_width),
                self.val_loss_buffer.last().unwrap_or(0.0)
            )
        };

        let alerts = self.render_alerts();

        format!(
            "\x1b[H\x1b[2J\
╔═══════════════════════════════════════════════════════════════════╗
║  ENTRENAR TRAINING MONITOR                              [RUNNING] ║
╠═══════════════════════════════════════════════════════════════════╣
║  Model: {:<20} │ Epoch: {}/{}                  ║
╠═══════════════════════════════════════════════════════════════════╣
║  Loss: {} {:.4}                                 ║
║  {}║  LR:   {} {:.2e}                                 ║
╠═══════════════════════════════════════════════════════════════════╣
║  Steps/s: {:.1}  │  Elapsed: {}                        ║
║  {}║
╚═══════════════════════════════════════════════════════════════════╝
{}",
            self.model_name,
            ctx.epoch + 1,
            ctx.max_epochs,
            loss_spark,
            ctx.loss,
            val_spark,
            lr_spark,
            ctx.lr,
            steps_per_sec,
            format_duration(elapsed),
            self.progress.render(),
            alerts
        )
    }

    /// Render recent alerts.
    fn render_alerts(&self) -> String {
        let alerts = self.andon.recent_alerts(3);
        if alerts.is_empty() {
            return String::new();
        }

        alerts
            .iter()
            .map(|a| {
                let prefix = match a.level {
                    AlertLevel::Info => "ℹ️ ",
                    AlertLevel::Warning => "⚠️ ",
                    AlertLevel::Critical => "🛑",
                };
                format!("{} {}", prefix, a.message)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Print the display to stdout.
    fn print_display(&self, ctx: &CallbackContext) {
        let output = self.render(ctx);
        print!("{output}");
        let _ = std::io::stdout().flush();
    }
}

impl TrainerCallback for TerminalMonitorCallback {
    fn on_train_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
        self.start_time = Instant::now();
        self.progress = ProgressBar::new(ctx.max_epochs * ctx.steps_per_epoch, 30);

        // Clear screen and hide cursor
        print!("\x1b[?25l\x1b[2J\x1b[H");
        let _ = std::io::stdout().flush();

        CallbackAction::Continue
    }

    fn on_train_end(&mut self, ctx: &CallbackContext) {
        // Final render
        self.print_display(ctx);

        // Show cursor
        println!("\x1b[?25h");
        let _ = std::io::stdout().flush();

        // Print summary
        println!("\nTraining complete!");
        if let Some(best) = self.loss_buffer.min() {
            println!("Best loss: {best:.4}");
        }
        println!(
            "Total time: {}",
            format_duration(self.start_time.elapsed().as_secs_f64())
        );
    }

    fn on_step_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        // Record metrics
        self.loss_buffer.push(ctx.loss);
        self.lr_buffer.push(ctx.lr);
        if let Some(val) = ctx.val_loss {
            self.val_loss_buffer.push(val);
        }

        // Update progress
        self.progress.update(ctx.global_step);

        // Check health
        if self.andon.check_loss(ctx.loss) {
            return CallbackAction::Stop;
        }

        // Rate-limited refresh
        if self.refresh_policy.should_refresh(ctx.global_step) {
            self.print_display(ctx);
        }

        CallbackAction::Continue
    }

    fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        // Force refresh at epoch boundaries
        self.refresh_policy.force_refresh(ctx.global_step);
        self.print_display(ctx);
        CallbackAction::Continue
    }

    fn name(&self) -> &'static str {
        "TerminalMonitorCallback"
    }
}

// =============================================================================
// Terminal Capability Detection (ENT-061)
// =============================================================================

/// Detected terminal capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerminalCapabilities {
    /// Terminal width in columns
    pub width: u16,
    /// Terminal height in rows
    pub height: u16,
    /// Supports Unicode characters
    pub unicode: bool,
    /// Supports ANSI color codes
    pub ansi_color: bool,
    /// Supports 24-bit true color
    pub true_color: bool,
    /// Is interactive TTY
    pub is_tty: bool,
}

impl Default for TerminalCapabilities {
    fn default() -> Self {
        Self {
            width: 80,
            height: 24,
            unicode: true,
            ansi_color: true,
            true_color: false,
            is_tty: true,
        }
    }
}

impl TerminalCapabilities {
    /// Detect terminal capabilities from environment.
    pub fn detect() -> Self {
        use std::env;
        use std::io::{stdout, IsTerminal};

        let is_tty = stdout().is_terminal();

        // Get size from environment or default
        let (width, height) = Self::get_size();

        // Check for Unicode support (most modern terminals)
        let lang = env::var("LANG").unwrap_or_default();
        let unicode = lang.contains("UTF") || lang.contains("utf");

        // Check for ANSI color support
        let term = env::var("TERM").unwrap_or_default();
        let ansi_color = !term.is_empty() && term != "dumb";

        // Check for true color support
        let colorterm = env::var("COLORTERM").unwrap_or_default();
        let true_color = colorterm == "truecolor" || colorterm == "24bit";

        Self {
            width,
            height,
            unicode,
            ansi_color,
            true_color,
            is_tty,
        }
    }

    /// Get terminal size.
    fn get_size() -> (u16, u16) {
        use std::env;

        // 1. Check environment variables (CI/headless)
        if let (Ok(cols), Ok(rows)) = (env::var("COLUMNS"), env::var("LINES")) {
            if let (Ok(c), Ok(r)) = (cols.parse(), rows.parse()) {
                return (c, r);
            }
        }

        // 2. Try ioctl on Unix
        #[cfg(unix)]
        {
            use std::io::{stdout, IsTerminal};
            if stdout().is_terminal() {
                // Use libc directly for TIOCGWINSZ
                #[repr(C)]
                struct WinSize {
                    ws_row: u16,
                    ws_col: u16,
                    ws_xpixel: u16,
                    ws_ypixel: u16,
                }
                extern "C" {
                    fn ioctl(fd: i32, request: u64, ...) -> i32;
                }
                const TIOCGWINSZ: u64 = 0x5413; // Linux
                let mut ws = WinSize {
                    ws_row: 0,
                    ws_col: 0,
                    ws_xpixel: 0,
                    ws_ypixel: 0,
                };
                // SAFETY: ioctl with TIOCGWINSZ is safe for reading terminal size
                #[allow(unsafe_code)]
                if unsafe { ioctl(1, TIOCGWINSZ, &mut ws) } == 0 && ws.ws_col > 0 {
                    return (ws.ws_col, ws.ws_row);
                }
            }
        }

        // 3. Fallback
        (80, 24)
    }

    /// Get recommended terminal mode based on capabilities.
    pub fn recommended_mode(&self) -> TerminalMode {
        if !self.is_tty {
            TerminalMode::Ascii
        } else if self.true_color {
            TerminalMode::Ansi
        } else if self.unicode {
            TerminalMode::Unicode
        } else {
            TerminalMode::Ascii
        }
    }
}

// =============================================================================
// Feature Importance Display (ENT-064)
// =============================================================================

/// Feature importance bar chart for terminal display.
#[derive(Debug, Clone)]
pub struct FeatureImportanceChart {
    /// Feature names
    names: Vec<String>,
    /// Importance scores
    scores: Vec<f32>,
    /// Bar width
    bar_width: usize,
    /// Number of features to show
    top_k: usize,
}

impl FeatureImportanceChart {
    /// Create a new feature importance chart.
    pub fn new(top_k: usize, bar_width: usize) -> Self {
        Self {
            names: Vec::new(),
            scores: Vec::new(),
            bar_width,
            top_k,
        }
    }

    /// Update with new importance scores.
    pub fn update(&mut self, importances: &[(usize, f32)], feature_names: Option<&[String]>) {
        let mut sorted: Vec<_> = importances.to_vec();
        sorted.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(self.top_k);

        self.names.clear();
        self.scores.clear();

        for (idx, score) in sorted {
            let name = feature_names
                .and_then(|n| n.get(idx))
                .cloned()
                .unwrap_or_else(|| format!("feature_{idx}"));
            self.names.push(name);
            self.scores.push(score);
        }
    }

    /// Render to string.
    pub fn render(&self) -> String {
        if self.names.is_empty() {
            return String::from("No feature importance data");
        }

        let max_name_len = self.names.iter().map(String::len).max().unwrap_or(10);
        let max_score = self.scores.iter().copied().fold(0.0f32, f32::max);

        let mut output = String::new();
        output.push_str("┌─ Feature Importance ─────────────────────────────┐\n");

        for (name, score) in self.names.iter().zip(self.scores.iter()) {
            let bar_len = if max_score > 0.0 {
                ((score / max_score) * self.bar_width as f32).round() as usize
            } else {
                0
            };
            let bar: String = "█".repeat(bar_len);
            output.push_str(&format!(
                "│  {:width$}  {:bar_width$}  {:.3}  │\n",
                name,
                bar,
                score,
                width = max_name_len,
                bar_width = self.bar_width
            ));
        }

        output.push_str("└──────────────────────────────────────────────────┘\n");
        output
    }
}

// =============================================================================
// Gradient Flow Heatmap (ENT-065)
// =============================================================================

/// Gradient flow heatmap for visualizing per-layer gradients.
#[derive(Debug, Clone)]
pub struct GradientFlowHeatmap {
    /// Layer names
    layer_names: Vec<String>,
    /// Gradient magnitudes per layer (log scale)
    gradients: Vec<Vec<f32>>,
    /// Column labels (Q, K, V, O, FFN, etc.)
    column_labels: Vec<String>,
}

impl GradientFlowHeatmap {
    /// Create a new gradient flow heatmap.
    pub fn new(layer_names: Vec<String>, column_labels: Vec<String>) -> Self {
        let num_layers = layer_names.len();
        Self {
            layer_names,
            gradients: vec![vec![0.0; column_labels.len()]; num_layers],
            column_labels,
        }
    }

    /// Update gradient for a specific layer and column.
    pub fn update(&mut self, layer: usize, col: usize, grad_norm: f32) {
        if layer < self.gradients.len() && col < self.column_labels.len() {
            // Store log scale for visualization
            self.gradients[layer][col] = (grad_norm + 1e-8).ln();
        }
    }

    /// Render to string.
    pub fn render(&self) -> String {
        let heatmap_chars = ['░', '▒', '▓', '█'];

        // Find min/max for normalization
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for row in &self.gradients {
            for &v in row {
                min = min.min(v);
                max = max.max(v);
            }
        }
        let range = max - min;

        let mut output = String::new();
        output.push_str("Gradient Flow (log scale):\n");

        // Header
        output.push_str("         ");
        for label in &self.column_labels {
            output.push_str(&format!("{label:^5}"));
        }
        output.push('\n');

        // Rows
        for (i, row) in self.gradients.iter().enumerate() {
            let name = self.layer_names.get(i).map_or("?", String::as_str);
            output.push_str(&format!("{name:>8} "));

            for &v in row {
                let normalized = if range > f32::EPSILON {
                    ((v - min) / range).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                let idx = (normalized * 3.0).round() as usize;
                let c = heatmap_chars[idx.min(3)];
                output.push_str(&format!("{c}{c}{c}{c} "));
            }
            output.push('\n');
        }

        output
    }
}

// =============================================================================
// Reference Curve Overlay (ENT-067)
// =============================================================================

/// Reference curve for comparison with current training run.
#[derive(Debug, Clone)]
pub struct ReferenceCurve {
    /// Reference values (from a "golden" run)
    values: Vec<f32>,
    /// Tolerance for deviation detection
    tolerance: f32,
}

impl ReferenceCurve {
    /// Create from a vector of reference values.
    pub fn new(values: Vec<f32>, tolerance: f32) -> Self {
        Self { values, tolerance }
    }

    /// Load from JSON file.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let values: Vec<f32> = serde_json::from_str(json)?;
        Ok(Self::new(values, 0.1))
    }

    /// Get reference value at epoch.
    pub fn get(&self, epoch: usize) -> Option<f32> {
        self.values.get(epoch).copied()
    }

    /// Check if current value deviates from reference.
    pub fn check_deviation(&self, epoch: usize, current: f32) -> Option<f32> {
        if let Some(reference) = self.get(epoch) {
            let deviation = (current - reference).abs() / reference.abs().max(f32::EPSILON);
            if deviation > self.tolerance {
                return Some(deviation);
            }
        }
        None
    }

    /// Generate comparison sparkline.
    pub fn comparison_sparkline(&self, current: &[f32], width: usize) -> String {
        let len = current.len().min(self.values.len());
        if len == 0 {
            return String::new();
        }

        // Show deviation from reference
        let deviations: Vec<f32> = current
            .iter()
            .zip(self.values.iter())
            .map(|(c, r)| (c - r) / r.abs().max(f32::EPSILON))
            .collect();

        // Use signed sparkline (negative = better, positive = worse for loss)
        sparkline_range(&deviations, width, -0.5, 0.5)
    }
}

// =============================================================================
// YAML Configuration (ENT-062)
// =============================================================================

/// Monitor configuration for YAML.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct MonitorConfig {
    /// Enable terminal monitoring
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Dashboard layout
    #[serde(default)]
    pub layout: String,
    /// Terminal mode (ascii, unicode, ansi)
    #[serde(default)]
    pub terminal_mode: String,
    /// Refresh interval in milliseconds
    #[serde(default = "default_refresh")]
    pub refresh_ms: u64,
    /// Sparkline width
    #[serde(default = "default_sparkline_width")]
    pub sparkline_width: usize,
    /// Show ETA
    #[serde(default = "default_true")]
    pub show_eta: bool,
    /// Reference curve path (optional)
    #[serde(default)]
    pub reference_curve: Option<String>,
}

fn default_true() -> bool {
    true
}
fn default_refresh() -> u64 {
    100
}
fn default_sparkline_width() -> usize {
    20
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            layout: "compact".to_string(),
            terminal_mode: "auto".to_string(),
            refresh_ms: 100,
            sparkline_width: 20,
            show_eta: true,
            reference_curve: None,
        }
    }
}

impl MonitorConfig {
    /// Create TerminalMonitorCallback from config.
    pub fn to_callback(&self) -> TerminalMonitorCallback {
        let layout = match self.layout.as_str() {
            "minimal" => DashboardLayout::Minimal,
            "full" => DashboardLayout::Full,
            _ => DashboardLayout::Compact,
        };

        let mode = match self.terminal_mode.as_str() {
            "ascii" => TerminalMode::Ascii,
            "ansi" => TerminalMode::Ansi,
            "unicode" => TerminalMode::Unicode,
            _ => TerminalCapabilities::detect().recommended_mode(),
        };

        TerminalMonitorCallback::new()
            .layout(layout)
            .mode(mode)
            .sparkline_width(self.sparkline_width)
            .refresh_interval_ms(self.refresh_ms)
    }
}

// =============================================================================
// LossCurve Integration (ENT-056)
// =============================================================================

/// Summary of a metric series: (name, min_value, last_smoothed, best_epoch).
pub type SeriesSummaryTuple = (String, Option<f32>, Option<f32>, Option<usize>);

/// Wrapper for trueno-viz LossCurve with terminal output support.
///
/// Provides streaming loss curve visualization with:
/// - Train and validation loss tracking
/// - Exponential moving average smoothing
/// - Best value markers
/// - ASCII/Unicode/ANSI terminal rendering modes
///
/// # Example
///
/// ```no_run
/// use entrenar::train::tui::LossCurveDisplay;
///
/// let mut display = LossCurveDisplay::new(80, 20);
/// display.push_train_loss(1.0);
/// display.push_val_loss(1.2);
/// println!("{}", display.render_terminal());
/// ```
pub struct LossCurveDisplay {
    loss_curve: LossCurve,
    width: u32,
    height: u32,
    terminal_mode: TerminalMode,
}

impl LossCurveDisplay {
    /// Create a new loss curve display.
    pub fn new(width: u32, height: u32) -> Self {
        let loss_curve = LossCurve::new()
            .add_series(MetricSeries::new("Train", Rgba::rgb(66, 133, 244)))
            .add_series(MetricSeries::new("Val", Rgba::rgb(255, 128, 0)))
            .dimensions(width, height)
            .margin(2)
            .best_markers(true)
            .lower_is_better(true)
            .build()
            .expect("LossCurve build should succeed");
        Self {
            loss_curve,
            width,
            height,
            terminal_mode: TerminalMode::Unicode,
        }
    }

    /// Set terminal rendering mode.
    pub fn terminal_mode(mut self, mode: TerminalMode) -> Self {
        self.terminal_mode = mode;
        self
    }

    /// Set smoothing factor (0.0 = none, 0.99 = heavy).
    pub fn smoothing(mut self, factor: f32) -> Self {
        // Re-create with smoothing applied
        self.loss_curve = LossCurve::new()
            .add_series(MetricSeries::new("Train", Rgba::rgb(66, 133, 244)).smoothing(factor))
            .add_series(MetricSeries::new("Val", Rgba::rgb(255, 128, 0)).smoothing(factor))
            .dimensions(self.width, self.height)
            .margin(2)
            .best_markers(true)
            .lower_is_better(true)
            .build()
            .expect("LossCurve build should succeed");
        self
    }

    /// Push a training loss value.
    pub fn push_train_loss(&mut self, value: f32) {
        self.loss_curve.push(0, value);
    }

    /// Push a validation loss value.
    pub fn push_val_loss(&mut self, value: f32) {
        self.loss_curve.push(1, value);
    }

    /// Push both train and val loss at once.
    pub fn push_losses(&mut self, train: f32, val: f32) {
        self.loss_curve.push_all(&[train, val]);
    }

    /// Get the number of epochs recorded.
    pub fn epochs(&self) -> usize {
        self.loss_curve.max_epochs()
    }

    /// Get summary of all series.
    pub fn summary(&self) -> Vec<SeriesSummaryTuple> {
        self.loss_curve
            .summary()
            .into_iter()
            .map(|s| (s.name, s.min, s.last_smoothed, s.best_epoch))
            .collect()
    }

    /// Render to terminal string.
    pub fn render_terminal(&self) -> String {
        if self.loss_curve.max_epochs() < 2 {
            return String::from("(waiting for data...)");
        }

        let fb = match self.loss_curve.to_framebuffer() {
            Ok(fb) => fb,
            Err(_) => return String::from("(render error)"),
        };

        let trueno_mode = match self.terminal_mode {
            TerminalMode::Ascii => TruenoTerminalMode::Ascii,
            TerminalMode::Unicode => TruenoTerminalMode::UnicodeHalfBlock,
            TerminalMode::Ansi => TruenoTerminalMode::AnsiTrueColor,
        };

        let encoder = TerminalEncoder::new()
            .mode(trueno_mode)
            .width(self.width)
            .height(self.height / 2); // Terminal chars are ~2:1 aspect

        encoder.render(&fb)
    }

    /// Print to stdout.
    pub fn print(&self) {
        println!("{}", self.render_terminal());
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // MetricsBuffer Tests (ENT-055)
    // =========================================================================

    #[test]
    fn test_metrics_buffer_new() {
        let buf = MetricsBuffer::new(10);
        assert_eq!(buf.capacity(), 10);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_metrics_buffer_push() {
        let mut buf = MetricsBuffer::new(5);
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);

        assert_eq!(buf.len(), 3);
        assert_eq!(buf.values(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_metrics_buffer_wraparound() {
        let mut buf = MetricsBuffer::new(3);
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);
        buf.push(4.0); // Overwrites 1.0
        buf.push(5.0); // Overwrites 2.0

        assert_eq!(buf.len(), 3);
        assert_eq!(buf.values(), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_metrics_buffer_last_n() {
        let mut buf = MetricsBuffer::new(10);
        for i in 0..10 {
            buf.push(i as f32);
        }

        assert_eq!(buf.last_n(3), vec![7.0, 8.0, 9.0]);
        assert_eq!(buf.last_n(1), vec![9.0]);
        assert_eq!(buf.last_n(0), Vec::<f32>::new());
    }

    #[test]
    fn test_metrics_buffer_last() {
        let mut buf = MetricsBuffer::new(5);
        assert_eq!(buf.last(), None);

        buf.push(1.0);
        assert_eq!(buf.last(), Some(1.0));

        buf.push(2.0);
        assert_eq!(buf.last(), Some(2.0));
    }

    #[test]
    fn test_metrics_buffer_stats() {
        let mut buf = MetricsBuffer::new(5);
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);
        buf.push(4.0);
        buf.push(5.0);

        assert_eq!(buf.min(), Some(1.0));
        assert_eq!(buf.max(), Some(5.0));
        assert_eq!(buf.mean(), Some(3.0));
    }

    #[test]
    fn test_metrics_buffer_clear() {
        let mut buf = MetricsBuffer::new(5);
        buf.push(1.0);
        buf.push(2.0);
        buf.clear();

        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    // =========================================================================
    // Sparkline Tests (ENT-057)
    // =========================================================================

    #[test]
    fn test_sparkline_empty() {
        assert_eq!(sparkline(&[], 10), "");
    }

    #[test]
    fn test_sparkline_single() {
        let result = sparkline(&[0.5], 10);
        assert_eq!(result.chars().count(), 1);
    }

    #[test]
    fn test_sparkline_constant() {
        let result = sparkline(&[0.5, 0.5, 0.5], 10);
        assert!(result.chars().all(|c| c == SPARK_CHARS[4]));
    }

    #[test]
    fn test_sparkline_increasing() {
        let values: Vec<f32> = (0..8).map(|i| i as f32 / 7.0).collect();
        let result = sparkline(&values, 8);

        // First should be lowest, last should be highest
        let chars: Vec<char> = result.chars().collect();
        assert_eq!(chars[0], SPARK_CHARS[0]);
        assert_eq!(chars[7], SPARK_CHARS[7]);
    }

    #[test]
    fn test_sparkline_subsample() {
        let values: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let result = sparkline(&values, 10);
        assert_eq!(result.chars().count(), 10);
    }

    #[test]
    fn test_sparkline_range() {
        let values = vec![0.5, 0.5, 0.5];
        let result = sparkline_range(&values, 3, 0.0, 1.0);
        // 0.5 normalized to [0,1] should be middle character
        assert!(result.chars().all(|c| c == SPARK_CHARS[4]));
    }

    // =========================================================================
    // Progress Bar Tests (ENT-058)
    // =========================================================================

    #[test]
    fn test_progress_bar_new() {
        let bar = ProgressBar::new(100, 20);
        assert_eq!(bar.percent(), 0.0);
    }

    #[test]
    fn test_progress_bar_update() {
        let mut bar = ProgressBar::new(100, 20);
        bar.update(50);
        assert!((bar.percent() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_progress_bar_render() {
        let mut bar = ProgressBar::new(100, 10);
        bar.update(50);
        let output = bar.render();
        assert!(output.contains('['));
        assert!(output.contains(']'));
        assert!(output.contains("50"));
    }

    #[test]
    fn test_kalman_eta() {
        let mut kalman = KalmanEta::new();
        kalman.update(1.0);
        kalman.update(1.0);
        kalman.update(1.0);

        let eta = kalman.eta_seconds(10);
        assert!((eta - 10.0).abs() < 1.0);
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30.0), "30s");
        assert_eq!(format_duration(90.0), "1m 30s");
        assert_eq!(format_duration(3700.0), "1h 01m");
    }

    // =========================================================================
    // Refresh Policy Tests (ENT-060)
    // =========================================================================

    #[test]
    fn test_refresh_policy_default() {
        let policy = RefreshPolicy::default();
        assert_eq!(policy.min_interval, Duration::from_millis(50));
        assert_eq!(policy.max_interval, Duration::from_millis(1000));
    }

    #[test]
    fn test_refresh_policy_step_interval() {
        let mut policy = RefreshPolicy::new(0, 10000, 5);
        // Force initial refresh
        policy.force_refresh(0);

        // Steps 1-4 should not trigger refresh
        assert!(!policy.should_refresh(1));
        assert!(!policy.should_refresh(2));
        assert!(!policy.should_refresh(3));
        assert!(!policy.should_refresh(4));
        // Step 5 should trigger (5 steps since last refresh at 0)
        assert!(policy.should_refresh(5));
        // Step 6 should not (only 1 step since last at 5)
        assert!(!policy.should_refresh(6));
    }

    // =========================================================================
    // Andon Tests (ENT-066)
    // =========================================================================

    #[test]
    fn test_andon_new() {
        let andon = AndonSystem::new();
        assert!(!andon.has_critical());
        assert!(!andon.should_stop());
    }

    #[test]
    fn test_andon_nan_detection() {
        let mut andon = AndonSystem::new();
        let should_stop = andon.check_loss(f32::NAN);
        assert!(should_stop);
        assert!(andon.has_critical());
    }

    #[test]
    fn test_andon_inf_detection() {
        let mut andon = AndonSystem::new();
        let should_stop = andon.check_loss(f32::INFINITY);
        assert!(should_stop);
        assert!(andon.has_critical());
    }

    #[test]
    fn test_andon_normal_loss() {
        let mut andon = AndonSystem::new();
        let should_stop = andon.check_loss(0.5);
        assert!(!should_stop);
        assert!(!andon.has_critical());
    }

    #[test]
    fn test_andon_alerts() {
        let mut andon = AndonSystem::new();
        andon.info("Test info");
        andon.warning("Test warning");
        andon.critical("Test critical");

        let alerts = andon.recent_alerts(10);
        assert_eq!(alerts.len(), 3);
        assert!(andon.has_critical());
    }

    // =========================================================================
    // Terminal Monitor Tests (ENT-054)
    // =========================================================================

    #[test]
    fn test_terminal_monitor_new() {
        let monitor = TerminalMonitorCallback::new();
        assert_eq!(monitor.mode, TerminalMode::Unicode);
        assert_eq!(monitor.layout, DashboardLayout::Compact);
    }

    #[test]
    fn test_terminal_monitor_builder() {
        let monitor = TerminalMonitorCallback::new()
            .mode(TerminalMode::Ascii)
            .layout(DashboardLayout::Full)
            .model_name("test-model")
            .sparkline_width(15);

        assert_eq!(monitor.mode, TerminalMode::Ascii);
        assert_eq!(monitor.layout, DashboardLayout::Full);
        assert_eq!(monitor.model_name, "test-model");
        assert_eq!(monitor.sparkline_width, 15);
    }

    #[test]
    fn test_terminal_monitor_render_minimal() {
        let monitor = TerminalMonitorCallback::new().layout(DashboardLayout::Minimal);

        let ctx = CallbackContext {
            epoch: 5,
            max_epochs: 10,
            loss: 0.1234,
            lr: 0.001,
            ..Default::default()
        };

        let output = monitor.render(&ctx);
        assert!(output.contains("6/10"));
        assert!(output.contains("0.1234"));
    }

    #[test]
    fn test_terminal_monitor_render_compact() {
        let mut monitor = TerminalMonitorCallback::new().layout(DashboardLayout::Compact);
        monitor.loss_buffer.push(0.5);
        monitor.loss_buffer.push(0.4);
        monitor.loss_buffer.push(0.3);

        let ctx = CallbackContext {
            epoch: 5,
            max_epochs: 10,
            loss: 0.3,
            lr: 0.001,
            ..Default::default()
        };

        let output = monitor.render(&ctx);
        assert!(output.contains("Training"));
        assert!(output.contains("Loss"));
    }

    // =========================================================================
    // Terminal Capabilities Tests (ENT-061)
    // =========================================================================

    #[test]
    fn test_terminal_capabilities_default() {
        let caps = TerminalCapabilities::default();
        assert_eq!(caps.width, 80);
        assert_eq!(caps.height, 24);
        assert!(caps.unicode);
    }

    #[test]
    fn test_terminal_capabilities_recommended_mode() {
        let mut caps = TerminalCapabilities::default();

        caps.is_tty = false;
        assert_eq!(caps.recommended_mode(), TerminalMode::Ascii);

        caps.is_tty = true;
        caps.true_color = true;
        assert_eq!(caps.recommended_mode(), TerminalMode::Ansi);

        caps.true_color = false;
        caps.unicode = true;
        assert_eq!(caps.recommended_mode(), TerminalMode::Unicode);
    }

    // =========================================================================
    // Feature Importance Tests (ENT-064)
    // =========================================================================

    #[test]
    fn test_feature_importance_chart_new() {
        let chart = FeatureImportanceChart::new(5, 20);
        assert_eq!(chart.top_k, 5);
        assert_eq!(chart.bar_width, 20);
    }

    #[test]
    fn test_feature_importance_chart_update() {
        let mut chart = FeatureImportanceChart::new(3, 10);
        let importances = vec![(0, 0.5), (1, 0.8), (2, 0.3), (3, 0.9)];
        let names = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];

        chart.update(&importances, Some(&names));

        assert_eq!(chart.names.len(), 3);
        assert_eq!(chart.names[0], "d"); // Highest score
    }

    #[test]
    fn test_feature_importance_chart_render() {
        let mut chart = FeatureImportanceChart::new(2, 10);
        let importances = vec![(0, 0.5), (1, 1.0)];

        chart.update(&importances, None);
        let output = chart.render();

        assert!(output.contains("Feature Importance"));
        assert!(output.contains("█"));
    }

    // =========================================================================
    // Gradient Flow Heatmap Tests (ENT-065)
    // =========================================================================

    #[test]
    fn test_gradient_flow_heatmap_new() {
        let layers = vec!["Layer0".to_string(), "Layer1".to_string()];
        let cols = vec!["Q".to_string(), "K".to_string(), "V".to_string()];
        let heatmap = GradientFlowHeatmap::new(layers, cols);

        assert_eq!(heatmap.layer_names.len(), 2);
        assert_eq!(heatmap.column_labels.len(), 3);
    }

    #[test]
    fn test_gradient_flow_heatmap_update() {
        let layers = vec!["L0".to_string()];
        let cols = vec!["Q".to_string(), "K".to_string()];
        let mut heatmap = GradientFlowHeatmap::new(layers, cols);

        heatmap.update(0, 0, 1.0);
        heatmap.update(0, 1, 2.0);

        assert!(heatmap.gradients[0][0] < heatmap.gradients[0][1]);
    }

    #[test]
    fn test_gradient_flow_heatmap_render() {
        let layers = vec!["L0".to_string(), "L1".to_string()];
        let cols = vec!["Q".to_string(), "K".to_string()];
        let mut heatmap = GradientFlowHeatmap::new(layers, cols);

        heatmap.update(0, 0, 0.1);
        heatmap.update(0, 1, 1.0);
        heatmap.update(1, 0, 0.5);
        heatmap.update(1, 1, 0.5);

        let output = heatmap.render();
        assert!(output.contains("Gradient Flow"));
        assert!(output.contains("L0"));
    }

    // =========================================================================
    // Reference Curve Tests (ENT-067)
    // =========================================================================

    #[test]
    fn test_reference_curve_new() {
        let curve = ReferenceCurve::new(vec![1.0, 0.8, 0.6, 0.4], 0.1);
        assert_eq!(curve.get(0), Some(1.0));
        assert_eq!(curve.get(3), Some(0.4));
        assert_eq!(curve.get(10), None);
    }

    #[test]
    fn test_reference_curve_deviation() {
        let curve = ReferenceCurve::new(vec![1.0, 0.8, 0.6], 0.1);

        // Within tolerance
        assert!(curve.check_deviation(0, 1.05).is_none());

        // Outside tolerance
        assert!(curve.check_deviation(0, 1.5).is_some());
    }

    #[test]
    fn test_reference_curve_from_json() {
        let json = "[1.0, 0.8, 0.6]";
        let curve = ReferenceCurve::from_json(json).unwrap();
        assert_eq!(curve.values.len(), 3);
    }

    #[test]
    fn test_reference_curve_comparison_sparkline() {
        let reference = ReferenceCurve::new(vec![1.0, 0.8, 0.6, 0.4], 0.1);
        let current = vec![1.0, 0.85, 0.55, 0.35];

        let spark = reference.comparison_sparkline(&current, 4);
        assert_eq!(spark.chars().count(), 4);
    }

    // =========================================================================
    // Monitor Config Tests (ENT-062)
    // =========================================================================

    #[test]
    fn test_monitor_config_default() {
        let config = MonitorConfig::default();
        assert!(config.enabled);
        assert_eq!(config.layout, "compact");
        assert_eq!(config.refresh_ms, 100);
    }

    #[test]
    fn test_monitor_config_to_callback() {
        let config = MonitorConfig {
            layout: "full".to_string(),
            terminal_mode: "ascii".to_string(),
            sparkline_width: 15,
            ..Default::default()
        };

        let callback = config.to_callback();
        assert_eq!(callback.layout, DashboardLayout::Full);
        assert_eq!(callback.mode, TerminalMode::Ascii);
        assert_eq!(callback.sparkline_width, 15);
    }

    #[test]
    fn test_monitor_config_yaml_roundtrip() {
        let config = MonitorConfig::default();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let parsed: MonitorConfig = serde_yaml::from_str(&yaml).unwrap();

        assert_eq!(config.enabled, parsed.enabled);
        assert_eq!(config.layout, parsed.layout);
    }

    // =========================================================================
    // LossCurveDisplay Tests (ENT-056)
    // =========================================================================

    #[test]
    fn test_loss_curve_display_new() {
        let display = LossCurveDisplay::new(80, 40);
        assert_eq!(display.epochs(), 0);
    }

    #[test]
    fn test_loss_curve_display_push_train_loss() {
        let mut display = LossCurveDisplay::new(80, 40);
        display.push_train_loss(1.0);
        display.push_train_loss(0.8);
        display.push_train_loss(0.6);

        // Train series should have 3 epochs
        assert_eq!(display.epochs(), 3);
    }

    #[test]
    fn test_loss_curve_display_push_losses() {
        let mut display = LossCurveDisplay::new(80, 40);
        display.push_losses(1.0, 1.2);
        display.push_losses(0.8, 1.0);

        assert_eq!(display.epochs(), 2);
    }

    #[test]
    fn test_loss_curve_display_summary() {
        let mut display = LossCurveDisplay::new(80, 40);
        display.push_losses(1.0, 1.2);
        display.push_losses(0.5, 0.8);
        display.push_losses(0.3, 0.6);

        let summary = display.summary();
        assert_eq!(summary.len(), 2);
        // First series is Train
        assert_eq!(summary[0].0, "Train");
        assert_eq!(summary[0].1, Some(0.3)); // min
    }

    #[test]
    fn test_loss_curve_display_render_waiting() {
        let display = LossCurveDisplay::new(80, 40);
        let output = display.render_terminal();
        assert_eq!(output, "(waiting for data...)");
    }

    #[test]
    fn test_loss_curve_display_render_with_data() {
        let mut display = LossCurveDisplay::new(40, 20);
        for i in 0..10 {
            let t = i as f32 / 10.0;
            display.push_losses(1.0 - t * 0.5, 1.2 - t * 0.4);
        }

        let output = display.render_terminal();
        // Should produce non-empty output
        assert!(!output.is_empty());
        assert_ne!(output, "(waiting for data...)");
    }

    #[test]
    fn test_loss_curve_display_terminal_mode() {
        let display = LossCurveDisplay::new(40, 20).terminal_mode(TerminalMode::Ascii);
        assert_eq!(display.terminal_mode, TerminalMode::Ascii);
    }

    #[test]
    fn test_loss_curve_display_smoothing() {
        let mut display = LossCurveDisplay::new(40, 20).smoothing(0.9);
        display.push_losses(1.0, 1.2);
        display.push_losses(0.0, 0.0);

        // Should still work with smoothing
        assert_eq!(display.epochs(), 2);
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_andon_builder_methods() {
        let andon = AndonSystem::new()
            .with_sigma_threshold(5.0)
            .with_stall_threshold(500)
            .with_stop_on_critical(false);

        assert!(!andon.stop_on_critical);
        assert_eq!(andon.sigma_threshold, 5.0);
        assert_eq!(andon.stall_threshold, 500);
    }

    #[test]
    fn test_andon_clear_alerts() {
        let mut andon = AndonSystem::new();
        andon.warning("test warning");
        andon.info("test info");
        assert!(!andon.recent_alerts(10).is_empty());

        andon.clear_alerts();
        assert!(andon.recent_alerts(10).is_empty());
    }

    #[test]
    fn test_andon_loss_std_single_value() {
        let mut andon = AndonSystem::new();
        andon.check_loss(1.0);
        // With only one value, std should be None
    }

    #[test]
    fn test_andon_divergence_detection() {
        let mut andon = AndonSystem::new().with_sigma_threshold(2.0);

        // Fill history with stable values
        for _ in 0..20 {
            andon.check_loss(1.0);
        }

        // Spike should trigger warning
        andon.check_loss(100.0);
        let alerts = andon.recent_alerts(10);
        assert!(!alerts.is_empty());
    }

    #[test]
    fn test_andon_stall_detection() {
        let mut andon = AndonSystem::new().with_stall_threshold(5);

        // Initial best loss
        andon.check_loss(1.0);

        // No improvement for 5 steps
        for _ in 0..5 {
            andon.check_loss(1.1);
        }

        let alerts = andon.recent_alerts(10);
        assert!(alerts.iter().any(|a| a.message.contains("stall")));
    }

    #[test]
    fn test_refresh_policy_force_refresh() {
        let mut policy = RefreshPolicy::default();
        policy.force_refresh(100);
        assert_eq!(policy.last_step, 100);
    }

    #[test]
    fn test_refresh_policy_max_interval() {
        let mut policy = RefreshPolicy::new(1000, 1, 1000); // 1ms max
        policy.force_refresh(0);
        std::thread::sleep(Duration::from_millis(5));
        // After max interval, should refresh
        assert!(policy.should_refresh(0));
    }

    #[test]
    fn test_terminal_monitor_refresh_interval() {
        let monitor = TerminalMonitorCallback::new().refresh_interval_ms(50);
        assert_eq!(monitor.refresh_policy.min_interval.as_millis(), 50);
    }

    #[test]
    fn test_terminal_monitor_render_full() {
        let mut monitor = TerminalMonitorCallback::new().layout(DashboardLayout::Full);
        monitor.loss_buffer.push(0.5);

        let ctx = CallbackContext {
            epoch: 5,
            max_epochs: 10,
            loss: 0.3,
            lr: 0.001,
            global_step: 50,
            ..Default::default()
        };

        let output = monitor.render(&ctx);
        assert!(output.contains("ENTRENAR"));
    }

    #[test]
    fn test_terminal_monitor_with_val_loss() {
        let mut monitor = TerminalMonitorCallback::new();
        monitor.val_loss_buffer.push(0.6);

        let ctx = CallbackContext {
            epoch: 5,
            max_epochs: 10,
            loss: 0.3,
            val_loss: Some(0.6),
            lr: 0.001,
            ..Default::default()
        };

        let output = monitor.render(&ctx);
        assert!(output.contains("0.3") || output.contains("loss"));
    }

    #[test]
    fn test_terminal_monitor_render_alerts() {
        let mut monitor = TerminalMonitorCallback::new();
        monitor.andon.warning("Test warning");
        monitor.andon.critical("Test critical");

        let output = monitor.render_alerts();
        assert!(output.contains("Test warning") || output.contains("Test critical"));
    }

    #[test]
    fn test_feature_importance_empty_render() {
        let chart = FeatureImportanceChart::new(5, 20);
        let output = chart.render();
        assert!(output.contains("No feature importance"));
    }

    #[test]
    fn test_feature_importance_without_names() {
        let mut chart = FeatureImportanceChart::new(3, 10);
        chart.update(&[(0, 0.8), (1, 0.5)], None);

        let output = chart.render();
        assert!(output.contains("feature_0"));
    }

    #[test]
    fn test_feature_importance_max_score_zero() {
        let mut chart = FeatureImportanceChart::new(3, 10);
        chart.update(&[(0, 0.0), (1, 0.0)], None);
        let output = chart.render();
        assert!(output.contains("feature_"));
    }

    #[test]
    fn test_gradient_flow_bounds_check() {
        let layers = vec!["L0".to_string()];
        let cols = vec!["Q".to_string()];
        let mut heatmap = GradientFlowHeatmap::new(layers, cols);

        // Out of bounds should be ignored
        heatmap.update(10, 10, 1.0);
        heatmap.update(0, 0, 1.0);
    }

    #[test]
    fn test_gradient_flow_constant_gradients() {
        let layers = vec!["L0".to_string(), "L1".to_string()];
        let cols = vec!["Q".to_string()];
        let mut heatmap = GradientFlowHeatmap::new(layers, cols);

        heatmap.update(0, 0, 1.0);
        heatmap.update(1, 0, 1.0);

        let output = heatmap.render();
        assert!(output.contains("L0"));
    }

    #[test]
    fn test_reference_curve_empty_comparison() {
        let curve = ReferenceCurve::new(vec![], 0.1);
        let spark = curve.comparison_sparkline(&[], 10);
        assert!(spark.is_empty());
    }

    #[test]
    fn test_reference_curve_deviation_none() {
        let curve = ReferenceCurve::new(vec![1.0], 0.5);
        // Out of bounds epoch
        assert!(curve.check_deviation(10, 1.0).is_none());
    }

    #[test]
    fn test_monitor_config_layouts() {
        let minimal = MonitorConfig {
            layout: "minimal".to_string(),
            ..Default::default()
        };
        let cb = minimal.to_callback();
        assert_eq!(cb.layout, DashboardLayout::Minimal);

        let unknown = MonitorConfig {
            layout: "unknown".to_string(),
            ..Default::default()
        };
        let cb2 = unknown.to_callback();
        assert_eq!(cb2.layout, DashboardLayout::Compact);
    }

    #[test]
    fn test_monitor_config_terminal_modes() {
        let unicode = MonitorConfig {
            terminal_mode: "unicode".to_string(),
            ..Default::default()
        };
        let cb = unicode.to_callback();
        assert_eq!(cb.mode, TerminalMode::Unicode);
    }

    #[test]
    fn test_kalman_eta_string_formats() {
        let mut kalman = KalmanEta::new();
        kalman.update(1.0);

        // Seconds
        assert!(kalman.eta_string(30).contains('s'));

        // Minutes
        assert!(kalman.eta_string(90).contains('m'));

        // Hours
        assert!(kalman.eta_string(4000).contains('h'));
    }

    #[test]
    fn test_sparkline_width_zero() {
        let result = sparkline(&[1.0, 2.0, 3.0], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sparkline_range_width_zero() {
        let result = sparkline_range(&[1.0, 2.0], 0, 0.0, 2.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_progress_bar_zero_total() {
        let bar = ProgressBar::new(0, 20);
        assert_eq!(bar.percent(), 100.0);
    }

    #[test]
    fn test_alert_level_variants() {
        let info = AlertLevel::Info;
        let warning = AlertLevel::Warning;
        let critical = AlertLevel::Critical;
        assert_ne!(info, warning);
        assert_ne!(warning, critical);
    }

    #[test]
    fn test_terminal_mode_default() {
        let mode = TerminalMode::default();
        assert_eq!(mode, TerminalMode::Unicode);
    }

    #[test]
    fn test_dashboard_layout_default() {
        let layout = DashboardLayout::default();
        assert_eq!(layout, DashboardLayout::Compact);
    }

    #[test]
    fn test_metrics_buffer_empty_stats() {
        let buf = MetricsBuffer::new(10);
        assert!(buf.min().is_none());
        assert!(buf.max().is_none());
        assert!(buf.mean().is_none());
    }

    #[test]
    fn test_loss_curve_display_push_val_loss() {
        let mut display = LossCurveDisplay::new(40, 20);
        display.push_val_loss(0.5);
        display.push_val_loss(0.4);
        // Val series has data
    }

    #[test]
    fn test_loss_curve_display_print() {
        let display = LossCurveDisplay::new(40, 20);
        // Just verify it doesn't panic
        display.print();
    }

    // =========================================================================
    // Additional Edge Case Tests for Coverage
    // =========================================================================

    #[test]
    fn test_sparkline_zero_width() {
        let result = sparkline(&[1.0, 2.0, 3.0], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sparkline_range_zero_width() {
        let result = sparkline_range(&[1.0, 2.0, 3.0], 0, 0.0, 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sparkline_range_constant_min_max() {
        let result = sparkline_range(&[0.5, 0.5], 2, 0.5, 0.5);
        // Should handle zero range
        assert!(result.chars().all(|c| c == SPARK_CHARS[4]));
    }

    #[test]
    fn test_sparkline_range_subsample() {
        let values: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let result = sparkline_range(&values, 10, 0.0, 99.0);
        assert_eq!(result.chars().count(), 10);
    }

    #[test]
    fn test_sparkline_range_clamping() {
        let values = vec![-10.0, 50.0, 110.0];
        let result = sparkline_range(&values, 3, 0.0, 100.0);
        // Values outside range should be clamped
        let chars: Vec<char> = result.chars().collect();
        assert_eq!(chars[0], SPARK_CHARS[0]); // Clamped to min
        assert_eq!(chars[2], SPARK_CHARS[7]); // Clamped to max
    }

    #[test]
    fn test_kalman_eta_multiple_updates() {
        let mut kalman = KalmanEta::new();
        for i in 1..=10 {
            kalman.update(f64::from(i) * 0.1);
        }
        let eta = kalman.eta_seconds(5);
        assert!(eta > 0.0);
    }

    #[test]
    fn test_format_duration_hours() {
        assert_eq!(format_duration(7200.0), "2h 00m");
        assert_eq!(format_duration(3661.0), "1h 01m");
    }

    #[test]
    fn test_progress_bar_100_percent() {
        let mut bar = ProgressBar::new(100, 20);
        bar.update(100);
        assert!((bar.percent() - 100.0).abs() < 0.1);
        let output = bar.render();
        assert!(output.contains("100"));
    }

    #[test]
    fn test_progress_bar_over_100() {
        let mut bar = ProgressBar::new(100, 20);
        bar.update(150);
        // Should handle over 100%
        assert!(bar.percent() > 100.0);
    }

    #[test]
    fn test_metrics_buffer_mean_single() {
        let mut buf = MetricsBuffer::new(10);
        buf.push(5.0);
        assert_eq!(buf.mean(), Some(5.0));
    }

    #[test]
    fn test_metrics_buffer_values_after_wrap() {
        let mut buf = MetricsBuffer::new(3);
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);
        buf.push(4.0);
        // Values should be in chronological order
        assert_eq!(buf.values(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_terminal_mode_eq() {
        assert_eq!(TerminalMode::Unicode, TerminalMode::Unicode);
        assert_ne!(TerminalMode::Ascii, TerminalMode::Unicode);
        assert_ne!(TerminalMode::Ansi, TerminalMode::Ascii);
    }

    #[test]
    fn test_dashboard_layout_eq() {
        assert_eq!(DashboardLayout::Minimal, DashboardLayout::Minimal);
        assert_ne!(DashboardLayout::Compact, DashboardLayout::Full);
    }

    #[test]
    fn test_alert_level_clone() {
        let level = AlertLevel::Critical;
        let cloned = level;
        assert_eq!(level, cloned);
    }

    #[test]
    fn test_alert_clone() {
        let alert = Alert {
            level: AlertLevel::Warning,
            message: "test".to_string(),
            timestamp: Instant::now(),
        };
        let cloned = alert.clone();
        assert_eq!(alert.message, cloned.message);
        assert_eq!(alert.level, cloned.level);
    }

    #[test]
    fn test_progress_bar_total_zero() {
        let bar = ProgressBar::new(0, 20);
        assert_eq!(bar.percent(), 100.0);
    }

    #[test]
    fn test_refresh_policy_min_interval_block() {
        let mut policy = RefreshPolicy::new(1000, 10000, 10);
        policy.force_refresh(0);
        // Within min interval, should not refresh
        assert!(!policy.should_refresh(0));
    }

    #[test]
    fn test_kalman_eta_estimate() {
        let mut kalman = KalmanEta::new();
        kalman.update(1.0); // 1 second per step
                            // With 1 step remaining at 1s per step, ETA should be ~1s
        let eta = kalman.eta_seconds(1);
        assert!(eta > 0.5 && eta < 2.0);
    }

    #[test]
    fn test_kalman_eta_string() {
        let mut kalman = KalmanEta::new();
        kalman.update(1.0);
        let eta_str = kalman.eta_string(5);
        // Should contain time units
        assert!(!eta_str.is_empty());
    }

    #[test]
    fn test_metrics_buffer_empty_stats_edge() {
        let buf = MetricsBuffer::new(10);
        assert!(buf.min().is_none());
        assert!(buf.max().is_none());
        assert!(buf.mean().is_none());
    }

    #[test]
    fn test_terminal_capabilities_narrow() {
        let mut caps = TerminalCapabilities::default();
        caps.width = 40;
        // Still valid even with narrow terminal
        assert!(caps.recommended_mode() != TerminalMode::Ascii || !caps.is_tty);
    }

    #[test]
    fn test_terminal_monitor_name() {
        let monitor = TerminalMonitorCallback::new();
        assert!(!monitor.name().is_empty());
    }

    #[test]
    fn test_andon_neg_infinity() {
        let mut andon = AndonSystem::new();
        let should_stop = andon.check_loss(f32::NEG_INFINITY);
        assert!(should_stop);
        assert!(andon.has_critical());
    }

    #[test]
    fn test_refresh_policy_clone() {
        let policy = RefreshPolicy::default();
        let cloned = policy.clone();
        assert_eq!(policy.step_interval, cloned.step_interval);
    }

    #[test]
    fn test_metrics_buffer_clone() {
        let mut buf = MetricsBuffer::new(5);
        buf.push(1.0);
        buf.push(2.0);
        let cloned = buf.clone();
        assert_eq!(buf.len(), cloned.len());
    }

    #[test]
    fn test_progress_bar_initial_render() {
        let bar = ProgressBar::new(100, 10);
        let output = bar.render();
        assert!(output.contains("0.0%") || output.contains("  0.0%"));
    }

    #[test]
    fn test_andon_system_has_no_critical_initially() {
        let andon = AndonSystem::new();
        assert!(!andon.has_critical());
    }

    #[test]
    fn test_terminal_monitor_starts_with_unicode_mode() {
        let monitor = TerminalMonitorCallback::new();
        // Default mode is Unicode
        assert_eq!(monitor.mode, TerminalMode::Unicode);
    }

    #[test]
    fn test_progress_bar_kalman_clone() {
        let kalman = KalmanEta::new();
        let cloned = kalman.clone();
        // Both should have same initial estimate
        assert!(cloned.eta_seconds(0) >= 0.0);
    }

    #[test]
    fn test_feature_importance_empty_names() {
        let mut chart = FeatureImportanceChart::new(3, 10);
        let importances = vec![(0, 0.5), (1, 0.8)];
        chart.update(&importances, None);
        // Should use default names
        assert_eq!(chart.names.len(), 2);
    }

    #[test]
    fn test_gradient_heatmap_empty_render() {
        let heatmap = GradientFlowHeatmap::new(vec![], vec![]);
        let output = heatmap.render();
        assert!(output.contains("Gradient"));
    }

    #[test]
    fn test_reference_curve_deviation_within() {
        let curve = ReferenceCurve::new(vec![1.0, 1.0, 1.0], 0.5);
        // 1.0 is within tolerance of 1.0
        let deviation = curve.check_deviation(0, 1.0);
        assert!(deviation.is_none());
    }

    #[test]
    fn test_reference_curve_deviation_outside() {
        let curve = ReferenceCurve::new(vec![1.0, 1.0, 1.0], 0.1);
        // 2.0 is outside 10% tolerance of 1.0
        let deviation = curve.check_deviation(0, 2.0);
        assert!(deviation.is_some());
    }

    #[test]
    fn test_loss_curve_display_ansi_mode() {
        let display = LossCurveDisplay::new(40, 20).terminal_mode(TerminalMode::Ansi);
        assert_eq!(display.terminal_mode, TerminalMode::Ansi);
    }

    #[test]
    fn test_terminal_monitor_epoch_callback() {
        let mut monitor = TerminalMonitorCallback::new();
        let ctx = CallbackContext {
            epoch: 0,
            max_epochs: 10,
            loss: 1.0,
            lr: 0.001,
            ..Default::default()
        };
        // Should not panic
        let action = monitor.on_epoch_end(&ctx);
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn test_terminal_monitor_step_callback() {
        let mut monitor = TerminalMonitorCallback::new();
        let ctx = CallbackContext {
            epoch: 0,
            max_epochs: 10,
            loss: 1.0,
            lr: 0.001,
            global_step: 1,
            ..Default::default()
        };
        // Should not panic
        let action = monitor.on_step_end(&ctx);
        assert!(matches!(action, CallbackAction::Continue));
    }
}

// =============================================================================
// Property Tests
// =============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// MetricsBuffer length never exceeds capacity
        #[test]
        fn metrics_buffer_bounded(values in prop::collection::vec(-1000.0f32..1000.0, 0..1000)) {
            let mut buf = MetricsBuffer::new(100);
            for v in &values {
                buf.push(*v);
            }
            prop_assert!(buf.len() <= buf.capacity());
        }

        /// MetricsBuffer values are in chronological order
        #[test]
        fn metrics_buffer_order(values in prop::collection::vec(0.0f32..1000.0, 1..100)) {
            let mut buf = MetricsBuffer::new(values.len());
            for v in &values {
                buf.push(*v);
            }
            prop_assert_eq!(buf.values(), values);
        }

        /// Sparkline length matches input (or width if subsampled)
        #[test]
        fn sparkline_length(
            values in prop::collection::vec(-100.0f32..100.0, 1..100),
            width in 1usize..50
        ) {
            let result = sparkline(&values, width);
            let expected_len = values.len().min(width);
            prop_assert_eq!(result.chars().count(), expected_len);
        }

        /// Sparkline chars are valid
        #[test]
        fn sparkline_valid_chars(values in prop::collection::vec(-100.0f32..100.0, 1..100)) {
            let result = sparkline(&values, values.len());
            for c in result.chars() {
                prop_assert!(SPARK_CHARS.contains(&c));
            }
        }

        /// Progress percentage is bounded [0, 100]
        #[test]
        fn progress_bar_bounded(current in 0usize..1000, total in 1usize..1000) {
            let mut bar = ProgressBar::new(total, 20);
            bar.current = current;
            let pct = bar.percent();
            prop_assert!(pct >= 0.0);
            prop_assert!(pct <= 100.0 || current > total);
        }

        /// Kalman ETA is non-negative
        #[test]
        fn kalman_eta_nonnegative(
            durations in prop::collection::vec(0.001f64..10.0, 1..100),
            remaining in 0usize..1000
        ) {
            let mut kalman = KalmanEta::new();
            for d in durations {
                kalman.update(d);
            }
            prop_assert!(kalman.eta_seconds(remaining) >= 0.0);
        }

        /// Andon doesn't false positive on normal losses
        #[test]
        fn andon_no_false_positive(values in prop::collection::vec(0.0f32..100.0, 1..100)) {
            let mut andon = AndonSystem::new().with_stop_on_critical(false);
            for v in values {
                andon.check_loss(v);
            }
            // Normal losses should not trigger critical
            prop_assert!(!andon.has_critical());
        }
    }
}
