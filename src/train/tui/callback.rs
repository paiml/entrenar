//! Terminal Monitor Callback (ENT-054)
//!
//! Real-time terminal monitoring callback for training loop integration.

use std::io::Write;
use std::time::{Duration, Instant};

use crate::train::callback::{CallbackAction, CallbackContext, TrainerCallback};

use super::andon::{AlertLevel, AndonSystem};
use super::buffer::MetricsBuffer;
use super::capability::{DashboardLayout, TerminalMode};
use super::progress::{format_duration, ProgressBar};
use super::refresh::RefreshPolicy;
use super::sparkline::sparkline;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_context() -> CallbackContext {
        CallbackContext {
            epoch: 0,
            max_epochs: 10,
            step: 0,
            steps_per_epoch: 100,
            global_step: 0,
            loss: 1.0,
            lr: 0.001,
            best_loss: None,
            val_loss: None,
            elapsed_secs: 0.0,
        }
    }

    #[test]
    fn test_terminal_monitor_callback_new() {
        let callback = TerminalMonitorCallback::new();
        assert_eq!(callback.sparkline_width, 20);
        assert_eq!(callback.model_name, "model");
    }

    #[test]
    fn test_terminal_monitor_callback_builders() {
        let callback = TerminalMonitorCallback::new()
            .mode(TerminalMode::Ascii)
            .layout(DashboardLayout::Full)
            .model_name("test_model")
            .sparkline_width(30)
            .refresh_interval_ms(200);

        assert_eq!(callback.mode, TerminalMode::Ascii);
        assert_eq!(callback.layout, DashboardLayout::Full);
        assert_eq!(callback.model_name, "test_model");
        assert_eq!(callback.sparkline_width, 30);
    }

    #[test]
    fn test_terminal_monitor_callback_render_minimal() {
        let callback = TerminalMonitorCallback::new().layout(DashboardLayout::Minimal);
        let ctx = make_test_context();
        let output = callback.render(&ctx);
        assert!(output.contains("Epoch"));
        assert!(output.contains("loss="));
    }

    #[test]
    fn test_terminal_monitor_callback_render_compact() {
        let callback = TerminalMonitorCallback::new().layout(DashboardLayout::Compact);
        let ctx = make_test_context();
        let output = callback.render(&ctx);
        assert!(output.contains("Training"));
    }

    #[test]
    fn test_terminal_monitor_callback_render_full() {
        let callback = TerminalMonitorCallback::new().layout(DashboardLayout::Full);
        let ctx = make_test_context();
        let output = callback.render(&ctx);
        assert!(output.contains("ENTRENAR"));
    }

    #[test]
    fn test_terminal_monitor_callback_name() {
        let callback = TerminalMonitorCallback::new();
        assert_eq!(callback.name(), "TerminalMonitorCallback");
    }

    #[test]
    fn test_terminal_monitor_callback_on_step_end() {
        let mut callback = TerminalMonitorCallback::new();
        let ctx = make_test_context();

        // Initialize
        callback.start_time = Instant::now();
        callback.progress = ProgressBar::new(1000, 30);

        let action = callback.on_step_end(&ctx);
        assert_eq!(action, CallbackAction::Continue);
        assert_eq!(callback.loss_buffer.len(), 1);
    }

    #[test]
    fn test_terminal_monitor_callback_on_step_end_nan() {
        let mut callback = TerminalMonitorCallback::new();
        let mut ctx = make_test_context();
        ctx.loss = f32::NAN;

        callback.start_time = Instant::now();
        callback.progress = ProgressBar::new(1000, 30);

        let action = callback.on_step_end(&ctx);
        assert_eq!(action, CallbackAction::Stop);
    }

    #[test]
    fn test_terminal_monitor_callback_with_val_loss() {
        let callback = TerminalMonitorCallback::new().layout(DashboardLayout::Compact);
        let mut ctx = make_test_context();
        ctx.val_loss = Some(0.8);
        let output = callback.render(&ctx);
        assert!(output.contains("val="));
    }

    #[test]
    fn test_terminal_monitor_callback_default() {
        let callback = TerminalMonitorCallback::default();
        assert_eq!(callback.sparkline_width, 20);
    }

    #[test]
    fn test_terminal_monitor_callback_on_train_begin() {
        let mut callback = TerminalMonitorCallback::new();
        let ctx = make_test_context();
        let action = callback.on_train_begin(&ctx);
        assert_eq!(action, CallbackAction::Continue);
    }

    #[test]
    fn test_terminal_monitor_callback_on_epoch_end() {
        let mut callback = TerminalMonitorCallback::new();
        let ctx = make_test_context();
        callback.on_train_begin(&ctx);
        let action = callback.on_epoch_end(&ctx);
        assert_eq!(action, CallbackAction::Continue);
    }

    #[test]
    fn test_terminal_monitor_callback_on_train_end() {
        let mut callback = TerminalMonitorCallback::new();
        let ctx = make_test_context();
        callback.on_train_begin(&ctx);
        callback.loss_buffer.push(0.5);
        callback.on_train_end(&ctx);
        // No assertion - just verify it doesn't panic
    }

    #[test]
    fn test_terminal_monitor_callback_render_full_with_val() {
        let mut callback = TerminalMonitorCallback::new().layout(DashboardLayout::Full);
        // Push some validation losses
        callback.val_loss_buffer.push(0.9);
        callback.val_loss_buffer.push(0.8);
        let ctx = make_test_context();
        let output = callback.render(&ctx);
        assert!(output.contains("ENTRENAR"));
    }

    #[test]
    fn test_terminal_monitor_callback_render_alerts_empty() {
        let callback = TerminalMonitorCallback::new();
        let alerts = callback.render_alerts();
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_terminal_monitor_callback_render_alerts_with_alerts() {
        let mut callback = TerminalMonitorCallback::new();
        callback.andon.warning("Test warning");
        callback.andon.critical("Test critical");
        callback.andon.info("Test info");
        let alerts = callback.render_alerts();
        assert!(!alerts.is_empty());
    }

    #[test]
    fn test_terminal_monitor_callback_step_with_val_loss() {
        let mut callback = TerminalMonitorCallback::new();
        callback.on_train_begin(&make_test_context());

        let mut ctx = make_test_context();
        ctx.val_loss = Some(0.75);

        let action = callback.on_step_end(&ctx);
        assert_eq!(action, CallbackAction::Continue);
        assert_eq!(callback.val_loss_buffer.len(), 1);
    }

    #[test]
    fn test_terminal_monitor_callback_render_compact_with_best() {
        let mut callback = TerminalMonitorCallback::new().layout(DashboardLayout::Compact);
        // Push losses to establish a minimum
        callback.loss_buffer.push(1.0);
        callback.loss_buffer.push(0.5);
        callback.loss_buffer.push(0.8);
        let ctx = make_test_context();
        let output = callback.render(&ctx);
        assert!(output.contains("best="));
    }

    #[test]
    fn test_terminal_monitor_callback_render_full_empty_val() {
        let callback = TerminalMonitorCallback::new().layout(DashboardLayout::Full);
        let ctx = make_test_context();
        let output = callback.render(&ctx);
        // Should render without validation spark since buffer is empty
        assert!(output.contains("ENTRENAR"));
    }

    #[test]
    fn test_terminal_monitor_callback_print_display() {
        let callback = TerminalMonitorCallback::new().layout(DashboardLayout::Minimal);
        let ctx = make_test_context();
        // This will print to stdout, but shouldn't panic
        callback.print_display(&ctx);
    }
}
