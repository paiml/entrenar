//! Presentar-based Training Dashboard Widget (ALB-047/048/057)
//!
//! Composes sovereign-stack presentar-terminal widgets:
//! - Layout for flex-based arrangement
//! - Border for section panels
//! - Meter for progress bar
//! - GpuPanel for GPU telemetry
//! - Sparkline for loss history
//! - Text for information lines
//!
//! ALB-057: Replaces monolithic draw_text() rendering with composable widget tree.
//! Each dashboard section is a standalone widget composed via Layout::rows().

use super::state::{TrainingSnapshot, TrainingState, TrainingStatus};
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use presentar_terminal::widgets::{
    Border, GpuDevice, GpuPanel as PresentarGpuPanel, GpuProcess as PresentarGpuProcess, GpuVendor,
    Layout, LayoutItem, Meter, Sparkline, Text,
};
use std::any::Any;
use std::path::PathBuf;
use std::time::Duration;

/// Training dashboard widget for `apr monitor`.
///
/// Composes presentar-terminal widgets (Layout, Border, Meter, GpuPanel,
/// Sparkline) via a widget tree rebuilt each frame from `training_state.json`.
/// Delegates layout/paint to the composed tree, getting responsive arrangement
/// and consistent theming from the sovereign stack.
pub struct TrainingDashboard {
    /// Latest snapshot (updated each layout pass)
    snapshot: Option<TrainingSnapshot>,
    /// Experiment directory (contains training_state.json)
    experiment_dir: PathBuf,
    /// Cached bounds from layout
    bounds: Rect,
    /// Composed widget tree — rebuilt each frame from snapshot
    widget_tree: Option<Layout>,
}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for TrainingDashboard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainingDashboard")
            .field("experiment_dir", &self.experiment_dir)
            .field("has_snapshot", &self.snapshot.is_some())
            .field("has_widget_tree", &self.widget_tree.is_some())
            .finish()
    }
}

impl TrainingDashboard {
    /// Create a new training dashboard for an experiment directory.
    #[must_use]
    pub fn new(experiment_dir: PathBuf) -> Self {
        Self { snapshot: None, experiment_dir, bounds: Rect::default(), widget_tree: None }
    }

    /// Refresh the snapshot from training_state.json.
    pub fn refresh(&mut self) {
        let mut state = TrainingState::new(&self.experiment_dir);
        if let Ok(Some(snap)) = state.read() {
            self.snapshot = Some(snap);
        }
    }

    /// Check if training is done (for app quit logic).
    pub fn is_finished(&self) -> bool {
        self.snapshot.as_ref().is_some_and(|s| {
            matches!(s.status, TrainingStatus::Completed | TrainingStatus::Failed(_))
        })
    }

    /// Rebuild composed widget tree from current snapshot.
    ///
    /// Called each frame after refresh(). Constructs a Layout::rows tree
    /// of bordered section panels from the latest training data.
    fn rebuild_widgets(&mut self) {
        let Some(snap) = &self.snapshot else {
            self.widget_tree = None;
            return;
        };

        let mut items = Vec::with_capacity(5);

        // Header: title + status badge + model info (2 lines)
        items.push(build_header(snap).into_item().fixed(2.0));

        // Training metrics panel: progress, loss, timing (4 content + 2 border)
        items.push(LayoutItem::new(build_metrics_panel(snap)).fixed(6.0));

        // Shows device, utilization, VRAM, temp/power (3 content + 2 border)
        if snap.gpu.is_some() {
            items.push(LayoutItem::new(build_gpu_panel(snap)).fixed(5.0));
        }

        // Loss history sparkline (2 content + 2 border)
        if !snap.loss_history.is_empty() {
            items.push(LayoutItem::new(build_loss_panel(snap)).fixed(4.0));
        }

        // Error message if failed
        if let TrainingStatus::Failed(msg) = &snap.status {
            let err = Text::new(format!("ERROR: {msg}")).with_color(Color::new(1.0, 0.3, 0.3, 1.0));
            items.push(LayoutItem::new(err).fixed(1.0));
        }

        self.widget_tree = Some(Layout::rows(items));
    }
}

// =============================================================================
// Widget builders — each returns a composed widget for one dashboard section
// =============================================================================

/// Build header section: title + status badge + model info.
fn build_header(snap: &TrainingSnapshot) -> Layout {
    let (status_str, status_color) = status_display(&snap.status);

    let title = Text::new(format!("apr monitor — {}", truncate_str(&snap.experiment_id, 30)))
        .with_color(Color::WHITE)
        .bold();

    let status = Text::new(status_str).with_color(status_color).right();

    let title_row =
        Layout::columns([LayoutItem::new(title).expanded(), LayoutItem::new(status).fixed(10.0)]);

    let info = Text::new(format!(
        "Model: {} | Opt: {} | Batch: {}",
        truncate_str(&snap.model_name, 30),
        snap.optimizer_name,
        snap.batch_size
    ))
    .with_color(Color::new(0.6, 0.6, 0.6, 1.0));

    Layout::rows([title_row.into_item().fixed(1.0), LayoutItem::new(info).fixed(1.0)])
}

/// Build metrics panel: progress bar, loss, timing — inside a rounded border.
fn build_metrics_panel(snap: &TrainingSnapshot) -> Border {
    let progress = snap.progress_percent();
    let trend = snap.loss_trend();

    let progress_text = Text::new(format!(
        "Epoch {}/{}  Step {}/{}",
        snap.epoch, snap.total_epochs, snap.step, snap.steps_per_epoch
    ))
    .with_color(Color::WHITE);

    let meter = Meter::percentage(f64::from(progress))
        .with_label(format!("{progress:.1}%"))
        .with_color(Color::new(0.3, 0.9, 0.3, 1.0));

    let loss_text = Text::new(format!(
        "Loss: {:.6} {}  LR: {:.2e}  Tok/s: {:.0}  Grad: {:.2}",
        snap.loss,
        trend.arrow(),
        snap.learning_rate,
        snap.tokens_per_second,
        snap.gradient_norm
    ))
    .with_color(Color::WHITE);

    let elapsed = snap.elapsed();
    let eta = snap
        .estimated_remaining()
        .map(|r| format!("ETA: {}", format_duration(r)))
        .unwrap_or_default();
    let time_text = Text::new(format!("Elapsed: {}  {}", format_duration(elapsed), eta))
        .with_color(Color::new(1.0, 0.9, 0.3, 1.0));

    let content = Layout::rows([
        LayoutItem::new(progress_text).fixed(1.0),
        LayoutItem::new(meter).fixed(1.0),
        LayoutItem::new(loss_text).fixed(1.0),
        LayoutItem::new(time_text).fixed(1.0),
    ]);

    Border::rounded("Training").child(content)
}

/// Build GPU panel: utilization, VRAM, temp/power — inside a rounded border.
fn build_gpu_panel(snap: &TrainingSnapshot) -> Border {
    let Some(gpu) = &snap.gpu else {
        return Border::rounded("GPU")
            .child(Text::new("N/A (CPU training)").with_color(Color::new(0.5, 0.5, 0.5, 1.0)));
    };

    let device = convert_gpu_device(gpu);
    let processes = convert_gpu_processes(&gpu.processes);

    let gpu_widget = PresentarGpuPanel::new()
        .with_device(device)
        .with_processes(processes)
        .show_processes(false);

    Border::rounded(format!("GPU: {}", truncate_str(&gpu.device_name, 20))).child(gpu_widget)
}

/// Build loss history panel: sparkline + range — inside a rounded border.
fn build_loss_panel(snap: &TrainingSnapshot) -> Border {
    let values: Vec<f64> = snap.loss_history.iter().map(|v| f64::from(*v)).collect();
    let first = snap.loss_history.first().copied().unwrap_or(0.0);
    let last = snap.loss_history.last().copied().unwrap_or(0.0);

    let sparkline =
        Sparkline::new(values).with_color(Color::new(0.3, 0.7, 1.0, 1.0)).with_trend(true);

    let range =
        Text::new(format!("{first:.4} → {last:.4}")).with_color(Color::new(0.5, 0.5, 0.5, 1.0));

    let content =
        Layout::rows([LayoutItem::new(sparkline).fixed(1.0), LayoutItem::new(range).fixed(1.0)]);

    Border::rounded("Loss History").child(content)
}

// =============================================================================
// Helpers
// =============================================================================

/// Convert entrenar `GpuTelemetry` to presentar `GpuDevice`.
fn convert_gpu_device(gpu: &super::state::GpuTelemetry) -> GpuDevice {
    GpuDevice::new(&gpu.device_name)
        .with_vendor(GpuVendor::Nvidia)
        .with_utilization(gpu.utilization_percent)
        .with_temperature(gpu.temperature_celsius)
        .with_vram(
            (gpu.vram_used_gb.max(0.0) * 1_073_741_824.0) as u64,
            (gpu.vram_total_gb.max(0.0) * 1_073_741_824.0) as u64,
        )
        .with_power(gpu.power_watts, Some(gpu.power_limit_watts))
}

/// Convert entrenar `GpuProcessInfo` to presentar `GpuProcess`.
fn convert_gpu_processes(processes: &[super::state::GpuProcessInfo]) -> Vec<PresentarGpuProcess> {
    processes
        .iter()
        .map(|p| {
            let name = p.exe_path.rsplit('/').next().unwrap_or(&p.exe_path);
            PresentarGpuProcess::new(name, p.pid, p.gpu_memory_mb * 1024 * 1024)
        })
        .collect()
}

/// Get display string and color for training status.
fn status_display(status: &TrainingStatus) -> (&'static str, Color) {
    match status {
        TrainingStatus::Initializing => ("INIT", Color::new(0.7, 0.7, 0.7, 1.0)),
        TrainingStatus::Running => ("RUNNING", Color::new(0.3, 0.9, 0.3, 1.0)),
        TrainingStatus::Paused => ("PAUSED", Color::new(0.7, 0.7, 0.7, 1.0)),
        TrainingStatus::Completed => ("DONE", Color::new(0.3, 0.7, 1.0, 1.0)),
        TrainingStatus::Failed(_) => ("FAILED", Color::new(1.0, 0.3, 0.3, 1.0)),
    }
}

/// Truncate a string with no-alloc slicing.
fn truncate_str(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        &s[..max]
    }
}

/// Format a Duration as human-readable.
fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    if secs > 3600 {
        format!("{}h {}m {}s", secs / 3600, (secs % 3600) / 60, secs % 60)
    } else if secs > 60 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{secs}s")
    }
}

// =============================================================================
// Brick trait implementation (PROBAR-SPEC-009)
// =============================================================================

impl Brick for TrainingDashboard {
    fn brick_name(&self) -> &'static str {
        "training_dashboard"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16) // 60fps
    }

    fn verify(&self) -> BrickVerification {
        // Wire panel verification system into Brick verify (ALB-057).
        // Uses panel::layout_can_render() to validate snapshot data before rendering.
        let start = std::time::Instant::now();
        if let Some(snap) = &self.snapshot {
            if !super::panel::layout_can_render(snap) {
                return BrickVerification {
                    passed: vec![],
                    failed: vec![(
                        BrickAssertion::max_latency_ms(16),
                        "snapshot data failed panel verification".to_string(),
                    )],
                    verification_time: start.elapsed(),
                };
            }
        }
        BrickVerification {
            passed: vec![BrickAssertion::max_latency_ms(16)],
            failed: vec![],
            verification_time: start.elapsed(),
        }
    }

    fn to_html(&self) -> String {
        String::new() // Terminal-only widget
    }

    fn to_css(&self) -> String {
        String::new() // Terminal-only widget
    }
}

// =============================================================================
// Widget trait implementation
// =============================================================================

impl Widget for TrainingDashboard {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        Size { width: constraints.max_width, height: constraints.max_height }
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        // Refresh data and rebuild widget tree each frame
        self.refresh();
        self.rebuild_widgets();

        // Delegate layout to composed widget tree
        if let Some(tree) = &mut self.widget_tree {
            tree.layout(bounds);
        }

        LayoutResult { size: Size { width: bounds.width, height: bounds.height } }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        // Delegate to composed widget tree
        if let Some(tree) = &self.widget_tree {
            tree.paint(canvas);
            return;
        }

        // Fallback: waiting state
        let dim = TextStyle { color: Color::new(0.5, 0.5, 0.5, 1.0), ..Default::default() };
        canvas.draw_text("Waiting for training data...", Point { x: 1.0, y: 1.0 }, &dim);
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        if let Some(tree) = &mut self.widget_tree {
            return tree.event(event);
        }
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }
}
