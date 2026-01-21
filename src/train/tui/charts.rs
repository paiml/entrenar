//! Chart components for terminal visualization
//!
//! - Feature Importance Display (ENT-064)
//! - Gradient Flow Heatmap (ENT-065)
//! - LossCurve Integration (ENT-056)

use trueno_viz::output::{TerminalEncoder, TerminalMode as TruenoTerminalMode};
use trueno_viz::plots::{LossCurve, MetricSeries};
use trueno_viz::prelude::Rgba;

use super::capability::TerminalMode;

/// Summary of a metric series: (name, min_value, last_smoothed, best_epoch).
pub type SeriesSummaryTuple = (String, Option<f32>, Option<f32>, Option<usize>);

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

#[cfg(test)]
mod tests {
    use super::*;

    // FeatureImportanceChart tests
    #[test]
    fn test_feature_importance_chart_new() {
        let chart = FeatureImportanceChart::new(5, 20);
        assert_eq!(chart.top_k, 5);
        assert_eq!(chart.bar_width, 20);
    }

    #[test]
    fn test_feature_importance_chart_update() {
        let mut chart = FeatureImportanceChart::new(3, 20);
        let importances = vec![(0, 0.5), (1, 0.8), (2, 0.3), (3, 0.9)];
        chart.update(&importances, None);

        assert_eq!(chart.names.len(), 3);
        assert_eq!(chart.scores.len(), 3);
        // Should be sorted by importance: 0.9, 0.8, 0.5
        assert_eq!(chart.scores[0], 0.9);
        assert_eq!(chart.scores[1], 0.8);
        assert_eq!(chart.scores[2], 0.5);
    }

    #[test]
    fn test_feature_importance_chart_with_names() {
        let mut chart = FeatureImportanceChart::new(2, 20);
        let importances = vec![(0, 0.5), (1, 0.8)];
        let names = vec!["feature_a".to_string(), "feature_b".to_string()];
        chart.update(&importances, Some(&names));

        assert!(chart.names.contains(&"feature_b".to_string()));
    }

    #[test]
    fn test_feature_importance_chart_render_empty() {
        let chart = FeatureImportanceChart::new(5, 20);
        assert!(chart.render().contains("No feature importance"));
    }

    #[test]
    fn test_feature_importance_chart_render() {
        let mut chart = FeatureImportanceChart::new(2, 10);
        let importances = vec![(0, 1.0), (1, 0.5)];
        chart.update(&importances, None);

        let rendered = chart.render();
        assert!(rendered.contains("Feature Importance"));
        assert!(rendered.contains("█"));
    }

    // GradientFlowHeatmap tests
    #[test]
    fn test_gradient_flow_heatmap_new() {
        let layers = vec!["layer_0".to_string(), "layer_1".to_string()];
        let cols = vec!["Q".to_string(), "K".to_string(), "V".to_string()];
        let heatmap = GradientFlowHeatmap::new(layers, cols);

        assert_eq!(heatmap.layer_names.len(), 2);
        assert_eq!(heatmap.column_labels.len(), 3);
        assert_eq!(heatmap.gradients.len(), 2);
        assert_eq!(heatmap.gradients[0].len(), 3);
    }

    #[test]
    fn test_gradient_flow_heatmap_update() {
        let layers = vec!["layer_0".to_string()];
        let cols = vec!["Q".to_string(), "K".to_string()];
        let mut heatmap = GradientFlowHeatmap::new(layers, cols);

        heatmap.update(0, 0, 1.0);
        heatmap.update(0, 1, 0.1);

        // Values stored as log scale
        assert!(heatmap.gradients[0][0] > heatmap.gradients[0][1]);
    }

    #[test]
    fn test_gradient_flow_heatmap_update_out_of_bounds() {
        let layers = vec!["layer_0".to_string()];
        let cols = vec!["Q".to_string()];
        let mut heatmap = GradientFlowHeatmap::new(layers, cols);

        // Should not panic
        heatmap.update(10, 10, 1.0);
    }

    #[test]
    fn test_gradient_flow_heatmap_render() {
        let layers = vec!["layer_0".to_string()];
        let cols = vec!["Q".to_string(), "K".to_string()];
        let mut heatmap = GradientFlowHeatmap::new(layers, cols);
        heatmap.update(0, 0, 1.0);
        heatmap.update(0, 1, 0.01);

        let rendered = heatmap.render();
        assert!(rendered.contains("Gradient Flow"));
        assert!(rendered.contains("layer_0"));
    }

    // LossCurveDisplay tests
    #[test]
    fn test_loss_curve_display_new() {
        let display = LossCurveDisplay::new(80, 20);
        assert_eq!(display.epochs(), 0);
    }

    #[test]
    fn test_loss_curve_display_push() {
        let mut display = LossCurveDisplay::new(80, 20);
        display.push_train_loss(1.0);
        display.push_val_loss(1.2);
        assert_eq!(display.epochs(), 1);
    }

    #[test]
    fn test_loss_curve_display_push_losses() {
        let mut display = LossCurveDisplay::new(80, 20);
        display.push_losses(1.0, 1.2);
        assert_eq!(display.epochs(), 1);
    }

    #[test]
    fn test_loss_curve_display_render_waiting() {
        let display = LossCurveDisplay::new(80, 20);
        assert!(display.render_terminal().contains("waiting"));
    }

    #[test]
    fn test_loss_curve_display_terminal_mode() {
        let display = LossCurveDisplay::new(80, 20).terminal_mode(TerminalMode::Ascii);
        assert_eq!(display.terminal_mode, TerminalMode::Ascii);
    }

    #[test]
    fn test_loss_curve_display_smoothing() {
        let display = LossCurveDisplay::new(80, 20).smoothing(0.9);
        // Just verify it doesn't panic
        assert_eq!(display.epochs(), 0);
    }
}
