//! WebAssembly bindings for training monitor.
//!
//! Provides JavaScript-accessible functions for real-time training
//! visualization in the browser.
//!
//! # Usage (JavaScript)
//!
//! ```javascript
//! import init, { WasmMetricsCollector, WasmDashboard } from 'entrenar-monitor';
//!
//! await init();
//!
//! const collector = new WasmMetricsCollector();
//! collector.record_loss(0.5);
//! collector.record_accuracy(0.85);
//!
//! const stats = collector.summary_json();
//! console.log(JSON.parse(stats));
//!
//! const dashboard = new WasmDashboard(800, 400);
//! dashboard.update(collector);
//! const pngData = dashboard.render_png();
//! ```

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use super::{Metric, MetricStats, MetricsCollector};
use std::collections::HashMap;

// ============================================================================
// WASM Metrics Collector
// ============================================================================

/// WASM-compatible metrics collector.
///
/// Wraps MetricsCollector with JavaScript-friendly API.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
#[derive(Debug)]
pub struct WasmMetricsCollector {
    inner: MetricsCollector,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl WasmMetricsCollector {
    /// Create a new metrics collector.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        Self {
            inner: MetricsCollector::new(),
        }
    }

    /// Record a loss value.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn record_loss(&mut self, value: f64) {
        self.inner.record(Metric::Loss, value);
    }

    /// Record an accuracy value.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn record_accuracy(&mut self, value: f64) {
        self.inner.record(Metric::Accuracy, value);
    }

    /// Record a learning rate value.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn record_learning_rate(&mut self, value: f64) {
        self.inner.record(Metric::LearningRate, value);
    }

    /// Record a gradient norm value.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn record_gradient_norm(&mut self, value: f64) {
        self.inner.record(Metric::GradientNorm, value);
    }

    /// Record a custom metric.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn record_custom(&mut self, name: &str, value: f64) {
        self.inner.record(Metric::Custom(name.to_string()), value);
    }

    /// Get number of recorded metrics.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn count(&self) -> usize {
        self.inner.count()
    }

    /// Check if collector is empty.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clear all recorded metrics.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Get summary as JSON string.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn summary_json(&self) -> String {
        let summary = self.inner.summary();
        let json_map: HashMap<String, WasmMetricStats> = summary
            .into_iter()
            .map(|(k, v)| (k.as_str().to_string(), WasmMetricStats::from(v)))
            .collect();
        serde_json::to_string(&json_map).unwrap_or_else(|_| "{}".to_string())
    }

    /// Get loss statistics.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn loss_mean(&self) -> f64 {
        self.inner
            .summary()
            .get(&Metric::Loss)
            .map(|s| s.mean)
            .unwrap_or(f64::NAN)
    }

    /// Get accuracy statistics.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn accuracy_mean(&self) -> f64 {
        self.inner
            .summary()
            .get(&Metric::Accuracy)
            .map(|s| s.mean)
            .unwrap_or(f64::NAN)
    }

    /// Get loss values as a Vec for JavaScript Float64Array conversion.
    pub fn loss_values(&self) -> Vec<f64> {
        self.inner
            .to_records()
            .iter()
            .filter(|r| r.metric == Metric::Loss)
            .map(|r| r.value)
            .collect()
    }

    /// Get accuracy values as a Vec for JavaScript Float64Array conversion.
    pub fn accuracy_values(&self) -> Vec<f64> {
        self.inner
            .to_records()
            .iter()
            .filter(|r| r.metric == Metric::Accuracy)
            .map(|r| r.value)
            .collect()
    }

    /// Get all timestamps as milliseconds since epoch.
    pub fn timestamps(&self) -> Vec<u64> {
        self.inner.to_records().iter().map(|r| r.timestamp).collect()
    }

    /// Get loss standard deviation.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn loss_std(&self) -> f64 {
        self.inner
            .summary()
            .get(&Metric::Loss)
            .map(|s| s.std)
            .unwrap_or(f64::NAN)
    }

    /// Get accuracy standard deviation.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn accuracy_std(&self) -> f64 {
        self.inner
            .summary()
            .get(&Metric::Accuracy)
            .map(|s| s.std)
            .unwrap_or(f64::NAN)
    }

    /// Check if NaN was detected in loss.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn loss_has_nan(&self) -> bool {
        self.inner
            .summary()
            .get(&Metric::Loss)
            .map(|s| s.has_nan)
            .unwrap_or(false)
    }

    /// Check if Inf was detected in loss.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn loss_has_inf(&self) -> bool {
        self.inner
            .summary()
            .get(&Metric::Loss)
            .map(|s| s.has_inf)
            .unwrap_or(false)
    }
}

impl Default for WasmMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// WASM Metric Stats (JSON-serializable)
// ============================================================================

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct WasmMetricStats {
    count: usize,
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
    has_nan: bool,
    has_inf: bool,
}

impl From<MetricStats> for WasmMetricStats {
    fn from(s: MetricStats) -> Self {
        Self {
            count: s.count,
            mean: s.mean,
            std: s.std,
            min: s.min,
            max: s.max,
            has_nan: s.has_nan,
            has_inf: s.has_inf,
        }
    }
}

// ============================================================================
// WASM Dashboard Options
// ============================================================================

/// Dashboard rendering options.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
#[derive(Debug, Clone)]
pub struct WasmDashboardOptions {
    width: u32,
    height: u32,
    background_color: String,
    loss_color: String,
    accuracy_color: String,
    show_sparklines: bool,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl WasmDashboardOptions {
    /// Create default dashboard options.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        Self {
            width: 800,
            height: 400,
            background_color: "#1a1a2e".to_string(),
            loss_color: "#ff6b6b".to_string(),
            accuracy_color: "#4ecdc4".to_string(),
            show_sparklines: true,
        }
    }

    /// Set width in pixels.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn width(mut self, width: u32) -> Self {
        self.width = width;
        self
    }

    /// Set height in pixels.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn height(mut self, height: u32) -> Self {
        self.height = height;
        self
    }

    /// Set background color (hex format).
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn background_color(mut self, color: &str) -> Self {
        self.background_color = color.to_string();
        self
    }

    /// Set loss color (hex format).
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn loss_color(mut self, color: &str) -> Self {
        self.loss_color = color.to_string();
        self
    }

    /// Set accuracy color (hex format).
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn accuracy_color(mut self, color: &str) -> Self {
        self.accuracy_color = color.to_string();
        self
    }

    /// Enable/disable sparklines.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn show_sparklines(mut self, show: bool) -> Self {
        self.show_sparklines = show;
        self
    }

    /// Get width.
    pub fn get_width(&self) -> u32 {
        self.width
    }

    /// Get height.
    pub fn get_height(&self) -> u32 {
        self.height
    }
}

impl Default for WasmDashboardOptions {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_collector_new() {
        let collector = WasmMetricsCollector::new();
        assert!(collector.is_empty());
        assert_eq!(collector.count(), 0);
    }

    #[test]
    fn test_wasm_collector_record_loss() {
        let mut collector = WasmMetricsCollector::new();
        collector.record_loss(0.5);
        collector.record_loss(0.3);
        assert_eq!(collector.count(), 2);
        assert!((collector.loss_mean() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_wasm_collector_record_accuracy() {
        let mut collector = WasmMetricsCollector::new();
        collector.record_accuracy(0.8);
        collector.record_accuracy(0.9);
        assert_eq!(collector.count(), 2);
        assert!((collector.accuracy_mean() - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_wasm_collector_record_custom() {
        let mut collector = WasmMetricsCollector::new();
        collector.record_custom("perplexity", 15.5);
        assert_eq!(collector.count(), 1);
    }

    #[test]
    fn test_wasm_collector_clear() {
        let mut collector = WasmMetricsCollector::new();
        collector.record_loss(0.5);
        collector.record_accuracy(0.8);
        assert_eq!(collector.count(), 2);
        collector.clear();
        assert!(collector.is_empty());
    }

    #[test]
    fn test_wasm_collector_summary_json() {
        let mut collector = WasmMetricsCollector::new();
        collector.record_loss(0.5);
        collector.record_loss(0.3);

        let json = collector.summary_json();
        assert!(json.contains("loss"));
        assert!(json.contains("mean"));

        // Parse and validate
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.get("loss").is_some());
    }

    #[test]
    fn test_wasm_collector_missing_metric() {
        let collector = WasmMetricsCollector::new();
        assert!(collector.loss_mean().is_nan());
        assert!(collector.accuracy_mean().is_nan());
    }

    #[test]
    fn test_wasm_dashboard_options_default() {
        let opts = WasmDashboardOptions::new();
        assert_eq!(opts.width, 800);
        assert_eq!(opts.height, 400);
        assert_eq!(opts.background_color, "#1a1a2e");
    }

    #[test]
    fn test_wasm_dashboard_options_builder() {
        let opts = WasmDashboardOptions::new()
            .width(1024)
            .height(768)
            .background_color("#ffffff")
            .loss_color("#ff0000")
            .accuracy_color("#00ff00")
            .show_sparklines(false);

        assert_eq!(opts.width, 1024);
        assert_eq!(opts.height, 768);
        assert_eq!(opts.background_color, "#ffffff");
        assert_eq!(opts.loss_color, "#ff0000");
        assert_eq!(opts.accuracy_color, "#00ff00");
        assert!(!opts.show_sparklines);
    }

    #[test]
    fn test_wasm_metric_stats_from() {
        let stats = MetricStats {
            count: 10,
            mean: 0.5,
            std: 0.1,
            min: 0.2,
            max: 0.8,
            sum: 5.0,
            has_nan: false,
            has_inf: false,
        };

        let wasm_stats = WasmMetricStats::from(stats);
        assert_eq!(wasm_stats.count, 10);
        assert!((wasm_stats.mean - 0.5).abs() < 1e-6);
        assert!((wasm_stats.std - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_wasm_collector_all_metric_types() {
        let mut collector = WasmMetricsCollector::new();
        collector.record_loss(0.5);
        collector.record_accuracy(0.8);
        collector.record_learning_rate(0.001);
        collector.record_gradient_norm(1.5);
        collector.record_custom("perplexity", 15.5);

        assert_eq!(collector.count(), 5);

        let json = collector.summary_json();
        assert!(json.contains("loss"));
        assert!(json.contains("accuracy"));
        assert!(json.contains("learning_rate"));
        assert!(json.contains("gradient_norm"));
        assert!(json.contains("perplexity"));
    }

    #[test]
    fn test_wasm_collector_default() {
        let collector = WasmMetricsCollector::default();
        assert!(collector.is_empty());
    }

    #[test]
    fn test_wasm_dashboard_options_default_trait() {
        let opts = WasmDashboardOptions::default();
        assert_eq!(opts.get_width(), 800);
        assert_eq!(opts.get_height(), 400);
    }

    #[test]
    fn test_wasm_collector_loss_values() {
        let mut collector = WasmMetricsCollector::new();
        collector.record_loss(0.5);
        collector.record_loss(0.3);
        collector.record_accuracy(0.8); // Should not be in loss values

        let values = collector.loss_values();
        assert_eq!(values.len(), 2);
        assert!((values[0] - 0.5).abs() < 1e-6);
        assert!((values[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_wasm_collector_accuracy_values() {
        let mut collector = WasmMetricsCollector::new();
        collector.record_accuracy(0.8);
        collector.record_accuracy(0.9);
        collector.record_loss(0.5); // Should not be in accuracy values

        let values = collector.accuracy_values();
        assert_eq!(values.len(), 2);
        assert!((values[0] - 0.8).abs() < 1e-6);
        assert!((values[1] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_wasm_collector_timestamps() {
        let mut collector = WasmMetricsCollector::new();
        collector.record_loss(0.5);
        collector.record_loss(0.3);

        let timestamps = collector.timestamps();
        assert_eq!(timestamps.len(), 2);
        assert!(timestamps[0] > 0);
        assert!(timestamps[1] >= timestamps[0]);
    }

    #[test]
    fn test_wasm_collector_loss_std() {
        let mut collector = WasmMetricsCollector::new();
        collector.record_loss(0.2);
        collector.record_loss(0.4);
        collector.record_loss(0.6);
        collector.record_loss(0.8);

        let std = collector.loss_std();
        assert!(std > 0.0);
        assert!(std < 1.0);
    }

    #[test]
    fn test_wasm_collector_nan_detection() {
        let mut collector = WasmMetricsCollector::new();
        collector.record_loss(0.5);
        assert!(!collector.loss_has_nan());

        collector.record_loss(f64::NAN);
        assert!(collector.loss_has_nan());
    }

    #[test]
    fn test_wasm_collector_inf_detection() {
        let mut collector = WasmMetricsCollector::new();
        collector.record_loss(0.5);
        assert!(!collector.loss_has_inf());

        collector.record_loss(f64::INFINITY);
        assert!(collector.loss_has_inf());
    }

    #[test]
    fn test_wasm_collector_empty_std() {
        let collector = WasmMetricsCollector::new();
        assert!(collector.loss_std().is_nan());
        assert!(collector.accuracy_std().is_nan());
    }
}
