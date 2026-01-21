//! Prometheus Metrics Export Module (MLOPS-006)
//!
//! Integration with standard observability stacks.
//!
//! # Toyota Way: アンドン (Andon)
//!
//! Visual alerting through Prometheus/Grafana dashboards.
//!
//! # Example
//!
//! ```ignore
//! use entrenar::monitor::prometheus::PrometheusExporter;
//!
//! let exporter = PrometheusExporter::new("my-experiment", "run-1");
//! exporter.record_epoch(1, 0.5, 0.001);
//! let metrics = exporter.export();
//! println!("{}", metrics);
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

/// Metric type for Prometheus
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    /// Gauge: current value that can go up or down
    Gauge,
    /// Counter: monotonically increasing value
    Counter,
    /// Histogram: distribution of values
    Histogram,
}

/// A single metric definition
#[derive(Debug, Clone)]
pub struct MetricDef {
    /// Metric name (must be valid Prometheus name)
    pub name: String,
    /// Help text describing the metric
    pub help: String,
    /// Metric type
    pub metric_type: MetricType,
    /// Label names
    pub labels: Vec<String>,
}

impl MetricDef {
    /// Create a new gauge metric
    pub fn gauge(name: &str, help: &str) -> Self {
        Self {
            name: name.to_string(),
            help: help.to_string(),
            metric_type: MetricType::Gauge,
            labels: Vec::new(),
        }
    }

    /// Create a new counter metric
    pub fn counter(name: &str, help: &str) -> Self {
        Self {
            name: name.to_string(),
            help: help.to_string(),
            metric_type: MetricType::Counter,
            labels: Vec::new(),
        }
    }

    /// Add labels to the metric
    pub fn with_labels(mut self, labels: &[&str]) -> Self {
        self.labels = labels.iter().map(|s| (*s).to_string()).collect();
        self
    }
}

/// Label values for a specific metric series
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LabelSet {
    values: Vec<(String, String)>,
}

impl LabelSet {
    /// Create an empty label set
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }

    /// Create label set from key-value pairs
    pub fn from_pairs(pairs: &[(&str, &str)]) -> Self {
        Self {
            values: pairs
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect(),
        }
    }

    /// Add a label
    pub fn add(mut self, key: &str, value: &str) -> Self {
        self.values.push((key.to_string(), value.to_string()));
        self
    }

    /// Format labels for Prometheus output
    fn format(&self) -> String {
        if self.values.is_empty() {
            return String::new();
        }

        let parts: Vec<String> = self
            .values
            .iter()
            .map(|(k, v)| format!("{}=\"{}\"", k, escape_label_value(v)))
            .collect();

        format!("{{{}}}", parts.join(","))
    }
}

impl Default for LabelSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Escape label values for Prometheus format
fn escape_label_value(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// A metric value with optional labels
#[derive(Debug, Clone)]
struct MetricValue {
    labels: LabelSet,
    value: f64,
    timestamp: Option<u64>,
}

/// Prometheus metrics exporter for training monitoring
#[derive(Debug)]
pub struct PrometheusExporter {
    /// Default labels applied to all metrics
    default_labels: LabelSet,
    /// Metric definitions
    definitions: HashMap<String, MetricDef>,
    /// Current metric values
    values: RwLock<HashMap<String, Vec<MetricValue>>>,
    /// Total samples counter
    total_samples: AtomicU64,
}

impl PrometheusExporter {
    /// Create a new exporter with experiment and run labels
    pub fn new(experiment: &str, run: &str) -> Self {
        let default_labels = LabelSet::from_pairs(&[("experiment", experiment), ("run", run)]);

        let mut exporter = Self {
            default_labels,
            definitions: HashMap::new(),
            values: RwLock::new(HashMap::new()),
            total_samples: AtomicU64::new(0),
        };

        // Register default training metrics
        exporter.register_default_metrics();
        exporter
    }

    /// Create with custom default labels
    pub fn with_labels(labels: LabelSet) -> Self {
        let mut exporter = Self {
            default_labels: labels,
            definitions: HashMap::new(),
            values: RwLock::new(HashMap::new()),
            total_samples: AtomicU64::new(0),
        };

        exporter.register_default_metrics();
        exporter
    }

    /// Register default training metrics
    fn register_default_metrics(&mut self) {
        // Training metrics
        self.register(
            MetricDef::gauge("entrenar_epoch_loss", "Training loss per epoch")
                .with_labels(&["experiment", "run"]),
        );
        self.register(
            MetricDef::gauge("entrenar_validation_loss", "Validation loss per epoch")
                .with_labels(&["experiment", "run"]),
        );
        self.register(
            MetricDef::gauge("entrenar_learning_rate", "Current learning rate")
                .with_labels(&["experiment", "run"]),
        );
        self.register(
            MetricDef::gauge("entrenar_batch_throughput", "Batches processed per second")
                .with_labels(&["experiment", "run"]),
        );
        self.register(
            MetricDef::gauge("entrenar_validation_accuracy", "Validation accuracy")
                .with_labels(&["experiment", "run"]),
        );
        self.register(
            MetricDef::counter(
                "entrenar_training_steps_total",
                "Total training steps completed",
            )
            .with_labels(&["experiment", "run"]),
        );

        // GPU metrics
        self.register(
            MetricDef::gauge("entrenar_gpu_utilization", "GPU utilization percentage")
                .with_labels(&["experiment", "run", "device"]),
        );
        self.register(
            MetricDef::gauge("entrenar_gpu_memory_used_bytes", "GPU memory used in bytes")
                .with_labels(&["experiment", "run", "device"]),
        );
        self.register(
            MetricDef::gauge(
                "entrenar_gpu_temperature_celsius",
                "GPU temperature in Celsius",
            )
            .with_labels(&["experiment", "run", "device"]),
        );
        self.register(
            MetricDef::gauge("entrenar_gpu_power_watts", "GPU power draw in watts").with_labels(&[
                "experiment",
                "run",
                "device",
            ]),
        );

        // System metrics
        self.register(
            MetricDef::gauge(
                "entrenar_memory_used_bytes",
                "Process memory usage in bytes",
            )
            .with_labels(&["experiment", "run"]),
        );
    }

    /// Register a metric definition
    pub fn register(&mut self, def: MetricDef) {
        self.definitions.insert(def.name.clone(), def);
    }

    /// Record a metric value
    pub fn record(&self, name: &str, value: f64) {
        self.record_with_labels(name, value, self.default_labels.clone());
    }

    /// Record a metric value with additional labels
    pub fn record_with_labels(&self, name: &str, value: f64, mut labels: LabelSet) {
        // Merge default labels
        for (k, v) in &self.default_labels.values {
            if !labels.values.iter().any(|(lk, _)| lk == k) {
                labels.values.push((k.clone(), v.clone()));
            }
        }

        let metric_value = MetricValue {
            labels,
            value,
            timestamp: Some(current_timestamp_ms()),
        };

        if let Ok(mut values) = self.values.write() {
            values
                .entry(name.to_string())
                .or_default()
                .push(metric_value);
        }

        self.total_samples.fetch_add(1, Ordering::Relaxed);
    }

    /// Record epoch metrics
    pub fn record_epoch(&self, epoch: u32, loss: f64, lr: f64) {
        self.record("entrenar_epoch_loss", loss);
        self.record("entrenar_learning_rate", lr);
        self.record("entrenar_training_steps_total", f64::from(epoch));
    }

    /// Record validation metrics
    pub fn record_validation(&self, loss: f64, accuracy: f64) {
        self.record("entrenar_validation_loss", loss);
        self.record("entrenar_validation_accuracy", accuracy);
    }

    /// Record batch throughput
    pub fn record_batch(&self, batches_per_second: f64) {
        self.record("entrenar_batch_throughput", batches_per_second);
    }

    /// Record GPU metrics for a device
    pub fn record_gpu(
        &self,
        device_id: u32,
        utilization: f64,
        memory_bytes: f64,
        temp: f64,
        power: f64,
    ) {
        let labels = self
            .default_labels
            .clone()
            .add("device", &device_id.to_string());

        self.record_with_labels("entrenar_gpu_utilization", utilization, labels.clone());
        self.record_with_labels(
            "entrenar_gpu_memory_used_bytes",
            memory_bytes,
            labels.clone(),
        );
        self.record_with_labels("entrenar_gpu_temperature_celsius", temp, labels.clone());
        self.record_with_labels("entrenar_gpu_power_watts", power, labels);
    }

    /// Record system memory usage
    pub fn record_memory(&self, used_bytes: f64) {
        self.record("entrenar_memory_used_bytes", used_bytes);
    }

    /// Get total samples recorded
    pub fn total_samples(&self) -> u64 {
        self.total_samples.load(Ordering::Relaxed)
    }

    /// Clear all recorded values
    pub fn clear(&self) {
        if let Ok(mut values) = self.values.write() {
            values.clear();
        }
    }

    /// Export metrics in Prometheus text format
    pub fn export(&self) -> String {
        let mut output = String::new();

        let values = match self.values.read() {
            Ok(v) => v,
            Err(_) => return output,
        };

        for (name, def) in &self.definitions {
            if let Some(metric_values) = values.get(name) {
                // Only export if we have values
                if metric_values.is_empty() {
                    continue;
                }

                // HELP line
                output.push_str(&format!("# HELP {} {}\n", name, def.help));

                // TYPE line
                let type_str = match def.metric_type {
                    MetricType::Gauge => "gauge",
                    MetricType::Counter => "counter",
                    MetricType::Histogram => "histogram",
                };
                output.push_str(&format!("# TYPE {name} {type_str}\n"));

                // Get latest value for each unique label set
                let mut latest: HashMap<String, &MetricValue> = HashMap::new();
                for mv in metric_values {
                    let key = mv.labels.format();
                    latest.insert(key, mv);
                }

                // Metric lines
                for mv in latest.values() {
                    let labels_str = mv.labels.format();
                    if let Some(ts) = mv.timestamp {
                        output.push_str(&format!("{}{} {} {}\n", name, labels_str, mv.value, ts));
                    } else {
                        output.push_str(&format!("{}{} {}\n", name, labels_str, mv.value));
                    }
                }
            }
        }

        output
    }

    /// Export metrics as JSON (for programmatic access)
    pub fn export_json(&self) -> String {
        let values = match self.values.read() {
            Ok(v) => v,
            Err(_) => return "{}".to_string(),
        };

        let mut metrics: HashMap<String, Vec<serde_json::Value>> = HashMap::new();

        for (name, metric_values) in values.iter() {
            let json_values: Vec<serde_json::Value> = metric_values
                .iter()
                .map(|mv| {
                    let mut obj = serde_json::Map::new();
                    for (k, v) in &mv.labels.values {
                        obj.insert(k.clone(), serde_json::Value::String(v.clone()));
                    }
                    obj.insert("value".to_string(), serde_json::json!(mv.value));
                    if let Some(ts) = mv.timestamp {
                        obj.insert("timestamp".to_string(), serde_json::json!(ts));
                    }
                    serde_json::Value::Object(obj)
                })
                .collect();

            metrics.insert(name.clone(), json_values);
        }

        serde_json::to_string_pretty(&metrics).unwrap_or_default()
    }
}

/// Get current timestamp in milliseconds
fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// =============================================================================
// Tests (TDD - Written First)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // MetricDef Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_metric_def_gauge() {
        let def = MetricDef::gauge("test_metric", "A test metric");
        assert_eq!(def.name, "test_metric");
        assert_eq!(def.help, "A test metric");
        assert_eq!(def.metric_type, MetricType::Gauge);
    }

    #[test]
    fn test_metric_def_counter() {
        let def = MetricDef::counter("test_counter", "A test counter");
        assert_eq!(def.metric_type, MetricType::Counter);
    }

    #[test]
    fn test_metric_def_with_labels() {
        let def = MetricDef::gauge("test", "help").with_labels(&["foo", "bar"]);
        assert_eq!(def.labels, vec!["foo", "bar"]);
    }

    // -------------------------------------------------------------------------
    // LabelSet Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_label_set_new() {
        let labels = LabelSet::new();
        assert!(labels.values.is_empty());
    }

    #[test]
    fn test_label_set_from_pairs() {
        let labels = LabelSet::from_pairs(&[("foo", "bar"), ("baz", "qux")]);
        assert_eq!(labels.values.len(), 2);
    }

    #[test]
    fn test_label_set_add() {
        let labels = LabelSet::new().add("key", "value");
        assert_eq!(labels.values.len(), 1);
        assert_eq!(labels.values[0], ("key".to_string(), "value".to_string()));
    }

    #[test]
    fn test_label_set_format_empty() {
        let labels = LabelSet::new();
        assert_eq!(labels.format(), "");
    }

    #[test]
    fn test_label_set_format() {
        let labels = LabelSet::from_pairs(&[("foo", "bar"), ("baz", "qux")]);
        let formatted = labels.format();
        assert!(formatted.contains("foo=\"bar\""));
        assert!(formatted.contains("baz=\"qux\""));
    }

    #[test]
    fn test_label_set_format_escaping() {
        let labels = LabelSet::from_pairs(&[("key", "value with \"quotes\"")]);
        let formatted = labels.format();
        assert!(formatted.contains("\\\""));
    }

    // -------------------------------------------------------------------------
    // PrometheusExporter Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_exporter_new() {
        let exporter = PrometheusExporter::new("test-exp", "run-1");
        assert!(!exporter.definitions.is_empty());
    }

    #[test]
    fn test_exporter_record() {
        let exporter = PrometheusExporter::new("test-exp", "run-1");
        exporter.record("entrenar_epoch_loss", 0.5);
        assert_eq!(exporter.total_samples(), 1);
    }

    #[test]
    fn test_exporter_record_epoch() {
        let exporter = PrometheusExporter::new("test-exp", "run-1");
        exporter.record_epoch(1, 0.5, 0.001);
        assert_eq!(exporter.total_samples(), 3); // loss, lr, steps
    }

    #[test]
    fn test_exporter_record_validation() {
        let exporter = PrometheusExporter::new("test-exp", "run-1");
        exporter.record_validation(0.4, 0.85);
        assert_eq!(exporter.total_samples(), 2);
    }

    #[test]
    fn test_exporter_record_batch() {
        let exporter = PrometheusExporter::new("test-exp", "run-1");
        exporter.record_batch(100.0);
        assert_eq!(exporter.total_samples(), 1);
    }

    #[test]
    fn test_exporter_record_gpu() {
        let exporter = PrometheusExporter::new("test-exp", "run-1");
        exporter.record_gpu(0, 85.0, 8_000_000_000.0, 65.0, 250.0);
        assert_eq!(exporter.total_samples(), 4); // util, mem, temp, power
    }

    #[test]
    fn test_exporter_record_memory() {
        let exporter = PrometheusExporter::new("test-exp", "run-1");
        exporter.record_memory(4_000_000_000.0);
        assert_eq!(exporter.total_samples(), 1);
    }

    #[test]
    fn test_exporter_clear() {
        let exporter = PrometheusExporter::new("test-exp", "run-1");
        exporter.record("entrenar_epoch_loss", 0.5);
        exporter.clear();

        let output = exporter.export();
        // Should not have any metric values after clear
        assert!(!output.contains("entrenar_epoch_loss{"));
    }

    // -------------------------------------------------------------------------
    // Export Format Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_exporter_export_prometheus_format() {
        let exporter = PrometheusExporter::new("test-exp", "run-1");
        exporter.record("entrenar_epoch_loss", 0.5);

        let output = exporter.export();

        // Should have HELP line
        assert!(output.contains("# HELP entrenar_epoch_loss"));
        // Should have TYPE line
        assert!(output.contains("# TYPE entrenar_epoch_loss gauge"));
        // Should have metric line with labels
        assert!(output.contains("entrenar_epoch_loss{"));
        assert!(output.contains("experiment=\"test-exp\""));
        assert!(output.contains("run=\"run-1\""));
    }

    #[test]
    fn test_exporter_export_with_timestamp() {
        let exporter = PrometheusExporter::new("test-exp", "run-1");
        exporter.record("entrenar_epoch_loss", 0.5);

        let output = exporter.export();

        // Should have timestamp at end of line
        let lines: Vec<&str> = output.lines().collect();
        let metric_line = lines.iter().find(|l| l.starts_with("entrenar_epoch_loss{"));
        assert!(metric_line.is_some());

        // Line format: metric{labels} value timestamp
        let parts: Vec<&str> = metric_line.unwrap().split_whitespace().collect();
        assert!(parts.len() >= 2); // metric+labels, value, optional timestamp
    }

    #[test]
    fn test_exporter_export_json() {
        let exporter = PrometheusExporter::new("test-exp", "run-1");
        exporter.record("entrenar_epoch_loss", 0.5);

        let json = exporter.export_json();

        // Should be valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_object());

        // Should have the metric
        let obj = parsed.as_object().unwrap();
        assert!(obj.contains_key("entrenar_epoch_loss"));
    }

    #[test]
    fn test_exporter_export_multiple_values() {
        let exporter = PrometheusExporter::new("test-exp", "run-1");
        exporter.record("entrenar_epoch_loss", 0.5);
        exporter.record("entrenar_epoch_loss", 0.4);
        exporter.record("entrenar_epoch_loss", 0.3);

        let output = exporter.export();

        // Should only have latest value in export (gauges show current value)
        let loss_lines: Vec<&str> = output
            .lines()
            .filter(|l| l.starts_with("entrenar_epoch_loss{"))
            .collect();
        assert_eq!(loss_lines.len(), 1);
    }

    #[test]
    fn test_exporter_export_multiple_gpus() {
        let exporter = PrometheusExporter::new("test-exp", "run-1");
        exporter.record_gpu(0, 80.0, 8e9, 60.0, 200.0);
        exporter.record_gpu(1, 90.0, 12e9, 70.0, 300.0);

        let output = exporter.export();

        // Should have metrics for both devices
        assert!(output.contains("device=\"0\""));
        assert!(output.contains("device=\"1\""));
    }

    // -------------------------------------------------------------------------
    // Custom Label Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_exporter_with_custom_labels() {
        let labels = LabelSet::from_pairs(&[("model", "llama-7b"), ("dataset", "alpaca")]);
        let exporter = PrometheusExporter::with_labels(labels);
        exporter.record("entrenar_epoch_loss", 0.5);

        let output = exporter.export();
        assert!(output.contains("model=\"llama-7b\""));
        assert!(output.contains("dataset=\"alpaca\""));
    }

    #[test]
    fn test_exporter_record_with_extra_labels() {
        let exporter = PrometheusExporter::new("test-exp", "run-1");

        let extra_labels = LabelSet::from_pairs(&[("layer", "attention")]);
        exporter.record_with_labels("entrenar_epoch_loss", 0.5, extra_labels);

        let output = exporter.export();
        assert!(output.contains("layer=\"attention\""));
        assert!(output.contains("experiment=\"test-exp\"")); // Default labels preserved
    }

    // -------------------------------------------------------------------------
    // Metric Registration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_exporter_register_custom_metric() {
        let mut exporter = PrometheusExporter::new("test-exp", "run-1");

        exporter.register(MetricDef::gauge("custom_metric", "A custom metric"));
        exporter.record("custom_metric", 42.0);

        let output = exporter.export();
        assert!(output.contains("# HELP custom_metric A custom metric"));
        assert!(output.contains("custom_metric{"));
    }

    #[test]
    fn test_exporter_default_metrics_registered() {
        let exporter = PrometheusExporter::new("test-exp", "run-1");

        // Check that default metrics are registered
        assert!(exporter.definitions.contains_key("entrenar_epoch_loss"));
        assert!(exporter.definitions.contains_key("entrenar_learning_rate"));
        assert!(exporter
            .definitions
            .contains_key("entrenar_gpu_utilization"));
        assert!(exporter
            .definitions
            .contains_key("entrenar_validation_accuracy"));
    }
}

// =============================================================================
// Property Tests
// =============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_label_escape_roundtrip(s in "[a-zA-Z0-9 \"\\\\\\n]{0,50}") {
            let escaped = escape_label_value(&s);
            // Escaped string should not have unescaped quotes or newlines
            prop_assert!(!escaped.contains('"') || escaped.contains("\\\""));
        }

        #[test]
        fn prop_label_set_format_balanced_braces(
            pairs in prop::collection::vec(
                ("[a-z]{1,10}", "[a-zA-Z0-9]{1,20}"),
                1..5
            )
        ) {
            let labels = LabelSet::from_pairs(
                &pairs.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect::<Vec<_>>()
            );
            let formatted = labels.format();

            // Count braces
            let open = formatted.chars().filter(|&c| c == '{').count();
            let close = formatted.chars().filter(|&c| c == '}').count();
            prop_assert_eq!(open, close);
        }

        #[test]
        fn prop_export_valid_format(
            loss in 0.0f64..10.0,
            lr in 1e-10f64..1.0
        ) {
            let exporter = PrometheusExporter::new("test", "run");
            exporter.record_epoch(1, loss, lr);

            let output = exporter.export();

            // Should have required components
            prop_assert!(output.contains("# HELP"));
            prop_assert!(output.contains("# TYPE"));
        }

        #[test]
        fn prop_total_samples_accurate(count in 1usize..100) {
            let exporter = PrometheusExporter::new("test", "run");
            for i in 0..count {
                exporter.record("entrenar_epoch_loss", i as f64 * 0.1);
            }
            prop_assert_eq!(exporter.total_samples() as usize, count);
        }

        #[test]
        fn prop_gpu_metrics_all_recorded(device_id in 0u32..8) {
            let exporter = PrometheusExporter::new("test", "run");
            exporter.record_gpu(device_id, 80.0, 8e9, 65.0, 200.0);

            let output = exporter.export();
            let device_str = format!("device=\"{device_id}\"");
            prop_assert!(output.contains(&device_str));
        }
    }
}
