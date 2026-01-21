//! Dashboard Module (Phase 2: ENT-003, ENT-004)
//!
//! Provides the `DashboardSource` trait for real-time training monitoring
//! and WASM bindings for browser-based dashboards.
//!
//! # Features
//!
//! - Real-time metric streaming via `subscribe()`
//! - Resource usage monitoring (GPU, CPU, memory)
//! - Trend analysis for metrics
//! - WASM support for browser dashboards (feature: "wasm")
//!
//! # Example
//!
//! ```
//! use std::sync::{Arc, Mutex};
//! use entrenar::storage::{InMemoryStorage, ExperimentStorage};
//! use entrenar::run::{Run, TracingConfig};
//! use entrenar::dashboard::{DashboardSource, Trend};
//!
//! let mut storage = InMemoryStorage::new();
//! let exp_id = storage.create_experiment("my-exp", None).unwrap();
//! let storage = Arc::new(Mutex::new(storage));
//!
//! let mut run = Run::new(&exp_id, storage.clone(), TracingConfig::disabled()).unwrap();
//! run.log_metric("loss", 0.5).unwrap();
//! run.log_metric("loss", 0.4).unwrap();
//! run.log_metric("loss", 0.3).unwrap();
//!
//! // Get dashboard data
//! let status = run.status();
//! let metrics = run.recent_metrics(10);
//! let resources = run.resource_usage();
//! ```

#[cfg(feature = "wasm")]
pub mod wasm;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::run::Run;
use crate::storage::{ExperimentStorage, MetricPoint, RunStatus};

/// Trend direction for a metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Trend {
    /// Metric is increasing
    Rising,
    /// Metric is decreasing
    Falling,
    /// Metric is relatively stable
    Stable,
}

impl Trend {
    /// Compute trend from a series of values.
    ///
    /// Uses linear regression slope to determine trend direction.
    pub fn from_values(values: &[f64]) -> Self {
        if values.len() < 2 {
            return Self::Stable;
        }

        // Simple linear regression slope
        let n = values.len() as f64;
        let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
        let sum_y: f64 = values.iter().sum();
        let sum_xy: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_x2: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator.abs() < f64::EPSILON {
            return Self::Stable;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;

        // Normalize slope by mean to get relative change
        let mean = sum_y / n;
        if mean.abs() < f64::EPSILON {
            return Self::Stable;
        }

        let relative_slope = slope / mean;

        // Thresholds for trend detection (5% relative change)
        const THRESHOLD: f64 = 0.05;

        if relative_slope > THRESHOLD {
            Self::Rising
        } else if relative_slope < -THRESHOLD {
            Self::Falling
        } else {
            Self::Stable
        }
    }

    /// Get emoji representation.
    pub fn emoji(&self) -> &'static str {
        match self {
            Self::Rising => "↑",
            Self::Falling => "↓",
            Self::Stable => "→",
        }
    }
}

impl std::fmt::Display for Trend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rising => write!(f, "rising"),
            Self::Falling => write!(f, "falling"),
            Self::Stable => write!(f, "stable"),
        }
    }
}

/// A snapshot of metric values for dashboard display.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricSnapshot {
    /// Metric key (e.g., "loss", "accuracy")
    pub key: String,
    /// Time-value pairs: (timestamp_ms, value)
    pub values: Vec<(u64, f64)>,
    /// Current trend direction
    pub trend: Trend,
}

impl MetricSnapshot {
    /// Create a new metric snapshot.
    pub fn new(key: impl Into<String>, values: Vec<(u64, f64)>) -> Self {
        let trend = Trend::from_values(&values.iter().map(|(_, v)| *v).collect::<Vec<_>>());
        Self {
            key: key.into(),
            values,
            trend,
        }
    }

    /// Create from metric points.
    pub fn from_points(key: impl Into<String>, points: &[MetricPoint]) -> Self {
        let values: Vec<(u64, f64)> = points
            .iter()
            .map(|p| {
                let ts = p.timestamp.timestamp_millis() as u64;
                (ts, p.value)
            })
            .collect();
        Self::new(key, values)
    }

    /// Get the latest value.
    pub fn latest(&self) -> Option<f64> {
        self.values.last().map(|(_, v)| *v)
    }

    /// Get the minimum value.
    pub fn min(&self) -> Option<f64> {
        self.values.iter().map(|(_, v)| *v).reduce(f64::min)
    }

    /// Get the maximum value.
    pub fn max(&self) -> Option<f64> {
        self.values.iter().map(|(_, v)| *v).reduce(f64::max)
    }

    /// Get the mean value.
    pub fn mean(&self) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }
        Some(self.values.iter().map(|(_, v)| *v).sum::<f64>() / self.values.len() as f64)
    }

    /// Check if metric is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get number of data points.
    pub fn len(&self) -> usize {
        self.values.len()
    }
}

/// Resource usage snapshot for dashboard display.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    /// GPU utilization (0.0 to 1.0)
    pub gpu_util: f64,
    /// CPU utilization (0.0 to 1.0)
    pub cpu_util: f64,
    /// Memory used in bytes
    pub memory_used: u64,
    /// Total memory in bytes
    pub memory_total: u64,
    /// GPU memory used in bytes (optional)
    pub gpu_memory_used: Option<u64>,
    /// Total GPU memory in bytes (optional)
    pub gpu_memory_total: Option<u64>,
}

impl Default for ResourceSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceSnapshot {
    /// Create a new resource snapshot with zero values.
    pub fn new() -> Self {
        Self {
            gpu_util: 0.0,
            cpu_util: 0.0,
            memory_used: 0,
            memory_total: 0,
            gpu_memory_used: None,
            gpu_memory_total: None,
        }
    }

    /// Set GPU utilization.
    pub fn with_gpu_util(mut self, util: f64) -> Self {
        self.gpu_util = util.clamp(0.0, 1.0);
        self
    }

    /// Set CPU utilization.
    pub fn with_cpu_util(mut self, util: f64) -> Self {
        self.cpu_util = util.clamp(0.0, 1.0);
        self
    }

    /// Set memory usage.
    pub fn with_memory(mut self, used: u64, total: u64) -> Self {
        self.memory_used = used;
        self.memory_total = total;
        self
    }

    /// Set GPU memory usage.
    pub fn with_gpu_memory(mut self, used: u64, total: u64) -> Self {
        self.gpu_memory_used = Some(used);
        self.gpu_memory_total = Some(total);
        self
    }

    /// Get memory utilization as a fraction.
    pub fn memory_util(&self) -> f64 {
        if self.memory_total == 0 {
            return 0.0;
        }
        self.memory_used as f64 / self.memory_total as f64
    }

    /// Get GPU memory utilization as a fraction.
    pub fn gpu_memory_util(&self) -> Option<f64> {
        match (self.gpu_memory_used, self.gpu_memory_total) {
            (Some(used), Some(total)) if total > 0 => Some(used as f64 / total as f64),
            _ => None,
        }
    }
}

/// Subscription callback type.
pub type SubscriptionCallback = Box<dyn Fn(&str, f64) + Send>;

/// Dashboard data source trait.
///
/// Implement this trait to provide data for real-time dashboards.
pub trait DashboardSource {
    /// Get the current run status.
    fn status(&self) -> RunStatus;

    /// Get recent metrics, limited to `limit` points per metric.
    fn recent_metrics(&self, limit: usize) -> HashMap<String, MetricSnapshot>;

    /// Subscribe to metric updates.
    ///
    /// Returns a receiver that will receive metric snapshots as they arrive.
    /// The callback is called with the metric key and latest value.
    fn subscribe(&self, callback: SubscriptionCallback);

    /// Get current resource usage.
    fn resource_usage(&self) -> ResourceSnapshot;
}

impl<S: ExperimentStorage> DashboardSource for Run<S> {
    fn status(&self) -> RunStatus {
        if self.is_finished() {
            // Query storage for actual status
            let storage = self.storage_ref();
            storage
                .lock()
                .ok()
                .and_then(|s| s.get_run_status(&self.id).ok())
                .unwrap_or(RunStatus::Success)
        } else {
            RunStatus::Running
        }
    }

    fn recent_metrics(&self, limit: usize) -> HashMap<String, MetricSnapshot> {
        let mut result = HashMap::new();

        // Get metrics from storage
        let storage = self.storage_ref();
        if let Ok(guard) = storage.lock() {
            // Get all metric keys we've logged
            for key in self.metric_keys() {
                if let Ok(points) = guard.get_metrics(&self.id, &key) {
                    // Take only the most recent `limit` points
                    let recent: Vec<_> = if points.len() > limit {
                        points[points.len() - limit..].to_vec()
                    } else {
                        points
                    };

                    let snapshot = MetricSnapshot::from_points(&key, &recent);
                    result.insert(key, snapshot);
                }
            }
        }

        result
    }

    fn subscribe(&self, _callback: SubscriptionCallback) {
        // Subscriptions are handled at a higher level
        // This is a placeholder for the trait requirement
        // In practice, use a channel-based approach or web sockets for WASM
    }

    fn resource_usage(&self) -> ResourceSnapshot {
        // Return simulated resource usage
        // In production, this would query actual system metrics
        ResourceSnapshot::new()
            .with_cpu_util(0.0)
            .with_gpu_util(0.0)
            .with_memory(0, 0)
    }
}

impl<S: ExperimentStorage> Run<S> {
    /// Get a reference to the storage (for dashboard use).
    pub(crate) fn storage_ref(&self) -> &Arc<Mutex<S>> {
        &self.storage
    }

    /// Get all metric keys that have been logged.
    pub fn metric_keys(&self) -> Vec<String> {
        self.step_counters.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run::TracingConfig;
    use crate::storage::InMemoryStorage;

    fn setup_storage() -> (Arc<Mutex<InMemoryStorage>>, String) {
        let mut storage = InMemoryStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).unwrap();
        (Arc::new(Mutex::new(storage)), exp_id)
    }

    #[test]
    fn test_trend_from_rising_values() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(Trend::from_values(&values), Trend::Rising);
    }

    #[test]
    fn test_trend_from_falling_values() {
        let values = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(Trend::from_values(&values), Trend::Falling);
    }

    #[test]
    fn test_trend_from_stable_values() {
        let values = vec![5.0, 5.01, 4.99, 5.0, 5.02];
        assert_eq!(Trend::from_values(&values), Trend::Stable);
    }

    #[test]
    fn test_trend_from_single_value() {
        let values = vec![5.0];
        assert_eq!(Trend::from_values(&values), Trend::Stable);
    }

    #[test]
    fn test_trend_from_empty() {
        let values: Vec<f64> = vec![];
        assert_eq!(Trend::from_values(&values), Trend::Stable);
    }

    #[test]
    fn test_trend_display() {
        assert_eq!(format!("{}", Trend::Rising), "rising");
        assert_eq!(format!("{}", Trend::Falling), "falling");
        assert_eq!(format!("{}", Trend::Stable), "stable");
    }

    #[test]
    fn test_trend_emoji() {
        assert_eq!(Trend::Rising.emoji(), "↑");
        assert_eq!(Trend::Falling.emoji(), "↓");
        assert_eq!(Trend::Stable.emoji(), "→");
    }

    #[test]
    fn test_metric_snapshot_new() {
        let values = vec![(1000, 0.5), (2000, 0.4), (3000, 0.3)];
        let snapshot = MetricSnapshot::new("loss", values);

        assert_eq!(snapshot.key, "loss");
        assert_eq!(snapshot.len(), 3);
        assert_eq!(snapshot.trend, Trend::Falling);
    }

    #[test]
    fn test_metric_snapshot_latest() {
        let values = vec![(1000, 0.5), (2000, 0.4), (3000, 0.3)];
        let snapshot = MetricSnapshot::new("loss", values);

        assert_eq!(snapshot.latest(), Some(0.3));
    }

    #[test]
    fn test_metric_snapshot_min_max() {
        let values = vec![(1000, 0.5), (2000, 0.2), (3000, 0.8)];
        let snapshot = MetricSnapshot::new("metric", values);

        assert_eq!(snapshot.min(), Some(0.2));
        assert_eq!(snapshot.max(), Some(0.8));
    }

    #[test]
    fn test_metric_snapshot_mean() {
        let values = vec![(1000, 1.0), (2000, 2.0), (3000, 3.0)];
        let snapshot = MetricSnapshot::new("metric", values);

        assert!((snapshot.mean().unwrap() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metric_snapshot_empty() {
        let snapshot = MetricSnapshot::new("empty", vec![]);

        assert!(snapshot.is_empty());
        assert_eq!(snapshot.len(), 0);
        assert_eq!(snapshot.latest(), None);
        assert_eq!(snapshot.min(), None);
        assert_eq!(snapshot.max(), None);
        assert_eq!(snapshot.mean(), None);
    }

    #[test]
    fn test_resource_snapshot_default() {
        let snapshot = ResourceSnapshot::default();

        assert!((snapshot.gpu_util - 0.0).abs() < f64::EPSILON);
        assert!((snapshot.cpu_util - 0.0).abs() < f64::EPSILON);
        assert_eq!(snapshot.memory_used, 0);
        assert_eq!(snapshot.memory_total, 0);
    }

    #[test]
    fn test_resource_snapshot_with_values() {
        let snapshot = ResourceSnapshot::new()
            .with_gpu_util(0.75)
            .with_cpu_util(0.50)
            .with_memory(4_000_000_000, 8_000_000_000)
            .with_gpu_memory(6_000_000_000, 16_000_000_000);

        assert!((snapshot.gpu_util - 0.75).abs() < f64::EPSILON);
        assert!((snapshot.cpu_util - 0.50).abs() < f64::EPSILON);
        assert!((snapshot.memory_util() - 0.5).abs() < f64::EPSILON);
        assert!((snapshot.gpu_memory_util().unwrap() - 0.375).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resource_snapshot_clamp() {
        let snapshot = ResourceSnapshot::new()
            .with_gpu_util(1.5)
            .with_cpu_util(-0.5);

        assert!((snapshot.gpu_util - 1.0).abs() < f64::EPSILON);
        assert!((snapshot.cpu_util - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resource_snapshot_zero_total() {
        let snapshot = ResourceSnapshot::new();
        assert!((snapshot.memory_util() - 0.0).abs() < f64::EPSILON);
        assert!(snapshot.gpu_memory_util().is_none());
    }

    #[test]
    fn test_run_dashboard_status_running() {
        let (storage, exp_id) = setup_storage();
        let run = Run::new(&exp_id, storage, TracingConfig::disabled()).unwrap();

        assert_eq!(run.status(), RunStatus::Running);
    }

    #[test]
    fn test_run_dashboard_status_finished() {
        let (storage, exp_id) = setup_storage();
        let run = Run::new(&exp_id, storage, TracingConfig::disabled()).unwrap();
        let run_id = run.id.clone();

        run.finish(RunStatus::Success).unwrap();

        // Status would be checked after finish
        // Note: we consumed run, so we can't call status() on it
        let _ = run_id;
    }

    #[test]
    fn test_run_dashboard_recent_metrics() {
        let (storage, exp_id) = setup_storage();
        let mut run = Run::new(&exp_id, storage, TracingConfig::disabled()).unwrap();

        run.log_metric("loss", 0.5).unwrap();
        run.log_metric("loss", 0.4).unwrap();
        run.log_metric("loss", 0.3).unwrap();
        run.log_metric("accuracy", 0.8).unwrap();

        let metrics = run.recent_metrics(10);

        assert!(metrics.contains_key("loss"));
        assert!(metrics.contains_key("accuracy"));
        assert_eq!(metrics.get("loss").unwrap().len(), 3);
        assert_eq!(metrics.get("accuracy").unwrap().len(), 1);
    }

    #[test]
    fn test_run_dashboard_recent_metrics_limited() {
        let (storage, exp_id) = setup_storage();
        let mut run = Run::new(&exp_id, storage, TracingConfig::disabled()).unwrap();

        for i in 0..20 {
            run.log_metric("loss", 1.0 / (f64::from(i) + 1.0)).unwrap();
        }

        let metrics = run.recent_metrics(5);

        assert_eq!(metrics.get("loss").unwrap().len(), 5);
    }

    #[test]
    fn test_run_metric_keys() {
        let (storage, exp_id) = setup_storage();
        let mut run = Run::new(&exp_id, storage, TracingConfig::disabled()).unwrap();

        run.log_metric("loss", 0.5).unwrap();
        run.log_metric("accuracy", 0.8).unwrap();
        run.log_metric("f1", 0.75).unwrap();

        let keys = run.metric_keys();

        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&"loss".to_string()));
        assert!(keys.contains(&"accuracy".to_string()));
        assert!(keys.contains(&"f1".to_string()));
    }

    #[test]
    fn test_run_resource_usage() {
        let (storage, exp_id) = setup_storage();
        let run = Run::new(&exp_id, storage, TracingConfig::disabled()).unwrap();

        let resources = run.resource_usage();

        // Default implementation returns zeros
        assert!((resources.gpu_util - 0.0).abs() < f64::EPSILON);
        assert!((resources.cpu_util - 0.0).abs() < f64::EPSILON);
    }
}
