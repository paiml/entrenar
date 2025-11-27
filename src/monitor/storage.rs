//! Metrics Storage Module
//!
//! Persists training metrics to Parquet files using trueno-db.
//! Feature-gated behind `monitor` feature flag.

use super::{Metric, MetricRecord, MetricStats};
use std::path::Path;

/// Result type for storage operations
pub type StorageResult<T> = Result<T, StorageError>;

/// Storage errors
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Storage not initialized")]
    NotInitialized,
}

/// Metrics storage backend trait
pub trait MetricsStore: Send + Sync {
    /// Write a batch of metric records
    fn write_batch(&mut self, records: &[MetricRecord]) -> StorageResult<()>;

    /// Query metrics by name within a time range
    fn query_range(
        &self,
        metric: &Metric,
        start_ts: u64,
        end_ts: u64,
    ) -> StorageResult<Vec<MetricRecord>>;

    /// Get all records for a metric
    fn query_all(&self, metric: &Metric) -> StorageResult<Vec<MetricRecord>>;

    /// Get summary statistics for a metric
    fn query_stats(&self, metric: &Metric) -> StorageResult<Option<MetricStats>>;

    /// Get total record count
    fn count(&self) -> StorageResult<usize>;

    /// Flush pending writes
    fn flush(&mut self) -> StorageResult<()>;
}

/// In-memory metrics store (always available, no feature flag)
#[derive(Debug, Default)]
pub struct InMemoryStore {
    records: Vec<MetricRecord>,
}

impl InMemoryStore {
    /// Create a new in-memory store
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    /// Get all records
    pub fn all_records(&self) -> &[MetricRecord] {
        &self.records
    }
}

impl MetricsStore for InMemoryStore {
    fn write_batch(&mut self, records: &[MetricRecord]) -> StorageResult<()> {
        self.records.extend(records.iter().cloned());
        Ok(())
    }

    fn query_range(
        &self,
        metric: &Metric,
        start_ts: u64,
        end_ts: u64,
    ) -> StorageResult<Vec<MetricRecord>> {
        Ok(self
            .records
            .iter()
            .filter(|r| &r.metric == metric && r.timestamp >= start_ts && r.timestamp <= end_ts)
            .cloned()
            .collect())
    }

    fn query_all(&self, metric: &Metric) -> StorageResult<Vec<MetricRecord>> {
        Ok(self
            .records
            .iter()
            .filter(|r| &r.metric == metric)
            .cloned()
            .collect())
    }

    fn query_stats(&self, metric: &Metric) -> StorageResult<Option<MetricStats>> {
        let values: Vec<f64> = self
            .records
            .iter()
            .filter(|r| &r.metric == metric)
            .map(|r| r.value)
            .collect();

        if values.is_empty() {
            return Ok(None);
        }

        let count = values.len();
        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let variance = if count > 1 {
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1) as f64
        } else {
            0.0
        };
        let std = variance.sqrt();

        let has_nan = values.iter().any(|v| v.is_nan());
        let has_inf = values.iter().any(|v| v.is_infinite());

        Ok(Some(MetricStats {
            count,
            mean,
            std,
            min,
            max,
            sum,
            has_nan,
            has_inf,
        }))
    }

    fn count(&self) -> StorageResult<usize> {
        Ok(self.records.len())
    }

    fn flush(&mut self) -> StorageResult<()> {
        Ok(())
    }
}

/// JSON file-based metrics store
pub struct JsonFileStore {
    path: std::path::PathBuf,
    records: Vec<MetricRecord>,
    dirty: bool,
}

impl JsonFileStore {
    /// Create or open a JSON file store
    pub fn open<P: AsRef<Path>>(path: P) -> StorageResult<Self> {
        let path = path.as_ref().to_path_buf();
        let records = if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            serde_json::from_str(&content)
                .map_err(|e| StorageError::Serialization(e.to_string()))?
        } else {
            Vec::new()
        };

        Ok(Self {
            path,
            records,
            dirty: false,
        })
    }

    /// Get the file path
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl MetricsStore for JsonFileStore {
    fn write_batch(&mut self, records: &[MetricRecord]) -> StorageResult<()> {
        self.records.extend(records.iter().cloned());
        self.dirty = true;
        Ok(())
    }

    fn query_range(
        &self,
        metric: &Metric,
        start_ts: u64,
        end_ts: u64,
    ) -> StorageResult<Vec<MetricRecord>> {
        Ok(self
            .records
            .iter()
            .filter(|r| &r.metric == metric && r.timestamp >= start_ts && r.timestamp <= end_ts)
            .cloned()
            .collect())
    }

    fn query_all(&self, metric: &Metric) -> StorageResult<Vec<MetricRecord>> {
        Ok(self
            .records
            .iter()
            .filter(|r| &r.metric == metric)
            .cloned()
            .collect())
    }

    fn query_stats(&self, metric: &Metric) -> StorageResult<Option<MetricStats>> {
        // Reuse InMemoryStore logic
        let mem_store = InMemoryStore {
            records: self.records.clone(),
        };
        mem_store.query_stats(metric)
    }

    fn count(&self) -> StorageResult<usize> {
        Ok(self.records.len())
    }

    fn flush(&mut self) -> StorageResult<()> {
        if self.dirty {
            let json = serde_json::to_string_pretty(&self.records)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
            std::fs::write(&self.path, json)?;
            self.dirty = false;
        }
        Ok(())
    }
}

impl Drop for JsonFileStore {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_store_new() {
        let store = InMemoryStore::new();
        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn test_in_memory_write_batch() {
        let mut store = InMemoryStore::new();
        let records = vec![
            MetricRecord::new(Metric::Loss, 0.5),
            MetricRecord::new(Metric::Accuracy, 0.85),
        ];
        store.write_batch(&records).unwrap();
        assert_eq!(store.count().unwrap(), 2);
    }

    #[test]
    fn test_in_memory_query_all() {
        let mut store = InMemoryStore::new();
        store
            .write_batch(&[
                MetricRecord::new(Metric::Loss, 0.5),
                MetricRecord::new(Metric::Loss, 0.4),
                MetricRecord::new(Metric::Accuracy, 0.85),
            ])
            .unwrap();

        let loss_records = store.query_all(&Metric::Loss).unwrap();
        assert_eq!(loss_records.len(), 2);
    }

    #[test]
    fn test_in_memory_query_stats() {
        let mut store = InMemoryStore::new();
        store
            .write_batch(&[
                MetricRecord::new(Metric::Loss, 1.0),
                MetricRecord::new(Metric::Loss, 2.0),
                MetricRecord::new(Metric::Loss, 3.0),
            ])
            .unwrap();

        let stats = store.query_stats(&Metric::Loss).unwrap().unwrap();
        assert!((stats.mean - 2.0).abs() < 1e-6);
        assert!((stats.min - 1.0).abs() < 1e-6);
        assert!((stats.max - 3.0).abs() < 1e-6);
        assert_eq!(stats.count, 3);
    }

    #[test]
    fn test_in_memory_query_stats_empty() {
        let store = InMemoryStore::new();
        let stats = store.query_stats(&Metric::Loss).unwrap();
        assert!(stats.is_none());
    }

    #[test]
    fn test_json_file_store_create() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("metrics.json");

        let store = JsonFileStore::open(&path).unwrap();
        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn test_json_file_store_write_and_flush() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("metrics.json");

        {
            let mut store = JsonFileStore::open(&path).unwrap();
            store
                .write_batch(&[MetricRecord::new(Metric::Loss, 0.5)])
                .unwrap();
            store.flush().unwrap();
        }

        // Reopen and verify
        let store = JsonFileStore::open(&path).unwrap();
        assert_eq!(store.count().unwrap(), 1);
    }

    #[test]
    fn test_json_file_store_persistence() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("metrics.json");

        // Write some records
        {
            let mut store = JsonFileStore::open(&path).unwrap();
            store
                .write_batch(&[
                    MetricRecord::new(Metric::Loss, 0.5),
                    MetricRecord::new(Metric::Accuracy, 0.85),
                ])
                .unwrap();
            // Drop triggers flush
        }

        // Reopen and add more
        {
            let mut store = JsonFileStore::open(&path).unwrap();
            assert_eq!(store.count().unwrap(), 2);
            store
                .write_batch(&[MetricRecord::new(Metric::Loss, 0.4)])
                .unwrap();
        }

        // Verify final count
        let store = JsonFileStore::open(&path).unwrap();
        assert_eq!(store.count().unwrap(), 3);
    }
}
