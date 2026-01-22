//! Benchmark result types
//!
//! Data structures for quantization benchmark results.

use super::super::granularity::{QuantGranularity, QuantMode};
use serde::{Deserialize, Serialize};

/// Benchmark results for quantization accuracy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantBenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Number of elements tested
    pub num_elements: usize,
    /// Bits used for quantization
    pub bits: u8,
    /// Granularity used
    pub granularity: QuantGranularity,
    /// Mode used (symmetric/asymmetric)
    pub mode: QuantMode,
    /// MSE error
    pub mse: f32,
    /// Max error
    pub max_error: f32,
    /// SQNR in dB
    pub sqnr_db: f32,
    /// Compression ratio
    pub compression_ratio: f32,
}

impl QuantBenchmarkResult {
    /// Quality score (higher is better): SQNR / compression overhead
    pub fn quality_score(&self) -> f32 {
        if self.compression_ratio > 0.0 {
            self.sqnr_db / self.compression_ratio.max(1.0)
        } else {
            0.0
        }
    }
}

/// Suite of benchmark results
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct BenchmarkSuite {
    pub results: Vec<QuantBenchmarkResult>,
}

impl BenchmarkSuite {
    /// Add a benchmark result
    pub fn add(&mut self, result: QuantBenchmarkResult) {
        self.results.push(result);
    }

    /// Get best result by SQNR
    pub fn best_by_sqnr(&self) -> Option<&QuantBenchmarkResult> {
        self.results
            .iter()
            .max_by(|a, b| a.sqnr_db.partial_cmp(&b.sqnr_db).unwrap())
    }

    /// Get best result by MSE (lowest)
    pub fn best_by_mse(&self) -> Option<&QuantBenchmarkResult> {
        self.results
            .iter()
            .min_by(|a, b| a.mse.partial_cmp(&b.mse).unwrap())
    }

    /// Get results sorted by quality score
    pub fn sorted_by_quality(&self) -> Vec<&QuantBenchmarkResult> {
        let mut sorted: Vec<_> = self.results.iter().collect();
        sorted.sort_by(|a, b| b.quality_score().partial_cmp(&a.quality_score()).unwrap());
        sorted
    }
}
