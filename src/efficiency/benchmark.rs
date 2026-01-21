//! Cost-Performance Benchmarking (ENT-011)
//!
//! Provides benchmarking infrastructure with Pareto frontier analysis
//! for cost-performance optimization.

use super::{ComputeDevice, CostMetrics, EnergyMetrics, ModelParadigm};
use serde::{Deserialize, Serialize};

/// A single benchmark entry
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkEntry {
    /// Unique run identifier
    pub run_id: String,
    /// Model paradigm used
    pub paradigm: ModelParadigm,
    /// Compute device used
    pub device: ComputeDevice,
    /// Quality score achieved (accuracy, F1, etc.)
    pub quality_score: f64,
    /// Cost metrics
    pub cost: CostMetrics,
    /// Energy metrics
    pub energy: EnergyMetrics,
}

impl BenchmarkEntry {
    /// Create a new benchmark entry
    pub fn new(
        run_id: impl Into<String>,
        paradigm: ModelParadigm,
        device: ComputeDevice,
        quality_score: f64,
        cost: CostMetrics,
        energy: EnergyMetrics,
    ) -> Self {
        Self {
            run_id: run_id.into(),
            paradigm,
            device,
            quality_score,
            cost,
            energy,
        }
    }

    /// Get efficiency score (quality per dollar)
    pub fn efficiency_score(&self) -> f64 {
        if self.cost.total_cost_usd > 0.0 {
            self.quality_score / self.cost.total_cost_usd
        } else {
            f64::INFINITY
        }
    }

    /// Get energy efficiency (quality per kWh)
    pub fn energy_efficiency(&self) -> f64 {
        let kwh = self.energy.kwh();
        if kwh > 0.0 {
            self.quality_score / kwh
        } else {
            f64::INFINITY
        }
    }

    /// Get carbon efficiency (quality per kg CO2)
    pub fn carbon_efficiency(&self) -> f64 {
        if self.energy.carbon_kg > 0.0 {
            self.quality_score / self.energy.carbon_kg
        } else {
            f64::INFINITY
        }
    }

    /// Check if this entry dominates another (better in all metrics)
    pub fn dominates(&self, other: &Self) -> bool {
        self.quality_score >= other.quality_score
            && self.cost.total_cost_usd <= other.cost.total_cost_usd
            && self.energy.joules_total <= other.energy.joules_total
            && (self.quality_score > other.quality_score
                || self.cost.total_cost_usd < other.cost.total_cost_usd
                || self.energy.joules_total < other.energy.joules_total)
    }
}

/// Cost-performance benchmark collection
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostPerformanceBenchmark {
    /// Benchmark entries
    pub entries: Vec<BenchmarkEntry>,
}

impl CostPerformanceBenchmark {
    /// Create a new empty benchmark
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a benchmark entry
    pub fn add(&mut self, entry: BenchmarkEntry) {
        self.entries.push(entry);
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Find the Pareto frontier (non-dominated entries)
    ///
    /// Returns entries where no other entry is better in both quality AND cost.
    pub fn pareto_frontier(&self) -> Vec<&BenchmarkEntry> {
        if self.entries.is_empty() {
            return Vec::new();
        }

        let mut frontier = Vec::new();

        for entry in &self.entries {
            let is_dominated = self
                .entries
                .iter()
                .any(|other| !std::ptr::eq(entry, other) && other.dominates(entry));

            if !is_dominated {
                frontier.push(entry);
            }
        }

        // Sort by quality (descending)
        frontier.sort_by(|a, b| {
            b.quality_score
                .partial_cmp(&a.quality_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        frontier
    }

    /// Find best entry within a budget
    pub fn best_for_budget(&self, max_usd: f64) -> Option<&BenchmarkEntry> {
        self.entries
            .iter()
            .filter(|e| e.cost.total_cost_usd <= max_usd)
            .max_by(|a, b| {
                a.quality_score
                    .partial_cmp(&b.quality_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Find cheapest entry that meets quality threshold
    pub fn cheapest_for_quality(&self, min_quality: f64) -> Option<&BenchmarkEntry> {
        self.entries
            .iter()
            .filter(|e| e.quality_score >= min_quality)
            .min_by(|a, b| {
                a.cost
                    .total_cost_usd
                    .partial_cmp(&b.cost.total_cost_usd)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Calculate efficiency score for an entry
    pub fn efficiency_score(&self, entry: &BenchmarkEntry) -> f64 {
        entry.efficiency_score()
    }

    /// Find most efficient entry (highest quality/cost ratio)
    pub fn most_efficient(&self) -> Option<&BenchmarkEntry> {
        self.entries.iter().max_by(|a, b| {
            a.efficiency_score()
                .partial_cmp(&b.efficiency_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Find greenest entry (highest quality/energy ratio)
    pub fn greenest(&self) -> Option<&BenchmarkEntry> {
        self.entries.iter().max_by(|a, b| {
            a.energy_efficiency()
                .partial_cmp(&b.energy_efficiency())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get best entry by quality
    pub fn best_quality(&self) -> Option<&BenchmarkEntry> {
        self.entries.iter().max_by(|a, b| {
            a.quality_score
                .partial_cmp(&b.quality_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get cheapest entry
    pub fn cheapest(&self) -> Option<&BenchmarkEntry> {
        self.entries.iter().min_by(|a, b| {
            a.cost
                .total_cost_usd
                .partial_cmp(&b.cost.total_cost_usd)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Filter entries by paradigm
    pub fn filter_by_paradigm(&self, paradigm: &ModelParadigm) -> Vec<&BenchmarkEntry> {
        self.entries
            .iter()
            .filter(|e| std::mem::discriminant(&e.paradigm) == std::mem::discriminant(paradigm))
            .collect()
    }

    /// Filter entries by device type
    pub fn filter_by_device_type<F>(&self, predicate: F) -> Vec<&BenchmarkEntry>
    where
        F: Fn(&ComputeDevice) -> bool,
    {
        self.entries
            .iter()
            .filter(|e| predicate(&e.device))
            .collect()
    }

    /// Get statistics summary
    pub fn statistics(&self) -> BenchmarkStatistics {
        if self.entries.is_empty() {
            return BenchmarkStatistics::default();
        }

        let qualities: Vec<f64> = self.entries.iter().map(|e| e.quality_score).collect();
        let costs: Vec<f64> = self.entries.iter().map(|e| e.cost.total_cost_usd).collect();
        let energies: Vec<f64> = self.entries.iter().map(|e| e.energy.joules_total).collect();

        BenchmarkStatistics {
            count: self.entries.len(),
            quality_min: qualities.iter().copied().fold(f64::INFINITY, f64::min),
            quality_max: qualities.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            quality_avg: qualities.iter().sum::<f64>() / qualities.len() as f64,
            cost_min: costs.iter().copied().fold(f64::INFINITY, f64::min),
            cost_max: costs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            cost_avg: costs.iter().sum::<f64>() / costs.len() as f64,
            energy_min: energies.iter().copied().fold(f64::INFINITY, f64::min),
            energy_max: energies.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            energy_avg: energies.iter().sum::<f64>() / energies.len() as f64,
            pareto_count: self.pareto_frontier().len(),
        }
    }

    /// Generate a comparison report
    pub fn comparison_report(&self) -> String {
        let stats = self.statistics();
        let frontier = self.pareto_frontier();

        let mut report = String::new();
        report.push_str(&format!(
            "=== Benchmark Report ({} entries) ===\n\n",
            stats.count
        ));

        report.push_str("Quality Scores:\n");
        report.push_str(&format!(
            "  Min: {:.4}  Max: {:.4}  Avg: {:.4}\n\n",
            stats.quality_min, stats.quality_max, stats.quality_avg
        ));

        report.push_str("Costs (USD):\n");
        report.push_str(&format!(
            "  Min: ${:.2}  Max: ${:.2}  Avg: ${:.2}\n\n",
            stats.cost_min, stats.cost_max, stats.cost_avg
        ));

        report.push_str(&format!("Pareto Frontier ({} entries):\n", frontier.len()));
        for entry in frontier.iter().take(5) {
            report.push_str(&format!(
                "  - {} ({}): quality={:.4}, cost=${:.2}\n",
                entry.run_id, entry.paradigm, entry.quality_score, entry.cost.total_cost_usd
            ));
        }

        if let Some(best) = self.most_efficient() {
            report.push_str(&format!(
                "\nMost Efficient: {} (score={:.2})\n",
                best.run_id,
                best.efficiency_score()
            ));
        }

        report
    }
}

/// Benchmark statistics summary
#[derive(Debug, Clone, Default)]
pub struct BenchmarkStatistics {
    /// Number of entries
    pub count: usize,
    /// Minimum quality score
    pub quality_min: f64,
    /// Maximum quality score
    pub quality_max: f64,
    /// Average quality score
    pub quality_avg: f64,
    /// Minimum cost
    pub cost_min: f64,
    /// Maximum cost
    pub cost_max: f64,
    /// Average cost
    pub cost_avg: f64,
    /// Minimum energy (joules)
    pub energy_min: f64,
    /// Maximum energy (joules)
    pub energy_max: f64,
    /// Average energy (joules)
    pub energy_avg: f64,
    /// Number of Pareto-optimal entries
    pub pareto_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::efficiency::{CpuInfo, SimdCapability};

    fn test_device() -> ComputeDevice {
        ComputeDevice::Cpu(CpuInfo::new(8, 16, SimdCapability::Avx2, "Test CPU"))
    }

    fn test_entry(run_id: &str, quality: f64, cost_usd: f64, joules: f64) -> BenchmarkEntry {
        BenchmarkEntry::new(
            run_id,
            ModelParadigm::DeepLearning,
            test_device(),
            quality,
            CostMetrics::new(cost_usd, 1.0, 1000, 10),
            EnergyMetrics::new(100.0, joules, 1000),
        )
    }

    #[test]
    fn test_benchmark_entry_new() {
        let entry = test_entry("run-001", 0.95, 10.0, 1000.0);

        assert_eq!(entry.run_id, "run-001");
        assert!((entry.quality_score - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_benchmark_entry_efficiency_score() {
        let entry = test_entry("run-001", 0.90, 10.0, 1000.0);
        assert!((entry.efficiency_score() - 0.09).abs() < 0.01);
    }

    #[test]
    fn test_benchmark_entry_dominates() {
        let a = test_entry("a", 0.95, 10.0, 1000.0);
        let b = test_entry("b", 0.90, 15.0, 1500.0);

        assert!(a.dominates(&b)); // Better quality, lower cost, lower energy
        assert!(!b.dominates(&a));
    }

    #[test]
    fn test_benchmark_entry_no_domination() {
        let a = test_entry("a", 0.95, 20.0, 1000.0); // Higher quality, higher cost
        let b = test_entry("b", 0.90, 10.0, 1500.0); // Lower quality, lower cost

        assert!(!a.dominates(&b));
        assert!(!b.dominates(&a));
    }

    #[test]
    fn test_benchmark_new() {
        let benchmark = CostPerformanceBenchmark::new();
        assert!(benchmark.is_empty());
        assert_eq!(benchmark.len(), 0);
    }

    #[test]
    fn test_benchmark_add() {
        let mut benchmark = CostPerformanceBenchmark::new();
        benchmark.add(test_entry("run-001", 0.95, 10.0, 1000.0));

        assert_eq!(benchmark.len(), 1);
        assert!(!benchmark.is_empty());
    }

    #[test]
    fn test_benchmark_pareto_frontier() {
        let mut benchmark = CostPerformanceBenchmark::new();

        // Entry A: High quality, high cost
        benchmark.add(test_entry("a", 0.98, 50.0, 5000.0));
        // Entry B: Medium quality, medium cost (Pareto optimal)
        benchmark.add(test_entry("b", 0.92, 20.0, 2000.0));
        // Entry C: Low quality, low cost (Pareto optimal)
        benchmark.add(test_entry("c", 0.85, 5.0, 500.0));
        // Entry D: Dominated by B (worse quality, same cost)
        benchmark.add(test_entry("d", 0.88, 20.0, 2000.0));

        let frontier = benchmark.pareto_frontier();
        assert_eq!(frontier.len(), 3);

        // D should not be in frontier (dominated by B)
        assert!(!frontier.iter().any(|e| e.run_id == "d"));
    }

    #[test]
    fn test_benchmark_best_for_budget() {
        let mut benchmark = CostPerformanceBenchmark::new();
        benchmark.add(test_entry("cheap", 0.85, 5.0, 500.0));
        benchmark.add(test_entry("mid", 0.92, 15.0, 1500.0));
        benchmark.add(test_entry("expensive", 0.98, 50.0, 5000.0));

        let best = benchmark.best_for_budget(20.0).unwrap();
        assert_eq!(best.run_id, "mid");

        let best = benchmark.best_for_budget(10.0).unwrap();
        assert_eq!(best.run_id, "cheap");

        let best = benchmark.best_for_budget(1.0);
        assert!(best.is_none());
    }

    #[test]
    fn test_benchmark_cheapest_for_quality() {
        let mut benchmark = CostPerformanceBenchmark::new();
        benchmark.add(test_entry("a", 0.95, 30.0, 3000.0));
        benchmark.add(test_entry("b", 0.92, 15.0, 1500.0));
        benchmark.add(test_entry("c", 0.90, 10.0, 1000.0));

        let cheapest = benchmark.cheapest_for_quality(0.90).unwrap();
        assert_eq!(cheapest.run_id, "c");

        let cheapest = benchmark.cheapest_for_quality(0.93).unwrap();
        assert_eq!(cheapest.run_id, "a");

        let cheapest = benchmark.cheapest_for_quality(0.99);
        assert!(cheapest.is_none());
    }

    #[test]
    fn test_benchmark_most_efficient() {
        let mut benchmark = CostPerformanceBenchmark::new();
        benchmark.add(test_entry("low_eff", 0.50, 50.0, 5000.0)); // 0.01
        benchmark.add(test_entry("high_eff", 0.90, 10.0, 1000.0)); // 0.09

        let most = benchmark.most_efficient().unwrap();
        assert_eq!(most.run_id, "high_eff");
    }

    #[test]
    fn test_benchmark_best_quality() {
        let mut benchmark = CostPerformanceBenchmark::new();
        benchmark.add(test_entry("a", 0.85, 10.0, 1000.0));
        benchmark.add(test_entry("b", 0.98, 50.0, 5000.0));
        benchmark.add(test_entry("c", 0.92, 20.0, 2000.0));

        let best = benchmark.best_quality().unwrap();
        assert_eq!(best.run_id, "b");
    }

    #[test]
    fn test_benchmark_cheapest() {
        let mut benchmark = CostPerformanceBenchmark::new();
        benchmark.add(test_entry("a", 0.85, 10.0, 1000.0));
        benchmark.add(test_entry("b", 0.98, 50.0, 5000.0));
        benchmark.add(test_entry("c", 0.92, 5.0, 500.0));

        let cheapest = benchmark.cheapest().unwrap();
        assert_eq!(cheapest.run_id, "c");
    }

    #[test]
    fn test_benchmark_filter_by_paradigm() {
        let mut benchmark = CostPerformanceBenchmark::new();

        let mut entry1 = test_entry("dl", 0.90, 10.0, 1000.0);
        entry1.paradigm = ModelParadigm::DeepLearning;
        benchmark.add(entry1);

        let mut entry2 = test_entry("lora", 0.88, 5.0, 500.0);
        entry2.paradigm = ModelParadigm::lora(64, 64.0);
        benchmark.add(entry2);

        let dl_entries = benchmark.filter_by_paradigm(&ModelParadigm::DeepLearning);
        assert_eq!(dl_entries.len(), 1);
        assert_eq!(dl_entries[0].run_id, "dl");
    }

    #[test]
    fn test_benchmark_filter_by_device() {
        let mut benchmark = CostPerformanceBenchmark::new();
        benchmark.add(test_entry("cpu-run", 0.90, 10.0, 1000.0));

        let cpu_entries =
            benchmark.filter_by_device_type(super::super::device::ComputeDevice::is_cpu);
        assert_eq!(cpu_entries.len(), 1);

        let gpu_entries =
            benchmark.filter_by_device_type(super::super::device::ComputeDevice::is_gpu);
        assert_eq!(gpu_entries.len(), 0);
    }

    #[test]
    fn test_benchmark_statistics() {
        let mut benchmark = CostPerformanceBenchmark::new();
        benchmark.add(test_entry("a", 0.80, 10.0, 1000.0));
        benchmark.add(test_entry("b", 0.90, 20.0, 2000.0));
        benchmark.add(test_entry("c", 1.00, 30.0, 3000.0));

        let stats = benchmark.statistics();
        assert_eq!(stats.count, 3);
        assert!((stats.quality_min - 0.80).abs() < f64::EPSILON);
        assert!((stats.quality_max - 1.00).abs() < f64::EPSILON);
        assert!((stats.quality_avg - 0.90).abs() < 0.01);
        assert!((stats.cost_min - 10.0).abs() < f64::EPSILON);
        assert!((stats.cost_max - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_benchmark_comparison_report() {
        let mut benchmark = CostPerformanceBenchmark::new();
        benchmark.add(test_entry("run-001", 0.90, 10.0, 1000.0));
        benchmark.add(test_entry("run-002", 0.95, 20.0, 2000.0));

        let report = benchmark.comparison_report();
        assert!(report.contains("Benchmark Report"));
        assert!(report.contains("2 entries"));
        assert!(report.contains("Quality Scores"));
        assert!(report.contains("Pareto Frontier"));
    }

    #[test]
    fn test_benchmark_empty_operations() {
        let benchmark = CostPerformanceBenchmark::new();

        assert!(benchmark.pareto_frontier().is_empty());
        assert!(benchmark.best_for_budget(100.0).is_none());
        assert!(benchmark.cheapest_for_quality(0.5).is_none());
        assert!(benchmark.most_efficient().is_none());
        assert!(benchmark.best_quality().is_none());
        assert!(benchmark.cheapest().is_none());
    }

    #[test]
    fn test_benchmark_serialization() {
        let mut benchmark = CostPerformanceBenchmark::new();
        benchmark.add(test_entry("run-001", 0.90, 10.0, 1000.0));

        let json = serde_json::to_string(&benchmark).unwrap();
        let parsed: CostPerformanceBenchmark = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed.entries[0].run_id, "run-001");
    }
}
