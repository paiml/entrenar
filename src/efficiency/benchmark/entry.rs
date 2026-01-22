//! Benchmark entry type

use crate::efficiency::{ComputeDevice, CostMetrics, EnergyMetrics, ModelParadigm};
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
