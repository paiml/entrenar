//! Type definitions for drift detection.

use std::collections::HashMap;

/// Statistical test for drift detection
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DriftTest {
    /// Kolmogorov-Smirnov test (continuous features)
    KS { threshold: f64 },
    /// Chi-square test (categorical features)
    ChiSquare { threshold: f64 },
    /// Population Stability Index (standard industry metric)
    PSI { threshold: f64 },
}

impl DriftTest {
    /// Get the name of this test
    pub fn name(&self) -> &'static str {
        match self {
            DriftTest::KS { .. } => "Kolmogorov-Smirnov",
            DriftTest::ChiSquare { .. } => "Chi-Square",
            DriftTest::PSI { .. } => "PSI",
        }
    }

    /// Get the threshold for this test
    pub fn threshold(&self) -> f64 {
        match self {
            DriftTest::KS { threshold }
            | DriftTest::ChiSquare { threshold }
            | DriftTest::PSI { threshold } => *threshold,
        }
    }
}

/// Severity levels for drift
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    /// No drift detected
    None,
    /// Warning: potential drift, log warning and continue
    Warning,
    /// Critical: stop inference or trigger retrain
    Critical,
}

/// Drift detection result
#[derive(Clone, Debug)]
pub struct DriftResult {
    /// Feature name or index
    pub feature: String,
    /// Test used for detection
    pub test: DriftTest,
    /// Test statistic value
    pub statistic: f64,
    /// P-value (for KS and Chi-sq) or PSI value
    pub p_value: f64,
    /// Whether drift was detected
    pub drifted: bool,
    /// Severity of the drift
    pub severity: Severity,
}

/// Summary of drift detection results
#[derive(Debug, Clone)]
pub struct DriftSummary {
    /// Total number of features checked
    pub total_features: usize,
    /// Number of features with detected drift
    pub drifted_features: usize,
    /// Number of warning-level drifts
    pub warnings: usize,
    /// Number of critical-level drifts
    pub critical: usize,
}

impl DriftSummary {
    /// Whether any critical drift was detected
    pub fn has_critical(&self) -> bool {
        self.critical > 0
    }

    /// Whether any drift was detected (warning or critical)
    pub fn has_drift(&self) -> bool {
        self.drifted_features > 0
    }

    /// Percentage of features that drifted
    pub fn drift_percentage(&self) -> f64 {
        if self.total_features == 0 {
            0.0
        } else {
            100.0 * self.drifted_features as f64 / self.total_features as f64
        }
    }
}

/// Callback type for drift events (Andon Cord)
pub type DriftCallback = Box<dyn Fn(&[DriftResult]) + Send + Sync>;

/// Baseline data for categorical features (histogram per feature)
pub type CategoricalBaseline = Vec<HashMap<usize, usize>>;
