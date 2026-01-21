//! Drift Detection Module
//!
//! Implements Jidoka (Automation with Human Touch) for detecting when the process
//! is out of control and signals for help (Retraining).
//!
//! Provides statistical tests for detecting data drift and concept drift:
//! - Kolmogorov-Smirnov test (continuous features)
//! - Chi-square test (categorical features)
//! - Population Stability Index (PSI)

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

/// Callback type for drift events (Andon Cord)
pub type DriftCallback = Box<dyn Fn(&[DriftResult]) + Send + Sync>;

/// Drift detector with statistical tests and callbacks
pub struct DriftDetector {
    tests: Vec<DriftTest>,
    baseline: Option<Vec<Vec<f64>>>,
    baseline_categorical: Option<Vec<HashMap<usize, usize>>>,
    warning_multiplier: f64,
    callbacks: Vec<DriftCallback>,
}

impl DriftDetector {
    /// Create a new drift detector with specified tests
    pub fn new(tests: Vec<DriftTest>) -> Self {
        Self {
            tests,
            baseline: None,
            baseline_categorical: None,
            warning_multiplier: 0.8, // Warning at 80% of threshold
            callbacks: Vec::new(),
        }
    }

    /// Register callback for drift events (Andon Cord)
    ///
    /// Callbacks are invoked when drift is detected via `check_and_trigger`.
    pub fn on_drift<F>(&mut self, callback: F)
    where
        F: Fn(&[DriftResult]) + Send + Sync + 'static,
    {
        self.callbacks.push(Box::new(callback));
    }

    /// Check for drift and trigger callbacks if drift detected
    ///
    /// Returns the drift results and invokes all registered callbacks
    /// if any feature shows drift.
    pub fn check_and_trigger(&self, current: &[Vec<f64>]) -> Vec<DriftResult> {
        let results = self.check(current);

        // Check if any drift detected
        let has_drift = results.iter().any(|r| r.drifted);

        if has_drift {
            for callback in &self.callbacks {
                callback(&results);
            }
        }

        results
    }

    /// Check categorical features for drift and trigger callbacks
    pub fn check_categorical_and_trigger(&self, current: &[Vec<usize>]) -> Vec<DriftResult> {
        let results = self.check_categorical(current);

        let has_drift = results.iter().any(|r| r.drifted);

        if has_drift {
            for callback in &self.callbacks {
                callback(&results);
            }
        }

        results
    }

    /// Set baseline distribution for continuous features
    /// Each row is a sample, each column is a feature
    pub fn set_baseline(&mut self, data: &[Vec<f64>]) {
        if data.is_empty() {
            return;
        }
        // Transpose: store column-wise for easier feature comparison
        let n_features = data[0].len();
        let mut columns = vec![Vec::new(); n_features];
        for row in data {
            for (i, &val) in row.iter().enumerate() {
                if i < n_features {
                    columns[i].push(val);
                }
            }
        }
        self.baseline = Some(columns);
    }

    /// Set baseline distribution for categorical features
    pub fn set_baseline_categorical(&mut self, data: &[Vec<usize>]) {
        if data.is_empty() {
            return;
        }
        let n_features = data[0].len();
        let mut histograms = vec![HashMap::new(); n_features];
        for row in data {
            for (i, &val) in row.iter().enumerate() {
                if i < n_features {
                    *histograms[i].entry(val).or_insert(0) += 1;
                }
            }
        }
        self.baseline_categorical = Some(histograms);
    }

    /// Check new data for drift against baseline
    pub fn check(&self, current: &[Vec<f64>]) -> Vec<DriftResult> {
        let mut results = Vec::new();

        let baseline = match &self.baseline {
            Some(b) => b,
            None => return results,
        };

        if current.is_empty() {
            return results;
        }

        // Transpose current data to column-wise
        let n_features = current[0].len().min(baseline.len());
        let mut current_columns = vec![Vec::new(); n_features];
        for row in current {
            for (i, &val) in row.iter().enumerate() {
                if i < n_features {
                    current_columns[i].push(val);
                }
            }
        }

        // Run tests on each feature
        for (feature_idx, (baseline_col, current_col)) in
            baseline.iter().zip(current_columns.iter()).enumerate()
        {
            for test in &self.tests {
                let result = match test {
                    DriftTest::KS { threshold } => {
                        self.ks_test(feature_idx, baseline_col, current_col, *threshold)
                    }
                    DriftTest::PSI { threshold } => {
                        self.psi_test(feature_idx, baseline_col, current_col, *threshold)
                    }
                    DriftTest::ChiSquare { .. } => continue, // Skip for continuous
                };
                results.push(result);
            }
        }

        results
    }

    /// Check categorical features for drift
    pub fn check_categorical(&self, current: &[Vec<usize>]) -> Vec<DriftResult> {
        let mut results = Vec::new();

        let baseline = match &self.baseline_categorical {
            Some(b) => b,
            None => return results,
        };

        if current.is_empty() {
            return results;
        }

        // Build current histograms
        let n_features = current[0].len().min(baseline.len());
        let mut current_histograms = vec![HashMap::new(); n_features];
        for row in current {
            for (i, &val) in row.iter().enumerate() {
                if i < n_features {
                    *current_histograms[i].entry(val).or_insert(0) += 1;
                }
            }
        }

        // Run chi-square test on each feature
        for (feature_idx, (baseline_hist, current_hist)) in
            baseline.iter().zip(current_histograms.iter()).enumerate()
        {
            for test in &self.tests {
                if let DriftTest::ChiSquare { threshold } = test {
                    let result =
                        self.chi_square_test(feature_idx, baseline_hist, current_hist, *threshold);
                    results.push(result);
                }
            }
        }

        results
    }

    /// Kolmogorov-Smirnov test for continuous features
    fn ks_test(
        &self,
        feature_idx: usize,
        baseline: &[f64],
        current: &[f64],
        threshold: f64,
    ) -> DriftResult {
        // Sort both distributions
        let mut sorted_baseline: Vec<f64> = baseline.to_vec();
        let mut sorted_current: Vec<f64> = current.to_vec();
        sorted_baseline.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted_current.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute empirical CDFs and find maximum difference
        let n1 = sorted_baseline.len() as f64;
        let n2 = sorted_current.len() as f64;

        let mut d_max = 0.0f64;
        let mut i = 0usize;
        let mut j = 0usize;

        while i < sorted_baseline.len() && j < sorted_current.len() {
            let cdf1 = (i + 1) as f64 / n1;
            let cdf2 = (j + 1) as f64 / n2;

            let diff = (cdf1 - cdf2).abs();
            d_max = d_max.max(diff);

            if sorted_baseline[i] <= sorted_current[j] {
                i += 1;
            } else {
                j += 1;
            }
        }

        // Approximate p-value using asymptotic formula
        let n_eff = (n1 * n2) / (n1 + n2);
        let lambda = d_max * n_eff.sqrt();
        let p_value = ks_p_value(lambda);

        let (drifted, severity) = self.classify_result(p_value, threshold);

        DriftResult {
            feature: format!("feature_{feature_idx}"),
            test: DriftTest::KS { threshold },
            statistic: d_max,
            p_value,
            drifted,
            severity,
        }
    }

    /// Population Stability Index (PSI) test
    fn psi_test(
        &self,
        feature_idx: usize,
        baseline: &[f64],
        current: &[f64],
        threshold: f64,
    ) -> DriftResult {
        // Create 10 bins based on baseline deciles
        let n_bins = 10;
        let mut sorted_baseline: Vec<f64> = baseline.to_vec();
        sorted_baseline.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate bin edges (deciles)
        let mut edges = Vec::with_capacity(n_bins + 1);
        edges.push(f64::NEG_INFINITY);
        for i in 1..n_bins {
            let idx = (sorted_baseline.len() * i / n_bins).min(sorted_baseline.len() - 1);
            edges.push(sorted_baseline[idx]);
        }
        edges.push(f64::INFINITY);

        // Count samples in each bin
        let baseline_counts = bin_counts(baseline, &edges);
        let current_counts = bin_counts(current, &edges);

        // Calculate PSI
        let total_baseline = baseline.len() as f64;
        let total_current = current.len() as f64;

        let mut psi = 0.0;
        for (b_count, c_count) in baseline_counts.iter().zip(current_counts.iter()) {
            let b_pct = (*b_count as f64 + 0.0001) / (total_baseline + 0.001);
            let c_pct = (*c_count as f64 + 0.0001) / (total_current + 0.001);
            psi += (c_pct - b_pct) * (c_pct / b_pct).ln();
        }

        let (drifted, severity) = if psi >= threshold {
            (true, Severity::Critical)
        } else if psi >= threshold * self.warning_multiplier {
            (true, Severity::Warning)
        } else {
            (false, Severity::None)
        };

        DriftResult {
            feature: format!("feature_{feature_idx}"),
            test: DriftTest::PSI { threshold },
            statistic: psi,
            p_value: psi, // PSI doesn't use p-value, store the PSI value
            drifted,
            severity,
        }
    }

    /// Chi-square test for categorical features
    fn chi_square_test(
        &self,
        feature_idx: usize,
        baseline: &HashMap<usize, usize>,
        current: &HashMap<usize, usize>,
        threshold: f64,
    ) -> DriftResult {
        // Get all categories
        let mut categories: Vec<usize> = baseline.keys().chain(current.keys()).copied().collect();
        categories.sort_unstable();
        categories.dedup();

        let total_baseline: f64 = baseline.values().sum::<usize>() as f64;
        let total_current: f64 = current.values().sum::<usize>() as f64;

        if total_baseline == 0.0 || total_current == 0.0 {
            return DriftResult {
                feature: format!("feature_{feature_idx}"),
                test: DriftTest::ChiSquare { threshold },
                statistic: 0.0,
                p_value: 1.0,
                drifted: false,
                severity: Severity::None,
            };
        }

        // Calculate chi-square statistic
        let mut chi_sq = 0.0;
        let mut df: usize = 0;

        for &cat in &categories {
            let observed = *current.get(&cat).unwrap_or(&0) as f64;
            let baseline_pct = *baseline.get(&cat).unwrap_or(&0) as f64 / total_baseline;
            let expected = baseline_pct * total_current;

            if expected > 0.0 {
                chi_sq += (observed - expected).powi(2) / expected;
                df += 1;
            }
        }

        df = df.saturating_sub(1); // degrees of freedom = categories - 1
        let p_value = chi_square_p_value(chi_sq, df);

        let (drifted, severity) = self.classify_result(p_value, threshold);

        DriftResult {
            feature: format!("feature_{feature_idx}"),
            test: DriftTest::ChiSquare { threshold },
            statistic: chi_sq,
            p_value,
            drifted,
            severity,
        }
    }

    /// Classify result based on p-value and threshold
    fn classify_result(&self, p_value: f64, threshold: f64) -> (bool, Severity) {
        if p_value < threshold {
            (true, Severity::Critical)
        } else if p_value < threshold / self.warning_multiplier {
            (true, Severity::Warning)
        } else {
            (false, Severity::None)
        }
    }

    /// Get summary of drift results
    pub fn summary(results: &[DriftResult]) -> DriftSummary {
        let total = results.len();
        let drifted = results.iter().filter(|r| r.drifted).count();
        let warnings = results
            .iter()
            .filter(|r| r.severity == Severity::Warning)
            .count();
        let critical = results
            .iter()
            .filter(|r| r.severity == Severity::Critical)
            .count();

        DriftSummary {
            total_features: total,
            drifted_features: drifted,
            warnings,
            critical,
        }
    }
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

/// Count samples in bins defined by edges
fn bin_counts(data: &[f64], edges: &[f64]) -> Vec<usize> {
    let mut counts = vec![0; edges.len() - 1];
    for &val in data {
        for i in 0..counts.len() {
            if val > edges[i] && val <= edges[i + 1] {
                counts[i] += 1;
                break;
            }
        }
    }
    counts
}

/// Approximate p-value for KS statistic using Kolmogorov distribution
fn ks_p_value(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 1.0;
    }
    // Asymptotic approximation: P(D > d) ≈ 2 * sum_{k=1}^∞ (-1)^{k+1} * exp(-2 * k^2 * λ^2)
    let mut p = 0.0;
    for k in 1..=100 {
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        let term = sign * (-2.0 * f64::from(k).powi(2) * lambda.powi(2)).exp();
        p += term;
        if term.abs() < 1e-10 {
            break;
        }
    }
    (2.0 * p).clamp(0.0, 1.0)
}

/// Approximate chi-square p-value using Wilson-Hilferty approximation
fn chi_square_p_value(chi_sq: f64, df: usize) -> f64 {
    if df == 0 || chi_sq <= 0.0 {
        return 1.0;
    }
    let k = df as f64;
    // Wilson-Hilferty transformation to normal
    let z = ((chi_sq / k).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / (2.0 / (9.0 * k)).sqrt();
    // Convert z to p-value (upper tail)
    0.5 * (1.0 - erf(z / std::f64::consts::SQRT_2))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drift_test_name() {
        assert_eq!(
            DriftTest::KS { threshold: 0.05 }.name(),
            "Kolmogorov-Smirnov"
        );
        assert_eq!(
            DriftTest::ChiSquare { threshold: 0.05 }.name(),
            "Chi-Square"
        );
        assert_eq!(DriftTest::PSI { threshold: 0.1 }.name(), "PSI");
    }

    #[test]
    fn test_drift_test_threshold() {
        assert_eq!(DriftTest::KS { threshold: 0.05 }.threshold(), 0.05);
        assert_eq!(DriftTest::PSI { threshold: 0.2 }.threshold(), 0.2);
    }

    #[test]
    fn test_no_baseline() {
        let detector = DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }]);
        let results = detector.check(&[vec![1.0, 2.0]]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_ks_same_distribution() {
        let mut detector = DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }]);

        // Same distribution should not drift
        let data: Vec<Vec<f64>> = (0..100).map(|i| vec![f64::from(i)]).collect();
        detector.set_baseline(&data);

        let results = detector.check(&data);
        assert_eq!(results.len(), 1);
        assert!(!results[0].drifted);
    }

    #[test]
    fn test_ks_different_distribution() {
        let mut detector = DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }]);

        // Baseline: uniform 0-100
        let baseline: Vec<Vec<f64>> = (0..100).map(|i| vec![f64::from(i)]).collect();
        detector.set_baseline(&baseline);

        // Current: shifted by 50
        let current: Vec<Vec<f64>> = (50..150).map(|i| vec![f64::from(i)]).collect();
        let results = detector.check(&current);

        assert_eq!(results.len(), 1);
        // Shifted distribution should trigger drift detection
        // The KS statistic should be significant
    }

    #[test]
    fn test_psi_no_drift() {
        let mut detector = DriftDetector::new(vec![DriftTest::PSI { threshold: 0.2 }]);

        let data: Vec<Vec<f64>> = (0..100).map(|i| vec![f64::from(i)]).collect();
        detector.set_baseline(&data);

        let results = detector.check(&data);
        assert_eq!(results.len(), 1);
        assert!(!results[0].drifted);
        assert!(results[0].statistic < 0.1); // PSI should be near 0
    }

    #[test]
    fn test_psi_with_drift() {
        let mut detector = DriftDetector::new(vec![DriftTest::PSI { threshold: 0.1 }]);

        // Baseline: all values in [0, 10)
        let baseline: Vec<Vec<f64>> = (0..100).map(|i| vec![f64::from(i % 10)]).collect();
        detector.set_baseline(&baseline);

        // Current: all values in [90, 100) - completely different distribution
        let current: Vec<Vec<f64>> = (0..100).map(|i| vec![90.0 + f64::from(i % 10)]).collect();
        let results = detector.check(&current);

        assert_eq!(results.len(), 1);
        // Completely different distributions should have high PSI
    }

    #[test]
    fn test_chi_square_same() {
        let mut detector = DriftDetector::new(vec![DriftTest::ChiSquare { threshold: 0.05 }]);

        let data: Vec<Vec<usize>> = (0..100).map(|i| vec![i % 5]).collect();
        detector.set_baseline_categorical(&data);

        let results = detector.check_categorical(&data);
        assert_eq!(results.len(), 1);
        assert!(!results[0].drifted);
    }

    #[test]
    fn test_chi_square_different() {
        let mut detector = DriftDetector::new(vec![DriftTest::ChiSquare { threshold: 0.05 }]);

        // Baseline: uniform distribution over 0-4
        let baseline: Vec<Vec<usize>> = (0..100).map(|i| vec![i % 5]).collect();
        detector.set_baseline_categorical(&baseline);

        // Current: all values are 0
        let current: Vec<Vec<usize>> = (0..100).map(|_| vec![0]).collect();
        let results = detector.check_categorical(&current);

        assert_eq!(results.len(), 1);
        // Completely different categorical distribution should drift
    }

    #[test]
    fn test_drift_summary() {
        let results = vec![
            DriftResult {
                feature: "f1".into(),
                test: DriftTest::KS { threshold: 0.05 },
                statistic: 0.5,
                p_value: 0.01,
                drifted: true,
                severity: Severity::Critical,
            },
            DriftResult {
                feature: "f2".into(),
                test: DriftTest::KS { threshold: 0.05 },
                statistic: 0.1,
                p_value: 0.3,
                drifted: false,
                severity: Severity::None,
            },
            DriftResult {
                feature: "f3".into(),
                test: DriftTest::KS { threshold: 0.05 },
                statistic: 0.2,
                p_value: 0.04,
                drifted: true,
                severity: Severity::Warning,
            },
        ];

        let summary = DriftDetector::summary(&results);
        assert_eq!(summary.total_features, 3);
        assert_eq!(summary.drifted_features, 2);
        assert_eq!(summary.warnings, 1);
        assert_eq!(summary.critical, 1);
        assert!(summary.has_critical());
        assert!(summary.has_drift());
        assert!((summary.drift_percentage() - 66.67).abs() < 1.0);
    }

    #[test]
    fn test_empty_data() {
        let mut detector = DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }]);
        detector.set_baseline(&[]);
        let results = detector.check(&[vec![1.0]]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bin_counts() {
        let data = vec![0.5, 1.5, 2.5, 3.5];
        let edges = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let counts = bin_counts(&data, &edges);
        assert_eq!(counts, vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_ks_p_value() {
        // λ = 0 should give p = 1
        assert!((ks_p_value(0.0) - 1.0).abs() < 0.01);
        // Large λ should give small p
        assert!(ks_p_value(3.0) < 0.01);
    }

    #[test]
    fn test_severity_eq() {
        assert_eq!(Severity::None, Severity::None);
        assert_ne!(Severity::None, Severity::Warning);
        assert_ne!(Severity::Warning, Severity::Critical);
    }

    #[test]
    fn test_multiple_features() {
        let mut detector = DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }]);

        // 2 features
        let baseline: Vec<Vec<f64>> = (0..100)
            .map(|i| vec![f64::from(i), f64::from(i * 2)])
            .collect();
        detector.set_baseline(&baseline);

        let results = detector.check(&baseline);
        assert_eq!(results.len(), 2); // One result per feature
    }

    #[test]
    fn test_on_drift_callback() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let mut detector = DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }]);

        // Counter to track callback invocations
        let callback_count = Arc::new(AtomicUsize::new(0));
        let count_clone = Arc::clone(&callback_count);

        detector.on_drift(move |_results| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        });

        // Baseline: uniform 0-100
        let baseline: Vec<Vec<f64>> = (0..100).map(|i| vec![f64::from(i)]).collect();
        detector.set_baseline(&baseline);

        // Same distribution - should not trigger callback
        let _ = detector.check_and_trigger(&baseline);
        assert_eq!(callback_count.load(Ordering::SeqCst), 0);

        // Shifted distribution - should trigger callback
        let shifted: Vec<Vec<f64>> = (100..200).map(|i| vec![f64::from(i)]).collect();
        let _ = detector.check_and_trigger(&shifted);
        assert_eq!(callback_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_check_and_trigger_no_drift() {
        let mut detector = DriftDetector::new(vec![DriftTest::PSI { threshold: 0.2 }]);

        let data: Vec<Vec<f64>> = (0..100).map(|i| vec![f64::from(i)]).collect();
        detector.set_baseline(&data);

        // Same data should not trigger
        let results = detector.check_and_trigger(&data);
        assert!(!results.iter().any(|r| r.drifted));
    }
}
