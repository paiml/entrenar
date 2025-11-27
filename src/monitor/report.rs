//! Hansei (反省) Post-Training Report Generator
//!
//! Toyota Way principle: Reflection and continuous improvement through
//! systematic analysis of training outcomes.
//!
//! Reference: Liker, J.K. (2004). The Toyota Way: 14 Management Principles.

use super::{Metric, MetricStats, MetricsCollector};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

/// Severity level for identified issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl std::fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IssueSeverity::Info => write!(f, "INFO"),
            IssueSeverity::Warning => write!(f, "WARNING"),
            IssueSeverity::Error => write!(f, "ERROR"),
            IssueSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// An identified issue from post-training analysis
#[derive(Debug, Clone)]
pub struct TrainingIssue {
    pub severity: IssueSeverity,
    pub category: String,
    pub description: String,
    pub recommendation: String,
}

/// Post-training analysis report (Hansei)
#[derive(Debug, Clone)]
pub struct PostTrainingReport {
    pub training_id: String,
    pub duration_secs: f64,
    pub total_steps: u64,
    pub final_metrics: HashMap<Metric, f64>,
    pub metric_summaries: HashMap<Metric, MetricSummary>,
    pub issues: Vec<TrainingIssue>,
    pub recommendations: Vec<String>,
}

/// Summary statistics for a metric over the training run
#[derive(Debug, Clone)]
pub struct MetricSummary {
    pub initial: f64,
    pub final_value: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub trend: Trend,
}

/// Trend direction for a metric
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend {
    Improving,
    Degrading,
    Stable,
    Oscillating,
}

impl std::fmt::Display for Trend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Trend::Improving => write!(f, "↑ Improving"),
            Trend::Degrading => write!(f, "↓ Degrading"),
            Trend::Stable => write!(f, "→ Stable"),
            Trend::Oscillating => write!(f, "~ Oscillating"),
        }
    }
}

/// Hansei report generator
pub struct HanseiAnalyzer {
    /// Threshold for loss increase to trigger warning
    pub loss_increase_threshold: f64,
    /// Threshold for gradient norm to indicate explosion
    pub gradient_explosion_threshold: f64,
    /// Threshold for gradient norm to indicate vanishing
    pub gradient_vanishing_threshold: f64,
    /// Minimum expected accuracy improvement
    pub min_accuracy_improvement: f64,
}

impl Default for HanseiAnalyzer {
    fn default() -> Self {
        Self {
            loss_increase_threshold: 0.1,      // 10% increase
            gradient_explosion_threshold: 100.0,
            gradient_vanishing_threshold: 1e-7,
            min_accuracy_improvement: 0.01,    // 1% improvement
        }
    }
}

impl HanseiAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze a completed training run and generate a report
    pub fn analyze(
        &self,
        training_id: &str,
        collector: &MetricsCollector,
        duration_secs: f64,
    ) -> PostTrainingReport {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();
        let mut metric_summaries = HashMap::new();
        let mut final_metrics = HashMap::new();

        let summary = collector.summary();
        let total_steps = summary.values().map(|s| s.count).sum::<usize>() as u64;

        // Analyze each metric
        for (metric, stats) in &summary {
            let metric_summary = self.analyze_metric(metric, stats);
            metric_summaries.insert(metric.clone(), metric_summary.clone());
            final_metrics.insert(metric.clone(), stats.mean);

            // Check for issues based on metric type
            self.check_metric_issues(metric, &metric_summary, stats, &mut issues);
        }

        // Generate recommendations based on issues
        self.generate_recommendations(&issues, &mut recommendations);

        // Check for missing expected metrics
        self.check_missing_metrics(&summary, &mut issues);

        // Sort issues by severity (critical first)
        issues.sort_by(|a, b| b.severity.cmp(&a.severity));

        PostTrainingReport {
            training_id: training_id.to_string(),
            duration_secs,
            total_steps,
            final_metrics,
            metric_summaries,
            issues,
            recommendations,
        }
    }

    fn analyze_metric(&self, metric: &Metric, stats: &MetricStats) -> MetricSummary {
        // Determine trend based on metric type and statistics
        let trend = self.determine_trend(metric, stats);

        MetricSummary {
            initial: stats.min, // Approximation - would need history for actual initial
            final_value: stats.mean, // Approximation - would need last value
            min: stats.min,
            max: stats.max,
            mean: stats.mean,
            std_dev: stats.std,
            trend,
        }
    }

    fn determine_trend(&self, metric: &Metric, stats: &MetricStats) -> Trend {
        let cv = if stats.mean.abs() > 1e-10 {
            stats.std / stats.mean.abs()
        } else {
            0.0
        };

        // High coefficient of variation indicates oscillation
        if cv > 0.5 {
            return Trend::Oscillating;
        }

        // For loss, lower is better
        // For accuracy, higher is better
        match metric {
            Metric::Loss => {
                if stats.max - stats.min < stats.std * 0.5 {
                    Trend::Stable
                } else if stats.mean < (stats.min + stats.max) / 2.0 {
                    Trend::Improving
                } else {
                    Trend::Degrading
                }
            }
            Metric::Accuracy => {
                if stats.max - stats.min < stats.std * 0.5 {
                    Trend::Stable
                } else if stats.mean > (stats.min + stats.max) / 2.0 {
                    Trend::Improving
                } else {
                    Trend::Degrading
                }
            }
            Metric::GradientNorm => {
                if cv < 0.2 {
                    Trend::Stable
                } else {
                    Trend::Oscillating
                }
            }
            _ => Trend::Stable,
        }
    }

    fn check_metric_issues(
        &self,
        metric: &Metric,
        summary: &MetricSummary,
        stats: &MetricStats,
        issues: &mut Vec<TrainingIssue>,
    ) {
        match metric {
            Metric::Loss => {
                // Check for NaN/Inf
                if stats.has_nan {
                    issues.push(TrainingIssue {
                        severity: IssueSeverity::Critical,
                        category: "Numerical Stability".to_string(),
                        description: "NaN values detected in loss".to_string(),
                        recommendation: "Reduce learning rate, add gradient clipping, or check data preprocessing".to_string(),
                    });
                }
                if stats.has_inf {
                    issues.push(TrainingIssue {
                        severity: IssueSeverity::Critical,
                        category: "Numerical Stability".to_string(),
                        description: "Infinity values detected in loss".to_string(),
                        recommendation: "Check for division by zero, reduce learning rate".to_string(),
                    });
                }
                // Check for loss not decreasing
                if summary.trend == Trend::Degrading {
                    issues.push(TrainingIssue {
                        severity: IssueSeverity::Warning,
                        category: "Convergence".to_string(),
                        description: "Loss appears to be increasing over training".to_string(),
                        recommendation: "Consider reducing learning rate or checking data quality".to_string(),
                    });
                }
                // Check for high variance (oscillating loss)
                if summary.trend == Trend::Oscillating {
                    issues.push(TrainingIssue {
                        severity: IssueSeverity::Warning,
                        category: "Stability".to_string(),
                        description: "Loss is oscillating significantly".to_string(),
                        recommendation: "Reduce learning rate or increase batch size".to_string(),
                    });
                }
            }
            Metric::Accuracy => {
                // Check for low final accuracy
                if summary.final_value < 0.5 && stats.count > 100 {
                    issues.push(TrainingIssue {
                        severity: IssueSeverity::Warning,
                        category: "Performance".to_string(),
                        description: format!("Final accuracy is low: {:.2}%", summary.final_value * 100.0),
                        recommendation: "Consider model architecture changes or hyperparameter tuning".to_string(),
                    });
                }
                // Check for no improvement
                if summary.trend == Trend::Stable && summary.max - summary.min < self.min_accuracy_improvement {
                    issues.push(TrainingIssue {
                        severity: IssueSeverity::Info,
                        category: "Convergence".to_string(),
                        description: "Accuracy shows minimal improvement".to_string(),
                        recommendation: "Model may have converged or may be stuck in local minimum".to_string(),
                    });
                }
            }
            Metric::GradientNorm => {
                // Check for gradient explosion
                if stats.max > self.gradient_explosion_threshold {
                    issues.push(TrainingIssue {
                        severity: IssueSeverity::Error,
                        category: "Gradient Health".to_string(),
                        description: format!("Gradient explosion detected: max norm = {:.2e}", stats.max),
                        recommendation: "Enable gradient clipping (e.g., max_norm=1.0)".to_string(),
                    });
                }
                // Check for vanishing gradients
                if stats.mean < self.gradient_vanishing_threshold && stats.count > 10 {
                    issues.push(TrainingIssue {
                        severity: IssueSeverity::Warning,
                        category: "Gradient Health".to_string(),
                        description: format!("Possible vanishing gradients: mean norm = {:.2e}", stats.mean),
                        recommendation: "Consider using residual connections or different activation functions".to_string(),
                    });
                }
            }
            Metric::LearningRate => {
                // Check if LR is too high based on loss behavior
                if summary.std_dev > summary.mean * 0.5 {
                    issues.push(TrainingIssue {
                        severity: IssueSeverity::Info,
                        category: "Hyperparameters".to_string(),
                        description: "Learning rate schedule shows high variance".to_string(),
                        recommendation: "Review learning rate schedule configuration".to_string(),
                    });
                }
            }
            _ => {}
        }
    }

    fn check_missing_metrics(
        &self,
        metrics: &HashMap<Metric, MetricStats>,
        issues: &mut Vec<TrainingIssue>,
    ) {
        // Check for essential metrics
        if !metrics.contains_key(&Metric::Loss) {
            issues.push(TrainingIssue {
                severity: IssueSeverity::Warning,
                category: "Observability".to_string(),
                description: "No loss metric recorded".to_string(),
                recommendation: "Ensure loss is being tracked for proper monitoring".to_string(),
            });
        }
    }

    fn generate_recommendations(&self, issues: &[TrainingIssue], recommendations: &mut Vec<String>) {
        let has_numerical_issues = issues.iter().any(|i| i.category == "Numerical Stability");
        let has_gradient_issues = issues.iter().any(|i| i.category == "Gradient Health");
        let has_convergence_issues = issues.iter().any(|i| i.category == "Convergence");

        if has_numerical_issues {
            recommendations.push("Priority 1: Address numerical stability before continuing training".to_string());
        }

        if has_gradient_issues {
            recommendations.push("Enable gradient clipping in optimizer configuration".to_string());
        }

        if has_convergence_issues {
            recommendations.push("Consider hyperparameter search for learning rate and batch size".to_string());
        }

        if issues.is_empty() {
            recommendations.push("Training completed without detected issues. Consider running validation tests.".to_string());
        }
    }

    /// Generate a human-readable report
    pub fn format_report(&self, report: &PostTrainingReport) -> String {
        let mut output = String::new();

        writeln!(output, "═══════════════════════════════════════════════════════════════").unwrap();
        writeln!(output, "                    HANSEI POST-TRAINING REPORT                 ").unwrap();
        writeln!(output, "═══════════════════════════════════════════════════════════════").unwrap();
        writeln!(output).unwrap();
        writeln!(output, "Training ID: {}", report.training_id).unwrap();
        writeln!(output, "Duration: {:.2}s", report.duration_secs).unwrap();
        writeln!(output, "Total Steps: {}", report.total_steps).unwrap();
        writeln!(output).unwrap();

        // Metric summaries
        writeln!(output, "─── Metric Summaries ───────────────────────────────────────────").unwrap();
        for (metric_type, summary) in &report.metric_summaries {
            writeln!(output, "\n{:?}:", metric_type).unwrap();
            writeln!(output, "  Mean: {:.6}  Std: {:.6}", summary.mean, summary.std_dev).unwrap();
            writeln!(output, "  Min: {:.6}   Max: {:.6}", summary.min, summary.max).unwrap();
            writeln!(output, "  Trend: {}", summary.trend).unwrap();
        }
        writeln!(output).unwrap();

        // Issues
        if !report.issues.is_empty() {
            writeln!(output, "─── Issues Detected ────────────────────────────────────────────").unwrap();
            for issue in &report.issues {
                writeln!(output, "\n[{}] {}", issue.severity, issue.category).unwrap();
                writeln!(output, "  {}", issue.description).unwrap();
                writeln!(output, "  → {}", issue.recommendation).unwrap();
            }
            writeln!(output).unwrap();
        }

        // Recommendations
        writeln!(output, "─── Recommendations ────────────────────────────────────────────").unwrap();
        for (i, rec) in report.recommendations.iter().enumerate() {
            writeln!(output, "{}. {}", i + 1, rec).unwrap();
        }
        writeln!(output).unwrap();

        writeln!(output, "═══════════════════════════════════════════════════════════════").unwrap();

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hansei_analyzer_default() {
        let analyzer = HanseiAnalyzer::default();
        assert_eq!(analyzer.loss_increase_threshold, 0.1);
        assert_eq!(analyzer.gradient_explosion_threshold, 100.0);
    }

    #[test]
    fn test_analyze_healthy_training() {
        let analyzer = HanseiAnalyzer::new();
        let mut collector = MetricsCollector::new();

        // Simulate healthy training: decreasing loss, increasing accuracy
        for i in 0..100 {
            let loss = 1.0 - (i as f64 * 0.008); // 1.0 -> 0.2
            let accuracy = 0.5 + (i as f64 * 0.004); // 0.5 -> 0.9
            collector.record(Metric::Loss, loss);
            collector.record(Metric::Accuracy, accuracy);
        }

        let report = analyzer.analyze("test-run-1", &collector, 120.0);

        assert_eq!(report.training_id, "test-run-1");
        assert_eq!(report.total_steps, 200); // 100 loss + 100 accuracy
        assert!(report.duration_secs == 120.0);
    }

    #[test]
    fn test_detect_nan_loss() {
        let analyzer = HanseiAnalyzer::new();
        let mut collector = MetricsCollector::new();

        collector.record(Metric::Loss, 1.0);
        collector.record(Metric::Loss, f64::NAN);

        let report = analyzer.analyze("nan-test", &collector, 10.0);

        let critical_issues: Vec<_> = report
            .issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Critical)
            .collect();

        assert!(!critical_issues.is_empty());
        assert!(critical_issues[0].description.contains("NaN"));
    }

    #[test]
    fn test_detect_inf_loss() {
        let analyzer = HanseiAnalyzer::new();
        let mut collector = MetricsCollector::new();

        collector.record(Metric::Loss, 1.0);
        collector.record(Metric::Loss, f64::INFINITY);

        let report = analyzer.analyze("inf-test", &collector, 10.0);

        let critical_issues: Vec<_> = report
            .issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Critical)
            .collect();

        assert!(!critical_issues.is_empty());
        assert!(critical_issues[0].description.contains("Infinity"));
    }

    #[test]
    fn test_detect_gradient_explosion() {
        let analyzer = HanseiAnalyzer::new();
        let mut collector = MetricsCollector::new();

        collector.record(Metric::GradientNorm, 1.0);
        collector.record(Metric::GradientNorm, 500.0); // Explosion!

        let report = analyzer.analyze("grad-explosion", &collector, 10.0);

        let gradient_issues: Vec<_> = report
            .issues
            .iter()
            .filter(|i| i.category == "Gradient Health")
            .collect();

        assert!(!gradient_issues.is_empty());
        assert!(gradient_issues[0].description.contains("explosion"));
    }

    #[test]
    fn test_detect_vanishing_gradients() {
        let analyzer = HanseiAnalyzer::new();
        let mut collector = MetricsCollector::new();

        // Very small gradients
        for _ in 0..20 {
            collector.record(Metric::GradientNorm, 1e-10);
        }

        let report = analyzer.analyze("vanishing-grad", &collector, 10.0);

        let gradient_issues: Vec<_> = report
            .issues
            .iter()
            .filter(|i| i.description.contains("vanishing"))
            .collect();

        assert!(!gradient_issues.is_empty());
    }

    #[test]
    fn test_missing_loss_metric() {
        let analyzer = HanseiAnalyzer::new();
        let mut collector = MetricsCollector::new();

        // Only record accuracy, no loss
        collector.record(Metric::Accuracy, 0.5);

        let report = analyzer.analyze("no-loss", &collector, 10.0);

        let observability_issues: Vec<_> = report
            .issues
            .iter()
            .filter(|i| i.category == "Observability")
            .collect();

        assert!(!observability_issues.is_empty());
    }

    #[test]
    fn test_format_report() {
        let analyzer = HanseiAnalyzer::new();
        let mut collector = MetricsCollector::new();

        collector.record(Metric::Loss, 1.0);
        collector.record(Metric::Loss, 0.5);
        collector.record(Metric::Accuracy, 0.8);

        let report = analyzer.analyze("format-test", &collector, 60.0);
        let formatted = analyzer.format_report(&report);

        assert!(formatted.contains("HANSEI POST-TRAINING REPORT"));
        assert!(formatted.contains("format-test"));
        assert!(formatted.contains("Duration: 60.00s"));
    }

    #[test]
    fn test_trend_detection_improving_loss() {
        let analyzer = HanseiAnalyzer::new();
        let mut collector = MetricsCollector::new();

        // Loss that is biased toward lower values (mean < midpoint)
        // Use values with low CV to avoid oscillation detection
        // Range: 1.0 to 2.0, mean around 1.2 (below midpoint of 1.5)
        for _ in 0..40 {
            collector.record(Metric::Loss, 1.0);
        }
        for _ in 0..10 {
            collector.record(Metric::Loss, 2.0);
        }

        let report = analyzer.analyze("improving", &collector, 10.0);
        let loss_summary = report.metric_summaries.get(&Metric::Loss).unwrap();

        // Mean = 1.2, midpoint = 1.5, so mean < midpoint → Improving
        assert!(
            loss_summary.trend == Trend::Improving,
            "Expected Improving, got {:?} (mean={:.2}, mid={:.2})",
            loss_summary.trend,
            loss_summary.mean,
            (loss_summary.min + loss_summary.max) / 2.0
        );
    }

    #[test]
    fn test_trend_detection_oscillating() {
        let analyzer = HanseiAnalyzer::new();
        let mut collector = MetricsCollector::new();

        // Highly oscillating values
        for i in 0..50 {
            let value = if i % 2 == 0 { 10.0 } else { 1.0 };
            collector.record(Metric::Loss, value);
        }

        let report = analyzer.analyze("oscillating", &collector, 10.0);
        let loss_summary = report.metric_summaries.get(&Metric::Loss).unwrap();

        assert_eq!(loss_summary.trend, Trend::Oscillating);
    }

    #[test]
    fn test_issue_severity_ordering() {
        assert!(IssueSeverity::Critical > IssueSeverity::Error);
        assert!(IssueSeverity::Error > IssueSeverity::Warning);
        assert!(IssueSeverity::Warning > IssueSeverity::Info);
    }

    #[test]
    fn test_recommendations_generated() {
        let analyzer = HanseiAnalyzer::new();
        let mut collector = MetricsCollector::new();

        collector.record(Metric::Loss, f64::NAN);

        let report = analyzer.analyze("rec-test", &collector, 10.0);

        assert!(!report.recommendations.is_empty());
        assert!(report.recommendations[0].contains("numerical stability"));
    }

    #[test]
    fn test_low_accuracy_warning() {
        let analyzer = HanseiAnalyzer::new();
        let mut collector = MetricsCollector::new();

        // Low accuracy over many steps
        for _ in 0..150 {
            collector.record(Metric::Accuracy, 0.3);
        }

        let report = analyzer.analyze("low-acc", &collector, 100.0);

        let perf_issues: Vec<_> = report
            .issues
            .iter()
            .filter(|i| i.category == "Performance")
            .collect();

        assert!(!perf_issues.is_empty());
    }

    #[test]
    fn test_empty_collector() {
        let analyzer = HanseiAnalyzer::new();
        let collector = MetricsCollector::new();

        let report = analyzer.analyze("empty", &collector, 0.0);

        assert_eq!(report.total_steps, 0);
        assert!(report.metric_summaries.is_empty());
        // Should have warning about missing loss
        assert!(report.issues.iter().any(|i| i.category == "Observability"));
    }
}
