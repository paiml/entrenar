//! Failure Context Structured Diagnostics (ENT-007)
//!
//! Provides structured failure diagnostics with categorization,
//! suggested fixes, and Pareto analysis for training runs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Categories of training failures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FailureCategory {
    /// Data quality issues (corrupt files, missing features, invalid formats)
    DataQuality,

    /// Model convergence failures (NaN loss, exploding gradients, divergence)
    ModelConvergence,

    /// Resource exhaustion (OOM, disk full, timeout)
    ResourceExhaustion,

    /// Dependency failures (missing crates, version conflicts, build errors)
    DependencyFailure,

    /// Configuration errors (invalid hyperparameters, missing required fields)
    ConfigurationError,

    /// Unknown or uncategorized failure
    Unknown,
}

impl FailureCategory {
    /// Get a human-readable description of the category
    pub fn description(&self) -> &'static str {
        match self {
            Self::DataQuality => "Data quality issue",
            Self::ModelConvergence => "Model convergence failure",
            Self::ResourceExhaustion => "Resource exhaustion",
            Self::DependencyFailure => "Dependency failure",
            Self::ConfigurationError => "Configuration error",
            Self::Unknown => "Unknown failure",
        }
    }

    /// Attempt to categorize from error message patterns
    pub fn from_error_message(message: &str) -> Self {
        let lower = message.to_lowercase();

        // Model convergence patterns
        if lower.contains("nan")
            || lower.contains("inf")
            || lower.contains("exploding")
            || lower.contains("diverge")
            || lower.contains("gradient")
        {
            return Self::ModelConvergence;
        }

        // Resource exhaustion patterns
        if lower.contains("out of memory")
            || lower.contains("oom")
            || lower.contains("memory")
            || lower.contains("timeout")
            || lower.contains("disk full")
            || lower.contains("no space")
        {
            return Self::ResourceExhaustion;
        }

        // Data quality patterns
        if lower.contains("corrupt")
            || lower.contains("invalid data")
            || lower.contains("missing feature")
            || lower.contains("data format")
            || lower.contains("parse error")
            || lower.contains("invalid shape")
        {
            return Self::DataQuality;
        }

        // Dependency patterns
        if lower.contains("dependency")
            || lower.contains("crate")
            || lower.contains("version")
            || lower.contains("build error")
            || lower.contains("compile")
        {
            return Self::DependencyFailure;
        }

        // Configuration patterns
        if lower.contains("config")
            || lower.contains("parameter")
            || lower.contains("invalid value")
            || lower.contains("missing field")
            || lower.contains("required")
        {
            return Self::ConfigurationError;
        }

        Self::Unknown
    }
}

impl std::fmt::Display for FailureCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// Structured failure context for a training run
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureContext {
    /// Error code (e.g., "E001", "NAN_LOSS")
    pub error_code: String,

    /// Human-readable error message
    pub message: String,

    /// Failure category for aggregation
    pub category: FailureCategory,

    /// Optional stack trace
    pub stack_trace: Option<String>,

    /// Suggested fix or remediation
    pub suggested_fix: Option<String>,

    /// Related run IDs that may have similar issues
    pub related_runs: Vec<String>,
}

impl FailureContext {
    /// Create a new failure context
    pub fn new(error_code: impl Into<String>, message: impl Into<String>) -> Self {
        let message_str = message.into();
        let category = FailureCategory::from_error_message(&message_str);

        Self {
            error_code: error_code.into(),
            message: message_str,
            category,
            stack_trace: None,
            suggested_fix: None,
            related_runs: Vec::new(),
        }
    }

    /// Create with explicit category
    pub fn with_category(
        error_code: impl Into<String>,
        message: impl Into<String>,
        category: FailureCategory,
    ) -> Self {
        Self {
            error_code: error_code.into(),
            message: message.into(),
            category,
            stack_trace: None,
            suggested_fix: None,
            related_runs: Vec::new(),
        }
    }

    /// Add a stack trace
    pub fn with_stack_trace(mut self, trace: impl Into<String>) -> Self {
        self.stack_trace = Some(trace.into());
        self
    }

    /// Add a suggested fix
    pub fn with_suggested_fix(mut self, fix: impl Into<String>) -> Self {
        self.suggested_fix = Some(fix.into());
        self
    }

    /// Add related run IDs
    pub fn with_related_runs(mut self, runs: Vec<String>) -> Self {
        self.related_runs = runs;
        self
    }

    /// Generate a suggested fix based on the category
    pub fn generate_suggested_fix(&self) -> String {
        match self.category {
            FailureCategory::ModelConvergence => {
                "Try reducing the learning rate, enabling gradient clipping, \
                 or checking for NaN values in input data."
                    .to_string()
            }
            FailureCategory::ResourceExhaustion => {
                "Try reducing batch size, using gradient checkpointing, \
                 or enabling mixed-precision training."
                    .to_string()
            }
            FailureCategory::DataQuality => {
                "Validate input data format, check for missing values, \
                 and verify data preprocessing pipeline."
                    .to_string()
            }
            FailureCategory::DependencyFailure => {
                "Run `cargo update`, check Cargo.lock for version conflicts, \
                 and verify all required features are enabled."
                    .to_string()
            }
            FailureCategory::ConfigurationError => {
                "Review configuration file for typos, missing required fields, \
                 and invalid parameter values."
                    .to_string()
            }
            FailureCategory::Unknown => {
                "Review the error message and stack trace for more details. \
                 Consider enabling debug logging."
                    .to_string()
            }
        }
    }
}

impl<E: std::error::Error> From<&E> for FailureContext {
    fn from(error: &E) -> Self {
        let message = error.to_string();
        let category = FailureCategory::from_error_message(&message);

        let mut context = Self::new("ERR_GENERIC", message);
        context.category = category;

        // Try to get source chain for stack trace
        let mut trace = String::new();
        let mut source = error.source();
        while let Some(s) = source {
            trace.push_str(&format!("Caused by: {s}\n"));
            source = s.source();
        }
        if !trace.is_empty() {
            context.stack_trace = Some(trace);
        }

        context
    }
}

/// Pareto analysis result for failure categories
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParetoAnalysis {
    /// Category counts sorted by frequency (descending)
    pub categories: Vec<(FailureCategory, u32)>,

    /// Total failures analyzed
    pub total_failures: u32,
}

impl ParetoAnalysis {
    /// Perform Pareto analysis on a list of failure contexts
    pub fn from_failures(failures: &[FailureContext]) -> Self {
        let mut counts: HashMap<FailureCategory, u32> = HashMap::new();

        for failure in failures {
            *counts.entry(failure.category).or_insert(0) += 1;
        }

        let mut categories: Vec<(FailureCategory, u32)> = counts.into_iter().collect();
        categories.sort_by(|a, b| b.1.cmp(&a.1)); // Sort descending by count

        Self {
            categories,
            total_failures: failures.len() as u32,
        }
    }

    /// Get the top N failure categories
    pub fn top_categories(&self, n: usize) -> Vec<(FailureCategory, u32)> {
        self.categories.iter().take(n).copied().collect()
    }

    /// Get percentage of failures for each category
    pub fn percentages(&self) -> Vec<(FailureCategory, f64)> {
        if self.total_failures == 0 {
            return Vec::new();
        }

        let total = f64::from(self.total_failures);
        self.categories
            .iter()
            .map(|(cat, count)| (*cat, (f64::from(*count) / total) * 100.0))
            .collect()
    }

    /// Get cumulative percentages (for Pareto chart)
    pub fn cumulative_percentages(&self) -> Vec<(FailureCategory, f64)> {
        if self.total_failures == 0 {
            return Vec::new();
        }

        let total = f64::from(self.total_failures);
        let mut cumulative = 0.0;
        self.categories
            .iter()
            .map(|(cat, count)| {
                cumulative += (f64::from(*count) / total) * 100.0;
                (*cat, cumulative)
            })
            .collect()
    }

    /// Find categories that account for ~80% of failures (Pareto principle)
    pub fn vital_few(&self) -> Vec<(FailureCategory, u32)> {
        let mut cumulative = 0u32;
        let threshold = (f64::from(self.total_failures) * 0.8) as u32;

        self.categories
            .iter()
            .take_while(|(_, count)| {
                let result = cumulative < threshold;
                cumulative += count;
                result
            })
            .copied()
            .collect()
    }
}

/// Convenience function for Pareto analysis
pub fn top_failure_categories(failures: &[FailureContext]) -> Vec<(FailureCategory, u32)> {
    ParetoAnalysis::from_failures(failures).categories
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_failure_category_from_error_message_convergence() {
        assert_eq!(
            FailureCategory::from_error_message("Loss is NaN at step 100"),
            FailureCategory::ModelConvergence
        );
        assert_eq!(
            FailureCategory::from_error_message("Gradient exploding detected"),
            FailureCategory::ModelConvergence
        );
        assert_eq!(
            FailureCategory::from_error_message("Training diverged"),
            FailureCategory::ModelConvergence
        );
    }

    #[test]
    fn test_failure_category_from_error_message_resource() {
        assert_eq!(
            FailureCategory::from_error_message("Out of memory"),
            FailureCategory::ResourceExhaustion
        );
        assert_eq!(
            FailureCategory::from_error_message("CUDA OOM error"),
            FailureCategory::ResourceExhaustion
        );
        assert_eq!(
            FailureCategory::from_error_message("Operation timeout"),
            FailureCategory::ResourceExhaustion
        );
    }

    #[test]
    fn test_failure_category_from_error_message_data() {
        assert_eq!(
            FailureCategory::from_error_message("Corrupt data file"),
            FailureCategory::DataQuality
        );
        assert_eq!(
            FailureCategory::from_error_message("Invalid shape: expected [32, 512]"),
            FailureCategory::DataQuality
        );
    }

    #[test]
    fn test_failure_category_from_error_message_dependency() {
        assert_eq!(
            FailureCategory::from_error_message("Failed to compile dependency"),
            FailureCategory::DependencyFailure
        );
        assert_eq!(
            FailureCategory::from_error_message("Version conflict in crate foo"),
            FailureCategory::DependencyFailure
        );
    }

    #[test]
    fn test_failure_category_from_error_message_config() {
        assert_eq!(
            FailureCategory::from_error_message("Missing required field 'lr'"),
            FailureCategory::ConfigurationError
        );
        assert_eq!(
            FailureCategory::from_error_message("Invalid parameter value"),
            FailureCategory::ConfigurationError
        );
    }

    #[test]
    fn test_failure_category_from_error_message_unknown() {
        assert_eq!(
            FailureCategory::from_error_message("Something went wrong"),
            FailureCategory::Unknown
        );
    }

    #[test]
    fn test_failure_category_description() {
        assert_eq!(
            FailureCategory::ModelConvergence.description(),
            "Model convergence failure"
        );
        assert_eq!(FailureCategory::Unknown.description(), "Unknown failure");
    }

    #[test]
    fn test_failure_context_new() {
        let ctx = FailureContext::new("E001", "Loss is NaN at step 100");

        assert_eq!(ctx.error_code, "E001");
        assert_eq!(ctx.message, "Loss is NaN at step 100");
        assert_eq!(ctx.category, FailureCategory::ModelConvergence);
        assert!(ctx.stack_trace.is_none());
        assert!(ctx.suggested_fix.is_none());
        assert!(ctx.related_runs.is_empty());
    }

    #[test]
    fn test_failure_context_with_category() {
        let ctx = FailureContext::with_category(
            "E002",
            "Generic error",
            FailureCategory::ResourceExhaustion,
        );

        assert_eq!(ctx.category, FailureCategory::ResourceExhaustion);
    }

    #[test]
    fn test_failure_context_builders() {
        let ctx = FailureContext::new("E001", "Test error")
            .with_stack_trace("at line 100\nat line 200")
            .with_suggested_fix("Try this fix")
            .with_related_runs(vec!["run-1".to_string(), "run-2".to_string()]);

        assert_eq!(
            ctx.stack_trace,
            Some("at line 100\nat line 200".to_string())
        );
        assert_eq!(ctx.suggested_fix, Some("Try this fix".to_string()));
        assert_eq!(ctx.related_runs, vec!["run-1", "run-2"]);
    }

    #[test]
    fn test_failure_context_generate_suggested_fix() {
        let ctx =
            FailureContext::with_category("E001", "NaN loss", FailureCategory::ModelConvergence);
        let fix = ctx.generate_suggested_fix();
        assert!(fix.contains("learning rate"));

        let ctx = FailureContext::with_category("E002", "OOM", FailureCategory::ResourceExhaustion);
        let fix = ctx.generate_suggested_fix();
        assert!(fix.contains("batch size"));
    }

    #[test]
    fn test_failure_context_from_error() {
        use std::io;

        let error = io::Error::new(io::ErrorKind::OutOfMemory, "Out of memory");
        let ctx = FailureContext::from(&error);

        assert_eq!(ctx.category, FailureCategory::ResourceExhaustion);
        assert!(ctx.message.contains("Out of memory"));
    }

    #[test]
    fn test_pareto_analysis_from_failures() {
        let failures = vec![
            FailureContext::with_category("E001", "NaN", FailureCategory::ModelConvergence),
            FailureContext::with_category("E002", "NaN", FailureCategory::ModelConvergence),
            FailureContext::with_category("E003", "NaN", FailureCategory::ModelConvergence),
            FailureContext::with_category("E004", "OOM", FailureCategory::ResourceExhaustion),
            FailureContext::with_category("E005", "Config", FailureCategory::ConfigurationError),
        ];

        let analysis = ParetoAnalysis::from_failures(&failures);

        assert_eq!(analysis.total_failures, 5);
        assert_eq!(analysis.categories[0].0, FailureCategory::ModelConvergence);
        assert_eq!(analysis.categories[0].1, 3);
    }

    #[test]
    fn test_pareto_analysis_top_categories() {
        let failures = vec![
            FailureContext::with_category("E001", "NaN", FailureCategory::ModelConvergence),
            FailureContext::with_category("E002", "NaN", FailureCategory::ModelConvergence),
            FailureContext::with_category("E003", "OOM", FailureCategory::ResourceExhaustion),
            FailureContext::with_category("E004", "Config", FailureCategory::ConfigurationError),
            FailureContext::with_category("E005", "Data", FailureCategory::DataQuality),
        ];

        let analysis = ParetoAnalysis::from_failures(&failures);
        let top2 = analysis.top_categories(2);

        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, FailureCategory::ModelConvergence);
        assert_eq!(top2[0].1, 2);
    }

    #[test]
    fn test_pareto_analysis_percentages() {
        let failures = vec![
            FailureContext::with_category("E001", "NaN", FailureCategory::ModelConvergence),
            FailureContext::with_category("E002", "NaN", FailureCategory::ModelConvergence),
            FailureContext::with_category("E003", "OOM", FailureCategory::ResourceExhaustion),
            FailureContext::with_category("E004", "OOM", FailureCategory::ResourceExhaustion),
        ];

        let analysis = ParetoAnalysis::from_failures(&failures);
        let percentages = analysis.percentages();

        // Both categories should be 50%
        assert!((percentages[0].1 - 50.0).abs() < f64::EPSILON);
        assert!((percentages[1].1 - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pareto_analysis_cumulative_percentages() {
        let failures = vec![
            FailureContext::with_category("E001", "NaN", FailureCategory::ModelConvergence),
            FailureContext::with_category("E002", "NaN", FailureCategory::ModelConvergence),
            FailureContext::with_category("E003", "NaN", FailureCategory::ModelConvergence),
            FailureContext::with_category("E004", "OOM", FailureCategory::ResourceExhaustion),
        ];

        let analysis = ParetoAnalysis::from_failures(&failures);
        let cumulative = analysis.cumulative_percentages();

        // ModelConvergence is 75%, cumulative should be 75%
        assert!((cumulative[0].1 - 75.0).abs() < f64::EPSILON);
        // Adding ResourceExhaustion (25%), cumulative should be 100%
        assert!((cumulative[1].1 - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pareto_analysis_vital_few() {
        // Create 10 failures: 6 convergence, 2 resource, 1 config, 1 data
        let mut failures = Vec::new();
        for i in 0..6 {
            failures.push(FailureContext::with_category(
                format!("E{i:03}"),
                "NaN",
                FailureCategory::ModelConvergence,
            ));
        }
        for i in 6..8 {
            failures.push(FailureContext::with_category(
                format!("E{i:03}"),
                "OOM",
                FailureCategory::ResourceExhaustion,
            ));
        }
        failures.push(FailureContext::with_category(
            "E008",
            "Config",
            FailureCategory::ConfigurationError,
        ));
        failures.push(FailureContext::with_category(
            "E009",
            "Data",
            FailureCategory::DataQuality,
        ));

        let analysis = ParetoAnalysis::from_failures(&failures);
        let vital = analysis.vital_few();

        // ModelConvergence (60%) + ResourceExhaustion (20%) = 80%
        // So vital_few should include at least ModelConvergence
        assert!(!vital.is_empty());
        assert_eq!(vital[0].0, FailureCategory::ModelConvergence);
    }

    #[test]
    fn test_pareto_analysis_empty() {
        let analysis = ParetoAnalysis::from_failures(&[]);

        assert_eq!(analysis.total_failures, 0);
        assert!(analysis.categories.is_empty());
        assert!(analysis.percentages().is_empty());
        assert!(analysis.cumulative_percentages().is_empty());
    }

    #[test]
    fn test_top_failure_categories_function() {
        let failures = vec![
            FailureContext::with_category("E001", "NaN", FailureCategory::ModelConvergence),
            FailureContext::with_category("E002", "NaN", FailureCategory::ModelConvergence),
            FailureContext::with_category("E003", "OOM", FailureCategory::ResourceExhaustion),
        ];

        let categories = top_failure_categories(&failures);

        assert_eq!(categories.len(), 2);
        assert_eq!(categories[0].0, FailureCategory::ModelConvergence);
        assert_eq!(categories[0].1, 2);
    }

    #[test]
    fn test_failure_context_serialization() {
        let ctx = FailureContext::new("E001", "Test error")
            .with_suggested_fix("Try this")
            .with_related_runs(vec!["run-1".to_string()]);

        let json = serde_json::to_string(&ctx).unwrap();
        let parsed: FailureContext = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.error_code, ctx.error_code);
        assert_eq!(parsed.category, ctx.category);
        assert_eq!(parsed.suggested_fix, ctx.suggested_fix);
    }
}
