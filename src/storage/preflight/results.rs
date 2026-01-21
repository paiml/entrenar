//! Preflight validation results.

use super::{CheckMetadata, CheckResult};

/// Results from running preflight checks
#[derive(Debug, Clone)]
pub struct PreflightResults {
    /// Individual check results
    results: Vec<(CheckMetadata, CheckResult)>,
    /// Overall pass/fail
    passed: bool,
    /// Number of checks that passed
    passed_count: usize,
    /// Number of checks that failed
    failed_count: usize,
    /// Number of warnings
    warning_count: usize,
    /// Number of skipped checks
    skipped_count: usize,
}

impl PreflightResults {
    /// Create a new PreflightResults instance
    pub(crate) fn new(
        results: Vec<(CheckMetadata, CheckResult)>,
        passed: bool,
        passed_count: usize,
        failed_count: usize,
        warning_count: usize,
        skipped_count: usize,
    ) -> Self {
        Self {
            results,
            passed,
            passed_count,
            failed_count,
            warning_count,
            skipped_count,
        }
    }

    /// Check if all required checks passed
    pub fn all_passed(&self) -> bool {
        self.passed
    }

    /// Get number of passed checks
    pub fn passed_count(&self) -> usize {
        self.passed_count
    }

    /// Get number of failed checks
    pub fn failed_count(&self) -> usize {
        self.failed_count
    }

    /// Get number of warnings
    pub fn warning_count(&self) -> usize {
        self.warning_count
    }

    /// Get number of skipped checks
    pub fn skipped_count(&self) -> usize {
        self.skipped_count
    }

    /// Get all results
    pub fn results(&self) -> &[(CheckMetadata, CheckResult)] {
        &self.results
    }

    /// Get failed checks only
    pub fn failed_checks(&self) -> Vec<(&CheckMetadata, &CheckResult)> {
        self.results
            .iter()
            .filter(|(check, result)| check.required && result.is_failed())
            .map(|(c, r)| (c, r))
            .collect()
    }

    /// Get warnings only
    pub fn warnings(&self) -> Vec<(&CheckMetadata, &CheckResult)> {
        self.results
            .iter()
            .filter(|(_, result)| result.is_warning())
            .map(|(c, r)| (c, r))
            .collect()
    }

    /// Format results as a report
    pub fn report(&self) -> String {
        let mut lines = Vec::new();
        lines.push("=== Preflight Check Results ===".to_string());
        lines.push(format!(
            "Status: {}",
            if self.passed { "PASSED" } else { "FAILED" }
        ));
        lines.push(format!(
            "Passed: {}, Failed: {}, Warnings: {}, Skipped: {}",
            self.passed_count, self.failed_count, self.warning_count, self.skipped_count
        ));
        lines.push(String::new());

        for (check, result) in &self.results {
            let status = match result {
                CheckResult::Passed { .. } => "✓",
                CheckResult::Failed { .. } => "✗",
                CheckResult::Warning { .. } => "⚠",
                CheckResult::Skipped { .. } => "○",
            };

            let message = match result {
                CheckResult::Passed { message } => message.clone(),
                CheckResult::Failed { message, details } => {
                    if let Some(d) = details {
                        format!("{message} ({d})")
                    } else {
                        message.clone()
                    }
                }
                CheckResult::Warning { message } => message.clone(),
                CheckResult::Skipped { reason } => reason.clone(),
            };

            lines.push(format!("{status} {}: {message}", check.name));
        }

        lines.join("\n")
    }
}
