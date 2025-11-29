//! Model validation (Andon principle - surface problems immediately).

use entrenar_common::Result;
use std::path::Path;

/// Result of model validation.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the model is valid
    pub valid: bool,
    /// List of issues found
    pub issues: Vec<ValidationIssue>,
    /// List of warnings
    pub warnings: Vec<String>,
    /// Validation checks performed
    pub checks: Vec<ValidationCheck>,
}

impl ValidationResult {
    /// Check if there are any errors (not just warnings).
    pub fn has_errors(&self) -> bool {
        self.issues.iter().any(|i| i.severity == Severity::Error)
    }

    /// Format as human-readable report.
    pub fn to_report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!(
            "Validation Result: {}\n\n",
            if self.valid { "PASS" } else { "FAIL" }
        ));

        if !self.issues.is_empty() {
            report.push_str("Issues:\n");
            for issue in &self.issues {
                let prefix = match issue.severity {
                    Severity::Error => "✗",
                    Severity::Warning => "⚠",
                    Severity::Info => "ℹ",
                };
                report.push_str(&format!("  {} {}: {}\n", prefix, issue.code, issue.message));
                if let Some(suggestion) = &issue.suggestion {
                    report.push_str(&format!("    → {}\n", suggestion));
                }
            }
            report.push('\n');
        }

        report.push_str("Checks Performed:\n");
        for check in &self.checks {
            let status = if check.passed { "✓" } else { "✗" };
            report.push_str(&format!("  {} {}\n", status, check.name));
        }

        report
    }
}

/// A validation issue.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Issue code for programmatic handling
    pub code: String,
    /// Human-readable message
    pub message: String,
    /// Severity level
    pub severity: Severity,
    /// Actionable suggestion
    pub suggestion: Option<String>,
    /// Affected tensor name (if applicable)
    pub tensor: Option<String>,
}

/// Severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Critical error - model unusable
    Error,
    /// Warning - model usable but may have issues
    Warning,
    /// Informational - not a problem
    Info,
}

/// A validation check that was performed.
#[derive(Debug, Clone)]
pub struct ValidationCheck {
    /// Check name
    pub name: String,
    /// Whether check passed
    pub passed: bool,
    /// Duration in milliseconds
    pub duration_ms: u64,
}

/// Model integrity checker.
pub struct IntegrityChecker {
    strict: bool,
}

impl Default for IntegrityChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl IntegrityChecker {
    /// Create a new integrity checker.
    pub fn new() -> Self {
        Self { strict: false }
    }

    /// Enable strict mode (treat warnings as errors).
    pub fn strict(mut self) -> Self {
        self.strict = true;
        self
    }

    /// Validate a model file.
    pub fn validate(&self, path: &Path) -> Result<ValidationResult> {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();
        let mut checks = Vec::new();

        // Check file exists
        let file_check = self.check_file_exists(path);
        checks.push(file_check.clone());
        if !file_check.passed {
            issues.push(ValidationIssue {
                code: "V001".to_string(),
                message: format!("File not found: {}", path.display()),
                severity: Severity::Error,
                suggestion: Some("Check the file path".to_string()),
                tensor: None,
            });
            return Ok(ValidationResult {
                valid: false,
                issues,
                warnings,
                checks,
            });
        }

        // Check format
        let format_check = self.check_format(path);
        checks.push(format_check.clone());
        if !format_check.passed {
            issues.push(ValidationIssue {
                code: "V002".to_string(),
                message: "Unsupported or potentially unsafe format".to_string(),
                severity: if self.strict {
                    Severity::Error
                } else {
                    Severity::Warning
                },
                suggestion: Some("Use SafeTensors format for security".to_string()),
                tensor: None,
            });
        }

        // Check file size
        let size_check = self.check_file_size(path);
        checks.push(size_check.clone());
        if !size_check.passed {
            warnings.push("File size is unusually small - may be corrupted".to_string());
        }

        // In real implementation, would also check:
        // - Tensor shapes consistency
        // - NaN/Inf values
        // - Data type consistency
        // - Architecture constraints

        let valid = !issues.iter().any(|i| i.severity == Severity::Error);

        Ok(ValidationResult {
            valid,
            issues,
            warnings,
            checks,
        })
    }

    fn check_file_exists(&self, path: &Path) -> ValidationCheck {
        let start = std::time::Instant::now();
        let passed = path.exists();
        ValidationCheck {
            name: "File exists".to_string(),
            passed,
            duration_ms: start.elapsed().as_millis() as u64,
        }
    }

    fn check_format(&self, path: &Path) -> ValidationCheck {
        let start = std::time::Instant::now();
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let passed = matches!(
            extension.to_lowercase().as_str(),
            "safetensors" | "gguf" | "apr"
        );
        ValidationCheck {
            name: "Safe format".to_string(),
            passed,
            duration_ms: start.elapsed().as_millis() as u64,
        }
    }

    fn check_file_size(&self, path: &Path) -> ValidationCheck {
        let start = std::time::Instant::now();
        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let passed = size > 1000; // At least 1KB
        ValidationCheck {
            name: "Valid file size".to_string(),
            passed,
            duration_ms: start.elapsed().as_millis() as u64,
        }
    }
}

/// Validate a model file.
pub fn validate_model(path: impl AsRef<Path>) -> Result<ValidationResult> {
    IntegrityChecker::new().validate(path.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_validation_missing_file() {
        let result = validate_model("/nonexistent/model.safetensors").unwrap();
        assert!(!result.valid);
        assert!(result.issues.iter().any(|i| i.code == "V001"));
    }

    #[test]
    fn test_validation_safe_format() {
        let mut file = NamedTempFile::with_suffix(".safetensors").unwrap();
        file.write_all(&[0u8; 2000]).unwrap();

        let result = validate_model(file.path()).unwrap();
        assert!(result
            .checks
            .iter()
            .any(|c| c.name == "Safe format" && c.passed));
    }

    #[test]
    fn test_validation_unsafe_format() {
        let mut file = NamedTempFile::with_suffix(".pt").unwrap();
        file.write_all(&[0u8; 2000]).unwrap();

        let result = validate_model(file.path()).unwrap();
        // In non-strict mode, unsafe format is a warning not error
        let format_check = result
            .checks
            .iter()
            .find(|c| c.name == "Safe format")
            .unwrap();
        assert!(!format_check.passed);
    }

    #[test]
    fn test_strict_mode() {
        let mut file = NamedTempFile::with_suffix(".pt").unwrap();
        file.write_all(&[0u8; 2000]).unwrap();

        let result = IntegrityChecker::new()
            .strict()
            .validate(file.path())
            .unwrap();
        assert!(!result.valid); // Unsafe format is error in strict mode
    }

    #[test]
    fn test_validation_report() {
        let result = ValidationResult {
            valid: false,
            issues: vec![ValidationIssue {
                code: "V001".to_string(),
                message: "Test error".to_string(),
                severity: Severity::Error,
                suggestion: Some("Fix it".to_string()),
                tensor: None,
            }],
            warnings: vec![],
            checks: vec![ValidationCheck {
                name: "Test check".to_string(),
                passed: false,
                duration_ms: 1,
            }],
        };

        let report = result.to_report();
        assert!(report.contains("FAIL"));
        assert!(report.contains("V001"));
        assert!(report.contains("Fix it"));
    }
}
