//! Failure types and context structures.

use serde::{Deserialize, Serialize};

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
