//! Metamorphic violation types for behavioral testing
//!
//! Defines violation types and metamorphic relation categories
//! used in behavioral integrity verification.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A metamorphic violation detected during behavioral testing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetamorphicViolation {
    /// Unique identifier for this violation
    pub id: String,

    /// Type of metamorphic relation violated
    pub relation_type: MetamorphicRelationType,

    /// Description of the violation
    pub description: String,

    /// Input that caused the violation
    pub input_description: String,

    /// Expected behavior
    pub expected: String,

    /// Actual behavior observed
    pub actual: String,

    /// Severity of the violation (0.0 - 1.0, higher = more severe)
    pub severity: f64,

    /// When the violation was detected
    pub detected_at: DateTime<Utc>,
}

impl MetamorphicViolation {
    /// Create a new metamorphic violation
    pub fn new(
        id: impl Into<String>,
        relation_type: MetamorphicRelationType,
        description: impl Into<String>,
        input_description: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
        severity: f64,
    ) -> Self {
        Self {
            id: id.into(),
            relation_type,
            description: description.into(),
            input_description: input_description.into(),
            expected: expected.into(),
            actual: actual.into(),
            severity: severity.clamp(0.0, 1.0),
            detected_at: Utc::now(),
        }
    }

    /// Check if this is a critical violation (severity >= 0.8)
    pub fn is_critical(&self) -> bool {
        self.severity >= 0.8
    }

    /// Check if this is a warning-level violation (severity >= 0.5)
    pub fn is_warning(&self) -> bool {
        self.severity >= 0.5
    }
}

/// Types of metamorphic relations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetamorphicRelationType {
    /// Additive: f(x + c) should relate to f(x) in predictable way
    Additive,
    /// Multiplicative: f(k * x) should relate to f(x)
    Multiplicative,
    /// Permutation: f(permute(x)) should relate to f(x)
    Permutation,
    /// Composition: f(g(x)) should relate to g(f(x)) or similar
    Composition,
    /// Negation: f(-x) should relate to f(x)
    Negation,
    /// Inclusion: f(x ⊂ y) implies relation between f(x) and f(y)
    Inclusion,
    /// Identity: f(x) should equal f(x) across invocations
    Identity,
    /// Custom relation type
    Custom,
}

impl std::fmt::Display for MetamorphicRelationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Additive => write!(f, "additive"),
            Self::Multiplicative => write!(f, "multiplicative"),
            Self::Permutation => write!(f, "permutation"),
            Self::Composition => write!(f, "composition"),
            Self::Negation => write!(f, "negation"),
            Self::Inclusion => write!(f, "inclusion"),
            Self::Identity => write!(f, "identity"),
            Self::Custom => write!(f, "custom"),
        }
    }
}
