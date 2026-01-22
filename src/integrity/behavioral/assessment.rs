//! Integrity assessment categories
//!
//! Provides human-readable assessment levels for behavioral integrity.

/// Overall integrity assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrityAssessment {
    /// Score >= 0.9, no critical violations
    Excellent,
    /// Score >= 0.7
    Good,
    /// Score >= 0.5
    Fair,
    /// Score < 0.5
    Poor,
    /// Has critical violations
    Critical,
}

impl std::fmt::Display for IntegrityAssessment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Excellent => write!(f, "Excellent"),
            Self::Good => write!(f, "Good"),
            Self::Fair => write!(f, "Fair"),
            Self::Poor => write!(f, "Poor"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}
