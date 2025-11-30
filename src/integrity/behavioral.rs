//! Behavioral Integrity Metrics (ENT-013)
//!
//! Provides behavioral integrity verification for ML model promotion gates.
//! Tracks metamorphic testing results, syscall patterns, timing variance,
//! and semantic equivalence to ensure model behavior consistency.

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

/// Behavioral integrity metrics for model promotion gates
///
/// Tracks multiple dimensions of behavioral consistency to determine
/// if a model is ready for promotion to production.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BehavioralIntegrity {
    /// Overall equivalence score (0.0 - 1.0)
    /// Measures how well the model's behavior matches expected patterns
    pub equivalence_score: f64,

    /// Syscall pattern match score (0.0 - 1.0)
    /// Measures consistency of system call patterns during inference
    pub syscall_match: f64,

    /// Timing variance score (0.0 - 1.0, lower = more consistent)
    /// Measures consistency of inference timing
    pub timing_variance: f64,

    /// Semantic equivalence score (0.0 - 1.0)
    /// Measures semantic consistency of model outputs
    pub semantic_equiv: f64,

    /// List of metamorphic violations detected
    pub violations: Vec<MetamorphicViolation>,

    /// Timestamp when metrics were collected
    pub timestamp: DateTime<Utc>,

    /// Number of test cases evaluated
    pub test_count: u32,

    /// Model version or identifier being evaluated
    pub model_id: String,
}

impl BehavioralIntegrity {
    /// Create new behavioral integrity metrics
    pub fn new(
        equivalence_score: f64,
        syscall_match: f64,
        timing_variance: f64,
        semantic_equiv: f64,
        model_id: impl Into<String>,
    ) -> Self {
        Self {
            equivalence_score: equivalence_score.clamp(0.0, 1.0),
            syscall_match: syscall_match.clamp(0.0, 1.0),
            timing_variance: timing_variance.clamp(0.0, 1.0),
            semantic_equiv: semantic_equiv.clamp(0.0, 1.0),
            violations: Vec::new(),
            timestamp: Utc::now(),
            test_count: 0,
            model_id: model_id.into(),
        }
    }

    /// Create perfect behavioral integrity (all scores = 1.0, variance = 0.0)
    pub fn perfect(model_id: impl Into<String>) -> Self {
        Self::new(1.0, 1.0, 0.0, 1.0, model_id)
    }

    /// Add a metamorphic violation
    pub fn add_violation(&mut self, violation: MetamorphicViolation) {
        self.violations.push(violation);
    }

    /// Set the test count
    pub fn with_test_count(mut self, count: u32) -> Self {
        self.test_count = count;
        self
    }

    /// Calculate the composite integrity score
    ///
    /// Weighted average of all metrics (timing variance is inverted)
    pub fn composite_score(&self) -> f64 {
        // Weights for each metric
        const W_EQUIV: f64 = 0.3;
        const W_SYSCALL: f64 = 0.2;
        const W_TIMING: f64 = 0.2;
        const W_SEMANTIC: f64 = 0.3;

        let timing_score = 1.0 - self.timing_variance; // Invert: lower variance = better

        W_EQUIV * self.equivalence_score
            + W_SYSCALL * self.syscall_match
            + W_TIMING * timing_score
            + W_SEMANTIC * self.semantic_equiv
    }

    /// Check if the model passes promotion gate
    ///
    /// Requires:
    /// - Composite score >= threshold (default 0.9)
    /// - No critical violations
    /// - Timing variance < 0.2
    pub fn passes_gate(&self, threshold: f64) -> bool {
        self.composite_score() >= threshold
            && !self.has_critical_violations()
            && self.timing_variance < 0.2
    }

    /// Check if there are any critical violations
    pub fn has_critical_violations(&self) -> bool {
        self.violations
            .iter()
            .any(MetamorphicViolation::is_critical)
    }

    /// Get count of violations by severity level
    pub fn violation_counts(&self) -> ViolationCounts {
        let critical = self.violations.iter().filter(|v| v.is_critical()).count() as u32;
        let warnings = self
            .violations
            .iter()
            .filter(|v| v.is_warning() && !v.is_critical())
            .count() as u32;
        let minor = self.violations.iter().filter(|v| !v.is_warning()).count() as u32;

        ViolationCounts {
            critical,
            warnings,
            minor,
            total: self.violations.len() as u32,
        }
    }

    /// Get violations grouped by relation type
    pub fn violations_by_type(
        &self,
    ) -> std::collections::HashMap<MetamorphicRelationType, Vec<&MetamorphicViolation>> {
        let mut map = std::collections::HashMap::new();
        for v in &self.violations {
            map.entry(v.relation_type).or_insert_with(Vec::new).push(v);
        }
        map
    }

    /// Get the most severe violation, if any
    pub fn most_severe_violation(&self) -> Option<&MetamorphicViolation> {
        self.violations.iter().max_by(|a, b| {
            a.severity
                .partial_cmp(&b.severity)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get a human-readable assessment
    pub fn assessment(&self) -> IntegrityAssessment {
        let score = self.composite_score();
        let counts = self.violation_counts();

        if counts.critical > 0 {
            IntegrityAssessment::Critical
        } else if score < 0.5 {
            IntegrityAssessment::Poor
        } else if score < 0.7 {
            IntegrityAssessment::Fair
        } else if score < 0.9 {
            IntegrityAssessment::Good
        } else {
            IntegrityAssessment::Excellent
        }
    }

    /// Generate a summary report
    pub fn summary(&self) -> String {
        let counts = self.violation_counts();
        format!(
            "Model: {}\n\
             Composite Score: {:.1}%\n\
             Assessment: {}\n\
             Violations: {} critical, {} warnings, {} minor\n\
             Tests Run: {}\n\
             Gate Status: {}",
            self.model_id,
            self.composite_score() * 100.0,
            self.assessment(),
            counts.critical,
            counts.warnings,
            counts.minor,
            self.test_count,
            if self.passes_gate(0.9) {
                "PASS"
            } else {
                "FAIL"
            }
        )
    }
}

/// Counts of violations by severity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViolationCounts {
    /// Critical violations (severity >= 0.8)
    pub critical: u32,
    /// Warning violations (0.5 <= severity < 0.8)
    pub warnings: u32,
    /// Minor violations (severity < 0.5)
    pub minor: u32,
    /// Total violations
    pub total: u32,
}

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

/// Builder for BehavioralIntegrity
#[derive(Debug, Clone)]
pub struct BehavioralIntegrityBuilder {
    equivalence_score: f64,
    syscall_match: f64,
    timing_variance: f64,
    semantic_equiv: f64,
    violations: Vec<MetamorphicViolation>,
    test_count: u32,
    model_id: String,
}

impl BehavioralIntegrityBuilder {
    /// Create a new builder
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            equivalence_score: 0.0,
            syscall_match: 0.0,
            timing_variance: 1.0,
            semantic_equiv: 0.0,
            violations: Vec::new(),
            test_count: 0,
            model_id: model_id.into(),
        }
    }

    /// Set equivalence score
    pub fn equivalence_score(mut self, score: f64) -> Self {
        self.equivalence_score = score;
        self
    }

    /// Set syscall match score
    pub fn syscall_match(mut self, score: f64) -> Self {
        self.syscall_match = score;
        self
    }

    /// Set timing variance
    pub fn timing_variance(mut self, variance: f64) -> Self {
        self.timing_variance = variance;
        self
    }

    /// Set semantic equivalence score
    pub fn semantic_equiv(mut self, score: f64) -> Self {
        self.semantic_equiv = score;
        self
    }

    /// Add a violation
    pub fn violation(mut self, violation: MetamorphicViolation) -> Self {
        self.violations.push(violation);
        self
    }

    /// Set test count
    pub fn test_count(mut self, count: u32) -> Self {
        self.test_count = count;
        self
    }

    /// Build the BehavioralIntegrity instance
    pub fn build(self) -> BehavioralIntegrity {
        let mut integrity = BehavioralIntegrity::new(
            self.equivalence_score,
            self.syscall_match,
            self.timing_variance,
            self.semantic_equiv,
            self.model_id,
        );
        integrity.violations = self.violations;
        integrity.test_count = self.test_count;
        integrity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metamorphic_violation_new() {
        let violation = MetamorphicViolation::new(
            "V001",
            MetamorphicRelationType::Identity,
            "Output differs on identical input",
            "Input: [1, 2, 3]",
            "[0.5, 0.3, 0.2]",
            "[0.4, 0.4, 0.2]",
            0.7,
        );

        assert_eq!(violation.id, "V001");
        assert_eq!(violation.relation_type, MetamorphicRelationType::Identity);
        assert!((violation.severity - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metamorphic_violation_severity_clamped() {
        let high = MetamorphicViolation::new(
            "V001",
            MetamorphicRelationType::Identity,
            "desc",
            "input",
            "exp",
            "act",
            1.5,
        );
        assert!((high.severity - 1.0).abs() < f64::EPSILON);

        let low = MetamorphicViolation::new(
            "V002",
            MetamorphicRelationType::Identity,
            "desc",
            "input",
            "exp",
            "act",
            -0.5,
        );
        assert!((low.severity - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metamorphic_violation_is_critical() {
        let critical = MetamorphicViolation::new(
            "V001",
            MetamorphicRelationType::Identity,
            "desc",
            "input",
            "exp",
            "act",
            0.8,
        );
        assert!(critical.is_critical());

        let not_critical = MetamorphicViolation::new(
            "V002",
            MetamorphicRelationType::Identity,
            "desc",
            "input",
            "exp",
            "act",
            0.79,
        );
        assert!(!not_critical.is_critical());
    }

    #[test]
    fn test_metamorphic_violation_is_warning() {
        let warning = MetamorphicViolation::new(
            "V001",
            MetamorphicRelationType::Identity,
            "desc",
            "input",
            "exp",
            "act",
            0.5,
        );
        assert!(warning.is_warning());

        let not_warning = MetamorphicViolation::new(
            "V002",
            MetamorphicRelationType::Identity,
            "desc",
            "input",
            "exp",
            "act",
            0.49,
        );
        assert!(!not_warning.is_warning());
    }

    #[test]
    fn test_metamorphic_relation_type_display() {
        assert_eq!(format!("{}", MetamorphicRelationType::Additive), "additive");
        assert_eq!(
            format!("{}", MetamorphicRelationType::Permutation),
            "permutation"
        );
        assert_eq!(format!("{}", MetamorphicRelationType::Identity), "identity");
    }

    #[test]
    fn test_behavioral_integrity_new() {
        let integrity = BehavioralIntegrity::new(0.95, 0.90, 0.05, 0.92, "model-v1");

        assert!((integrity.equivalence_score - 0.95).abs() < f64::EPSILON);
        assert!((integrity.syscall_match - 0.90).abs() < f64::EPSILON);
        assert!((integrity.timing_variance - 0.05).abs() < f64::EPSILON);
        assert!((integrity.semantic_equiv - 0.92).abs() < f64::EPSILON);
        assert_eq!(integrity.model_id, "model-v1");
        assert!(integrity.violations.is_empty());
    }

    #[test]
    fn test_behavioral_integrity_scores_clamped() {
        let integrity = BehavioralIntegrity::new(1.5, -0.1, 2.0, -0.5, "model");

        assert!((integrity.equivalence_score - 1.0).abs() < f64::EPSILON);
        assert!((integrity.syscall_match - 0.0).abs() < f64::EPSILON);
        assert!((integrity.timing_variance - 1.0).abs() < f64::EPSILON);
        assert!((integrity.semantic_equiv - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_behavioral_integrity_perfect() {
        let integrity = BehavioralIntegrity::perfect("model-v1");

        assert!((integrity.equivalence_score - 1.0).abs() < f64::EPSILON);
        assert!((integrity.syscall_match - 1.0).abs() < f64::EPSILON);
        assert!((integrity.timing_variance - 0.0).abs() < f64::EPSILON);
        assert!((integrity.semantic_equiv - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_behavioral_integrity_composite_score_perfect() {
        let integrity = BehavioralIntegrity::perfect("model");
        let score = integrity.composite_score();

        // Perfect scores should yield composite of 1.0
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_behavioral_integrity_composite_score_mixed() {
        let integrity = BehavioralIntegrity::new(0.8, 0.7, 0.3, 0.9, "model");
        let score = integrity.composite_score();

        // Manual calculation:
        // 0.3 * 0.8 + 0.2 * 0.7 + 0.2 * (1-0.3) + 0.3 * 0.9
        // = 0.24 + 0.14 + 0.14 + 0.27 = 0.79
        assert!((score - 0.79).abs() < 0.01);
    }

    #[test]
    fn test_behavioral_integrity_passes_gate() {
        let good = BehavioralIntegrity::perfect("model");
        assert!(good.passes_gate(0.9));

        let poor = BehavioralIntegrity::new(0.5, 0.5, 0.5, 0.5, "model");
        assert!(!poor.passes_gate(0.9));
    }

    #[test]
    fn test_behavioral_integrity_passes_gate_timing_variance() {
        let mut integrity = BehavioralIntegrity::perfect("model");
        integrity.timing_variance = 0.25; // Too high

        // Even with perfect other scores, high timing variance fails
        assert!(!integrity.passes_gate(0.9));
    }

    #[test]
    fn test_behavioral_integrity_passes_gate_critical_violation() {
        let mut integrity = BehavioralIntegrity::perfect("model");
        integrity.add_violation(MetamorphicViolation::new(
            "V001",
            MetamorphicRelationType::Identity,
            "desc",
            "input",
            "exp",
            "act",
            0.85, // Critical
        ));

        assert!(!integrity.passes_gate(0.9));
    }

    #[test]
    fn test_behavioral_integrity_violation_counts() {
        let mut integrity = BehavioralIntegrity::new(0.9, 0.9, 0.1, 0.9, "model");

        // Add violations of different severities
        integrity.add_violation(MetamorphicViolation::new(
            "V1",
            MetamorphicRelationType::Identity,
            "d",
            "i",
            "e",
            "a",
            0.9, // Critical
        ));
        integrity.add_violation(MetamorphicViolation::new(
            "V2",
            MetamorphicRelationType::Identity,
            "d",
            "i",
            "e",
            "a",
            0.6, // Warning
        ));
        integrity.add_violation(MetamorphicViolation::new(
            "V3",
            MetamorphicRelationType::Identity,
            "d",
            "i",
            "e",
            "a",
            0.3, // Minor
        ));
        integrity.add_violation(MetamorphicViolation::new(
            "V4",
            MetamorphicRelationType::Identity,
            "d",
            "i",
            "e",
            "a",
            0.2, // Minor
        ));

        let counts = integrity.violation_counts();
        assert_eq!(counts.critical, 1);
        assert_eq!(counts.warnings, 1);
        assert_eq!(counts.minor, 2);
        assert_eq!(counts.total, 4);
    }

    #[test]
    fn test_behavioral_integrity_violations_by_type() {
        let mut integrity = BehavioralIntegrity::new(0.9, 0.9, 0.1, 0.9, "model");

        integrity.add_violation(MetamorphicViolation::new(
            "V1",
            MetamorphicRelationType::Identity,
            "d",
            "i",
            "e",
            "a",
            0.5,
        ));
        integrity.add_violation(MetamorphicViolation::new(
            "V2",
            MetamorphicRelationType::Additive,
            "d",
            "i",
            "e",
            "a",
            0.5,
        ));
        integrity.add_violation(MetamorphicViolation::new(
            "V3",
            MetamorphicRelationType::Identity,
            "d",
            "i",
            "e",
            "a",
            0.5,
        ));

        let by_type = integrity.violations_by_type();
        assert_eq!(
            by_type
                .get(&MetamorphicRelationType::Identity)
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            by_type
                .get(&MetamorphicRelationType::Additive)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn test_behavioral_integrity_most_severe_violation() {
        let mut integrity = BehavioralIntegrity::new(0.9, 0.9, 0.1, 0.9, "model");

        integrity.add_violation(MetamorphicViolation::new(
            "V1",
            MetamorphicRelationType::Identity,
            "d",
            "i",
            "e",
            "a",
            0.3,
        ));
        integrity.add_violation(MetamorphicViolation::new(
            "V2",
            MetamorphicRelationType::Identity,
            "d",
            "i",
            "e",
            "a",
            0.9,
        ));
        integrity.add_violation(MetamorphicViolation::new(
            "V3",
            MetamorphicRelationType::Identity,
            "d",
            "i",
            "e",
            "a",
            0.5,
        ));

        let most_severe = integrity.most_severe_violation().unwrap();
        assert_eq!(most_severe.id, "V2");
    }

    #[test]
    fn test_behavioral_integrity_assessment() {
        let excellent = BehavioralIntegrity::perfect("model");
        assert_eq!(excellent.assessment(), IntegrityAssessment::Excellent);

        let good = BehavioralIntegrity::new(0.8, 0.8, 0.1, 0.8, "model");
        assert_eq!(good.assessment(), IntegrityAssessment::Good);

        let fair = BehavioralIntegrity::new(0.6, 0.6, 0.3, 0.6, "model");
        assert_eq!(fair.assessment(), IntegrityAssessment::Fair);

        let poor = BehavioralIntegrity::new(0.3, 0.3, 0.5, 0.3, "model");
        assert_eq!(poor.assessment(), IntegrityAssessment::Poor);

        let mut critical = BehavioralIntegrity::perfect("model");
        critical.add_violation(MetamorphicViolation::new(
            "V1",
            MetamorphicRelationType::Identity,
            "d",
            "i",
            "e",
            "a",
            0.85,
        ));
        assert_eq!(critical.assessment(), IntegrityAssessment::Critical);
    }

    #[test]
    fn test_integrity_assessment_display() {
        assert_eq!(format!("{}", IntegrityAssessment::Excellent), "Excellent");
        assert_eq!(format!("{}", IntegrityAssessment::Critical), "Critical");
    }

    #[test]
    fn test_behavioral_integrity_summary() {
        let integrity = BehavioralIntegrity::perfect("model-v1").with_test_count(100);

        let summary = integrity.summary();
        assert!(summary.contains("model-v1"));
        assert!(summary.contains("100.0%"));
        assert!(summary.contains("Excellent"));
        assert!(summary.contains("PASS"));
    }

    #[test]
    fn test_behavioral_integrity_builder() {
        let integrity = BehavioralIntegrityBuilder::new("model-v2")
            .equivalence_score(0.95)
            .syscall_match(0.90)
            .timing_variance(0.05)
            .semantic_equiv(0.92)
            .test_count(500)
            .build();

        assert_eq!(integrity.model_id, "model-v2");
        assert!((integrity.equivalence_score - 0.95).abs() < f64::EPSILON);
        assert_eq!(integrity.test_count, 500);
    }

    #[test]
    fn test_behavioral_integrity_builder_with_violation() {
        let violation = MetamorphicViolation::new(
            "V001",
            MetamorphicRelationType::Identity,
            "desc",
            "input",
            "exp",
            "act",
            0.5,
        );

        let integrity = BehavioralIntegrityBuilder::new("model")
            .equivalence_score(0.9)
            .violation(violation)
            .build();

        assert_eq!(integrity.violations.len(), 1);
    }

    #[test]
    fn test_behavioral_integrity_serialization() {
        let integrity = BehavioralIntegrity::new(0.9, 0.85, 0.1, 0.88, "model-v1");
        let json = serde_json::to_string(&integrity).unwrap();
        let parsed: BehavioralIntegrity = serde_json::from_str(&json).unwrap();

        assert!((parsed.equivalence_score - integrity.equivalence_score).abs() < f64::EPSILON);
        assert_eq!(parsed.model_id, integrity.model_id);
    }

    #[test]
    fn test_behavioral_integrity_has_critical_violations_empty() {
        let integrity = BehavioralIntegrity::perfect("model");
        assert!(!integrity.has_critical_violations());
    }

    #[test]
    fn test_behavioral_integrity_has_critical_violations_minor_only() {
        let mut integrity = BehavioralIntegrity::perfect("model");
        integrity.add_violation(MetamorphicViolation::new(
            "V1",
            MetamorphicRelationType::Identity,
            "d",
            "i",
            "e",
            "a",
            0.3,
        ));
        assert!(!integrity.has_critical_violations());
    }

    #[test]
    fn test_behavioral_integrity_most_severe_violation_empty() {
        let integrity = BehavioralIntegrity::perfect("model");
        assert!(integrity.most_severe_violation().is_none());
    }
}
