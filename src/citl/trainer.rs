//! Compiler-in-the-Loop (CITL) trainer for error-fix correlation
//!
//! Correlates compiler decision traces with compilation outcomes
//! for fault localization using statistical debugging techniques.
//!
//! # References
//! - Zeller (2002): Isolating Cause-Effect Chains
//! - Jones & Harrold (2005): Tarantula Fault Localization
//! - Chilimbi et al. (2009): HOLMES Statistical Debugging

use super::{DecisionPatternStore, FixPattern, FixSuggestion};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A source code span (location in source file)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceSpan {
    /// File path
    pub file: String,
    /// Start line (1-indexed)
    pub start_line: u32,
    /// Start column (1-indexed)
    pub start_col: u32,
    /// End line (1-indexed)
    pub end_line: u32,
    /// End column (1-indexed)
    pub end_col: u32,
}

impl SourceSpan {
    /// Create a new source span
    #[must_use]
    pub fn new(
        file: impl Into<String>,
        start_line: u32,
        start_col: u32,
        end_line: u32,
        end_col: u32,
    ) -> Self {
        Self {
            file: file.into(),
            start_line,
            start_col,
            end_line,
            end_col,
        }
    }

    /// Create a single-line span
    #[must_use]
    pub fn line(file: impl Into<String>, line: u32) -> Self {
        Self::new(file, line, 1, line, u32::MAX)
    }

    /// Check if this span overlaps with another
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        if self.file != other.file {
            return false;
        }

        // Check if ranges overlap
        !(self.end_line < other.start_line || other.end_line < self.start_line)
    }

    /// Check if this span contains another
    #[must_use]
    pub fn contains(&self, other: &Self) -> bool {
        if self.file != other.file {
            return false;
        }

        self.start_line <= other.start_line && self.end_line >= other.end_line
    }
}

impl std::fmt::Display for SourceSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{}-{}:{}",
            self.file, self.start_line, self.start_col, self.end_line, self.end_col
        )
    }
}

/// A single decision in the compiler trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTrace {
    /// Unique ID for this decision
    pub id: String,
    /// Type of decision (e.g., "type_inference", "borrow_check", "lifetime_resolution")
    pub decision_type: String,
    /// Description of the decision
    pub description: String,
    /// Source span where this decision was made
    pub span: Option<SourceSpan>,
    /// Timestamp (nanoseconds since session start)
    pub timestamp_ns: u64,
    /// Dependencies on other decisions
    pub depends_on: Vec<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl DecisionTrace {
    /// Create a new decision trace
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        decision_type: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            decision_type: decision_type.into(),
            description: description.into(),
            span: None,
            timestamp_ns: 0,
            depends_on: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set the source span
    #[must_use]
    pub fn with_span(mut self, span: SourceSpan) -> Self {
        self.span = Some(span);
        self
    }

    /// Set the timestamp
    #[must_use]
    pub fn with_timestamp(mut self, timestamp_ns: u64) -> Self {
        self.timestamp_ns = timestamp_ns;
        self
    }

    /// Add a dependency
    #[must_use]
    pub fn with_dependency(mut self, dep_id: impl Into<String>) -> Self {
        self.depends_on.push(dep_id.into());
        self
    }

    /// Add dependencies
    #[must_use]
    pub fn with_dependencies(mut self, deps: Vec<String>) -> Self {
        self.depends_on.extend(deps);
        self
    }
}

/// Outcome of a compilation session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompilationOutcome {
    /// Compilation succeeded
    Success,
    /// Compilation failed with errors
    Failure {
        /// Error codes encountered
        error_codes: Vec<String>,
        /// Error spans
        error_spans: Vec<SourceSpan>,
        /// Error messages
        messages: Vec<String>,
    },
}

impl CompilationOutcome {
    /// Create a success outcome
    #[must_use]
    pub fn success() -> Self {
        Self::Success
    }

    /// Create a failure outcome
    #[must_use]
    pub fn failure(
        error_codes: Vec<String>,
        error_spans: Vec<SourceSpan>,
        messages: Vec<String>,
    ) -> Self {
        Self::Failure {
            error_codes,
            error_spans,
            messages,
        }
    }

    /// Check if the outcome is success
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Get error codes if failure
    #[must_use]
    pub fn error_codes(&self) -> Vec<&str> {
        match self {
            Self::Success => vec![],
            Self::Failure { error_codes, .. } => error_codes.iter().map(String::as_str).collect(),
        }
    }

    /// Get error spans if failure
    #[must_use]
    pub fn error_spans(&self) -> Vec<&SourceSpan> {
        match self {
            Self::Success => vec![],
            Self::Failure { error_spans, .. } => error_spans.iter().collect(),
        }
    }
}

/// Result of error correlation analysis
#[derive(Debug, Clone)]
pub struct ErrorCorrelation {
    /// The error code being analyzed
    pub error_code: String,
    /// Error span where the error occurred
    pub error_span: SourceSpan,
    /// Decisions that may have contributed to the error (sorted by suspiciousness)
    pub suspicious_decisions: Vec<SuspiciousDecision>,
    /// Suggested fixes from the pattern store
    pub fix_suggestions: Vec<FixSuggestion>,
}

/// A decision suspected of contributing to an error
#[derive(Debug, Clone)]
pub struct SuspiciousDecision {
    /// The decision trace
    pub decision: DecisionTrace,
    /// Suspiciousness score (0.0 to 1.0)
    pub suspiciousness: f32,
    /// Reason for suspicion
    pub reason: String,
}

impl SuspiciousDecision {
    /// Create a new suspicious decision
    #[must_use]
    pub fn new(decision: DecisionTrace, suspiciousness: f32, reason: impl Into<String>) -> Self {
        Self {
            decision,
            suspiciousness: suspiciousness.clamp(0.0, 1.0),
            reason: reason.into(),
        }
    }
}

/// Configuration for the CITL trainer
#[derive(Debug, Clone)]
pub struct CITLConfig {
    /// Maximum number of fix suggestions to return
    pub max_suggestions: usize,
    /// Minimum suspiciousness score to report
    pub min_suspiciousness: f32,
    /// Whether to build dependency graphs
    pub enable_dependency_graph: bool,
}

impl Default for CITLConfig {
    fn default() -> Self {
        Self {
            max_suggestions: 5,
            min_suspiciousness: 0.3,
            enable_dependency_graph: true,
        }
    }
}

/// Session data for a single compilation
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used for future session replay/analysis
struct Session {
    /// Session ID
    id: String,
    /// Decision traces
    decisions: Vec<DecisionTrace>,
    /// Compilation outcome
    outcome: CompilationOutcome,
    /// Optional fix diff (if error was fixed)
    fix_diff: Option<String>,
}

/// Compiler-in-the-Loop (CITL) trainer
///
/// Correlates compiler decision traces with compilation outcomes
/// to identify error-contributing decisions and suggest fixes.
///
/// # Example
///
/// ```ignore
/// use entrenar::citl::{DecisionCITL, DecisionTrace, CompilationOutcome, SourceSpan};
///
/// let mut trainer = DecisionCITL::new()?;
///
/// // Ingest a failed session
/// let traces = vec![
///     DecisionTrace::new("d1", "type_inference", "Inferred type i32")
///         .with_span(SourceSpan::line("main.rs", 5)),
/// ];
/// let outcome = CompilationOutcome::failure(
///     vec!["E0308".to_string()],
///     vec![SourceSpan::line("main.rs", 5)],
///     vec!["mismatched types".to_string()],
/// );
///
/// trainer.ingest_session(traces, outcome, None)?;
///
/// // Correlate errors
/// let correlations = trainer.correlate_error("E0308", &SourceSpan::line("main.rs", 5))?;
/// ```
pub struct DecisionCITL {
    /// Pattern store for fix suggestions
    pattern_store: DecisionPatternStore,
    /// Sessions indexed by outcome type
    success_sessions: Vec<Session>,
    failed_sessions: Vec<Session>,
    /// Decision frequency in successful vs failed sessions (for Tarantula)
    decision_stats: HashMap<String, DecisionStats>,
    /// Configuration
    config: CITLConfig,
    /// Session counter
    session_counter: u64,
}

/// Statistics for a decision type across sessions
#[derive(Debug, Clone, Default)]
pub struct DecisionStats {
    /// Times seen in successful sessions
    pub success_count: u32,
    /// Times seen in failed sessions
    pub fail_count: u32,
    /// Total successful sessions
    pub total_success: u32,
    /// Total failed sessions
    pub total_fail: u32,
}

impl DecisionStats {
    /// Calculate Tarantula suspiciousness score
    ///
    /// Suspiciousness = (fail_freq) / (fail_freq + success_freq)
    /// where fail_freq = fail_count / total_fail
    /// and success_freq = success_count / total_success
    #[must_use]
    pub fn tarantula_score(&self) -> f32 {
        if self.total_fail == 0 || self.fail_count == 0 {
            return 0.0;
        }

        let fail_freq = self.fail_count as f32 / self.total_fail as f32;
        let success_freq = if self.total_success > 0 {
            self.success_count as f32 / self.total_success as f32
        } else {
            0.0
        };

        if fail_freq + success_freq < f32::EPSILON {
            0.0
        } else {
            fail_freq / (fail_freq + success_freq)
        }
    }
}

impl DecisionCITL {
    /// Create a new CITL trainer with default configuration
    pub fn new() -> Result<Self, crate::Error> {
        Self::with_config(CITLConfig::default())
    }

    /// Create a new CITL trainer with custom configuration
    pub fn with_config(config: CITLConfig) -> Result<Self, crate::Error> {
        Ok(Self {
            pattern_store: DecisionPatternStore::new()?,
            success_sessions: Vec::new(),
            failed_sessions: Vec::new(),
            decision_stats: HashMap::new(),
            config,
            session_counter: 0,
        })
    }

    /// Ingest a compilation session
    ///
    /// # Arguments
    ///
    /// * `traces` - Decision traces from the compilation
    /// * `outcome` - The compilation outcome
    /// * `fix_diff` - Optional fix diff if the error was fixed
    pub fn ingest_session(
        &mut self,
        traces: Vec<DecisionTrace>,
        outcome: CompilationOutcome,
        fix_diff: Option<String>,
    ) -> Result<(), crate::Error> {
        self.session_counter += 1;
        let session_id = format!("session_{}", self.session_counter);

        let session = Session {
            id: session_id,
            decisions: traces.clone(),
            outcome: outcome.clone(),
            fix_diff: fix_diff.clone(),
        };

        // Update decision statistics
        let is_success = outcome.is_success();
        for trace in &traces {
            let stats = self
                .decision_stats
                .entry(trace.decision_type.clone())
                .or_default();
            if is_success {
                stats.success_count += 1;
            } else {
                stats.fail_count += 1;
            }
        }

        // Update totals
        for stats in self.decision_stats.values_mut() {
            if is_success {
                stats.total_success += 1;
            } else {
                stats.total_fail += 1;
            }
        }

        // Store session
        if is_success {
            self.success_sessions.push(session);
        } else {
            // If we have a fix diff, create a pattern
            if let Some(diff) = fix_diff {
                for error_code in outcome.error_codes() {
                    let decisions: Vec<String> =
                        traces.iter().map(|t| t.decision_type.clone()).collect();

                    let pattern = FixPattern::new(error_code, &diff).with_decisions(decisions);

                    self.pattern_store.index_fix(pattern)?;
                }
            }
            self.failed_sessions.push(session);
        }

        Ok(())
    }

    /// Correlate an error with decisions that may have caused it
    ///
    /// # Arguments
    ///
    /// * `error_code` - The error code to analyze
    /// * `error_span` - The source span where the error occurred
    ///
    /// # Returns
    ///
    /// Error correlation with suspicious decisions and fix suggestions
    pub fn correlate_error(
        &self,
        error_code: &str,
        error_span: &SourceSpan,
    ) -> Result<ErrorCorrelation, crate::Error> {
        // Find decisions that overlap with the error span
        let overlapping_decisions = self.find_overlapping_decisions(error_span);

        // Calculate suspiciousness scores using Tarantula
        let mut suspicious: Vec<SuspiciousDecision> = overlapping_decisions
            .into_iter()
            .map(|d| {
                let score = self
                    .decision_stats
                    .get(&d.decision_type)
                    .map_or(0.0, DecisionStats::tarantula_score);

                SuspiciousDecision::new(
                    d.clone(),
                    score,
                    format!(
                        "Decision '{}' overlaps error span with suspiciousness {:.2}",
                        d.decision_type, score
                    ),
                )
            })
            .filter(|s| s.suspiciousness >= self.config.min_suspiciousness)
            .collect();

        // Sort by suspiciousness
        suspicious.sort_by(|a, b| {
            b.suspiciousness
                .partial_cmp(&a.suspiciousness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build dependency chain if enabled
        if self.config.enable_dependency_graph && !suspicious.is_empty() {
            suspicious = self.expand_with_dependencies(suspicious);
        }

        // Get fix suggestions from pattern store
        let decision_context: Vec<String> = suspicious
            .iter()
            .map(|s| s.decision.decision_type.clone())
            .collect();

        let fix_suggestions = self.pattern_store.suggest_fix(
            error_code,
            &decision_context,
            self.config.max_suggestions,
        )?;

        Ok(ErrorCorrelation {
            error_code: error_code.to_string(),
            error_span: error_span.clone(),
            suspicious_decisions: suspicious,
            fix_suggestions,
        })
    }

    /// Find decisions whose spans overlap with the given span
    fn find_overlapping_decisions(&self, span: &SourceSpan) -> Vec<&DecisionTrace> {
        let mut result = Vec::new();

        for session in &self.failed_sessions {
            for decision in &session.decisions {
                if let Some(decision_span) = &decision.span {
                    if decision_span.overlaps(span) {
                        result.push(decision);
                    }
                }
            }
        }

        result
    }

    /// Expand suspicious decisions with their dependencies
    fn expand_with_dependencies(
        &self,
        mut suspicious: Vec<SuspiciousDecision>,
    ) -> Vec<SuspiciousDecision> {
        let mut seen: HashSet<String> = suspicious.iter().map(|s| s.decision.id.clone()).collect();
        let mut i = 0;

        while i < suspicious.len() {
            let deps = suspicious[i].decision.depends_on.clone();

            for dep_id in deps {
                if seen.contains(&dep_id) {
                    continue;
                }

                // Find the dependency in failed sessions
                for session in &self.failed_sessions {
                    for decision in &session.decisions {
                        if decision.id == dep_id {
                            let score = self
                                .decision_stats
                                .get(&decision.decision_type)
                                .map_or(0.0, DecisionStats::tarantula_score);

                            // Reduce suspiciousness for indirect dependencies
                            let adjusted_score = score * 0.8;

                            if adjusted_score >= self.config.min_suspiciousness {
                                suspicious.push(SuspiciousDecision::new(
                                    decision.clone(),
                                    adjusted_score,
                                    "Dependency of suspicious decision (indirect)",
                                ));
                                seen.insert(dep_id.clone());
                            }
                            break;
                        }
                    }
                }
            }

            i += 1;
        }

        // Re-sort after adding dependencies
        suspicious.sort_by(|a, b| {
            b.suspiciousness
                .partial_cmp(&a.suspiciousness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        suspicious
    }

    /// Get the pattern store
    #[must_use]
    pub fn pattern_store(&self) -> &DecisionPatternStore {
        &self.pattern_store
    }

    /// Get mutable pattern store
    pub fn pattern_store_mut(&mut self) -> &mut DecisionPatternStore {
        &mut self.pattern_store
    }

    /// Get the number of successful sessions
    #[must_use]
    pub fn success_count(&self) -> usize {
        self.success_sessions.len()
    }

    /// Get the number of failed sessions
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.failed_sessions.len()
    }

    /// Get the total number of sessions
    #[must_use]
    pub fn session_count(&self) -> usize {
        self.success_sessions.len() + self.failed_sessions.len()
    }

    /// Get decision statistics
    #[must_use]
    pub fn decision_stats(&self) -> &HashMap<String, DecisionStats> {
        &self.decision_stats
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &CITLConfig {
        &self.config
    }

    /// Get top suspicious decision types (by Tarantula score)
    #[must_use]
    pub fn top_suspicious_types(&self, k: usize) -> Vec<(&str, f32)> {
        let mut scores: Vec<_> = self
            .decision_stats
            .iter()
            .map(|(t, s)| (t.as_str(), s.tarantula_score()))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    /// Group decisions by source file
    #[must_use]
    pub fn decisions_by_file(&self) -> HashMap<String, Vec<&DecisionTrace>> {
        let mut by_file: HashMap<String, Vec<&DecisionTrace>> = HashMap::new();

        for session in self
            .failed_sessions
            .iter()
            .chain(self.success_sessions.iter())
        {
            for decision in &session.decisions {
                if let Some(span) = &decision.span {
                    by_file.entry(span.file.clone()).or_default().push(decision);
                }
            }
        }

        by_file
    }

    /// Build a dependency graph for all decisions
    #[must_use]
    pub fn build_dependency_graph(&self) -> HashMap<String, Vec<String>> {
        let mut graph: HashMap<String, Vec<String>> = HashMap::new();

        for session in self
            .failed_sessions
            .iter()
            .chain(self.success_sessions.iter())
        {
            for decision in &session.decisions {
                graph
                    .entry(decision.id.clone())
                    .or_default()
                    .extend(decision.depends_on.clone());
            }
        }

        graph
    }

    /// Find the root causes (decisions with no dependencies that are suspicious)
    #[must_use]
    pub fn find_root_causes(&self, error_span: &SourceSpan) -> Vec<&DecisionTrace> {
        let overlapping = self.find_overlapping_decisions(error_span);
        let _graph = self.build_dependency_graph();

        // Find decisions that are depended upon but don't depend on others in the set
        let all_ids: HashSet<_> = overlapping.iter().map(|d| &d.id).collect();

        overlapping
            .into_iter()
            .filter(|d| {
                // Check if this decision's dependencies are outside the suspicious set
                d.depends_on.iter().all(|dep| !all_ids.contains(dep))
            })
            .collect()
    }
}

impl std::fmt::Debug for DecisionCITL {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionCITL")
            .field("success_sessions", &self.success_sessions.len())
            .field("failed_sessions", &self.failed_sessions.len())
            .field("decision_types", &self.decision_stats.len())
            .field("patterns", &self.pattern_store.len())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============ SourceSpan Tests ============

    #[test]
    fn test_source_span_new() {
        let span = SourceSpan::new("main.rs", 1, 1, 10, 80);
        assert_eq!(span.file, "main.rs");
        assert_eq!(span.start_line, 1);
        assert_eq!(span.end_line, 10);
    }

    #[test]
    fn test_source_span_line() {
        let span = SourceSpan::line("main.rs", 5);
        assert_eq!(span.file, "main.rs");
        assert_eq!(span.start_line, 5);
        assert_eq!(span.end_line, 5);
    }

    #[test]
    fn test_source_span_overlaps_same_line() {
        let span1 = SourceSpan::line("main.rs", 5);
        let span2 = SourceSpan::line("main.rs", 5);
        assert!(span1.overlaps(&span2));
    }

    #[test]
    fn test_source_span_overlaps_different_lines() {
        let span1 = SourceSpan::new("main.rs", 1, 1, 10, 80);
        let span2 = SourceSpan::new("main.rs", 5, 1, 15, 80);
        assert!(span1.overlaps(&span2));
    }

    #[test]
    fn test_source_span_no_overlap() {
        let span1 = SourceSpan::new("main.rs", 1, 1, 5, 80);
        let span2 = SourceSpan::new("main.rs", 10, 1, 15, 80);
        assert!(!span1.overlaps(&span2));
    }

    #[test]
    fn test_source_span_no_overlap_different_files() {
        let span1 = SourceSpan::line("main.rs", 5);
        let span2 = SourceSpan::line("lib.rs", 5);
        assert!(!span1.overlaps(&span2));
    }

    #[test]
    fn test_source_span_contains() {
        let outer = SourceSpan::new("main.rs", 1, 1, 20, 80);
        let inner = SourceSpan::new("main.rs", 5, 1, 10, 80);
        assert!(outer.contains(&inner));
        assert!(!inner.contains(&outer));
    }

    #[test]
    fn test_source_span_display() {
        let span = SourceSpan::new("main.rs", 5, 10, 5, 20);
        let display = format!("{span}");
        assert!(display.contains("main.rs"));
        assert!(display.contains("5"));
    }

    #[test]
    fn test_source_span_serialization() {
        let span = SourceSpan::line("main.rs", 5);
        let json = serde_json::to_string(&span).unwrap();
        let deserialized: SourceSpan = serde_json::from_str(&json).unwrap();
        assert_eq!(span, deserialized);
    }

    // ============ DecisionTrace Tests ============

    #[test]
    fn test_decision_trace_new() {
        let trace = DecisionTrace::new("d1", "type_inference", "Inferred type i32");
        assert_eq!(trace.id, "d1");
        assert_eq!(trace.decision_type, "type_inference");
        assert_eq!(trace.description, "Inferred type i32");
    }

    #[test]
    fn test_decision_trace_with_span() {
        let trace =
            DecisionTrace::new("d1", "type", "desc").with_span(SourceSpan::line("main.rs", 5));
        assert!(trace.span.is_some());
    }

    #[test]
    fn test_decision_trace_with_timestamp() {
        let trace = DecisionTrace::new("d1", "type", "desc").with_timestamp(1000);
        assert_eq!(trace.timestamp_ns, 1000);
    }

    #[test]
    fn test_decision_trace_with_dependencies() {
        let trace = DecisionTrace::new("d1", "type", "desc")
            .with_dependency("d0")
            .with_dependencies(vec!["d2".to_string(), "d3".to_string()]);
        assert_eq!(trace.depends_on.len(), 3);
    }

    // ============ CompilationOutcome Tests ============

    #[test]
    fn test_compilation_outcome_success() {
        let outcome = CompilationOutcome::success();
        assert!(outcome.is_success());
        assert!(outcome.error_codes().is_empty());
    }

    #[test]
    fn test_compilation_outcome_failure() {
        let outcome = CompilationOutcome::failure(
            vec!["E0308".to_string()],
            vec![SourceSpan::line("main.rs", 5)],
            vec!["type mismatch".to_string()],
        );
        assert!(!outcome.is_success());
        assert_eq!(outcome.error_codes(), vec!["E0308"]);
        assert_eq!(outcome.error_spans().len(), 1);
    }

    // ============ SuspiciousDecision Tests ============

    #[test]
    fn test_suspicious_decision_new() {
        let trace = DecisionTrace::new("d1", "type", "desc");
        let suspicious = SuspiciousDecision::new(trace, 0.8, "high suspicion");
        assert_eq!(suspicious.suspiciousness, 0.8);
    }

    #[test]
    fn test_suspicious_decision_clamped() {
        let trace = DecisionTrace::new("d1", "type", "desc");
        let suspicious = SuspiciousDecision::new(trace, 1.5, "over max");
        assert_eq!(suspicious.suspiciousness, 1.0);
    }

    // ============ DecisionStats Tests ============

    #[test]
    fn test_decision_stats_tarantula() {
        let mut stats = DecisionStats::default();
        stats.success_count = 2;
        stats.fail_count = 8;
        stats.total_success = 10;
        stats.total_fail = 10;

        // fail_freq = 8/10 = 0.8
        // success_freq = 2/10 = 0.2
        // suspiciousness = 0.8 / (0.8 + 0.2) = 0.8
        assert!((stats.tarantula_score() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_decision_stats_tarantula_no_failures() {
        let stats = DecisionStats {
            success_count: 5,
            fail_count: 0,
            total_success: 5,
            total_fail: 0,
        };
        assert_eq!(stats.tarantula_score(), 0.0);
    }

    #[test]
    fn test_decision_stats_tarantula_only_failures() {
        let stats = DecisionStats {
            success_count: 0,
            fail_count: 5,
            total_success: 0,
            total_fail: 5,
        };
        // fail_freq = 1.0, success_freq = 0.0
        // suspiciousness = 1.0 / 1.0 = 1.0
        assert_eq!(stats.tarantula_score(), 1.0);
    }

    // ============ CITLConfig Tests ============

    #[test]
    fn test_citl_config_default() {
        let config = CITLConfig::default();
        assert_eq!(config.max_suggestions, 5);
        assert!((config.min_suspiciousness - 0.3).abs() < 0.01);
        assert!(config.enable_dependency_graph);
    }

    // ============ DecisionCITL Tests ============

    #[test]
    fn test_decision_citl_new() {
        let trainer = DecisionCITL::new().unwrap();
        assert_eq!(trainer.session_count(), 0);
        assert_eq!(trainer.success_count(), 0);
        assert_eq!(trainer.failure_count(), 0);
    }

    #[test]
    fn test_decision_citl_ingest_success() {
        let mut trainer = DecisionCITL::new().unwrap();

        let traces = vec![DecisionTrace::new("d1", "type_inference", "desc")];
        let outcome = CompilationOutcome::success();

        trainer.ingest_session(traces, outcome, None).unwrap();

        assert_eq!(trainer.session_count(), 1);
        assert_eq!(trainer.success_count(), 1);
        assert_eq!(trainer.failure_count(), 0);
    }

    #[test]
    fn test_decision_citl_ingest_failure() {
        let mut trainer = DecisionCITL::new().unwrap();

        let traces = vec![DecisionTrace::new("d1", "type_inference", "desc")];
        let outcome = CompilationOutcome::failure(
            vec!["E0308".to_string()],
            vec![SourceSpan::line("main.rs", 5)],
            vec![],
        );

        trainer.ingest_session(traces, outcome, None).unwrap();

        assert_eq!(trainer.failure_count(), 1);
    }

    #[test]
    fn test_decision_citl_ingest_with_fix() {
        let mut trainer = DecisionCITL::new().unwrap();

        let traces = vec![DecisionTrace::new("d1", "type_inference", "desc")];
        let outcome = CompilationOutcome::failure(vec!["E0308".to_string()], vec![], vec![]);
        let fix = Some("- i32\n+ &str".to_string());

        trainer.ingest_session(traces, outcome, fix).unwrap();

        // Pattern should be indexed
        assert_eq!(trainer.pattern_store().len(), 1);
    }

    #[test]
    fn test_decision_citl_correlate_error() {
        let mut trainer = DecisionCITL::new().unwrap();

        // Ingest a failed session
        let traces = vec![
            DecisionTrace::new("d1", "type_inference", "Inferred wrong type")
                .with_span(SourceSpan::line("main.rs", 5)),
        ];
        let outcome = CompilationOutcome::failure(
            vec!["E0308".to_string()],
            vec![SourceSpan::line("main.rs", 5)],
            vec![],
        );
        trainer.ingest_session(traces, outcome, None).unwrap();

        // Correlate
        let error_span = SourceSpan::line("main.rs", 5);
        let correlation = trainer.correlate_error("E0308", &error_span).unwrap();

        assert_eq!(correlation.error_code, "E0308");
    }

    #[test]
    fn test_decision_citl_top_suspicious_types() {
        let mut trainer = DecisionCITL::new().unwrap();

        // Add some sessions
        for _ in 0..5 {
            trainer
                .ingest_session(
                    vec![DecisionTrace::new("d", "bad_decision", "")],
                    CompilationOutcome::failure(vec!["E0001".to_string()], vec![], vec![]),
                    None,
                )
                .unwrap();
        }

        for _ in 0..3 {
            trainer
                .ingest_session(
                    vec![DecisionTrace::new("d", "good_decision", "")],
                    CompilationOutcome::success(),
                    None,
                )
                .unwrap();
        }

        let top = trainer.top_suspicious_types(5);
        assert!(!top.is_empty());
    }

    #[test]
    fn test_decision_citl_decisions_by_file() {
        let mut trainer = DecisionCITL::new().unwrap();

        trainer
            .ingest_session(
                vec![
                    DecisionTrace::new("d1", "type", "").with_span(SourceSpan::line("main.rs", 1)),
                    DecisionTrace::new("d2", "type", "").with_span(SourceSpan::line("lib.rs", 1)),
                ],
                CompilationOutcome::success(),
                None,
            )
            .unwrap();

        let by_file = trainer.decisions_by_file();
        assert!(by_file.contains_key("main.rs"));
        assert!(by_file.contains_key("lib.rs"));
    }

    #[test]
    fn test_decision_citl_build_dependency_graph() {
        let mut trainer = DecisionCITL::new().unwrap();

        trainer
            .ingest_session(
                vec![
                    DecisionTrace::new("d1", "type", "").with_dependency("d0"),
                    DecisionTrace::new("d2", "type", "")
                        .with_dependencies(vec!["d0".to_string(), "d1".to_string()]),
                ],
                CompilationOutcome::success(),
                None,
            )
            .unwrap();

        let graph = trainer.build_dependency_graph();
        assert_eq!(graph.get("d1").unwrap(), &vec!["d0".to_string()]);
        assert_eq!(graph.get("d2").unwrap().len(), 2);
    }

    #[test]
    fn test_decision_citl_find_root_causes() {
        let mut trainer = DecisionCITL::new().unwrap();

        let span = SourceSpan::line("main.rs", 5);
        trainer
            .ingest_session(
                vec![
                    DecisionTrace::new("root", "type", "").with_span(span.clone()),
                    DecisionTrace::new("child", "type", "")
                        .with_span(span.clone())
                        .with_dependency("root"),
                ],
                CompilationOutcome::failure(vec!["E0308".to_string()], vec![span.clone()], vec![]),
                None,
            )
            .unwrap();

        let roots = trainer.find_root_causes(&span);
        assert!(!roots.is_empty());
        assert!(roots.iter().any(|r| r.id == "root"));
    }

    #[test]
    fn test_decision_citl_debug() {
        let trainer = DecisionCITL::new().unwrap();
        let debug = format!("{trainer:?}");
        assert!(debug.contains("DecisionCITL"));
    }

    // ============ Property Tests ============

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_source_span_overlap_symmetric(
            line1 in 1u32..100,
            line2 in 1u32..100
        ) {
            let span1 = SourceSpan::line("file.rs", line1);
            let span2 = SourceSpan::line("file.rs", line2);

            prop_assert_eq!(span1.overlaps(&span2), span2.overlaps(&span1));
        }

        #[test]
        fn prop_tarantula_score_bounded(
            success in 0u32..100,
            fail in 0u32..100,
            total_success in 1u32..100,
            total_fail in 1u32..100
        ) {
            let stats = DecisionStats {
                success_count: success.min(total_success),
                fail_count: fail.min(total_fail),
                total_success,
                total_fail,
            };

            let score = stats.tarantula_score();
            prop_assert!(score >= 0.0);
            prop_assert!(score <= 1.0);
        }

        #[test]
        fn prop_suspiciousness_clamped(score in -10.0f32..10.0) {
            let trace = DecisionTrace::new("d", "type", "desc");
            let suspicious = SuspiciousDecision::new(trace, score, "reason");
            prop_assert!(suspicious.suspiciousness >= 0.0);
            prop_assert!(suspicious.suspiciousness <= 1.0);
        }

        #[test]
        fn prop_session_count_consistent(
            n_success in 0usize..10,
            n_fail in 0usize..10
        ) {
            let mut trainer = DecisionCITL::new().unwrap();

            for _ in 0..n_success {
                trainer.ingest_session(
                    vec![DecisionTrace::new("d", "type", "")],
                    CompilationOutcome::success(),
                    None,
                ).unwrap();
            }

            for _ in 0..n_fail {
                trainer.ingest_session(
                    vec![DecisionTrace::new("d", "type", "")],
                    CompilationOutcome::failure(vec![], vec![], vec![]),
                    None,
                ).unwrap();
            }

            prop_assert_eq!(trainer.success_count(), n_success);
            prop_assert_eq!(trainer.failure_count(), n_fail);
            prop_assert_eq!(trainer.session_count(), n_success + n_fail);
        }
    }
}
