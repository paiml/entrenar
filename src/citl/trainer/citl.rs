//! Compiler-in-the-Loop (CITL) trainer for error-fix correlation
//!
//! Correlates compiler decision traces with compilation outcomes
//! for fault localization using statistical debugging techniques.
//!
//! # References
//! - Zeller (2002): Isolating Cause-Effect Chains
//! - Jones & Harrold (2005): Tarantula Fault Localization
//! - Chilimbi et al. (2009): HOLMES Statistical Debugging

#![allow(clippy::field_reassign_with_default)]

use super::stats::Session;
use super::{
    CITLConfig, CompilationOutcome, DecisionStats, DecisionTrace, ErrorCorrelation, SourceSpan,
    SuspiciousDecision,
};
use crate::citl::{DecisionPatternStore, FixPattern};
use std::collections::{HashMap, HashSet};

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
}

#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
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
