//! Tests for the CITL trainer

use super::*;
use crate::citl::trainer::{CompilationOutcome, DecisionTrace, SourceSpan};

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
