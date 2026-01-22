//! Tests for pattern store module.

use super::*;

// ============ ChunkId Tests ============

#[test]
fn test_chunk_id_unique() {
    let id1 = ChunkId::new();
    let id2 = ChunkId::new();
    assert_ne!(id1, id2);
}

#[test]
fn test_chunk_id_display() {
    let id = ChunkId::new();
    let display = format!("{id}");
    assert!(!display.is_empty());
    assert!(display.contains('-')); // UUID format
}

#[test]
fn test_chunk_id_serialization() {
    let id = ChunkId::new();
    let json = serde_json::to_string(&id).unwrap();
    let deserialized: ChunkId = serde_json::from_str(&json).unwrap();
    assert_eq!(id, deserialized);
}

// ============ FixPattern Tests ============

#[test]
fn test_fix_pattern_new() {
    let pattern = FixPattern::new("E0308", "- old\n+ new");
    assert_eq!(pattern.error_code, "E0308");
    assert_eq!(pattern.fix_diff, "- old\n+ new");
    assert!(pattern.decision_sequence.is_empty());
    assert_eq!(pattern.success_count, 0);
    assert_eq!(pattern.attempt_count, 0);
}

#[test]
fn test_fix_pattern_with_decision() {
    let pattern = FixPattern::new("E0308", "diff")
        .with_decision("detect_mismatch")
        .with_decision("suggest_fix");

    assert_eq!(pattern.decision_sequence.len(), 2);
    assert_eq!(pattern.decision_sequence[0], "detect_mismatch");
    assert_eq!(pattern.decision_sequence[1], "suggest_fix");
}

#[test]
fn test_fix_pattern_with_decisions() {
    let decisions = vec!["step1".to_string(), "step2".to_string()];
    let pattern = FixPattern::new("E0308", "diff").with_decisions(decisions);

    assert_eq!(pattern.decision_sequence.len(), 2);
}

#[test]
fn test_fix_pattern_record_success() {
    let mut pattern = FixPattern::new("E0308", "diff");
    pattern.record_success();
    pattern.record_success();

    assert_eq!(pattern.success_count, 2);
    assert_eq!(pattern.attempt_count, 2);
}

#[test]
fn test_fix_pattern_record_failure() {
    let mut pattern = FixPattern::new("E0308", "diff");
    pattern.record_success();
    pattern.record_failure();

    assert_eq!(pattern.success_count, 1);
    assert_eq!(pattern.attempt_count, 2);
}

#[test]
fn test_fix_pattern_success_rate() {
    let mut pattern = FixPattern::new("E0308", "diff");
    assert_eq!(pattern.success_rate(), 0.0);

    pattern.record_success();
    pattern.record_success();
    pattern.record_failure();

    assert!((pattern.success_rate() - 0.666).abs() < 0.01);
}

#[test]
fn test_fix_pattern_to_searchable_text() {
    let pattern = FixPattern::new("E0308", "- i32\n+ &str").with_decision("type_mismatch");

    let text = pattern.to_searchable_text();
    assert!(text.contains("E0308"));
    assert!(text.contains("type_mismatch"));
    assert!(text.contains("- i32"));
}

#[test]
fn test_fix_pattern_serialization() {
    let pattern = FixPattern::new("E0308", "diff").with_decision("step1");

    let json = serde_json::to_string(&pattern).unwrap();
    let deserialized: FixPattern = serde_json::from_str(&json).unwrap();

    assert_eq!(pattern.error_code, deserialized.error_code);
    assert_eq!(pattern.fix_diff, deserialized.fix_diff);
    assert_eq!(pattern.decision_sequence, deserialized.decision_sequence);
}

// ============ FixSuggestion Tests ============

#[test]
fn test_fix_suggestion_new() {
    let pattern = FixPattern::new("E0308", "diff");
    let suggestion = FixSuggestion::new(pattern, 0.85, 0);

    assert_eq!(suggestion.score, 0.85);
    assert_eq!(suggestion.rank, 0);
}

#[test]
fn test_fix_suggestion_weighted_score() {
    let mut pattern = FixPattern::new("E0308", "diff");
    pattern.record_success();
    pattern.record_success();

    let suggestion = FixSuggestion::new(pattern, 0.8, 0);
    // weighted = 0.8 * (0.5 + 0.5 * 1.0) = 0.8 * 1.0 = 0.8
    assert!((suggestion.weighted_score() - 0.8).abs() < 0.01);
}

#[test]
fn test_fix_suggestion_weighted_score_partial_success() {
    let mut pattern = FixPattern::new("E0308", "diff");
    pattern.record_success();
    pattern.record_failure();

    let suggestion = FixSuggestion::new(pattern, 1.0, 0);
    // success_rate = 0.5, weighted = 1.0 * (0.5 + 0.5 * 0.5) = 1.0 * 0.75 = 0.75
    assert!((suggestion.weighted_score() - 0.75).abs() < 0.01);
}

// ============ PatternStoreConfig Tests ============

#[test]
fn test_pattern_store_config_default() {
    let config = PatternStoreConfig::default();
    assert_eq!(config.chunk_size, 256);
    assert_eq!(config.embedding_dim, 384);
    assert!((config.rrf_k - 60.0).abs() < 0.01);
}

// ============ DecisionPatternStore Tests ============

#[test]
fn test_pattern_store_new() {
    let store = DecisionPatternStore::new().unwrap();
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);
}

#[test]
fn test_pattern_store_with_config() {
    let config = PatternStoreConfig {
        chunk_size: 512,
        embedding_dim: 768,
        rrf_k: 30.0,
    };
    let store = DecisionPatternStore::with_config(config.clone()).unwrap();
    assert_eq!(store.config().chunk_size, 512);
    assert_eq!(store.config().embedding_dim, 768);
}

#[test]
fn test_pattern_store_index_fix() {
    let mut store = DecisionPatternStore::new().unwrap();

    let pattern = FixPattern::new("E0308", "- i32\n+ &str").with_decision("type_mismatch");

    store.index_fix(pattern).unwrap();

    assert_eq!(store.len(), 1);
    assert!(!store.is_empty());
}

#[test]
fn test_pattern_store_index_multiple() {
    let mut store = DecisionPatternStore::new().unwrap();

    store.index_fix(FixPattern::new("E0308", "fix1")).unwrap();
    store.index_fix(FixPattern::new("E0308", "fix2")).unwrap();
    store.index_fix(FixPattern::new("E0382", "fix3")).unwrap();

    assert_eq!(store.len(), 3);
}

#[test]
fn test_pattern_store_get() {
    let mut store = DecisionPatternStore::new().unwrap();

    let pattern = FixPattern::new("E0308", "diff");
    let id = pattern.id;

    store.index_fix(pattern).unwrap();

    let retrieved = store.get(&id);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().error_code, "E0308");
}

#[test]
fn test_pattern_store_get_mut() {
    let mut store = DecisionPatternStore::new().unwrap();

    let pattern = FixPattern::new("E0308", "diff");
    let id = pattern.id;

    store.index_fix(pattern).unwrap();

    if let Some(p) = store.get_mut(&id) {
        p.record_success();
    }

    assert_eq!(store.get(&id).unwrap().success_count, 1);
}

#[test]
fn test_pattern_store_record_outcome() {
    let mut store = DecisionPatternStore::new().unwrap();

    let pattern = FixPattern::new("E0308", "diff");
    let id = pattern.id;

    store.index_fix(pattern).unwrap();
    store.record_outcome(&id, true);
    store.record_outcome(&id, false);

    let p = store.get(&id).unwrap();
    assert_eq!(p.success_count, 1);
    assert_eq!(p.attempt_count, 2);
}

#[test]
fn test_pattern_store_patterns_for_error() {
    let mut store = DecisionPatternStore::new().unwrap();

    store.index_fix(FixPattern::new("E0308", "fix1")).unwrap();
    store.index_fix(FixPattern::new("E0308", "fix2")).unwrap();
    store.index_fix(FixPattern::new("E0382", "fix3")).unwrap();

    let e0308_patterns = store.patterns_for_error("E0308");
    assert_eq!(e0308_patterns.len(), 2);

    let e0382_patterns = store.patterns_for_error("E0382");
    assert_eq!(e0382_patterns.len(), 1);

    let e0000_patterns = store.patterns_for_error("E0000");
    assert!(e0000_patterns.is_empty());
}

#[test]
fn test_pattern_store_suggest_fix() {
    let mut store = DecisionPatternStore::new().unwrap();

    let pattern = FixPattern::new(
        "E0308",
        "- let x: i32 = \"hello\";\n+ let x: &str = \"hello\";",
    )
    .with_decision("type_mismatch_detected")
    .with_decision("infer_correct_type");

    store.index_fix(pattern).unwrap();

    let context = vec!["type_mismatch".to_string()];
    let suggestions = store.suggest_fix("E0308", &context, 5).unwrap();

    assert!(!suggestions.is_empty());
    assert_eq!(suggestions[0].pattern.error_code, "E0308");
}

#[test]
fn test_pattern_store_suggest_fix_empty() {
    let store = DecisionPatternStore::new().unwrap();

    let suggestions = store.suggest_fix("E0308", &[], 5).unwrap();
    assert!(suggestions.is_empty());
}

#[test]
fn test_pattern_store_suggest_fix_ranking() {
    let mut store = DecisionPatternStore::new().unwrap();

    // Pattern with high success rate
    let mut pattern1 = FixPattern::new("E0308", "fix1 high success");
    pattern1.record_success();
    pattern1.record_success();
    store.index_fix(pattern1).unwrap();

    // Pattern with low success rate
    let mut pattern2 = FixPattern::new("E0308", "fix2 low success");
    pattern2.record_failure();
    pattern2.record_failure();
    store.index_fix(pattern2).unwrap();

    let suggestions = store.suggest_fix("E0308", &[], 5).unwrap();

    // Higher success rate should be ranked first
    assert!(!suggestions.is_empty());
    // The first suggestion should have higher weighted score
    if suggestions.len() >= 2 {
        assert!(suggestions[0].weighted_score() >= suggestions[1].weighted_score());
    }
}

#[test]
fn test_pattern_store_export_import_json() {
    let mut store = DecisionPatternStore::new().unwrap();

    store.index_fix(FixPattern::new("E0308", "fix1")).unwrap();
    store.index_fix(FixPattern::new("E0382", "fix2")).unwrap();

    let json = store.export_json().unwrap();

    let mut new_store = DecisionPatternStore::new().unwrap();
    let count = new_store.import_json(&json).unwrap();

    assert_eq!(count, 2);
    assert_eq!(new_store.len(), 2);
}

#[test]
fn test_pattern_store_debug() {
    let mut store = DecisionPatternStore::new().unwrap();
    store.index_fix(FixPattern::new("E0308", "fix")).unwrap();

    let debug = format!("{store:?}");
    assert!(debug.contains("DecisionPatternStore"));
    assert!(debug.contains("pattern_count"));
}

// ============ Property Tests ============

use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_fix_pattern_success_rate_bounded(
        successes in 0u32..100,
        failures in 0u32..100
    ) {
        let mut pattern = FixPattern::new("E0308", "diff");
        for _ in 0..successes {
            pattern.record_success();
        }
        for _ in 0..failures {
            pattern.record_failure();
        }

        let rate = pattern.success_rate();
        prop_assert!(rate >= 0.0);
        prop_assert!(rate <= 1.0);
    }

    #[test]
    fn prop_fix_pattern_searchable_contains_error_code(
        error_code in "[A-Z][0-9]{4}"
    ) {
        let pattern = FixPattern::new(&error_code, "diff");
        let text = pattern.to_searchable_text();
        prop_assert!(text.contains(&error_code));
    }

    #[test]
    fn prop_chunk_id_serialization_roundtrip(n in 0u128..1000) {
        let id = ChunkId(uuid::Uuid::from_u128(n));
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: ChunkId = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(id, deserialized);
    }

    #[test]
    fn prop_suggestion_weighted_score_positive(
        score in 0.0f32..1.0,
        successes in 0u32..10,
        attempts in 1u32..20
    ) {
        let mut pattern = FixPattern::new("E0308", "diff");
        let actual_attempts = attempts.max(successes);
        for _ in 0..successes {
            pattern.record_success();
        }
        for _ in 0..(actual_attempts - successes) {
            pattern.record_failure();
        }

        let suggestion = FixSuggestion::new(pattern, score, 0);
        prop_assert!(suggestion.weighted_score() >= 0.0);
    }
}

// ============ APR Persistence Tests ============

#[test]
fn test_pattern_store_save_load_apr() {
    let mut store = DecisionPatternStore::new().unwrap();

    // Add patterns with various states
    let mut pattern1 = FixPattern::new("E0308", "- i32\n+ &str")
        .with_decision("type_mismatch")
        .with_decision("infer_string");
    pattern1.record_success();
    pattern1.record_success();
    store.index_fix(pattern1).unwrap();

    let pattern2 = FixPattern::new("E0382", "- x\n+ x.clone()").with_decision("borrow_after_move");
    store.index_fix(pattern2).unwrap();

    // Save to temp file
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("patterns.apr");

    store.save_apr(&path).unwrap();
    assert!(path.exists());

    // Load and verify
    let loaded = DecisionPatternStore::load_apr(&path).unwrap();
    assert_eq!(loaded.len(), 2);

    // Verify patterns for each error code
    let e0308_patterns = loaded.patterns_for_error("E0308");
    assert_eq!(e0308_patterns.len(), 1);
    assert_eq!(e0308_patterns[0].success_count, 2);
    assert_eq!(e0308_patterns[0].decision_sequence.len(), 2);

    let e0382_patterns = loaded.patterns_for_error("E0382");
    assert_eq!(e0382_patterns.len(), 1);
}

#[test]
fn test_pattern_store_apr_config_preserved() {
    let config = PatternStoreConfig {
        chunk_size: 512,
        embedding_dim: 768,
        rrf_k: 30.0,
    };
    let mut store = DecisionPatternStore::with_config(config).unwrap();
    store.index_fix(FixPattern::new("E0308", "fix")).unwrap();

    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("patterns.apr");

    store.save_apr(&path).unwrap();
    let loaded = DecisionPatternStore::load_apr(&path).unwrap();

    assert_eq!(loaded.config().chunk_size, 512);
    assert_eq!(loaded.config().embedding_dim, 768);
    assert!((loaded.config().rrf_k - 30.0).abs() < 0.01);
}

#[test]
fn test_pattern_store_apr_empty() {
    let store = DecisionPatternStore::new().unwrap();

    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("empty.apr");

    store.save_apr(&path).unwrap();
    let loaded = DecisionPatternStore::load_apr(&path).unwrap();

    assert!(loaded.is_empty());
}

#[test]
fn test_pattern_store_apr_suggest_after_load() {
    let mut store = DecisionPatternStore::new().unwrap();

    let pattern = FixPattern::new("E0308", "type fix").with_decision("detect_mismatch");
    store.index_fix(pattern).unwrap();

    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("patterns.apr");

    store.save_apr(&path).unwrap();
    let loaded = DecisionPatternStore::load_apr(&path).unwrap();

    // RAG index should be rebuilt and queryable
    let suggestions = loaded
        .suggest_fix("E0308", &["detect_mismatch".into()], 5)
        .unwrap();
    assert!(!suggestions.is_empty());
    assert_eq!(suggestions[0].pattern.error_code, "E0308");
}

proptest! {
    #[test]
    fn prop_apr_roundtrip_preserves_patterns(
        error_codes in proptest::collection::vec("[A-Z][0-9]{4}", 1..5),
        successes in proptest::collection::vec(0u32..10, 1..5)
    ) {
        let mut store = DecisionPatternStore::new().unwrap();

        for (i, code) in error_codes.iter().enumerate() {
            let mut pattern = FixPattern::new(code, format!("fix{i}"));
            let success_count = successes.get(i).copied().unwrap_or(0);
            for _ in 0..success_count {
                pattern.record_success();
            }
            store.index_fix(pattern).unwrap();
        }

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("patterns.apr");

        store.save_apr(&path).unwrap();
        let loaded = DecisionPatternStore::load_apr(&path).unwrap();

        prop_assert_eq!(store.len(), loaded.len());

        for code in &error_codes {
            let orig = store.patterns_for_error(code);
            let load = loaded.patterns_for_error(code);
            prop_assert_eq!(orig.len(), load.len());
        }
    }
}
