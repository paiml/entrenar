//! Tests for the monitor module - EXTREME TDD
//!
//! These tests are written BEFORE implementation.

use super::*;
use proptest::prelude::*;

// =============================================================================
// Unit Tests - MetricsCollector
// =============================================================================

#[test]
fn test_metrics_collector_new() {
    let collector = MetricsCollector::new();
    assert_eq!(collector.count(), 0);
    assert!(collector.is_empty());
}

#[test]
fn test_record_single_metric() {
    let mut collector = MetricsCollector::new();
    collector.record(Metric::Loss, 0.5);
    assert_eq!(collector.count(), 1);
    assert!(!collector.is_empty());
}

#[test]
fn test_record_multiple_metrics() {
    let mut collector = MetricsCollector::new();
    collector.record(Metric::Loss, 0.5);
    collector.record(Metric::Accuracy, 0.85);
    collector.record(Metric::LearningRate, 0.001);
    collector.record(Metric::GradientNorm, 1.2);
    assert_eq!(collector.count(), 4);
}

#[test]
fn test_record_batch() {
    let mut collector = MetricsCollector::new();
    collector.record_batch(&[
        (Metric::Loss, 0.5),
        (Metric::Accuracy, 0.85),
    ]);
    assert_eq!(collector.count(), 2);
}

#[test]
fn test_summary_single_value() {
    let mut collector = MetricsCollector::new();
    collector.record(Metric::Loss, 0.5);

    let summary = collector.summary();
    let loss_stats = summary.get(&Metric::Loss).unwrap();

    assert!((loss_stats.mean - 0.5).abs() < 1e-6);
    assert!((loss_stats.min - 0.5).abs() < 1e-6);
    assert!((loss_stats.max - 0.5).abs() < 1e-6);
    assert_eq!(loss_stats.count, 1);
}

#[test]
fn test_summary_multiple_values() {
    let mut collector = MetricsCollector::new();
    collector.record(Metric::Loss, 1.0);
    collector.record(Metric::Loss, 2.0);
    collector.record(Metric::Loss, 3.0);

    let summary = collector.summary();
    let loss_stats = summary.get(&Metric::Loss).unwrap();

    assert!((loss_stats.mean - 2.0).abs() < 1e-6);
    assert!((loss_stats.min - 1.0).abs() < 1e-6);
    assert!((loss_stats.max - 3.0).abs() < 1e-6);
    assert_eq!(loss_stats.count, 3);
}

#[test]
fn test_summary_std_dev() {
    let mut collector = MetricsCollector::new();
    // Values: 2, 4, 4, 4, 5, 5, 7, 9 -> mean=5, sample std≈2.138
    for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
        collector.record(Metric::Loss, v);
    }

    let summary = collector.summary();
    let loss_stats = summary.get(&Metric::Loss).unwrap();

    assert!((loss_stats.mean - 5.0).abs() < 1e-6);
    // Sample std = sqrt(32/7) ≈ 2.138
    assert!((loss_stats.std - 2.138).abs() < 0.1);
}

#[test]
fn test_clear() {
    let mut collector = MetricsCollector::new();
    collector.record(Metric::Loss, 0.5);
    collector.clear();
    assert!(collector.is_empty());
}

#[test]
fn test_to_records() {
    let mut collector = MetricsCollector::new();
    collector.record(Metric::Loss, 0.5);
    collector.record(Metric::Accuracy, 0.85);

    let records = collector.to_records();
    assert_eq!(records.len(), 2);
}

#[test]
fn test_metric_display() {
    assert_eq!(Metric::Loss.as_str(), "loss");
    assert_eq!(Metric::Accuracy.as_str(), "accuracy");
    assert_eq!(Metric::LearningRate.as_str(), "learning_rate");
    assert_eq!(Metric::GradientNorm.as_str(), "gradient_norm");
}

#[test]
fn test_metric_from_str() {
    assert_eq!(Metric::from_str("loss"), Some(Metric::Loss));
    assert_eq!(Metric::from_str("accuracy"), Some(Metric::Accuracy));
    assert_eq!(Metric::from_str("unknown"), None);
}

#[test]
fn test_metric_record_timestamp() {
    let mut collector = MetricsCollector::new();
    collector.record(Metric::Loss, 0.5);

    let records = collector.to_records();
    assert!(records[0].timestamp > 0);
}

#[test]
fn test_json_export() {
    let mut collector = MetricsCollector::new();
    collector.record(Metric::Loss, 0.5);

    let json = collector.to_json().unwrap();
    // Check for "Loss" (enum variant) or value
    assert!(json.contains("Loss") || json.contains("loss"));
    assert!(json.contains("0.5"));
}

// =============================================================================
// Property Tests
// =============================================================================

proptest! {
    #[test]
    fn prop_mean_within_bounds(values in proptest::collection::vec(-1000.0f64..1000.0, 1..100)) {
        let mut collector = MetricsCollector::new();
        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        for v in &values {
            collector.record(Metric::Loss, *v);
        }

        let summary = collector.summary();
        let stats = summary.get(&Metric::Loss).unwrap();

        prop_assert!(stats.mean >= min_val);
        prop_assert!(stats.mean <= max_val);
    }

    #[test]
    fn prop_std_non_negative(values in proptest::collection::vec(-1000.0f64..1000.0, 2..100)) {
        let mut collector = MetricsCollector::new();
        for v in &values {
            collector.record(Metric::Loss, *v);
        }

        let summary = collector.summary();
        let stats = summary.get(&Metric::Loss).unwrap();

        prop_assert!(stats.std >= 0.0);
    }

    #[test]
    fn prop_count_matches_insertions(values in proptest::collection::vec(-1000.0f64..1000.0, 1..100)) {
        let mut collector = MetricsCollector::new();
        for v in &values {
            collector.record(Metric::Loss, *v);
        }

        let summary = collector.summary();
        let stats = summary.get(&Metric::Loss).unwrap();

        prop_assert_eq!(stats.count, values.len());
    }

    #[test]
    fn prop_min_max_correct(values in proptest::collection::vec(-1000.0f64..1000.0, 1..100)) {
        let mut collector = MetricsCollector::new();
        let expected_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let expected_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        for v in &values {
            collector.record(Metric::Loss, *v);
        }

        let summary = collector.summary();
        let stats = summary.get(&Metric::Loss).unwrap();

        prop_assert!((stats.min - expected_min).abs() < 1e-10);
        prop_assert!((stats.max - expected_max).abs() < 1e-10);
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_empty_summary() {
    let collector = MetricsCollector::new();
    let summary = collector.summary();
    assert!(summary.is_empty());
}

#[test]
fn test_nan_handling() {
    let mut collector = MetricsCollector::new();
    collector.record(Metric::Loss, f64::NAN);

    let summary = collector.summary();
    let stats = summary.get(&Metric::Loss).unwrap();

    // NaN values should be detected
    assert!(stats.has_nan);
}

#[test]
fn test_inf_handling() {
    let mut collector = MetricsCollector::new();
    collector.record(Metric::Loss, f64::INFINITY);

    let summary = collector.summary();
    let stats = summary.get(&Metric::Loss).unwrap();

    // Inf values should be detected
    assert!(stats.has_inf);
}

#[test]
fn test_custom_metric() {
    let mut collector = MetricsCollector::new();
    collector.record(Metric::Custom("my_metric".to_string()), 42.0);

    let summary = collector.summary();
    assert!(summary.contains_key(&Metric::Custom("my_metric".to_string())));
}
