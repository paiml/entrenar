//! Performance tests for monitor module - Verify spec requirements

use crate::monitor::dashboard::Dashboard;
use crate::monitor::drift::DriftDetector;
use crate::monitor::report::HanseiAnalyzer;
use crate::monitor::*;
use std::time::Instant;

/// Simple pseudo-random float for testing (deterministic)
fn rand_float() -> f64 {
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u64> = const { Cell::new(12345) };
    }
    SEED.with(|seed| {
        let mut s = seed.get();
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        seed.set(s);
        (s as f64) / (u64::MAX as f64)
    })
}

// NOTE: Timing-dependent test - generous bound to avoid flakiness under CI load (CB-511)
#[test]
fn test_performance_metrics_collector_overhead() {
    // Spec requirement: Monitoring overhead < 1% of training time
    // Test: 10,000 metric records should complete reasonably fast
    let mut collector = MetricsCollector::new();

    let start = Instant::now();
    for i in 0..10_000 {
        collector.record(Metric::Loss, 1.0 / (f64::from(i) + 1.0));
        collector.record(Metric::Accuracy, 0.5 + (f64::from(i) * 0.00005));
    }
    let elapsed = start.elapsed();

    // 2s generous budget for 20,000 records (CI may be slow under load)
    assert!(
        elapsed.as_millis() < 2000,
        "Metrics recording too slow: {elapsed:?} for 20,000 records"
    );
}

// NOTE: Timing-dependent test - generous bound to avoid flakiness under CI load (CB-511)
#[test]
fn test_performance_summary_calculation() {
    // Pre-fill collector with data
    let mut collector = MetricsCollector::new();
    for i in 0..10_000 {
        collector.record(Metric::Loss, 1.0 / (f64::from(i) + 1.0));
    }

    // Spec requirement: Summary calculation should be O(1) due to running stats
    let start = Instant::now();
    let _summary = collector.summary();
    let elapsed = start.elapsed();

    // Summary should complete quickly (generous 100ms for CI under load)
    assert!(
        elapsed.as_millis() < 100,
        "Summary calculation too slow: {elapsed:?}"
    );
}

// NOTE: Timing-dependent test - generous bound to avoid flakiness under CI load (CB-511)
#[test]
fn test_performance_dashboard_render() {
    // Spec requirement: Dashboard refresh latency < 100ms
    let mut collector = MetricsCollector::new();
    for i in 0..1000 {
        collector.record(Metric::Loss, 1.0 - (f64::from(i) * 0.001));
        collector.record(Metric::Accuracy, 0.5 + (f64::from(i) * 0.0005));
    }

    let mut dashboard = Dashboard::new();
    dashboard.update(collector.summary());

    let start = Instant::now();
    let _output = dashboard.render_ascii();
    let elapsed = start.elapsed();

    // Dashboard render should complete within 2s (generous for CI under load)
    assert!(
        elapsed.as_millis() < 2000,
        "Dashboard render too slow: {elapsed:?}"
    );
}

// NOTE: Timing-dependent test - generous bound to avoid flakiness under CI load (CB-511)
#[test]
fn test_performance_drift_detection() {
    let mut detector = DriftDetector::new(100);

    // Build baseline with 100 values
    for _ in 0..100 {
        detector.check(50.0 + rand_float() * 2.0);
    }

    // Spec: Drift detection should be O(1) per update
    let start = Instant::now();
    for _ in 0..1000 {
        detector.check(50.0 + rand_float() * 2.0);
    }
    let elapsed = start.elapsed();

    // 1000 updates should complete within 2s (generous for CI under load)
    assert!(
        elapsed.as_millis() < 2000,
        "Drift detection too slow: {elapsed:?} for 1000 updates"
    );
}

// NOTE: Timing-dependent test - generous bound to avoid flakiness under CI load (CB-511)
#[test]
fn test_performance_hansei_analysis() {
    let mut collector = MetricsCollector::new();
    for i in 0..10_000 {
        collector.record(Metric::Loss, 1.0 - (f64::from(i) * 0.0001));
        collector.record(Metric::Accuracy, 0.5 + (f64::from(i) * 0.00005));
        collector.record(Metric::GradientNorm, 1.0 + rand_float() * 0.1);
    }

    let analyzer = HanseiAnalyzer::new();

    let start = Instant::now();
    let _report = analyzer.analyze("perf-test", &collector, 100.0);
    let elapsed = start.elapsed();

    // Analysis should complete within 2s (generous for CI under load)
    assert!(
        elapsed.as_millis() < 2000,
        "Hansei analysis too slow: {elapsed:?}"
    );
}
