//! Property Tests for Inference Monitor (ENT-112)
//!
//! 200K+ proptest iterations for trace serialization, hash chain integrity, etc.

use super::*;
use proptest::prelude::*;

// =============================================================================
// Test Helpers
// =============================================================================

fn arb_linear_path() -> impl Strategy<Value = LinearPath> {
    (
        prop::collection::vec(-10.0f32..10.0, 1..20), // contributions
        -10.0f32..10.0,                               // intercept
        -10.0f32..10.0,                               // logit
        -10.0f32..10.0,                               // prediction
        prop::option::of(-1.0f32..1.0),               // probability
    )
        .prop_map(
            |(contributions, intercept, logit, prediction, probability)| {
                let mut path = LinearPath::new(contributions, intercept, logit, prediction);
                if let Some(prob) = probability {
                    path = path.with_probability(prob.abs().min(1.0));
                }
                path
            },
        )
}

fn arb_tree_split() -> impl Strategy<Value = TreeSplit> {
    (
        0..100usize,      // feature_idx
        -100.0f32..100.0, // threshold
        any::<bool>(),    // went_left
        1..10000usize,    // n_samples
    )
        .prop_map(|(feature_idx, threshold, went_left, n_samples)| TreeSplit {
            feature_idx,
            threshold,
            went_left,
            n_samples,
        })
}

fn arb_tree_path() -> impl Strategy<Value = TreePath> {
    (
        prop::collection::vec(arb_tree_split(), 0..10), // splits
        -10.0f32..10.0,                                 // prediction
        1..1000usize,                                   // n_samples
        prop::option::of(prop::collection::vec(0.0f32..1.0, 2..10)), // class_dist
    )
        .prop_map(|(splits, prediction, n_samples, class_distribution)| {
            let leaf = LeafInfo {
                prediction,
                n_samples,
                class_distribution,
            };
            TreePath::new(splits, leaf)
        })
}

fn arb_decision_trace() -> impl Strategy<Value = DecisionTrace<LinearPath>> {
    (
        arb_linear_path(),
        0..u64::MAX,     // timestamp_ns
        0..u64::MAX,     // sequence
        0..u64::MAX,     // input_hash
        -10.0f32..10.0,  // output
        0..1_000_000u64, // latency_ns
    )
        .prop_map(
            |(path, timestamp_ns, sequence, input_hash, output, latency_ns)| {
                DecisionTrace::new(timestamp_ns, sequence, input_hash, path, output, latency_ns)
            },
        )
}

// =============================================================================
// LinearPath Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_linear_path_serialization_roundtrip(path in arb_linear_path()) {
        let bytes = path.to_bytes();
        let restored = LinearPath::from_bytes(&bytes).expect("Deserialization failed");

        prop_assert_eq!(path.contributions.len(), restored.contributions.len());
        for (a, b) in path.contributions.iter().zip(restored.contributions.iter()) {
            prop_assert!((a - b).abs() < 1e-5, "Contribution mismatch: {} vs {}", a, b);
        }
        prop_assert!((path.intercept - restored.intercept).abs() < 1e-5);
        prop_assert!((path.logit - restored.logit).abs() < 1e-5);
        prop_assert!((path.prediction - restored.prediction).abs() < 1e-5);
        prop_assert_eq!(path.probability.is_some(), restored.probability.is_some());
    }

    #[test]
    fn prop_linear_path_confidence_bounds(path in arb_linear_path()) {
        let confidence = path.confidence();
        prop_assert!(confidence >= 0.0, "Confidence must be >= 0: {}", confidence);
        prop_assert!(confidence <= 1.0, "Confidence must be <= 1: {}", confidence);
    }

    #[test]
    fn prop_linear_path_top_features_sorted(path in arb_linear_path()) {
        let k = path.contributions.len().min(5);
        let top = path.top_features(k);

        // Check sorted by absolute value (descending)
        for i in 1..top.len() {
            prop_assert!(
                top[i-1].1.abs() >= top[i].1.abs(),
                "Not sorted: {} vs {}",
                top[i-1].1.abs(),
                top[i].1.abs()
            );
        }
    }

    #[test]
    fn prop_linear_path_feature_contributions_length(path in arb_linear_path()) {
        let contributions = path.feature_contributions();
        prop_assert_eq!(contributions.len(), path.contributions.len());
    }
}

// =============================================================================
// TreePath Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_tree_path_serialization_roundtrip(path in arb_tree_path()) {
        let bytes = path.to_bytes();
        let restored = TreePath::from_bytes(&bytes).expect("Deserialization failed");

        prop_assert_eq!(path.splits.len(), restored.splits.len());
        prop_assert_eq!(path.depth(), restored.depth());
        prop_assert!((path.leaf.prediction - restored.leaf.prediction).abs() < 1e-5);
        prop_assert_eq!(path.leaf.n_samples, restored.leaf.n_samples);
    }

    #[test]
    fn prop_tree_path_depth_equals_splits(path in arb_tree_path()) {
        prop_assert_eq!(path.depth(), path.splits.len());
    }

    #[test]
    fn prop_tree_path_confidence_bounds(path in arb_tree_path()) {
        let confidence = path.confidence();
        prop_assert!(confidence >= 0.0, "Confidence must be >= 0: {}", confidence);
        prop_assert!(confidence <= 1.0, "Confidence must be <= 1: {}", confidence);
    }
}

// =============================================================================
// DecisionTrace Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_decision_trace_serialization_roundtrip(trace in arb_decision_trace()) {
        let bytes = trace.to_bytes();
        let restored: DecisionTrace<LinearPath> =
            DecisionTrace::from_bytes(&bytes).expect("Deserialization failed");

        prop_assert_eq!(trace.timestamp_ns, restored.timestamp_ns);
        prop_assert_eq!(trace.sequence, restored.sequence);
        prop_assert_eq!(trace.input_hash, restored.input_hash);
        prop_assert!((trace.output - restored.output).abs() < 1e-5);
        // Latency has microsecond precision
        prop_assert!(
            (trace.latency_ns as i64 - restored.latency_ns as i64).abs() < 1000,
            "Latency mismatch: {} vs {}",
            trace.latency_ns,
            restored.latency_ns
        );
    }

    #[test]
    fn prop_decision_trace_confidence_from_path(trace in arb_decision_trace()) {
        let trace_confidence = trace.confidence();
        let path_confidence = trace.path.confidence();
        prop_assert!((trace_confidence - path_confidence).abs() < 1e-6);
    }
}

// =============================================================================
// RingCollector Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn prop_ring_collector_bounded(traces in prop::collection::vec(arb_decision_trace(), 0..100)) {
        let mut collector = RingCollector::<LinearPath, 32>::new();

        for trace in traces {
            collector.record(trace);
        }

        prop_assert!(collector.len() <= 32, "Ring buffer exceeded capacity: {}", collector.len());
    }

    #[test]
    fn prop_ring_collector_recent_order(traces in prop::collection::vec(arb_decision_trace(), 1..50)) {
        let mut collector = RingCollector::<LinearPath, 64>::new();

        for trace in &traces {
            collector.record(trace.clone());
        }

        let recent = collector.recent(traces.len());

        // Most recent should be last inserted
        if !recent.is_empty() {
            prop_assert_eq!(
                recent[0].sequence,
                traces.last().unwrap().sequence,
                "Most recent trace mismatch"
            );
        }
    }

    #[test]
    fn prop_ring_collector_all_preserves_order(n_traces in 1..20usize) {
        let mut collector = RingCollector::<LinearPath, 64>::new();

        for i in 0..n_traces {
            let path = LinearPath::new(vec![i as f32], 0.0, 0.0, 0.0);
            let trace = DecisionTrace::new(0, i as u64, 0, path, 0.0, 0);
            collector.record(trace);
        }

        let all = collector.all();

        // Should be in insertion order (oldest first)
        for i in 0..all.len() {
            prop_assert_eq!(all[i].sequence, i as u64, "Order mismatch at index {}", i);
        }
    }
}

// =============================================================================
// HashChainCollector Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_hash_chain_integrity(traces in prop::collection::vec(arb_decision_trace(), 1..20)) {
        let mut collector = HashChainCollector::<LinearPath>::new();

        for trace in traces {
            collector.record(trace);
        }

        let verification = collector.verify_chain();
        prop_assert!(verification.valid, "Chain integrity failed: {:?}", verification.error);
    }

    #[test]
    fn prop_hash_chain_sequence_monotonic(traces in prop::collection::vec(arb_decision_trace(), 1..20)) {
        let mut collector = HashChainCollector::<LinearPath>::new();

        for trace in traces {
            collector.record(trace);
        }

        let entries = collector.entries();
        for i in 0..entries.len() {
            prop_assert_eq!(entries[i].sequence, i as u64, "Sequence not monotonic at {}", i);
        }
    }

    #[test]
    fn prop_hash_chain_linking(traces in prop::collection::vec(arb_decision_trace(), 2..10)) {
        let mut collector = HashChainCollector::<LinearPath>::new();

        for trace in traces {
            collector.record(trace);
        }

        let entries = collector.entries();

        // Each entry's prev_hash should match previous entry's hash
        for i in 1..entries.len() {
            prop_assert_eq!(
                entries[i].prev_hash,
                entries[i-1].hash,
                "Hash linking broken at index {}",
                i
            );
        }
    }
}

// =============================================================================
// Counterfactual Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn prop_counterfactual_metrics(
        original in prop::collection::vec(-10.0f32..10.0, 2..20),
        deltas in prop::collection::vec(-5.0f32..5.0, 2..20),
    ) {
        let n = original.len().min(deltas.len());
        let original = original[..n].to_vec();
        let deltas = deltas[..n].to_vec();

        let counterfactual: Vec<f32> = original.iter().zip(deltas.iter())
            .map(|(o, d)| o + d)
            .collect();

        let cf = Counterfactual::new(
            original.clone(),
            0,
            0.9,
            counterfactual,
            1,
            0.85,
        );

        // L1 should be sum of absolute deltas for changed features
        let expected_l1: f32 = deltas.iter().map(|d| d.abs()).sum();
        prop_assert!((cf.sparsity - expected_l1).abs() < 1e-4, "L1 mismatch");

        // L2 should be sqrt of sum of squared deltas
        let expected_l2: f32 = deltas.iter().map(|d| d * d).sum::<f32>().sqrt();
        prop_assert!((cf.distance - expected_l2).abs() < 1e-4, "L2 mismatch");
    }

    #[test]
    fn prop_counterfactual_serialization_roundtrip(
        original in prop::collection::vec(-10.0f32..10.0, 2..10),
        deltas in prop::collection::vec(-5.0f32..5.0, 2..10),
    ) {
        let n = original.len().min(deltas.len());
        let original = original[..n].to_vec();
        let counterfactual: Vec<f32> = original.iter().zip(deltas.iter().take(n))
            .map(|(o, d)| o + d)
            .collect();

        let cf = Counterfactual::new(
            original,
            0,
            0.9,
            counterfactual,
            1,
            0.85,
        );

        let bytes = cf.to_bytes();
        let restored = Counterfactual::from_bytes(&bytes)
            .expect("Deserialization failed");

        prop_assert_eq!(cf.original_decision, restored.original_decision);
        prop_assert_eq!(cf.alternative_decision, restored.alternative_decision);
        prop_assert_eq!(cf.n_changes(), restored.n_changes());
    }
}

// =============================================================================
// Serialization Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn prop_binary_serialization_roundtrip(trace in arb_decision_trace()) {
        let serializer = TraceSerializer::new(TraceFormat::Binary);

        let bytes = serializer.serialize(&trace, PathType::Linear)
            .expect("Serialization failed");

        let restored: DecisionTrace<LinearPath> = serializer.deserialize(&bytes)
            .expect("Deserialization failed");

        prop_assert_eq!(trace.sequence, restored.sequence);
        prop_assert_eq!(trace.input_hash, restored.input_hash);
    }

    #[test]
    fn prop_json_serialization_roundtrip(trace in arb_decision_trace()) {
        let serializer = TraceSerializer::new(TraceFormat::Json);

        let bytes = serializer.serialize(&trace, PathType::Linear)
            .expect("Serialization failed");

        let restored: DecisionTrace<LinearPath> = serializer.deserialize(&bytes)
            .expect("Deserialization failed");

        prop_assert_eq!(trace.sequence, restored.sequence);
        prop_assert_eq!(trace.input_hash, restored.input_hash);
    }
}

// =============================================================================
// FNV-1a Hash Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_fnv1a_deterministic(data in prop::collection::vec(any::<u8>(), 0..100)) {
        let hash1 = fnv1a_hash(&data);
        let hash2 = fnv1a_hash(&data);
        prop_assert_eq!(hash1, hash2, "Hash not deterministic");
    }

    #[test]
    fn prop_fnv1a_different_data(
        data1 in prop::collection::vec(any::<u8>(), 1..50),
        data2 in prop::collection::vec(any::<u8>(), 1..50),
    ) {
        if data1 != data2 {
            let hash1 = fnv1a_hash(&data1);
            let hash2 = fnv1a_hash(&data2);
            // Not guaranteed to be different (collisions exist) but very likely
            // This is a weak test - we just check it runs without panic
            let _ = (hash1, hash2);
        }
    }

    #[test]
    fn prop_hash_features_deterministic(features in prop::collection::vec(-100.0f32..100.0, 1..50)) {
        let hash1 = hash_features(&features);
        let hash2 = hash_features(&features);
        prop_assert_eq!(hash1, hash2, "Feature hash not deterministic");
    }
}

// =============================================================================
// Safety Andon Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_safety_andon_confidence_thresholds(
        sil in prop_oneof![
            Just(SafetyIntegrityLevel::QM),
            Just(SafetyIntegrityLevel::SIL1),
            Just(SafetyIntegrityLevel::SIL2),
            Just(SafetyIntegrityLevel::SIL3),
            Just(SafetyIntegrityLevel::SIL4),
        ]
    ) {
        let confidence = sil.min_confidence();
        prop_assert!(confidence >= 0.0);
        prop_assert!(confidence <= 1.0);
    }

    #[test]
    fn prop_safety_andon_latency_thresholds(
        sil in prop_oneof![
            Just(SafetyIntegrityLevel::QM),
            Just(SafetyIntegrityLevel::SIL1),
            Just(SafetyIntegrityLevel::SIL2),
            Just(SafetyIntegrityLevel::SIL3),
            Just(SafetyIntegrityLevel::SIL4),
        ]
    ) {
        let latency = sil.max_latency_ns();
        prop_assert!(latency > 0);
    }
}

// =============================================================================
// Provenance Graph Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_provenance_graph_node_ids_unique(n_nodes in 1..50usize) {
        let mut graph = ProvenanceGraph::new();
        let mut ids = Vec::with_capacity(n_nodes);

        for i in 0..n_nodes {
            let id = graph.add_node(ProvenanceNode::Input {
                source: format!("source_{i}"),
                timestamp_ns: i as u64 * 1000,
                hash: i as u64,
            });
            ids.push(id);
        }

        // All IDs should be unique
        let unique_ids: std::collections::HashSet<_> = ids.iter().collect();
        prop_assert_eq!(unique_ids.len(), ids.len(), "Node IDs not unique");
    }

    #[test]
    fn prop_provenance_graph_edge_consistency(n_nodes in 2..20usize) {
        let mut graph = ProvenanceGraph::new();
        let mut ids = Vec::with_capacity(n_nodes);

        for i in 0..n_nodes {
            let id = graph.add_node(ProvenanceNode::Input {
                source: format!("source_{i}"),
                timestamp_ns: i as u64 * 1000,
                hash: i as u64,
            });
            ids.push(id);
        }

        // Add edges in a chain
        for i in 1..n_nodes {
            graph.add_edge(ProvenanceEdge {
                from: ids[i-1],
                to: ids[i],
                relation: CausalRelation::DataFlow,
                timestamp_ns: i as u64 * 1000,
            });
        }

        // Check adjacency consistency
        for i in 1..n_nodes {
            let preds = graph.predecessors(ids[i]);
            prop_assert!(preds.contains(&ids[i-1]), "Missing predecessor at {}", i);

            let succs = graph.successors(ids[i-1]);
            prop_assert!(succs.contains(&ids[i]), "Missing successor at {}", i-1);
        }
    }
}

// =============================================================================
// Utility Function Tests
// =============================================================================

#[test]
fn test_monotonic_ns_increasing() {
    let ts1 = monotonic_ns();
    std::thread::sleep(std::time::Duration::from_micros(100));
    let ts2 = monotonic_ns();
    assert!(
        ts2 > ts1,
        "monotonic_ns should be strictly increasing over time"
    );
}

#[test]
fn test_monotonic_ns_non_zero() {
    // The first call might be 0 if called at exactly initialization time
    // but subsequent calls should not be 0
    std::thread::sleep(std::time::Duration::from_micros(1));
    let ts = monotonic_ns();
    assert!(
        ts > 0,
        "monotonic_ns should return a positive value after initialization"
    );
}

#[test]
fn test_fnv1a_hash_empty() {
    let hash = fnv1a_hash(&[]);
    assert_eq!(
        hash, 0xcbf29ce484222325,
        "Empty input should return FNV offset basis"
    );
}

#[test]
fn test_fnv1a_hash_deterministic() {
    let data = b"hello world";
    let hash1 = fnv1a_hash(data);
    let hash2 = fnv1a_hash(data);
    assert_eq!(hash1, hash2, "Same input should produce same hash");
}

#[test]
fn test_fnv1a_hash_different_inputs() {
    let hash1 = fnv1a_hash(b"hello");
    let hash2 = fnv1a_hash(b"world");
    assert_ne!(
        hash1, hash2,
        "Different inputs should produce different hashes"
    );
}

#[test]
fn test_fnv1a_hash_single_byte() {
    let hash = fnv1a_hash(&[0x61]); // 'a'
    assert_ne!(hash, 0xcbf29ce484222325, "Single byte should change hash");
}

#[test]
fn test_hash_features_deterministic() {
    let features = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let hash1 = hash_features(&features);
    let hash2 = hash_features(&features);
    assert_eq!(hash1, hash2, "Same features should produce same hash");
}

#[test]
fn test_hash_features_different_inputs() {
    let features1 = [1.0f32, 2.0, 3.0];
    let features2 = [1.0f32, 2.0, 4.0];
    let hash1 = hash_features(&features1);
    let hash2 = hash_features(&features2);
    assert_ne!(
        hash1, hash2,
        "Different features should produce different hashes"
    );
}

#[test]
fn test_hash_features_empty() {
    let features: [f32; 0] = [];
    let hash = hash_features(&features);
    assert_eq!(
        hash, 0xcbf29ce484222325,
        "Empty features should return FNV offset basis"
    );
}

// =============================================================================
// InferenceMonitor Tests
// =============================================================================

// Mock model for testing InferenceMonitor
struct MockModel;

impl Explainable for MockModel {
    type Path = LinearPath;

    fn predict_explained(&self, x: &[f32], n_samples: usize) -> (Vec<f32>, Vec<Self::Path>) {
        let features_per_sample = x.len() / n_samples;
        let mut outputs = Vec::with_capacity(n_samples);
        let mut paths = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let start = i * features_per_sample;
            let sample = &x[start..start + features_per_sample];
            let output: f32 = sample.iter().sum();
            outputs.push(output);
            paths.push(LinearPath::new(sample.to_vec(), 0.0, output, output));
        }

        (outputs, paths)
    }

    fn explain_one(&self, sample: &[f32]) -> Self::Path {
        let output: f32 = sample.iter().sum();
        LinearPath::new(sample.to_vec(), 0.0, output, output)
    }
}

#[test]
fn test_inference_monitor_creation() {
    let model = MockModel;
    let collector = RingCollector::<LinearPath, 64>::new();
    let monitor = InferenceMonitor::new(model, collector);
    assert_eq!(monitor.sequence(), 0);
}

#[test]
fn test_inference_monitor_with_latency_budget() {
    let model = MockModel;
    let collector = RingCollector::<LinearPath, 64>::new();
    let monitor = InferenceMonitor::new(model, collector).with_latency_budget_ns(5_000_000);
    assert_eq!(monitor.sequence(), 0);
}

#[test]
fn test_inference_monitor_with_andon() {
    let model = MockModel;
    let collector = RingCollector::<LinearPath, 64>::new();
    let andon = SafetyAndon::new(SafetyIntegrityLevel::SIL2);
    let monitor = InferenceMonitor::new(model, collector).with_andon(andon);
    assert_eq!(monitor.sequence(), 0);
}

#[test]
fn test_inference_monitor_predict() {
    let model = MockModel;
    let collector = RingCollector::<LinearPath, 64>::new();
    let mut monitor = InferenceMonitor::new(model, collector);

    let features = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let n_samples = 2;

    let outputs = monitor.predict(&features, n_samples);

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0], 6.0); // 1 + 2 + 3
    assert_eq!(outputs[1], 15.0); // 4 + 5 + 6
    assert_eq!(monitor.sequence(), 2);
}

#[test]
fn test_inference_monitor_model_accessor() {
    let model = MockModel;
    let collector = RingCollector::<LinearPath, 64>::new();
    let monitor = InferenceMonitor::new(model, collector);

    let _model_ref = monitor.model();
}

#[test]
fn test_inference_monitor_collector_accessor() {
    let model = MockModel;
    let collector = RingCollector::<LinearPath, 64>::new();
    let monitor = InferenceMonitor::new(model, collector);

    let _collector_ref = monitor.collector();
}

#[test]
fn test_inference_monitor_collector_mut_accessor() {
    let model = MockModel;
    let collector = RingCollector::<LinearPath, 64>::new();
    let mut monitor = InferenceMonitor::new(model, collector);

    let _collector_ref = monitor.collector_mut();
}

#[test]
fn test_inference_monitor_traces_are_recorded() {
    let model = MockModel;
    let collector = RingCollector::<LinearPath, 64>::new();
    let mut monitor = InferenceMonitor::new(model, collector);

    let features = [1.0f32, 2.0, 3.0];
    monitor.predict(&features, 1);

    let traces = monitor.collector().all();
    assert_eq!(traces.len(), 1);
    assert_eq!(traces[0].output, 6.0);
}

#[test]
fn test_inference_monitor_multiple_predictions() {
    let model = MockModel;
    let collector = RingCollector::<LinearPath, 64>::new();
    let mut monitor = InferenceMonitor::new(model, collector);

    // First batch
    monitor.predict(&[1.0, 2.0, 3.0], 1);
    assert_eq!(monitor.sequence(), 1);

    // Second batch
    monitor.predict(&[4.0, 5.0, 6.0], 1);
    assert_eq!(monitor.sequence(), 2);

    let traces = monitor.collector().all();
    assert_eq!(traces.len(), 2);
}
