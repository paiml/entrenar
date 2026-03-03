use super::*;
use crate::autograd::Tensor;

#[test]
fn test_classification_head_shape() {
    let head = ClassificationHead::new(128, 5);
    assert_eq!(head.num_classes(), 5);
    assert_eq!(head.hidden_size(), 128);
    assert_eq!(head.num_parameters(), 128 * 5 + 5);
}

#[test]
fn test_classification_head_forward() {
    let head = ClassificationHead::new(64, 5);
    // Simulate hidden states: 3 tokens, hidden_size=64
    let hidden = Tensor::from_vec(vec![0.1f32; 3 * 64], false);
    let logits = head.forward(&hidden, 3);
    assert_eq!(logits.len(), 5, "F-CLASS-001: must produce 5 logits");
}

#[test]
fn test_classification_head_parameters() {
    let mut head = ClassificationHead::new(64, 5);
    assert_eq!(head.parameters().len(), 2); // weight + bias
    assert_eq!(head.parameters_mut().len(), 2);
}

#[test]
fn test_cross_entropy_loss_finite() {
    let logits = Tensor::from_vec(vec![1.0, 2.0, -1.0, 0.5, 3.0], false);
    let loss = cross_entropy_loss(&logits, 2, 5);
    let loss_val = loss.data()[0];
    assert!(loss_val.is_finite(), "F-CLASS-005: loss must be finite");
    assert!(loss_val > 0.0, "Cross-entropy loss must be positive");
}

#[test]
fn test_cross_entropy_loss_correct_class() {
    // If logit for target class is much larger, loss should be small
    let logits = Tensor::from_vec(vec![-100.0, -100.0, 100.0, -100.0, -100.0], false);
    let loss = cross_entropy_loss(&logits, 2, 5);
    let loss_val = loss.data()[0];
    assert!(loss_val < 0.01, "Loss for correct high-confidence should be ~0");
}

#[test]
fn test_cross_entropy_loss_wrong_class() {
    // If logit for target class is much smaller, loss should be large
    let logits = Tensor::from_vec(vec![100.0, -100.0, -100.0, -100.0, -100.0], false);
    let loss = cross_entropy_loss(&logits, 2, 5);
    let loss_val = loss.data()[0];
    assert!(loss_val > 1.0, "Loss for wrong class should be large");
}

#[test]
fn test_mean_pool() {
    let head = ClassificationHead::new(4, 2);
    // 2 tokens, hidden_size=4: [[1,2,3,4], [5,6,7,8]] → mean = [3,4,5,6]
    let hidden = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], false);
    let pooled = head.mean_pool(&hidden, 2);
    let data = pooled.data();
    let slice = data.as_slice().expect("contiguous");
    assert_eq!(slice.len(), 4);
    assert!((slice[0] - 3.0).abs() < 1e-6);
    assert!((slice[1] - 4.0).abs() < 1e-6);
    assert!((slice[2] - 5.0).abs() < 1e-6);
    assert!((slice[3] - 6.0).abs() < 1e-6);
}

#[test]
fn test_corpus_stats_empty() {
    let stats = corpus_stats(&[], 5);
    assert_eq!(stats.total, 0);
    assert_eq!(stats.class_counts, vec![0; 5]);
}

#[test]
fn test_corpus_stats_distribution() {
    let samples = vec![
        SafetySample { input: "echo hello".into(), label: 0 },
        SafetySample { input: "echo $HOME".into(), label: 1 },
        SafetySample { input: "echo $RANDOM".into(), label: 2 },
        SafetySample { input: "mkdir /tmp/x".into(), label: 3 },
        SafetySample { input: "eval $x".into(), label: 4 },
        SafetySample { input: "ls".into(), label: 0 },
    ];
    let stats = corpus_stats(&samples, 5);
    assert_eq!(stats.total, 6);
    assert_eq!(stats.class_counts, vec![2, 1, 1, 1, 1]);
}

#[test]
#[should_panic(expected = "F-CLASS-001")]
fn test_cross_entropy_wrong_logit_count() {
    let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
    let _ = cross_entropy_loss(&logits, 0, 5);
}

#[test]
#[should_panic(expected = "F-CLASS-002")]
fn test_cross_entropy_label_out_of_range() {
    let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], false);
    let _ = cross_entropy_loss(&logits, 5, 5);
}

// ── Multi-label tests ──────────────────────────────────────────

#[test]
fn test_multi_label_from_single_label() {
    let single = SafetySample { input: "echo $RANDOM".into(), label: 2 };
    let multi = MultiLabelSafetySample::from_single_label(&single, 5);
    assert_eq!(multi.labels, vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    assert_eq!(multi.active_classes(), vec![2]);
}

#[test]
fn test_multi_label_active_classes() {
    let sample = MultiLabelSafetySample {
        input: "echo $RANDOM $HOME".into(),
        labels: vec![0.0, 1.0, 1.0, 0.0, 0.0],
    };
    assert_eq!(sample.active_classes(), vec![1, 2]);
}

#[test]
fn test_multi_label_no_active_classes() {
    let sample =
        MultiLabelSafetySample { input: String::new(), labels: vec![0.0, 0.0, 0.0, 0.0, 0.0] };
    assert!(sample.active_classes().is_empty());
}

#[test]
fn test_multi_label_all_active() {
    let sample = MultiLabelSafetySample {
        input: "eval $RANDOM; mkdir /x".into(),
        labels: vec![1.0, 1.0, 1.0, 1.0, 1.0],
    };
    assert_eq!(sample.active_classes(), vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_bce_with_logits_loss_basic() {
    let logits = Tensor::from_vec(vec![2.0, -1.0, 0.5, -2.0, 3.0], false);
    let targets = [1.0, 0.0, 1.0, 0.0, 0.0];
    let loss = bce_with_logits_loss(&logits, &targets, 5);
    let loss_val = loss.data()[0];
    assert!(loss_val.is_finite(), "F-CLASS-005: loss must be finite");
    assert!(loss_val > 0.0, "BCE loss must be positive");
}

#[test]
fn test_bce_with_logits_loss_perfect() {
    let logits = Tensor::from_vec(vec![100.0, -100.0, 100.0, -100.0, -100.0], false);
    let targets = [1.0, 0.0, 1.0, 0.0, 0.0];
    let loss = bce_with_logits_loss(&logits, &targets, 5);
    assert!(loss.data()[0] < 0.01, "Perfect prediction should have near-zero loss");
}

#[test]
#[should_panic(expected = "F-CLASS-001")]
fn test_bce_logit_shape_mismatch() {
    let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
    let _ = bce_with_logits_loss(&logits, &[1.0, 0.0, 1.0, 0.0, 0.0], 5);
}

#[test]
#[should_panic(expected = "F-CLASS-001")]
fn test_bce_target_shape_mismatch() {
    let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], false);
    let _ = bce_with_logits_loss(&logits, &[1.0, 0.0, 1.0], 5);
}

// ── Class weight tests (SPEC-TUNE-2026-001) ─────────────────────

#[test]
fn test_class_weights_uniform() {
    let stats =
        SafetyCorpusStats { total: 100, class_counts: vec![60, 10, 10, 10, 10], avg_input_len: 50 };
    let weights = compute_class_weights(&stats, ClassWeightStrategy::Uniform, 5);
    assert_eq!(weights.len(), 5);
    for &w in &weights {
        assert!((w - 1.0).abs() < 1e-5, "Uniform weights should all be 1.0, got {w}");
    }
    let sum: f32 = weights.iter().sum();
    assert!((sum - 5.0).abs() < 1e-5, "F-TUNE-005: weights must sum to num_classes");
}

#[test]
fn test_class_weights_inverse_freq() {
    let stats =
        SafetyCorpusStats { total: 100, class_counts: vec![60, 10, 10, 10, 10], avg_input_len: 50 };
    let weights = compute_class_weights(&stats, ClassWeightStrategy::InverseFreq, 5);
    assert_eq!(weights.len(), 5);
    // Majority class (60 samples) should have lowest weight
    assert!(weights[0] < weights[1], "Majority class weight should be smallest");
    // Minority classes should have equal weights
    assert!(
        (weights[1] - weights[2]).abs() < 1e-5,
        "Equal-sized classes should have equal weights"
    );
    let sum: f32 = weights.iter().sum();
    assert!((sum - 5.0).abs() < 1e-5, "F-TUNE-005: weights must sum to num_classes");
}

#[test]
fn test_class_weights_sqrt_inverse() {
    let stats =
        SafetyCorpusStats { total: 100, class_counts: vec![60, 10, 10, 10, 10], avg_input_len: 50 };
    let weights = compute_class_weights(&stats, ClassWeightStrategy::SqrtInverse, 5);
    assert_eq!(weights.len(), 5);
    // Majority class should have lowest weight (but less extreme than inverse_freq)
    assert!(weights[0] < weights[1], "Majority class should have lowest weight");
    let sum: f32 = weights.iter().sum();
    assert!((sum - 5.0).abs() < 1e-5, "F-TUNE-005: weights must sum to num_classes");

    // sqrt_inverse should be less extreme than inverse_freq
    let inv_weights = compute_class_weights(&stats, ClassWeightStrategy::InverseFreq, 5);
    let inv_ratio = inv_weights[1] / inv_weights[0];
    let sqrt_ratio = weights[1] / weights[0];
    assert!(
        sqrt_ratio < inv_ratio,
        "sqrt_inverse should be less extreme than inverse_freq"
    );
}

#[test]
fn test_class_weights_strategy_parse() {
    assert_eq!(
        "uniform".parse::<ClassWeightStrategy>().ok(),
        Some(ClassWeightStrategy::Uniform)
    );
    assert_eq!(
        "inverse_freq".parse::<ClassWeightStrategy>().ok(),
        Some(ClassWeightStrategy::InverseFreq)
    );
    assert_eq!(
        "sqrt_inverse".parse::<ClassWeightStrategy>().ok(),
        Some(ClassWeightStrategy::SqrtInverse)
    );
    assert!("invalid".parse::<ClassWeightStrategy>().is_err());
}

#[test]
fn test_class_weights_strategy_aliases() {
    assert_eq!(
        "inverse".parse::<ClassWeightStrategy>().ok(),
        Some(ClassWeightStrategy::InverseFreq)
    );
    assert_eq!(
        "sqrt".parse::<ClassWeightStrategy>().ok(),
        Some(ClassWeightStrategy::SqrtInverse)
    );
}

#[test]
fn test_class_weights_strategy_display() {
    assert_eq!(format!("{}", ClassWeightStrategy::Uniform), "uniform");
    assert_eq!(format!("{}", ClassWeightStrategy::InverseFreq), "inverse_freq");
    assert_eq!(format!("{}", ClassWeightStrategy::SqrtInverse), "sqrt_inverse");
}

#[test]
fn test_class_weights_all_equal_classes() {
    let stats =
        SafetyCorpusStats { total: 50, class_counts: vec![10, 10, 10, 10, 10], avg_input_len: 30 };
    let inv_weights = compute_class_weights(&stats, ClassWeightStrategy::InverseFreq, 5);
    // All equal → all weights should be 1.0
    for &w in &inv_weights {
        assert!((w - 1.0).abs() < 1e-5, "Equal classes should all get weight 1.0, got {w}");
    }
}

// ── Corpus I/O tests ───────────────────────────────────────────

#[test]
fn test_load_safety_corpus_valid() {
    use std::io::Write;
    let dir = std::env::temp_dir().join("entrenar_test_corpus");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("valid.jsonl");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"input":"echo hello","label":0}}"#).unwrap();
        writeln!(f, r#"{{"input":"eval $x","label":4}}"#).unwrap();
        writeln!(f, "").unwrap(); // empty line should be skipped
        writeln!(f, r#"{{"input":"ls","label":1}}"#).unwrap();
    }
    let samples = load_safety_corpus(&path, 5).unwrap();
    assert_eq!(samples.len(), 3);
    assert_eq!(samples[0].label, 0);
    assert_eq!(samples[1].label, 4);
    assert_eq!(samples[2].label, 1);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_load_safety_corpus_label_out_of_range() {
    use std::io::Write;
    let path = std::env::temp_dir().join("entrenar_test_corpus_oor.jsonl");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"input":"bad","label":10}}"#).unwrap();
    }
    let result = load_safety_corpus(&path, 5);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("F-CLASS-002"), "Expected F-CLASS-002 error, got: {err}");
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_load_safety_corpus_file_not_found() {
    let result = load_safety_corpus(std::path::Path::new("/tmp/nonexistent_xyz_abc.jsonl"), 5);
    assert!(result.is_err());
}

#[test]
fn test_load_safety_corpus_invalid_json() {
    use std::io::Write;
    let path = std::env::temp_dir().join("entrenar_test_corpus_badjson.jsonl");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "not json at all").unwrap();
    }
    let result = load_safety_corpus(&path, 5);
    assert!(result.is_err());
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_load_multi_label_corpus_single_label_format() {
    use std::io::Write;
    let path = std::env::temp_dir().join("entrenar_test_ml_single.jsonl");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"input":"echo hi","label":2}}"#).unwrap();
    }
    let samples = load_multi_label_corpus(&path, 5).unwrap();
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].labels, vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_load_multi_label_corpus_multi_label_format() {
    use std::io::Write;
    let path = std::env::temp_dir().join("entrenar_test_ml_multi.jsonl");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"input":"eval $RANDOM","labels":[0.0,1.0,1.0,0.0,0.0]}}"#).unwrap();
    }
    let samples = load_multi_label_corpus(&path, 5).unwrap();
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].active_classes(), vec![1, 2]);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_load_multi_label_corpus_wrong_label_length() {
    use std::io::Write;
    let path = std::env::temp_dir().join("entrenar_test_ml_wronglen.jsonl");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"input":"x","labels":[1.0,0.0]}}"#).unwrap();
    }
    let result = load_multi_label_corpus(&path, 5);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("F-CLASS-001"), "Expected F-CLASS-001 error, got: {err}");
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_load_multi_label_corpus_single_label_out_of_range() {
    use std::io::Write;
    let path = std::env::temp_dir().join("entrenar_test_ml_oor.jsonl");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"input":"bad","label":99}}"#).unwrap();
    }
    let result = load_multi_label_corpus(&path, 5);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("F-CLASS-002"), "Expected F-CLASS-002 error, got: {err}");
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_load_multi_label_corpus_invalid_format() {
    use std::io::Write;
    let path = std::env::temp_dir().join("entrenar_test_ml_badfmt.jsonl");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"foo":"bar"}}"#).unwrap();
    }
    let result = load_multi_label_corpus(&path, 5);
    assert!(result.is_err());
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_load_multi_label_corpus_file_not_found() {
    let result =
        load_multi_label_corpus(std::path::Path::new("/tmp/nonexistent_ml_xyz.jsonl"), 5);
    assert!(result.is_err());
}

#[test]
fn test_safety_sample_input_ids() {
    let sample = SafetySample { input: "AB".into(), label: 0 };
    let ids = sample.input_ids();
    assert_eq!(ids, vec![65, 66]); // 'A'=65, 'B'=66
}

#[test]
fn test_classification_head_num_parameters() {
    let head = ClassificationHead::new(256, 10);
    assert_eq!(head.num_parameters(), 256 * 10 + 10);
}

#[test]
fn test_cross_entropy_uniform_logits() {
    // Uniform logits → loss should be -log(1/K) = log(K)
    let logits = Tensor::from_vec(vec![0.0; 5], false);
    let loss = cross_entropy_loss(&logits, 0, 5);
    let expected = (5.0f32).ln();
    assert!((loss.data()[0] - expected).abs() < 1e-4);
}

#[test]
fn test_bce_with_logits_loss_all_zeros() {
    let logits = Tensor::from_vec(vec![0.0; 5], false);
    let targets = [0.0, 0.0, 0.0, 0.0, 0.0];
    let loss = bce_with_logits_loss(&logits, &targets, 5);
    // BCE(0, 0) = max(0,0) - 0*0 + log(1+exp(0)) = 0 + log(2) ≈ 0.693
    let expected = 2.0f32.ln();
    assert!((loss.data()[0] - expected).abs() < 1e-4);
}
