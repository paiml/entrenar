//! Integration tests for HuggingFace pipeline
//!
//! These tests verify end-to-end functionality.

use super::*;

#[test]
fn test_module_exports() {
    // Verify all public types are accessible
    let _: FetchError = FetchError::MissingToken;
    let _: FetchOptions = FetchOptions::default();
    let _: WeightFormat = WeightFormat::SafeTensors;
}

#[test]
fn test_end_to_end_fetch_error_handling() {
    let temp_dir = std::env::temp_dir().join("hf_e2e_test");
    let _ = std::fs::remove_dir_all(&temp_dir);

    // Create fetcher with invalid token (intentionally)
    let fetcher = HfModelFetcher::with_token("test_token").cache_dir(&temp_dir);

    // Attempt download - should fail with appropriate error
    let result = fetcher.download_model(
        "nonexistent-org-12345/nonexistent-model-67890",
        FetchOptions::new()
            .files(&["model.safetensors", "config.json"])
            .cache_dir(&temp_dir),
    );

    // Expect error for nonexistent model
    assert!(result.is_err(), "Should fail for nonexistent model");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
#[ignore] // Requires network access and valid HF token
fn test_end_to_end_real_fetch() {
    let temp_dir = std::env::temp_dir().join("hf_e2e_real_test");
    let _ = std::fs::remove_dir_all(&temp_dir);

    // Create fetcher (uses HF_TOKEN env var if available)
    let fetcher = HfModelFetcher::new().unwrap().cache_dir(&temp_dir);

    // Download a small, known model
    let artifact = fetcher
        .download_model(
            "hf-internal-testing/tiny-random-bert",
            FetchOptions::new()
                .files(&["model.safetensors", "config.json"])
                .cache_dir(&temp_dir),
        )
        .expect("Download should succeed for test model");

    // Verify artifact
    assert_eq!(artifact.format, WeightFormat::SafeTensors);
    assert!(artifact.path.exists());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_distillation_pipeline_mock() {
    use ndarray::Array2;

    // Create mock teacher
    let teacher = SafeTensorsTeacher::mock(12, 768);

    // Verify teacher interface
    assert_eq!(teacher.num_layers(), 12);
    assert_eq!(teacher.hidden_size(), 768);

    // Mock forward pass
    let input = Array2::<f32>::zeros((4, 768));
    let output = teacher.forward(&input).expect("Forward should work");
    assert_eq!(output.dim(), (4, 768));

    // Get hidden states
    let hidden = teacher
        .hidden_states(&input)
        .expect("Hidden states should work");
    assert_eq!(hidden.len(), 12);

    // Memory estimation
    let mem = teacher.estimate_memory(32, 512);
    assert!(mem.total() > 0);
}

#[test]
fn test_full_distillation_loss_flow() {
    use ndarray::Array2;

    // Create loss function
    let loss_fn = DistillationLoss::new(4.0, 0.7);

    // Mock student and teacher outputs
    let student_logits =
        Array2::from_shape_vec((2, 10), (0..20).map(|x| x as f32).collect()).expect("Valid shape");
    let teacher_logits = Array2::from_shape_vec((2, 10), (0..20).map(|x| (x + 1) as f32).collect())
        .expect("Valid shape");
    let targets = vec![5, 3];

    // Compute loss
    let loss = loss_fn.forward(&student_logits, &teacher_logits, &targets);

    // Should be positive and finite
    assert!(loss >= 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_progressive_distillation_flow() {
    use ndarray::Array2;

    // Setup progressive distillation
    let prog = ProgressiveDistillation::new(vec![(0, 3), (1, 7), (2, 11)]).with_weight(0.5);

    // Mock hidden states (4 student layers, 12 teacher layers)
    let student_hidden: Vec<Array2<f32>> = (0..4).map(|_| Array2::zeros((4, 768))).collect();
    let teacher_hidden: Vec<Array2<f32>> = (0..12).map(|_| Array2::ones((4, 768))).collect();

    // Compute loss
    let loss = prog.hidden_state_loss(&student_hidden, &teacher_hidden);

    // Should be positive (student zeros, teacher ones)
    assert!(loss > 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_attention_transfer_flow() {
    use ndarray::Array2;

    let at = AttentionTransfer::new(0.1);

    // Mock attention weights
    let student_attn: Vec<Array2<f32>> = (0..4).map(|_| Array2::zeros((16, 16))).collect();
    let teacher_attn: Vec<Array2<f32>> = (0..12).map(|_| Array2::ones((16, 16))).collect();

    // Compute loss
    let loss = at.loss(&student_attn, &teacher_attn);

    // Should be positive
    assert!(loss > 0.0);
}

#[test]
fn test_security_pytorch_rejection() {
    let fetcher = HfModelFetcher::with_token("test");

    // Should reject PyTorch files by default
    let result = fetcher.download_model(
        "test/model",
        FetchOptions::new().files(&["pytorch_model.bin"]),
    );

    assert!(matches!(result, Err(FetchError::PickleSecurityRisk)));
}

#[test]
fn test_error_retryability() {
    let errors = vec![
        (
            FetchError::NetworkTimeout {
                repo: "r".into(),
                elapsed: std::time::Duration::from_secs(30),
            },
            true,
        ),
        (
            FetchError::RateLimited {
                retry_after: std::time::Duration::from_secs(60),
            },
            true,
        ),
        (FetchError::ModelNotFound { repo: "r".into() }, false),
        (FetchError::MissingToken, false),
    ];

    for (err, expected_retryable) in errors {
        assert_eq!(
            err.is_retryable(),
            expected_retryable,
            "Retryability mismatch for {:?}",
            err
        );
    }
}

#[test]
fn test_memory_estimation_realistic() {
    // CodeBERT-base: ~110M params
    let codebert_mem = MemoryEstimate::fp16(110_000_000, 32, 512, 768);
    // Should fit in 8GB
    assert!(codebert_mem.fits_in(8 * 1024 * 1024 * 1024));

    // StarCoder-1B: ~1B params
    let starcoder_mem = MemoryEstimate::fp16(1_000_000_000, 8, 2048, 2048);
    // 1B params * 2 bytes = 2GB, plus activations
    assert!(starcoder_mem.weights == 2_000_000_000);
    assert!(starcoder_mem.total() > 1_500_000_000); // At least 1.5GB

    // Llama-7B quantized: ~7B params
    let llama_int4 = MemoryEstimate::int4(7_000_000_000, 1, 4096, 4096);
    // Should fit in 16GB with INT4
    assert!(llama_int4.fits_in(16 * 1024 * 1024 * 1024));
}

// =========================================================================
// ENT-082: Integration Tests
// =========================================================================

#[test]
fn test_integration_full_pipeline_flow() {
    // 1. Create teacher model
    let teacher = SafeTensorsTeacher::mock(12, 768);

    // 2. Create trainer config
    let config = TrainerConfig::new("teacher/model", "student/model")
        .temperature(4.0)
        .alpha(0.7)
        .with_progressive(vec![(0, 3), (1, 7), (2, 11)])
        .epochs(3);

    // 3. Create trainer
    let mut trainer = DistillationTrainer::new(config, teacher);

    // 4. Simulate training loop
    for epoch in 0..3 {
        for step in 0..10 {
            let loss = 1.0 / (1.0 + (epoch * 10 + step) as f32);
            trainer.simulate_step(loss);
        }
        trainer.simulate_epoch();
    }

    // 5. Verify training state
    assert_eq!(trainer.state().epoch, 3);
    assert_eq!(trainer.state().global_step, 30);
    assert!(trainer.state().avg_loss(10).unwrap() < 0.5);
}

#[test]
fn test_integration_dataset_to_batch() {
    // 1. Create dataset
    let dataset = Dataset::mock(100, 64);
    assert_eq!(dataset.len(), 100);

    // 2. Create collator
    let collator = DistillationCollator::new(0).max_length(64);

    // 3. Create batches
    let batches = collator.batch_dataset(&dataset, 16);
    assert_eq!(batches.len(), 7); // 100 / 16 = 6 full + 1 partial

    // 4. Verify batch shapes
    assert_eq!(batches[0].batch_size(), 16);
    assert!(batches[0].max_seq_len() <= 64);
    assert!(batches[0].labels.is_some());
}

#[test]
fn test_integration_config_to_trainer() {
    let yaml = r#"
teacher:
  model_id: "microsoft/codebert-base"
student:
  model_id: "distilbert-base-uncased"
  lora:
    rank: 8
    alpha: 16.0
distillation:
  temperature: 6.0
  progressive:
    layer_mapping: [[0, 2], [1, 5]]
training:
  epochs: 5
  batch_size: 32
dataset:
  path: "wikitext"
"#;

    let config = DistillationYamlConfig::from_yaml(yaml).unwrap();
    assert!(config.validate().is_ok());

    let trainer_config = config.to_trainer_config().unwrap();
    assert_eq!(trainer_config.epochs, 5);
    assert!(trainer_config.progressive.is_some());
}

#[test]
fn test_integration_export_flow() {
    // 1. Create mock weights
    let weights = ModelWeights::mock(4, 256);

    // 2. Export to different formats
    let exporter = Exporter::new().output_dir("/tmp/hf_export_test");

    // SafeTensors
    let st_result = exporter.export(&weights, ExportFormat::SafeTensors, "model.safetensors");
    assert!(st_result.is_ok());

    // APR
    let apr_result = exporter.export(&weights, ExportFormat::APR, "model.apr.json");
    assert!(apr_result.is_ok());

    // GGUF
    let gguf_result = exporter.export(&weights, ExportFormat::GGUF, "model.gguf");
    assert!(gguf_result.is_ok());

    // Cleanup
    std::fs::remove_dir_all("/tmp/hf_export_test").ok();
}

#[test]
fn test_integration_teacher_cache() {
    use ndarray::Array2;

    let mut cache = TeacherCache::new();

    // Simulate caching teacher outputs
    for i in 0..100 {
        let logits = Array2::<f32>::zeros((4, 100));
        cache.cache_logits(i, logits);
    }

    // Verify cache stats
    let stats = cache.stats();
    assert_eq!(stats.logits_cached, 100);

    // Test hit rate after accesses
    for i in 0..50 {
        let _ = cache.get_logits(i);
    }
    for i in 100..110 {
        let _ = cache.get_logits(i);
    }

    let stats = cache.stats();
    assert!(stats.hit_rate() > 0.8); // Most should be hits
}

#[test]
fn test_integration_loss_with_trainer() {
    use ndarray::Array2;

    let teacher = SafeTensorsTeacher::mock(12, 768);
    let config = TrainerConfig::new("t", "s")
        .temperature(4.0)
        .alpha(0.7)
        .with_progressive(vec![(0, 0)])
        .with_attention_transfer(0.1);

    let trainer = DistillationTrainer::new(config, teacher);

    // Create mock data
    let student_logits = Array2::from_shape_fn((8, 100), |(i, j)| (i + j) as f32 * 0.01);
    let teacher_logits = Array2::from_shape_fn((8, 100), |(i, j)| (i + j + 1) as f32 * 0.01);
    let targets: Vec<usize> = (0..8).collect();

    let sh = vec![Array2::<f32>::zeros((8, 768))];
    let th = vec![Array2::<f32>::ones((8, 768))];
    let sa = vec![Array2::<f32>::zeros((8, 8))];
    let ta = vec![Array2::<f32>::ones((8, 8))];

    // Compute combined loss
    let loss = trainer.compute_loss(
        &student_logits,
        &teacher_logits,
        &targets,
        Some(&sh),
        Some(&th),
        Some(&sa),
        Some(&ta),
    );

    assert!(loss > 0.0);
    assert!(loss.is_finite());
}

// =========================================================================
// ENT-083: Property Tests
// =========================================================================

mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]

        /// Memory estimation should scale linearly with param count
        #[test]
        fn prop_memory_scales_linearly(
            param_count in 1_000_000u64..1_000_000_000,
            batch_size in 1usize..64,
            seq_len in 64usize..2048,
            hidden_size in 256usize..4096,
        ) {
            let mem1 = MemoryEstimate::fp16(param_count, batch_size, seq_len, hidden_size);
            let mem2 = MemoryEstimate::fp16(param_count * 2, batch_size, seq_len, hidden_size);

            // Weights should double
            prop_assert_eq!(mem2.weights, mem1.weights * 2);
        }

        /// Distillation loss should be positive for different logits
        #[test]
        fn prop_distillation_loss_positive(
            temp in 1.0f32..20.0,
            alpha in 0.0f32..1.0,
        ) {
            use ndarray::Array2;

            let loss_fn = DistillationLoss::new(temp, alpha);
            let student = Array2::from_shape_fn((2, 10), |(i, j)| (i + j) as f32);
            let teacher = Array2::from_shape_fn((2, 10), |(i, j)| (i + j + 1) as f32);

            let loss = loss_fn.forward(&student, &teacher, &[5, 3]);
            prop_assert!(loss >= 0.0);
            prop_assert!(loss.is_finite());
        }

        /// Collator should preserve sequence content
        #[test]
        fn prop_collator_preserves_content(
            seq_len in 1usize..100,
            batch_size in 1usize..16,
        ) {
            let examples: Vec<Example> = (0..batch_size)
                .map(|i| Example::from_tokens((0..seq_len).map(|j| (i * 1000 + j) as u32).collect()))
                .collect();

            let collator = DistillationCollator::new(0).max_length(512);
            let batch = collator.collate(&examples);

            // First token of each sequence should match
            for (i, example) in examples.iter().enumerate() {
                prop_assert_eq!(batch.input_ids[[i, 0]], example.input_ids[0]);
            }
        }

        /// Dataset shuffle should be deterministic with same seed
        #[test]
        fn prop_shuffle_deterministic(seed in 0u64..10000) {
            let mut ds1 = Dataset::mock(50, 16);
            let mut ds2 = Dataset::mock(50, 16);

            ds1.shuffle(seed);
            ds2.shuffle(seed);

            // Same seed should produce same order
            for (e1, e2) in ds1.examples().iter().zip(ds2.examples().iter()) {
                prop_assert_eq!(&e1.input_ids, &e2.input_ids);
            }
        }

        /// Export format detection should round-trip
        #[test]
        fn prop_format_detection_roundtrip(format_idx in 0usize..3) {
            let formats = [ExportFormat::SafeTensors, ExportFormat::APR, ExportFormat::GGUF];
            let format = formats[format_idx];

            let path = format!("model.{}", format.extension());
            let detected = ExportFormat::from_path(std::path::Path::new(&path));

            prop_assert_eq!(detected, Some(format));
        }

        /// Trainer state should track progress correctly
        #[test]
        fn prop_trainer_state_consistent(
            num_steps in 1usize..1000,
            num_epochs in 1usize..10,
        ) {
            let mut state = TrainingState::new();

            for _ in 0..num_epochs {
                for _ in 0..num_steps {
                    state.record_loss(1.0);
                    state.step();
                }
                state.new_epoch();
            }

            prop_assert_eq!(state.global_step, num_steps * num_epochs);
            prop_assert_eq!(state.epoch, num_epochs);
            prop_assert_eq!(state.loss_history.len(), num_steps * num_epochs);
        }

        /// Model weights should count params correctly
        #[test]
        fn prop_weights_param_count(
            num_tensors in 1usize..10,
            tensor_size in 100usize..1000,
        ) {
            let mut weights = ModelWeights::new();
            for i in 0..num_tensors {
                let data = vec![0.0f32; tensor_size];
                weights.add_tensor(format!("tensor_{}", i), data, vec![tensor_size]);
            }

            prop_assert_eq!(weights.param_count(), (num_tensors * tensor_size) as u64);
            prop_assert_eq!(weights.tensor_names().len(), num_tensors);
        }

        /// FineTuneConfig memory estimate should be positive
        #[test]
        fn prop_finetune_memory_positive(
            total_params in 1_000_000u64..10_000_000_000,
        ) {
            let config = FineTuneConfig::new("model");
            let mem = config.estimate_memory(total_params);

            prop_assert!(mem.total() > 0);
            prop_assert!(mem.model > 0);
        }
    }
}
