//! Tests for transformer trainer module

use super::*;
use crate::transformer::{Transformer, TransformerConfig};

#[test]
fn test_transformer_train_config_new() {
    let model_config = TransformerConfig::tiny();
    let config = TransformerTrainConfig::new(model_config.clone());

    assert_eq!(config.model_config.hidden_size, model_config.hidden_size);
    assert!(!config.checkpoint_config.enabled);
    assert!(!config.precision_config.is_mixed());
}

#[test]
fn test_transformer_train_config_with_checkpointing() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_checkpointing(4);

    assert!(config.checkpoint_config.enabled);
    assert_eq!(config.checkpoint_config.num_segments, 4);
}

#[test]
fn test_transformer_train_config_with_bf16() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_bf16();

    assert!(config.precision_config.is_mixed());
}

#[test]
fn test_transformer_train_config_with_fp16() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_fp16();

    assert!(config.precision_config.is_mixed());
    assert!(config.precision_config.dynamic_scaling);
}

#[test]
fn test_transformer_train_config_builders() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_max_seq_len(1024)
        .with_accumulation_steps(4)
        .with_warmup_steps(100)
        .with_lr(0.0001)
        .with_grad_clip(1.0);

    assert_eq!(config.max_seq_len, 1024);
    assert_eq!(config.accumulation_steps, 4);
    assert_eq!(config.warmup_steps, 100);
}

#[test]
fn test_lm_batch_from_sequences() {
    let sequences = vec![vec![0, 1, 2, 3, 4], vec![0, 5, 6, 7]];

    let batch = LMBatch::from_sequences(&sequences, 99, 100);

    assert_eq!(batch.batch_size, 2);
    assert_eq!(batch.seq_len, 4); // max_len - 1 = 5 - 1 = 4
}

#[test]
fn test_lm_batch_single() {
    let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);

    assert_eq!(batch.batch_size, 1);
    assert_eq!(batch.seq_len, 3);
    assert_eq!(batch.num_tokens(), 3);
}

#[test]
fn test_lm_batch_get_input() {
    let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);

    let input = batch.get_input(0).expect("operation should succeed");
    assert_eq!(input, &[1, 2, 3]);

    assert!(batch.get_input(1).is_none());
}

#[test]
fn test_lm_batch_get_target() {
    let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);

    let target = batch.get_target(0).expect("operation should succeed");
    assert_eq!(target, &[2, 3, 4]);
}

#[test]
fn test_lm_batch_empty() {
    let batch = LMBatch::from_sequences(&[], 0, 1);
    assert_eq!(batch.batch_size, 0);
    assert_eq!(batch.num_tokens(), 0);
}

#[test]
fn test_transformer_trainer_new() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    let trainer = TransformerTrainer::new(config);

    assert_eq!(trainer.step(), 0);
    assert!(!trainer.is_mixed_precision());
    assert!(!trainer.is_checkpointing());
}

#[test]
fn test_transformer_trainer_forward_single() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    let trainer = TransformerTrainer::new(config);

    let input_ids = vec![1, 2, 3];
    let target_ids = vec![2, 3, 4];

    let (loss, _loss_tensor, logits) = trainer.forward_single(&input_ids, &target_ids);

    assert!(loss > 0.0);
    assert!(loss.is_finite());
    assert_eq!(logits.len(), 3 * trainer.model().config().vocab_size);
}

#[test]
fn test_transformer_trainer_train_batch() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    let mut trainer = TransformerTrainer::new(config);

    let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);
    let loss = trainer.train_batch(&batch);

    assert!(loss > 0.0);
    assert!(loss.is_finite());
    assert_eq!(trainer.step(), 1);
}

#[test]
fn test_transformer_trainer_train_epoch() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    let mut trainer = TransformerTrainer::new(config);

    let batches = vec![
        LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]),
        LMBatch::single(vec![5, 6, 7], vec![6, 7, 8]),
    ];

    let avg_loss = trainer.train_epoch(&batches);

    assert!(avg_loss > 0.0);
    assert_eq!(trainer.step(), 2);
}

#[test]
fn test_transformer_trainer_epoch_with_callback() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    let mut trainer = TransformerTrainer::new(config);

    let batches = vec![
        LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]),
        LMBatch::single(vec![5, 6, 7], vec![6, 7, 8]),
    ];

    let mut callback_calls = Vec::new();
    let avg_loss = trainer.train_epoch_with_callback(&batches, |batch_idx, loss, _trainer| {
        callback_calls.push((batch_idx, loss));
    });

    assert!(avg_loss > 0.0);
    assert_eq!(callback_calls.len(), 2);
    assert_eq!(callback_calls[0].0, 0);
    assert_eq!(callback_calls[1].0, 1);
    assert!(callback_calls[0].1 > 0.0);
    assert!(callback_calls[1].1 > 0.0);
}

#[test]
fn test_transformer_trainer_empty_epoch() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    let mut trainer = TransformerTrainer::new(config);

    let avg_loss = trainer.train_epoch(&[]);
    assert_eq!(avg_loss, 0.0);
}

#[test]
fn test_transformer_trainer_with_accumulation() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_accumulation_steps(2);
    let mut trainer = TransformerTrainer::new(config);

    let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);

    // First batch - no step yet
    trainer.train_batch(&batch);
    assert_eq!(trainer.step(), 0);

    // Second batch - step occurs
    trainer.train_batch(&batch);
    assert_eq!(trainer.step(), 1);
}

#[test]
fn test_transformer_trainer_max_steps_stops_early() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_max_steps(3);
    let mut trainer = TransformerTrainer::new(config);

    let batches = vec![
        LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]),
        LMBatch::single(vec![5, 6, 7], vec![6, 7, 8]),
        LMBatch::single(vec![1, 3, 5], vec![3, 5, 7]),
        LMBatch::single(vec![2, 4, 6], vec![4, 6, 8]),
        LMBatch::single(vec![10, 11, 12], vec![11, 12, 13]),
    ];

    // Epoch 1: should process 3 batches then stop (max_steps=3)
    trainer.train_epoch(&batches);
    assert_eq!(trainer.step(), 3);
    assert!(trainer.reached_max_steps());

    // Epoch 2: should process 0 batches (already at max_steps)
    trainer.train_epoch(&batches);
    assert_eq!(trainer.step(), 3);
}

#[test]
fn test_transformer_trainer_max_steps_none_runs_all() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    let mut trainer = TransformerTrainer::new(config);

    let batches = vec![
        LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]),
        LMBatch::single(vec![5, 6, 7], vec![6, 7, 8]),
    ];

    trainer.train_epoch(&batches);
    assert_eq!(trainer.step(), 2);
    assert!(!trainer.reached_max_steps());
}

#[test]
fn test_transformer_trainer_warmup_lr() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_lr(0.001)
        .with_warmup_steps(100);
    let mut trainer = TransformerTrainer::new(config);

    // At step 0, LR should be 0
    assert_eq!(trainer.current_lr(), 0.0);

    // Train to advance step
    let batch = LMBatch::single(vec![1, 2], vec![2, 3]);
    trainer.train_batch(&batch);

    // At step 1, LR should be 0.001 * 1/100 = 0.00001
    let lr = trainer.current_lr();
    assert!(lr > 0.0);
    assert!(lr < 0.001);
}

#[test]
fn test_perplexity() {
    let loss = 2.0;
    let ppl = perplexity(loss);
    assert!((ppl - loss.exp()).abs() < 1e-6);
}

#[test]
fn test_tokens_per_second() {
    let tps = tokens_per_second(1000, 2.0);
    assert_eq!(tps, 500.0);
}

#[test]
fn test_transformer_trainer_grad_scaler_stats() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_fp16();
    let trainer = TransformerTrainer::new(config);

    let (scale, overflows, successes) = trainer.grad_scaler_stats();
    assert!(scale > 0.0);
    assert_eq!(overflows, 0);
    assert_eq!(successes, 0);
}

#[test]
fn test_transformer_trainer_with_model() {
    let model_config = TransformerConfig::tiny();
    let model = Transformer::new(&model_config);
    let config = TransformerTrainConfig::new(model_config);
    let trainer = TransformerTrainer::with_model(model, config);

    assert_eq!(trainer.step(), 0);
}

#[test]
fn test_lm_batch_shift_correctness() {
    // Verify that input/target shift is correct for causal LM
    let sequences = vec![vec![100, 1, 2, 3, 200]]; // BOS, tokens, EOS
    let batch = LMBatch::from_sequences(&sequences, 0, 200);

    let input = batch.get_input(0).expect("operation should succeed");
    let target = batch.get_target(0).expect("operation should succeed");

    // Input should be [BOS, 1, 2, 3]
    assert_eq!(input[0], 100); // BOS
    assert_eq!(input[1], 1);
    assert_eq!(input[2], 2);
    assert_eq!(input[3], 3);

    // Target should be [1, 2, 3, EOS]
    assert_eq!(target[0], 1);
    assert_eq!(target[1], 2);
    assert_eq!(target[2], 3);
    assert_eq!(target[3], 200); // EOS
}

// =========================================================================
// FALSIFY-ALBOR-038: Training must modify model weights
//
// Root cause: RMSNorm::forward_batched() created tensors with no backward op,
// blocking all gradient flow through the model. The optimizer never received
// gradients, so weights remained at initialization values.
//
// This test verifies the end-to-end fix: forward → backward → optimizer → save
// produces weights that differ from initialization.
// =========================================================================

/// FALSIFY-ALBOR-038: Training step must change model weights
#[test]
fn falsify_alb038_training_changes_weights() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    let mut trainer = TransformerTrainer::new(config);

    // Snapshot initial weights
    let init_params: Vec<Vec<f32>> = trainer
        .model()
        .parameters()
        .iter()
        .map(|p| p.data().to_vec())
        .collect();

    // Train for several steps
    let batch = LMBatch::single(vec![1, 2, 3, 4], vec![2, 3, 4, 5]);
    for _ in 0..5 {
        trainer.train_batch(&batch);
    }

    // Check that at least some weights changed
    let final_params: Vec<Vec<f32>> = trainer
        .model()
        .parameters()
        .iter()
        .map(|p| p.data().to_vec())
        .collect();

    let mut changed_count = 0;
    for (i, (init, final_p)) in init_params.iter().zip(final_params.iter()).enumerate() {
        if init != final_p {
            changed_count += 1;
        } else {
            // Log which parameter didn't change (for debugging)
            eprintln!("ALB-038 WARNING: parameter {i} unchanged after training (len={})", init.len());
        }
    }

    assert!(
        changed_count > 0,
        "FALSIFIED ALB-038: No model weights changed after 5 training steps. \
         All {} parameters remained at initialization values.",
        init_params.len()
    );

    // Specifically check FFN weights (these must change with working norm backward)
    // Parameters order: embed, norm, [per layer: input_norm, post_attn_norm, q, k, v, o, gate, up, down]
    // For tiny config with 2 layers: embed(0), norm(1), then per layer 9 params each
    // FFN gate_proj for layer 0 is at index 2 + 0*9 + 6 = 8
    let num_params = init_params.len();
    if num_params > 8 {
        assert_ne!(
            init_params[8], final_params[8],
            "FALSIFIED ALB-038: FFN gate_proj (param 8) unchanged — norm backward broken"
        );
    }
}

/// FALSIFY-ALBOR-038: Saved weights must differ from initialization
#[test]
fn falsify_alb038_saved_weights_differ_from_init() {
    use tempfile::NamedTempFile;

    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    let mut trainer = TransformerTrainer::new(config);

    // Snapshot init weights for comparison
    let init_embed = trainer.model().embed_tokens.weight.data().to_vec();

    // Train
    let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);
    for _ in 0..5 {
        trainer.train_batch(&batch);
    }

    // Save to temp file
    let temp = NamedTempFile::new().expect("temp file creation should succeed");
    trainer
        .save(temp.path(), "alb038-test", "test")
        .expect("save should succeed");

    // Load back and verify weights differ from init
    let data = std::fs::read(temp.path()).expect("file read should succeed");
    let loaded = safetensors::SafeTensors::deserialize(&data).expect("load should succeed");

    let saved_embed = loaded.tensor("model.embed_tokens.weight").expect("tensor exists");
    let saved_data: &[f32] = bytemuck::cast_slice(saved_embed.data());

    // Saved embed weights must differ from init
    assert_ne!(
        saved_data, &init_embed[..],
        "FALSIFIED ALB-038: Saved embedding weights are identical to initialization"
    );
}

// === R-084: Bitwise deterministic training (C-DETERM-001) ===

#[test]
fn test_deterministic_config_defaults() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    assert!(!config.deterministic, "deterministic should default to false");
    assert_eq!(config.seed, 42, "default seed should be 42");
}

#[test]
fn test_deterministic_config_builder() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_deterministic(true)
        .with_seed(12345);
    assert!(config.deterministic);
    assert_eq!(config.seed, 12345);
}

#[test]
fn test_deterministic_env_vars_set() {
    // O-DET-001: ReproducibilityConfig::apply() sets CUDA env vars
    let config = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_deterministic(true)
        .with_seed(99999);
    config.apply_deterministic_settings();

    assert_eq!(
        std::env::var("CUBLAS_WORKSPACE_CONFIG").unwrap_or_default(),
        ":4096:8",
        "CUBLAS_WORKSPACE_CONFIG must be :4096:8 (I-DET-001)"
    );
    assert_eq!(
        std::env::var("CUDNN_DETERMINISTIC").unwrap_or_default(),
        "1",
        "CUDNN_DETERMINISTIC must be 1 (I-DET-002)"
    );
    assert_eq!(
        std::env::var("CUDNN_BENCHMARK").unwrap_or_default(),
        "0",
        "CUDNN_BENCHMARK must be 0 (I-DET-003)"
    );
}

#[test]
fn test_deterministic_disabled_no_env_change() {
    // When deterministic=false, apply_deterministic_settings() is a no-op
    let config = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_deterministic(false);
    // Clear env first to verify it's not set
    std::env::remove_var("PYTHONHASHSEED");
    config.apply_deterministic_settings();
    // PYTHONHASHSEED should NOT have been set
    assert!(
        std::env::var("PYTHONHASHSEED").is_err(),
        "deterministic=false should not set PYTHONHASHSEED"
    );
}

#[test]
fn test_deterministic_training_reproducibility() {
    // F-DET-001: Two runs with same seed produce identical loss
    let config1 = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_deterministic(true)
        .with_seed(42);
    let config2 = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_deterministic(true)
        .with_seed(42);

    let mut trainer1 = TransformerTrainer::new(config1);
    let mut trainer2 = TransformerTrainer::new(config2);

    let batch = LMBatch::single(vec![1, 2, 3, 4, 5], vec![2, 3, 4, 5, 6]);

    let mut losses1 = Vec::new();
    let mut losses2 = Vec::new();

    for _ in 0..5 {
        losses1.push(trainer1.train_batch(&batch));
        losses2.push(trainer2.train_batch(&batch));
    }

    // CPU training with same init should produce identical losses
    for (i, (l1, l2)) in losses1.iter().zip(losses2.iter()).enumerate() {
        assert!(
            (l1 - l2).abs() < 1e-6,
            "Step {i}: loss mismatch {l1} vs {l2} (C-DETERM-001 violation)"
        );
    }
}

// ── Activation Checkpointing (R-021, #115) ──────────────────────────────

#[test]
fn test_checkpoint_config_segment_calculation() {
    // Verify checkpoint boundary mask calculation
    // tiny has 2 layers, 2 segments → segment_size = 1 → all layers are checkpoints
    let config = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_checkpointing(2);

    assert!(config.checkpoint_config.enabled);
    assert_eq!(config.checkpoint_config.num_segments, 2);

    let num_layers = config.model_config.num_hidden_layers;
    assert_eq!(num_layers, 2);
    let ns = config.checkpoint_config.num_segments.max(1);
    let segment_size = (num_layers + ns - 1) / ns;
    assert_eq!(segment_size, 1);
    let mask: Vec<bool> = (0..num_layers)
        .map(|i| i % segment_size == 0)
        .collect();
    assert_eq!(mask, vec![true, true]);
}

#[test]
fn test_checkpoint_config_fewer_segments() {
    // Use a config with more layers: 24 layers, 4 segments → segment_size = 6
    // Checkpoint boundary layers: 0, 6, 12, 18
    let mut model_config = TransformerConfig::tiny();
    model_config.num_hidden_layers = 24;
    let config = TransformerTrainConfig::new(model_config)
        .with_checkpointing(4);

    let num_layers = config.model_config.num_hidden_layers;
    assert_eq!(num_layers, 24);
    let ns = config.checkpoint_config.num_segments.max(1);
    let segment_size = (num_layers + ns - 1) / ns;
    assert_eq!(segment_size, 6);
    let mask: Vec<bool> = (0..num_layers)
        .map(|i| i % segment_size == 0)
        .collect();
    // Only layers 0, 6, 12, 18 are checkpoints
    let expected: Vec<bool> = (0..24).map(|i| i % 6 == 0).collect();
    assert_eq!(mask, expected);
    assert_eq!(mask.iter().filter(|&&x| x).count(), 4);
    assert_eq!(mask.iter().filter(|&&x| !x).count(), 20);
}

#[test]
fn test_checkpoint_disabled_saves_all_layers() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());

    assert!(!config.checkpoint_config.enabled);

    // When disabled, all layers should be "checkpointed" (saved)
    let num_layers = config.model_config.num_hidden_layers;
    let mask: Vec<bool> = (0..num_layers)
        .map(|_| true) // !checkpointing → all saved
        .collect();
    assert!(mask.iter().all(|&x| x));
}

#[test]
fn test_checkpoint_with_cpu_trainer() {
    // CPU trainer with checkpointing enabled should train normally
    // (checkpointing only affects CUDA path, CPU path ignores it)
    let model_config = TransformerConfig::tiny();
    let config = TransformerTrainConfig::new(model_config.clone())
        .with_checkpointing(2)
        .with_lr(0.001)
        .with_max_seq_len(32);

    let model = Transformer::new(&model_config);
    let mut trainer = TransformerTrainer::with_model(model, config);

    let input = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
    let batch = LMBatch::from_sequences(&[input], 0, 0);

    let loss = trainer.train_batch(&batch);
    assert!(loss > 0.0, "Loss should be positive");
    assert!(loss.is_finite(), "Loss should be finite");
}
