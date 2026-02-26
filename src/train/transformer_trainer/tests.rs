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
