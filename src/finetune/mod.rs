//! Fine-tuning pipeline for code generation models
//!
//! This module implements SPEC-FT-001: Rust Test Generation Fine-Tuning Pipeline.
//!
//! # Features
//!
//! - CUDA auto-detection with fallback to CPU
//! - QLoRA fine-tuning with 4-bit quantization
//! - Popperian 100-point falsification QA
//! - Scientific reproducibility protocol
//!
//! # References
//!
//! - Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
//! - Dettmers et al. (2023) "QLoRA: Efficient Finetuning of Quantized LLMs"
//! - Popper (1959) "The Logic of Scientific Discovery"

pub mod classification;
pub mod classify_pipeline;
pub mod classify_trainer;
pub mod classify_tuner;
pub mod tune_searchers;
mod corpus;
mod device;
mod eval;
mod popperian;
mod reproducibility;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_classification_contract_falsify;

pub use classification::{
    bce_with_logits_loss, compute_class_weights, corpus_stats, cross_entropy_loss,
    load_multi_label_corpus, load_safety_corpus, ClassWeightStrategy, ClassificationHead,
    MultiLabelSafetySample, SafetyCorpusStats, SafetySample,
};
pub use classify_pipeline::{BatchResult, ClassifyConfig, ClassifyPipeline};
pub use classify_trainer::{
    evaluate_checkpoint, ClassifyEvalReport, ClassifyTrainer, EpochMetrics, TrainResult,
    TrainingConfig, SSC_LABELS,
};
pub use classify_tuner::{
    default_classify_search_space, extract_trial_params, ClassifyTuner, SchedulerKind,
    TrialSummary, TuneConfig, TuneResult, TuneScheduler, TuneSearcher, TuneStrategy,
};
pub use corpus::{CorpusStats, SampleMetadata, TestGenCorpus, TestGenSample};
pub use device::{ComputeDevice, DeviceInfo};
pub use eval::{
    contains_tautology, count_test_functions, has_edge_case_tests, has_meaningful_assertions,
    EvalMetrics, EvalResult, TestEvaluator,
};
pub use popperian::{PopperianQA, QAGrade};
pub use reproducibility::{ExperimentLock, ReproducibilityConfig};
