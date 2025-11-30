//! Declarative YAML configuration
//!
//! This module provides Ludwig-style declarative training configuration via YAML.
//!
//! # Example
//!
//! ```yaml
//! model:
//!   path: base-model.gguf
//!   layers: [q_proj, v_proj]
//!
//! data:
//!   train: train.parquet
//!   batch_size: 8
//!
//! optimizer:
//!   name: adam
//!   lr: 1e-4
//!
//! lora:
//!   rank: 64
//!   alpha: 16
//! ```

mod builder;
mod cli;
mod infer;
mod schema;
mod train;
mod validate;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod property_tests;

pub use builder::{build_model, build_optimizer};
pub use cli::{
    apply_overrides, parse_args, ArchiveProviderArg, ArtifactTypeArg, BundleArgs, CitationFormat,
    CiteArgs, Cli, Command, DepositArgs, ExportArgs, ExportFormat, InfoArgs, InitArgs, InitTemplate,
    LicenseArg, MergeArgs, MergeMethod, OutputFormat, PreregisterArgs, QuantMethod, QuantizeArgs,
    ResearchArgs, ResearchCommand, ResearchInitArgs, TrainArgs, ValidateArgs, VerifyArgs,
};
pub use infer::{
    collect_stats_from_samples, infer_schema, infer_schema_from_path, infer_type, ColumnStats,
    FeatureType, InferenceConfig, InferredSchema,
};
pub use schema::{
    DataConfig, LoRASpec, MergeSpec, ModelRef, OptimSpec, QuantSpec, TrainSpec, TrainingParams,
};
pub use train::{load_config, train_from_yaml};
pub use validate::{validate_config, ValidationError};
