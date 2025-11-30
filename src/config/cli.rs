//! CLI argument parsing and validation
//!
//! This module provides the command-line interface for entrenar training.
//!
//! # Usage
//!
//! ```bash
//! entrenar train config.yaml
//! entrenar train config.yaml --output-dir ./checkpoints
//! entrenar train config.yaml --resume checkpoint.json
//! entrenar validate config.yaml
//! entrenar info config.yaml
//! ```

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Entrenar: Training & Optimization Library
#[derive(Parser, Debug, Clone, PartialEq)]
#[command(name = "entrenar")]
#[command(author = "PAIML")]
#[command(version)]
#[command(
    about = "Training & Optimization Library with autograd, LoRA, quantization, and model merging"
)]
pub struct Cli {
    /// Subcommand to execute
    #[command(subcommand)]
    pub command: Command,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Suppress all output except errors
    #[arg(short, long, global = true)]
    pub quiet: bool,
}

/// Available commands
#[derive(Subcommand, Debug, Clone, PartialEq)]
pub enum Command {
    /// Train a model from YAML configuration
    Train(TrainArgs),

    /// Validate a configuration file without training
    Validate(ValidateArgs),

    /// Display information about a configuration
    Info(InfoArgs),

    /// Initialize a new YAML Mode training manifest
    Init(InitArgs),

    /// Quantize a model
    Quantize(QuantizeArgs),

    /// Merge multiple models
    Merge(MergeArgs),

    /// Academic research artifacts and workflows
    Research(ResearchArgs),
}

/// Arguments for the train command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct TrainArgs {
    /// Path to YAML configuration file
    #[arg(value_name = "CONFIG")]
    pub config: PathBuf,

    /// Override output directory
    #[arg(short, long)]
    pub output_dir: Option<PathBuf>,

    /// Resume training from checkpoint
    #[arg(short, long)]
    pub resume: Option<PathBuf>,

    /// Override number of epochs
    #[arg(short, long)]
    pub epochs: Option<usize>,

    /// Override batch size
    #[arg(short, long)]
    pub batch_size: Option<usize>,

    /// Override learning rate
    #[arg(short, long)]
    pub lr: Option<f32>,

    /// Dry run (validate config but don't train)
    #[arg(long)]
    pub dry_run: bool,

    /// Save checkpoint every N steps
    #[arg(long)]
    pub save_every: Option<usize>,

    /// Log metrics every N steps
    #[arg(long)]
    pub log_every: Option<usize>,

    /// Random seed for reproducibility
    #[arg(long)]
    pub seed: Option<u64>,
}

/// Arguments for the validate command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct ValidateArgs {
    /// Path to YAML configuration file
    #[arg(value_name = "CONFIG")]
    pub config: PathBuf,

    /// Show detailed validation report
    #[arg(short, long)]
    pub detailed: bool,
}

/// Arguments for the info command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct InfoArgs {
    /// Path to YAML configuration file
    #[arg(value_name = "CONFIG")]
    pub config: PathBuf,

    /// Output format (text, json, yaml)
    #[arg(short, long, default_value = "text")]
    pub format: OutputFormat,
}

/// Arguments for the init command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct InitArgs {
    /// Template to use for initialization
    #[arg(short, long, default_value = "minimal")]
    pub template: InitTemplate,

    /// Output path (stdout if not specified)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Experiment name
    #[arg(long, default_value = "my-experiment")]
    pub name: String,

    /// Model source path or URI
    #[arg(long)]
    pub model: Option<String>,

    /// Data source path or URI
    #[arg(long)]
    pub data: Option<String>,
}

/// Init template type
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum InitTemplate {
    /// Minimal manifest with required fields only
    #[default]
    Minimal,
    /// LoRA fine-tuning template
    Lora,
    /// QLoRA fine-tuning template
    Qlora,
    /// Full template with all sections
    Full,
}

impl std::str::FromStr for InitTemplate {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "minimal" | "min" => Ok(InitTemplate::Minimal),
            "lora" => Ok(InitTemplate::Lora),
            "qlora" => Ok(InitTemplate::Qlora),
            "full" | "complete" => Ok(InitTemplate::Full),
            _ => Err(format!(
                "Unknown template: {s}. Valid templates: minimal, lora, qlora, full"
            )),
        }
    }
}

impl std::fmt::Display for InitTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InitTemplate::Minimal => write!(f, "minimal"),
            InitTemplate::Lora => write!(f, "lora"),
            InitTemplate::Qlora => write!(f, "qlora"),
            InitTemplate::Full => write!(f, "full"),
        }
    }
}

/// Arguments for the quantize command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct QuantizeArgs {
    /// Path to model file
    #[arg(value_name = "MODEL")]
    pub model: PathBuf,

    /// Output path for quantized model
    #[arg(short, long)]
    pub output: PathBuf,

    /// Quantization bits (4 or 8)
    #[arg(short, long, default_value = "4")]
    pub bits: u8,

    /// Quantization method (symmetric or asymmetric)
    #[arg(short, long, default_value = "symmetric")]
    pub method: QuantMethod,

    /// Use per-channel quantization
    #[arg(long)]
    pub per_channel: bool,

    /// Path to calibration data
    #[arg(long)]
    pub calibration_data: Option<PathBuf>,
}

/// Arguments for the merge command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct MergeArgs {
    /// Paths to models to merge
    #[arg(value_name = "MODELS", num_args = 2..)]
    pub models: Vec<PathBuf>,

    /// Output path for merged model
    #[arg(short, long)]
    pub output: PathBuf,

    /// Merge method (ties, dare, slerp, average)
    #[arg(short, long, default_value = "ties")]
    pub method: MergeMethod,

    /// Interpolation weight (for slerp)
    #[arg(short, long)]
    pub weight: Option<f32>,

    /// Density threshold (for ties/dare)
    #[arg(short, long)]
    pub density: Option<f32>,

    /// Model weights (comma-separated, for weighted average)
    #[arg(long)]
    pub weights: Option<String>,
}

/// Arguments for the research command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct ResearchArgs {
    /// Research subcommand to execute
    #[command(subcommand)]
    pub command: ResearchCommand,
}

/// Research subcommands
#[derive(Subcommand, Debug, Clone, PartialEq)]
pub enum ResearchCommand {
    /// Initialize a new research artifact
    Init(ResearchInitArgs),

    /// Create a pre-registration with cryptographic commitment
    Preregister(PreregisterArgs),

    /// Generate citations in various formats
    Cite(CiteArgs),

    /// Export artifacts to various formats
    Export(ExportArgs),

    /// Deposit to academic archives
    Deposit(DepositArgs),

    /// Bundle artifacts into RO-Crate package
    Bundle(BundleArgs),

    /// Verify pre-registration commitments or signatures
    Verify(VerifyArgs),
}

/// Arguments for research init command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct ResearchInitArgs {
    /// Artifact ID (unique identifier)
    #[arg(long)]
    pub id: String,

    /// Artifact title
    #[arg(long)]
    pub title: String,

    /// Artifact type
    #[arg(long, default_value = "dataset")]
    pub artifact_type: ArtifactTypeArg,

    /// License (e.g., CC-BY-4.0, MIT, Apache-2.0)
    #[arg(long, default_value = "cc-by-4.0")]
    pub license: LicenseArg,

    /// Output path for artifact YAML
    #[arg(short, long, default_value = "artifact.yaml")]
    pub output: PathBuf,

    /// Author name
    #[arg(long)]
    pub author: Option<String>,

    /// Author ORCID (format: 0000-0002-1825-0097)
    #[arg(long)]
    pub orcid: Option<String>,

    /// Author affiliation
    #[arg(long)]
    pub affiliation: Option<String>,

    /// Description of the artifact
    #[arg(long)]
    pub description: Option<String>,

    /// Keywords (comma-separated)
    #[arg(long)]
    pub keywords: Option<String>,

    /// DOI (if already assigned)
    #[arg(long)]
    pub doi: Option<String>,
}

/// Arguments for preregister command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct PreregisterArgs {
    /// Research question or title
    #[arg(long)]
    pub title: String,

    /// Hypothesis being tested
    #[arg(long)]
    pub hypothesis: String,

    /// Methodology description
    #[arg(long)]
    pub methodology: String,

    /// Statistical analysis plan
    #[arg(long)]
    pub analysis_plan: String,

    /// Additional notes
    #[arg(long)]
    pub notes: Option<String>,

    /// Output path for pre-registration
    #[arg(short, long, default_value = "preregistration.yaml")]
    pub output: PathBuf,

    /// Path to Ed25519 private key for signing
    #[arg(long)]
    pub sign_key: Option<PathBuf>,

    /// Add git commit hash as timestamp proof
    #[arg(long)]
    pub git_timestamp: bool,
}

/// Arguments for cite command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct CiteArgs {
    /// Path to artifact YAML file
    #[arg(value_name = "ARTIFACT")]
    pub artifact: PathBuf,

    /// Publication year
    #[arg(long)]
    pub year: u16,

    /// Output format
    #[arg(short, long, default_value = "bibtex")]
    pub format: CitationFormat,

    /// Output file (stdout if not specified)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Journal name
    #[arg(long)]
    pub journal: Option<String>,

    /// Volume number
    #[arg(long)]
    pub volume: Option<String>,

    /// Page range
    #[arg(long)]
    pub pages: Option<String>,

    /// URL
    #[arg(long)]
    pub url: Option<String>,
}

/// Arguments for export command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct ExportArgs {
    /// Path to artifact or document
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Export format
    #[arg(short, long)]
    pub format: ExportFormat,

    /// Output file
    #[arg(short, long)]
    pub output: PathBuf,

    /// Anonymize for double-blind review
    #[arg(long)]
    pub anonymize: bool,

    /// Salt for anonymization (required with --anonymize)
    #[arg(long)]
    pub anon_salt: Option<String>,

    /// Jupyter kernel (for notebook export)
    #[arg(long, default_value = "python3")]
    pub kernel: String,
}

/// Arguments for deposit command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct DepositArgs {
    /// Path to artifact YAML
    #[arg(value_name = "ARTIFACT")]
    pub artifact: PathBuf,

    /// Archive provider
    #[arg(short, long)]
    pub provider: ArchiveProviderArg,

    /// API token (or use env var ZENODO_TOKEN, etc.)
    #[arg(long)]
    pub token: Option<String>,

    /// Use sandbox/test environment
    #[arg(long)]
    pub sandbox: bool,

    /// Community to submit to
    #[arg(long)]
    pub community: Option<String>,

    /// Files to include (can be repeated)
    #[arg(short, long)]
    pub file: Vec<PathBuf>,

    /// Dry run (validate without uploading)
    #[arg(long)]
    pub dry_run: bool,
}

/// Arguments for bundle command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct BundleArgs {
    /// Path to artifact YAML
    #[arg(value_name = "ARTIFACT")]
    pub artifact: PathBuf,

    /// Output directory for RO-Crate
    #[arg(short, long)]
    pub output: PathBuf,

    /// Files to include (can be repeated)
    #[arg(short, long)]
    pub file: Vec<PathBuf>,

    /// Create ZIP archive instead of directory
    #[arg(long)]
    pub zip: bool,

    /// Include citation graph
    #[arg(long)]
    pub include_citations: bool,
}

/// Arguments for verify command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct VerifyArgs {
    /// Path to pre-registration or signed artifact
    #[arg(value_name = "FILE")]
    pub file: PathBuf,

    /// Path to Ed25519 public key for signature verification
    #[arg(long)]
    pub public_key: Option<PathBuf>,

    /// Original content to verify against commitment
    #[arg(long)]
    pub original: Option<PathBuf>,

    /// Verify git timestamp proof
    #[arg(long)]
    pub verify_git: bool,
}

/// Artifact type for CLI
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ArtifactTypeArg {
    #[default]
    Dataset,
    Paper,
    Model,
    Code,
    Notebook,
    Workflow,
}

impl std::str::FromStr for ArtifactTypeArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "dataset" => Ok(ArtifactTypeArg::Dataset),
            "paper" => Ok(ArtifactTypeArg::Paper),
            "model" => Ok(ArtifactTypeArg::Model),
            "code" => Ok(ArtifactTypeArg::Code),
            "notebook" => Ok(ArtifactTypeArg::Notebook),
            "workflow" => Ok(ArtifactTypeArg::Workflow),
            _ => Err(format!(
                "Unknown artifact type: {s}. Valid types: dataset, paper, model, code, notebook, workflow"
            )),
        }
    }
}

impl std::fmt::Display for ArtifactTypeArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArtifactTypeArg::Dataset => write!(f, "dataset"),
            ArtifactTypeArg::Paper => write!(f, "paper"),
            ArtifactTypeArg::Model => write!(f, "model"),
            ArtifactTypeArg::Code => write!(f, "code"),
            ArtifactTypeArg::Notebook => write!(f, "notebook"),
            ArtifactTypeArg::Workflow => write!(f, "workflow"),
        }
    }
}

/// License for CLI
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum LicenseArg {
    #[default]
    CcBy4,
    CcBySa4,
    Cc0,
    Mit,
    Apache2,
    Gpl3,
    Bsd3,
}

impl std::str::FromStr for LicenseArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().replace(['-', '.'], "").as_str() {
            "ccby4" | "ccby40" => Ok(LicenseArg::CcBy4),
            "ccbysa4" | "ccbysa40" => Ok(LicenseArg::CcBySa4),
            "cc0" => Ok(LicenseArg::Cc0),
            "mit" => Ok(LicenseArg::Mit),
            "apache2" | "apache20" => Ok(LicenseArg::Apache2),
            "gpl3" | "gplv3" => Ok(LicenseArg::Gpl3),
            "bsd3" | "bsd3clause" => Ok(LicenseArg::Bsd3),
            _ => Err(format!(
                "Unknown license: {s}. Valid licenses: CC-BY-4.0, CC-BY-SA-4.0, CC0, MIT, Apache-2.0, GPL-3.0, BSD-3-Clause"
            )),
        }
    }
}

impl std::fmt::Display for LicenseArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LicenseArg::CcBy4 => write!(f, "CC-BY-4.0"),
            LicenseArg::CcBySa4 => write!(f, "CC-BY-SA-4.0"),
            LicenseArg::Cc0 => write!(f, "CC0"),
            LicenseArg::Mit => write!(f, "MIT"),
            LicenseArg::Apache2 => write!(f, "Apache-2.0"),
            LicenseArg::Gpl3 => write!(f, "GPL-3.0"),
            LicenseArg::Bsd3 => write!(f, "BSD-3-Clause"),
        }
    }
}

/// Citation format for CLI
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CitationFormat {
    #[default]
    Bibtex,
    Cff,
    Json,
}

impl std::str::FromStr for CitationFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bibtex" | "bib" => Ok(CitationFormat::Bibtex),
            "cff" | "citation.cff" => Ok(CitationFormat::Cff),
            "json" => Ok(CitationFormat::Json),
            _ => Err(format!(
                "Unknown citation format: {s}. Valid formats: bibtex, cff, json"
            )),
        }
    }
}

impl std::fmt::Display for CitationFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CitationFormat::Bibtex => write!(f, "bibtex"),
            CitationFormat::Cff => write!(f, "cff"),
            CitationFormat::Json => write!(f, "json"),
        }
    }
}

/// Export format for CLI
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExportFormat {
    Notebook,
    Html,
    AnonymizedJson,
    RoCrate,
}

impl std::str::FromStr for ExportFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "notebook" | "ipynb" | "jupyter" => Ok(ExportFormat::Notebook),
            "html" => Ok(ExportFormat::Html),
            "anonymized" | "anon" | "anonymized-json" => Ok(ExportFormat::AnonymizedJson),
            "ro-crate" | "rocrate" => Ok(ExportFormat::RoCrate),
            _ => Err(format!(
                "Unknown export format: {s}. Valid formats: notebook, html, anonymized, ro-crate"
            )),
        }
    }
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportFormat::Notebook => write!(f, "notebook"),
            ExportFormat::Html => write!(f, "html"),
            ExportFormat::AnonymizedJson => write!(f, "anonymized-json"),
            ExportFormat::RoCrate => write!(f, "ro-crate"),
        }
    }
}

/// Archive provider for CLI
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ArchiveProviderArg {
    #[default]
    Zenodo,
    Figshare,
    Dryad,
    Dataverse,
}

impl std::str::FromStr for ArchiveProviderArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "zenodo" => Ok(ArchiveProviderArg::Zenodo),
            "figshare" => Ok(ArchiveProviderArg::Figshare),
            "dryad" => Ok(ArchiveProviderArg::Dryad),
            "dataverse" => Ok(ArchiveProviderArg::Dataverse),
            _ => Err(format!(
                "Unknown archive provider: {s}. Valid providers: zenodo, figshare, dryad, dataverse"
            )),
        }
    }
}

impl std::fmt::Display for ArchiveProviderArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArchiveProviderArg::Zenodo => write!(f, "zenodo"),
            ArchiveProviderArg::Figshare => write!(f, "figshare"),
            ArchiveProviderArg::Dryad => write!(f, "dryad"),
            ArchiveProviderArg::Dataverse => write!(f, "dataverse"),
        }
    }
}

/// Output format for info command
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
    Yaml,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "text" => Ok(OutputFormat::Text),
            "json" => Ok(OutputFormat::Json),
            "yaml" => Ok(OutputFormat::Yaml),
            _ => Err(format!(
                "Unknown output format: {s}. Valid formats: text, json, yaml"
            )),
        }
    }
}

/// Quantization method
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum QuantMethod {
    #[default]
    Symmetric,
    Asymmetric,
}

impl std::str::FromStr for QuantMethod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "symmetric" | "sym" => Ok(QuantMethod::Symmetric),
            "asymmetric" | "asym" => Ok(QuantMethod::Asymmetric),
            _ => Err(format!(
                "Unknown quantization method: {s}. Valid methods: symmetric, asymmetric"
            )),
        }
    }
}

/// Merge method
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum MergeMethod {
    #[default]
    Ties,
    Dare,
    Slerp,
    Average,
}

impl std::str::FromStr for MergeMethod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ties" => Ok(MergeMethod::Ties),
            "dare" => Ok(MergeMethod::Dare),
            "slerp" => Ok(MergeMethod::Slerp),
            "average" | "avg" => Ok(MergeMethod::Average),
            _ => Err(format!(
                "Unknown merge method: {s}. Valid methods: ties, dare, slerp, average"
            )),
        }
    }
}

/// Parse CLI arguments from a string slice (for testing)
pub fn parse_args<I, T>(args: I) -> Result<Cli, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    Cli::try_parse_from(args)
}

/// Apply command-line overrides to a TrainSpec
pub fn apply_overrides(spec: &mut super::TrainSpec, args: &TrainArgs) {
    if let Some(output_dir) = &args.output_dir {
        spec.training.output_dir = output_dir.clone();
    }
    if let Some(epochs) = args.epochs {
        spec.training.epochs = epochs;
    }
    if let Some(batch_size) = args.batch_size {
        spec.data.batch_size = batch_size;
    }
    if let Some(lr) = args.lr {
        spec.optimizer.lr = lr;
    }
    if let Some(save_every) = args.save_every {
        spec.training.save_interval = save_every;
    }
    // Note: log_every and seed are CLI-only options not persisted in config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_train_command() {
        let cli = parse_args(["entrenar", "train", "config.yaml"]).unwrap();
        match cli.command {
            Command::Train(args) => {
                assert_eq!(args.config, PathBuf::from("config.yaml"));
                assert!(!args.dry_run);
            }
            _ => panic!("Expected Train command"),
        }
    }

    #[test]
    fn test_parse_train_with_overrides() {
        let cli = parse_args([
            "entrenar",
            "train",
            "config.yaml",
            "--epochs",
            "10",
            "--batch-size",
            "32",
            "--lr",
            "0.001",
            "--output-dir",
            "./output",
        ])
        .unwrap();

        match cli.command {
            Command::Train(args) => {
                assert_eq!(args.epochs, Some(10));
                assert_eq!(args.batch_size, Some(32));
                assert!((args.lr.unwrap() - 0.001).abs() < 1e-6);
                assert_eq!(args.output_dir, Some(PathBuf::from("./output")));
            }
            _ => panic!("Expected Train command"),
        }
    }

    #[test]
    fn test_parse_train_with_resume() {
        let cli = parse_args([
            "entrenar",
            "train",
            "config.yaml",
            "--resume",
            "checkpoint.json",
        ])
        .unwrap();

        match cli.command {
            Command::Train(args) => {
                assert_eq!(args.resume, Some(PathBuf::from("checkpoint.json")));
            }
            _ => panic!("Expected Train command"),
        }
    }

    #[test]
    fn test_parse_train_dry_run() {
        let cli = parse_args(["entrenar", "train", "config.yaml", "--dry-run"]).unwrap();
        match cli.command {
            Command::Train(args) => {
                assert!(args.dry_run);
            }
            _ => panic!("Expected Train command"),
        }
    }

    #[test]
    fn test_parse_validate_command() {
        let cli = parse_args(["entrenar", "validate", "config.yaml"]).unwrap();
        match cli.command {
            Command::Validate(args) => {
                assert_eq!(args.config, PathBuf::from("config.yaml"));
                assert!(!args.detailed);
            }
            _ => panic!("Expected Validate command"),
        }
    }

    #[test]
    fn test_parse_validate_detailed() {
        let cli = parse_args(["entrenar", "validate", "config.yaml", "--detailed"]).unwrap();
        match cli.command {
            Command::Validate(args) => {
                assert!(args.detailed);
            }
            _ => panic!("Expected Validate command"),
        }
    }

    #[test]
    fn test_parse_info_command() {
        let cli = parse_args(["entrenar", "info", "config.yaml"]).unwrap();
        match cli.command {
            Command::Info(args) => {
                assert_eq!(args.config, PathBuf::from("config.yaml"));
                assert_eq!(args.format, OutputFormat::Text);
            }
            _ => panic!("Expected Info command"),
        }
    }

    #[test]
    fn test_parse_info_json_format() {
        let cli = parse_args(["entrenar", "info", "config.yaml", "--format", "json"]).unwrap();
        match cli.command {
            Command::Info(args) => {
                assert_eq!(args.format, OutputFormat::Json);
            }
            _ => panic!("Expected Info command"),
        }
    }

    #[test]
    fn test_parse_quantize_command() {
        let cli = parse_args([
            "entrenar",
            "quantize",
            "model.gguf",
            "--output",
            "model_q4.gguf",
        ])
        .unwrap();

        match cli.command {
            Command::Quantize(args) => {
                assert_eq!(args.model, PathBuf::from("model.gguf"));
                assert_eq!(args.output, PathBuf::from("model_q4.gguf"));
                assert_eq!(args.bits, 4);
                assert_eq!(args.method, QuantMethod::Symmetric);
            }
            _ => panic!("Expected Quantize command"),
        }
    }

    #[test]
    fn test_parse_quantize_with_options() {
        let cli = parse_args([
            "entrenar",
            "quantize",
            "model.gguf",
            "--output",
            "model_q8.gguf",
            "--bits",
            "8",
            "--method",
            "asymmetric",
            "--per-channel",
        ])
        .unwrap();

        match cli.command {
            Command::Quantize(args) => {
                assert_eq!(args.bits, 8);
                assert_eq!(args.method, QuantMethod::Asymmetric);
                assert!(args.per_channel);
            }
            _ => panic!("Expected Quantize command"),
        }
    }

    #[test]
    fn test_parse_merge_command() {
        let cli = parse_args([
            "entrenar",
            "merge",
            "model1.gguf",
            "model2.gguf",
            "--output",
            "merged.gguf",
        ])
        .unwrap();

        match cli.command {
            Command::Merge(args) => {
                assert_eq!(args.models.len(), 2);
                assert_eq!(args.output, PathBuf::from("merged.gguf"));
                assert_eq!(args.method, MergeMethod::Ties);
            }
            _ => panic!("Expected Merge command"),
        }
    }

    #[test]
    fn test_parse_merge_slerp() {
        let cli = parse_args([
            "entrenar",
            "merge",
            "model1.gguf",
            "model2.gguf",
            "--output",
            "merged.gguf",
            "--method",
            "slerp",
            "--weight",
            "0.7",
        ])
        .unwrap();

        match cli.command {
            Command::Merge(args) => {
                assert_eq!(args.method, MergeMethod::Slerp);
                assert!((args.weight.unwrap() - 0.7).abs() < 1e-6);
            }
            _ => panic!("Expected Merge command"),
        }
    }

    #[test]
    fn test_parse_merge_multiple_models() {
        let cli = parse_args([
            "entrenar",
            "merge",
            "model1.gguf",
            "model2.gguf",
            "model3.gguf",
            "--output",
            "merged.gguf",
            "--method",
            "average",
        ])
        .unwrap();

        match cli.command {
            Command::Merge(args) => {
                assert_eq!(args.models.len(), 3);
                assert_eq!(args.method, MergeMethod::Average);
            }
            _ => panic!("Expected Merge command"),
        }
    }

    #[test]
    fn test_global_verbose_flag() {
        let cli = parse_args(["entrenar", "-v", "train", "config.yaml"]).unwrap();
        assert!(cli.verbose);
        assert!(!cli.quiet);
    }

    #[test]
    fn test_global_quiet_flag() {
        let cli = parse_args(["entrenar", "-q", "train", "config.yaml"]).unwrap();
        assert!(!cli.verbose);
        assert!(cli.quiet);
    }

    #[test]
    fn test_output_format_from_str() {
        assert_eq!("text".parse::<OutputFormat>().unwrap(), OutputFormat::Text);
        assert_eq!("json".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert_eq!("yaml".parse::<OutputFormat>().unwrap(), OutputFormat::Yaml);
        assert_eq!("JSON".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert!("invalid".parse::<OutputFormat>().is_err());
    }

    #[test]
    fn test_quant_method_from_str() {
        assert_eq!(
            "symmetric".parse::<QuantMethod>().unwrap(),
            QuantMethod::Symmetric
        );
        assert_eq!(
            "sym".parse::<QuantMethod>().unwrap(),
            QuantMethod::Symmetric
        );
        assert_eq!(
            "asymmetric".parse::<QuantMethod>().unwrap(),
            QuantMethod::Asymmetric
        );
        assert_eq!(
            "asym".parse::<QuantMethod>().unwrap(),
            QuantMethod::Asymmetric
        );
        assert!("invalid".parse::<QuantMethod>().is_err());
    }

    #[test]
    fn test_merge_method_from_str() {
        assert_eq!("ties".parse::<MergeMethod>().unwrap(), MergeMethod::Ties);
        assert_eq!("dare".parse::<MergeMethod>().unwrap(), MergeMethod::Dare);
        assert_eq!("slerp".parse::<MergeMethod>().unwrap(), MergeMethod::Slerp);
        assert_eq!(
            "average".parse::<MergeMethod>().unwrap(),
            MergeMethod::Average
        );
        assert_eq!("avg".parse::<MergeMethod>().unwrap(), MergeMethod::Average);
        assert!("invalid".parse::<MergeMethod>().is_err());
    }

    #[test]
    fn test_missing_config_file() {
        let result = parse_args(["entrenar", "train"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_command() {
        let result = parse_args(["entrenar", "unknown"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_overrides_output_dir() {
        let mut spec = create_test_spec();
        let args = TrainArgs {
            config: PathBuf::from("config.yaml"),
            output_dir: Some(PathBuf::from("./custom_output")),
            resume: None,
            epochs: None,
            batch_size: None,
            lr: None,
            save_every: None,
            log_every: None,
            dry_run: false,
            seed: None,
        };
        apply_overrides(&mut spec, &args);
        assert_eq!(spec.training.output_dir, PathBuf::from("./custom_output"));
    }

    #[test]
    fn test_apply_overrides_epochs() {
        let mut spec = create_test_spec();
        let args = TrainArgs {
            config: PathBuf::from("config.yaml"),
            output_dir: None,
            resume: None,
            epochs: Some(50),
            batch_size: None,
            lr: None,
            save_every: None,
            log_every: None,
            dry_run: false,
            seed: None,
        };
        apply_overrides(&mut spec, &args);
        assert_eq!(spec.training.epochs, 50);
    }

    #[test]
    fn test_apply_overrides_batch_size() {
        let mut spec = create_test_spec();
        let args = TrainArgs {
            config: PathBuf::from("config.yaml"),
            output_dir: None,
            resume: None,
            epochs: None,
            batch_size: Some(64),
            lr: None,
            save_every: None,
            log_every: None,
            dry_run: false,
            seed: None,
        };
        apply_overrides(&mut spec, &args);
        assert_eq!(spec.data.batch_size, 64);
    }

    #[test]
    fn test_apply_overrides_lr() {
        let mut spec = create_test_spec();
        let args = TrainArgs {
            config: PathBuf::from("config.yaml"),
            output_dir: None,
            resume: None,
            epochs: None,
            batch_size: None,
            lr: Some(0.0001),
            save_every: None,
            log_every: None,
            dry_run: false,
            seed: None,
        };
        apply_overrides(&mut spec, &args);
        assert!((spec.optimizer.lr - 0.0001).abs() < 1e-8);
    }

    #[test]
    fn test_apply_overrides_save_every() {
        let mut spec = create_test_spec();
        let args = TrainArgs {
            config: PathBuf::from("config.yaml"),
            output_dir: None,
            resume: None,
            epochs: None,
            batch_size: None,
            lr: None,
            save_every: Some(5),
            log_every: None,
            dry_run: false,
            seed: None,
        };
        apply_overrides(&mut spec, &args);
        assert_eq!(spec.training.save_interval, 5);
    }

    #[test]
    fn test_apply_overrides_all() {
        let mut spec = create_test_spec();
        let args = TrainArgs {
            config: PathBuf::from("config.yaml"),
            output_dir: Some(PathBuf::from("./all_overrides")),
            resume: Some(PathBuf::from("checkpoint.json")),
            epochs: Some(100),
            batch_size: Some(128),
            lr: Some(0.01),
            save_every: Some(10),
            log_every: Some(50),
            dry_run: true,
            seed: Some(42),
        };
        apply_overrides(&mut spec, &args);
        assert_eq!(spec.training.output_dir, PathBuf::from("./all_overrides"));
        assert_eq!(spec.training.epochs, 100);
        assert_eq!(spec.data.batch_size, 128);
        assert!((spec.optimizer.lr - 0.01).abs() < 1e-8);
        assert_eq!(spec.training.save_interval, 10);
    }

    fn create_test_spec() -> super::super::TrainSpec {
        super::super::TrainSpec {
            model: super::super::ModelRef {
                path: PathBuf::from("model.gguf"),
                layers: vec![],
            },
            data: super::super::DataConfig {
                train: PathBuf::from("train.parquet"),
                val: None,
                batch_size: 8,
                auto_infer_types: true,
                seq_len: None,
            },
            optimizer: super::super::OptimSpec {
                name: "adam".to_string(),
                lr: 0.001,
                params: Default::default(),
            },
            lora: None,
            quantize: None,
            merge: None,
            training: Default::default(),
        }
    }

    #[test]
    fn test_output_format_from_str_yaml() {
        assert_eq!("yaml".parse::<OutputFormat>().unwrap(), OutputFormat::Yaml);
    }

    #[test]
    fn test_output_format_from_str_invalid() {
        assert!("invalid_format".parse::<OutputFormat>().is_err());
    }

    #[test]
    fn test_verbose_and_quiet_flags() {
        let cli = parse_args(["entrenar", "--verbose", "train", "config.yaml"]).unwrap();
        assert!(cli.verbose);
        assert!(!cli.quiet);

        let cli = parse_args(["entrenar", "--quiet", "train", "config.yaml"]).unwrap();
        assert!(!cli.verbose);
        assert!(cli.quiet);
    }

    #[test]
    fn test_info_yaml_format() {
        let cli = parse_args(["entrenar", "info", "config.yaml", "--format", "yaml"]).unwrap();
        match cli.command {
            Command::Info(args) => {
                assert_eq!(args.format, OutputFormat::Yaml);
            }
            _ => panic!("Expected Info command"),
        }
    }

    // Research command tests

    #[test]
    fn test_parse_research_init() {
        let cli = parse_args([
            "entrenar",
            "research",
            "init",
            "--id",
            "dataset-2024",
            "--title",
            "My Dataset",
        ])
        .unwrap();

        match cli.command {
            Command::Research(args) => match args.command {
                ResearchCommand::Init(init_args) => {
                    assert_eq!(init_args.id, "dataset-2024");
                    assert_eq!(init_args.title, "My Dataset");
                    assert_eq!(init_args.artifact_type, ArtifactTypeArg::Dataset);
                    assert_eq!(init_args.license, LicenseArg::CcBy4);
                }
                _ => panic!("Expected Init subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_init_with_options() {
        let cli = parse_args([
            "entrenar",
            "research",
            "init",
            "--id",
            "paper-2024",
            "--title",
            "Novel Approach",
            "--artifact-type",
            "paper",
            "--license",
            "mit",
            "--author",
            "Alice Smith",
            "--orcid",
            "0000-0002-1825-0097",
            "--affiliation",
            "MIT",
            "--output",
            "custom.yaml",
        ])
        .unwrap();

        match cli.command {
            Command::Research(args) => match args.command {
                ResearchCommand::Init(init_args) => {
                    assert_eq!(init_args.artifact_type, ArtifactTypeArg::Paper);
                    assert_eq!(init_args.license, LicenseArg::Mit);
                    assert_eq!(init_args.author, Some("Alice Smith".to_string()));
                    assert_eq!(init_args.orcid, Some("0000-0002-1825-0097".to_string()));
                    assert_eq!(init_args.output, PathBuf::from("custom.yaml"));
                }
                _ => panic!("Expected Init subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_preregister() {
        let cli = parse_args([
            "entrenar",
            "research",
            "preregister",
            "--title",
            "Effect of X on Y",
            "--hypothesis",
            "X improves Y by 20%",
            "--methodology",
            "RCT, n=100",
            "--analysis-plan",
            "t-test, alpha=0.05",
        ])
        .unwrap();

        match cli.command {
            Command::Research(args) => match args.command {
                ResearchCommand::Preregister(prereg_args) => {
                    assert_eq!(prereg_args.title, "Effect of X on Y");
                    assert_eq!(prereg_args.hypothesis, "X improves Y by 20%");
                    assert_eq!(prereg_args.methodology, "RCT, n=100");
                    assert_eq!(prereg_args.analysis_plan, "t-test, alpha=0.05");
                }
                _ => panic!("Expected Preregister subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_cite() {
        let cli = parse_args([
            "entrenar",
            "research",
            "cite",
            "artifact.yaml",
            "--year",
            "2024",
        ])
        .unwrap();

        match cli.command {
            Command::Research(args) => match args.command {
                ResearchCommand::Cite(cite_args) => {
                    assert_eq!(cite_args.artifact, PathBuf::from("artifact.yaml"));
                    assert_eq!(cite_args.year, 2024);
                    assert_eq!(cite_args.format, CitationFormat::Bibtex);
                }
                _ => panic!("Expected Cite subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_cite_cff_format() {
        let cli = parse_args([
            "entrenar",
            "research",
            "cite",
            "artifact.yaml",
            "--year",
            "2024",
            "--format",
            "cff",
            "--journal",
            "Nature",
            "--output",
            "CITATION.cff",
        ])
        .unwrap();

        match cli.command {
            Command::Research(args) => match args.command {
                ResearchCommand::Cite(cite_args) => {
                    assert_eq!(cite_args.format, CitationFormat::Cff);
                    assert_eq!(cite_args.journal, Some("Nature".to_string()));
                    assert_eq!(cite_args.output, Some(PathBuf::from("CITATION.cff")));
                }
                _ => panic!("Expected Cite subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_export() {
        let cli = parse_args([
            "entrenar",
            "research",
            "export",
            "document.md",
            "--format",
            "notebook",
            "--output",
            "analysis.ipynb",
        ])
        .unwrap();

        match cli.command {
            Command::Research(args) => match args.command {
                ResearchCommand::Export(export_args) => {
                    assert_eq!(export_args.input, PathBuf::from("document.md"));
                    assert_eq!(export_args.format, ExportFormat::Notebook);
                    assert_eq!(export_args.output, PathBuf::from("analysis.ipynb"));
                }
                _ => panic!("Expected Export subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_export_anonymized() {
        let cli = parse_args([
            "entrenar",
            "research",
            "export",
            "artifact.yaml",
            "--format",
            "anonymized",
            "--output",
            "anon.json",
            "--anonymize",
            "--anon-salt",
            "review-2024",
        ])
        .unwrap();

        match cli.command {
            Command::Research(args) => match args.command {
                ResearchCommand::Export(export_args) => {
                    assert!(export_args.anonymize);
                    assert_eq!(export_args.anon_salt, Some("review-2024".to_string()));
                }
                _ => panic!("Expected Export subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_deposit() {
        let cli = parse_args([
            "entrenar",
            "research",
            "deposit",
            "artifact.yaml",
            "--provider",
            "zenodo",
            "--sandbox",
            "--file",
            "data.csv",
        ])
        .unwrap();

        match cli.command {
            Command::Research(args) => match args.command {
                ResearchCommand::Deposit(deposit_args) => {
                    assert_eq!(deposit_args.provider, ArchiveProviderArg::Zenodo);
                    assert!(deposit_args.sandbox);
                    assert_eq!(deposit_args.file, vec![PathBuf::from("data.csv")]);
                }
                _ => panic!("Expected Deposit subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_bundle() {
        let cli = parse_args([
            "entrenar",
            "research",
            "bundle",
            "artifact.yaml",
            "--output",
            "./ro-crate",
            "--file",
            "data.csv",
            "--file",
            "README.md",
            "--zip",
        ])
        .unwrap();

        match cli.command {
            Command::Research(args) => match args.command {
                ResearchCommand::Bundle(bundle_args) => {
                    assert_eq!(bundle_args.output, PathBuf::from("./ro-crate"));
                    assert_eq!(bundle_args.file.len(), 2);
                    assert!(bundle_args.zip);
                }
                _ => panic!("Expected Bundle subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_verify() {
        let cli = parse_args([
            "entrenar",
            "research",
            "verify",
            "preregistration.yaml",
            "--verify-git",
        ])
        .unwrap();

        match cli.command {
            Command::Research(args) => match args.command {
                ResearchCommand::Verify(verify_args) => {
                    assert_eq!(verify_args.file, PathBuf::from("preregistration.yaml"));
                    assert!(verify_args.verify_git);
                }
                _ => panic!("Expected Verify subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_artifact_type_from_str() {
        assert_eq!(
            "dataset".parse::<ArtifactTypeArg>().unwrap(),
            ArtifactTypeArg::Dataset
        );
        assert_eq!(
            "paper".parse::<ArtifactTypeArg>().unwrap(),
            ArtifactTypeArg::Paper
        );
        assert_eq!(
            "model".parse::<ArtifactTypeArg>().unwrap(),
            ArtifactTypeArg::Model
        );
        assert_eq!(
            "code".parse::<ArtifactTypeArg>().unwrap(),
            ArtifactTypeArg::Code
        );
        assert!("invalid".parse::<ArtifactTypeArg>().is_err());
    }

    #[test]
    fn test_license_from_str() {
        assert_eq!(
            "cc-by-4.0".parse::<LicenseArg>().unwrap(),
            LicenseArg::CcBy4
        );
        assert_eq!("mit".parse::<LicenseArg>().unwrap(), LicenseArg::Mit);
        assert_eq!(
            "apache-2.0".parse::<LicenseArg>().unwrap(),
            LicenseArg::Apache2
        );
        assert_eq!("cc0".parse::<LicenseArg>().unwrap(), LicenseArg::Cc0);
        assert!("invalid".parse::<LicenseArg>().is_err());
    }

    #[test]
    fn test_citation_format_from_str() {
        assert_eq!(
            "bibtex".parse::<CitationFormat>().unwrap(),
            CitationFormat::Bibtex
        );
        assert_eq!(
            "bib".parse::<CitationFormat>().unwrap(),
            CitationFormat::Bibtex
        );
        assert_eq!(
            "cff".parse::<CitationFormat>().unwrap(),
            CitationFormat::Cff
        );
        assert_eq!(
            "json".parse::<CitationFormat>().unwrap(),
            CitationFormat::Json
        );
        assert!("invalid".parse::<CitationFormat>().is_err());
    }

    #[test]
    fn test_export_format_from_str() {
        assert_eq!(
            "notebook".parse::<ExportFormat>().unwrap(),
            ExportFormat::Notebook
        );
        assert_eq!(
            "ipynb".parse::<ExportFormat>().unwrap(),
            ExportFormat::Notebook
        );
        assert_eq!("html".parse::<ExportFormat>().unwrap(), ExportFormat::Html);
        assert_eq!(
            "anonymized".parse::<ExportFormat>().unwrap(),
            ExportFormat::AnonymizedJson
        );
        assert_eq!(
            "ro-crate".parse::<ExportFormat>().unwrap(),
            ExportFormat::RoCrate
        );
        assert!("invalid".parse::<ExportFormat>().is_err());
    }

    #[test]
    fn test_archive_provider_from_str() {
        assert_eq!(
            "zenodo".parse::<ArchiveProviderArg>().unwrap(),
            ArchiveProviderArg::Zenodo
        );
        assert_eq!(
            "figshare".parse::<ArchiveProviderArg>().unwrap(),
            ArchiveProviderArg::Figshare
        );
        assert_eq!(
            "dryad".parse::<ArchiveProviderArg>().unwrap(),
            ArchiveProviderArg::Dryad
        );
        assert_eq!(
            "dataverse".parse::<ArchiveProviderArg>().unwrap(),
            ArchiveProviderArg::Dataverse
        );
        assert!("invalid".parse::<ArchiveProviderArg>().is_err());
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // Strategy for valid config paths
    fn config_path_strategy() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9_-]{0,20}\\.(yaml|yml)"
    }

    // Strategy for valid output paths
    fn output_path_strategy() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9_/-]{0,30}"
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_train_command_parses(config in config_path_strategy()) {
            let result = parse_args(["entrenar", "train", &config]);
            prop_assert!(result.is_ok());
            let cli = result.unwrap();
            match cli.command {
                Command::Train(args) => {
                    prop_assert_eq!(args.config.to_str().unwrap(), &config);
                }
                _ => prop_assert!(false, "Expected Train command"),
            }
        }

        #[test]
        fn prop_validate_command_parses(config in config_path_strategy()) {
            let result = parse_args(["entrenar", "validate", &config]);
            prop_assert!(result.is_ok());
            let cli = result.unwrap();
            match cli.command {
                Command::Validate(args) => {
                    prop_assert_eq!(args.config.to_str().unwrap(), &config);
                }
                _ => prop_assert!(false, "Expected Validate command"),
            }
        }

        #[test]
        fn prop_info_command_parses(config in config_path_strategy()) {
            let result = parse_args(["entrenar", "info", &config]);
            prop_assert!(result.is_ok());
        }

        #[test]
        fn prop_epochs_override_positive(
            config in config_path_strategy(),
            epochs in 1usize..10000
        ) {
            let epochs_str = epochs.to_string();
            let result = parse_args([
                "entrenar", "train", &config,
                "--epochs", &epochs_str,
            ]);
            prop_assert!(result.is_ok());
            let cli = result.unwrap();
            match cli.command {
                Command::Train(args) => {
                    prop_assert_eq!(args.epochs, Some(epochs));
                }
                _ => prop_assert!(false, "Expected Train command"),
            }
        }

        #[test]
        fn prop_batch_size_override_positive(
            config in config_path_strategy(),
            batch_size in 1usize..1024
        ) {
            let batch_str = batch_size.to_string();
            let result = parse_args([
                "entrenar", "train", &config,
                "--batch-size", &batch_str,
            ]);
            prop_assert!(result.is_ok());
            let cli = result.unwrap();
            match cli.command {
                Command::Train(args) => {
                    prop_assert_eq!(args.batch_size, Some(batch_size));
                }
                _ => prop_assert!(false, "Expected Train command"),
            }
        }

        #[test]
        fn prop_learning_rate_override(
            config in config_path_strategy(),
            lr in 1e-10f32..1.0
        ) {
            let lr_str = format!("{:.10}", lr);
            let result = parse_args([
                "entrenar", "train", &config,
                "--lr", &lr_str,
            ]);
            prop_assert!(result.is_ok());
            let cli = result.unwrap();
            match cli.command {
                Command::Train(args) => {
                    let parsed_lr = args.lr.unwrap();
                    // Allow for float parsing precision
                    prop_assert!((parsed_lr - lr).abs() < 1e-6 || (parsed_lr / lr - 1.0).abs() < 1e-4);
                }
                _ => prop_assert!(false, "Expected Train command"),
            }
        }

        #[test]
        fn prop_seed_override(
            config in config_path_strategy(),
            seed in 0u64..u64::MAX
        ) {
            let seed_str = seed.to_string();
            let result = parse_args([
                "entrenar", "train", &config,
                "--seed", &seed_str,
            ]);
            prop_assert!(result.is_ok());
            let cli = result.unwrap();
            match cli.command {
                Command::Train(args) => {
                    prop_assert_eq!(args.seed, Some(seed));
                }
                _ => prop_assert!(false, "Expected Train command"),
            }
        }

        #[test]
        fn prop_quantize_bits_valid(
            model in output_path_strategy(),
            bits in prop::sample::select(vec![4u8, 8])
        ) {
            let bits_str = bits.to_string();
            let result = parse_args([
                "entrenar", "quantize", &model,
                "--output", "out.gguf",
                "--bits", &bits_str,
            ]);
            prop_assert!(result.is_ok());
            let cli = result.unwrap();
            match cli.command {
                Command::Quantize(args) => {
                    prop_assert_eq!(args.bits, bits);
                }
                _ => prop_assert!(false, "Expected Quantize command"),
            }
        }

        #[test]
        fn prop_merge_weight_valid(
            weight in 0.0f32..=1.0
        ) {
            let weight_str = format!("{:.4}", weight);
            let result = parse_args([
                "entrenar", "merge",
                "model1.gguf", "model2.gguf",
                "--output", "merged.gguf",
                "--method", "slerp",
                "--weight", &weight_str,
            ]);
            prop_assert!(result.is_ok());
            let cli = result.unwrap();
            match cli.command {
                Command::Merge(args) => {
                    let parsed_weight = args.weight.unwrap();
                    prop_assert!((parsed_weight - weight).abs() < 1e-3);
                }
                _ => prop_assert!(false, "Expected Merge command"),
            }
        }

        #[test]
        fn prop_output_format_case_insensitive(
            format in prop::sample::select(vec!["text", "TEXT", "Text", "json", "JSON", "Json", "yaml", "YAML", "Yaml"])
        ) {
            let result = format.parse::<OutputFormat>();
            prop_assert!(result.is_ok());
        }

        #[test]
        fn prop_merge_method_case_insensitive(
            method in prop::sample::select(vec!["ties", "TIES", "dare", "DARE", "slerp", "SLERP", "average", "avg"])
        ) {
            let result = method.parse::<MergeMethod>();
            prop_assert!(result.is_ok());
        }

        #[test]
        fn prop_quant_method_case_insensitive(
            method in prop::sample::select(vec!["symmetric", "SYMMETRIC", "sym", "SYM", "asymmetric", "asym"])
        ) {
            let result = method.parse::<QuantMethod>();
            prop_assert!(result.is_ok());
        }

        #[test]
        fn prop_verbose_quiet_exclusive(config in config_path_strategy()) {
            // Can't have both verbose and quiet
            let cli_v = parse_args(["entrenar", "-v", "train", &config]).unwrap();
            let cli_q = parse_args(["entrenar", "-q", "train", &config]).unwrap();

            prop_assert!(cli_v.verbose && !cli_v.quiet);
            prop_assert!(!cli_q.verbose && cli_q.quiet);
        }

        #[test]
        fn prop_multiple_models_merge(
            model_count in 2usize..=5
        ) {
            let mut args: Vec<String> = vec!["entrenar".to_string(), "merge".to_string()];
            let models: Vec<String> = (0..model_count).map(|i| format!("model{}.gguf", i)).collect();
            for m in &models {
                args.push(m.clone());
            }
            args.push("--output".to_string());
            args.push("merged.gguf".to_string());

            let result = parse_args(&args);
            prop_assert!(result.is_ok());
            let cli = result.unwrap();
            match cli.command {
                Command::Merge(merge_args) => {
                    prop_assert_eq!(merge_args.models.len(), model_count);
                }
                _ => prop_assert!(false, "Expected Merge command"),
            }
        }
    }
}
