//! Extended command types - Completion, Bench, Inspect, Audit, Monitor

use clap::Parser;
use std::path::PathBuf;

use super::types::{AuditType, InspectMode, OutputFormat, ShellType};

/// Arguments for completion command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct CompletionArgs {
    /// Shell to generate completions for
    #[arg(value_name = "SHELL")]
    pub shell: ShellType,
}

/// Arguments for bench command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct BenchArgs {
    /// Path to model or config file
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Number of warmup iterations
    #[arg(long, default_value = "10")]
    pub warmup: usize,

    /// Number of benchmark iterations
    #[arg(long, default_value = "100")]
    pub iterations: usize,

    /// Batch sizes to test (comma-separated)
    #[arg(long, default_value = "1,8,32")]
    pub batch_sizes: String,

    /// Output format (text, json)
    #[arg(short, long, default_value = "text")]
    pub format: OutputFormat,
}

/// Arguments for inspect command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct InspectArgs {
    /// Path to data file or model
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Inspection mode
    #[arg(short, long, default_value = "summary")]
    pub mode: InspectMode,

    /// Columns to inspect (comma-separated)
    #[arg(long)]
    pub columns: Option<String>,

    /// Z-score threshold for outlier detection
    #[arg(long, default_value = "3.0")]
    pub z_threshold: f32,
}

/// Arguments for audit command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct AuditArgs {
    /// Path to model or config file
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Audit type
    #[arg(short, long, default_value = "bias")]
    pub audit_type: AuditType,

    /// Protected attribute column
    #[arg(long)]
    pub protected_attr: Option<String>,

    /// Fairness threshold (0.0-1.0)
    #[arg(long, default_value = "0.8")]
    pub threshold: f32,

    /// Output format (text, json, html)
    #[arg(short, long, default_value = "text")]
    pub format: OutputFormat,
}

/// Arguments for monitor command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct MonitorArgs {
    /// Path to model or config file
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Baseline statistics file
    #[arg(long)]
    pub baseline: Option<PathBuf>,

    /// Drift detection threshold (PSI)
    #[arg(long, default_value = "0.2")]
    pub threshold: f32,

    /// Monitoring interval in seconds
    #[arg(long, default_value = "60")]
    pub interval: u64,

    /// Output format (text, json)
    #[arg(short, long, default_value = "text")]
    pub format: OutputFormat,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::cli::parse_args;

    #[test]
    fn test_parse_completion_command() {
        let cli = parse_args(["entrenar", "completion", "bash"]).unwrap();
        match cli.command {
            crate::config::cli::Command::Completion(args) => {
                assert_eq!(args.shell, ShellType::Bash);
            }
            _ => panic!("Expected Completion command"),
        }
    }

    #[test]
    fn test_parse_bench_command() {
        let cli = parse_args(["entrenar", "bench", "model.gguf"]).unwrap();
        match cli.command {
            crate::config::cli::Command::Bench(args) => {
                assert_eq!(args.input, PathBuf::from("model.gguf"));
                assert_eq!(args.warmup, 10);
                assert_eq!(args.iterations, 100);
                assert_eq!(args.batch_sizes, "1,8,32");
            }
            _ => panic!("Expected Bench command"),
        }
    }

    #[test]
    fn test_parse_bench_with_options() {
        let cli = parse_args([
            "entrenar",
            "bench",
            "model.gguf",
            "--warmup",
            "5",
            "--iterations",
            "50",
            "--batch-sizes",
            "1,2,4,8",
            "--format",
            "json",
        ])
        .unwrap();
        match cli.command {
            crate::config::cli::Command::Bench(args) => {
                assert_eq!(args.warmup, 5);
                assert_eq!(args.iterations, 50);
                assert_eq!(args.batch_sizes, "1,2,4,8");
                assert_eq!(args.format, OutputFormat::Json);
            }
            _ => panic!("Expected Bench command"),
        }
    }

    #[test]
    fn test_parse_inspect_command() {
        let cli = parse_args(["entrenar", "inspect", "data.parquet"]).unwrap();
        match cli.command {
            crate::config::cli::Command::Inspect(args) => {
                assert_eq!(args.input, PathBuf::from("data.parquet"));
                assert_eq!(args.mode, InspectMode::Summary);
                assert!((args.z_threshold - 3.0).abs() < 1e-6);
            }
            _ => panic!("Expected Inspect command"),
        }
    }

    #[test]
    fn test_parse_inspect_with_options() {
        let cli = parse_args([
            "entrenar",
            "inspect",
            "data.parquet",
            "--mode",
            "outliers",
            "--columns",
            "col1,col2",
            "--z-threshold",
            "2.5",
        ])
        .unwrap();
        match cli.command {
            crate::config::cli::Command::Inspect(args) => {
                assert_eq!(args.mode, InspectMode::Outliers);
                assert_eq!(args.columns, Some("col1,col2".to_string()));
                assert!((args.z_threshold - 2.5).abs() < 1e-6);
            }
            _ => panic!("Expected Inspect command"),
        }
    }

    #[test]
    fn test_parse_audit_command() {
        let cli = parse_args(["entrenar", "audit", "model.gguf"]).unwrap();
        match cli.command {
            crate::config::cli::Command::Audit(args) => {
                assert_eq!(args.input, PathBuf::from("model.gguf"));
                assert_eq!(args.audit_type, AuditType::Bias);
                assert!((args.threshold - 0.8).abs() < 1e-6);
            }
            _ => panic!("Expected Audit command"),
        }
    }

    #[test]
    fn test_parse_audit_with_options() {
        let cli = parse_args([
            "entrenar",
            "audit",
            "model.gguf",
            "--audit-type",
            "fairness",
            "--protected-attr",
            "gender",
            "--threshold",
            "0.9",
            "--format",
            "json",
        ])
        .unwrap();
        match cli.command {
            crate::config::cli::Command::Audit(args) => {
                assert_eq!(args.audit_type, AuditType::Fairness);
                assert_eq!(args.protected_attr, Some("gender".to_string()));
                assert!((args.threshold - 0.9).abs() < 1e-6);
                assert_eq!(args.format, OutputFormat::Json);
            }
            _ => panic!("Expected Audit command"),
        }
    }

    #[test]
    fn test_parse_monitor_command() {
        let cli = parse_args(["entrenar", "monitor", "model.gguf"]).unwrap();
        match cli.command {
            crate::config::cli::Command::Monitor(args) => {
                assert_eq!(args.input, PathBuf::from("model.gguf"));
                assert!((args.threshold - 0.2).abs() < 1e-6);
                assert_eq!(args.interval, 60);
            }
            _ => panic!("Expected Monitor command"),
        }
    }

    #[test]
    fn test_parse_monitor_with_options() {
        let cli = parse_args([
            "entrenar",
            "monitor",
            "model.gguf",
            "--baseline",
            "baseline.json",
            "--threshold",
            "0.3",
            "--interval",
            "120",
            "--format",
            "json",
        ])
        .unwrap();
        match cli.command {
            crate::config::cli::Command::Monitor(args) => {
                assert_eq!(args.baseline, Some(PathBuf::from("baseline.json")));
                assert!((args.threshold - 0.3).abs() < 1e-6);
                assert_eq!(args.interval, 120);
                assert_eq!(args.format, OutputFormat::Json);
            }
            _ => panic!("Expected Monitor command"),
        }
    }
}
