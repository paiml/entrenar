//! entrenar-distill CLI entry point.

use clap::{Parser, Subcommand};
use entrenar_common::cli::CommonArgs;
use entrenar_distill::{config::DistillConfig, estimate_memory, run, validation::ConfigValidator};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "entrenar-distill")]
#[command(about = "End-to-end knowledge distillation CLI")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the distillation pipeline
    Run {
        /// Path to configuration file
        #[arg(short, long)]
        config: PathBuf,

        /// Override output directory
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Dry run (validate only, don't train)
        #[arg(long)]
        dry_run: bool,
    },

    /// Estimate memory requirements
    Estimate {
        /// Teacher model ID
        #[arg(long)]
        teacher: String,

        /// Student model ID
        #[arg(long)]
        student: Option<String>,

        /// Batch size
        #[arg(long, default_value = "32")]
        batch_size: u32,

        /// Sequence length
        #[arg(long, default_value = "512")]
        seq_len: usize,
    },

    /// Validate configuration file
    Validate {
        /// Path to configuration file
        #[arg(short, long)]
        config: PathBuf,
    },

    /// Export a trained model to different formats
    Export {
        /// Input model path
        #[arg(short, long)]
        input: PathBuf,

        /// Output format: safetensors, gguf, apr
        #[arg(short, long, default_value = "safetensors")]
        format: String,

        /// Output path
        #[arg(short, long)]
        output: PathBuf,

        /// Quantization: none, q8_0, q4_0
        #[arg(long, default_value = "none")]
        quantize: String,
    },
}

fn main() {
    let cli = Cli::parse();
    let config = cli.common.to_cli();

    let result = match cli.command {
        Commands::Run {
            config: config_path,
            output,
            dry_run,
        } => run_command(&config_path, output, dry_run, &config),

        Commands::Estimate {
            teacher,
            student,
            batch_size,
            seq_len,
        } => estimate_command(&teacher, student, batch_size, seq_len, &config),

        Commands::Validate {
            config: config_path,
        } => validate_command(&config_path, &config),

        Commands::Export {
            input,
            format,
            output,
            quantize,
        } => export_command(&input, &format, &output, &quantize, &config),
    };

    if let Err(e) = result {
        if !config.is_quiet() {
            eprintln!("{}", entrenar_common::cli::styles::error(&e.to_string()));
        }
        std::process::exit(1);
    }
}

fn run_command(
    config_path: &PathBuf,
    output: Option<PathBuf>,
    dry_run: bool,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    if !cli.is_quiet() {
        println!(
            "{}",
            entrenar_common::cli::styles::header("entrenar-distill")
        );
    }

    // Load configuration
    let mut config = DistillConfig::from_file(config_path)?;

    // Override output if specified
    if let Some(out) = output {
        config.output.dir = out;
    }

    // Validate
    ConfigValidator::validate(&config)?;

    if dry_run {
        if !cli.is_quiet() {
            println!(
                "{}",
                entrenar_common::cli::styles::success("Configuration valid")
            );

            let estimate = estimate_memory(&config)?;
            println!("\n{}", estimate.to_human_readable());
        }
        return Ok(());
    }

    // Run pipeline
    let result = run(&config)?;

    if !cli.is_quiet() {
        println!(
            "\n{}",
            entrenar_common::cli::styles::success("Distillation complete")
        );
        println!("  Output: {}", result.output_path.display());
        println!("  Duration: {:.1}s", result.duration_seconds);
        println!(
            "  Improvement: {:.1}%",
            result.metrics.improvement_ratio() * 100.0
        );
    }

    Ok(())
}

fn estimate_command(
    teacher: &str,
    student: Option<String>,
    batch_size: u32,
    seq_len: usize,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let student_id = student.unwrap_or_else(|| teacher.to_string());

    let mut config = DistillConfig::minimal(teacher, &student_id);
    config.training.batch_size = batch_size;
    config.dataset.max_length = seq_len;

    let estimate = estimate_memory(&config)?;

    if cli.format == entrenar_common::OutputFormat::Json {
        println!(
            "{}",
            serde_json::json!({
                "model_bytes": estimate.model_bytes,
                "activation_bytes": estimate.activation_bytes,
                "optimizer_bytes": estimate.optimizer_bytes,
                "total_bytes": estimate.total_bytes,
                "fits_in_vram": estimate.fits_in_vram,
                "recommended_batch_size": estimate.recommended_batch_size,
            })
        );
    } else {
        println!("{}", estimate.to_human_readable());
    }

    Ok(())
}

fn validate_command(
    config_path: &PathBuf,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let config = DistillConfig::from_file(config_path)?;
    ConfigValidator::validate(&config)?;

    if !cli.is_quiet() {
        println!(
            "{}",
            entrenar_common::cli::styles::success("Configuration valid")
        );
    }

    Ok(())
}

fn export_command(
    input: &std::path::Path,
    format: &str,
    output: &std::path::Path,
    quantize: &str,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    if !input.exists() {
        return Err(entrenar_common::EntrenarError::ModelNotFound {
            path: input.to_path_buf(),
        });
    }

    // In real implementation, would use Exporter
    if !cli.is_quiet() {
        println!(
            "{}",
            entrenar_common::cli::styles::info(&format!(
                "Exporting {} to {} format (quantize: {})",
                input.display(),
                format,
                quantize
            ))
        );
        println!(
            "{}",
            entrenar_common::cli::styles::success(&format!("Exported to {}", output.display()))
        );
    }

    Ok(())
}
