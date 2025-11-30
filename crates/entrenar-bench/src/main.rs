//! entrenar-bench CLI entry point.

use clap::{Parser, Subcommand};
use entrenar_bench::{
    strategies::{compare, DistillStrategy},
    sweep::{SweepConfig, Sweeper},
};
use entrenar_common::cli::{styles, CommonArgs};

#[derive(Parser)]
#[command(name = "entrenar-bench")]
#[command(about = "Distillation benchmarking and hyperparameter sweep tool")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Subcommand)]
enum Commands {
    /// Sweep temperature hyperparameter
    Temperature {
        /// Start of range
        #[arg(long, default_value = "1.0")]
        start: f32,

        /// End of range
        #[arg(long, default_value = "8.0")]
        end: f32,

        /// Step size
        #[arg(long, default_value = "0.5")]
        step: f32,

        /// Runs per configuration
        #[arg(long, default_value = "3")]
        runs: usize,
    },

    /// Sweep alpha hyperparameter
    Alpha {
        /// Start of range
        #[arg(long, default_value = "0.1")]
        start: f32,

        /// End of range
        #[arg(long, default_value = "0.9")]
        end: f32,

        /// Step size
        #[arg(long, default_value = "0.1")]
        step: f32,

        /// Runs per configuration
        #[arg(long, default_value = "3")]
        runs: usize,
    },

    /// Compare distillation strategies
    Compare {
        /// Strategies to compare (kd, progressive, attention, combined, all)
        #[arg(long, value_delimiter = ',', default_value = "all")]
        strategies: Vec<String>,

        /// Runs per strategy
        #[arg(long, default_value = "5")]
        runs: usize,
    },

    /// Run ablation study
    Ablation {
        /// Base configuration file
        #[arg(short, long)]
        config: Option<std::path::PathBuf>,
    },
}

fn main() {
    let cli = Cli::parse();
    let config = cli.common.to_cli();

    let result = match cli.command {
        Commands::Temperature {
            start,
            end,
            step,
            runs,
        } => temperature_command(start, end, step, runs, &config),
        Commands::Alpha {
            start,
            end,
            step,
            runs,
        } => alpha_command(start, end, step, runs, &config),
        Commands::Compare { strategies, runs } => compare_command(&strategies, runs, &config),
        Commands::Ablation { config: cfg_path } => ablation_command(cfg_path.as_deref(), &config),
    };

    if let Err(e) = result {
        if !config.is_quiet() {
            eprintln!("{}", styles::error(&e.to_string()));
        }
        std::process::exit(1);
    }
}

fn temperature_command(
    start: f32,
    end: f32,
    step: f32,
    runs: usize,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    if !cli.is_quiet() {
        println!("{}", styles::header("Temperature Sweep"));
        println!("Range: {start:.1} to {end:.1}, step {step:.1}, {runs} runs per point\n");
    }

    let config = SweepConfig::temperature(start..end, step).with_runs(runs);
    let sweeper = Sweeper::new(config);
    let result = sweeper.run()?;

    if cli.format == entrenar_common::OutputFormat::Json {
        let json: Vec<_> = result
            .data_points
            .iter()
            .map(|p| {
                serde_json::json!({
                    "value": p.parameter_value,
                    "loss": p.mean_loss,
                    "loss_std": p.std_loss,
                    "accuracy": p.mean_accuracy,
                    "accuracy_std": p.std_accuracy,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&json).unwrap());
    } else {
        println!("{}", result.to_table());
    }

    Ok(())
}

fn alpha_command(
    start: f32,
    end: f32,
    step: f32,
    runs: usize,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    if !cli.is_quiet() {
        println!("{}", styles::header("Alpha Sweep"));
        println!("Range: {start:.1} to {end:.1}, step {step:.1}, {runs} runs per point\n");
    }

    let config = SweepConfig::alpha(start..end, step).with_runs(runs);
    let sweeper = Sweeper::new(config);
    let result = sweeper.run()?;

    if cli.format == entrenar_common::OutputFormat::Json {
        let json: Vec<_> = result
            .data_points
            .iter()
            .map(|p| {
                serde_json::json!({
                    "value": p.parameter_value,
                    "loss": p.mean_loss,
                    "loss_std": p.std_loss,
                    "accuracy": p.mean_accuracy,
                    "accuracy_std": p.std_accuracy,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&json).unwrap());
    } else {
        println!("{}", result.to_table());
    }

    Ok(())
}

fn compare_command(
    strategy_names: &[String],
    _runs: usize,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let strategies: Vec<DistillStrategy> = if strategy_names.iter().any(|s| s == "all") {
        vec![
            DistillStrategy::kd_only(),
            DistillStrategy::progressive(),
            DistillStrategy::attention(),
            DistillStrategy::combined(),
        ]
    } else {
        strategy_names
            .iter()
            .filter_map(|name| match name.to_lowercase().as_str() {
                "kd" | "kd-only" | "kdonly" => Some(DistillStrategy::kd_only()),
                "progressive" | "prog" => Some(DistillStrategy::progressive()),
                "attention" | "attn" => Some(DistillStrategy::attention()),
                "combined" | "all" => Some(DistillStrategy::combined()),
                _ => None,
            })
            .collect()
    };

    if strategies.is_empty() {
        return Err(entrenar_common::EntrenarError::ConfigValue {
            field: "strategies".into(),
            message: "No valid strategies specified".into(),
            suggestion: "Use: kd, progressive, attention, combined, all".into(),
        });
    }

    if !cli.is_quiet() {
        println!("{}", styles::header("Strategy Comparison"));
        println!("Comparing {} strategies\n", strategies.len());
    }

    let comparison = compare(&strategies)?;

    if cli.format == entrenar_common::OutputFormat::Json {
        let json = serde_json::json!({
            "results": comparison.results.iter().map(|r| {
                serde_json::json!({
                    "strategy": r.name,
                    "loss": r.mean_loss,
                    "loss_std": r.std_loss,
                    "accuracy": r.mean_accuracy,
                    "accuracy_std": r.std_accuracy,
                    "time_hours": r.mean_time_hours,
                })
            }).collect::<Vec<_>>(),
            "best_by_loss": comparison.best_by_loss,
            "best_by_accuracy": comparison.best_by_accuracy,
        });
        println!("{}", serde_json::to_string_pretty(&json).unwrap());
    } else {
        println!("{}", comparison.to_table());

        if let Some(best) = &comparison.best_by_accuracy {
            println!(
                "\n{}",
                styles::success(&format!("Recommendation: {best} for best accuracy"))
            );
        }
    }

    Ok(())
}

fn ablation_command(
    _config_path: Option<&std::path::Path>,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    if !cli.is_quiet() {
        println!("{}", styles::header("Ablation Study"));
        println!("Testing contribution of each component...\n");
    }

    // Run ablation by progressively adding components
    let ablations = [
        (
            "Baseline (CE only)",
            DistillStrategy::KDOnly {
                temperature: 1.0,
                alpha: 0.0, // No KD, just CE
            },
        ),
        (
            "+ KD (T=4)",
            DistillStrategy::KDOnly {
                temperature: 4.0,
                alpha: 0.7,
            },
        ),
        (
            "+ Progressive",
            DistillStrategy::Progressive {
                temperature: 4.0,
                alpha: 0.7,
                layer_weight: 0.3,
            },
        ),
        (
            "+ Attention",
            DistillStrategy::Combined {
                temperature: 4.0,
                alpha: 0.7,
                layer_weight: 0.3,
                attention_weight: 0.1,
            },
        ),
    ];

    let strategies: Vec<DistillStrategy> = ablations.iter().map(|(_, s)| s.clone()).collect();
    let comparison = compare(&strategies)?;

    // Custom output for ablation
    println!("Ablation Results:");
    println!("┌─────────────────────┬────────────┬────────────┬────────────┐");
    println!("│ Configuration       │ Loss       │ Δ Loss     │ Accuracy   │");
    println!("├─────────────────────┼────────────┼────────────┼────────────┤");

    let mut prev_loss = None;
    for (i, (name, _)) in ablations.iter().enumerate() {
        let result = &comparison.results[i];
        let delta = prev_loss
            .map(|p: f64| result.mean_loss - p)
            .map_or_else(|| "-".to_string(), |d| format!("{d:+.4}"));

        println!(
            "│ {:19} │ {:>10.4} │ {:>10} │ {:>9.1}% │",
            name,
            result.mean_loss,
            delta,
            result.mean_accuracy * 100.0
        );

        prev_loss = Some(result.mean_loss);
    }

    println!("└─────────────────────┴────────────┴────────────┴────────────┘");

    Ok(())
}
