//! Inspect command implementation

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{InspectArgs, InspectMode};

pub fn run_inspect(args: InspectArgs, level: LogLevel) -> Result<(), String> {
    log(
        level,
        LogLevel::Normal,
        &format!("Inspecting: {}", args.input.display()),
    );

    // Check if file exists
    if !args.input.exists() {
        return Err(format!("File not found: {}", args.input.display()));
    }

    // Determine file type and load data
    let ext = args
        .input
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    log(level, LogLevel::Normal, &format!("  Mode: {}", args.mode));

    match ext {
        "safetensors" => {
            // Load and inspect SafeTensors model file
            use safetensors::SafeTensors;

            let data =
                std::fs::read(&args.input).map_err(|e| format!("Failed to read file: {e}"))?;

            let tensors = SafeTensors::deserialize(&data)
                .map_err(|e| format!("Failed to parse SafeTensors: {e}"))?;

            let tensor_names: Vec<String> =
                tensors.names().iter().map(|s| (*s).to_string()).collect();
            let mut total_params: u64 = 0;

            for name in &tensor_names {
                if let Ok(tensor) = tensors.tensor(name) {
                    let params: u64 = tensor.shape().iter().product::<usize>() as u64;
                    total_params += params;
                }
            }

            let file_size = data.len();

            log(level, LogLevel::Normal, "Model Information:");
            log(
                level,
                LogLevel::Normal,
                &format!("  File size: {:.2} MB", file_size as f64 / 1_000_000.0),
            );
            log(
                level,
                LogLevel::Normal,
                &format!("  Parameters: {:.2}B", total_params as f64 / 1e9),
            );
            log(
                level,
                LogLevel::Normal,
                &format!("  Tensors: {}", tensor_names.len()),
            );

            if level == LogLevel::Verbose {
                log(level, LogLevel::Verbose, "\nTensor Details:");
                for name in &tensor_names[..tensor_names.len().min(20)] {
                    if let Ok(tensor) = tensors.tensor(name) {
                        log(
                            level,
                            LogLevel::Verbose,
                            &format!("  {}: {:?} ({:?})", name, tensor.shape(), tensor.dtype()),
                        );
                    }
                }
                if tensor_names.len() > 20 {
                    log(
                        level,
                        LogLevel::Verbose,
                        &format!("  ... and {} more tensors", tensor_names.len() - 20),
                    );
                }
            }
        }
        "gguf" => {
            // GGUF inspection - basic file stats
            let metadata = std::fs::metadata(&args.input)
                .map_err(|e| format!("Failed to read metadata: {e}"))?;

            log(level, LogLevel::Normal, "GGUF Model Information:");
            log(
                level,
                LogLevel::Normal,
                &format!("  File size: {:.2} MB", metadata.len() as f64 / 1_000_000.0),
            );
            log(
                level,
                LogLevel::Normal,
                "  Format: GGUF (llama.cpp compatible)",
            );
            log(
                level,
                LogLevel::Normal,
                "  (Use llama.cpp for detailed GGUF inspection)",
            );
        }
        "parquet" | "csv" => {
            // Data file inspection
            let metadata = std::fs::metadata(&args.input)
                .map_err(|e| format!("Failed to read metadata: {e}"))?;

            match args.mode {
                InspectMode::Summary => {
                    log(level, LogLevel::Normal, "Data Summary:");
                    log(
                        level,
                        LogLevel::Normal,
                        &format!("  File size: {:.2} MB", metadata.len() as f64 / 1_000_000.0),
                    );
                    log(level, LogLevel::Normal, &format!("  Format: {ext}"));
                }
                InspectMode::Outliers => {
                    log(
                        level,
                        LogLevel::Normal,
                        &format!("Outlier Detection (z-threshold: {}):", args.z_threshold),
                    );
                    log(
                        level,
                        LogLevel::Normal,
                        "  Load data with alimentar for outlier analysis",
                    );
                }
                InspectMode::Distribution => {
                    log(level, LogLevel::Normal, "Distribution Statistics:");
                    log(
                        level,
                        LogLevel::Normal,
                        "  Load data with alimentar for distribution analysis",
                    );
                }
                InspectMode::Schema => {
                    log(level, LogLevel::Normal, "Schema:");
                    log(level, LogLevel::Normal, &format!("  Format: {ext}"));
                }
            }
        }
        _ => {
            return Err(format!(
                "Unsupported file format: {ext}. Use .safetensors, .gguf, .parquet, or .csv"
            ));
        }
    }

    Ok(())
}
