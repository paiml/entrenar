//! Inspect command implementation

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{InspectArgs, InspectMode};
use std::path::Path;

/// Inspect a SafeTensors model file
fn inspect_safetensors(path: &Path, level: LogLevel) -> Result<(), String> {
    use safetensors::SafeTensors;

    let data = std::fs::read(path).map_err(|e| format!("Failed to read file: {e}"))?;

    let tensors =
        SafeTensors::deserialize(&data).map_err(|e| format!("Failed to parse SafeTensors: {e}"))?;

    let tensor_names: Vec<String> = tensors.names().iter().map(|s| (*s).to_string()).collect();
    let total_params = count_total_parameters(&tensors, &tensor_names);
    let file_size = data.len();

    log_model_info(level, file_size, total_params, tensor_names.len());

    if level == LogLevel::Verbose {
        log_tensor_details(level, &tensors, &tensor_names);
    }

    Ok(())
}

/// Count total parameters across all tensors
fn count_total_parameters(tensors: &safetensors::SafeTensors<'_>, names: &[String]) -> u64 {
    let mut total: u64 = 0;
    for name in names {
        if let Ok(tensor) = tensors.tensor(name) {
            let params: u64 = tensor.shape().iter().product::<usize>() as u64;
            total += params;
        }
    }
    total
}

/// Log basic model information
fn log_model_info(level: LogLevel, file_size: usize, total_params: u64, tensor_count: usize) {
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
        &format!("  Tensors: {tensor_count}"),
    );
}

/// Log detailed tensor information
fn log_tensor_details(level: LogLevel, tensors: &safetensors::SafeTensors<'_>, names: &[String]) {
    log(level, LogLevel::Verbose, "\nTensor Details:");
    for name in &names[..names.len().min(20)] {
        if let Ok(tensor) = tensors.tensor(name) {
            log(
                level,
                LogLevel::Verbose,
                &format!("  {}: {:?} ({:?})", name, tensor.shape(), tensor.dtype()),
            );
        }
    }
    if names.len() > 20 {
        log(
            level,
            LogLevel::Verbose,
            &format!("  ... and {} more tensors", names.len() - 20),
        );
    }
}

/// Inspect a GGUF model file
fn inspect_gguf(path: &Path, level: LogLevel) -> Result<(), String> {
    let metadata = std::fs::metadata(path).map_err(|e| format!("Failed to read metadata: {e}"))?;

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

    Ok(())
}

/// Inspect a data file (parquet or csv)
fn inspect_data_file(
    path: &Path,
    ext: &str,
    mode: InspectMode,
    z_threshold: f32,
    level: LogLevel,
) -> Result<(), String> {
    let metadata = std::fs::metadata(path).map_err(|e| format!("Failed to read metadata: {e}"))?;

    match mode {
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
                &format!("Outlier Detection (z-threshold: {z_threshold}):"),
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

    Ok(())
}

/// Get file extension as lowercase string
fn get_extension(path: &Path) -> &str {
    path.extension().and_then(|s| s.to_str()).unwrap_or("")
}

pub fn run_inspect(args: InspectArgs, level: LogLevel) -> Result<(), String> {
    log(
        level,
        LogLevel::Normal,
        &format!("Inspecting: {}", args.input.display()),
    );

    if !args.input.exists() {
        return Err(format!("File not found: {}", args.input.display()));
    }

    let ext = get_extension(&args.input);
    log(level, LogLevel::Normal, &format!("  Mode: {}", args.mode));

    match ext {
        "safetensors" => inspect_safetensors(&args.input, level),
        "gguf" => inspect_gguf(&args.input, level),
        "parquet" | "csv" => {
            inspect_data_file(&args.input, ext, args.mode, args.z_threshold, level)
        }
        _ => Err(format!(
            "Unsupported file format: {ext}. Use .safetensors, .gguf, .parquet, or .csv"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_get_extension_safetensors() {
        let path = PathBuf::from("model.safetensors");
        assert_eq!(get_extension(&path), "safetensors");
    }

    #[test]
    fn test_get_extension_gguf() {
        let path = PathBuf::from("model.gguf");
        assert_eq!(get_extension(&path), "gguf");
    }

    #[test]
    fn test_get_extension_parquet() {
        let path = PathBuf::from("data.parquet");
        assert_eq!(get_extension(&path), "parquet");
    }

    #[test]
    fn test_get_extension_csv() {
        let path = PathBuf::from("data.csv");
        assert_eq!(get_extension(&path), "csv");
    }

    #[test]
    fn test_get_extension_none() {
        let path = PathBuf::from("noextension");
        assert_eq!(get_extension(&path), "");
    }

    #[test]
    fn test_run_inspect_file_not_found() {
        let args = InspectArgs {
            input: PathBuf::from("/nonexistent/path/model.safetensors"),
            mode: InspectMode::Summary,
            columns: None,
            z_threshold: 3.0,
        };
        let result = run_inspect(args, LogLevel::Normal);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("File not found"));
    }

    #[test]
    fn test_run_inspect_unsupported_format() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_inspect.xyz");
        std::fs::write(&path, "test").unwrap();

        let args = InspectArgs {
            input: path.clone(),
            mode: InspectMode::Summary,
            columns: None,
            z_threshold: 3.0,
        };
        let result = run_inspect(args, LogLevel::Normal);

        let _ = std::fs::remove_file(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unsupported file format"));
    }
}
