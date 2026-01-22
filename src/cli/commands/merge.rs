//! Merge command implementation

use crate::autograd::Tensor;
use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{MergeArgs, MergeMethod};
use crate::merge::{
    dare_merge, ensemble_merge, slerp_merge, ties_merge, DareConfig, EnsembleConfig, Model,
    SlerpConfig, TiesConfig,
};
use safetensors::SafeTensors;
use std::collections::HashMap;

pub fn run_merge(args: MergeArgs, level: LogLevel) -> Result<(), String> {
    log(
        level,
        LogLevel::Normal,
        &format!(
            "Merging {} models using {:?}",
            args.models.len(),
            args.method
        ),
    );

    for (i, model) in args.models.iter().enumerate() {
        log(
            level,
            LogLevel::Verbose,
            &format!("  Model {}: {}", i + 1, model.display()),
        );
    }
    log(
        level,
        LogLevel::Verbose,
        &format!("  Output: {}", args.output.display()),
    );

    // Validate we have enough models
    if args.models.len() < 2 {
        return Err("Need at least 2 models to merge".to_string());
    }

    // Load all models
    let mut models: Vec<Model> = Vec::new();
    for path in &args.models {
        let data =
            std::fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?;

        let tensors = SafeTensors::deserialize(&data)
            .map_err(|e| format!("Failed to parse {}: {e}", path.display()))?;

        let mut model: Model = HashMap::new();
        for name in tensors.names() {
            let tensor = tensors
                .tensor(name)
                .map_err(|e| format!("Failed to get tensor {name}: {e}"))?;

            // Only process F32 tensors
            if tensor.dtype() != safetensors::tensor::Dtype::F32 {
                continue;
            }

            let bytes = tensor.data();
            let values: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            model.insert((*name).to_string(), Tensor::from_vec(values, false));
        }
        models.push(model);

        let tensor_count = models.last().map_or(0, HashMap::len);
        log(
            level,
            LogLevel::Verbose,
            &format!("  Loaded {} tensors from {}", tensor_count, path.display()),
        );
    }

    // Perform merge
    let merged = match args.method {
        MergeMethod::Ties => {
            let config = TiesConfig {
                density: args.density.unwrap_or(0.2),
            };
            // First model is base, rest are task-specific
            let base = &models[0];
            ties_merge(&models[1..], base, &config)
                .map_err(|e| format!("TIES merge failed: {e}"))?
        }
        MergeMethod::Dare => {
            let config = DareConfig {
                drop_prob: 1.0 - args.density.unwrap_or(0.5), // density -> drop_prob
                seed: None,
            };
            let base = &models[0];
            dare_merge(&models[1..], base, &config)
                .map_err(|e| format!("DARE merge failed: {e}"))?
        }
        MergeMethod::Slerp => {
            if models.len() != 2 {
                return Err("SLERP requires exactly 2 models".to_string());
            }
            let config = SlerpConfig {
                t: args.weight.unwrap_or(0.5),
            };
            slerp_merge(&models[0], &models[1], &config)
                .map_err(|e| format!("SLERP merge failed: {e}"))?
        }
        MergeMethod::Average => {
            // Parse weights if provided
            let config = if let Some(w_str) = &args.weights {
                let weights: Vec<f32> = w_str
                    .split(',')
                    .map(|s| s.trim().parse::<f32>())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| format!("Invalid weights: {e}"))?;
                EnsembleConfig::weighted_average(weights)
            } else {
                EnsembleConfig::uniform_average()
            };

            ensemble_merge(&models, &config).map_err(|e| format!("Average merge failed: {e}"))?
        }
    };

    // Determine output format from file extension
    let output_ext = args
        .output
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("json");

    if output_ext == "safetensors" {
        // Export to SafeTensors format (HuggingFace compatible)
        use safetensors::tensor::{Dtype, TensorView};

        // Collect tensor data with proper lifetime management
        let tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = merged
            .iter()
            .map(|(name, tensor)| {
                let data = tensor.data();
                let bytes: Vec<u8> = bytemuck::cast_slice(data.as_slice().unwrap()).to_vec();
                let shape = vec![tensor.len()];
                (name.clone(), bytes, shape)
            })
            .collect();

        // Create TensorViews
        let views: Vec<(&str, TensorView<'_>)> = tensor_data
            .iter()
            .map(|(name, bytes, shape)| {
                let view = TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap();
                (name.as_str(), view)
            })
            .collect();

        // Create metadata
        let mut metadata = HashMap::new();
        metadata.insert("name".to_string(), "merged-model".to_string());
        metadata.insert("merge_method".to_string(), format!("{:?}", args.method));
        metadata.insert("tensor_count".to_string(), merged.len().to_string());

        // Serialize
        let safetensor_bytes = safetensors::serialize(views, Some(metadata))
            .map_err(|e| format!("Failed to serialize SafeTensors: {e}"))?;

        std::fs::write(&args.output, safetensor_bytes)
            .map_err(|e| format!("Failed to write output: {e}"))?;
    } else {
        // Fall back to JSON for other formats
        let output_data: HashMap<String, Vec<f32>> = merged
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.data().to_vec()))
            .collect();

        let json_data = serde_json::to_vec_pretty(&output_data)
            .map_err(|e| format!("Failed to serialize: {e}"))?;

        std::fs::write(&args.output, &json_data)
            .map_err(|e| format!("Failed to write output: {e}"))?;
    }

    log(
        level,
        LogLevel::Normal,
        &format!(
            "Merge complete: {} tensors written to {}",
            merged.len(),
            args.output.display()
        ),
    );

    Ok(())
}
