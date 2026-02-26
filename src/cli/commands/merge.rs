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
use std::path::Path;

pub fn run_merge(args: MergeArgs, level: LogLevel) -> Result<(), String> {
    log_merge_start(&args, level);
    validate_model_count(&args)?;

    let models = load_all_models(&args.models, level)?;
    let merged = perform_merge(&models, &args)?;
    export_merged_model(&merged, &args)?;

    log_merge_complete(&merged, &args, level);
    Ok(())
}

/// Log merge operation start
fn log_merge_start(args: &MergeArgs, level: LogLevel) {
    log(
        level,
        LogLevel::Normal,
        &format!("Merging {} models using {:?}", args.models.len(), args.method),
    );

    for (i, model) in args.models.iter().enumerate() {
        log(level, LogLevel::Verbose, &format!("  Model {}: {}", i + 1, model.display()));
    }
    log(level, LogLevel::Verbose, &format!("  Output: {}", args.output.display()));
}

/// Validate we have enough models
fn validate_model_count(args: &MergeArgs) -> Result<(), String> {
    if args.models.len() < 2 {
        return Err("Need at least 2 models to merge".to_string());
    }
    Ok(())
}

/// Load all models from paths
fn load_all_models(paths: &[std::path::PathBuf], level: LogLevel) -> Result<Vec<Model>, String> {
    let mut models: Vec<Model> = Vec::new();
    for path in paths {
        let model = load_single_model(path)?;
        let tensor_count = model.len();
        models.push(model);

        log(
            level,
            LogLevel::Verbose,
            &format!("  Loaded {} tensors from {}", tensor_count, path.display()),
        );
    }
    Ok(models)
}

/// Load a single model from a SafeTensors file
fn load_single_model(path: &Path) -> Result<Model, String> {
    let data =
        std::fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?;

    let tensors = SafeTensors::deserialize(&data)
        .map_err(|e| format!("Failed to parse {}: {e}", path.display()))?;

    let mut model: Model = HashMap::new();
    for name in tensors.names() {
        if let Some(tensor) = extract_f32_tensor(&tensors, name)? {
            model.insert((*name).to_string(), tensor);
        }
    }
    Ok(model)
}

/// Extract a tensor as f32 values (returns None for non-F32 tensors)
fn extract_f32_tensor(tensors: &SafeTensors<'_>, name: &str) -> Result<Option<Tensor>, String> {
    let tensor = tensors.tensor(name).map_err(|e| format!("Failed to get tensor {name}: {e}"))?;

    if tensor.dtype() != safetensors::tensor::Dtype::F32 {
        return Ok(None);
    }

    let bytes = tensor.data();
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(Some(Tensor::from_vec(values, false)))
}

/// Perform the merge based on the specified method
fn perform_merge(models: &[Model], args: &MergeArgs) -> Result<Model, String> {
    match args.method {
        MergeMethod::Ties => perform_ties_merge(models, args),
        MergeMethod::Dare => perform_dare_merge(models, args),
        MergeMethod::Slerp => perform_slerp_merge(models, args),
        MergeMethod::Average => perform_average_merge(models, args),
    }
}

/// TIES merge: first model is base, rest are task-specific
fn perform_ties_merge(models: &[Model], args: &MergeArgs) -> Result<Model, String> {
    let config = TiesConfig { density: args.density.unwrap_or(0.2) };
    let base = &models[0];
    ties_merge(models.get(1..).unwrap_or_default(), base, &config)
        .map_err(|e| format!("TIES merge failed: {e}"))
}

/// DARE merge with dropout
fn perform_dare_merge(models: &[Model], args: &MergeArgs) -> Result<Model, String> {
    let config = DareConfig { drop_prob: 1.0 - args.density.unwrap_or(0.5), seed: None };
    let base = &models[0];
    dare_merge(models.get(1..).unwrap_or_default(), base, &config)
        .map_err(|e| format!("DARE merge failed: {e}"))
}

/// SLERP merge (requires exactly 2 models)
fn perform_slerp_merge(models: &[Model], args: &MergeArgs) -> Result<Model, String> {
    if models.len() != 2 {
        return Err("SLERP requires exactly 2 models".to_string());
    }
    let config = SlerpConfig { t: args.weight.unwrap_or(0.5) };
    slerp_merge(&models[0], &models[1], &config).map_err(|e| format!("SLERP merge failed: {e}"))
}

/// Average/ensemble merge with optional weights
fn perform_average_merge(models: &[Model], args: &MergeArgs) -> Result<Model, String> {
    let config = build_ensemble_config(args)?;
    ensemble_merge(models, &config).map_err(|e| format!("Average merge failed: {e}"))
}

/// Build ensemble config from args
fn build_ensemble_config(args: &MergeArgs) -> Result<EnsembleConfig, String> {
    if let Some(w_str) = &args.weights {
        let weights: Vec<f32> = w_str
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Invalid weights: {e}"))?;
        Ok(EnsembleConfig::weighted_average(weights))
    } else {
        Ok(EnsembleConfig::uniform_average())
    }
}

/// Export merged model to file
fn export_merged_model(merged: &Model, args: &MergeArgs) -> Result<(), String> {
    let output_ext = args.output.extension().and_then(|s| s.to_str()).unwrap_or("json");

    if output_ext == "safetensors" {
        export_safetensors(merged, args)
    } else {
        export_json(merged, args)
    }
}

/// Export to SafeTensors format
fn export_safetensors(merged: &Model, args: &MergeArgs) -> Result<(), String> {
    use safetensors::tensor::{Dtype, TensorView};

    let tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = merged
        .iter()
        .map(|(name, tensor)| {
            let data = tensor.data();
            let bytes: Vec<u8> = bytemuck::cast_slice(data.as_slice().unwrap_or(&[])).to_vec();
            let shape = vec![tensor.len()];
            (name.clone(), bytes, shape)
        })
        .collect();

    let views: Vec<(&str, TensorView<'_>)> = tensor_data
        .iter()
        .filter_map(|(name, bytes, shape)| {
            TensorView::new(Dtype::F32, shape.clone(), bytes).ok().map(|view| (name.as_str(), view))
        })
        .collect();

    let metadata = build_safetensor_metadata(merged, args);
    let safetensor_bytes = safetensors::serialize(views, Some(metadata))
        .map_err(|e| format!("Failed to serialize SafeTensors: {e}"))?;

    std::fs::write(&args.output, safetensor_bytes)
        .map_err(|e| format!("Failed to write output: {e}"))
}

/// Build SafeTensors metadata
fn build_safetensor_metadata(merged: &Model, args: &MergeArgs) -> HashMap<String, String> {
    let mut metadata = HashMap::new();
    metadata.insert("name".to_string(), "merged-model".to_string());
    metadata.insert("merge_method".to_string(), format!("{:?}", args.method));
    metadata.insert("tensor_count".to_string(), merged.len().to_string());
    metadata
}

/// Export to JSON format
fn export_json(merged: &Model, args: &MergeArgs) -> Result<(), String> {
    let output_data: HashMap<String, Vec<f32>> =
        merged.iter().map(|(name, tensor)| (name.clone(), tensor.data().to_vec())).collect();

    let json_data =
        serde_json::to_vec_pretty(&output_data).map_err(|e| format!("Failed to serialize: {e}"))?;

    std::fs::write(&args.output, &json_data).map_err(|e| format!("Failed to write output: {e}"))
}

/// Log merge completion
fn log_merge_complete(merged: &Model, args: &MergeArgs, level: LogLevel) {
    log(
        level,
        LogLevel::Normal,
        &format!("Merge complete: {} tensors written to {}", merged.len(), args.output.display()),
    );
}
