//! Architecture detection and SafeTensors file discovery

use super::Architecture;
use crate::error::Result;

/// Find SafeTensors files in a directory or return single file
pub(crate) fn find_safetensors_files(path: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
    if path.is_file() {
        return Ok(if path.extension().is_some_and(|e| e == "safetensors") {
            vec![path.to_path_buf()]
        } else {
            vec![]
        });
    }

    if !path.is_dir() {
        return Ok(vec![]);
    }

    // Check for single model.safetensors first
    let single = path.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    // Look for sharded files
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(path)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    files.sort();
    Ok(files)
}

/// Auto-detect model architecture from tensor names
pub(crate) fn detect_architecture(tensors: &safetensors::SafeTensors<'_>) -> Architecture {
    let names: Vec<String> = tensors.names().iter().map(std::string::ToString::to_string).collect();

    // RoBERTa / CodeBERT: look for "roberta." or "bert." prefix with encoder layers
    let is_roberta = names
        .iter()
        .any(|n| n.starts_with("roberta.") || (n.starts_with("bert.") && n.contains(".encoder.")));
    if is_roberta {
        return Architecture::RoBERTa;
    }

    // Qwen2 uses "model.layers.X.self_attn.q_proj" with biases
    // LLaMA uses same pattern but no biases
    // Check for bias tensors to distinguish
    let has_attn_bias = names.iter().any(|n: &String| {
        n.contains("self_attn.q_proj.bias") || n.contains("self_attn.k_proj.bias")
    });

    if has_attn_bias {
        // Qwen2 has attention biases
        return Architecture::Qwen2;
    }

    // Check for Mistral-specific patterns
    // Mistral uses sliding window attention config but same weight names as LLaMA
    // We default to LLaMA if no Qwen2 bias markers found

    Architecture::Llama
}
