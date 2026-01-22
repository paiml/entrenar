//! LoRA adapter I/O convenience functions

use super::error::AdapterError;
use super::lora_adapter::LoRAAdapter;
use crate::lora::LoRALayer;
use crate::Tensor;
use std::path::Path;

/// Save LoRA adapter to file
///
/// # Arguments
/// * `layer` - LoRALayer to save
/// * `rank` - LoRA rank
/// * `alpha` - LoRA alpha parameter
/// * `path` - File path to save to
pub fn save_adapter<P: AsRef<Path>>(
    layer: &LoRALayer,
    rank: usize,
    alpha: f32,
    path: P,
) -> Result<(), AdapterError> {
    let adapter = LoRAAdapter::from_layer(layer, rank, alpha);
    adapter.save(path)
}

/// Load LoRA adapter from file
///
/// # Arguments
/// * `base_weight` - Frozen base weight to apply adapter to
/// * `path` - File path to load from
pub fn load_adapter<P: AsRef<Path>>(
    base_weight: Tensor,
    path: P,
) -> Result<LoRALayer, AdapterError> {
    let adapter = LoRAAdapter::load(path)?;
    adapter.to_layer(base_weight)
}
