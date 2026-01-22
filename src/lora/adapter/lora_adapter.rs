//! LoRA adapter serialization and deserialization
//!
//! Contains the main LoRAAdapter struct for saving and loading adapters.

use super::error::AdapterError;
use super::metadata::AdapterMetadata;
use crate::lora::LoRALayer;
use crate::Tensor;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Serializable LoRA adapter format
///
/// Contains all information needed to reconstruct a LoRA adapter
/// (excluding the base weight, which remains frozen and separate)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LoRAAdapter {
    /// Format version for future compatibility
    version: String,
    /// LoRA rank
    rank: usize,
    /// LoRA alpha parameter
    alpha: f32,
    /// Output dimension
    d_out: usize,
    /// Input dimension
    d_in: usize,
    /// Computed scale factor (alpha/rank)
    scale: f32,
    /// LoRA A matrix weights [rank * d_in]
    lora_a: Vec<f32>,
    /// LoRA B matrix weights [d_out * rank]
    lora_b: Vec<f32>,
}

impl LoRAAdapter {
    /// Current adapter format version
    const VERSION: &'static str = "1.0";

    /// Create adapter from LoRALayer
    ///
    /// # Arguments
    /// * `layer` - LoRALayer to extract adapter from
    /// * `rank` - LoRA rank
    /// * `alpha` - LoRA alpha parameter
    pub fn from_layer(layer: &LoRALayer, rank: usize, alpha: f32) -> Self {
        Self {
            version: Self::VERSION.to_string(),
            rank,
            alpha,
            d_out: layer.d_out(),
            d_in: layer.d_in(),
            scale: layer.scale(),
            lora_a: layer.lora_a().data().to_vec(),
            lora_b: layer.lora_b().data().to_vec(),
        }
    }

    /// Load adapter and apply to base weight
    ///
    /// # Arguments
    /// * `base_weight` - Frozen base weight tensor [d_out * d_in]
    ///
    /// # Returns
    /// LoRALayer with loaded adapter weights
    pub fn to_layer(&self, base_weight: Tensor) -> Result<LoRALayer, AdapterError> {
        // Validate dimensions
        if base_weight.len() != self.d_out * self.d_in {
            return Err(AdapterError::DimensionMismatch {
                expected: format!("{}x{} = {}", self.d_out, self.d_in, self.d_out * self.d_in),
                actual: base_weight.len().to_string(),
            });
        }

        if self.lora_a.len() != self.rank * self.d_in {
            return Err(AdapterError::Validation(format!(
                "LoRA A size mismatch: expected {} (rank {} * d_in {}), got {}",
                self.rank * self.d_in,
                self.rank,
                self.d_in,
                self.lora_a.len()
            )));
        }

        if self.lora_b.len() != self.d_out * self.rank {
            return Err(AdapterError::Validation(format!(
                "LoRA B size mismatch: expected {} (d_out {} * rank {}), got {}",
                self.d_out * self.rank,
                self.d_out,
                self.rank,
                self.lora_b.len()
            )));
        }

        // Create layer with loaded weights
        let mut layer = LoRALayer::new(base_weight, self.d_out, self.d_in, self.rank, self.alpha);

        // Replace LoRA weights with loaded values
        *layer.lora_a_mut().data_mut() = ndarray::arr1(&self.lora_a);
        *layer.lora_b_mut().data_mut() = ndarray::arr1(&self.lora_b);

        Ok(layer)
    }

    /// Save adapter to JSON file
    ///
    /// # Arguments
    /// * `path` - File path to save to
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), AdapterError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Load adapter from JSON file
    ///
    /// # Arguments
    /// * `path` - File path to load from
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, AdapterError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let adapter: LoRAAdapter = serde_json::from_reader(reader)?;

        // Validate version
        if adapter.version != Self::VERSION {
            return Err(AdapterError::Validation(format!(
                "Unsupported adapter version: {} (expected {})",
                adapter.version,
                Self::VERSION
            )));
        }

        Ok(adapter)
    }

    /// Get adapter metadata
    pub fn metadata(&self) -> AdapterMetadata {
        AdapterMetadata {
            version: self.version.clone(),
            rank: self.rank,
            alpha: self.alpha,
            d_out: self.d_out,
            d_in: self.d_in,
            scale: self.scale,
            num_params: self.lora_a.len() + self.lora_b.len(),
        }
    }
}
