//! Parquet batch loading using alimentar

use super::super::arrow::arrow_array_to_f32;
use super::super::demo::create_demo_batches;
use super::rebatch::rebatch;
use crate::error::{Error, Result};
use crate::train::Batch;
use crate::Tensor;
use alimentar::{ArrowDataset, Dataset};
use std::path::Path;

/// Load batches from parquet file using alimentar
pub fn load_parquet_batches(path: &Path, batch_size: usize) -> Result<Vec<Batch>> {
    println!("  Loading parquet: {}", path.display());

    let dataset = ArrowDataset::from_parquet(path).map_err(|e| {
        Error::ConfigError(format!("Failed to load parquet {}: {}", path.display(), e))
    })?;

    println!("  Loaded {} rows from parquet", dataset.len());

    // Convert Arrow RecordBatches to training Batches
    let mut batches = Vec::new();
    let schema = dataset.schema();

    // Get column names for input/output detection
    let column_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

    // Look for input/output columns (common patterns)
    let input_col = column_names
        .iter()
        .find(|&&n| n == "input" || n == "input_ids" || n == "x" || n == "features")
        .copied();
    let target_col = column_names
        .iter()
        .find(|&&n| n == "target" || n == "output" || n == "labels" || n == "y")
        .copied();

    if input_col.is_none() || target_col.is_none() {
        eprintln!(
            "Warning: Could not find input/target columns in parquet (found: {column_names:?})"
        );
        eprintln!("  Expected columns like: input/target, x/y, features/labels");
        return Ok(create_demo_batches(batch_size));
    }

    let input_name = input_col.unwrap();
    let target_name = target_col.unwrap();
    println!("  Using columns: input='{input_name}', target='{target_name}'");

    // Process each Arrow batch
    for record_batch in dataset.iter() {
        // Extract input and target arrays
        let input_idx = schema
            .index_of(input_name)
            .map_err(|e| Error::ConfigError(format!("Column not found: {e}")))?;
        let target_idx = schema
            .index_of(target_name)
            .map_err(|e| Error::ConfigError(format!("Column not found: {e}")))?;

        let input_array = record_batch.column(input_idx);
        let target_array = record_batch.column(target_idx);

        // Convert to f32 vectors (simplified - assumes numeric data)
        let input_data = arrow_array_to_f32(input_array)?;
        let target_data = arrow_array_to_f32(target_array)?;

        // Create batch
        let batch = Batch::new(
            Tensor::from_vec(input_data, false),
            Tensor::from_vec(target_data, false),
        );
        batches.push(batch);
    }

    // Re-batch to desired batch size if needed
    if batches.len() > 1 && batch_size > 0 {
        batches = rebatch(batches, batch_size);
    }

    Ok(batches)
}
