//! Parquet batch loading using alimentar

use super::super::arrow::arrow_array_to_f32;
use super::super::demo::create_demo_batches;
use super::rebatch::rebatch;
use crate::error::{Error, Result};
use crate::train::Batch;
use crate::Tensor;
use alimentar::{ArrowDataset, Dataset};
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use std::path::Path;

/// Column detection result
struct ColumnPair<'a> {
    input_name: &'a str,
    target_name: &'a str,
}

/// Detect input column from schema
fn detect_input_column<'a>(column_names: &[&'a str]) -> Option<&'a str> {
    column_names
        .iter()
        .find(|&&n| n == "input" || n == "input_ids" || n == "x" || n == "features")
        .copied()
}

/// Detect target column from schema
fn detect_target_column<'a>(column_names: &[&'a str]) -> Option<&'a str> {
    column_names
        .iter()
        .find(|&&n| n == "target" || n == "output" || n == "labels" || n == "y")
        .copied()
}

/// Detect input/target column pair from schema
fn detect_columns<'a>(column_names: &[&'a str]) -> Option<ColumnPair<'a>> {
    let input_name = detect_input_column(column_names)?;
    let target_name = detect_target_column(column_names)?;
    Some(ColumnPair {
        input_name,
        target_name,
    })
}

/// Log column detection warning and return demo batches
fn handle_missing_columns(column_names: &[&str], batch_size: usize) -> Vec<Batch> {
    eprintln!("Warning: Could not find input/target columns in parquet (found: {column_names:?})");
    eprintln!("  Expected columns like: input/target, x/y, features/labels");
    create_demo_batches(batch_size)
}

/// Convert a single record batch to a training batch
fn record_batch_to_training_batch(
    record_batch: &RecordBatch,
    schema: &Schema,
    input_name: &str,
    target_name: &str,
) -> Result<Batch> {
    let input_idx = schema
        .index_of(input_name)
        .map_err(|e| Error::ConfigError(format!("Column not found: {e}")))?;
    let target_idx = schema
        .index_of(target_name)
        .map_err(|e| Error::ConfigError(format!("Column not found: {e}")))?;

    let input_array = record_batch.column(input_idx);
    let target_array = record_batch.column(target_idx);

    let input_data = arrow_array_to_f32(input_array)?;
    let target_data = arrow_array_to_f32(target_array)?;

    Ok(Batch::new(
        Tensor::from_vec(input_data, false),
        Tensor::from_vec(target_data, false),
    ))
}

/// Process all record batches from dataset
fn process_record_batches(dataset: &ArrowDataset, columns: &ColumnPair<'_>) -> Result<Vec<Batch>> {
    let schema = dataset.schema();
    let mut batches = Vec::new();

    for record_batch in dataset.iter() {
        let batch = record_batch_to_training_batch(
            &record_batch,
            &schema,
            columns.input_name,
            columns.target_name,
        )?;
        batches.push(batch);
    }

    Ok(batches)
}

/// Load batches from parquet file using alimentar
pub fn load_parquet_batches(path: &Path, batch_size: usize) -> Result<Vec<Batch>> {
    println!("  Loading parquet: {}", path.display());

    let dataset = ArrowDataset::from_parquet(path).map_err(|e| {
        Error::ConfigError(format!("Failed to load parquet {}: {}", path.display(), e))
    })?;

    println!("  Loaded {} rows from parquet", dataset.len());

    let schema = dataset.schema();
    let column_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

    let columns = match detect_columns(&column_names) {
        Some(cols) => cols,
        None => return Ok(handle_missing_columns(&column_names, batch_size)),
    };

    println!(
        "  Using columns: input='{}', target='{}'",
        columns.input_name, columns.target_name
    );

    let mut batches = process_record_batches(&dataset, &columns)?;

    // Re-batch to desired batch size if needed
    if batches.len() > 1 && batch_size > 0 {
        batches = rebatch(batches, batch_size);
    }

    Ok(batches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_input_column_input() {
        let cols = vec!["input", "target"];
        assert_eq!(detect_input_column(&cols), Some("input"));
    }

    #[test]
    fn test_detect_input_column_input_ids() {
        let cols = vec!["input_ids", "labels"];
        assert_eq!(detect_input_column(&cols), Some("input_ids"));
    }

    #[test]
    fn test_detect_input_column_x() {
        let cols = vec!["x", "y"];
        assert_eq!(detect_input_column(&cols), Some("x"));
    }

    #[test]
    fn test_detect_input_column_features() {
        let cols = vec!["features", "labels"];
        assert_eq!(detect_input_column(&cols), Some("features"));
    }

    #[test]
    fn test_detect_input_column_none() {
        let cols = vec!["foo", "bar"];
        assert_eq!(detect_input_column(&cols), None);
    }

    #[test]
    fn test_detect_target_column_target() {
        let cols = vec!["input", "target"];
        assert_eq!(detect_target_column(&cols), Some("target"));
    }

    #[test]
    fn test_detect_target_column_output() {
        let cols = vec!["input", "output"];
        assert_eq!(detect_target_column(&cols), Some("output"));
    }

    #[test]
    fn test_detect_target_column_labels() {
        let cols = vec!["features", "labels"];
        assert_eq!(detect_target_column(&cols), Some("labels"));
    }

    #[test]
    fn test_detect_target_column_y() {
        let cols = vec!["x", "y"];
        assert_eq!(detect_target_column(&cols), Some("y"));
    }

    #[test]
    fn test_detect_target_column_none() {
        let cols = vec!["foo", "bar"];
        assert_eq!(detect_target_column(&cols), None);
    }

    #[test]
    fn test_detect_columns_success() {
        let cols = vec!["input", "target"];
        let result = detect_columns(&cols);
        assert!(result.is_some());
        let pair = result.unwrap();
        assert_eq!(pair.input_name, "input");
        assert_eq!(pair.target_name, "target");
    }

    #[test]
    fn test_detect_columns_missing_input() {
        let cols = vec!["foo", "target"];
        assert!(detect_columns(&cols).is_none());
    }

    #[test]
    fn test_detect_columns_missing_target() {
        let cols = vec!["input", "bar"];
        assert!(detect_columns(&cols).is_none());
    }

    #[test]
    fn test_handle_missing_columns_returns_demo_batches() {
        let cols = vec!["foo", "bar"];
        let batches = handle_missing_columns(&cols, 32);
        assert!(!batches.is_empty());
    }
}
