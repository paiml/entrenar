//! Training batch loading from various data formats

use super::demo::create_demo_batches;
use crate::config::schema::TrainSpec;
use crate::error::{Error, Result};
use crate::train::Batch;
use crate::Tensor;
use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
use alimentar::{ArrowDataset, Dataset};

#[cfg(not(target_arch = "wasm32"))]
use super::arrow::arrow_array_to_f32;

/// Load training batches from data file using alimentar
///
/// Supports parquet, JSON, and CSV formats via alimentar.
/// Falls back to demo data if the file doesn't exist (for testing).
pub fn load_training_batches(spec: &TrainSpec) -> Result<Vec<Batch>> {
    let data_path = &spec.data.train;
    let batch_size = spec.data.batch_size;

    // Check if data file exists
    if !data_path.exists() {
        eprintln!(
            "Warning: Training data not found at '{}', using demo data",
            data_path.display()
        );
        return Ok(create_demo_batches(batch_size));
    }

    // Load data using alimentar (only on non-WASM)
    #[cfg(not(target_arch = "wasm32"))]
    {
        let ext = data_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "parquet" => load_parquet_batches(data_path, batch_size),
            "json" => load_json_batches(data_path, batch_size),
            _ => {
                eprintln!("Warning: Unsupported data format '{ext}', using demo data");
                Ok(create_demo_batches(batch_size))
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        eprintln!("Warning: Data loading not available in WASM, using demo data");
        Ok(create_demo_batches(batch_size))
    }
}

/// Load batches from parquet file using alimentar
#[cfg(not(target_arch = "wasm32"))]
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

/// Load batches from JSON file
#[cfg(not(target_arch = "wasm32"))]
pub fn load_json_batches(path: &Path, batch_size: usize) -> Result<Vec<Batch>> {
    println!("  Loading JSON: {}", path.display());

    // Try to load as JSON array of {input, target} objects
    let content = std::fs::read_to_string(path).map_err(|e| {
        Error::ConfigError(format!("Failed to read JSON {}: {}", path.display(), e))
    })?;

    #[derive(serde::Deserialize)]
    struct Example {
        input: Vec<f32>,
        target: Vec<f32>,
    }

    #[derive(serde::Deserialize)]
    struct DataFile {
        examples: Vec<Example>,
    }

    // Try structured format first
    if let Ok(data) = serde_json::from_str::<DataFile>(&content) {
        println!("  Loaded {} examples from JSON", data.examples.len());
        let batches: Vec<Batch> = data
            .examples
            .chunks(batch_size.max(1))
            .map(|chunk| {
                let input_data: Vec<f32> = chunk.iter().flat_map(|ex| ex.input.clone()).collect();
                let target_data: Vec<f32> = chunk.iter().flat_map(|ex| ex.target.clone()).collect();
                Batch::new(
                    Tensor::from_vec(input_data, false),
                    Tensor::from_vec(target_data, false),
                )
            })
            .collect();
        return Ok(batches);
    }

    // Try array of examples
    if let Ok(examples) = serde_json::from_str::<Vec<Example>>(&content) {
        println!("  Loaded {} examples from JSON array", examples.len());
        let batches: Vec<Batch> = examples
            .chunks(batch_size.max(1))
            .map(|chunk| {
                let input_data: Vec<f32> = chunk.iter().flat_map(|ex| ex.input.clone()).collect();
                let target_data: Vec<f32> = chunk.iter().flat_map(|ex| ex.target.clone()).collect();
                Batch::new(
                    Tensor::from_vec(input_data, false),
                    Tensor::from_vec(target_data, false),
                )
            })
            .collect();
        return Ok(batches);
    }

    eprintln!("Warning: Could not parse JSON data format, using demo data");
    Ok(create_demo_batches(batch_size))
}

/// Re-batch data into specified batch size
pub fn rebatch(batches: Vec<Batch>, batch_size: usize) -> Vec<Batch> {
    // Flatten all data
    let all_inputs: Vec<f32> = batches
        .iter()
        .flat_map(|b| b.inputs.data().iter().copied())
        .collect();
    let all_targets: Vec<f32> = batches
        .iter()
        .flat_map(|b| b.targets.data().iter().copied())
        .collect();

    if all_inputs.is_empty() {
        return Vec::new();
    }

    // Determine feature dimensions from first batch
    let input_dim = batches[0].inputs.len();
    let target_dim = batches[0].targets.len();

    // Re-batch
    let num_examples = all_inputs.len() / input_dim;
    let mut new_batches = Vec::new();

    for chunk_start in (0..num_examples).step_by(batch_size) {
        let chunk_end = (chunk_start + batch_size).min(num_examples);
        let input_start = chunk_start * input_dim;
        let input_end = chunk_end * input_dim;
        let target_start = chunk_start * target_dim;
        let target_end = chunk_end * target_dim;

        new_batches.push(Batch::new(
            Tensor::from_vec(all_inputs[input_start..input_end].to_vec(), false),
            Tensor::from_vec(all_targets[target_start..target_end].to_vec(), false),
        ));
    }

    new_batches
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_batch(input: Vec<f32>, target: Vec<f32>) -> Batch {
        Batch::new(
            Tensor::from_vec(input, false),
            Tensor::from_vec(target, false),
        )
    }

    #[test]
    fn test_rebatch_empty() {
        let batches: Vec<Batch> = Vec::new();
        let result = rebatch(batches, 4);
        assert!(result.is_empty());
    }

    #[test]
    fn test_rebatch_single_batch() {
        // 4 inputs with input_dim=2 means 2 examples
        // batch_size=2 -> 1 batch
        let batches = vec![make_batch(vec![1.0, 2.0, 3.0, 4.0], vec![0.1, 0.2])];
        let result = rebatch(batches, 2);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_rebatch_multiple_batches() {
        // 3 batches, each with input_dim=2, total 6 inputs = 3 examples
        // batch_size=2 -> 2 batches (2 + 1)
        let batches = vec![
            make_batch(vec![1.0, 2.0], vec![0.1]),
            make_batch(vec![3.0, 4.0], vec![0.2]),
            make_batch(vec![5.0, 6.0], vec![0.3]),
        ];
        let result = rebatch(batches, 2);
        assert_eq!(result.len(), 2); // 3 examples -> 2 batches
    }

    #[test]
    fn test_rebatch_batch_size_one() {
        // 3 inputs with input_dim=3 means 1 example
        // batch_size=1 -> 1 batch
        let batches = vec![make_batch(vec![1.0, 2.0, 3.0], vec![0.1, 0.2, 0.3])];
        let result = rebatch(batches, 1);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_rebatch_large_batch_size() {
        let batches = vec![
            make_batch(vec![1.0], vec![0.1]),
            make_batch(vec![2.0], vec![0.2]),
        ];
        let result = rebatch(batches, 100);
        assert_eq!(result.len(), 1); // All fit in one batch
    }

    #[test]
    fn test_load_training_batches_nonexistent_file() {
        use std::path::PathBuf;

        let spec = TrainSpec {
            model: crate::config::ModelRef {
                path: PathBuf::from("model.bin"),
                layers: vec![],
            },
            data: crate::config::DataConfig {
                train: PathBuf::from("/nonexistent/path/data.parquet"),
                val: None,
                batch_size: 8,
                auto_infer_types: false,
                seq_len: None,
            },
            optimizer: crate::config::OptimSpec {
                name: "adam".to_string(),
                lr: 0.001,
                params: Default::default(),
            },
            lora: None,
            quantize: None,
            merge: None,
            training: Default::default(),
        };

        let result = load_training_batches(&spec);
        assert!(result.is_ok());
        let batches = result.unwrap();
        assert!(!batches.is_empty()); // Should return demo batches
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_load_training_batches_unsupported_format() {
        use std::path::PathBuf;

        // Create a temp file with unsupported extension
        let mut temp_file = NamedTempFile::with_suffix(".xyz").unwrap();
        writeln!(temp_file, "test data").unwrap();

        let spec = TrainSpec {
            model: crate::config::ModelRef {
                path: PathBuf::from("model.bin"),
                layers: vec![],
            },
            data: crate::config::DataConfig {
                train: temp_file.path().to_path_buf(),
                val: None,
                batch_size: 8,
                auto_infer_types: false,
                seq_len: None,
            },
            optimizer: crate::config::OptimSpec {
                name: "adam".to_string(),
                lr: 0.001,
                params: Default::default(),
            },
            lora: None,
            quantize: None,
            merge: None,
            training: Default::default(),
        };

        let result = load_training_batches(&spec);
        assert!(result.is_ok());
        let batches = result.unwrap();
        assert!(!batches.is_empty()); // Should return demo batches
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_load_json_batches_structured_format() {
        let mut temp_file = NamedTempFile::with_suffix(".json").unwrap();
        writeln!(
            temp_file,
            r#"{{
                "examples": [
                    {{"input": [1.0, 2.0], "target": [0.1]}},
                    {{"input": [3.0, 4.0], "target": [0.2]}}
                ]
            }}"#
        )
        .unwrap();

        let result = load_json_batches(temp_file.path(), 2);
        assert!(result.is_ok());
        let batches = result.unwrap();
        assert!(!batches.is_empty());
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_load_json_batches_array_format() {
        let mut temp_file = NamedTempFile::with_suffix(".json").unwrap();
        writeln!(
            temp_file,
            r#"[
                {{"input": [1.0, 2.0], "target": [0.1]}},
                {{"input": [3.0, 4.0], "target": [0.2]}},
                {{"input": [5.0, 6.0], "target": [0.3]}}
            ]"#
        )
        .unwrap();

        let result = load_json_batches(temp_file.path(), 2);
        assert!(result.is_ok());
        let batches = result.unwrap();
        assert!(!batches.is_empty());
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_load_json_batches_invalid_format() {
        let mut temp_file = NamedTempFile::with_suffix(".json").unwrap();
        writeln!(temp_file, r#"{{"some": "other", "format": true}}"#).unwrap();

        let result = load_json_batches(temp_file.path(), 2);
        assert!(result.is_ok());
        // Should fall back to demo batches
        let batches = result.unwrap();
        assert!(!batches.is_empty());
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_load_json_batches_file_not_found() {
        let result = load_json_batches(Path::new("/nonexistent/file.json"), 2);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_load_json_batches_batch_size_one() {
        let mut temp_file = NamedTempFile::with_suffix(".json").unwrap();
        writeln!(
            temp_file,
            r#"[
                {{"input": [1.0], "target": [0.1]}},
                {{"input": [2.0], "target": [0.2]}}
            ]"#
        )
        .unwrap();

        let result = load_json_batches(temp_file.path(), 1);
        assert!(result.is_ok());
        let batches = result.unwrap();
        assert_eq!(batches.len(), 2);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_load_json_batches_batch_size_zero() {
        let mut temp_file = NamedTempFile::with_suffix(".json").unwrap();
        writeln!(temp_file, r#"[{{"input": [1.0], "target": [0.1]}}]"#).unwrap();

        let result = load_json_batches(temp_file.path(), 0);
        assert!(result.is_ok());
        // batch_size.max(1) ensures at least 1
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_load_training_batches_json_file() {
        use std::path::PathBuf;

        let mut temp_file = NamedTempFile::with_suffix(".json").unwrap();
        writeln!(temp_file, r#"[{{"input": [1.0, 2.0], "target": [0.5]}}]"#).unwrap();

        let spec = TrainSpec {
            model: crate::config::ModelRef {
                path: PathBuf::from("model.bin"),
                layers: vec![],
            },
            data: crate::config::DataConfig {
                train: temp_file.path().to_path_buf(),
                val: None,
                batch_size: 4,
                auto_infer_types: false,
                seq_len: None,
            },
            optimizer: crate::config::OptimSpec {
                name: "adam".to_string(),
                lr: 0.001,
                params: Default::default(),
            },
            lora: None,
            quantize: None,
            merge: None,
            training: Default::default(),
        };

        let result = load_training_batches(&spec);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rebatch_preserves_data() {
        // 2 batches, each with input_dim=2, total 4 inputs = 2 examples
        let batches = vec![
            make_batch(vec![1.0, 2.0], vec![10.0]),
            make_batch(vec![3.0, 4.0], vec![20.0]),
        ];

        // batch_size=1, 2 examples -> 2 batches
        let result = rebatch(batches, 1);
        assert_eq!(result.len(), 2);

        // Check data is preserved
        let all_inputs: Vec<f32> = result
            .iter()
            .flat_map(|b| b.inputs.data().iter().copied())
            .collect();
        assert_eq!(all_inputs, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_rebatch_exact_batch_size() {
        // 4 batches, each with input_dim=1, total 4 inputs = 4 examples
        // batch_size=2 -> 2 batches
        let batches = vec![
            make_batch(vec![1.0], vec![0.1]),
            make_batch(vec![2.0], vec![0.2]),
            make_batch(vec![3.0], vec![0.3]),
            make_batch(vec![4.0], vec![0.4]),
        ];
        let result = rebatch(batches, 2);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_rebatch_remainder() {
        // 5 batches, each with input_dim=1, total 5 inputs = 5 examples
        // batch_size=2 -> 3 batches (2 + 2 + 1)
        let batches = vec![
            make_batch(vec![1.0], vec![0.1]),
            make_batch(vec![2.0], vec![0.2]),
            make_batch(vec![3.0], vec![0.3]),
            make_batch(vec![4.0], vec![0.4]),
            make_batch(vec![5.0], vec![0.5]),
        ];
        let result = rebatch(batches, 2);
        assert_eq!(result.len(), 3);
    }
}
