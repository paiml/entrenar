//! Build training components from configuration

use super::schema::{OptimSpec, TrainSpec};
use crate::error::{Error, Result};
use crate::io::{Model, ModelMetadata};
use crate::optim::{Adam, AdamW, Optimizer, SGD};
use crate::Tensor;

/// Build optimizer from configuration
pub fn build_optimizer(spec: &OptimSpec) -> Result<Box<dyn Optimizer>> {
    match spec.name.to_lowercase().as_str() {
        "sgd" => {
            let momentum = spec
                .params
                .get("momentum")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;

            Ok(Box::new(SGD::new(spec.lr, momentum)))
        }
        "adam" => {
            let beta1 = spec
                .params
                .get("beta1")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.9) as f32;

            let beta2 = spec
                .params
                .get("beta2")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.999) as f32;

            let eps = spec
                .params
                .get("eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-8) as f32;

            Ok(Box::new(Adam::new(spec.lr, beta1, beta2, eps)))
        }
        "adamw" => {
            let beta1 = spec
                .params
                .get("beta1")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.9) as f32;

            let beta2 = spec
                .params
                .get("beta2")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.999) as f32;

            let eps = spec
                .params
                .get("eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-8) as f32;

            let weight_decay = spec
                .params
                .get("weight_decay")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.01) as f32;

            Ok(Box::new(AdamW::new(
                spec.lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            )))
        }
        name => Err(Error::ConfigError(format!(
            "Unknown optimizer: {}. Supported: sgd, adam, adamw",
            name
        ))),
    }
}

/// Build a simple model from configuration
///
/// NOTE: This is a placeholder implementation until we have GGUF loading via Realizar.
/// For now, it creates a simple model with random parameters for demonstration.
pub fn build_model(spec: &TrainSpec) -> Result<Model> {
    // For demonstration, create a simple 2-layer model
    let params = vec![
        (
            "layer1.weight".to_string(),
            Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], true),
        ),
        (
            "layer1.bias".to_string(),
            Tensor::from_vec(vec![0.01, 0.02], true),
        ),
        (
            "layer2.weight".to_string(),
            Tensor::from_vec(vec![0.5, 0.6], true),
        ),
        ("layer2.bias".to_string(), Tensor::from_vec(vec![0.1], true)),
    ];

    let metadata = ModelMetadata::new(
        format!("model-from-{}", spec.model.path.display()),
        "simple-mlp",
    )
    .with_custom("config_path", serde_json::json!(spec.model.path))
    .with_custom("optimizer", serde_json::json!(spec.optimizer.name))
    .with_custom("learning_rate", serde_json::json!(spec.optimizer.lr))
    .with_custom("batch_size", serde_json::json!(spec.data.batch_size));

    Ok(Model::new(metadata, params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_build_optimizer_adam() {
        let mut params = std::collections::HashMap::new();
        params.insert("beta1".to_string(), serde_json::json!(0.9));
        params.insert("beta2".to_string(), serde_json::json!(0.999));

        let spec = OptimSpec {
            name: "adam".to_string(),
            lr: 0.001,
            params,
        };

        let optimizer = build_optimizer(&spec).unwrap();
        assert_eq!(optimizer.lr(), 0.001);
    }

    #[test]
    fn test_build_optimizer_sgd() {
        let mut params = std::collections::HashMap::new();
        params.insert("momentum".to_string(), serde_json::json!(0.9));

        let spec = OptimSpec {
            name: "sgd".to_string(),
            lr: 0.01,
            params,
        };

        let optimizer = build_optimizer(&spec).unwrap();
        assert_eq!(optimizer.lr(), 0.01);
    }

    #[test]
    fn test_build_optimizer_adamw() {
        let mut params = std::collections::HashMap::new();
        params.insert("weight_decay".to_string(), serde_json::json!(0.01));

        let spec = OptimSpec {
            name: "adamw".to_string(),
            lr: 0.001,
            params,
        };

        let optimizer = build_optimizer(&spec).unwrap();
        assert_eq!(optimizer.lr(), 0.001);
    }

    #[test]
    fn test_build_optimizer_unknown() {
        let spec = OptimSpec {
            name: "unknown".to_string(),
            lr: 0.001,
            params: std::collections::HashMap::new(),
        };

        let result = build_optimizer(&spec);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_model() {
        use super::super::schema::{DataConfig, ModelRef, TrainSpec, TrainingParams};

        let spec = TrainSpec {
            model: ModelRef {
                path: PathBuf::from("test.gguf"),
                layers: vec![],
            },
            data: DataConfig {
                train: PathBuf::from("train.parquet"),
                val: None,
                batch_size: 8,
                auto_infer_types: true,
                seq_len: None,
            },
            optimizer: OptimSpec {
                name: "adam".to_string(),
                lr: 0.001,
                params: std::collections::HashMap::new(),
            },
            lora: None,
            quantize: None,
            merge: None,
            training: TrainingParams::default(),
        };

        let model = build_model(&spec).unwrap();
        assert_eq!(model.parameters.len(), 4);
        assert!(model.get_parameter("layer1.weight").is_some());
    }
}
