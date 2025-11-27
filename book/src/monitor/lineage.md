# Model Lineage

Track model versions, derivations, and identify regression sources.

## Usage

```rust
use entrenar::monitor::{ModelLineage, ModelMetadata, ChangeType};

let mut lineage = ModelLineage::new();

// Register base model
let base = ModelMetadata {
    id: "llama-7b-base".to_string(),
    version: "1.0.0".to_string(),
    parent_id: None,
    metrics: [("accuracy".to_string(), 0.75)].into(),
    timestamp: std::time::SystemTime::now(),
};
lineage.register(base);

// Register fine-tuned model
let finetuned = ModelMetadata {
    id: "llama-7b-lora".to_string(),
    version: "1.1.0".to_string(),
    parent_id: Some("llama-7b-base".to_string()),
    metrics: [("accuracy".to_string(), 0.82)].into(),
    timestamp: std::time::SystemTime::now(),
};
lineage.register(finetuned);
```

## Change Types

| Type | Description |
|------|-------------|
| `FineTune` | LoRA/QLoRA adaptation |
| `Merge` | Model merging (TIES/DARE/SLERP) |
| `Quantize` | Quantization (Q4_0, Q8_0) |
| `Distill` | Knowledge distillation |

## Regression Analysis

Find the source of a regression:

```rust
// If new model performs worse
if new_accuracy < old_accuracy * 0.95 {
    if let Some(source) = lineage.find_regression_source("llama-7b-v3") {
        println!("Regression introduced in: {}", source.id);
        println!("Change type: {:?}", source.change_type);
    }
}
```

## Lineage Visualization

```
llama-7b-base (v1.0.0)
├── llama-7b-lora (v1.1.0) [FineTune]
│   └── llama-7b-lora-q4 (v1.1.1) [Quantize]
└── llama-7b-merged (v1.2.0) [Merge]
```
