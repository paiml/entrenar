# Examples

Example models and configurations for testing and development purposes only.

## Files

- `example-model.gguf` - Mock GGUF model used in integration tests and CLI examples
- `train.parquet` - Sample training data for YAML config examples
- `yaml/` - YAML training configurations for all 100 Toyota Way QA scenarios
- `yaml_fixed/` - Fixed YAML configs matching binary TrainSpec schema
- `*.rs` - Rust example programs demonstrating entrenar features

## Usage

```bash
# Run a Rust example
cargo run --example training_loop

# Train from YAML config
cargo run --bin entrenar -- train examples/yaml/qlora.yaml
```

## Note

The `.gguf` model file in this directory is a mock artifact for testing.
It does not contain a real trained model and should not be used for inference.
