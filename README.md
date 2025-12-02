# Entrenar

<div align="center">
  <img src="docs/images/entrenar-logo.svg" alt="Entrenar" width="400">

  <p><strong>Production-grade neural network training in pure Rust</strong></p>

[![Crates.io](https://img.shields.io/crates/v/entrenar.svg)](https://crates.io/crates/entrenar)
[![Documentation](https://docs.rs/entrenar/badge.svg)](https://docs.rs/entrenar)
[![Tests](https://img.shields.io/badge/tests-2155%20passing-brightgreen)](https://github.com/paiml/entrenar)
[![Coverage](https://img.shields.io/badge/coverage-%3E90%25-brightgreen)](https://github.com/paiml/entrenar)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.83+-orange.svg)](https://www.rust-lang.org)

[Getting Started](#getting-started) | [Features](#features) | [Examples](#examples) | [Documentation](https://docs.rs/entrenar)

</div>

---

## What is Entrenar?

**Entrenar** (Spanish: "to train") provides everything needed to train neural networks in Rust:

- **Autograd Engine** - Tape-based automatic differentiation
- **Optimizers** - SGD, Adam, AdamW with schedulers and gradient clipping
- **LoRA/QLoRA** - Parameter-efficient fine-tuning (4-bit quantized)
- **Quantization** - QAT, PTQ, GGUF-compatible Q4_0/Q8_0
- **Model Merging** - TIES, DARE, SLERP algorithms
- **Knowledge Distillation** - Multi-teacher, progressive layer-wise
- **Training Loop** - Callbacks, checkpoints, early stopping
- **Monitoring** - Real-time metrics, drift detection, Andon alerts
- **Explainability** - Feature attribution via SHAP, Integrated Gradients

Part of the [PAIML Stack](https://github.com/paiml), built on [trueno](https://crates.io/crates/trueno) for SIMD-accelerated operations.

## Getting Started

Add to your `Cargo.toml`:

```toml
[dependencies]
entrenar = "0.2"
```

### Basic Training

```rust
use entrenar::train::{Trainer, TrainConfig, Batch, MSELoss, EarlyStopping};
use entrenar::optim::Adam;
use entrenar::Tensor;

fn main() {
    // Model parameters
    let params = vec![Tensor::zeros(784 * 128, true)];
    let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);

    // Create trainer with callbacks
    let mut trainer = Trainer::new(params, Box::new(optimizer), TrainConfig::default());
    trainer.set_loss(Box::new(MSELoss));
    trainer.add_callback(EarlyStopping::new(5, 0.001));

    // Train
    let result = trainer.train(100, || batches.clone(), |x| model.forward(x));
    println!("Final loss: {:.4}", result.final_loss);
}
```

### Declarative Configuration

```yaml
# train.yaml
model:
  path: base-model.gguf
data:
  train: train.parquet
  batch_size: 8
optimizer:
  name: adamw
  lr: 0.0001
lora:
  rank: 64
  alpha: 16
training:
  epochs: 10
  grad_clip: 1.0
```

```bash
entrenar train train.yaml
```

## Features

### Autograd

Tape-based automatic differentiation with verified gradients:

```rust
use entrenar::autograd::{matmul, softmax, layer_norm, attention};

let y = matmul(&x, &w);                    // Matrix multiplication
let s = softmax(&logits);                  // Softmax activation
let n = layer_norm(&x, &gamma, &beta);     // Layer normalization
let a = attention(&q, &k, &v);             // Scaled dot-product attention
```

### Optimizers

```rust
use entrenar::optim::{SGD, Adam, AdamW, CosineScheduler};

let sgd = SGD::new(0.01, 0.9);
let adam = Adam::new(0.001, 0.9, 0.999, 1e-8);
let adamw = AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01);

// Learning rate scheduling
let scheduler = CosineScheduler::new(0.001, 0.0001, 100);
```

### LoRA / QLoRA

Parameter-efficient fine-tuning with up to 99.75% parameter reduction:

```rust
use entrenar::lora::{LoRALayer, QLoRALayer, LoRAConfig};

// Standard LoRA
let lora = LoRALayer::new(4096, 4096, 16, 32.0);

// QLoRA: 4-bit base + FP16 adapters
// 7B model: 28GB -> 3.5GB memory
let qlora = QLoRALayer::new(base_weights, 16, 32.0);
```

### Quantization

```rust
use entrenar::quant::{FakeQuantize, PTQCalibrator, GGUFQuantizer};

// QAT with straight-through estimator
let fq = FakeQuantize::new(8, true);

// Post-training quantization
let calibrator = PTQCalibrator::percentile(0.999);

// GGUF export (llama.cpp compatible)
let quantizer = GGUFQuantizer::q4_0();
```

### Model Merging

```rust
use entrenar::merge::{TiesMerge, DareMerge, SlerpMerge};

// TIES: Trim + Sign Election
let merged = TiesMerge::new(0.2).merge(&models, &weights);

// DARE: Dropout + Rescale
let merged = DareMerge::new(0.9).merge(&base, &finetuned);

// SLERP: Spherical interpolation
let merged = SlerpMerge::new().merge(&a, &b, 0.5);
```

### Knowledge Distillation

```rust
use entrenar::distill::{DistillationLoss, EnsembleDistiller};

// Temperature-scaled KD loss
let kd = DistillationLoss::new(4.0, 0.7);
let loss = kd.compute(&student, &teacher, &labels);

// Multi-teacher ensemble
let ensemble = EnsembleDistiller::weighted(&[0.5, 0.3, 0.2]);
```

### Training Callbacks

```rust
use entrenar::train::{
    EarlyStopping, CheckpointCallback, ProgressCallback,
    MonitorCallback, ExplainabilityCallback, ExplainMethod,
};

trainer.add_callback(EarlyStopping::new(5, 0.001));
trainer.add_callback(CheckpointCallback::new("./checkpoints"));
trainer.add_callback(ProgressCallback::new(10));
trainer.add_callback(MonitorCallback::new());  // NaN/Inf detection

// Feature importance tracking
trainer.add_callback(
    ExplainabilityCallback::new(ExplainMethod::PermutationImportance)
        .with_top_k(10)
);
```

### Real-Time Monitoring

Toyota Way-inspired quality monitoring:

```rust
use entrenar::monitor::{MetricsCollector, DriftDetector, AndonSystem};

let mut collector = MetricsCollector::new();
let mut drift = DriftDetector::new(10);
let mut andon = AndonSystem::new();

// Automatic drift detection and Andon alerts
if let DriftStatus::Drift(z) = drift.check(loss) {
    andon.warning(format!("Loss drift: z={:.2}", z));
}
```

## Examples

### Programmatic

```bash
cargo run --example training_loop      # Basic training
cargo run --example explainability     # Feature attribution
cargo run --example distillation       # Knowledge distillation
cargo run --example merge_models       # Model merging
cargo run --example model_io           # Save/load models
cargo run --example cli_bench          # Latency benchmarking
cargo run --example cli_audit          # Bias detection
cargo run --example cli_monitor        # Drift detection (PSI)
```

### CLI Commands

```bash
# Training
entrenar train config.yaml --epochs 10

# Model operations
entrenar quantize model.safetensors --bits 4 --output model_q4.json
entrenar merge model1.safetensors model2.safetensors --method ties

# Benchmarking & Monitoring
entrenar bench config.yaml --warmup 5 --iterations 100
entrenar inspect model.safetensors -v
entrenar audit predictions.parquet --type bias --threshold 0.8
entrenar monitor data.parquet --threshold 0.2

# Shell completions
entrenar completion bash > ~/.local/share/bash-completion/completions/entrenar
```

## Architecture

```
entrenar/
├── autograd/     Tape-based automatic differentiation
├── optim/        SGD, Adam, AdamW, schedulers
├── lora/         LoRA, QLoRA fine-tuning
├── quant/        QAT, PTQ, GGUF quantization
├── merge/        TIES, DARE, SLERP merging
├── distill/      Knowledge distillation
├── train/        Trainer, callbacks, metrics
├── monitor/      Real-time monitoring, Andon
├── config/       Declarative YAML config
└── io/           Model persistence
```

## Quality

| Metric | Value |
|--------|-------|
| Tests | 2155 passing |
| Coverage | >90% |
| Property Tests | 200K+ iterations |
| Gradient Checking | Finite difference validated |
| Mutation Testing | >80% kill rate |

## PAIML Stack

| Library | Purpose | Version |
|---------|---------|---------|
| [trueno](https://crates.io/crates/trueno) | SIMD tensor operations | 0.7.3 |
| **entrenar** | Training & optimization | 0.2.3 |
| [aprender](https://crates.io/crates/aprender) | ML algorithms & explainability | 0.14.0 |
| [realizar](https://crates.io/crates/realizar) | GGUF inference | 0.2.1 |

## Documentation

- [API Reference](https://docs.rs/entrenar)
- [Book](book/) - Comprehensive guide
- [Roadmap](roadmap.yaml) - 53/53 tickets complete

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with Extreme TDD | Part of <a href="https://github.com/paiml">PAIML</a></sub>
</div>
