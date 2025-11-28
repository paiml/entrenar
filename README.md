# Entrenar

<div align="center">
  <img src="docs/images/entrenar-logo.svg" alt="Entrenar - Training & Optimization Library" width="400">
</div>

<div align="center">

[![Crates.io](https://img.shields.io/crates/v/entrenar.svg)](https://crates.io/crates/entrenar)
[![Tests](https://img.shields.io/badge/tests-717%20passing-brightgreen)](https://github.com/paiml/entrenar)
[![Coverage](https://img.shields.io/badge/coverage-%3E90%25-brightgreen)](https://github.com/paiml/entrenar)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.83+-orange.svg)](https://www.rust-lang.org)

**Production-grade neural network training with autograd, optimizers, LoRA, quantization, model merging, and real-time monitoring.**

[Quick Start](#quick-start) | [Features](#features) | [Documentation](#documentation) | [PAIML Stack](#paiml-stack)

</div>

---

## Overview

**Entrenar** (Spanish: "to train") is a complete training and optimization library for neural networks, providing:

- **Tape-based Autograd** - Automatic differentiation with gradient checking
- **Optimizers** - SGD, Adam, AdamW with LR schedulers and gradient clipping
- **LoRA/QLoRA** - Parameter-efficient fine-tuning (99.75% param reduction)
- **Quantization** - QAT, PTQ, GGUF-compatible Q4_0/Q8_0 formats
- **Model Merging** - TIES, DARE, SLERP for combining fine-tuned models
- **Knowledge Distillation** - Temperature-scaled softmax, multi-teacher ensemble
- **Training Loop** - Callback system with early stopping, checkpoints, monitoring
- **Real-time Monitoring** - Toyota Way-inspired Andon system with drift detection

Part of the [PAIML Stack](#paiml-stack), built on [trueno](https://github.com/paiml/trueno) for SIMD-accelerated tensor operations.

## Quick Start

```bash
cargo add entrenar
```

### Basic Training Loop

```rust
use entrenar::train::{Trainer, TrainConfig, Batch, MSELoss, EarlyStopping};
use entrenar::optim::Adam;
use entrenar::Tensor;

// Setup
let params = vec![Tensor::zeros(784 * 128, true)];
let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
let config = TrainConfig::default();

let mut trainer = Trainer::new(params, Box::new(optimizer), config);
trainer.set_loss(Box::new(MSELoss));
trainer.add_callback(EarlyStopping::new(5, 0.001));

// Train with callbacks
let result = trainer.train(100, || batches.clone(), |x| model.forward(x));

println!("Trained {} epochs, loss: {:.4}", result.final_epoch, result.final_loss);
if result.stopped_early {
    println!("Early stopping triggered");
}
```

### Declarative Training (Ludwig-style)

```yaml
# config.yaml
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
entrenar train config.yaml
```

## Features

### Autograd Engine

Tape-based automatic differentiation with property-tested gradient checking:

```rust
use entrenar::autograd::{matmul, softmax, layer_norm, attention};

// All ops have verified backward passes
let y = matmul(&x, &w);           // dL/dX, dL/dW
let s = softmax(&logits);          // Jacobian-vector product
let n = layer_norm(&x, &gamma, &beta);  // dx, dgamma, dbeta
let a = attention(&q, &k, &v);     // dQ, dK, dV
```

**Operations:** matmul, add, mul, relu, gelu, swish, softmax, layer_norm, attention

### Optimizers

```rust
use entrenar::optim::{SGD, Adam, AdamW, CosineScheduler, clip_grad_norm};

let sgd = SGD::new(0.01, 0.9);              // With momentum
let adam = Adam::new(0.001, 0.9, 0.999, 1e-8);
let adamw = AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01);

// Learning rate scheduling
let scheduler = CosineScheduler::new(0.001, 0.0001, 100);

// Gradient clipping
clip_grad_norm(&mut params, 1.0);
```

### LoRA / QLoRA

Parameter-efficient fine-tuning with 99.75% parameter reduction:

```rust
use entrenar::lora::{LoRALayer, LoRAConfig, QLoRALayer};

let config = LoRAConfig::new()
    .with_rank(16)
    .with_alpha(32.0)
    .with_targets(&["q_proj", "k_proj", "v_proj", "o_proj"]);

// Standard LoRA
let lora = LoRALayer::new(4096, 4096, 16, 32.0);

// QLoRA (4-bit base + FP16 adapters)
let qlora = QLoRALayer::new(base_weights, 16, 32.0);
// 7B model: 28GB → 3.5GB (87% memory reduction)
```

### Quantization

QAT, PTQ, and GGUF-compatible formats:

```rust
use entrenar::quant::{FakeQuantize, PTQCalibrator, GGUFQuantizer, QuantConfig};

// Quantization-Aware Training (STE backward)
let fq = FakeQuantize::new(8, true);  // 8-bit symmetric
let quantized = fq.forward(&weights);

// Post-Training Quantization
let calibrator = PTQCalibrator::percentile(0.999);
calibrator.observe(&activations);
let scale = calibrator.compute_scale();

// GGUF export (llama.cpp compatible)
let quantizer = GGUFQuantizer::q4_0();
let packed = quantizer.quantize(&weights);  // Q4_0 block format
```

### Model Merging

Combine fine-tuned models with TIES, DARE, or SLERP:

```rust
use entrenar::merge::{TiesMerge, DareMerge, SlerpMerge};

// TIES: Trim + Sign Election + Merge
let ties = TiesMerge::new(0.2);  // 20% density
let merged = ties.merge(&[model_a, model_b, model_c], &[0.4, 0.3, 0.3]);

// DARE: Dropout + Rescale
let dare = DareMerge::new(0.9);  // 90% dropout
let merged = dare.merge(&base, &finetuned);

// SLERP: Spherical interpolation (2 models)
let slerp = SlerpMerge::new();
let merged = slerp.merge(&model_a, &model_b, 0.5);
```

### Knowledge Distillation

```rust
use entrenar::distill::{DistillationLoss, EnsembleDistiller, ProgressiveDistiller};

// Temperature-scaled KD
let kd_loss = DistillationLoss::new(4.0, 0.7);  // temp=4, alpha=0.7
let loss = kd_loss.compute(&student_logits, &teacher_logits, &labels);

// Multi-teacher ensemble
let ensemble = EnsembleDistiller::weighted(&[0.5, 0.3, 0.2]);
let combined = ensemble.combine(&teacher_outputs);

// Progressive layer-wise distillation
let progressive = ProgressiveDistiller::new(layer_weights);
let loss = progressive.layer_wise_loss(&student_hiddens, &teacher_hiddens);
```

### Training Loop & Callbacks

```rust
use entrenar::train::{
    Trainer, TrainResult, CallbackManager,
    EarlyStopping, CheckpointCallback, ProgressCallback, MonitorCallback,
};

let mut trainer = Trainer::new(params, optimizer, config);
trainer.set_loss(loss_fn);

// Add callbacks
trainer.add_callback(EarlyStopping::new(5, 0.001));       // Stop after 5 epochs without improvement
trainer.add_callback(CheckpointCallback::new("./ckpt")); // Save best + periodic
trainer.add_callback(ProgressCallback::new(10));         // Log every 10 steps
trainer.add_callback(MonitorCallback::new());            // NaN/Inf detection, metrics

let result: TrainResult = trainer.train(100, batch_fn, forward_fn);
// result.final_epoch, result.best_loss, result.stopped_early, result.elapsed_secs
```

### Real-time Monitoring

Toyota Way-inspired quality monitoring with Andon alerts:

```rust
use entrenar::monitor::{MetricsCollector, DriftDetector, AndonSystem, HanseiAnalyzer};

let mut collector = MetricsCollector::new();
let mut drift = DriftDetector::new(10);  // 10-epoch window
let mut andon = AndonSystem::new();

for epoch in 0..max_epochs {
    collector.record(Metric::Loss, loss);
    collector.record(Metric::LearningRate, lr);

    // Detect statistical drift (z-score based)
    if let DriftStatus::Drift(z) = drift.check(loss) {
        andon.warning(format!("Loss drift: z={:.2}", z));
    }

    // Andon stop on critical issues
    if andon.should_stop() {
        break;
    }
}

// Post-training Hansei (reflection) report
let analyzer = HanseiAnalyzer::new();
let report = analyzer.analyze("run-001", &collector, elapsed);
println!("{}", analyzer.format_report(&report));
```

## Architecture

```
entrenar/
├── autograd/      # Tape-based automatic differentiation
├── optim/         # SGD, Adam, AdamW, schedulers, gradient clipping
├── lora/          # LoRA, QLoRA parameter-efficient fine-tuning
├── quant/         # QAT, PTQ, GGUF Q4_0/Q8_0 quantization
├── merge/         # TIES, DARE, SLERP model merging
├── distill/       # Knowledge distillation (KD, ensemble, progressive)
├── config/        # Declarative YAML training configuration
├── train/         # Trainer, callbacks, training loop
├── monitor/       # Real-time metrics, drift detection, Andon alerts
└── io/            # Model save/load (JSON, YAML, GGUF)
```

## Development

### Quality Gates

```bash
# Tier 1 (<5s) - Before commit
make tier1    # Format, clippy, unit tests

# Tier 2 (<30s) - Before push
make tier2    # + Integration tests

# Tier 3 (<5m) - Before PR
make tier3    # + Property tests, coverage
```

### Test Coverage

- **717 tests** passing
- **>90% code coverage**
- **Property tests**: 200+ cases per property (proptest)
- **Gradient checking**: Finite difference validation

## PAIML Stack

Entrenar is part of the PAIML production ML stack:

| Library | Purpose | Status |
|---------|---------|--------|
| **[trueno](https://github.com/paiml/trueno)** | SIMD/GPU compute | v0.7.3 |
| **entrenar** | Training & optimization | v0.2.0 |
| **[aprender](https://github.com/paiml/aprender)** | .apr model format | v0.9.1 |
| **[realizar](https://github.com/paiml/realizar)** | GGUF export | planned |
| **[alimentar](https://github.com/paiml/alimentar)** | Dataset loading | planned |

## Documentation

- [API Reference](https://docs.rs/entrenar) - Complete API documentation
- [Training Loop Guide](docs/book/src/training-loop.md) - Callbacks and monitoring
- [Monitoring Spec](docs/specifications/training-monitoring-spec.md) - Real-time monitoring design
- [Roadmap](roadmap.yaml) - Development progress (52/52 tickets complete)

## License

MIT

---

<div align="center">

**Built with Extreme TDD** | Part of the [PAIML Stack](https://github.com/paiml)

</div>
