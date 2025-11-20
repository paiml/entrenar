# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Entrenar** is a Rust training and optimization library for neural networks, part of the PAIML stack. It provides autograd, optimizers, quantization (QAT/PTQ), LoRA/QLoRA, model merging (TIES/DARE/SLERP), and knowledge distillation.

**Status:** Specification phase - implementation not yet started.

**Stack Dependencies:**
- `trueno` - SIMD-accelerated tensor operations (compute layer)
- `realizar` - GGUF model I/O
- `aprender` - Loss functions

**Critical Constraint:** Entrenar depends on backward propagation operations that do not yet exist in Trueno. Phase 1 cannot start until `trueno/src/ops/backward.rs` is implemented.

## Architecture

### Core Type System

The foundation is a tape-based autograd system with lifetime-tracked gradient computation:

```rust
pub struct Tensor<'g, T> {
    data: trueno::Tensor<'g, T>,
    grad: Option<trueno::Tensor<'g, T>>,
    op: Option<Box<dyn BackwardOp<'g, T>>>,
}
```

The lifetime `'g` ensures gradient tape validity throughout backpropagation.

### Layer Structure

1. **Autograd Engine** (`src/autograd/`) - Tape-based eager-mode differentiation with backward operations
2. **Optimizers** (`src/optim/`) - SGD, Adam, AdamW, learning rate schedulers
3. **LoRA** (`src/lora/`) - Low-rank adaptation with QLoRA (4-bit base) support
4. **Quantization** (`src/quant/`) - QAT (fake quantize + STE) and PTQ (calibration)
5. **Model Merging** (`src/merge/`) - TIES, DARE, SLERP methods
6. **Distillation** (`src/distill/`) - Knowledge distillation with temperature-scaled softmax
7. **Declarative Config** (`src/config/`) - Ludwig-inspired YAML → trained model pipeline
8. **Training Loop** (`src/train/`) - High-level trainer with epoch/step abstractions

## Commands

### Quality Gates (Tiered TDD Workflow)

**Tier 1 (Fast <5s)** - Run before every commit:
```bash
make tier1    # Format, clippy, unit tests
```

**Tier 2 (Integration <30s)** - Run before push:
```bash
make tier2    # Tier 1 + integration tests
```

**Tier 3 (Full <5m)** - Run before PR:
```bash
make tier3    # Tier 1+2 + property tests
```

**Pre-Commit** - All quality gates:
```bash
make pre-commit    # Tier 1 + PMAT TDG check
```

**Full CI Pipeline** - Complete validation:
```bash
make ci    # Tier 3 + coverage + mutants + PMAT + security
```

### Development

```bash
# Build
make build              # Debug build
make release            # Release build

# Testing
make test               # Fast tests
cargo test test_name    # Specific test

# Code Quality
make lint               # Clippy
make format             # Format code
make check              # Type check

# Coverage (>90% required)
make coverage           # Generate HTML report
make coverage-clean     # Clean coverage data

# Mutation Testing (>80% kill rate required)
make mutants            # Full mutation testing
make mutants-quick      # Quick check on git diff

# Dependency Security
make deny-check         # Check for vulnerabilities

# Clean
make clean              # Remove build artifacts
```

### Ticket-Based Development (Required)

**ALL WORK MUST BE TRACKED VIA TICKETS** (ENT-001 through ENT-040).

Roadmap is tracked in `roadmap.yaml`. To start work:

1. **Find next ticket** in roadmap.yaml (status: pending)
2. **Update status** to `in-progress`
3. **Implement** following EXTREME TDD
4. **Update status** to `complete` with actual hours
5. **Commit** changes mentioning ticket ID

**Example workflow for ENT-002 (Matmul backward)**:

```bash
# 1. Edit roadmap.yaml: Set ENT-002 status to in-progress
vim roadmap.yaml

# 2. Implement with TDD
make tier1              # Fast feedback loop
cargo test matmul       # Specific test

# 3. Verify quality gates
make tier3              # Full validation
make coverage           # Check >90% coverage
make mutants-quick      # Quick mutation check

# 4. Update roadmap.yaml with completion
vim roadmap.yaml        # Set status: complete, actual_hours: 4

# 5. Commit
git add . && git commit -m "feat: ENT-002 matmul backward with gradient checking

- Implement matmul forward/backward operations
- Add property tests (1000+ cases)
- Gradient validation via finite difference
- Coverage: 95%, Mutation score: 85%

Closes ENT-002"
```

**Check progress**:
```bash
make roadmap-status     # View summary
grep "status: complete" roadmap.yaml | wc -l  # Count completed
```

**PMAT quality analysis** (optional):
```bash
make pmat-complexity    # Check complexity (<10)
make pmat-tdg           # Check TDG score (>90)
```

## Testing Requirements

**Zero tolerance for defects.** All code must meet these quality metrics:

- **Test Coverage:** >90% (cargo llvm-cov)
- **Mutation Kill Rate:** >80% (cargo-mutants)
- **TDG Score:** >90 (A grade via PMAT)
- **Cyclomatic Complexity:** ≤10
- **Cognitive Complexity:** ≤15
- **Gradient Error:** <1e-3 (property tests with finite difference validation)
- **Property Test Iterations:** 200K+ (proptest)

### Test-First Workflow

1. Write property test with gradient checking (finite difference validation)
2. Implement function to pass tests
3. Run mutation testing
4. Refactor if needed

Example test structure:
```rust
proptest! {
    #[test]
    fn softmax_backward_gradient_check(x in prop::collection::vec(-10.0f32..10.0, 1..100)) {
        let y = softmax(&x);
        let dy = vec![1.0; x.len()];
        let analytical = softmax_backward(&y, &dy);
        let numerical = finite_diff(|x| softmax(x).sum(), &x, 1e-5);
        prop_assert!((analytical - numerical).abs() < 1e-3);
    }
}
```

## Critical Implementation Details

### Backward Operations Required in Trueno

Before Entrenar implementation can begin, Trueno needs these operations in `trueno/src/ops/backward.rs`:

- `softmax_backward`: `∂L/∂x = softmax(x) * (∂L/∂y - dot(∂L/∂y, softmax(x)))`
- `layer_norm_backward`: Returns gradients for input, gamma, and beta
- `attention_backward`: Returns gradients for Q, K, V matrices
- `relu_backward`, `gelu_backward`, `swish_backward`: Element-wise activation gradients

All backward ops must use SIMD for small batches and GPU for large batches (5× threshold rule from Trueno).

### Quantization Output Format

- QAT/PTQ must output Realizar-compatible GGUF files
- Supports Q4_0 and Q8_0 block formats
- Per-channel and per-tensor quantization modes

### LoRA Memory Requirements

QLoRA provides 4× memory reduction vs full fine-tuning:
- Base weights: 4-bit quantized (frozen)
- Adapters: FP16/FP32 (trainable, rank r << min(d_in, d_out))
- On-the-fly dequantization during forward pass

## Declarative Configuration

Entrenar supports Ludwig-inspired YAML configs for zero-code training:

```yaml
model:
  path: llama-3-7b.gguf
  layers: [q_proj, k_proj, v_proj, o_proj]

data:
  train: train.parquet
  batch_size: 8
  auto_infer_types: true

optimizer:
  name: adam
  lr: 1e-4

lora:
  rank: 64
  alpha: 16

quantize:
  bits: 4
  symmetric: true

merge:
  method: TIES
  density: 0.2
```

Single command execution: `entrenar train config.yaml`

## Development Roadmap

**Total Estimate:** 824 hours (103 days @ 8h/day)

### Phase 1: Autograd (200h)
Core gradient computation with tape-based backpropagation. Depends on Trueno backward ops.

### Phase 2: Optimizers (120h)
SGD, Adam, AdamW with momentum and learning rate scheduling.

### Phase 3: LoRA (144h)
Low-rank adaptation with 4-bit base weights (QLoRA).

### Phase 4: Quantization (136h)
QAT with straight-through estimator, PTQ calibration, GGUF export.

### Phase 5: Model Merging (96h)
TIES (trim + sign election), DARE (dropout), SLERP (spherical interpolation).

### Phase 6: Declarative Config (64h)
YAML schema with auto-feature inference.

### Phase 7: Distillation (64h)
KD loss with temperature-scaled softmax.

All tickets tracked with PMAT (`ENT-001` through `ENT-040`).

## Pre-Commit Requirements

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test --lib  # Fast unit tests only
pmat analyze tdg src/ --min-score 90
```

## Benchmarks (Target Performance)

| Operation | Size | Target | Backend |
|-----------|------|--------|---------|
| Matmul backward | 512×512 | 3× forward | GPU |
| Adam step | 1M params | <10ms | SIMD |
| Softmax backward | 10K | 2× forward | SIMD |
| Q4_0 quantize | 1GB | <1s | Scalar |
| LoRA merge | 7B, r=64 | <5s | SIMD |

## Dependencies

```toml
[dependencies]
trueno = "0.x"  # SIMD tensor operations
realizar = "0.x"  # GGUF I/O
aprender = "0.x"  # Loss functions
serde = { version = "1", features = ["derive"] }
serde_yaml = "0.9"
serde_json = "1"

[dev-dependencies]
proptest = "1.4"  # Property-based testing
cargo-mutants = "25.3"  # Mutation testing
```

**Note:** Always use latest versions from crates.io. Never use git dependencies for PAIML stack libraries.
