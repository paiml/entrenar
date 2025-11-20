# entrenar

**Rust Training & Optimization Library**

Entrenar provides a tape-based autograd engine with optimizers, designed for training neural networks with support for LoRA, quantization, model merging, and knowledge distillation.

## Status

**Phase 1 (Autograd Engine): ✅ COMPLETED**

- ✅ Core Tensor type with gradient tracking
- ✅ Tape-based automatic differentiation
- ✅ Backward operations: Add, Mul, Scale, ReLU, Softmax, Sum
- ✅ Finite difference gradient validation
- ✅ Property-based tests (1000+ test cases per operation)
- ✅ SGD and Adam optimizers
- ✅ All tests passing (18 tests)
- ✅ Clippy clean (zero warnings)
- ✅ 1,130 lines of code

## Features

### Implemented

#### Autograd Engine
- **Tensor**: Core type with automatic gradient tracking
- **Operations**:
  - Arithmetic: `add`, `mul`, `scale`, `sum`
  - Activations: `relu`, `softmax`
- **Backward Pass**: Automatic differentiation via computational graph
- **Gradient Checking**: Finite difference validation for all operations

#### Optimizers
- **SGD**: With optional momentum
- **Adam**: Adaptive moment estimation with bias correction

### Testing

All implementations follow **EXTREME TDD** methodology:

```bash
# Run all tests
cargo test

# Run specific test module
cargo test autograd

# Run with output
cargo test -- --nocapture

# Check code quality
cargo clippy -- -D warnings
cargo fmt --check
```

**Test Coverage:**
- Unit tests for all operations
- Property-based tests (1000+ cases per operation using proptest)
- Gradient checking via finite difference (epsilon=1e-3, tolerance=0.1)
- Optimizer convergence tests

## Usage

### Basic Autograd

```rust
use entrenar::autograd::*;

// Create tensors
let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);  // requires_grad=true
let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], true);

// Forward pass
let c = add(&a, &b);
let d = relu(&c);
let mut loss = sum(&d);

// Backward pass
backward(&mut loss, None);

// Access gradients
let grad_a = a.grad().unwrap();
let grad_b = b.grad().unwrap();
```

### Using Optimizers

```rust
use entrenar::autograd::*;
use entrenar::optim::*;

// Create parameters
let mut params = vec![
    Tensor::from_vec(vec![0.5, -0.3], true),
];

// Create optimizer
let mut optimizer = Adam::default_params(0.01);

for epoch in 0..100 {
    // Compute gradients (your forward pass here)
    // ...

    // Update parameters
    optimizer.step(&mut params);
    optimizer.zero_grad(&mut params);
}
```

## Architecture

```
src/
├── autograd/         ✅ Tape-based automatic differentiation
│   ├── tensor.rs     ✅ Tensor with gradient tracking
│   ├── ops.rs        ✅ Forward/backward operations
│   ├── backward.rs   ✅ BackwardOp trait
│   ├── context.rs    ✅ Execution context
│   └── tests.rs      ✅ Comprehensive test suite
├── optim/            ✅ Optimizers
│   ├── optimizer.rs  ✅ Optimizer trait
│   ├── sgd.rs        ✅ SGD with momentum
│   └── adam.rs       ✅ Adam optimizer
├── lora/             🚧 Placeholder (Phase 3)
├── quant/            🚧 Placeholder (Phase 4)
├── merge/            🚧 Placeholder (Phase 5)
├── distill/          🚧 Placeholder (Phase 7)
├── config/           🚧 Placeholder (Phase 6)
└── train/            🚧 Placeholder
```

## Roadmap

See `docs/specifications/entrenar-spec.md` for complete specification.

### ✅ Phase 1: Autograd Engine (COMPLETED)
- Tape-based context with gradient tracking
- Core backward operations with gradient validation
- Property-based tests (1000+ iterations)

### 🚧 Phase 2: Optimizers (IN PROGRESS)
- ✅ SGD with momentum
- ✅ Adam
- ⏳ AdamW (decoupled weight decay)
- ⏳ Learning rate schedulers
- ⏳ Gradient clipping

### ⏳ Phase 3: LoRA (144h estimated)
- Low-rank adaptation layers
- QLoRA (4-bit base weights)
- Adapter save/load

### ⏳ Phase 4: Quantization (136h estimated)
- QAT (Quantization-Aware Training)
- PTQ (Post-Training Quantization)
- GGUF export (Q4_0/Q8_0)

### ⏳ Phase 5: Model Merging (96h estimated)
- TIES (Task Inference via Elimination and Sign)
- DARE (Drop And REscale)
- SLERP (Spherical Linear Interpolation)

### ⏳ Phase 6: Declarative Config (64h estimated)
- YAML configuration schema
- Auto-feature type inference
- Single-command training

### ⏳ Phase 7: Distillation (64h estimated)
- Knowledge distillation loss
- Multi-teacher ensemble
- Progressive distillation

## Development

### Quality Gates (Tiered Workflow)

```bash
# Tier 1 (Fast <5s) - Before every commit
make tier1

# Tier 2 (Integration <30s) - Before push
make tier2

# Tier 3 (Full <5m) - Before PR
make tier3

# Full CI Pipeline
make ci    # Tier 3 + coverage + mutants + PMAT + security
```

### Standard Commands

```bash
# Build
make build              # Debug
make release            # Release

# Testing
make test               # Fast tests
make coverage           # Coverage report (>90% required)
make mutants            # Mutation testing (>80% kill rate)

# Code Quality
make lint               # Clippy
make format             # Format code
make deny-check         # Dependency security

# Clean
make clean

# View all commands
make help
```

### Ticket-Based Development

All work is tracked via tickets (ENT-001 through ENT-040) in `roadmap.yaml`. See CLAUDE.md for workflow details.

```bash
make roadmap-status     # View progress
```

## Quality Metrics

Current status:
- ✅ All tests passing (18 tests)
- ✅ Clippy clean (0 warnings)
- ✅ Property-based testing (1000+ cases per operation)
- ✅ Gradient validation (finite difference checking)

## License

MIT
