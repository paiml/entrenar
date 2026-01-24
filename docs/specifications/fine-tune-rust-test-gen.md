# Specification: Rust Test Generation Fine-Tuning Pipeline

**Document ID:** SPEC-FT-001
**Version:** 1.1.0
**Status:** APPROVED
**Author:** Claude Opus 4.5
**Reviewer:** Dr. Karl Popper
**Date:** 2026-01-24

---

## 1. Executive Summary

This specification defines a production-ready fine-tuning pipeline for training Qwen2-0.5B-Coder to generate Rust unit tests and property-based tests from function implementations. The system emphasizes scientific reproducibility, falsifiability, and Toyota Production System principles.

### 1.1 Objectives

| Objective | Success Criteria |
|-----------|------------------|
| **Reproducibility** | Identical results across runs with same seed |
| **Falsifiability** | 100-point Popperian QA checklist |
| **Memory Efficiency** | Train on 8GB VRAM via QLoRA |
| **Quality** | ≥90% compile rate, ≥70% mutation score |
| **Usability** | Single command execution |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Fine-Tuning Pipeline                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  HuggingFace │───▶│  Tokenizer   │───▶│  DataLoader  │          │
│  │  Model Hub   │    │  + Corpus    │    │  (Batched)   │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                                        │                   │
│         ▼                                        ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Qwen2-0.5B  │───▶│  QLoRA       │───▶│  Trainer     │          │
│  │  (4-bit)     │    │  Adapters    │    │  (AdamW)     │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                  │                   │
│                                                  ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Checkpoint  │◀───│  Evaluation  │◀───│  Validation  │          │
│  │  Manager     │    │  Pipeline    │    │  Loop        │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                    │                                      │
│         ▼                    ▼                                      │
│  ┌──────────────┐    ┌──────────────┐                              │
│  │  Adapter     │    │  Metrics     │                              │
│  │  .safetensors│    │  Report      │                              │
│  └──────────────┘    └──────────────┘                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Pipeline

### 3.1 Corpus Options

**Option A: Use Existing Corpus**
```
https://huggingface.co/datasets/paiml/rust-cli-docs-corpus
```

**Option B: Build Test Generation Corpus**

Create a new dataset specifically for function → test pairs:

```yaml
dataset:
  name: paiml/rust-test-generation-corpus
  format: jsonl
  splits:
    train: 8000 samples
    validation: 1000 samples
    test: 1000 samples

  schema:
    - function_code: str      # Source function
    - unit_tests: str         # #[test] functions
    - property_tests: str     # proptest! macros (optional)
    - metadata:
        crate: str            # Source crate name
        complexity: int       # Cyclomatic complexity
        has_generics: bool    # Uses generics
        has_lifetimes: bool   # Uses lifetimes
```

**Data Sources for Option B:**
1. `rustdoc` test examples from top 1000 crates
2. Proptest examples from crates using it
3. Hand-curated edge case examples

### 3.2 Data Format

```jsonl
{
  "input": "/// Checks if n is prime\npub fn is_prime(n: u64) -> bool {\n    if n < 2 { return false; }\n    for i in 2..=(n as f64).sqrt() as u64 {\n        if n % i == 0 { return false; }\n    }\n    true\n}",
  "output": "#[test]\nfn test_is_prime_small_primes() {\n    assert!(is_prime(2));\n    assert!(is_prime(3));\n    assert!(is_prime(5));\n}\n\n#[test]\nfn test_is_prime_composites() {\n    assert!(!is_prime(4));\n    assert!(!is_prime(9));\n}\n\nproptest! {\n    #[test]\n    fn prop_prime_greater_than_one(p in 2u64..1000) {\n        if is_prime(p) {\n            prop_assert!(p > 1);\n        }\n    }\n}",
  "metadata": {"crate": "num-prime", "complexity": 4}
}
```

### 3.3 Prompt Template

```
<|im_start|>system
You are a Rust testing expert. Generate comprehensive unit tests and property-based tests for the given function.
<|im_end|>
<|im_start|>user
Generate tests for this function:

```rust
{function_code}
```
<|im_end|>
<|im_start|>assistant
{tests}
<|im_end|>
```

---

## 4. Model Configuration

### 4.1 Base Model

```yaml
model:
  name: Qwen/Qwen2.5-Coder-0.5B-Instruct
  source: huggingface
  quantization: 4-bit (NF4)
  dtype: bfloat16 (compute)

  architecture:
    hidden_size: 896
    num_attention_heads: 14
    num_kv_heads: 2
    intermediate_size: 4864
    num_hidden_layers: 24
    vocab_size: 151936
    max_position_embeddings: 32768
```

### 4.2 LoRA Configuration

```yaml
lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

  # Trainable parameters: ~4.2M (0.8% of 527M)
```

### 4.3 Training Hyperparameters

```yaml
training:
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  effective_batch_size: 16

  optimizer:
    name: AdamW
    learning_rate: 2e-4
    betas: [0.9, 0.999]
    weight_decay: 0.01

  scheduler:
    name: cosine
    warmup_ratio: 0.03
    min_lr_ratio: 0.1

  precision:
    compute: bfloat16
    gradients: float32

  gradient_clipping: 1.0

  checkpointing:
    strategy: best_loss
    save_steps: 500
    max_checkpoints: 3
```

---

## 5. Evaluation Pipeline

### 5.1 Automatic Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Compile Rate** | % of generated tests that compile | ≥90% |
| **Test Pass Rate** | % of compiled tests that pass | ≥85% |
| **Mutation Score** | % of mutants killed by tests | ≥70% |
| **Coverage Delta** | Branch coverage improvement | ≥+10% |
| **BLEU-4** | N-gram overlap with reference | ≥0.45 |
| **CodeBLEU** | Syntax-aware code similarity | ≥0.50 |

### 5.2 Evaluation Protocol

```rust
struct EvaluationResult {
    // Compilation metrics
    compile_success: f32,      // [0, 1]
    compile_errors: Vec<CompileError>,

    // Execution metrics
    tests_passed: f32,         // [0, 1]
    tests_failed: Vec<TestFailure>,

    // Quality metrics
    mutation_score: f32,       // [0, 1]
    branch_coverage: f32,      // [0, 1]
    edge_cases_covered: f32,   // [0, 1]

    // Similarity metrics
    bleu4: f32,
    codebleu: f32,

    // Resource metrics
    inference_latency_ms: f32,
    tokens_per_second: f32,
}
```

### 5.3 Holdout Test Set

Reserved 100 functions never seen during training:
- 25 from stdlib patterns
- 25 from async/tokio patterns
- 25 with generics/lifetimes
- 25 with error handling (Result/Option)

---

## 6. CUDA Integration

### 6.1 Device Selection

```rust
pub enum ComputeDevice {
    Cpu,
    Cuda { device_id: usize },
    Auto,  // Prefer CUDA if available
}

impl ComputeDevice {
    pub fn auto_detect() -> Self {
        if cuda_available() && cuda_memory_gb() >= 6.0 {
            ComputeDevice::Cuda { device_id: 0 }
        } else {
            ComputeDevice::Cpu
        }
    }
}
```

### 6.2 Memory Requirements

| Configuration | VRAM | RAM | Notes |
|---------------|------|-----|-------|
| QLoRA (4-bit) | 6 GB | 8 GB | Recommended |
| LoRA (fp16) | 12 GB | 16 GB | Higher quality |
| Full fine-tune | 24 GB | 32 GB | Not recommended |

### 6.3 Mixed Precision

```yaml
precision:
  base_weights: nf4          # 4-bit NormalFloat
  lora_weights: bfloat16     # Brain floating point
  activations: bfloat16      # Forward pass
  gradients: float32         # Backward pass (stability)
  optimizer_states: float32  # Adam moments
```

---

## 7. Reproducibility Protocol

### 7.1 Random Seed Management

```rust
pub struct ReproducibilityConfig {
    seed: u64,
    deterministic_algorithms: bool,
    cudnn_benchmark: bool,
    cudnn_deterministic: bool,
}

impl Default for ReproducibilityConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            deterministic_algorithms: true,
            cudnn_benchmark: false,
            cudnn_deterministic: true,
        }
    }
}
```

### 7.2 Environment Lockfile

```yaml
# reproducibility.lock
rust_version: "1.75.0"
cuda_version: "12.1"
cudnn_version: "8.9.0"
torch_version: "2.1.0"

dependencies:
  entrenar: "0.5.6"
  trueno: "0.14.0"
  tokenizers: "0.15.0"
  safetensors: "0.4.0"

hardware:
  gpu: "NVIDIA RTX 4090"
  driver: "545.23.08"

checksum:
  model: "sha256:abc123..."
  dataset: "sha256:def456..."
  adapters: "sha256:ghi789..."
```

### 7.3 Experiment Tracking

```yaml
experiment:
  id: "ft-testgen-001"
  timestamp: "2025-01-24T12:00:00Z"
  git_commit: "ef30bd8"

  config_hash: "sha256:..."

  metrics:
    final_loss: 0.412
    best_compile_rate: 0.94
    training_time_hours: 2.5
```

---

## 8. Artifact Management

### 8.1 .gitignore Additions

```gitignore
# Fine-tuning artifacts
checkpoints/
adapters/
*.safetensors
*.gguf
*.bin

# HuggingFace cache
.cache/huggingface/

# Training outputs
runs/
logs/
tensorboard/

# Evaluation artifacts
eval_results/
generated_tests/

# Large model files
models/
*.ckpt
```

### 8.2 Directory Structure

```
experiments/
├── ft-testgen-001/
│   ├── config.yaml           # Full configuration
│   ├── reproducibility.lock  # Environment lockfile
│   ├── training.log          # Training logs
│   ├── metrics.json          # Final metrics
│   ├── checkpoints/
│   │   ├── step-500/
│   │   ├── step-1000/
│   │   └── best/
│   │       ├── adapter_model.safetensors
│   │       └── adapter_config.json
│   └── evaluation/
│       ├── compile_results.json
│       ├── test_results.json
│       └── mutation_report.html
```

---

## 9. Popperian Falsification QA (100 Points)

Following Karl Popper's philosophy of science, we define falsifiable hypotheses and tests to validate them. Each test can **disprove** a claim, ensuring scientific rigor.

### 9.1 Scoring Rubric

| Category | Points | Description |
|----------|--------|-------------|
| **Reproducibility** | 20 | Identical results across runs |
| **Compilation** | 20 | Generated code compiles |
| **Correctness** | 20 | Tests catch actual bugs |
| **Coverage** | 15 | Tests exercise code paths |
| **Efficiency** | 10 | Resource usage bounds |
| **Edge Cases** | 10 | Handles boundary conditions |
| **Documentation** | 5 | Output is self-documenting |

### 9.2 Falsifiable Hypotheses

```rust
/// 100-Point Popperian Falsification Checklist
pub struct PopperianQA {
    /// REPRODUCIBILITY (20 points)
    /// H1: Training is deterministic with fixed seed
    r1_same_loss_curve: bool,           // 5 pts - Loss identical across runs
    r2_same_final_weights: bool,        // 5 pts - Adapter weights match exactly
    r3_same_eval_metrics: bool,         // 5 pts - Eval metrics identical
    r4_environment_locked: bool,        // 5 pts - All deps version-locked

    /// COMPILATION (20 points)
    /// H2: Generated tests are syntactically valid Rust
    c1_parses_as_rust: bool,            // 5 pts - rustfmt succeeds
    c2_type_checks: bool,               // 5 pts - cargo check succeeds
    c3_no_unused_warnings: bool,        // 5 pts - No clippy warnings
    c4_links_correctly: bool,           // 5 pts - Links against target crate

    /// CORRECTNESS (20 points)
    /// H3: Generated tests are semantically meaningful
    x1_tests_pass_on_correct: bool,     // 5 pts - Pass on original impl
    x2_tests_fail_on_mutant: bool,      // 5 pts - Fail on mutated impl
    x3_assertions_meaningful: bool,     // 5 pts - Not just `assert!(true)`
    x4_no_tautologies: bool,            // 5 pts - No `assert_eq!(x, x)`

    /// COVERAGE (15 points)
    /// H4: Tests exercise meaningful code paths
    v1_branch_coverage_delta: bool,     // 5 pts - ≥+5% branch coverage
    v2_line_coverage_delta: bool,       // 5 pts - ≥+10% line coverage
    v3_edge_cases_present: bool,        // 5 pts - Tests empty/null/max cases

    /// EFFICIENCY (10 points)
    /// H5: Training completes within resource bounds
    e1_vram_under_8gb: bool,            // 3 pts - Peak VRAM < 8GB
    e2_training_under_4hrs: bool,       // 4 pts - Completes in <4 hours
    e3_inference_under_1s: bool,        // 3 pts - Generation < 1s/function

    /// EDGE CASES (10 points)
    /// H6: Handles difficult inputs gracefully
    g1_handles_generics: bool,          // 2 pts - Generic functions
    g2_handles_lifetimes: bool,         // 2 pts - Lifetime annotations
    g3_handles_async: bool,             // 2 pts - Async functions
    g4_handles_unsafe: bool,            // 2 pts - Unsafe blocks
    g5_handles_macros: bool,            // 2 pts - Macro-heavy code

    /// DOCUMENTATION (5 points)
    /// H7: Output is self-explanatory
    d1_test_names_descriptive: bool,    // 2 pts - test_* names explain intent
    d2_comments_present: bool,          // 2 pts - Explains edge cases
    d3_proptest_strategies_clear: bool, // 1 pt  - Strategy names meaningful
}

impl PopperianQA {
    pub fn score(&self) -> u8 {
        let mut score = 0u8;

        // Reproducibility (20)
        if self.r1_same_loss_curve { score += 5; }
        if self.r2_same_final_weights { score += 5; }
        if self.r3_same_eval_metrics { score += 5; }
        if self.r4_environment_locked { score += 5; }

        // Compilation (20)
        if self.c1_parses_as_rust { score += 5; }
        if self.c2_type_checks { score += 5; }
        if self.c3_no_unused_warnings { score += 5; }
        if self.c4_links_correctly { score += 5; }

        // Correctness (20)
        if self.x1_tests_pass_on_correct { score += 5; }
        if self.x2_tests_fail_on_mutant { score += 5; }
        if self.x3_assertions_meaningful { score += 5; }
        if self.x4_no_tautologies { score += 5; }

        // Coverage (15)
        if self.v1_branch_coverage_delta { score += 5; }
        if self.v2_line_coverage_delta { score += 5; }
        if self.v3_edge_cases_present { score += 5; }

        // Efficiency (10)
        if self.e1_vram_under_8gb { score += 3; }
        if self.e2_training_under_4hrs { score += 4; }
        if self.e3_inference_under_1s { score += 3; }

        // Edge Cases (10)
        if self.g1_handles_generics { score += 2; }
        if self.g2_handles_lifetimes { score += 2; }
        if self.g3_handles_async { score += 2; }
        if self.g4_handles_unsafe { score += 2; }
        if self.g5_handles_macros { score += 2; }

        // Documentation (5)
        if self.d1_test_names_descriptive { score += 2; }
        if self.d2_comments_present { score += 2; }
        if self.d3_proptest_strategies_clear { score += 1; }

        score
    }

    pub fn grade(&self) -> &'static str {
        match self.score() {
            95..=100 => "A+ (Excellent)",
            90..=94  => "A  (Very Good)",
            85..=89  => "B+ (Good)",
            80..=84  => "B  (Satisfactory)",
            70..=79  => "C  (Needs Improvement)",
            _        => "F  (Failing)",
        }
    }
}
```

### 9.3 Automated Falsification Tests

```rust
#[cfg(test)]
mod popperian_tests {
    /// FALSIFIABLE: If training is reproducible, two runs with same seed
    /// must produce identical loss curves (±1e-6)
    #[test]
    fn falsify_reproducibility() {
        let run1 = train_with_seed(42);
        let run2 = train_with_seed(42);

        for (l1, l2) in run1.losses.iter().zip(run2.losses.iter()) {
            assert!((l1 - l2).abs() < 1e-6,
                "FALSIFIED: Runs differ at step - not reproducible");
        }
    }

    /// FALSIFIABLE: If tests are meaningful, they must fail on mutated code
    #[test]
    fn falsify_mutation_detection() {
        let original_fn = "pub fn add(a: i32, b: i32) -> i32 { a + b }";
        let mutant_fn   = "pub fn add(a: i32, b: i32) -> i32 { a - b }";  // Bug!

        let tests = generate_tests(original_fn);

        assert!(tests.pass_on(original_fn), "Tests should pass on original");
        assert!(!tests.pass_on(mutant_fn),
            "FALSIFIED: Tests pass on mutant - not detecting bugs");
    }

    /// FALSIFIABLE: Generated code must be valid Rust
    #[test]
    fn falsify_syntax_validity() {
        for sample in test_set.iter() {
            let tests = generate_tests(&sample.function);
            let parse_result = syn::parse_file(&tests);

            assert!(parse_result.is_ok(),
                "FALSIFIED: Generated invalid Rust syntax");
        }
    }
}
```

---

## 10. Academic Citations

### 10.1 Core Methodology

| Citation | Relevance |
|----------|-----------|
| **Hu et al. (2021)** "LoRA: Low-Rank Adaptation of Large Language Models" [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) | Foundation for parameter-efficient fine-tuning |
| **Dettmers et al. (2023)** "QLoRA: Efficient Finetuning of Quantized LLMs" [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) | 4-bit quantization with LoRA |
| **Rozière et al. (2023)** "Code Llama: Open Foundation Models for Code" [arXiv:2308.12950](https://arxiv.org/abs/2308.12950) | Code model fine-tuning practices |

### 10.2 Test Generation

| Citation | Relevance |
|----------|-----------|
| **Tufano et al. (2020)** "Unit Test Case Generation with Transformers" [arXiv:2009.05617](https://arxiv.org/abs/2009.05617) | Transformer-based test generation |
| **Alagarsamy et al. (2023)** "A3Test: Assertion-Augmented Automated Test Generation" [arXiv:2302.10352](https://arxiv.org/abs/2302.10352) | Assertion quality in generated tests |
| **Lemieux et al. (2023)** "CodaMosa: Escaping Coverage Plateaus in Test Generation" [ICSE 2023](https://dl.acm.org/doi/10.1109/ICSE48619.2023.00085) | Coverage-guided generation |

### 10.3 Evaluation Methodology

| Citation | Relevance |
|----------|-----------|
| **Jia & Harman (2011)** "An Analysis and Survey of the Development of Mutation Testing" [IEEE TSE](https://ieeexplore.ieee.org/document/5487526) | Mutation testing as oracle |
| **Ren et al. (2020)** "CodeBLEU: a Method for Automatic Evaluation of Code Synthesis" [arXiv:2009.10297](https://arxiv.org/abs/2009.10297) | Code-aware similarity metrics |

### 10.4 Toyota Production System

| Citation | Relevance |
|----------|-----------|
| **Liker (2004)** "The Toyota Way: 14 Management Principles" ISBN: 0071392319 | Jidoka, Kaizen, Poka-yoke |
| **Ohno (1988)** "Toyota Production System: Beyond Large-Scale Production" ISBN: 0915299143 | Continuous improvement |

### 10.5 Philosophy of Science

| Citation | Relevance |
|----------|-----------|
| **Popper (1959)** "The Logic of Scientific Discovery" ISBN: 0415278449 | Falsifiability criterion |
| **Popper (1963)** "Conjectures and Refutations" ISBN: 0415285941 | Hypothesis testing methodology |

---

## 11. Implementation Plan

### 11.1 Phase 1: Infrastructure (Week 1)

- [ ] Add HuggingFace Hub integration
- [ ] Implement CUDA device detection
- [ ] Setup checkpoint management
- [ ] Update .gitignore

### 11.2 Phase 2: Training Pipeline (Week 2)

- [ ] Implement data loading from HF datasets
- [ ] Add QLoRA training loop with mixed precision
- [ ] Implement validation metrics
- [ ] Add TensorBoard logging

### 11.3 Phase 3: Evaluation (Week 3)

- [ ] Compile rate evaluation
- [ ] Mutation testing integration
- [ ] Coverage delta measurement
- [ ] Popperian QA automation

### 11.4 Phase 4: Documentation (Week 4)

- [ ] Update book documentation
- [ ] Record training run with metrics
- [ ] Publish trained adapters to HF Hub
- [ ] Write reproducibility guide

---

## 12. CLI Interface

```bash
# Full training pipeline
cargo run --example finetune_test_gen -- \
    --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --dataset paiml/rust-test-generation-corpus \
    --output ./experiments/ft-testgen-001 \
    --epochs 3 \
    --batch-size 4 \
    --lora-rank 16 \
    --seed 42 \
    --device auto

# Evaluation only
cargo run --example finetune_test_gen -- \
    --eval-only \
    --adapter ./experiments/ft-testgen-001/checkpoints/best \
    --test-set ./eval/holdout.jsonl

# Generate tests for a single file
cargo run --example finetune_test_gen -- \
    --generate \
    --adapter ./experiments/ft-testgen-001/checkpoints/best \
    --input ./src/my_function.rs \
    --output ./tests/generated_tests.rs
```

---

## 13. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CUDA OOM | Medium | High | Gradient checkpointing, reduce batch size |
| Poor generation quality | Medium | High | More training data, larger LoRA rank |
| Non-reproducible results | Low | Critical | Lock all dependencies, fix seeds |
| HF Hub rate limiting | Low | Medium | Local model cache, retry logic |
| Compile errors in output | High | Medium | Syntax validation, iterative refinement |

---

## 14. Success Metrics

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| Compile Rate | 85% | 92% | 97% |
| Test Pass Rate | 80% | 88% | 95% |
| Mutation Score | 60% | 72% | 80% |
| Coverage Delta | +5% | +12% | +18% |
| Popperian Score | 80/100 | 90/100 | 95/100 |
| Training Time | <6h | <3h | <1.5h |
| VRAM Usage | <10GB | <7GB | <5GB |

---

## 15. Resolved Decisions (Popperian Review)

1. **Corpus Strategy**: **Option B (Create `paiml/rust-test-generation-corpus`)**
   *Rationale:* To rigorously test the hypothesis "The model understands Rust testing idioms," we must control independent variables. A specialized corpus with metadata allows for precise hypothesis testing (e.g., "fails when complexity > 10").

2. **Model Size**: **Start with 0.5B**
   *Rationale:* We prefer the simplest theory that has not been falsified. If 0.5B meets the correctness targets, it is scientifically superior. 1.5B is only justified if 0.5B is falsified.

3. **Proptest Ratio**: **20-25% Proptest Data**
   *Rationale:* Property-based tests are "engines of falsification." The model needs enough structural training (unit tests) but must learn to generate these high-value falsifiers.

4. **Mutation Testing**: **Stratified Sampling (Rigorous)**
   *Rationale:* Full verification is ideal but expensive. We will use stratified sampling based on code complexity to ensure the "Correctness" score is not an illusion based on trivial mutants.

5. **Adapter Publishing**: **Mandatory**
   *Rationale:* Secrecy is the enemy of truth. Adapters must be published to the HuggingFace Hub to allow the community to attempt to falsify the results.

---

## Appendix A: Popper's Falsifiability Criterion

> "A theory is scientific if and only if it is falsifiable."
> — Karl Popper, *The Logic of Scientific Discovery* (1959)

We apply this to ML evaluation: every claim about model capability must have a test that could **disprove** it. If a test cannot fail, it provides no information.

**Example Application:**

| Claim | Falsifiable Test | How It Can Fail |
|-------|------------------|-----------------|
| "Model generates valid Rust" | Parse with `syn` | `syn::parse_file()` returns `Err` |
| "Tests detect bugs" | Run on mutants | Tests pass on buggy code |
| "Training is reproducible" | Compare two runs | Losses diverge beyond ε |

---

## 16. PMAT (Popperian Metric Analysis Tool) Integration

To ensure the 100-point QA checklist is enforced with scientific rigor, the pipeline must integrate with the project's PMAT system.

### 16.1 Automated Gates

| Gate ID | Logic | Action on Failure |
|---------|-------|-------------------|
| **P-GATE-001** | `PopperianScore < 90` | Reject adapter; forbid deployment |
| **P-GATE-002** | `CompileRate < 0.90` | Falsify "Syntactic Understanding" hypothesis |
| **P-GATE-003** | `MutationScore < 0.70` | Falsify "Semantic Utility" hypothesis |
| **P-GATE-004** | `VRAM > 8GB` | Falsify "Efficiency" hypothesis |

### 16.2 Continuous Falsification

The evaluation pipeline shall run on every commit to the `adapters` repository. Any degradation in the Popperian Score constitutes a "Regression of Knowledge" and requires immediate *Hansei* (reflection).

---

## Appendix C: PMAT ComputeBrick Integration

Following the "Brick Architecture" (v2.0), the evaluation results must be exportable as a standardized **ComputeBrick Metric Set**.

1. **Metrics Format**: JSON-LD with semantic pointers to the source functions.
2. **Observability**: Integrate with `cbtop` for real-time visualization of the "falsification rate" during batch evaluation.
3. **Traceability**: Every generated test must be linked to the specific model version and adapter weights via SHA-256 hashes.

---

## Appendix D: Toyota Way Principles Applied

| Principle | Application to Fine-Tuning |
|-----------|---------------------------|
| **Jidoka** (自働化) | Stop training on quality failures |
| **Kaizen** (改善) | Iterative hyperparameter improvement |
| **Genchi Genbutsu** (現地現物) | Inspect actual generated tests, not just metrics |
| **Poka-yoke** (ポカヨケ) | Type-safe configs prevent misconfiguration |
| **Andon** (アンドン) | Real-time training dashboard alerts |
| **Hansei** (反省) | Post-training reflection on what could improve |

---

**END OF SPECIFICATION**

*Approved for implementation.*
