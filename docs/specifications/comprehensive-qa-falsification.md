# Specification: Comprehensive QA & Falsification Protocol

**Document ID:** SPEC-QA-001
**Version:** 1.2.0
**Status:** ACTIVE
**Author:** Claude Opus 4.5
**Reviewer:** Dr. Karl Popper
**Date:** 2026-01-24

---

## 1. Executive Summary

This document defines the **Comprehensive QA Falsification Protocol** for the `entrenar` CUDA-first fine-tuning pipeline. It serves as the final gatekeeper before release, ensuring that the 100% ticket completion translates to 100% scientific validity.

**V1.2.0 Update:** Integrated **Probar Compliance Standards** and tightened falsification criteria. We now demand high-fidelity visualization and resilience to "Chaos" events.

---

## 2. The 100-Point Falsification Matrix

The system is graded on a strict 100-point scale. **Any score < 95 is a failure.** (Threshold raised from 90).

### 2.1 Category A: Infrastructure & Stability (15 Points)

| ID | Test / Claim | Falsification Method | Points |
|----|--------------|----------------------|--------|
| A1 | **CUDA Detection** | Run on CPU-only node; must fail gracefully or fallback | 5 |
| A2 | **Memory Safety** | Run under `cuda-memcheck`; 0 errors allowed | 5 |
| A3 | **Numerical Stability** | Train for 100 epochs; Gradient Norm must never be `NaN`/`Inf` | 5 |

### 2.2 Category B: Kernel Correctness (20 Points)

| ID | Test / Claim | Falsification Method | Points |
|----|--------------|----------------------|--------|
| B1 | **MatMul Parity** | Compare `CudaTensor` output vs `ndarray` (CPU) output (tol < 1e-5) | 4 |
| B2 | **Attention Parity** | Compare `FlashAttention` vs Standard Decomposition vs CPU | 4 |
| B3 | **RMSNorm Parity** | Compare `RmsNormKernel` vs Python/PyTorch reference | 4 |
| B4 | **Activation Parity** | Verify `ReLU`/`GELU`/`SiLU` forward & backward gradients | 4 |
| B5 | **Fused Optimizer** | Verify `AdamW` update matches manual calculation exactly | 4 |

### 2.3 Category C: Performance & Efficiency (20 Points)

*Thresholds tightened to force optimization.*

| ID | Test / Claim | Falsification Method | Points |
|----|--------------|----------------------|--------|
| C1 | **GPU Utilization** | `nvidia-smi` must show **>80%** (was 70%) during `finetune_real` | 5 |
| C2 | **Throughput** | Token generation must exceed **120 tokens/second** (was 100) | 5 |
| C3 | **Memory Footprint** | Peak VRAM < 8GB for 0.5B model (QLoRA) | 5 |
| C4 | **Kernel Overhead** | Kernel launch latency < 5% of total step time | 5 |

### 2.4 Category D: Learning Dynamics (20 Points)

| ID | Test / Claim | Falsification Method | Points |
|----|--------------|----------------------|--------|
| D1 | **Loss Reduction** | Loss must decrease monotonically (smoothed) over 15 epochs | 5 |
| D2 | **LoRA Efficacy** | LoRA (rank=16) must match Full FT loss within **5% margin** (was 10%) | 5 |
| D3 | **Gradient Flow** | Gradients for LoRA A/B matrices must be non-zero | 5 |
| D4 | **Overfitting Test** | Model must achieve near-zero loss on single-batch overfit | 5 |

### 2.5 Category E: Observability & Probar Compliance (25 Points)

*Expanded to enforce Visual Fidelity and Chaos Resilience.*

| ID | Test / Claim | Falsification Method | Points |
|----|--------------|----------------------|--------|
| E1 | **Boundary Enforcement** | Displaying `Epoch > Total` or `Step > Total` constitutes failure | 5 |
| E2 | **Numerical Sanitization** | Any raw `NaN`/`Inf` in UI falsifies the view | 5 |
| E3 | **Visual Fidelity** | **Pixel Coverage > 50%** AND **Unicode Richness > 15%** (Probar Std) | 5 |
| E4 | **Render Latency** | TUI Render time must be **< 1.0ms** (p99) | 5 |
| E5 | **Chaos Resilience** | UI must not crash if Backend dies or sends corrupt JSON | 5 |

---

## 3. Execution Plan

### 3.1 Automated Test Suite
Run the standard test suite to clear Category B (Kernels):
```bash
cargo test --features cuda --release
```

### 3.2 Performance Benchmark
Run the benchmark to clear Category C (Performance):
```bash
# Verify GPU isolation
nvidia-smi --query-compute-apps=pid,name --format=csv

# Run the benchmark:
cargo run --release --example cuda_training_benchmark --features cuda
```

### 3.3 Probar Compliance Run (Integration)
Run the full pipeline with monitoring to clear Categories A, D, and E:
```bash
# Terminal 1: Training (Producer)
cargo run --release --example finetune_real --features cuda -- \
    --output ./experiments/probar-test

# Terminal 2: Probar Monitor (Consumer)
cargo run --release --example finetune_real --features cuda -- \
    --monitor --experiment ./experiments/probar-test
```

---

## 4. Failure Protocol (Hansei)

If any check fails:
1.  **Stop:** The build is broken.
2.  **Audit:** Run `probar audit` (if available) or inspect `training_state.json`.
3.  **Visual Debug:** For Category E failures, take a screenshot and check character distribution.
4.  **Refactor:** Fix the underlying logic. Sanitization must happen at the *Source* (State generation), not just the *Sink* (Rendering).
5.  **Restart:** The 100-point check starts from 0.

---

**END OF SPECIFICATION**