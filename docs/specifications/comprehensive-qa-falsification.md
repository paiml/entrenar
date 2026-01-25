# Specification: Comprehensive QA & Falsification Protocol

**Document ID:** SPEC-QA-001
**Version:** 1.0.0
**Status:** ACTIVE
**Author:** Claude Opus 4.5
**Reviewer:** Dr. Karl Popper
**Date:** 2026-01-24

---

## 1. Executive Summary

This document defines the **Comprehensive QA Falsification Protocol** for the `entrenar` CUDA-first fine-tuning pipeline. It serves as the final gatekeeper before release, ensuring that the 100% ticket completion translates to 100% scientific validity.

The protocol is built on **Popperian principles**: we do not seek to prove the system works; we seek to prove it fails. Only if it survives this rigorous battery of tests do we accept it.

---

## 2. The 100-Point Falsification Matrix

The system is graded on a strict 100-point scale. **Any score < 90 is a failure.**

### 2.1 Category A: Infrastructure & Stability (20 Points)

| ID | Test / Claim | Falsification Method | Points |
|----|--------------|----------------------|--------|
| A1 | **CUDA Detection** | Run on CPU-only node; must fail gracefully or fallback | 5 |
| A2 | **Memory Safety** | Run under `cuda-memcheck`; 0 errors allowed | 5 |
| A3 | **Numerical Stability** | Train for 100 epochs; Gradient Norm must never be `NaN`/`Inf` | 5 |
| A4 | **Process Isolation** | Kill TUI process; Training must continue uninterrupted | 5 |

### 2.2 Category B: Kernel Correctness (30 Points)

| ID | Test / Claim | Falsification Method | Points |
|----|--------------|----------------------|--------|
| B1 | **MatMul Parity** | Compare `CudaTensor` output vs `ndarray` (CPU) output (tol < 1e-5) | 5 |
| B2 | **Attention Parity** | Compare `FlashAttention` vs Standard Decomposition vs CPU | 5 |
| B3 | **RMSNorm Parity** | Compare `RmsNormKernel` vs Python/PyTorch reference | 5 |
| B4 | **Activation Parity** | Verify `ReLU`/`GELU`/`SiLU` forward & backward gradients | 5 |
| B5 | **Fused SwiGLU** | Verify `SwiGLU` output matches component-wise execution | 5 |
| B6 | **Optimizer Step** | Verify `AdamW` update matches manual calculation | 5 |

### 2.3 Category C: Performance & Efficiency (25 Points)

| ID | Test / Claim | Falsification Method | Points |
|----|--------------|----------------------|--------|
| C1 | **GPU Utilization** | `nvidia-smi` must show >70% during `finetune_real` | 5 |
| C2 | **Throughput** | Token generation must exceed 100 tokens/second | 5 |
| C3 | **Memory Footprint** | Peak VRAM < 8GB for 0.5B model (QLoRA) | 5 |
| C4 | **Epoch Duration** | 15 Epochs must complete in < 30 seconds (0.5B model) | 5 |
| C5 | **Kernel Overhead** | Kernel launch latency < 10% of total step time | 5 |

### 2.4 Category D: Learning Dynamics (25 Points)

| ID | Test / Claim | Falsification Method | Points |
|----|--------------|----------------------|--------|
| D1 | **Loss Reduction** | Loss must decrease monotonically (smoothed) over 15 epochs | 5 |
| D2 | **LoRA Efficacy** | LoRA (rank=16) must match Full FT loss within 10% margin | 5 |
| D3 | **Gradient Flow** | Gradients for LoRA A/B matrices must be non-zero | 5 |
| D4 | **Overfitting Test** | Model must achieve near-zero loss on single-batch overfit | 5 |
| D5 | **Generalization** | Validation loss must track training loss (no immediate divergence) | 5 |

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
# PREREQUISITE: Fresh terminal session (no prior CUDA failures)
# The CUDA driver API maintains global state that can be corrupted by failures.
# If you encounter CUDA_ERROR_UNKNOWN (700) or CUDA_ERROR_NOT_INITIALIZED (3),
# start a new terminal session.

# Verify GPU is available and no other CUDA processes running:
nvidia-smi --query-compute-apps=pid,name --format=csv  # Should show no processes

# Run the benchmark:
cargo run --release --example cuda_training_benchmark --features cuda
```

**Environmental Requirements:**
- Fresh terminal session (trueno-gpu uses static CUDA_INITIALIZED flag)
- No concurrent CUDA processes (`nvidia-smi --query-compute-apps`)
- GPU not in error state from prior failures

**Common Errors:**
- `CUDA_ERROR_UNKNOWN (700)`: Illegal memory access or corrupted context - restart terminal
- `CUDA_ERROR_NOT_INITIALIZED (3)`: CUDA driver state corrupted - restart terminal

### 3.3 Integration Test (The "Real" Run)
Run the full pipeline to clear Category A and D:
```bash
cargo run --release --example finetune_real --features cuda -- \
    --monitor --epochs 15
```

---

## 4. Failure Protocol (Hansei)

If any check fails:
1.  **Stop:** Do not proceed to release.
2.  **Isolate:** Identify the specific falsified hypothesis (e.g., "RMSNorm is stable").
3.  **Root Cause:** Use `cuda-gdb` or `nsight` to inspect the kernel state.
4.  **Refactor:** Fix the kernel or logic.
5.  **Restart:** The 100-point check starts from 0.

---

**END OF SPECIFICATION**
