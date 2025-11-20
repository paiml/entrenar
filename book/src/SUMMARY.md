# Entrenar - Training & Optimization Library

[Introduction](./introduction.md)

# Getting Started

- [Installation](./getting-started/installation.md)
- [Quick Start](./getting-started/quick-start.md)
- [First Training Loop](./getting-started/first-training-loop.md)
- [Core Concepts](./getting-started/core-concepts.md)

# Architecture

- [Overview](./architecture/overview.md)
- [Design Philosophy](./architecture/design-philosophy.md)
- [Module Organization](./architecture/module-organization.md)
- [Type System](./architecture/type-system.md)
- [Memory Management](./architecture/memory-management.md)

# Autograd Engine

- [What is Automatic Differentiation?](./autograd/what-is-autograd.md)
- [Tape-Based Computation Graphs](./autograd/tape-based-graphs.md)
- [Tensor Operations](./autograd/tensor-operations.md)
  - [Matrix Multiplication](./autograd/matmul.md)
  - [Activations (ReLU, GELU, Swish)](./autograd/activations.md)
  - [Layer Normalization](./autograd/layer-norm.md)
  - [Attention Mechanism](./autograd/attention.md)
- [Backward Pass](./autograd/backward-pass.md)
- [Gradient Computation](./autograd/gradient-computation.md)
- [Finite Difference Validation](./autograd/finite-difference.md)

# Optimizers

- [Overview](./optimizers/overview.md)
- [Stochastic Gradient Descent (SGD)](./optimizers/sgd.md)
- [Adam Optimizer](./optimizers/adam.md)
- [AdamW (Decoupled Weight Decay)](./optimizers/adamw.md)
- [Learning Rate Schedulers](./optimizers/schedulers.md)
  - [Cosine Annealing](./optimizers/cosine-scheduler.md)
  - [Step Decay](./optimizers/step-decay.md)
  - [Exponential Decay](./optimizers/exponential-decay.md)
- [Gradient Clipping](./optimizers/gradient-clipping.md)
- [SIMD-Accelerated Updates](./optimizers/simd-acceleration.md)
- [Optimizer Theory](./optimizers/theory.md)

# LoRA (Low-Rank Adaptation)

- [What is LoRA?](./lora/what-is-lora.md)
- [Parameter-Efficient Fine-Tuning](./lora/parameter-efficient-finetuning.md)
- [LoRA Layer Architecture](./lora/layer-architecture.md)
  - [Low-Rank Matrices A and B](./lora/low-rank-matrices.md)
  - [Scaling Factor (alpha/rank)](./lora/scaling-factor.md)
  - [Merge and Unmerge](./lora/merge-unmerge.md)
- [Target Module Selection](./lora/target-modules.md)
- [Gradient Flow Isolation](./lora/gradient-flow.md)
- [Adapter Persistence](./lora/adapter-persistence.md)
  - [Saving Adapters](./lora/saving-adapters.md)
  - [Loading Adapters](./lora/loading-adapters.md)
  - [Sharing Adapters](./lora/sharing-adapters.md)

# QLoRA (Quantized LoRA)

- [Memory-Efficient Fine-Tuning](./quantization/qlora-overview.md)
- [4-bit Quantization](./quantization/4bit-quantization.md)
  - [Block-Wise Quantization](./quantization/block-wise-quantization.md)
  - [Scale Factors](./quantization/scale-factors.md)
  - [Quantization/Dequantization](./quantization/quant-dequant.md)
- [QLoRA Layer](./quantization/qlora-layer.md)
- [On-the-Fly Dequantization](./quantization/on-the-fly-dequant.md)
- [Memory Benchmarks](./quantization/memory-benchmarks.md)
  - [LoRA vs QLoRA Comparison](./quantization/lora-vs-qlora.md)
  - [Transformer Model Benchmarks](./quantization/transformer-benchmarks.md)
  - [Compression Ratios](./quantization/compression-ratios.md)
- [Trade-offs and Best Practices](./quantization/tradeoffs.md)

# Training Loops

- [Basic Training Loop](./training/basic-loop.md)
- [Batching and Data Loading](./training/batching.md)
- [Loss Functions](./training/loss-functions.md)
- [Validation and Testing](./training/validation.md)
- [Checkpointing](./training/checkpointing.md)
- [Early Stopping](./training/early-stopping.md)
- [Mixed Precision Training](./training/mixed-precision.md)

# API Reference

- [Tensor API](./api-reference/tensor-api.md)
- [Autograd Operations](./api-reference/autograd-ops.md)
- [Optimizer API](./api-reference/optimizer-api.md)
- [LoRA API](./api-reference/lora-api.md)
- [QLoRA API](./api-reference/qlora-api.md)
- [Configuration System](./api-reference/configuration.md)
- [Error Handling](./api-reference/error-handling.md)

# Examples

- [Linear Regression with Autograd](./examples/linear-regression.md)
- [Training a Simple MLP](./examples/simple-mlp.md)
- [Fine-Tuning with LoRA](./examples/lora-finetuning.md)
- [Memory-Efficient QLoRA](./examples/qlora-example.md)
- [Custom Loss Functions](./examples/custom-loss.md)
- [Learning Rate Scheduling](./examples/lr-scheduling.md)
- [Gradient Clipping](./examples/gradient-clipping-example.md)
- [Adapter Sharing](./examples/adapter-sharing.md)

# Development Guide

- [Contributing](./development/contributing.md)
- [EXTREME TDD Methodology](./development/extreme-tdd.md)
- [Testing Strategy](./development/testing-strategy.md)
  - [Unit Tests](./development/unit-tests.md)
  - [Property-Based Tests](./development/property-based-tests.md)
  - [Gradient Checking Tests](./development/gradient-checking.md)
  - [Mutation Testing](./development/mutation-testing.md)
- [Quality Gates](./development/quality-gates.md)
  - [Pre-Commit Hooks](./development/pre-commit-hooks.md)
  - [Continuous Integration](./development/continuous-integration.md)
  - [Code Coverage](./development/code-coverage.md)
  - [Clippy Linting](./development/clippy-linting.md)
- [Benchmarking](./development/benchmarking.md)
- [PMAT Toyota Workflow](./development/pmat-workflow.md)

# Best Practices

- [Optimizer Selection](./best-practices/optimizer-selection.md)
- [Learning Rate Tuning](./best-practices/lr-tuning.md)
- [LoRA Configuration](./best-practices/lora-configuration.md)
- [Memory Optimization](./best-practices/memory-optimization.md)
- [Gradient Stability](./best-practices/gradient-stability.md)
- [Debugging Training Issues](./best-practices/debugging-training.md)
- [Performance Profiling](./best-practices/profiling.md)

# Advanced Topics

- [Custom Backward Passes](./advanced/custom-backward.md)
- [Implementing New Optimizers](./advanced/new-optimizers.md)
- [Custom LoRA Variants](./advanced/custom-lora.md)
- [Advanced Quantization](./advanced/advanced-quantization.md)
- [Distributed Training](./advanced/distributed-training.md)
- [Model Parallelism](./advanced/model-parallelism.md)

# Specifications

- [Autograd Specification](./specifications/autograd-spec.md)
- [Optimizer Specification](./specifications/optimizer-spec.md)
- [LoRA Specification](./specifications/lora-spec.md)
- [Quantization Specification](./specifications/quantization-spec.md)
- [Academic Foundations](./specifications/academic-foundations.md)

# Appendix

- [Glossary](./appendix/glossary.md)
- [Mathematical Notation](./appendix/mathematical-notation.md)
- [References](./appendix/references.md)
- [FAQ](./appendix/faq.md)
- [Changelog](./appendix/changelog.md)
- [Migration Guide](./appendix/migration-guide.md)
- [Benchmarking Results](./appendix/benchmark-results.md)
