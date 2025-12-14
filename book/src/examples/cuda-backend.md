# CUDA Backend Configuration

This chapter demonstrates how to configure entrenar for NVIDIA CUDA acceleration using the trueno/cuda-monitor feature.

## Overview

entrenar v0.2.8 supports multiple compute backends via trueno:

| Backend | Feature | Use Case |
|---------|---------|----------|
| CPU SIMD | (default) | Portable, works everywhere |
| GPU | `--features gpu` | Cross-platform GPU via wgpu |
| CUDA | `--features cuda` | Maximum performance on NVIDIA |

## Cargo.toml Configuration

Enable CUDA support in your `Cargo.toml`:

```toml
[dependencies]
# Default CPU SIMD backend
entrenar = "0.2.8"

# With NVIDIA CUDA support
entrenar = { version = "0.2.8", features = ["cuda"] }

# With cross-platform GPU (wgpu)
entrenar = { version = "0.2.8", features = ["gpu"] }

# Both GPU and CUDA
entrenar = { version = "0.2.8", features = ["gpu", "cuda"] }
```

## Running the Example

```bash
# Without CUDA (shows feature availability)
cargo run --example cuda_backend

# With CUDA (detects NVIDIA GPU)
cargo run --example cuda_backend --features cuda

# With GPU (wgpu backend)
cargo run --example cuda_backend --features gpu
```

## Example Output (with RTX 4090)

```
╔══════════════════════════════════════════════════════════════╗
║      CUDA Backend Detection & Monitoring (trueno-gpu)        ║
╚══════════════════════════════════════════════════════════════╝

┌─ Feature Availability ──────────────────────────────────────┐
│ ✅ CUDA feature: ENABLED
│    trueno/cuda-monitor is available
│
│ Default backend: CPU SIMD (trueno)
└──────────────────────────────────────────────────────────────┘

┌─ CUDA Device Detection ─────────────────────────────────────┐
│ Querying NVIDIA driver via trueno-gpu...
│
│ ✅ NVIDIA driver detected
│
│ Device Information (via nvidia-smi):
│   GPU 0: NVIDIA GeForce RTX 4090
│   - Memory: 24564 MiB
│   - Compute: SM 8.9
│
│ With cuda feature, trueno-gpu provides:
│   - Pure Rust PTX generation (no nvcc needed)
│   - Runtime CUDA driver loading
│   - Device memory management
│   - Kernel execution
└──────────────────────────────────────────────────────────────┘
```

## Trueno Integration

entrenar uses trueno for compute acceleration:

```
trueno v0.8.3
├── CPU SIMD (AVX2, AVX-512, NEON)
├── trueno/gpu (wgpu compute shaders)
└── trueno/cuda-monitor (via trueno-gpu v0.2.0)
```

### trueno-gpu Features

The `cuda` feature enables trueno-gpu, which provides:

- **Pure Rust PTX Generation**: No LLVM or nvcc compiler required
- **Runtime Driver Loading**: Dynamically loads libcuda.so
- **Device Memory Management**: Safe GPU memory allocation
- **Kernel Execution**: Launch CUDA kernels from Rust

## Performance Expectations

| Backend | Relative Speed | Best For |
|---------|---------------|----------|
| CPU SIMD | 1x (baseline) | General workloads, portability |
| GPU (wgpu) | 5-50x | Cross-platform GPU acceleration |
| CUDA | 10-100x | Maximum NVIDIA performance |

## GPU Training Example

For GPU-accelerated training with real-time monitoring:

```bash
# MNIST training with GPU acceleration
cargo run --example mnist_train_gpu --features gpu

# With CUDA for NVIDIA GPUs
cargo run --example mnist_train_gpu --features cuda
```

## Andon Monitoring Integration

With CUDA enabled, entrenar provides GPU monitoring via the Andon system:

```rust
use entrenar::monitor::gpu::{GpuMonitor, AndonSystem};

// Create monitor
let monitor = GpuMonitor::new()?;

// Collect metrics
let metrics = monitor.collect_metrics()?;

for gpu in &metrics {
    println!("GPU {}: {}°C, {}% util, {:.1} GB / {:.1} GB",
        gpu.device_id,
        gpu.temperature_celsius,
        gpu.utilization_percent,
        gpu.memory_used_bytes as f64 / 1e9,
        gpu.memory_total_bytes as f64 / 1e9
    );
}

// Check for alerts (thermal throttling, memory pressure)
let andon = AndonSystem::default();
let alerts = andon.check(&metrics);
```

## Requirements

### For CUDA Feature

1. **NVIDIA GPU**: Any CUDA-capable GPU
2. **NVIDIA Driver**: 450.x or newer recommended
3. **No CUDA Toolkit Required**: trueno-gpu uses pure Rust PTX

Verify driver installation:

```bash
nvidia-smi
```

### For GPU Feature (wgpu)

1. **Vulkan** (Linux/Windows) or **Metal** (macOS)
2. **No special drivers** beyond standard GPU drivers

## See Also

- [GPU Monitoring (Andon)](../mlops/gpu-monitoring.md)
- [MNIST Training with GPU](./mnist-train-gpu.md)
- [Backend Comparison](../architecture/overview.md)
