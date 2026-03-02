#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::{Batched4DGemmKernel, FusedSwigluKernel, GemmKernel, Kernel, Nf4GemmKernel};

use crate::autograd::cuda_tensor::{CudaTensorError, Result};

#[cfg(feature = "cuda")]
use super::cache::FORWARD_KERNEL_CACHE;

/// Fused SwiGLU forward pass on GPU (ENT-150)
///
/// Computes: output = SiLU(gate) * up
/// Fuses two operations into one kernel for better memory bandwidth.
#[cfg(feature = "cuda")]
pub fn fused_swiglu_forward(
    gate: &GpuBuffer<f32>,
    up: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = FusedSwigluKernel::new(n);
    let ptx = kernel.emit_ptx_for_target(cache.sm_target());

    let key = format!("fused_swiglu_forward_{n}");
    let module = cache.get_or_compile(&key, &ptx)?;

    let config = LaunchConfig { grid: (n.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem: 0 };

    let gate_ptr = gate.as_ptr();
    let up_ptr = up.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: [*mut std::ffi::c_void; 4] = [
        &gate_ptr as *const _ as *mut _,
        &up_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, "fused_swiglu", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("Fused SwiGLU forward launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// GEMM forward pass on GPU
///
/// Computes: C = A @ B where A is MxK, B is KxN, C is MxN
#[cfg(feature = "cuda")]
pub fn gemm_forward(
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = GemmKernel::naive(m, n, k);
    let ptx = kernel.emit_ptx_for_target(cache.sm_target());

    let key = format!("gemm_forward_{m}_{k}_{n}");
    let module = cache.get_or_compile(&key, &ptx)?;

    // Use 16x16 thread blocks for GEMM
    // Kernel: col = ctaid.x * 16 + tid.x, row = ctaid.y * 16 + tid.y
    // So grid.x = ceil(N/16) for columns, grid.y = ceil(M/16) for rows
    let config = LaunchConfig {
        grid: (n.div_ceil(16), m.div_ceil(16), 1),
        block: (16, 16, 1),
        shared_mem: 0,
    };

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_ptr();

    // PTX kernel signature: (a_ptr, b_ptr, c_ptr, m, n, k)
    // CRITICAL: must match param declaration order in GemmKernel::build_naive()
    let mut args: [*mut std::ffi::c_void; 6] = [
        &a_ptr as *const _ as *mut _,
        &b_ptr as *const _ as *mut _,
        &c_ptr as *const _ as *mut _,
        &m as *const _ as *mut _,
        &n as *const _ as *mut _,
        &k as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, "gemm_naive", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("GEMM forward launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// Batched 4D GEMM forward pass on GPU for multi-head attention
///
/// Computes: C[b,h] = A[b,h] @ B[b,h] for each batch b and head h
/// Pattern: [batch, heads, m, k] @ [batch, heads, k, n] -> [batch, heads, m, n]
///
/// # Contract (C-B4DGEMM-001)
///
/// - **Precondition**: a.len() >= batch * heads * m * k, b.len() >= batch * heads * k * n,
///   c.len() >= batch * heads * m * n
/// - **Postcondition**: C[b,h] = A[b,h] @ B[b,h] for all (b,h) in [0,batch)×[0,heads)
/// - **Invariant**: Zero CPU-side data transfers
#[cfg(feature = "cuda")]
pub fn batched_4d_gemm_forward(
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    c: &mut GpuBuffer<f32>,
    batch: u32,
    heads: u32,
    m: u32,
    n: u32,
    k: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = Batched4DGemmKernel::new(batch, heads, m, n, k);
    let tile_size = kernel.config.tile_size;
    let ptx = kernel.emit_ptx_for_target(cache.sm_target());

    let key = format!("batched_4d_gemm_{batch}_{heads}_{m}_{n}_{k}");
    let module = cache.get_or_compile(&key, &ptx)?;

    // Grid: ((m+tile-1)/tile, (n+tile-1)/tile, batch * heads)
    // Block: (tile_size, tile_size, 1)
    // Shared memory: tile_size * tile_size * 4 * 2 bytes (tiles for A and B)
    let config = LaunchConfig {
        grid: (n.div_ceil(tile_size), m.div_ceil(tile_size), batch * heads),
        block: (tile_size, tile_size, 1),
        shared_mem: tile_size * tile_size * 4 * 2,
    };

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_ptr();

    // PTX kernel signature: (a_ptr, b_ptr, c_ptr, batch, heads, m, n, k)
    let mut args: [*mut std::ffi::c_void; 8] = [
        &a_ptr as *const _ as *mut _,
        &b_ptr as *const _ as *mut _,
        &c_ptr as *const _ as *mut _,
        &batch as *const _ as *mut _,
        &heads as *const _ as *mut _,
        &m as *const _ as *mut _,
        &n as *const _ as *mut _,
        &k as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, "batched_4d_gemm", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("Batched 4D GEMM forward launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// NF4 quantized GEMM forward pass on GPU (trueno#108).
///
/// Computes: C = A @ dequant(B_nf4) where:
/// - A is MxK (f32 activations)
/// - B_nf4 is packed 4-bit NF4 weights (u8)
/// - B_scales is per-block f32 scale factors
/// - C is MxN (f32 output)
///
/// The kernel fuses dequantization with matmul: no intermediate fp32 weight buffer needed.
///
/// # Contract: C-NF4-003 (GEMM Numerical Parity)
///
/// `nf4_gemm(A, Q) ≈ naive_gemm(A, dequantize(Q))` within 1e-3 per-element.
#[cfg(feature = "cuda")]
pub fn gemm_nf4_forward(
    a: &GpuBuffer<f32>,
    b_nf4: &GpuBuffer<u8>,
    b_scales: &GpuBuffer<f32>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = Nf4GemmKernel::new(m, n, k);
    let tile_size = kernel.tile_size;
    let ptx = kernel.emit_ptx_for_target(cache.sm_target());

    let key = format!("nf4_gemm_forward_{m}_{k}_{n}");
    let module = cache.get_or_compile(&key, &ptx)?;

    // Use tile_size × tile_size thread blocks (same as Q4K GEMM)
    let config = LaunchConfig {
        grid: (n.div_ceil(tile_size), m.div_ceil(tile_size), 1),
        block: (tile_size * tile_size, 1, 1),
        shared_mem: 16 * 4, // NF4 codebook LUT (16 × f32)
    };

    let a_ptr = a.as_ptr();
    let b_nf4_ptr = b_nf4.as_ptr();
    let b_scales_ptr = b_scales.as_ptr();
    let c_ptr = c.as_ptr();

    // PTX kernel signature: (a_ptr, b_nf4_ptr, b_scales_ptr, c_ptr, m, n, k)
    // CRITICAL: must match param declaration order in Nf4GemmKernel::build_ptx()
    let mut args: [*mut std::ffi::c_void; 7] = [
        &a_ptr as *const _ as *mut _,
        &b_nf4_ptr as *const _ as *mut _,
        &b_scales_ptr as *const _ as *mut _,
        &c_ptr as *const _ as *mut _,
        &m as *const _ as *mut _,
        &n as *const _ as *mut _,
        &k as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, "nf4_gemm_fused", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("NF4 GEMM forward launch failed: {e:?}"))
        })?;
    }

    Ok(())
}
