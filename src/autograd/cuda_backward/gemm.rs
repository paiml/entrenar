#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::backward::{GemmBackwardAKernel, GemmBackwardBKernel};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::Kernel;

use super::super::cuda_tensor::{CudaTensorError, Result};
#[cfg(feature = "cuda")]
use super::cache::KERNEL_CACHE;

/// Tile size for backward GEMM kernels (C-TILE-BWD-001).
///
/// Must be divisible by 4 (unroll factor). Shared memory per block = 2 * TILE^2 * 4 bytes.
/// TILE=16: 2KB smem, 256 threads/block. Safe for all dimensions including LoRA rank=16.
const BACKWARD_TILE_SIZE: u32 = 16;

/// GEMM backward pass for matrix A on GPU (trueno#109: tiled)
///
/// Given C = A @ B, computes: grad_A = grad_C @ B^T
///
/// Uses tiled GEMM with shared memory (C-TILE-BWD-001) and 4x unrolled inner loop.
#[cfg(feature = "cuda")]
pub fn gemm_backward_a(
    grad_output: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    grad_a: &mut GpuBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let tile = BACKWARD_TILE_SIZE;
    let kernel = GemmBackwardAKernel::tiled_unrolled(m, k, n, tile);
    let kernel_name = kernel.name();
    let ptx = kernel.emit_ptx_for_target(cache.sm_target());

    let key = format!("gemm_backward_a_{m}_{k}_{n}");
    let module = cache.get_or_compile(&key, &ptx)?;

    // Tiled launch: block = (TILE, TILE), grid covers output grad_a[M, K]
    let smem = 2 * tile * tile * 4; // 2 tiles of f32
    let config = LaunchConfig {
        grid: (k.div_ceil(tile), m.div_ceil(tile), 1),
        block: (tile, tile, 1),
        shared_mem: smem,
    };

    let grad_out_ptr = grad_output.as_ptr();
    let b_ptr = b.as_ptr();
    let grad_a_ptr = grad_a.as_ptr();

    // PTX kernel signature: (grad_c_ptr, b_ptr, grad_a_ptr, m, n, k)
    // CRITICAL: must match param declaration order in GemmBackwardAKernel::build_ptx()
    let mut args: [*mut std::ffi::c_void; 6] = [
        &grad_out_ptr as *const _ as *mut _,
        &b_ptr as *const _ as *mut _,
        &grad_a_ptr as *const _ as *mut _,
        &m as *const _ as *mut _,
        &n as *const _ as *mut _,
        &k as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, kernel_name, &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("GEMM backward A launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// GEMM backward pass for matrix B on GPU (trueno#109: tiled)
///
/// Given C = A @ B, computes: grad_B = A^T @ grad_C
///
/// Uses tiled GEMM with shared memory (C-TILE-BWD-002) and 4x unrolled inner loop.
#[cfg(feature = "cuda")]
pub fn gemm_backward_b(
    a: &GpuBuffer<f32>,
    grad_output: &GpuBuffer<f32>,
    grad_b: &mut GpuBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let tile = BACKWARD_TILE_SIZE;
    let kernel = GemmBackwardBKernel::tiled_unrolled(m, k, n, tile);
    let kernel_name = kernel.name();
    let ptx = kernel.emit_ptx_for_target(cache.sm_target());

    let key = format!("gemm_backward_b_{m}_{k}_{n}");
    let module = cache.get_or_compile(&key, &ptx)?;

    // Tiled launch: block = (TILE, TILE), grid covers output grad_b[K, N]
    let smem = 2 * tile * tile * 4;
    let config = LaunchConfig {
        grid: (n.div_ceil(tile), k.div_ceil(tile), 1),
        block: (tile, tile, 1),
        shared_mem: smem,
    };

    let a_ptr = a.as_ptr();
    let grad_out_ptr = grad_output.as_ptr();
    let grad_b_ptr = grad_b.as_ptr();

    // PTX kernel signature: (a_ptr, grad_c_ptr, grad_b_ptr, m, n, k)
    // CRITICAL: must match param declaration order in GemmBackwardBKernel::build_ptx()
    let mut args: [*mut std::ffi::c_void; 6] = [
        &a_ptr as *const _ as *mut _,
        &grad_out_ptr as *const _ as *mut _,
        &grad_b_ptr as *const _ as *mut _,
        &m as *const _ as *mut _,
        &n as *const _ as *mut _,
        &k as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, kernel_name, &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("GEMM backward B launch failed: {e:?}"))
        })?;
    }

    Ok(())
}
