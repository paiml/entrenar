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

/// GEMM backward pass for matrix A on GPU
///
/// Given C = A @ B, computes: grad_A = grad_C @ B^T
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

    let kernel = GemmBackwardAKernel::new(m, k, n);
    let ptx = kernel.emit_ptx_for_target(cache.sm_target());

    let key = format!("gemm_backward_a_{m}_{k}_{n}");
    let module = cache.get_or_compile(&key, &ptx)?;

    // Use 16x16 thread blocks for GEMM
    // Kernel: col = ctaid.x, row = ctaid.y; output is grad_a[M,K]
    // So grid.x = ceil(K/16) for columns, grid.y = ceil(M/16) for rows
    let config = LaunchConfig {
        grid: (k.div_ceil(16), m.div_ceil(16), 1),
        block: (16, 16, 1),
        shared_mem: 0,
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
        stream.launch_kernel(module, "gemm_backward_a", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("GEMM backward A launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// GEMM backward pass for matrix B on GPU
///
/// Given C = A @ B, computes: grad_B = A^T @ grad_C
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

    let kernel = GemmBackwardBKernel::new(m, k, n);
    let ptx = kernel.emit_ptx_for_target(cache.sm_target());

    let key = format!("gemm_backward_b_{m}_{k}_{n}");
    let module = cache.get_or_compile(&key, &ptx)?;

    // Use 16x16 thread blocks for GEMM
    // Kernel: col = ctaid.x, row = ctaid.y; output is grad_b[K,N]
    // So grid.x = ceil(N/16) for columns, grid.y = ceil(K/16) for rows
    let config = LaunchConfig {
        grid: (n.div_ceil(16), k.div_ceil(16), 1),
        block: (16, 16, 1),
        shared_mem: 0,
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
        stream.launch_kernel(module, "gemm_backward_b", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("GEMM backward B launch failed: {e:?}"))
        })?;
    }

    Ok(())
}
