#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CublasHandle, CudaStream, GemmOp, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::{
    Batched4DGemmKernel, FusedSwigluKernel, GemmKernel, Kernel, Nf4GemmKernel,
};

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

    let key = format!("fused_swiglu_forward_{n}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let kernel = FusedSwigluKernel::new(n);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

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
///
/// Dispatches to cuBLAS tensor cores when available (ALB-075), falling back
/// to hand-written PTX naive GEMM otherwise.
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

    // cuBLAS fast path: tensor core GEMM (ALB-075)
    if let Some(cublas) = cache.cublas() {
        return cublas_gemm_forward(cublas, a, b, c, m, k, n);
    }

    // PTX fallback
    let key = format!("gemm_forward_{m}_{k}_{n}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let kernel = GemmKernel::naive(m, n, k);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

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

/// cuBLAS GEMM forward: C[M,N] = A[M,K] @ B[K,N] (row-major via B^T@A^T identity)
#[cfg(feature = "cuda")]
fn cublas_gemm_forward(
    cublas: &CublasHandle,
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
) -> Result<()> {
    cublas
        .gemm_f32(
            GemmOp::NoTrans,
            GemmOp::NoTrans,
            n as i32,
            m as i32,
            k as i32,
            1.0,
            b.as_ptr(),
            n as i32,
            a.as_ptr(),
            k as i32,
            0.0,
            c.as_ptr(),
            n as i32,
        )
        .map_err(|e| CudaTensorError::KernelError(format!("cuBLAS GEMM forward failed: {e:?}")))
}

/// cuBLAS backward A: grad_A[M,K] = grad_C[M,N] @ B[K,N]^T
#[cfg(feature = "cuda")]
pub(crate) fn cublas_gemm_backward_a(
    cublas: &CublasHandle,
    grad_output: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    grad_a: &mut GpuBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
) -> Result<()> {
    cublas
        .gemm_f32(
            GemmOp::Trans,
            GemmOp::NoTrans,
            k as i32,
            m as i32,
            n as i32,
            1.0,
            b.as_ptr(),
            n as i32,
            grad_output.as_ptr(),
            n as i32,
            0.0,
            grad_a.as_ptr(),
            k as i32,
        )
        .map_err(|e| CudaTensorError::KernelError(format!("cuBLAS GEMM backward_a failed: {e:?}")))
}

/// cuBLAS backward B: grad_B[K,N] = A[M,K]^T @ grad_C[M,N]
#[cfg(feature = "cuda")]
pub(crate) fn cublas_gemm_backward_b(
    cublas: &CublasHandle,
    a: &GpuBuffer<f32>,
    grad_output: &GpuBuffer<f32>,
    grad_b: &mut GpuBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
) -> Result<()> {
    cublas
        .gemm_f32(
            GemmOp::NoTrans,
            GemmOp::Trans,
            n as i32,
            k as i32,
            m as i32,
            1.0,
            grad_output.as_ptr(),
            n as i32,
            a.as_ptr(),
            k as i32,
            0.0,
            grad_b.as_ptr(),
            n as i32,
        )
        .map_err(|e| CudaTensorError::KernelError(format!("cuBLAS GEMM backward_b failed: {e:?}")))
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

    // ALB-075 Phase 4: cuBLAS strided batched GEMM for attention (16x faster than PTX)
    if let Some(cublas) = cache.cublas() {
        let batch_count = (batch * heads) as i32;
        let stride_a = i64::from(m) * i64::from(k);
        let stride_b = i64::from(k) * i64::from(n);
        let stride_c = i64::from(m) * i64::from(n);
        return cublas
            .gemm_f32_strided_batched_row_major(
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                stride_a,
                b.as_ptr(),
                stride_b,
                0.0,
                c.as_ptr(),
                stride_c,
                batch_count,
            )
            .map_err(|e| {
                CudaTensorError::KernelError(format!("cuBLAS batched 4D GEMM failed: {e:?}"))
            });
    }

    let kernel = Batched4DGemmKernel::new(batch, heads, m, n, k);
    let tile_size = kernel.config.tile_size;

    let key = format!("batched_4d_gemm_{batch}_{heads}_{m}_{n}_{k}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

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

    let key = format!("nf4_gemm_forward_{m}_{k}_{n}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

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

/// BF16-precision GEMM forward pass on GPU (R-002: BF16 mixed precision).
///
/// Computes: C = A @ B where A is MxK, B is KxN, C is MxN
/// Both inputs are f32 (FP32 master weights), but compute is done at BF16
/// precision: each operand is truncated to BF16 (7-bit mantissa) before
/// multiply, with FP32 accumulation. Output is FP32.
///
/// This implements the standard mixed-precision pattern:
/// - FP32 storage (master weights stay in full precision)
/// - BF16 compute (reduced precision multiply for bandwidth savings)
/// - FP32 accumulation (no loss in reduction precision)
///
/// # Contract (C-BF16GEMM-001)
///
/// - `C[i,j] = Σ_k trunc_bf16(A[i,k]) * trunc_bf16(B[k,j])` accumulated in f32
/// - `trunc_bf16(x)` = f32::from_bits(x.to_bits() & 0xFFFF0000)
/// - Output matches CPU BF16 reference within f32 accumulation tolerance
#[cfg(feature = "cuda")]
pub fn gemm_forward_bf16(
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

    let key = format!("gemm_bf16_compute_{m}_{k}_{n}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let ptx = build_gemm_bf16_compute_ptx(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    let config = LaunchConfig {
        grid: (n.div_ceil(16), m.div_ceil(16), 1),
        block: (16, 16, 1),
        shared_mem: 0,
    };

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_ptr();

    // PTX kernel signature: (a_ptr, b_ptr, c_ptr, m, n, k)
    // CRITICAL: must match param declaration order in build_gemm_bf16_compute_ptx()
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
        stream.launch_kernel(module, "gemm_bf16_compute", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("BF16 GEMM forward launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// Build PTX for BF16-precision GEMM kernel.
///
/// Naive GEMM with inline BF16 truncation: loads f32, truncates to bf16 precision
/// (AND 0xFFFF0000), multiplies as f32, accumulates in f32. This matches the
/// precision characteristics of hardware bf16 tensor cores (bf16 multiply, f32 accum).
#[cfg(feature = "cuda")]
fn build_gemm_bf16_compute_ptx(sm_target: &str) -> String {
    format!(
        r#".version 7.0
.target {sm_target}
.address_size 64

.visible .entry gemm_bf16_compute(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u32 M,
    .param .u32 N,
    .param .u32 K
) {{
    .reg .u32 %r<20>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<4>;
    .reg .pred %p<4>;

    // col = ctaid.x * 16 + tid.x
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;

    // row = ctaid.y * 16 + tid.y
    mov.u32 %r4, %ctaid.y;
    mov.u32 %r5, %ntid.y;
    mov.u32 %r6, %tid.y;
    mad.lo.u32 %r7, %r4, %r5, %r6;

    // Load params
    ld.param.u64 %rd0, [a_ptr];
    ld.param.u64 %rd1, [b_ptr];
    ld.param.u64 %rd2, [c_ptr];
    ld.param.u32 %r8, [M];
    ld.param.u32 %r9, [N];
    ld.param.u32 %r10, [K];

    // Bounds check: row < M && col < N
    setp.ge.u32 %p0, %r7, %r8;
    setp.ge.u32 %p1, %r3, %r9;
    or.pred %p2, %p0, %p1;
    @%p2 bra exit;

    // acc = 0.0f
    mov.f32 %f0, 0f00000000;

    // Loop: for i = 0; i < K; i++
    mov.u32 %r11, 0;
loop_start:
    setp.ge.u32 %p3, %r11, %r10;
    @%p3 bra loop_end;

    // Load A[row, i] as u32 bits, truncate to bf16 precision
    mul.lo.u32 %r12, %r7, %r10;
    add.u32 %r12, %r12, %r11;
    mul.wide.u32 %rd3, %r12, 4;
    add.u64 %rd3, %rd0, %rd3;
    ld.global.u32 %r13, [%rd3];
    and.b32 %r13, %r13, 0xFFFF0000;
    mov.b32 %f1, %r13;

    // Load B[i, col] as u32 bits, truncate to bf16 precision
    mul.lo.u32 %r14, %r11, %r9;
    add.u32 %r14, %r14, %r3;
    mul.wide.u32 %rd4, %r14, 4;
    add.u64 %rd4, %rd1, %rd4;
    ld.global.u32 %r15, [%rd4];
    and.b32 %r15, %r15, 0xFFFF0000;
    mov.b32 %f2, %r15;

    // acc += a_bf16 * b_bf16 (FMA in f32 accumulator)
    fma.rn.f32 %f0, %f1, %f2, %f0;

    add.u32 %r11, %r11, 1;
    bra loop_start;

loop_end:
    // Store C[row, col]
    mul.lo.u32 %r16, %r7, %r9;
    add.u32 %r16, %r16, %r3;
    mul.wide.u32 %rd5, %r16, 4;
    add.u64 %rd5, %rd2, %rd5;
    st.global.f32 [%rd5], %f0;

exit:
    ret;
}}
"#
    )
}

/// NF4 transposed GEMM for backward pass (ENT-153: QLoRA backward).
///
/// Computes: `grad_input[M×K] = grad_output[M×N] @ dequant(W_nf4[K×N])^T`
///
/// This is the gradient-flow kernel: given upstream gradient and frozen NF4 weights,
/// computes the input gradient without materializing fp32 weights.
///
/// # Arguments
///
/// * `grad_output` - Upstream gradient `[M × N]` (f32)
/// * `w_nf4` - Frozen NF4-packed weights for `W[K × N]` (u8)
/// * `w_scales` - Per-block scales for `W[K × N]` (f32)
/// * `grad_input` - Output gradient `[M × K]` (f32)
/// * `m` - Rows of grad_output (seq_len)
/// * `n` - Columns of W (reduction dimension)
/// * `k` - Rows of W (output columns = input dimension)
///
/// # Contract: C-NF4T-001 (Transposed GEMM Parity)
///
/// `gemm_nf4_backward_a(grad, W_nf4) ≈ gemm(grad, dequant(W)^T)` within 1e-3.
#[cfg(feature = "cuda")]
pub fn gemm_nf4_backward_a(
    grad_output: &GpuBuffer<f32>,
    w_nf4: &GpuBuffer<u8>,
    w_scales: &GpuBuffer<f32>,
    grad_input: &mut GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
    stream: &CudaStream,
) -> Result<()> {
    let _cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;

    // TODO: Nf4GemmTransposeKernel not yet implemented in trueno-gpu
    let _ = (grad_output, w_nf4, w_scales, grad_input, m, n, k, stream);
    Err(CudaTensorError::KernelError("NF4 GEMM transpose not yet implemented".to_string()))
}
