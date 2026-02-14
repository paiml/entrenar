use super::*;

#[test]
#[cfg(feature = "cuda")]
fn test_gemm_forward_basic() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).unwrap();
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();

    // 2x2 @ 2x2 matrix multiplication
    // A = [[1, 2], [3, 4]]
    // B = [[5, 6], [7, 8]]
    // C = A @ B = [[19, 22], [43, 50]]
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
    let c_data: Vec<f32> = vec![0.0; 4];

    let a = GpuBuffer::from_host(&ctx, &a_data).unwrap();
    let b = GpuBuffer::from_host(&ctx, &b_data).unwrap();
    let mut c = GpuBuffer::from_host(&ctx, &c_data).unwrap();

    gemm_forward(&a, &b, &mut c, 2, 2, 2, &stream).unwrap();
    stream.synchronize().unwrap();

    let mut result = vec![0.0f32; 4];
    c.copy_to_host(&mut result).unwrap();
    assert!(
        (result[0] - 19.0).abs() < 1e-3,
        "C[0,0] should be 19, got {}",
        result[0]
    );
    assert!(
        (result[1] - 22.0).abs() < 1e-3,
        "C[0,1] should be 22, got {}",
        result[1]
    );
    assert!(
        (result[2] - 43.0).abs() < 1e-3,
        "C[1,0] should be 43, got {}",
        result[2]
    );
    assert!(
        (result[3] - 50.0).abs() < 1e-3,
        "C[1,1] should be 50, got {}",
        result[3]
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_fused_swiglu_forward_basic() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).unwrap();
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();

    // SwiGLU: output = SiLU(gate) * up
    let gate_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let up_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let n = gate_data.len() as u32;

    let gate = GpuBuffer::from_host(&ctx, &gate_data).unwrap();
    let up = GpuBuffer::from_host(&ctx, &up_data).unwrap();
    let output_data = vec![0.0f32; n as usize];
    let mut output = GpuBuffer::from_host(&ctx, &output_data).unwrap();

    fused_swiglu_forward(&gate, &up, &mut output, n, &stream).unwrap();
    stream.synchronize().unwrap();

    let mut result = vec![0.0f32; n as usize];
    output.copy_to_host(&mut result).unwrap();

    // With up=1, output should equal SiLU(gate)
    // SiLU(x) = x * sigmoid(x) > 0 for x > 0
    for (i, &r) in result.iter().enumerate() {
        assert!(
            r > 0.0,
            "SwiGLU output should be positive for positive gate, got {r} at {i}"
        );
    }
    // Output should increase with gate (monotonic for positive inputs)
    assert!(result[3] > result[2]);
    assert!(result[2] > result[1]);
    assert!(result[1] > result[0]);
}

#[test]
#[cfg(feature = "cuda")]
fn test_gemm_forward_2x2() {
    // Test GEMM with a simple 2x2 case for verification
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).unwrap();
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();

    // A = [[1, 2], [3, 4]] (2x2, row-major)
    // B = [[1, 0], [0, 1]] (2x2, identity)
    // C = A @ B = A
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let c_data: Vec<f32> = vec![0.0; 4];

    let a = GpuBuffer::from_host(&ctx, &a_data).unwrap();
    let b = GpuBuffer::from_host(&ctx, &b_data).unwrap();
    let mut c = GpuBuffer::from_host(&ctx, &c_data).unwrap();

    gemm_forward(&a, &b, &mut c, 2, 2, 2, &stream).unwrap();
    stream.synchronize().unwrap();

    let mut result = vec![0.0f32; 4];
    c.copy_to_host(&mut result).unwrap();

    // C should equal A (since B is identity)
    for (i, (&r, &expected)) in result.iter().zip(a_data.iter()).enumerate() {
        assert!(
            (r - expected).abs() < 1e-4,
            "GEMM mismatch at {i}: got {r}, expected {expected}"
        );
    }
}
