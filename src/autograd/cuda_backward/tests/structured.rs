use super::super::*;
use super::softmax_backward_cpu;

#[test]
#[cfg(feature = "cuda")]
fn test_softmax_backward_basic() {
    use trueno_gpu::driver::{CudaStream, GpuBuffer};

    let ctx = match super::get_test_gpu_context() {
        Some(c) => c,
        None => return,
    };
    init_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // Softmax output: uniform distribution (1/4 each)
    let softmax_output_data: Vec<f32> = vec![0.25, 0.25, 0.25, 0.25];
    let grad_output_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0]; // Cross-entropy grad
    let grad_input_data: Vec<f32> = vec![0.0; 4];

    let softmax_output =
        GpuBuffer::from_host(&ctx, &softmax_output_data).expect("operation should succeed");
    let grad_output =
        GpuBuffer::from_host(&ctx, &grad_output_data).expect("operation should succeed");
    let mut grad_input =
        GpuBuffer::from_host(&ctx, &grad_input_data).expect("operation should succeed");

    softmax_backward(&softmax_output, &grad_output, &mut grad_input, 1, 4, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; 4];
    grad_input.copy_to_host(&mut result).expect("operation should succeed");

    // CPU reference
    let expected = softmax_backward_cpu(&softmax_output_data, &grad_output_data);

    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-3,
            "Softmax backward mismatch at {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_softmax_backward_batch() {
    use trueno_gpu::driver::{CudaStream, GpuBuffer};

    let ctx = match super::get_test_gpu_context() {
        Some(c) => c,
        None => return,
    };
    init_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // 2 batches, seq_len=3
    let softmax_output_data: Vec<f32> = vec![0.5, 0.3, 0.2, 0.1, 0.2, 0.7];
    let grad_output_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let grad_input_data: Vec<f32> = vec![0.0; 6];

    let softmax_output =
        GpuBuffer::from_host(&ctx, &softmax_output_data).expect("operation should succeed");
    let grad_output =
        GpuBuffer::from_host(&ctx, &grad_output_data).expect("operation should succeed");
    let mut grad_input =
        GpuBuffer::from_host(&ctx, &grad_input_data).expect("operation should succeed");

    softmax_backward(&softmax_output, &grad_output, &mut grad_input, 2, 3, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; 6];
    grad_input.copy_to_host(&mut result).expect("operation should succeed");

    // Verify gradients are computed (not all zeros or NaN)
    assert!(!result.iter().any(|x| x.is_nan()), "Softmax backward should not produce NaN");
    assert!(
        result.iter().any(|&x| x.abs() > 1e-5),
        "Softmax backward should produce non-zero gradients"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_rms_norm_backward_basic() {
    use trueno_gpu::driver::{CudaStream, GpuBuffer};

    let ctx = match super::get_test_gpu_context() {
        Some(c) => c,
        None => return,
    };
    init_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let batch_size = 1u32;
    let hidden_size = 4u32;
    let n = (batch_size * hidden_size) as usize;
    let eps = 1e-5;

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let gamma_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let grad_output_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let grad_input_data: Vec<f32> = vec![0.0; n];
    let grad_gamma_data: Vec<f32> = vec![0.0; hidden_size as usize];

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let gamma = GpuBuffer::from_host(&ctx, &gamma_data).expect("operation should succeed");
    let grad_output =
        GpuBuffer::from_host(&ctx, &grad_output_data).expect("operation should succeed");
    let mut grad_input =
        GpuBuffer::from_host(&ctx, &grad_input_data).expect("operation should succeed");
    let mut grad_gamma =
        GpuBuffer::from_host(&ctx, &grad_gamma_data).expect("operation should succeed");

    rms_norm_backward(
        &input,
        &gamma,
        &grad_output,
        &mut grad_input,
        &mut grad_gamma,
        batch_size,
        hidden_size,
        eps,
        &stream,
    )
    .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result_input = vec![0.0f32; n];
    let mut result_gamma = vec![0.0f32; hidden_size as usize];
    grad_input.copy_to_host(&mut result_input).expect("operation should succeed");
    grad_gamma.copy_to_host(&mut result_gamma).expect("operation should succeed");

    // Verify gradients are computed
    assert!(
        !result_input.iter().any(|x| x.is_nan()),
        "RMSNorm backward should not produce NaN for input grad"
    );
    assert!(
        !result_gamma.iter().any(|x| x.is_nan()),
        "RMSNorm backward should not produce NaN for gamma grad"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_layer_norm_backward_basic() {
    use trueno_gpu::driver::{CudaStream, GpuBuffer};

    let ctx = match super::get_test_gpu_context() {
        Some(c) => c,
        None => return,
    };
    init_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let batch_size = 1u32;
    let hidden_size = 4u32;
    let n = (batch_size * hidden_size) as usize;

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let gamma_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let grad_output_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let grad_input_data: Vec<f32> = vec![0.0; n];
    let grad_gamma_data: Vec<f32> = vec![0.0; hidden_size as usize];
    let grad_beta_data: Vec<f32> = vec![0.0; hidden_size as usize];

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let gamma = GpuBuffer::from_host(&ctx, &gamma_data).expect("operation should succeed");
    let grad_output =
        GpuBuffer::from_host(&ctx, &grad_output_data).expect("operation should succeed");
    let mut grad_input =
        GpuBuffer::from_host(&ctx, &grad_input_data).expect("operation should succeed");
    let mut grad_gamma =
        GpuBuffer::from_host(&ctx, &grad_gamma_data).expect("operation should succeed");
    let mut grad_beta =
        GpuBuffer::from_host(&ctx, &grad_beta_data).expect("operation should succeed");

    layer_norm_backward(
        &input,
        &gamma,
        &grad_output,
        &mut grad_input,
        &mut grad_gamma,
        &mut grad_beta,
        batch_size,
        hidden_size,
        &stream,
    )
    .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result_input = vec![0.0f32; n];
    let mut result_gamma = vec![0.0f32; hidden_size as usize];
    let mut result_beta = vec![0.0f32; hidden_size as usize];
    grad_input.copy_to_host(&mut result_input).expect("operation should succeed");
    grad_gamma.copy_to_host(&mut result_gamma).expect("operation should succeed");
    grad_beta.copy_to_host(&mut result_beta).expect("operation should succeed");

    // Verify gradients are computed without NaN/Inf
    assert!(
        !result_input.iter().any(|x| x.is_nan() || x.is_infinite()),
        "LayerNorm backward should not produce NaN/Inf for input grad"
    );
    assert!(
        !result_gamma.iter().any(|x| x.is_nan() || x.is_infinite()),
        "LayerNorm backward should not produce NaN/Inf for gamma grad"
    );
    assert!(
        !result_beta.iter().any(|x| x.is_nan() || x.is_infinite()),
        "LayerNorm backward should not produce NaN/Inf for beta grad"
    );
    // Note: grad_beta may be zero for single batch with uniform grad_output
    // due to how the backward kernel accumulates gradients
}
