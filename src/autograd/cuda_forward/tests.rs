mod test_activations;
mod test_matmul;
mod test_normalization;

use super::*;

#[test]
fn test_cuda_forward_module_compiles() {
    // This test verifies the module compiles correctly
    assert!(true);
}

#[test]
#[cfg(feature = "cuda")]
fn test_forward_kernel_cache_initialization() {
    use trueno_gpu::driver::{cuda_available, CudaContext};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).unwrap();
    let ctx = std::sync::Arc::new(ctx);
    let result = init_forward_kernel_cache(ctx);
    assert!(result.is_ok());
}
