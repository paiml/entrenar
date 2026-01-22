//! SIMD-accelerated AXPY operation

use trueno::vector::Vector;

/// SIMD-accelerated AXPY operation: y = a*x + y
///
/// Used in SGD and momentum updates. Performs scalar-vector multiply and
/// vector addition in a single fused operation.
///
/// # Arguments
/// * `a` - Scalar coefficient
/// * `x` - Input vector (typically gradient or momentum)
/// * `y` - Output vector (updated in-place)
pub fn simd_axpy(a: f32, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    // Convert to Trueno vectors for SIMD operations
    let x_vec = Vector::from_slice(x);
    let y_vec = Vector::from_slice(y);

    // Compute: a*x + y
    let scaled_x = x_vec.scale(a).expect("Scale operation failed");
    let result = scaled_x.add(&y_vec).expect("Add operation failed");

    // Write back to output
    y.copy_from_slice(result.as_slice());
}
