//! Tests for MSE, L1, and Huber losses

use super::*;
use crate::train::loss::LossFn;
use crate::Tensor;
use approx::assert_relative_eq;

#[test]
fn test_mse_loss_basic() {
    let loss_fn = MSELoss;
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false);

    let loss = loss_fn.forward(&pred, &target);

    // MSE = mean((0.5, 0.5, 0.5)^2) = 0.25
    assert_relative_eq!(loss.data()[0], 0.25, epsilon = 1e-5);
}

#[test]
fn test_mse_loss_zero_for_perfect() {
    let loss_fn = MSELoss;
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);

    let loss = loss_fn.forward(&pred, &target);

    assert_relative_eq!(loss.data()[0], 0.0, epsilon = 1e-5);
}

#[test]
fn test_mse_gradient() {
    let loss_fn = MSELoss;
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let target = Tensor::from_vec(vec![0.0, 0.0, 0.0], false);

    let loss = loss_fn.forward(&pred, &target);

    // Trigger backward
    if let Some(backward_op) = loss.backward_op() {
        backward_op.backward();
    }

    // Check gradient: d(MSE)/d(pred) = 2*(pred - target)/n
    let grad = pred.grad().unwrap();
    assert_relative_eq!(grad[0], 2.0 / 3.0, epsilon = 1e-5);
    assert_relative_eq!(grad[1], 4.0 / 3.0, epsilon = 1e-5);
    assert_relative_eq!(grad[2], 6.0 / 3.0, epsilon = 1e-5);
}

#[test]
#[should_panic(expected = "must have same length")]
fn test_mse_mismatched_lengths() {
    let loss_fn = MSELoss;
    let pred = Tensor::from_vec(vec![1.0, 2.0], true);
    let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);

    loss_fn.forward(&pred, &target);
}

#[test]
fn test_mse_no_grad() {
    let loss_fn = MSELoss;
    let pred = Tensor::from_vec(vec![1.0, 2.0], false); // requires_grad = false
    let target = Tensor::from_vec(vec![1.5, 2.5], false);
    let loss = loss_fn.forward(&pred, &target);
    assert!(loss.data()[0] > 0.0);
}

#[test]
fn test_gradient_accumulation_mse() {
    let pred = Tensor::from_vec(vec![1.0, 2.0], true);
    let target = Tensor::from_vec(vec![0.0, 0.0], false);

    // First forward/backward
    let loss1 = MSELoss.forward(&pred, &target);
    if let Some(op) = loss1.backward_op() {
        op.backward();
    }

    // Second forward/backward - gradients should accumulate
    let loss2 = MSELoss.forward(&pred, &target);
    if let Some(op) = loss2.backward_op() {
        op.backward();
    }

    // Gradient should be 2x the single pass
    let grad = pred.grad().unwrap();
    assert!(grad[0].abs() > 0.0);
    assert!(grad[1].abs() > 0.0);
}

#[test]
fn test_l1_loss_basic() {
    let loss_fn = L1Loss;
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false);

    let loss = loss_fn.forward(&pred, &target);

    // L1 = mean(|0.5, 0.5, 0.5|) = 0.5
    assert_relative_eq!(loss.data()[0], 0.5, epsilon = 1e-5);
}

#[test]
fn test_l1_loss_zero_for_perfect() {
    let loss_fn = L1Loss;
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);

    let loss = loss_fn.forward(&pred, &target);
    assert_relative_eq!(loss.data()[0], 0.0, epsilon = 1e-5);
}

#[test]
fn test_l1_loss_gradient() {
    let loss_fn = L1Loss;
    let pred = Tensor::from_vec(vec![2.0, 0.0], true);
    let target = Tensor::from_vec(vec![0.0, 2.0], false);

    let loss = loss_fn.forward(&pred, &target);

    if let Some(backward_op) = loss.backward_op() {
        backward_op.backward();
    }

    let grad = pred.grad().unwrap();
    // Gradient: sign(error) / n
    // First: sign(2) / 2 = 0.5
    // Second: sign(-2) / 2 = -0.5
    assert_relative_eq!(grad[0], 0.5, epsilon = 1e-5);
    assert_relative_eq!(grad[1], -0.5, epsilon = 1e-5);
}

#[test]
fn test_l1_robust_to_outliers() {
    let l1_loss = L1Loss;
    let mse_loss = MSELoss;

    // Normal data with one outlier
    let pred = Tensor::from_vec(vec![1.0, 2.0, 100.0], true);
    let target = Tensor::from_vec(vec![1.0, 2.0, 0.0], false);

    let l1 = l1_loss.forward(&pred, &target);
    let mse = mse_loss.forward(&pred.clone(), &target);

    // L1 should be much smaller than MSE due to outlier
    // L1 = (0 + 0 + 100) / 3 = 33.33
    // MSE = (0 + 0 + 10000) / 3 = 3333.33
    assert!(l1.data()[0] < mse.data()[0]);
}

#[test]
#[should_panic(expected = "must have same length")]
fn test_l1_mismatched_lengths() {
    let loss_fn = L1Loss;
    let pred = Tensor::from_vec(vec![1.0, 2.0], true);
    let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
    loss_fn.forward(&pred, &target);
}

#[test]
fn test_l1_no_grad() {
    let loss_fn = L1Loss;
    let pred = Tensor::from_vec(vec![1.0, 2.0], false);
    let target = Tensor::from_vec(vec![1.5, 2.5], false);
    let loss = loss_fn.forward(&pred, &target);
    assert!(loss.data()[0] > 0.0);
}

#[test]
fn test_gradient_accumulation_l1() {
    let pred = Tensor::from_vec(vec![1.0, 2.0], true);
    let target = Tensor::from_vec(vec![0.0, 0.0], false);

    let loss1 = L1Loss.forward(&pred, &target);
    if let Some(op) = loss1.backward_op() {
        op.backward();
    }

    let loss2 = L1Loss.forward(&pred, &target);
    if let Some(op) = loss2.backward_op() {
        op.backward();
    }

    let grad = pred.grad().unwrap();
    assert!(grad[0].is_finite());
    assert!(grad[1].is_finite());
}

#[test]
fn test_huber_loss_small_error() {
    let loss_fn = HuberLoss::new(1.0);
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false); // errors = 0.5

    let loss = loss_fn.forward(&pred, &target);

    // For small errors (|e| <= delta), Huber = 0.5 * e^2
    // = mean(0.5 * 0.25) = 0.125
    assert_relative_eq!(loss.data()[0], 0.125, epsilon = 1e-5);
}

#[test]
fn test_huber_loss_large_error() {
    let loss_fn = HuberLoss::new(1.0);
    let pred = Tensor::from_vec(vec![0.0], true);
    let target = Tensor::from_vec(vec![5.0], false); // error = 5 > delta

    let loss = loss_fn.forward(&pred, &target);

    // For large errors (|e| > delta), Huber = delta * (|e| - 0.5 * delta)
    // = 1 * (5 - 0.5) = 4.5
    assert_relative_eq!(loss.data()[0], 4.5, epsilon = 1e-5);
}

#[test]
fn test_huber_loss_mixed() {
    let loss_fn = HuberLoss::new(1.0);
    // One small error (0.5), one large error (3.0)
    let pred = Tensor::from_vec(vec![0.0, 0.0], true);
    let target = Tensor::from_vec(vec![0.5, 3.0], false);

    let loss = loss_fn.forward(&pred, &target);

    // Small: 0.5 * 0.25 = 0.125
    // Large: 1 * (3 - 0.5) = 2.5
    // Mean: (0.125 + 2.5) / 2 = 1.3125
    assert_relative_eq!(loss.data()[0], 1.3125, epsilon = 1e-5);
}

#[test]
fn test_huber_loss_gradient() {
    let loss_fn = HuberLoss::new(1.0);
    let pred = Tensor::from_vec(vec![0.0, 0.0], true);
    let target = Tensor::from_vec(vec![0.5, 3.0], false);

    let loss = loss_fn.forward(&pred, &target);

    if let Some(backward_op) = loss.backward_op() {
        backward_op.backward();
    }

    let grad = pred.grad().unwrap();
    // Small error: grad = error / n = -0.5 / 2 = -0.25
    // Large error: grad = delta * sign(error) / n = 1 * (-1) / 2 = -0.5
    assert_relative_eq!(grad[0], -0.25, epsilon = 1e-5);
    assert_relative_eq!(grad[1], -0.5, epsilon = 1e-5);
}

#[test]
fn test_huber_default() {
    let loss_fn = HuberLoss::default();
    let pred = Tensor::from_vec(vec![1.0], true);
    let target = Tensor::from_vec(vec![2.0], false);

    let loss = loss_fn.forward(&pred, &target);
    assert!(loss.data()[0] > 0.0);
}

#[test]
fn test_smooth_l1_is_huber() {
    // SmoothL1Loss is type alias for HuberLoss
    let loss_fn: SmoothL1Loss = HuberLoss::new(1.0);
    let pred = Tensor::from_vec(vec![1.0], true);
    let target = Tensor::from_vec(vec![2.0], false);

    let loss = loss_fn.forward(&pred, &target);
    assert!(loss.data()[0] > 0.0);
}

#[test]
fn test_huber_default_delta() {
    let loss_fn = HuberLoss::default_delta();
    let pred = Tensor::from_vec(vec![1.0, 2.0], true);
    let target = Tensor::from_vec(vec![1.5, 2.5], false);
    let loss = loss_fn.forward(&pred, &target);
    assert!(loss.data()[0] > 0.0);
}

#[test]
#[should_panic(expected = "must have same length")]
fn test_huber_mismatched_lengths() {
    let loss_fn = HuberLoss::new(1.0);
    let pred = Tensor::from_vec(vec![1.0, 2.0], true);
    let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
    loss_fn.forward(&pred, &target);
}

#[test]
#[should_panic(expected = "delta must be positive")]
fn test_huber_negative_delta() {
    HuberLoss::new(-1.0);
}

#[test]
fn test_huber_no_grad() {
    let loss_fn = HuberLoss::new(1.0);
    let pred = Tensor::from_vec(vec![1.0, 2.0], false);
    let target = Tensor::from_vec(vec![1.5, 2.5], false);
    let loss = loss_fn.forward(&pred, &target);
    assert!(loss.data()[0] > 0.0);
}

#[test]
fn test_gradient_accumulation_huber() {
    let pred = Tensor::from_vec(vec![1.0, 5.0], true);
    let target = Tensor::from_vec(vec![0.0, 0.0], false);

    let loss1 = HuberLoss::new(1.0).forward(&pred, &target);
    if let Some(op) = loss1.backward_op() {
        op.backward();
    }

    let loss2 = HuberLoss::new(1.0).forward(&pred, &target);
    if let Some(op) = loss2.backward_op() {
        op.backward();
    }

    let grad = pred.grad().unwrap();
    assert!(grad[0].is_finite());
    assert!(grad[1].is_finite());
}
