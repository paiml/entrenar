//! Regression metrics: R2Score, MAE, RMSE

use crate::Tensor;

use super::Metric;

/// R² (coefficient of determination) for regression
///
/// R² = 1 - SS_res / SS_tot
///
/// Where:
/// - SS_res = sum((y - y_pred)²)
/// - SS_tot = sum((y - y_mean)²)
///
/// R² = 1.0 is perfect prediction, 0.0 means predicting the mean
///
/// # Example
///
/// ```
/// use entrenar::train::{R2Score, Metric};
/// use entrenar::Tensor;
///
/// let metric = R2Score;
/// let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
/// let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
///
/// let r2 = metric.compute(&pred, &target);
/// assert!((r2 - 1.0).abs() < 1e-5);  // Perfect prediction
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct R2Score;

impl Metric for R2Score {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(predictions.len(), targets.len());

        if predictions.is_empty() {
            return 0.0;
        }

        let y_mean: f32 = targets.data().mean().unwrap_or(0.0);

        let ss_res: f32 = predictions
            .data()
            .iter()
            .zip(targets.data().iter())
            .map(|(&p, &t)| (t - p).powi(2))
            .sum();

        let ss_tot: f32 = targets.data().iter().map(|&t| (t - y_mean).powi(2)).sum();

        if ss_tot == 0.0 {
            return if ss_res == 0.0 { 1.0 } else { 0.0 };
        }

        1.0 - (ss_res / ss_tot)
    }

    fn name(&self) -> &'static str {
        "R²"
    }
}

/// Mean Absolute Error (MAE) metric
///
/// MAE = mean(|y - y_pred|)
///
/// # Example
///
/// ```
/// use entrenar::train::{MAE, Metric};
/// use entrenar::Tensor;
///
/// let metric = MAE;
/// let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
/// let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false);
///
/// let mae = metric.compute(&pred, &target);
/// assert!((mae - 0.5).abs() < 1e-5);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct MAE;

impl Metric for MAE {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(predictions.len(), targets.len());

        if predictions.is_empty() {
            return 0.0;
        }

        predictions
            .data()
            .iter()
            .zip(targets.data().iter())
            .map(|(&p, &t)| (p - t).abs())
            .sum::<f32>()
            / predictions.len() as f32
    }

    fn name(&self) -> &'static str {
        "MAE"
    }

    fn higher_is_better(&self) -> bool {
        false
    }
}

/// Root Mean Squared Error (RMSE) metric
///
/// RMSE = sqrt(mean((y - y_pred)²))
///
/// # Example
///
/// ```
/// use entrenar::train::{RMSE, Metric};
/// use entrenar::Tensor;
///
/// let metric = RMSE;
/// let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
/// let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
///
/// let rmse = metric.compute(&pred, &target);
/// assert!(rmse < 1e-5);  // Perfect prediction
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct RMSE;

impl Metric for RMSE {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(predictions.len(), targets.len());

        if predictions.is_empty() {
            return 0.0;
        }

        let mse: f32 = predictions
            .data()
            .iter()
            .zip(targets.data().iter())
            .map(|(&p, &t)| (p - t).powi(2))
            .sum::<f32>()
            / predictions.len() as f32;

        mse.sqrt()
    }

    fn name(&self) -> &'static str {
        "RMSE"
    }

    fn higher_is_better(&self) -> bool {
        false
    }
}
