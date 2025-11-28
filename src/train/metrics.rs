//! Evaluation metrics for training and validation
//!
//! This module provides common metrics for evaluating model performance:
//! - Classification: Accuracy, Precision, Recall, F1
//! - Regression: MSE, MAE, R²

use crate::Tensor;

/// Trait for evaluation metrics
pub trait Metric {
    /// Compute the metric given predictions and targets
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f32;

    /// Name of the metric
    fn name(&self) -> &str;

    /// Whether higher values are better (true) or lower (false)
    fn higher_is_better(&self) -> bool {
        true
    }
}

// =============================================================================
// Classification Metrics
// =============================================================================

/// Accuracy metric for classification
///
/// For binary classification: fraction of correct predictions
/// For multi-class: fraction where argmax(pred) == argmax(target)
///
/// # Example
///
/// ```
/// use entrenar::train::{Accuracy, Metric};
/// use entrenar::Tensor;
///
/// let metric = Accuracy::new(0.5);  // threshold for binary
/// let pred = Tensor::from_vec(vec![0.9, 0.2, 0.8], false);
/// let target = Tensor::from_vec(vec![1.0, 0.0, 1.0], false);
///
/// let acc = metric.compute(&pred, &target);
/// assert_eq!(acc, 1.0);  // All correct
/// ```
#[derive(Debug, Clone)]
pub struct Accuracy {
    /// Threshold for binary classification
    threshold: f32,
}

impl Accuracy {
    /// Create new accuracy metric with given threshold for binary classification
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Create accuracy metric with default threshold of 0.5
    pub fn default_threshold() -> Self {
        Self::new(0.5)
    }
}

impl Default for Accuracy {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Metric for Accuracy {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(
            predictions.len(),
            targets.len(),
            "Predictions and targets must have same length"
        );

        if predictions.is_empty() {
            return 0.0;
        }

        let correct: usize = predictions
            .data()
            .iter()
            .zip(targets.data().iter())
            .filter(|(&p, &t)| {
                // Binary classification
                let pred_class = if p >= self.threshold { 1.0 } else { 0.0 };
                (pred_class - t).abs() < 0.5
            })
            .count();

        correct as f32 / predictions.len() as f32
    }

    fn name(&self) -> &str {
        "Accuracy"
    }
}

/// Precision metric (true positives / predicted positives)
///
/// # Example
///
/// ```
/// use entrenar::train::{Precision, Metric};
/// use entrenar::Tensor;
///
/// let metric = Precision::new(0.5);
/// let pred = Tensor::from_vec(vec![0.9, 0.8, 0.2], false);
/// let target = Tensor::from_vec(vec![1.0, 0.0, 0.0], false);
///
/// let prec = metric.compute(&pred, &target);
/// assert_eq!(prec, 0.5);  // 1 TP / 2 predicted positives
/// ```
#[derive(Debug, Clone)]
pub struct Precision {
    threshold: f32,
}

impl Precision {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl Default for Precision {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Metric for Precision {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(predictions.len(), targets.len());

        let mut true_positives = 0;
        let mut predicted_positives = 0;

        for (&p, &t) in predictions.data().iter().zip(targets.data().iter()) {
            let pred_positive = p >= self.threshold;
            let actual_positive = t >= 0.5;

            if pred_positive {
                predicted_positives += 1;
                if actual_positive {
                    true_positives += 1;
                }
            }
        }

        if predicted_positives == 0 {
            return 0.0; // No predictions made
        }

        true_positives as f32 / predicted_positives as f32
    }

    fn name(&self) -> &str {
        "Precision"
    }
}

/// Recall metric (true positives / actual positives)
///
/// # Example
///
/// ```
/// use entrenar::train::{Recall, Metric};
/// use entrenar::Tensor;
///
/// let metric = Recall::new(0.5);
/// let pred = Tensor::from_vec(vec![0.9, 0.2, 0.8], false);
/// let target = Tensor::from_vec(vec![1.0, 1.0, 0.0], false);
///
/// let rec = metric.compute(&pred, &target);
/// assert_eq!(rec, 0.5);  // 1 TP / 2 actual positives
/// ```
#[derive(Debug, Clone)]
pub struct Recall {
    threshold: f32,
}

impl Recall {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl Default for Recall {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Metric for Recall {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(predictions.len(), targets.len());

        let mut true_positives = 0;
        let mut actual_positives = 0;

        for (&p, &t) in predictions.data().iter().zip(targets.data().iter()) {
            let pred_positive = p >= self.threshold;
            let actual_positive = t >= 0.5;

            if actual_positive {
                actual_positives += 1;
                if pred_positive {
                    true_positives += 1;
                }
            }
        }

        if actual_positives == 0 {
            return 0.0; // No positive samples
        }

        true_positives as f32 / actual_positives as f32
    }

    fn name(&self) -> &str {
        "Recall"
    }
}

/// F1 Score (harmonic mean of precision and recall)
///
/// F1 = 2 * (precision * recall) / (precision + recall)
///
/// # Example
///
/// ```
/// use entrenar::train::{F1Score, Metric};
/// use entrenar::Tensor;
///
/// let metric = F1Score::new(0.5);
/// let pred = Tensor::from_vec(vec![0.9, 0.8, 0.2, 0.1], false);
/// let target = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], false);
///
/// let f1 = metric.compute(&pred, &target);
/// assert!(f1 > 0.0 && f1 <= 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct F1Score {
    precision: Precision,
    recall: Recall,
}

impl F1Score {
    pub fn new(threshold: f32) -> Self {
        Self {
            precision: Precision::new(threshold),
            recall: Recall::new(threshold),
        }
    }
}

impl Default for F1Score {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Metric for F1Score {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        let precision = self.precision.compute(predictions, targets);
        let recall = self.recall.compute(predictions, targets);

        if precision + recall == 0.0 {
            return 0.0;
        }

        2.0 * (precision * recall) / (precision + recall)
    }

    fn name(&self) -> &str {
        "F1"
    }
}

// =============================================================================
// Regression Metrics
// =============================================================================

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

    fn name(&self) -> &str {
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

    fn name(&self) -> &str {
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

    fn name(&self) -> &str {
        "RMSE"
    }

    fn higher_is_better(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_perfect() {
        let metric = Accuracy::default();
        let pred = Tensor::from_vec(vec![0.9, 0.1, 0.8, 0.2], false);
        let target = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], false);

        let acc = metric.compute(&pred, &target);
        assert!((acc - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_accuracy_half() {
        let metric = Accuracy::default();
        let pred = Tensor::from_vec(vec![0.9, 0.9, 0.1, 0.1], false);
        let target = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], false);

        let acc = metric.compute(&pred, &target);
        assert!((acc - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_precision() {
        let metric = Precision::default();
        // 2 predicted positives (0.9, 0.8), 1 TP (0.9 -> 1.0)
        let pred = Tensor::from_vec(vec![0.9, 0.8, 0.2], false);
        let target = Tensor::from_vec(vec![1.0, 0.0, 0.0], false);

        let prec = metric.compute(&pred, &target);
        assert!((prec - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_recall() {
        let metric = Recall::default();
        // 2 actual positives, 1 correctly predicted
        let pred = Tensor::from_vec(vec![0.9, 0.2, 0.8], false);
        let target = Tensor::from_vec(vec![1.0, 1.0, 0.0], false);

        let rec = metric.compute(&pred, &target);
        assert!((rec - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_f1_score() {
        let metric = F1Score::default();
        let pred = Tensor::from_vec(vec![0.9, 0.8, 0.2, 0.1], false);
        let target = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], false);

        let f1 = metric.compute(&pred, &target);
        // Precision = 0.5 (1/2), Recall = 0.5 (1/2)
        // F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
        assert!((f1 - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_f1_perfect() {
        let metric = F1Score::default();
        let pred = Tensor::from_vec(vec![0.9, 0.1], false);
        let target = Tensor::from_vec(vec![1.0, 0.0], false);

        let f1 = metric.compute(&pred, &target);
        assert!((f1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_r2_perfect() {
        let metric = R2Score;
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);

        let r2 = metric.compute(&pred, &target);
        assert!((r2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_r2_mean_prediction() {
        let metric = R2Score;
        // Predicting mean of targets
        let pred = Tensor::from_vec(vec![2.0, 2.0, 2.0], false);
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);

        let r2 = metric.compute(&pred, &target);
        assert!(r2.abs() < 1e-5); // R² ≈ 0
    }

    #[test]
    fn test_mae() {
        let metric = MAE;
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false);

        let mae = metric.compute(&pred, &target);
        assert!((mae - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_mae_perfect() {
        let metric = MAE;
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);

        let mae = metric.compute(&pred, &target);
        assert!(mae < 1e-5);
    }

    #[test]
    fn test_rmse() {
        let metric = RMSE;
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        let target = Tensor::from_vec(vec![2.0, 3.0, 4.0], false);

        let rmse = metric.compute(&pred, &target);
        // MSE = mean([1, 1, 1]) = 1, RMSE = 1
        assert!((rmse - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rmse_perfect() {
        let metric = RMSE;
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);

        let rmse = metric.compute(&pred, &target);
        assert!(rmse < 1e-5);
    }

    #[test]
    fn test_higher_is_better() {
        assert!(Accuracy::default().higher_is_better());
        assert!(Precision::default().higher_is_better());
        assert!(Recall::default().higher_is_better());
        assert!(F1Score::default().higher_is_better());
        assert!(R2Score.higher_is_better());
        assert!(!MAE.higher_is_better());
        assert!(!RMSE.higher_is_better());
    }

    #[test]
    fn test_metric_names() {
        assert_eq!(Accuracy::default().name(), "Accuracy");
        assert_eq!(Precision::default().name(), "Precision");
        assert_eq!(Recall::default().name(), "Recall");
        assert_eq!(F1Score::default().name(), "F1");
        assert_eq!(R2Score.name(), "R²");
        assert_eq!(MAE.name(), "MAE");
        assert_eq!(RMSE.name(), "RMSE");
    }

    #[test]
    fn test_empty_input() {
        let metric = Accuracy::default();
        let pred = Tensor::from_vec(vec![], false);
        let target = Tensor::from_vec(vec![], false);

        let acc = metric.compute(&pred, &target);
        assert_eq!(acc, 0.0);
    }

    #[test]
    fn test_precision_no_predictions() {
        let metric = Precision::default();
        let pred = Tensor::from_vec(vec![0.1, 0.2, 0.3], false);
        let target = Tensor::from_vec(vec![1.0, 1.0, 1.0], false);

        let prec = metric.compute(&pred, &target);
        assert_eq!(prec, 0.0); // No positive predictions
    }

    #[test]
    fn test_recall_no_positives() {
        let metric = Recall::default();
        let pred = Tensor::from_vec(vec![0.9, 0.8, 0.7], false);
        let target = Tensor::from_vec(vec![0.0, 0.0, 0.0], false);

        let rec = metric.compute(&pred, &target);
        assert_eq!(rec, 0.0); // No actual positives
    }
}
