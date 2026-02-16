//! Classification metrics: Accuracy, Precision, Recall, F1

use crate::Tensor;

use super::Metric;

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
    pub(crate) threshold: f32,
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

        let correct = predictions
            .data()
            .iter()
            .zip(targets.data().iter())
            .filter(|(&p, &t)| {
                // Binary classification
                let pred_class = if p >= self.threshold { 1.0 } else { 0.0 };
                (pred_class - t).abs() < 0.5
            })
            .count() as f32;

        correct / predictions.len().max(1) as f32
    }

    fn name(&self) -> &'static str {
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
    pub(crate) threshold: f32,
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

    fn name(&self) -> &'static str {
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
    pub(crate) threshold: f32,
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

    fn name(&self) -> &'static str {
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

    fn name(&self) -> &'static str {
        "F1"
    }
}
