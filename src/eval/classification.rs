//! Classification metrics for model evaluation
//!
//! Provides multi-class classification metrics including:
//! - Confusion matrix computation
//! - Per-class precision, recall, F1
//! - Macro, micro, and weighted averaging
//! - sklearn-style classification reports

use std::fmt;

/// Averaging strategy for multi-class metrics
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Average {
    /// Calculate metrics for each label, return unweighted mean
    Macro,
    /// Calculate metrics globally by counting total TP, FP, FN
    Micro,
    /// Weighted mean by support (number of true instances per label)
    Weighted,
    /// Return metrics per class (no averaging) - used internally
    None,
}

/// Confusion matrix for multi-class classification
///
/// Element [i][j] represents count of samples with true label i predicted as j
#[derive(Clone, Debug)]
pub struct ConfusionMatrix {
    /// The matrix data: matrix[true_label][predicted_label] = count
    matrix: Vec<Vec<usize>>,
    /// Number of classes
    n_classes: usize,
    /// Class labels (indices)
    labels: Vec<usize>,
}

impl ConfusionMatrix {
    /// Create a new confusion matrix with given number of classes
    pub fn new(n_classes: usize) -> Self {
        Self {
            matrix: vec![vec![0; n_classes]; n_classes],
            n_classes,
            labels: (0..n_classes).collect(),
        }
    }

    /// Create from predictions and ground truth
    pub fn from_predictions(y_pred: &[usize], y_true: &[usize]) -> Self {
        assert_eq!(
            y_pred.len(),
            y_true.len(),
            "Predictions and targets must have same length"
        );

        // Determine number of classes
        let n_classes = y_pred
            .iter()
            .chain(y_true.iter())
            .max()
            .map_or(0, |&m| m + 1);

        let mut cm = Self::new(n_classes);

        for (&pred, &true_label) in y_pred.iter().zip(y_true.iter()) {
            if pred < n_classes && true_label < n_classes {
                cm.matrix[true_label][pred] += 1;
            }
        }

        cm
    }

    /// Get the raw matrix
    pub fn matrix(&self) -> &Vec<Vec<usize>> {
        &self.matrix
    }

    /// Get the class labels
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    /// Get number of classes
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Get element at [true_label][predicted_label]
    pub fn get(&self, true_label: usize, predicted_label: usize) -> usize {
        self.matrix[true_label][predicted_label]
    }

    /// Calculate true positives for a class
    pub fn true_positives(&self, class: usize) -> usize {
        self.matrix[class][class]
    }

    /// Calculate false positives for a class (predicted as class but wasn't)
    pub fn false_positives(&self, class: usize) -> usize {
        (0..self.n_classes)
            .filter(|&i| i != class)
            .map(|i| self.matrix[i][class])
            .sum()
    }

    /// Calculate false negatives for a class (was class but predicted differently)
    pub fn false_negatives(&self, class: usize) -> usize {
        (0..self.n_classes)
            .filter(|&j| j != class)
            .map(|j| self.matrix[class][j])
            .sum()
    }

    /// Calculate true negatives for a class
    pub fn true_negatives(&self, class: usize) -> usize {
        let total: usize = self.matrix.iter().flatten().sum();
        total
            - self.true_positives(class)
            - self.false_positives(class)
            - self.false_negatives(class)
    }

    /// Calculate support (total true instances) for a class
    pub fn support(&self, class: usize) -> usize {
        self.matrix[class].iter().sum()
    }

    /// Total number of samples
    pub fn total(&self) -> usize {
        self.matrix.iter().flatten().sum()
    }

    /// Calculate accuracy
    pub fn accuracy(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        let correct: usize = (0..self.n_classes).map(|i| self.matrix[i][i]).sum();
        correct as f64 / total as f64
    }
}

impl fmt::Display for ConfusionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Confusion Matrix:")?;

        // Header
        write!(f, "      ")?;
        for j in 0..self.n_classes {
            write!(f, "Pred {j} ")?;
        }
        writeln!(f)?;

        // Rows
        for i in 0..self.n_classes {
            write!(f, "True {i}")?;
            for j in 0..self.n_classes {
                write!(f, "{:>6} ", self.matrix[i][j])?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

/// Multi-class classification metrics
#[derive(Clone, Debug)]
pub struct MultiClassMetrics {
    /// Per-class precision
    pub precision: Vec<f64>,
    /// Per-class recall
    pub recall: Vec<f64>,
    /// Per-class F1 score
    pub f1: Vec<f64>,
    /// Per-class support (count)
    pub support: Vec<usize>,
    /// Number of classes
    pub n_classes: usize,
}

impl MultiClassMetrics {
    /// Compute metrics from confusion matrix
    pub fn from_confusion_matrix(cm: &ConfusionMatrix) -> Self {
        let n_classes = cm.n_classes();
        let mut precision = Vec::with_capacity(n_classes);
        let mut recall = Vec::with_capacity(n_classes);
        let mut f1 = Vec::with_capacity(n_classes);
        let mut support = Vec::with_capacity(n_classes);

        for class in 0..n_classes {
            let tp = cm.true_positives(class) as f64;
            let fp = cm.false_positives(class) as f64;
            let fn_ = cm.false_negatives(class) as f64;

            let p = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let r = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
            let f = if p + r > 0.0 {
                2.0 * p * r / (p + r)
            } else {
                0.0
            };

            precision.push(p);
            recall.push(r);
            f1.push(f);
            support.push(cm.support(class));
        }

        Self {
            precision,
            recall,
            f1,
            support,
            n_classes,
        }
    }

    /// Compute from predictions and ground truth
    pub fn from_predictions(y_pred: &[usize], y_true: &[usize]) -> Self {
        let cm = ConfusionMatrix::from_predictions(y_pred, y_true);
        Self::from_confusion_matrix(&cm)
    }

    /// Get averaged precision
    pub fn precision_avg(&self, average: Average) -> f64 {
        self.average_metric(&self.precision, average)
    }

    /// Get averaged recall
    pub fn recall_avg(&self, average: Average) -> f64 {
        self.average_metric(&self.recall, average)
    }

    /// Get averaged F1
    pub fn f1_avg(&self, average: Average) -> f64 {
        self.average_metric(&self.f1, average)
    }

    fn average_metric(&self, values: &[f64], average: Average) -> f64 {
        match average {
            Average::Macro => {
                if values.is_empty() {
                    0.0
                } else {
                    values.iter().sum::<f64>() / values.len() as f64
                }
            }
            Average::Micro => {
                // For micro-averaging, we need to recalculate from totals
                // This is a simplified version - returns macro for now
                // TODO: Implement proper micro-averaging
                self.average_metric(values, Average::Macro)
            }
            Average::Weighted => {
                let total_support: usize = self.support.iter().sum();
                if total_support == 0 {
                    return 0.0;
                }
                values
                    .iter()
                    .zip(self.support.iter())
                    .map(|(&v, &s)| v * s as f64)
                    .sum::<f64>()
                    / total_support as f64
            }
            Average::None => {
                // Return macro as default for single value
                self.average_metric(values, Average::Macro)
            }
        }
    }
}

/// Compute confusion matrix from predictions and ground truth
///
/// # Arguments
/// * `y_pred` - Predicted class labels
/// * `y_true` - Ground truth class labels
///
/// # Returns
/// A ConfusionMatrix where element [i][j] is count of true label i predicted as j
///
/// # Example
/// ```ignore
/// use entrenar::eval::confusion_matrix;
///
/// let y_pred = vec![0, 1, 1, 2, 0];
/// let y_true = vec![0, 1, 0, 2, 1];
/// let cm = confusion_matrix(&y_pred, &y_true);
///
/// assert_eq!(cm.get(0, 0), 1);  // True 0, predicted 0
/// assert_eq!(cm.get(0, 1), 1);  // True 0, predicted 1
/// ```
pub fn confusion_matrix(y_pred: &[usize], y_true: &[usize]) -> ConfusionMatrix {
    ConfusionMatrix::from_predictions(y_pred, y_true)
}

/// Generate sklearn-style classification report
///
/// # Arguments
/// * `y_pred` - Predicted class labels
/// * `y_true` - Ground truth class labels
///
/// # Returns
/// A formatted string containing per-class and overall metrics
///
/// # Example
/// ```ignore
/// use entrenar::eval::classification_report;
///
/// let report = classification_report(&y_pred, &y_true);
/// println!("{}", report);
/// ```
pub fn classification_report(y_pred: &[usize], y_true: &[usize]) -> String {
    let cm = ConfusionMatrix::from_predictions(y_pred, y_true);
    let metrics = MultiClassMetrics::from_confusion_matrix(&cm);

    let mut report = String::new();

    // Header
    report.push_str(&format!(
        "{:>12} {:>10} {:>10} {:>10} {:>10}\n",
        "", "precision", "recall", "f1-score", "support"
    ));
    report.push_str(&"-".repeat(54));
    report.push('\n');

    // Per-class metrics
    for class in 0..metrics.n_classes {
        report.push_str(&format!(
            "{:>12} {:>10.2} {:>10.2} {:>10.2} {:>10}\n",
            format!("Class {}", class),
            metrics.precision[class],
            metrics.recall[class],
            metrics.f1[class],
            metrics.support[class]
        ));
    }

    report.push_str(&"-".repeat(54));
    report.push('\n');

    // Averages
    let total_support: usize = metrics.support.iter().sum();

    report.push_str(&format!(
        "{:>12} {:>10.2} {:>10.2} {:>10.2} {:>10}\n",
        "macro avg",
        metrics.precision_avg(Average::Macro),
        metrics.recall_avg(Average::Macro),
        metrics.f1_avg(Average::Macro),
        total_support
    ));

    report.push_str(&format!(
        "{:>12} {:>10.2} {:>10.2} {:>10.2} {:>10}\n",
        "weighted avg",
        metrics.precision_avg(Average::Weighted),
        metrics.recall_avg(Average::Weighted),
        metrics.f1_avg(Average::Weighted),
        total_support
    ));

    report.push_str(&format!("\nAccuracy: {:.4}\n", cm.accuracy()));

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusion_matrix_basic() {
        let y_pred = vec![0, 1, 1, 2, 0, 1];
        let y_true = vec![0, 1, 0, 2, 0, 2];
        let cm = confusion_matrix(&y_pred, &y_true);

        assert_eq!(cm.n_classes(), 3);
        assert_eq!(cm.get(0, 0), 2); // True 0, predicted 0
        assert_eq!(cm.get(0, 1), 1); // True 0, predicted 1
        assert_eq!(cm.get(1, 1), 1); // True 1, predicted 1
        assert_eq!(cm.get(2, 1), 1); // True 2, predicted 1
        assert_eq!(cm.get(2, 2), 1); // True 2, predicted 2
    }

    #[test]
    fn test_confusion_matrix_perfect() {
        let y_pred = vec![0, 1, 2, 0, 1, 2];
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let cm = confusion_matrix(&y_pred, &y_true);

        assert_eq!(cm.accuracy(), 1.0);
        assert_eq!(cm.get(0, 0), 2);
        assert_eq!(cm.get(1, 1), 2);
        assert_eq!(cm.get(2, 2), 2);
    }

    #[test]
    fn test_confusion_matrix_tp_fp_fn() {
        let y_pred = vec![1, 1, 0, 1];
        let y_true = vec![1, 0, 0, 1];
        let cm = confusion_matrix(&y_pred, &y_true);

        // For class 1:
        // TP = 2 (predicted 1, was 1)
        // FP = 1 (predicted 1, was 0)
        // FN = 0 (was 1, predicted 0)
        assert_eq!(cm.true_positives(1), 2);
        assert_eq!(cm.false_positives(1), 1);
        assert_eq!(cm.false_negatives(1), 0);

        // For class 0:
        // TP = 1 (predicted 0, was 0)
        // FP = 0 (predicted 0, was 1)
        // FN = 1 (was 0, predicted 1)
        assert_eq!(cm.true_positives(0), 1);
        assert_eq!(cm.false_positives(0), 0);
        assert_eq!(cm.false_negatives(0), 1);
    }

    #[test]
    fn test_multiclass_metrics() {
        let y_pred = vec![0, 1, 1, 2, 0];
        let y_true = vec![0, 1, 0, 2, 1];
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);

        // Class 0: TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1=0.5
        assert!((metrics.precision[0] - 0.5).abs() < 1e-6);
        assert!((metrics.recall[0] - 0.5).abs() < 1e-6);

        // Class 1: TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1=0.5
        assert!((metrics.precision[1] - 0.5).abs() < 1e-6);
        assert!((metrics.recall[1] - 0.5).abs() < 1e-6);

        // Class 2: TP=1, FP=0, FN=0 -> P=1, R=1, F1=1
        assert!((metrics.precision[2] - 1.0).abs() < 1e-6);
        assert!((metrics.recall[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_macro_average() {
        let y_pred = vec![0, 1, 1, 2, 0];
        let y_true = vec![0, 1, 0, 2, 1];
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);

        // Macro average: (0.5 + 0.5 + 1.0) / 3 = 0.667
        let macro_f1 = metrics.f1_avg(Average::Macro);
        assert!((macro_f1 - 0.6666666).abs() < 0.01);
    }

    #[test]
    fn test_weighted_average() {
        let y_pred = vec![0, 1, 1, 2, 0];
        let y_true = vec![0, 1, 0, 2, 1];
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);

        // Support: class 0 = 2, class 1 = 2, class 2 = 1
        // Weighted F1: (0.5*2 + 0.5*2 + 1.0*1) / 5 = 3/5 = 0.6
        let weighted_f1 = metrics.f1_avg(Average::Weighted);
        assert!((weighted_f1 - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_classification_report() {
        let y_pred = vec![0, 1, 1, 2, 0, 1];
        let y_true = vec![0, 1, 0, 2, 0, 2];
        let report = classification_report(&y_pred, &y_true);

        assert!(report.contains("precision"));
        assert!(report.contains("recall"));
        assert!(report.contains("f1-score"));
        assert!(report.contains("support"));
        assert!(report.contains("macro avg"));
        assert!(report.contains("weighted avg"));
        assert!(report.contains("Accuracy"));
    }

    #[test]
    fn test_empty_input() {
        let y_pred: Vec<usize> = vec![];
        let y_true: Vec<usize> = vec![];
        let cm = confusion_matrix(&y_pred, &y_true);

        assert_eq!(cm.n_classes(), 0);
        assert_eq!(cm.accuracy(), 0.0);
    }

    #[test]
    fn test_binary_classification() {
        let y_pred = vec![0, 1, 1, 0, 1, 0];
        let y_true = vec![0, 1, 0, 0, 1, 1];
        let cm = confusion_matrix(&y_pred, &y_true);

        assert_eq!(cm.n_classes(), 2);

        // Class 0: TP=2, FP=1, FN=1
        // Class 1: TP=2, FP=1, FN=1
        assert_eq!(cm.true_positives(0), 2);
        assert_eq!(cm.true_positives(1), 2);
    }

    #[test]
    fn test_support() {
        let y_pred = vec![0, 1, 1, 2, 0];
        let y_true = vec![0, 1, 0, 2, 1];
        let cm = confusion_matrix(&y_pred, &y_true);

        // Support is count of true instances per class
        assert_eq!(cm.support(0), 2); // Two samples with true label 0
        assert_eq!(cm.support(1), 2); // Two samples with true label 1
        assert_eq!(cm.support(2), 1); // One sample with true label 2
    }

    #[test]
    fn test_display() {
        let y_pred = vec![0, 1, 0];
        let y_true = vec![0, 1, 1];
        let cm = confusion_matrix(&y_pred, &y_true);

        let display = format!("{cm}");
        assert!(display.contains("Confusion Matrix"));
        assert!(display.contains("Pred"));
        assert!(display.contains("True"));
    }

    // =========================================================================
    // sklearn Parity Tests (APR-073 Section 10.1)
    //
    // These tests verify that our metrics match sklearn reference values
    // to within 1e-6 precision.
    //
    // Reference values computed with sklearn 1.4.0:
    // ```python
    // from sklearn.metrics import (accuracy_score, precision_score,
    //                              recall_score, f1_score, confusion_matrix)
    // ```
    // =========================================================================

    #[test]
    fn test_sklearn_parity_accuracy() {
        // sklearn: accuracy_score([0, 0, 1, 1, 2, 2, 0, 1, 2],
        //                         [0, 1, 1, 2, 2, 0, 0, 1, 2]) = 0.6666666666666666
        let y_true = vec![0, 0, 1, 1, 2, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 1, 2, 2, 0, 0, 1, 2];

        let cm = confusion_matrix(&y_pred, &y_true);
        let acc = cm.accuracy();

        assert!(
            (acc - 0.6666666666666666).abs() < 1e-6,
            "Accuracy {acc} does not match sklearn reference 0.6666666666666666"
        );
    }

    #[test]
    fn test_sklearn_parity_precision_macro() {
        // sklearn: precision_score(..., average='macro') = 0.6666666666666666
        let y_true = vec![0, 0, 1, 1, 2, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 1, 2, 2, 0, 0, 1, 2];

        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);
        let p_macro = metrics.precision_avg(Average::Macro);

        assert!(
            (p_macro - 0.6666666666666666).abs() < 1e-6,
            "Macro precision {p_macro} does not match sklearn reference"
        );
    }

    #[test]
    fn test_sklearn_parity_recall_macro() {
        // sklearn: recall_score(..., average='macro') = 0.6666666666666666
        let y_true = vec![0, 0, 1, 1, 2, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 1, 2, 2, 0, 0, 1, 2];

        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);
        let r_macro = metrics.recall_avg(Average::Macro);

        assert!(
            (r_macro - 0.6666666666666666).abs() < 1e-6,
            "Macro recall {r_macro} does not match sklearn reference"
        );
    }

    #[test]
    fn test_sklearn_parity_f1_macro() {
        // sklearn: f1_score(..., average='macro') = 0.6666666666666666
        let y_true = vec![0, 0, 1, 1, 2, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 1, 2, 2, 0, 0, 1, 2];

        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);
        let f1_macro = metrics.f1_avg(Average::Macro);

        assert!(
            (f1_macro - 0.6666666666666666).abs() < 1e-6,
            "Macro F1 {f1_macro} does not match sklearn reference"
        );
    }

    #[test]
    fn test_sklearn_parity_micro_averages() {
        // sklearn: For this dataset, micro = macro = 0.6666666666666666
        let y_true = vec![0, 0, 1, 1, 2, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 1, 2, 2, 0, 0, 1, 2];

        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);

        let p_micro = metrics.precision_avg(Average::Micro);
        let r_micro = metrics.recall_avg(Average::Micro);
        let f1_micro = metrics.f1_avg(Average::Micro);

        assert!(
            (p_micro - 0.6666666666666666).abs() < 1e-6,
            "Micro precision {p_micro} does not match sklearn reference"
        );
        assert!(
            (r_micro - 0.6666666666666666).abs() < 1e-6,
            "Micro recall {r_micro} does not match sklearn reference"
        );
        assert!(
            (f1_micro - 0.6666666666666666).abs() < 1e-6,
            "Micro F1 {f1_micro} does not match sklearn reference"
        );
    }

    #[test]
    fn test_sklearn_parity_weighted_averages() {
        // sklearn: weighted averages = 0.6666666666666666 for balanced classes
        let y_true = vec![0, 0, 1, 1, 2, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 1, 2, 2, 0, 0, 1, 2];

        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);

        let p_weighted = metrics.precision_avg(Average::Weighted);
        let r_weighted = metrics.recall_avg(Average::Weighted);
        let f1_weighted = metrics.f1_avg(Average::Weighted);

        assert!(
            (p_weighted - 0.6666666666666666).abs() < 1e-6,
            "Weighted precision {p_weighted} does not match sklearn reference"
        );
        assert!(
            (r_weighted - 0.6666666666666666).abs() < 1e-6,
            "Weighted recall {r_weighted} does not match sklearn reference"
        );
        assert!(
            (f1_weighted - 0.6666666666666666).abs() < 1e-6,
            "Weighted F1 {f1_weighted} does not match sklearn reference"
        );
    }

    #[test]
    fn test_sklearn_parity_imbalanced() {
        // Test with imbalanced classes
        // y_true = [0, 0, 0, 0, 0, 1, 1, 2]
        // y_pred = [0, 0, 0, 1, 1, 1, 0, 2]
        //
        // Manual calculation:
        // Class 0: TP=3, FP=1, FN=2 → P=0.75, R=0.6, F1=0.6667
        // Class 1: TP=1, FP=2, FN=1 → P=0.333, R=0.5, F1=0.4
        // Class 2: TP=1, FP=0, FN=0 → P=1, R=1, F1=1
        // Macro F1 = (0.6667 + 0.4 + 1.0) / 3 = 0.6889
        let y_true = vec![0, 0, 0, 0, 0, 1, 1, 2];
        let y_pred = vec![0, 0, 0, 1, 1, 1, 0, 2];

        let cm = confusion_matrix(&y_pred, &y_true);
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);

        // Accuracy: 5 correct out of 8
        assert!(
            (cm.accuracy() - 0.625).abs() < 1e-6,
            "Accuracy {} does not match reference 0.625",
            cm.accuracy()
        );

        // Macro F1: (0.6667 + 0.4 + 1.0) / 3 = 0.6889
        let f1_macro = metrics.f1_avg(Average::Macro);
        assert!(
            (f1_macro - 0.6888888888888888).abs() < 1e-6,
            "Macro F1 {f1_macro} does not match reference 0.6889"
        );
    }

    #[test]
    fn test_sklearn_parity_binary() {
        // Binary classification parity
        // y_true = [0, 0, 1, 1, 0, 1, 0, 1]
        // y_pred = [0, 1, 1, 0, 0, 1, 1, 1]
        //
        // Manual calculation:
        // Class 0: TP=2, FP=1, FN=2 → P=2/3=0.6667, R=2/4=0.5, F1=0.5714
        // Class 1: TP=3, FP=2, FN=1 → P=3/5=0.6, R=3/4=0.75, F1=0.6667
        // Macro F1 = (0.5714 + 0.6667) / 2 = 0.6190
        let y_true = vec![0, 0, 1, 1, 0, 1, 0, 1];
        let y_pred = vec![0, 1, 1, 0, 0, 1, 1, 1];

        let cm = confusion_matrix(&y_pred, &y_true);
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);

        // Accuracy: 5 correct out of 8
        assert!(
            (cm.accuracy() - 0.625).abs() < 1e-6,
            "Accuracy {} does not match reference 0.625",
            cm.accuracy()
        );

        // Macro F1: (0.5714 + 0.6667) / 2 = 0.6190
        let f1_macro = metrics.f1_avg(Average::Macro);
        assert!(
            (f1_macro - 0.6190476190476191).abs() < 1e-6,
            "Macro F1 {f1_macro} does not match reference 0.6190"
        );
    }
}
