//! Basic tests for classification metrics
//!
//! Tests for confusion matrix, multiclass metrics, and core functionality.

#[cfg(test)]
mod tests {
    use crate::eval::classification::{
        classification_report, confusion_matrix, Average, ConfusionMatrix, MultiClassMetrics,
    };

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

    #[test]
    fn test_average_enum_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Average::Macro);
        set.insert(Average::Micro);
        set.insert(Average::Weighted);
        set.insert(Average::None);
        set.insert(Average::Macro); // Duplicate
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn test_average_enum_clone_copy() {
        let avg = Average::Weighted;
        let copied = avg;
        let cloned = avg;
        assert_eq!(avg, copied);
        assert_eq!(avg, cloned);
    }

    #[test]
    fn test_confusion_matrix_labels() {
        let cm = ConfusionMatrix::new(3);
        let labels = cm.labels();
        assert_eq!(labels, &[0, 1, 2]);
    }

    #[test]
    fn test_confusion_matrix_matrix_accessor() {
        let y_pred = vec![0, 1, 0];
        let y_true = vec![0, 1, 1];
        let cm = confusion_matrix(&y_pred, &y_true);
        let matrix = cm.matrix();
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0][0], 1); // True 0, pred 0
        assert_eq!(matrix[1][0], 1); // True 1, pred 0
    }

    #[test]
    fn test_confusion_matrix_total() {
        let y_pred = vec![0, 1, 1, 2, 0];
        let y_true = vec![0, 1, 0, 2, 1];
        let cm = confusion_matrix(&y_pred, &y_true);
        assert_eq!(cm.total(), 5);
    }

    #[test]
    fn test_confusion_matrix_true_negatives() {
        // Simple binary case for clarity
        let y_pred = vec![1, 1, 0, 0];
        let y_true = vec![1, 0, 0, 1];
        let cm = confusion_matrix(&y_pred, &y_true);

        // For class 0: TN = samples that are not 0 and not predicted as 0
        // True 1, pred 1: 1 case = TN for class 0
        let tn_0 = cm.true_negatives(0);
        assert_eq!(tn_0, 1);

        // For class 1: TN = samples that are not 1 and not predicted as 1
        // True 0, pred 0: 1 case = TN for class 1
        let tn_1 = cm.true_negatives(1);
        assert_eq!(tn_1, 1);
    }

    #[test]
    fn test_confusion_matrix_clone() {
        let y_pred = vec![0, 1, 0];
        let y_true = vec![0, 1, 1];
        let cm = confusion_matrix(&y_pred, &y_true);
        let cloned = cm.clone();

        assert_eq!(cm.n_classes(), cloned.n_classes());
        assert_eq!(cm.get(0, 0), cloned.get(0, 0));
    }

    #[test]
    fn test_multiclass_metrics_clone() {
        let y_pred = vec![0, 1, 1, 2, 0];
        let y_true = vec![0, 1, 0, 2, 1];
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);
        let cloned = metrics.clone();

        assert_eq!(metrics.n_classes, cloned.n_classes);
        assert_eq!(metrics.precision.len(), cloned.precision.len());
    }
}
