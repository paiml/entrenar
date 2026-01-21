//! Model Evaluator for standardized evaluation and comparison
//!
//! Provides ModelEvaluator for running comprehensive evaluations,
//! comparing multiple models, and generating leaderboards.

use super::classification::{confusion_matrix, Average, MultiClassMetrics};
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

/// Available evaluation metrics
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Metric {
    // Classification
    /// Classification accuracy
    Accuracy,
    /// Precision with averaging strategy
    Precision(Average),
    /// Recall with averaging strategy
    Recall(Average),
    /// F1 score with averaging strategy
    F1(Average),
    // Regression
    /// R² coefficient of determination
    R2,
    /// Mean Squared Error
    MSE,
    /// Mean Absolute Error
    MAE,
    /// Root Mean Squared Error
    RMSE,
    // Clustering
    /// Silhouette score
    Silhouette,
    /// Inertia
    Inertia,
}

impl Metric {
    /// Whether higher values are better for this metric
    pub fn higher_is_better(&self) -> bool {
        !matches!(
            self,
            Metric::MSE | Metric::MAE | Metric::RMSE | Metric::Inertia
        )
    }

    /// Get metric name as string
    pub fn name(&self) -> &'static str {
        match self {
            Metric::Accuracy => "Accuracy",
            Metric::Precision(_) => "Precision",
            Metric::Recall(_) => "Recall",
            Metric::F1(_) => "F1",
            Metric::R2 => "R²",
            Metric::MSE => "MSE",
            Metric::MAE => "MAE",
            Metric::RMSE => "RMSE",
            Metric::Silhouette => "Silhouette",
            Metric::Inertia => "Inertia",
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Metric::Precision(avg) => write!(f, "Precision({avg:?})"),
            Metric::Recall(avg) => write!(f, "Recall({avg:?})"),
            Metric::F1(avg) => write!(f, "F1({avg:?})"),
            _ => write!(f, "{}", self.name()),
        }
    }
}

/// Configuration for model evaluation
#[derive(Clone, Debug)]
pub struct EvalConfig {
    /// Metrics to compute
    pub metrics: Vec<Metric>,
    /// Number of cross-validation folds (0 = no CV)
    pub cv_folds: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Parallel evaluation (requires rayon feature)
    pub parallel: bool,
    /// Enable tracing (for renacer integration)
    pub trace_enabled: bool,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            metrics: vec![Metric::Accuracy, Metric::F1(Average::Weighted)],
            cv_folds: 0,
            seed: 42,
            parallel: false,
            trace_enabled: false,
        }
    }
}

/// Model evaluation results
#[derive(Clone, Debug)]
pub struct EvalResult {
    /// Name of the model
    pub model_name: String,
    /// Computed metric scores
    pub scores: HashMap<Metric, f64>,
    /// Cross-validation scores per fold (if CV enabled)
    pub cv_scores: Option<Vec<f64>>,
    /// Mean CV score
    pub cv_mean: Option<f64>,
    /// CV score standard deviation
    pub cv_std: Option<f64>,
    /// Inference time in milliseconds
    pub inference_time_ms: f64,
    /// Optional trace ID for observability
    pub trace_id: Option<String>,
}

impl EvalResult {
    /// Create new eval result
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            scores: HashMap::new(),
            cv_scores: None,
            cv_mean: None,
            cv_std: None,
            inference_time_ms: 0.0,
            trace_id: None,
        }
    }

    /// Get score for a specific metric
    pub fn get_score(&self, metric: Metric) -> Option<f64> {
        self.scores.get(&metric).copied()
    }

    /// Add a score
    pub fn add_score(&mut self, metric: Metric, score: f64) {
        self.scores.insert(metric, score);
    }
}

impl fmt::Display for EvalResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model: {}", self.model_name)?;
        writeln!(f, "Metrics:")?;
        for (metric, score) in &self.scores {
            writeln!(f, "  {metric}: {score:.4}")?;
        }
        writeln!(f, "Inference time: {:.2}ms", self.inference_time_ms)?;
        Ok(())
    }
}

/// Leaderboard for comparing multiple models
#[derive(Clone, Debug)]
pub struct Leaderboard {
    /// Evaluation results for each model
    pub results: Vec<EvalResult>,
    /// Primary metric for ranking
    pub primary_metric: Metric,
}

impl Leaderboard {
    /// Create a new leaderboard
    pub fn new(primary_metric: Metric) -> Self {
        Self {
            results: Vec::new(),
            primary_metric,
        }
    }

    /// Add evaluation result
    pub fn add(&mut self, result: EvalResult) {
        self.results.push(result);
        self.sort();
    }

    /// Sort by primary metric
    pub fn sort(&mut self) {
        let higher_is_better = self.primary_metric.higher_is_better();
        self.results.sort_by(|a, b| {
            let score_a = a.get_score(self.primary_metric).unwrap_or(0.0);
            let score_b = b.get_score(self.primary_metric).unwrap_or(0.0);
            if higher_is_better {
                score_b
                    .partial_cmp(&score_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        });
    }

    /// Sort by a specific metric
    pub fn sort_by(&mut self, metric: Metric) {
        let higher_is_better = metric.higher_is_better();
        self.results.sort_by(|a, b| {
            let score_a = a.get_score(metric).unwrap_or(0.0);
            let score_b = b.get_score(metric).unwrap_or(0.0);
            if higher_is_better {
                score_b
                    .partial_cmp(&score_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        });
    }

    /// Get best model by primary metric
    pub fn best(&self) -> Option<&EvalResult> {
        self.results.first()
    }

    /// Print formatted leaderboard to stdout (Mieruka - visual control)
    pub fn print(&self) {
        println!("{self}");
    }

    /// Export as markdown table
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        // Collect all metrics
        let metrics: Vec<Metric> = if let Some(first) = self.results.first() {
            first.scores.keys().copied().collect()
        } else {
            return md;
        };

        // Header
        md.push_str("| Model |");
        for metric in &metrics {
            md.push_str(&format!(" {metric} |"));
        }
        md.push_str(" Inference (ms) |\n");

        // Separator
        md.push_str("|-------|");
        for _ in &metrics {
            md.push_str("----------|");
        }
        md.push_str("---------------|\n");

        // Rows
        for result in &self.results {
            md.push_str(&format!("| {} |", result.model_name));
            for metric in &metrics {
                let score = result.get_score(*metric).unwrap_or(0.0);
                md.push_str(&format!(" {score:.4} |"));
            }
            md.push_str(&format!(" {:.2} |\n", result.inference_time_ms));
        }

        md
    }
}

impl fmt::Display for Leaderboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.results.is_empty() {
            return writeln!(f, "Leaderboard: (empty)");
        }

        // Collect all metrics
        let metrics: Vec<Metric> = if let Some(first) = self.results.first() {
            first.scores.keys().copied().collect()
        } else {
            return Ok(());
        };

        // Calculate column widths
        let model_width = self
            .results
            .iter()
            .map(|r| r.model_name.len())
            .max()
            .unwrap_or(5)
            .max(5);

        // Header
        write!(f, "┌{:─<width$}┬", "", width = model_width + 2)?;
        for _ in &metrics {
            write!(f, "{:─<12}┬", "")?;
        }
        writeln!(f, "{:─<15}┐", "")?;

        write!(f, "│ {:width$} │", "Model", width = model_width)?;
        for metric in &metrics {
            write!(f, " {:>10} │", metric.name())?;
        }
        writeln!(f, " Inference (ms)│")?;

        // Separator
        write!(f, "├{:─<width$}┼", "", width = model_width + 2)?;
        for _ in &metrics {
            write!(f, "{:─<12}┼", "")?;
        }
        writeln!(f, "{:─<15}┤", "")?;

        // Rows
        for result in &self.results {
            write!(f, "│ {:width$} │", result.model_name, width = model_width)?;
            for metric in &metrics {
                let score = result.get_score(*metric).unwrap_or(0.0);
                write!(f, " {score:>10.4} │")?;
            }
            writeln!(f, " {:>13.2} │", result.inference_time_ms)?;
        }

        // Footer
        write!(f, "└{:─<width$}┴", "", width = model_width + 2)?;
        for _ in &metrics {
            write!(f, "{:─<12}┴", "")?;
        }
        writeln!(f, "{:─<15}┘", "")?;

        Ok(())
    }
}

/// K-Fold cross-validation splitter
#[derive(Clone, Debug)]
pub struct KFold {
    n_splits: usize,
    shuffle: bool,
    seed: u64,
}

impl KFold {
    /// Create a new KFold splitter
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            shuffle: true,
            seed: 42,
        }
    }

    /// Set random seed for shuffling
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Disable shuffling
    pub fn without_shuffle(mut self) -> Self {
        self.shuffle = false;
        self
    }

    /// Generate train/test indices for each fold
    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut indices: Vec<usize> = (0..n_samples).collect();

        if self.shuffle {
            // Simple LCG-based shuffle for reproducibility
            let mut rng_state = self.seed;
            for i in (1..n_samples).rev() {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = (rng_state >> 33) as usize % (i + 1);
                indices.swap(i, j);
            }
        }

        let fold_size = n_samples / self.n_splits;
        let remainder = n_samples % self.n_splits;

        let mut folds = Vec::with_capacity(self.n_splits);
        let mut start = 0;

        for i in 0..self.n_splits {
            let extra = usize::from(i < remainder);
            let end = start + fold_size + extra;

            let test_indices: Vec<usize> = indices[start..end].to_vec();
            let train_indices: Vec<usize> = indices[..start]
                .iter()
                .chain(indices[end..].iter())
                .copied()
                .collect();

            folds.push((train_indices, test_indices));
            start = end;
        }

        folds
    }
}

/// Model Evaluator for running evaluations
pub struct ModelEvaluator {
    config: EvalConfig,
}

impl ModelEvaluator {
    /// Create a new evaluator with given configuration
    pub fn new(config: EvalConfig) -> Self {
        Self { config }
    }

    /// Evaluate classification with cross-validation
    ///
    /// Takes a prediction function that maps (train_indices, test_indices) to predictions.
    /// Returns EvalResult with cv_scores, cv_mean, and cv_std populated.
    pub fn evaluate_cv<F>(
        &self,
        model_name: impl Into<String>,
        y_true: &[usize],
        predict_fn: F,
    ) -> Result<EvalResult>
    where
        F: Fn(&[usize], &[usize]) -> Vec<usize>,
    {
        if self.config.cv_folds == 0 {
            return Err(Error::InvalidParameter(
                "cv_folds must be > 0 for cross-validation".into(),
            ));
        }

        let start = Instant::now();
        let kfold = KFold::new(self.config.cv_folds).with_seed(self.config.seed);
        let folds = kfold.split(y_true.len());

        let mut fold_scores: Vec<f64> = Vec::with_capacity(self.config.cv_folds);

        // Get primary metric for CV scoring
        let primary_metric = self
            .config
            .metrics
            .first()
            .copied()
            .unwrap_or(Metric::Accuracy);

        for (train_idx, test_idx) in &folds {
            // Get predictions for this fold
            let predictions = predict_fn(train_idx, test_idx);

            // Get test labels
            let test_labels: Vec<usize> = test_idx.iter().map(|&i| y_true[i]).collect();

            // Compute primary metric for this fold
            let cm = confusion_matrix(&predictions, &test_labels);
            let metrics = MultiClassMetrics::from_confusion_matrix(&cm);

            let score = match primary_metric {
                Metric::Accuracy => cm.accuracy(),
                Metric::Precision(avg) => metrics.precision_avg(avg),
                Metric::Recall(avg) => metrics.recall_avg(avg),
                Metric::F1(avg) => metrics.f1_avg(avg),
                _ => cm.accuracy(),
            };

            fold_scores.push(score);
        }

        // Compute mean and std
        let cv_mean = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
        let cv_std = if fold_scores.len() > 1 {
            let variance = fold_scores
                .iter()
                .map(|s| (s - cv_mean).powi(2))
                .sum::<f64>()
                / (fold_scores.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let mut result = EvalResult::new(model_name);
        result.cv_scores = Some(fold_scores);
        result.cv_mean = Some(cv_mean);
        result.cv_std = Some(cv_std);
        result.add_score(primary_metric, cv_mean);
        result.inference_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(result)
    }

    /// Evaluate classification model with predictions and ground truth
    ///
    /// # Arguments
    /// * `model_name` - Name for the model in results
    /// * `y_pred` - Predicted class labels
    /// * `y_true` - Ground truth class labels
    ///
    /// # Returns
    /// EvalResult containing computed metrics
    pub fn evaluate_classification(
        &self,
        model_name: impl Into<String>,
        y_pred: &[usize],
        y_true: &[usize],
    ) -> Result<EvalResult> {
        if y_pred.len() != y_true.len() {
            return Err(Error::InvalidParameter(
                "Predictions and targets must have same length".into(),
            ));
        }

        let start = Instant::now();

        let cm = confusion_matrix(y_pred, y_true);
        let metrics = MultiClassMetrics::from_confusion_matrix(&cm);

        let mut result = EvalResult::new(model_name);

        for metric in &self.config.metrics {
            let score = match metric {
                Metric::Accuracy => cm.accuracy(),
                Metric::Precision(avg) => metrics.precision_avg(*avg),
                Metric::Recall(avg) => metrics.recall_avg(*avg),
                Metric::F1(avg) => metrics.f1_avg(*avg),
                _ => continue, // Skip non-classification metrics
            };
            result.add_score(*metric, score);
        }

        result.inference_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(result)
    }

    /// Compare multiple classification models
    ///
    /// # Arguments
    /// * `models` - Slice of (name, predictions) tuples
    /// * `y_true` - Ground truth labels
    ///
    /// # Returns
    /// Leaderboard with all models ranked by primary metric
    pub fn compare_classification(
        &self,
        models: &[(&str, &[usize])],
        y_true: &[usize],
    ) -> Result<Leaderboard> {
        let primary = self
            .config
            .metrics
            .first()
            .copied()
            .unwrap_or(Metric::Accuracy);
        let mut leaderboard = Leaderboard::new(primary);

        for (name, y_pred) in models {
            let result = self.evaluate_classification(*name, y_pred, y_true)?;
            leaderboard.add(result);
        }

        Ok(leaderboard)
    }

    /// Get the configuration
    pub fn config(&self) -> &EvalConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_config_default() {
        let config = EvalConfig::default();
        assert_eq!(config.metrics.len(), 2);
        assert_eq!(config.cv_folds, 0);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_metric_higher_is_better() {
        assert!(Metric::Accuracy.higher_is_better());
        assert!(Metric::F1(Average::Macro).higher_is_better());
        assert!(Metric::R2.higher_is_better());
        assert!(!Metric::MSE.higher_is_better());
        assert!(!Metric::MAE.higher_is_better());
        assert!(!Metric::RMSE.higher_is_better());
    }

    #[test]
    fn test_eval_result() {
        let mut result = EvalResult::new("TestModel");
        result.add_score(Metric::Accuracy, 0.95);
        result.add_score(Metric::F1(Average::Weighted), 0.93);

        assert_eq!(result.get_score(Metric::Accuracy), Some(0.95));
        assert_eq!(result.get_score(Metric::F1(Average::Weighted)), Some(0.93));
        assert_eq!(result.get_score(Metric::R2), None);
    }

    #[test]
    fn test_leaderboard() {
        let mut leaderboard = Leaderboard::new(Metric::Accuracy);

        let mut result1 = EvalResult::new("Model A");
        result1.add_score(Metric::Accuracy, 0.85);
        leaderboard.add(result1);

        let mut result2 = EvalResult::new("Model B");
        result2.add_score(Metric::Accuracy, 0.92);
        leaderboard.add(result2);

        let mut result3 = EvalResult::new("Model C");
        result3.add_score(Metric::Accuracy, 0.88);
        leaderboard.add(result3);

        // Best should be Model B (highest accuracy)
        assert_eq!(leaderboard.best().unwrap().model_name, "Model B");

        // Order should be B, C, A
        assert_eq!(leaderboard.results[0].model_name, "Model B");
        assert_eq!(leaderboard.results[1].model_name, "Model C");
        assert_eq!(leaderboard.results[2].model_name, "Model A");
    }

    #[test]
    fn test_leaderboard_lower_is_better() {
        let mut leaderboard = Leaderboard::new(Metric::MSE);

        let mut result1 = EvalResult::new("Model A");
        result1.add_score(Metric::MSE, 0.1);
        leaderboard.add(result1);

        let mut result2 = EvalResult::new("Model B");
        result2.add_score(Metric::MSE, 0.05);
        leaderboard.add(result2);

        // Best should be Model B (lowest MSE)
        assert_eq!(leaderboard.best().unwrap().model_name, "Model B");
    }

    #[test]
    fn test_evaluate_classification() {
        let config = EvalConfig {
            metrics: vec![
                Metric::Accuracy,
                Metric::Precision(Average::Macro),
                Metric::Recall(Average::Macro),
                Metric::F1(Average::Macro),
            ],
            ..Default::default()
        };

        let evaluator = ModelEvaluator::new(config);

        let y_pred = vec![0, 1, 1, 2, 0, 1];
        let y_true = vec![0, 1, 0, 2, 0, 2];

        let result = evaluator
            .evaluate_classification("TestModel", &y_pred, &y_true)
            .unwrap();

        assert!(result.get_score(Metric::Accuracy).is_some());
        assert!(result
            .get_score(Metric::Precision(Average::Macro))
            .is_some());
        assert!(result.get_score(Metric::Recall(Average::Macro)).is_some());
        assert!(result.get_score(Metric::F1(Average::Macro)).is_some());

        // Accuracy should be 4/6 = 0.667 (positions 0,1,3,4 correct)
        let acc = result.get_score(Metric::Accuracy).unwrap();
        assert!((acc - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_compare_classification() {
        let config = EvalConfig {
            metrics: vec![Metric::Accuracy],
            ..Default::default()
        };

        let evaluator = ModelEvaluator::new(config);

        let y_true = vec![0, 1, 1, 0, 1, 0];

        // Model A: 4/6 correct
        let y_pred_a = vec![0, 1, 1, 1, 1, 0];
        // Model B: 6/6 correct
        let y_pred_b = vec![0, 1, 1, 0, 1, 0];
        // Model C: 3/6 correct
        let y_pred_c = vec![1, 0, 1, 1, 0, 0];

        let models: Vec<(&str, &[usize])> = vec![
            ("Model A", &y_pred_a),
            ("Model B", &y_pred_b),
            ("Model C", &y_pred_c),
        ];

        let leaderboard = evaluator.compare_classification(&models, &y_true).unwrap();

        // Model B should be first (perfect accuracy)
        assert_eq!(leaderboard.best().unwrap().model_name, "Model B");
    }

    #[test]
    fn test_leaderboard_display() {
        let mut leaderboard = Leaderboard::new(Metric::Accuracy);

        let mut result = EvalResult::new("TestModel");
        result.add_score(Metric::Accuracy, 0.95);
        result.inference_time_ms = 1.5;
        leaderboard.add(result);

        let display = format!("{leaderboard}");
        assert!(display.contains("TestModel"));
        assert!(display.contains("Accuracy"));
    }

    #[test]
    fn test_leaderboard_markdown() {
        let mut leaderboard = Leaderboard::new(Metric::Accuracy);

        let mut result = EvalResult::new("TestModel");
        result.add_score(Metric::Accuracy, 0.95);
        result.inference_time_ms = 1.5;
        leaderboard.add(result);

        let md = leaderboard.to_markdown();
        assert!(md.contains("| Model |"));
        assert!(md.contains("| TestModel |"));
    }

    #[test]
    fn test_metric_display() {
        assert_eq!(format!("{}", Metric::Accuracy), "Accuracy");
        assert_eq!(format!("{}", Metric::F1(Average::Weighted)), "F1(Weighted)");
    }

    #[test]
    fn test_empty_leaderboard() {
        let leaderboard = Leaderboard::new(Metric::Accuracy);
        assert!(leaderboard.best().is_none());
    }

    #[test]
    fn test_kfold_split() {
        let kfold = KFold::new(5).without_shuffle();
        let folds = kfold.split(100);

        assert_eq!(folds.len(), 5);

        // Each fold should have 20 test samples
        for (train, test) in &folds {
            assert_eq!(test.len(), 20);
            assert_eq!(train.len(), 80);
        }

        // All indices should be covered exactly once across test sets
        let mut all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.iter().copied()).collect();
        all_test.sort_unstable();
        assert_eq!(all_test, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn test_kfold_uneven_split() {
        let kfold = KFold::new(3).without_shuffle();
        let folds = kfold.split(10);

        assert_eq!(folds.len(), 3);

        // With 10 samples and 3 folds: sizes should be 4, 3, 3
        let test_sizes: Vec<usize> = folds.iter().map(|(_, t)| t.len()).collect();
        assert_eq!(test_sizes.iter().sum::<usize>(), 10);
    }

    #[test]
    fn test_kfold_shuffled() {
        let kfold1 = KFold::new(5).with_seed(42);
        let kfold2 = KFold::new(5).with_seed(42);
        let kfold3 = KFold::new(5).with_seed(99);

        let folds1 = kfold1.split(100);
        let folds2 = kfold2.split(100);
        let folds3 = kfold3.split(100);

        // Same seed should produce same splits
        assert_eq!(folds1[0].1, folds2[0].1);

        // Different seed should produce different splits
        assert_ne!(folds1[0].1, folds3[0].1);
    }

    #[test]
    fn test_evaluate_cv() {
        let config = EvalConfig {
            metrics: vec![Metric::Accuracy],
            cv_folds: 5,
            seed: 42,
            ..Default::default()
        };

        let evaluator = ModelEvaluator::new(config);

        // Create simple data: labels 0-9 repeated 10 times
        let y_true: Vec<usize> = (0..100).map(|i| i % 10).collect();

        // Prediction function that returns the true labels for test indices
        let result = evaluator
            .evaluate_cv("PerfectModel", &y_true, |_train_idx, test_idx| {
                test_idx.iter().map(|&i| y_true[i]).collect()
            })
            .unwrap();

        // Perfect predictions should have ~1.0 accuracy
        assert!(result.cv_mean.is_some());
        let cv_mean = result.cv_mean.unwrap();
        assert!((cv_mean - 1.0).abs() < 0.01);

        // Should have 5 fold scores
        assert_eq!(result.cv_scores.as_ref().unwrap().len(), 5);
    }

    #[test]
    fn test_evaluate_cv_no_folds_error() {
        let config = EvalConfig {
            cv_folds: 0, // No CV
            ..Default::default()
        };

        let evaluator = ModelEvaluator::new(config);
        let y_true = vec![0, 1, 0, 1];

        let result = evaluator.evaluate_cv("Test", &y_true, |_, _| vec![0, 1, 0, 1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_metric_name_all_variants() {
        assert_eq!(Metric::Accuracy.name(), "Accuracy");
        assert_eq!(Metric::Precision(Average::Macro).name(), "Precision");
        assert_eq!(Metric::Recall(Average::Micro).name(), "Recall");
        assert_eq!(Metric::F1(Average::Weighted).name(), "F1");
        assert_eq!(Metric::R2.name(), "R²");
        assert_eq!(Metric::MSE.name(), "MSE");
        assert_eq!(Metric::MAE.name(), "MAE");
        assert_eq!(Metric::RMSE.name(), "RMSE");
        assert_eq!(Metric::Silhouette.name(), "Silhouette");
        assert_eq!(Metric::Inertia.name(), "Inertia");
    }

    #[test]
    fn test_metric_higher_is_better_all_variants() {
        assert!(Metric::Accuracy.higher_is_better());
        assert!(Metric::Precision(Average::Macro).higher_is_better());
        assert!(Metric::Recall(Average::Micro).higher_is_better());
        assert!(Metric::F1(Average::Weighted).higher_is_better());
        assert!(Metric::R2.higher_is_better());
        assert!(Metric::Silhouette.higher_is_better());
        assert!(!Metric::MSE.higher_is_better());
        assert!(!Metric::MAE.higher_is_better());
        assert!(!Metric::RMSE.higher_is_better());
        assert!(!Metric::Inertia.higher_is_better());
    }

    #[test]
    fn test_metric_display_all_variants() {
        assert_eq!(
            format!("{}", Metric::Precision(Average::Macro)),
            "Precision(Macro)"
        );
        assert_eq!(
            format!("{}", Metric::Recall(Average::Micro)),
            "Recall(Micro)"
        );
        assert_eq!(format!("{}", Metric::MSE), "MSE");
        assert_eq!(format!("{}", Metric::R2), "R²");
        assert_eq!(format!("{}", Metric::Silhouette), "Silhouette");
    }

    #[test]
    fn test_eval_result_display() {
        let mut result = EvalResult::new("TestModel");
        result.add_score(Metric::Accuracy, 0.95);
        result.inference_time_ms = 1.5;

        let display = format!("{result}");
        assert!(display.contains("TestModel"));
        assert!(display.contains("0.95"));
        assert!(display.contains("1.50ms"));
    }

    #[test]
    fn test_leaderboard_empty_display() {
        let leaderboard = Leaderboard::new(Metric::Accuracy);
        let display = format!("{leaderboard}");
        assert!(display.contains("empty"));
    }

    #[test]
    fn test_leaderboard_empty_markdown() {
        let leaderboard = Leaderboard::new(Metric::Accuracy);
        let md = leaderboard.to_markdown();
        assert!(md.is_empty());
    }

    #[test]
    fn test_leaderboard_sort_by() {
        let mut leaderboard = Leaderboard::new(Metric::Accuracy);

        let mut result1 = EvalResult::new("Model A");
        result1.add_score(Metric::Accuracy, 0.85);
        result1.add_score(Metric::F1(Average::Macro), 0.90);
        leaderboard.add(result1);

        let mut result2 = EvalResult::new("Model B");
        result2.add_score(Metric::Accuracy, 0.92);
        result2.add_score(Metric::F1(Average::Macro), 0.80);
        leaderboard.add(result2);

        // Initially sorted by Accuracy: B, A
        assert_eq!(leaderboard.results[0].model_name, "Model B");

        // Sort by F1 instead
        leaderboard.sort_by(Metric::F1(Average::Macro));

        // Now sorted by F1: A, B
        assert_eq!(leaderboard.results[0].model_name, "Model A");
    }

    #[test]
    fn test_eval_config_custom() {
        let config = EvalConfig {
            metrics: vec![Metric::MSE, Metric::MAE],
            cv_folds: 10,
            seed: 123,
            parallel: true,
            trace_enabled: true,
        };

        assert_eq!(config.cv_folds, 10);
        assert_eq!(config.seed, 123);
        assert!(config.parallel);
        assert!(config.trace_enabled);
    }

    #[test]
    fn test_model_evaluator_config() {
        let config = EvalConfig::default();
        let evaluator = ModelEvaluator::new(config.clone());
        assert_eq!(evaluator.config().seed, config.seed);
    }

    #[test]
    fn test_evaluate_classification_length_mismatch() {
        let config = EvalConfig::default();
        let evaluator = ModelEvaluator::new(config);

        let y_pred = vec![0, 1, 2];
        let y_true = vec![0, 1];

        let result = evaluator.evaluate_classification("Test", &y_pred, &y_true);
        assert!(result.is_err());
    }

    #[test]
    fn test_eval_result_cv_fields() {
        let mut result = EvalResult::new("TestModel");
        result.cv_scores = Some(vec![0.9, 0.92, 0.88, 0.91, 0.89]);
        result.cv_mean = Some(0.9);
        result.cv_std = Some(0.014);
        result.trace_id = Some("trace-123".to_string());

        assert_eq!(result.cv_scores.as_ref().unwrap().len(), 5);
        assert_eq!(result.cv_mean, Some(0.9));
        assert_eq!(result.cv_std, Some(0.014));
        assert_eq!(result.trace_id, Some("trace-123".to_string()));
    }

    #[test]
    fn test_evaluate_cv_with_precision_metric() {
        let config = EvalConfig {
            metrics: vec![Metric::Precision(Average::Macro)],
            cv_folds: 3,
            seed: 42,
            ..Default::default()
        };

        let evaluator = ModelEvaluator::new(config);
        let y_true: Vec<usize> = (0..30).map(|i| i % 3).collect();

        let result = evaluator
            .evaluate_cv("TestModel", &y_true, |_train_idx, test_idx| {
                test_idx.iter().map(|&i| y_true[i]).collect()
            })
            .unwrap();

        assert!(result.cv_mean.is_some());
    }

    #[test]
    fn test_evaluate_cv_with_recall_metric() {
        let config = EvalConfig {
            metrics: vec![Metric::Recall(Average::Weighted)],
            cv_folds: 3,
            seed: 42,
            ..Default::default()
        };

        let evaluator = ModelEvaluator::new(config);
        let y_true: Vec<usize> = (0..30).map(|i| i % 3).collect();

        let result = evaluator
            .evaluate_cv("TestModel", &y_true, |_train_idx, test_idx| {
                test_idx.iter().map(|&i| y_true[i]).collect()
            })
            .unwrap();

        assert!(result.cv_mean.is_some());
    }

    #[test]
    fn test_evaluate_cv_single_fold_zero_std() {
        let config = EvalConfig {
            metrics: vec![Metric::Accuracy],
            cv_folds: 1,
            seed: 42,
            ..Default::default()
        };

        let evaluator = ModelEvaluator::new(config);
        let y_true: Vec<usize> = vec![0, 1, 0, 1];

        let result = evaluator
            .evaluate_cv("TestModel", &y_true, |_train_idx, test_idx| {
                test_idx.iter().map(|&i| y_true[i]).collect()
            })
            .unwrap();

        // With single fold, std should be 0
        assert_eq!(result.cv_std, Some(0.0));
    }

    #[test]
    fn test_leaderboard_print() {
        let mut leaderboard = Leaderboard::new(Metric::Accuracy);
        let mut result = EvalResult::new("TestModel");
        result.add_score(Metric::Accuracy, 0.95);
        leaderboard.add(result);

        // Just test that print() doesn't panic
        // (output goes to stdout, can't easily capture in tests)
        leaderboard.print();
    }

    #[test]
    fn test_evaluate_classification_skips_non_classification_metrics() {
        let config = EvalConfig {
            metrics: vec![
                Metric::Accuracy,
                Metric::R2,  // Regression metric, should be skipped
                Metric::MSE, // Regression metric, should be skipped
            ],
            ..Default::default()
        };

        let evaluator = ModelEvaluator::new(config);
        let y_pred = vec![0, 1, 1, 0];
        let y_true = vec![0, 1, 0, 0];

        let result = evaluator
            .evaluate_classification("Test", &y_pred, &y_true)
            .unwrap();

        // Should only have Accuracy, R2 and MSE should be skipped
        assert!(result.get_score(Metric::Accuracy).is_some());
        assert!(result.get_score(Metric::R2).is_none());
        assert!(result.get_score(Metric::MSE).is_none());
    }
}
