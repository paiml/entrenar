# Model Evaluation Framework Specification

**Issue:** https://github.com/paiml/aprender/issues/73
**Version:** 1.0.0
**Status:** Draft
**Target:** aprender 0.10.0 + entrenar 0.2.0

---

## 1. Overview

A comprehensive model evaluation framework to compare multiple trained `.apr` models and trigger retraining via entrenar based on performance degradation.

### 1.1 Problem Statement

Currently aprender provides:
- Regression metrics: `r_squared`, `mse`, `mae`, `rmse`
- Clustering metrics: `silhouette_score`, `inertia`
- Per-model `.score()` methods

**Gaps:**
- No classification metrics (accuracy, F1, precision, recall)
- No multi-model comparison API
- No performance tracking over time
- No retraining triggers

### 1.2 Solution

Add `aprender::eval` module with:
1. Classification metrics
2. `ModelEvaluator` for multi-model comparison
3. Drift detection with retraining hooks
4. entrenar integration for automated retraining

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     aprender::eval Module                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │    metrics/     │  │   evaluator/    │  │     drift/      │     │
│  │                 │  │                 │  │                 │     │
│  │ - accuracy      │  │ - ModelEval     │  │ - DriftDetector │     │
│  │ - precision     │  │ - EvalConfig    │  │ - KSTest        │     │
│  │ - recall        │  │ - Leaderboard   │  │ - ChiSquare     │     │
│  │ - f1_score      │  │ - CVResults     │  │ - Callback      │     │
│  │ - confusion_mat │  │                 │  │                 │     │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
│           │                    │                    │               │
│           └────────────────────┼────────────────────┘               │
│                                │                                    │
│                                ▼                                    │
│                    ┌─────────────────────┐                          │
│                    │  entrenar::Trainer  │  (optional feature)      │
│                    │  - fit()            │                          │
│                    │  - from_config()    │                          │
│                    └─────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. API Specification

### 3.1 Classification Metrics (`src/metrics/classification.rs`)

```rust
/// Averaging strategy for multi-class metrics
#[derive(Clone, Copy, Debug)]
pub enum Average {
    /// Calculate metrics for each label, return unweighted mean
    Macro,
    /// Calculate metrics globally by counting total TP, FP, FN
    Micro,
    /// Weighted mean by support (number of true instances per label)
    Weighted,
    /// Return array of metrics per class (no averaging)
    None,
}

/// Compute classification accuracy
///
/// accuracy = (TP + TN) / (TP + TN + FP + FN)
pub fn accuracy(y_pred: &[usize], y_true: &[usize]) -> f32;

/// Compute precision score
///
/// precision = TP / (TP + FP)
pub fn precision(y_pred: &[usize], y_true: &[usize], average: Average) -> f32;

/// Compute recall score
///
/// recall = TP / (TP + FN)
pub fn recall(y_pred: &[usize], y_true: &[usize], average: Average) -> f32;

/// Compute F1 score (harmonic mean of precision and recall)
///
/// F1 = 2 * (precision * recall) / (precision + recall)
pub fn f1_score(y_pred: &[usize], y_true: &[usize], average: Average) -> f32;

/// Compute confusion matrix
///
/// Returns Matrix<usize> where element [i,j] is count of samples
/// with true label i and predicted label j
pub fn confusion_matrix(y_pred: &[usize], y_true: &[usize]) -> Matrix<usize>;

/// Generate text classification report (sklearn-style)
pub fn classification_report(y_pred: &[usize], y_true: &[usize]) -> String;
```

### 3.2 Model Evaluator (`src/eval/evaluator.rs`)

```rust
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
}

/// Available evaluation metrics
#[derive(Clone, Copy, Debug)]
pub enum Metric {
    // Classification
    Accuracy,
    Precision(Average),
    Recall(Average),
    F1(Average),
    // Regression
    R2,
    MSE,
    MAE,
    RMSE,
    // Clustering
    Silhouette,
    Inertia,
}

/// Model evaluation results
#[derive(Clone, Debug)]
pub struct EvalResult {
    pub model_name: String,
    pub scores: HashMap<Metric, f32>,
    pub cv_scores: Option<Vec<f32>>,
    pub cv_mean: Option<f32>,
    pub cv_std: Option<f32>,
    pub inference_time_ms: f64,
}

/// Leaderboard for comparing multiple models
#[derive(Clone, Debug)]
pub struct Leaderboard {
    pub results: Vec<EvalResult>,
    pub primary_metric: Metric,
}

impl Leaderboard {
    /// Print formatted leaderboard to stdout
    pub fn print(&self);

    /// Export as markdown table
    pub fn to_markdown(&self) -> String;

    /// Get best model by primary metric
    pub fn best(&self) -> &EvalResult;

    /// Sort by metric (descending for accuracy/F1, ascending for MSE)
    pub fn sort_by(&mut self, metric: Metric);
}

/// Main evaluator struct
pub struct ModelEvaluator {
    config: EvalConfig,
}

impl ModelEvaluator {
    pub fn new(config: EvalConfig) -> Self;

    /// Evaluate single model
    pub fn evaluate<M: Estimator>(
        &self,
        model: &M,
        x: &Matrix<f32>,
        y: &[usize],
    ) -> Result<EvalResult>;

    /// Compare multiple models, return leaderboard
    pub fn compare<M: Estimator>(
        &self,
        models: &[(&str, &M)],
        x: &Matrix<f32>,
        y: &[usize],
    ) -> Result<Leaderboard>;

    /// Track performance over time (append to history)
    pub fn track<M: Estimator>(
        &self,
        model: &M,
        x: &Matrix<f32>,
        y: &[usize],
        history: &mut PerformanceHistory,
    ) -> Result<EvalResult>;
}
```

### 3.3 Drift Detection (`src/eval/drift.rs`)

```rust
/// Statistical test for drift detection
#[derive(Clone, Copy, Debug)]
pub enum DriftTest {
    /// Kolmogorov-Smirnov test (continuous features)
    KS { threshold: f64 },
    /// Chi-square test (categorical features)
    ChiSquare { threshold: f64 },
    /// Population Stability Index
    PSI { threshold: f64 },
}

/// Drift detection result
#[derive(Clone, Debug)]
pub struct DriftResult {
    pub feature: String,
    pub test: DriftTest,
    pub statistic: f64,
    pub p_value: f64,
    pub drifted: bool,
}

/// Callback type for drift events
pub type DriftCallback = Box<dyn Fn(&str, f32) -> Result<()> + Send + Sync>;

/// Drift detector with retraining hooks
pub struct DriftDetector {
    tests: Vec<DriftTest>,
    baseline: Option<Matrix<f32>>,
    callbacks: Vec<DriftCallback>,
}

impl DriftDetector {
    pub fn new(tests: Vec<DriftTest>) -> Self;

    /// Set baseline distribution
    pub fn set_baseline(&mut self, data: &Matrix<f32>);

    /// Check for drift against baseline
    pub fn check(&self, current: &Matrix<f32>) -> Vec<DriftResult>;

    /// Register callback for drift events
    pub fn on_drift<F>(&mut self, callback: F)
    where
        F: Fn(&str, f32) -> Result<()> + Send + Sync + 'static;

    /// Check and trigger callbacks if drift detected
    pub fn check_and_trigger(&self, current: &Matrix<f32>) -> Result<Vec<DriftResult>>;
}
```

### 3.4 Entrenar Integration (`src/eval/retrain.rs`)

```rust
#[cfg(feature = "entrenar")]
pub mod retrain {
    use entrenar::Trainer;

    /// Retraining mode
    #[derive(Clone, Copy, Debug)]
    pub enum RetrainMode {
        /// Retrain on accumulated batch
        Batch { min_samples: usize },
        /// Retrain on each new sample (online learning)
        RealTime,
        /// Retrain on schedule
        Scheduled { interval_secs: u64 },
    }

    /// Retraining configuration
    #[derive(Clone, Debug)]
    pub struct RetrainConfig {
        pub mode: RetrainMode,
        pub config_path: PathBuf,
        pub output_path: PathBuf,
        pub keep_n_versions: usize,
    }

    /// Auto-retrainer with drift detection
    pub struct AutoRetrainer {
        drift_detector: DriftDetector,
        trainer: Trainer,
        config: RetrainConfig,
        buffer: Vec<(Matrix<f32>, Vec<usize>)>,
    }

    impl AutoRetrainer {
        pub fn new(config: RetrainConfig) -> Result<Self>;

        /// Process new data point/batch
        pub fn ingest(&mut self, x: &Matrix<f32>, y: &[usize]) -> Result<()>;

        /// Force retrain now
        pub fn retrain_now(&mut self) -> Result<PathBuf>;

        /// Check drift and retrain if needed
        pub fn check_and_retrain(&mut self) -> Result<Option<PathBuf>>;
    }
}
```

---

## 4. File Structure

```
aprender/src/
├── eval/
│   ├── mod.rs              # Module exports
│   ├── evaluator.rs        # ModelEvaluator, Leaderboard
│   ├── drift.rs            # DriftDetector, statistical tests
│   └── retrain.rs          # AutoRetrainer (feature-gated)
├── metrics/
│   ├── mod.rs              # Re-exports
│   ├── regression.rs       # Existing: r_squared, mse, mae, rmse
│   ├── clustering.rs       # Existing: silhouette_score, inertia
│   └── classification.rs   # NEW: accuracy, f1, precision, recall
```

---

## 5. Feature Flags

```toml
[features]
default = []
eval = []                           # Enable eval module
entrenar = ["eval", "dep:entrenar"] # Enable retraining integration
```

---

## 6. Dependencies

| Crate | Version | Purpose | Feature |
|-------|---------|---------|---------|
| aprender | 0.9.x | Base library | - |
| entrenar | 0.1.x | Retraining | `entrenar` |
| statrs | 0.16 | Statistical tests | `eval` |

---

## 7. Implementation Tickets

| ID | Task | Hours | Priority |
|----|------|-------|----------|
| APR-073-1 | Classification metrics | 8 | P0 |
| APR-073-2 | ModelEvaluator + Leaderboard | 16 | P0 |
| APR-073-3 | Cross-validation integration | 8 | P1 |
| APR-073-4 | Drift detection (KS, Chi-sq) | 12 | P1 |
| APR-073-5 | Entrenar integration | 16 | P2 |
| APR-073-6 | Property tests + docs | 8 | P0 |

**Total:** 68 hours

---

## 8. Quality Requirements

- Test coverage: ≥95%
- Mutation score: ≥85%
- Property tests: 1000+ iterations per metric
- Gradient validation: N/A (no gradients)
- WASM compatible: Yes (except entrenar feature)

---

## 9. Example Usage

```rust
use aprender::prelude::*;
use aprender::eval::{ModelEvaluator, EvalConfig, Metric, DriftDetector, DriftTest};
use aprender::format::load;

fn main() -> Result<()> {
    // Load models
    let rf = load::<RandomForestClassifier>("rf.apr")?;
    let gb = load::<GradientBoostingClassifier>("gb.apr")?;
    let svm = load::<SVM>("svm.apr")?;

    // Configure evaluator
    let evaluator = ModelEvaluator::new(EvalConfig {
        metrics: vec![
            Metric::Accuracy,
            Metric::F1(Average::Weighted),
            Metric::Precision(Average::Macro),
        ],
        cv_folds: 5,
        seed: 42,
        parallel: true,
    });

    // Compare models
    let leaderboard = evaluator.compare(&[
        ("RandomForest", &rf),
        ("GradientBoosting", &gb),
        ("SVM", &svm),
    ], &x_test, &y_test)?;

    leaderboard.print();
    // ┌─────────────────┬──────────┬─────────┬───────────┐
    // │ Model           │ Accuracy │ F1      │ Precision │
    // ├─────────────────┼──────────┼─────────┼───────────┤
    // │ GradientBoosting│ 0.9423   │ 0.9401  │ 0.9387    │
    // │ RandomForest    │ 0.9318   │ 0.9295  │ 0.9271    │
    // │ SVM             │ 0.9156   │ 0.9112  │ 0.9089    │
    // └─────────────────┴──────────┴─────────┴───────────┘

    // Setup drift detection
    let mut detector = DriftDetector::new(vec![
        DriftTest::KS { threshold: 0.05 },
    ]);
    detector.set_baseline(&x_train);

    // Check new data for drift
    let drift_results = detector.check(&x_new);
    for result in drift_results.iter().filter(|r| r.drifted) {
        println!("Drift detected in {}: p={:.4}", result.feature, result.p_value);
    }

    Ok(())
}
```

---

## 10. References

- Vision Sync: `entrenar/docs/specifications/paiml-sai-vision-sync.md`
- sklearn classification_report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
- PSI (Population Stability Index): https://www.listendata.com/2015/05/population-stability-index.html
- Kolmogorov-Smirnov Test: https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test

---

*Generated from GitHub Issue #73 via pmat workflow*
