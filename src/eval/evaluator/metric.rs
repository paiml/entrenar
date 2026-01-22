//! Evaluation metric definitions

use super::super::classification::Average;
use std::fmt;

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
