//! Evaluation metric definitions

use super::super::classification::Average;
use std::fmt;

/// ROUGE variant for text generation evaluation
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RougeVariant {
    /// Unigram overlap
    Rouge1,
    /// Bigram overlap
    Rouge2,
    /// Longest common subsequence
    RougeL,
}

impl fmt::Display for RougeVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RougeVariant::Rouge1 => write!(f, "ROUGE-1"),
            RougeVariant::Rouge2 => write!(f, "ROUGE-2"),
            RougeVariant::RougeL => write!(f, "ROUGE-L"),
        }
    }
}

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
    // ASR (Automatic Speech Recognition)
    /// Word Error Rate (lower is better)
    WER,
    /// Inverse Real-Time Factor (higher is better: RTFx=100 means 100x real-time)
    RTFx,
    // Text Generation
    /// BLEU score (higher is better)
    BLEU,
    /// ROUGE score with variant (higher is better)
    ROUGE(RougeVariant),
    /// Perplexity (lower is better)
    Perplexity,
    // LLM Benchmarks
    /// MMLU accuracy (higher is better, covers MMLU-PRO, BBH, etc.)
    MMLUAccuracy,
    // Code
    /// pass@k — unbiased estimator, parameterized by k (higher is better)
    PassAtK(usize),
    // Retrieval
    /// NDCG@k — normalized discounted cumulative gain (higher is better)
    NDCGAtK(usize),
}

impl Metric {
    /// Whether higher values are better for this metric
    pub fn higher_is_better(&self) -> bool {
        !matches!(
            self,
            Metric::MSE
                | Metric::MAE
                | Metric::RMSE
                | Metric::Inertia
                | Metric::WER
                | Metric::Perplexity
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
            Metric::WER => "WER",
            Metric::RTFx => "RTFx",
            Metric::BLEU => "BLEU",
            Metric::ROUGE(_) => "ROUGE",
            Metric::Perplexity => "Perplexity",
            Metric::MMLUAccuracy => "MMLU",
            Metric::PassAtK(_) => "pass@k",
            Metric::NDCGAtK(_) => "NDCG@k",
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Metric::Precision(avg) => write!(f, "Precision({avg:?})"),
            Metric::Recall(avg) => write!(f, "Recall({avg:?})"),
            Metric::F1(avg) => write!(f, "F1({avg:?})"),
            Metric::ROUGE(variant) => write!(f, "{variant}"),
            Metric::PassAtK(k) => write!(f, "pass@{k}"),
            Metric::NDCGAtK(k) => write!(f, "NDCG@{k}"),
            _ => write!(f, "{}", self.name()),
        }
    }
}
