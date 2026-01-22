//! Leaderboard for comparing multiple models

use super::metric::Metric;
use super::result::EvalResult;
use std::fmt;

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
