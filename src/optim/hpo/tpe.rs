//! Tree-structured Parzen Estimator (TPE) optimizer
//!
//! Based on Bergstra et al. (2011) - Algorithms for Hyper-Parameter Optimization

use rand::Rng;
use std::collections::HashMap;

use super::error::{HPOError, Result};
use super::types::{HyperparameterSpace, ParameterDomain, ParameterValue, Trial, TrialStatus};

/// Tree-structured Parzen Estimator optimizer
///
/// # Toyota Way: Kaizen
///
/// Uses accumulated knowledge from trials to make increasingly better suggestions.
/// Splits trials by quantile to model "good" vs "bad" configurations.
#[derive(Debug, Clone)]
pub struct TPEOptimizer {
    /// Search space
    space: HyperparameterSpace,
    /// Quantile for splitting good/bad (default: 0.25)
    pub(crate) gamma: f64,
    /// Number of startup trials (random sampling)
    n_startup: usize,
    /// KDE bandwidth
    kde_bandwidth: f64,
    /// Completed trials
    trials: Vec<Trial>,
    /// Next trial ID
    next_id: usize,
}

impl TPEOptimizer {
    /// Create a new TPE optimizer
    pub fn new(space: HyperparameterSpace) -> Self {
        Self {
            space,
            gamma: 0.25,
            n_startup: 10,
            kde_bandwidth: 1.0,
            trials: Vec::new(),
            next_id: 0,
        }
    }

    /// Set gamma (quantile for splitting)
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma.clamp(0.01, 0.99);
        self
    }

    /// Set number of startup trials
    pub fn with_startup(mut self, n: usize) -> Self {
        self.n_startup = n.max(1);
        self
    }

    /// Get number of completed trials
    pub fn n_trials(&self) -> usize {
        self.trials
            .iter()
            .filter(|t| t.status == TrialStatus::Completed)
            .count()
    }

    /// Get best trial so far
    pub fn best_trial(&self) -> Option<&Trial> {
        self.trials
            .iter()
            .filter(|t| t.status == TrialStatus::Completed)
            .min_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Suggest next configuration to try
    pub fn suggest(&mut self) -> Result<Trial> {
        if self.space.is_empty() {
            return Err(HPOError::EmptySpace);
        }

        let mut rng = rand::rng();
        let config = if self.n_trials() < self.n_startup {
            // Random sampling during startup phase
            self.space.sample_random(&mut rng)
        } else {
            // TPE-guided sampling
            self.tpe_sample(&mut rng)
        };

        let trial = Trial::new(self.next_id, config);
        self.next_id += 1;
        Ok(trial)
    }

    /// Record trial result
    pub fn record(&mut self, mut trial: Trial, score: f64, iterations: usize) {
        trial.complete(score, iterations);
        self.trials.push(trial);
    }

    /// Record failed trial
    pub fn record_failed(&mut self, mut trial: Trial) {
        trial.fail();
        self.trials.push(trial);
    }

    /// TPE sampling (internal)
    fn tpe_sample<R: Rng>(&self, rng: &mut R) -> HashMap<String, ParameterValue> {
        let completed: Vec<_> = self
            .trials
            .iter()
            .filter(|t| t.status == TrialStatus::Completed)
            .collect();

        if completed.is_empty() {
            return self.space.sample_random(rng);
        }

        // Split trials into good (l) and bad (g) by gamma quantile
        let n_good = ((completed.len() as f64) * self.gamma).ceil() as usize;
        let n_good = n_good.max(1).min(completed.len() - 1);

        let mut sorted: Vec<_> = completed.clone();
        sorted.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let (good_trials, bad_trials) = sorted.split_at(n_good);

        // Sample each parameter using TPE
        let mut config = HashMap::new();
        for (name, domain) in self.space.iter() {
            let value = self.sample_parameter_tpe(name, domain, good_trials, bad_trials, rng);
            config.insert(name.clone(), value);
        }

        config
    }

    /// Sample a single parameter using TPE
    fn sample_parameter_tpe<R: Rng>(
        &self,
        name: &str,
        domain: &ParameterDomain,
        good_trials: &[&Trial],
        bad_trials: &[&Trial],
        rng: &mut R,
    ) -> ParameterValue {
        match domain {
            ParameterDomain::Continuous {
                low,
                high,
                log_scale,
            } => {
                // Extract values from trials
                let good_values: Vec<f64> = good_trials
                    .iter()
                    .filter_map(|t| t.config.get(name)?.as_float())
                    .map(|v| if *log_scale { v.ln() } else { v })
                    .collect();

                let bad_values: Vec<f64> = bad_trials
                    .iter()
                    .filter_map(|t| t.config.get(name)?.as_float())
                    .map(|v| if *log_scale { v.ln() } else { v })
                    .collect();

                // Sample from l(x) / g(x) using simple KDE approximation
                let (effective_low, effective_high) = if *log_scale {
                    (low.ln(), high.ln())
                } else {
                    (*low, *high)
                };

                let value = self.sample_ei_ratio_continuous(
                    &good_values,
                    &bad_values,
                    effective_low,
                    effective_high,
                    rng,
                );

                let final_value = if *log_scale { value.exp() } else { value };
                ParameterValue::Float(final_value.clamp(*low, *high))
            }
            ParameterDomain::Discrete { low, high } => {
                // Extract values
                let good_values: Vec<i64> = good_trials
                    .iter()
                    .filter_map(|t| t.config.get(name)?.as_int())
                    .collect();

                let bad_values: Vec<i64> = bad_trials
                    .iter()
                    .filter_map(|t| t.config.get(name)?.as_int())
                    .collect();

                let value =
                    self.sample_ei_ratio_discrete(&good_values, &bad_values, *low, *high, rng);
                ParameterValue::Int(value)
            }
            ParameterDomain::Categorical { choices } => {
                // Count occurrences
                let good_counts = self.count_categorical(name, good_trials, choices);
                let bad_counts = self.count_categorical(name, bad_trials, choices);

                // Sample based on l(x) / g(x)
                let mut weights: Vec<f64> = choices
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let l = (good_counts[i] + 1) as f64; // Laplace smoothing
                        let g = (bad_counts[i] + 1) as f64;
                        l / g
                    })
                    .collect();

                // Normalize
                let total: f64 = weights.iter().sum();
                for w in &mut weights {
                    *w /= total;
                }

                // Sample
                let r: f64 = rng.random();
                let mut cumsum = 0.0;
                for (i, &w) in weights.iter().enumerate() {
                    cumsum += w;
                    if r < cumsum {
                        return ParameterValue::Categorical(choices[i].clone());
                    }
                }

                ParameterValue::Categorical(
                    choices
                        .last()
                        .expect("choices is non-empty per validate()")
                        .clone(),
                )
            }
        }
    }

    /// Sample continuous parameter with EI ratio
    fn sample_ei_ratio_continuous<R: Rng>(
        &self,
        good_values: &[f64],
        bad_values: &[f64],
        low: f64,
        high: f64,
        rng: &mut R,
    ) -> f64 {
        if good_values.is_empty() {
            return low + rng.random::<f64>() * (high - low);
        }

        // Generate candidate samples
        let n_candidates = 24;
        let mut best_value = low;
        let mut best_ei = f64::NEG_INFINITY;

        let bandwidth = self.kde_bandwidth * (high - low) / 10.0;

        for _ in 0..n_candidates {
            // Sample from good distribution (KDE)
            let idx = (rng.random::<f64>() * good_values.len() as f64).floor() as usize;
            let idx = idx.min(good_values.len() - 1);
            let base = good_values[idx];
            // Box-Muller transform for Gaussian noise
            let u1: f64 = rng.random::<f64>().max(1e-10);
            let u2: f64 = rng.random::<f64>();
            let noise =
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * bandwidth;
            let candidate = (base + noise).clamp(low, high);

            // Compute l(x) / g(x) approximately
            let l_score = self.kde_score(candidate, good_values, bandwidth);
            let g_score = self.kde_score(candidate, bad_values, bandwidth);
            let ei = l_score / (g_score + 1e-10);

            if ei > best_ei {
                best_ei = ei;
                best_value = candidate;
            }
        }

        best_value
    }

    /// Simple KDE score
    fn kde_score(&self, x: f64, values: &[f64], bandwidth: f64) -> f64 {
        if values.is_empty() {
            return 1.0;
        }
        values
            .iter()
            .map(|&v| (-(x - v).powi(2) / (2.0 * bandwidth.powi(2))).exp())
            .sum::<f64>()
            / values.len() as f64
    }

    /// Sample discrete parameter with EI ratio
    fn sample_ei_ratio_discrete<R: Rng>(
        &self,
        good_values: &[i64],
        bad_values: &[i64],
        low: i64,
        high: i64,
        rng: &mut R,
    ) -> i64 {
        if good_values.is_empty() {
            let range = (high - low + 1) as usize;
            let offset = (rng.random::<f64>() * range as f64).floor() as i64;
            return (low + offset).min(high);
        }

        // Count occurrences with Laplace smoothing
        let range = (high - low + 1) as usize;
        let mut good_counts = vec![1.0; range]; // Laplace smoothing
        let mut bad_counts = vec![1.0; range];

        for &v in good_values {
            good_counts[(v - low) as usize] += 1.0;
        }
        for &v in bad_values {
            bad_counts[(v - low) as usize] += 1.0;
        }

        // Compute weights (l/g)
        let mut weights: Vec<f64> = good_counts
            .iter()
            .zip(bad_counts.iter())
            .map(|(l, g)| l / g)
            .collect();

        // Normalize
        let total: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= total;
        }

        // Sample
        let r: f64 = rng.random();
        let mut cumsum = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            cumsum += w;
            if r < cumsum {
                return low + i as i64;
            }
        }

        high
    }

    /// Count categorical occurrences
    fn count_categorical(&self, name: &str, trials: &[&Trial], choices: &[String]) -> Vec<usize> {
        let mut counts = vec![0usize; choices.len()];
        for trial in trials {
            if let Some(ParameterValue::Categorical(s)) = trial.config.get(name) {
                if let Some(idx) = choices.iter().position(|c| c == s) {
                    counts[idx] += 1;
                }
            }
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tpe_new() {
        let space = HyperparameterSpace::new();
        let tpe = TPEOptimizer::new(space);
        assert_eq!(tpe.n_trials(), 0);
        assert!(tpe.best_trial().is_none());
    }

    #[test]
    fn test_tpe_suggest_empty_space() {
        let space = HyperparameterSpace::new();
        let mut tpe = TPEOptimizer::new(space);
        let result = tpe.suggest();
        assert!(matches!(result, Err(HPOError::EmptySpace)));
    }

    #[test]
    fn test_tpe_suggest_startup() {
        let mut space = HyperparameterSpace::new();
        space.add(
            "lr",
            ParameterDomain::Continuous {
                low: 1e-5,
                high: 1e-1,
                log_scale: true,
            },
        );

        let mut tpe = TPEOptimizer::new(space).with_startup(5);

        // First suggestions should work (startup phase)
        for _i in 0..5 {
            let trial = tpe.suggest().unwrap();
            assert!(trial.config.contains_key("lr"));
        }
    }

    #[test]
    fn test_tpe_record_and_best() {
        let mut space = HyperparameterSpace::new();
        space.add(
            "lr",
            ParameterDomain::Continuous {
                low: 0.0,
                high: 1.0,
                log_scale: false,
            },
        );

        let mut tpe = TPEOptimizer::new(space);

        let trial1 = tpe.suggest().unwrap();
        tpe.record(trial1, 0.5, 10);

        let trial2 = tpe.suggest().unwrap();
        tpe.record(trial2, 0.3, 10);

        assert_eq!(tpe.n_trials(), 2);
        let best = tpe.best_trial().unwrap();
        assert_eq!(best.score, 0.3);
    }

    #[test]
    fn test_tpe_with_gamma() {
        let space = HyperparameterSpace::new();
        let tpe = TPEOptimizer::new(space).with_gamma(0.15);
        assert!((tpe.gamma - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_tpe_guided_sampling() {
        let mut space = HyperparameterSpace::new();
        space.add(
            "x",
            ParameterDomain::Continuous {
                low: 0.0,
                high: 10.0,
                log_scale: false,
            },
        );

        let mut tpe = TPEOptimizer::new(space).with_startup(5);

        // Run startup phase
        for _i in 0..5 {
            let trial = tpe.suggest().unwrap();
            // Lower x values get better scores
            let score = trial.config.get("x").unwrap().as_float().unwrap();
            tpe.record(trial, score, 10);
        }

        // After startup, TPE should suggest values closer to 0
        // (where scores are better in our mock objective)
        assert_eq!(tpe.n_trials(), 5);
    }

    #[test]
    fn test_tpe_record_failed() {
        let mut space = HyperparameterSpace::new();
        space.add(
            "lr",
            ParameterDomain::Continuous {
                low: 0.0,
                high: 1.0,
                log_scale: false,
            },
        );

        let mut tpe = TPEOptimizer::new(space);
        let trial = tpe.suggest().unwrap();
        tpe.record_failed(trial);

        // Failed trials shouldn't count as completed
        assert_eq!(tpe.n_trials(), 0);
    }

    #[test]
    fn test_tpe_tpe_sampling_with_trials() {
        let mut space = HyperparameterSpace::new();
        space.add(
            "lr",
            ParameterDomain::Continuous {
                low: 0.0,
                high: 1.0,
                log_scale: false,
            },
        );
        space.add("batch_size", ParameterDomain::Discrete { low: 8, high: 32 });
        space.add(
            "activation",
            ParameterDomain::Categorical {
                choices: vec!["relu".to_string(), "gelu".to_string()],
            },
        );

        let mut tpe = TPEOptimizer::new(space).with_startup(3);

        // Run startup phase
        for _ in 0..3 {
            let trial = tpe.suggest().unwrap();
            let lr = trial.config.get("lr").unwrap().as_float().unwrap();
            tpe.record(trial, lr, 10); // Score equals lr
        }

        // Now TPE sampling kicks in
        for _ in 0..5 {
            let trial = tpe.suggest().unwrap();
            assert!(trial.config.contains_key("lr"));
            assert!(trial.config.contains_key("batch_size"));
            assert!(trial.config.contains_key("activation"));
            let lr = trial.config.get("lr").unwrap().as_float().unwrap();
            tpe.record(trial, lr, 10);
        }

        assert_eq!(tpe.n_trials(), 8);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_tpe_trials_increment(n_trials in 1usize..20) {
            let mut space = HyperparameterSpace::new();
            space.add("x", ParameterDomain::Continuous {
                low: 0.0,
                high: 1.0,
                log_scale: false,
            });

            let mut tpe = TPEOptimizer::new(space);
            for i in 0..n_trials {
                let trial = tpe.suggest().unwrap();
                let score = (i as f64) / 10.0;
                tpe.record(trial, score, 10);
            }
            prop_assert_eq!(tpe.n_trials(), n_trials);
        }
    }
}
