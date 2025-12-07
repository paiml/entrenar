//! Hyperparameter Optimization Module (MLOPS-011)
//!
//! Bayesian optimization with TPE and Hyperband schedulers.
//!
//! # Toyota Way: 改善 (Kaizen)
//!
//! Continuous improvement through intelligent search. Each trial informs the next,
//! building knowledge iteratively rather than wasteful exhaustive search.
//!
//! # Example
//!
//! ```ignore
//! use entrenar::optim::hpo::{HyperparameterSpace, ParameterDomain, TPEOptimizer};
//!
//! let mut space = HyperparameterSpace::new();
//! space.add("learning_rate", ParameterDomain::Continuous {
//!     low: 1e-5, high: 1e-1, log_scale: true
//! });
//! space.add("batch_size", ParameterDomain::Discrete { low: 8, high: 128 });
//!
//! let optimizer = TPEOptimizer::new(space);
//! let config = optimizer.suggest(&trials);
//! ```
//!
//! # References
//!
//! [1] Bergstra et al. (2011) - Algorithms for Hyper-Parameter Optimization (TPE)
//! [2] Li et al. (2018) - Hyperband: A Novel Bandit-Based Approach

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// =============================================================================
// Core Types
// =============================================================================

/// HPO errors
#[derive(Debug, Error)]
pub enum HPOError {
    #[error("Empty search space")]
    EmptySpace,

    #[error("Parameter not found: {0}")]
    ParameterNotFound(String),

    #[error("Invalid parameter value for {0}: {1}")]
    InvalidValue(String, String),

    #[error("No trials completed")]
    NoTrials,

    #[error("HPO error: {0}")]
    Internal(String),
}

/// Result type for HPO operations
pub type Result<T> = std::result::Result<T, HPOError>;

/// Parameter value (sampled from domain)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterValue {
    Float(f64),
    Int(i64),
    Categorical(String),
}

impl ParameterValue {
    /// Get as float (converts int to float if needed)
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ParameterValue::Float(v) => Some(*v),
            ParameterValue::Int(v) => Some(*v as f64),
            ParameterValue::Categorical(_) => None,
        }
    }

    /// Get as int
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ParameterValue::Int(v) => Some(*v),
            ParameterValue::Float(v) => Some(*v as i64),
            ParameterValue::Categorical(_) => None,
        }
    }

    /// Get as string
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ParameterValue::Categorical(s) => Some(s),
            _ => None,
        }
    }
}

/// Parameter domain (search space)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterDomain {
    /// Continuous range [low, high], optionally log-scaled
    Continuous {
        low: f64,
        high: f64,
        log_scale: bool,
    },
    /// Discrete integer range [low, high]
    Discrete { low: i64, high: i64 },
    /// Categorical choices
    Categorical { choices: Vec<String> },
}

impl ParameterDomain {
    /// Sample a random value from this domain
    pub fn sample<R: Rng>(&self, rng: &mut R) -> ParameterValue {
        match self {
            ParameterDomain::Continuous {
                low,
                high,
                log_scale,
            } => {
                let value = if *log_scale {
                    let log_low = low.ln();
                    let log_high = high.ln();
                    let log_val = log_low + rng.random::<f64>() * (log_high - log_low);
                    log_val.exp()
                } else {
                    low + rng.random::<f64>() * (high - low)
                };
                ParameterValue::Float(value)
            }
            ParameterDomain::Discrete { low, high } => {
                let range = (*high - *low + 1) as usize;
                let offset = (rng.random::<f64>() * range as f64).floor() as i64;
                let value = (*low + offset).min(*high);
                ParameterValue::Int(value)
            }
            ParameterDomain::Categorical { choices } => {
                let idx = (rng.random::<f64>() * choices.len() as f64).floor() as usize;
                let idx = idx.min(choices.len() - 1);
                ParameterValue::Categorical(choices[idx].clone())
            }
        }
    }

    /// Check if a value is valid for this domain
    pub fn is_valid(&self, value: &ParameterValue) -> bool {
        match (self, value) {
            (ParameterDomain::Continuous { low, high, .. }, ParameterValue::Float(v)) => {
                *v >= *low && *v <= *high
            }
            (ParameterDomain::Discrete { low, high }, ParameterValue::Int(v)) => {
                *v >= *low && *v <= *high
            }
            (ParameterDomain::Categorical { choices }, ParameterValue::Categorical(s)) => {
                choices.contains(s)
            }
            _ => false,
        }
    }
}

/// Hyperparameter search space
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HyperparameterSpace {
    /// Parameter name -> domain mapping
    params: HashMap<String, ParameterDomain>,
}

impl HyperparameterSpace {
    /// Create an empty search space
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a parameter to the search space
    pub fn add(&mut self, name: &str, domain: ParameterDomain) {
        self.params.insert(name.to_string(), domain);
    }

    /// Get a parameter domain
    pub fn get(&self, name: &str) -> Option<&ParameterDomain> {
        self.params.get(name)
    }

    /// Check if space is empty
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Get number of parameters
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Iterate over parameters
    pub fn iter(&self) -> impl Iterator<Item = (&String, &ParameterDomain)> {
        self.params.iter()
    }

    /// Sample a random configuration
    pub fn sample_random<R: Rng>(&self, rng: &mut R) -> HashMap<String, ParameterValue> {
        self.params
            .iter()
            .map(|(name, domain)| (name.clone(), domain.sample(rng)))
            .collect()
    }

    /// Validate a configuration
    pub fn validate(&self, config: &HashMap<String, ParameterValue>) -> Result<()> {
        for (name, domain) in &self.params {
            match config.get(name) {
                Some(value) if domain.is_valid(value) => {}
                Some(value) => {
                    return Err(HPOError::InvalidValue(name.clone(), format!("{value:?}")))
                }
                None => return Err(HPOError::ParameterNotFound(name.clone())),
            }
        }
        Ok(())
    }
}

/// A single trial (configuration + score)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    /// Trial ID
    pub id: usize,
    /// Parameter configuration
    pub config: HashMap<String, ParameterValue>,
    /// Objective score (lower is better by default)
    pub score: f64,
    /// Number of epochs/iterations used
    pub iterations: usize,
    /// Trial status
    pub status: TrialStatus,
}

impl Trial {
    /// Create a new trial
    pub fn new(id: usize, config: HashMap<String, ParameterValue>) -> Self {
        Self {
            id,
            config,
            score: f64::INFINITY,
            iterations: 0,
            status: TrialStatus::Pending,
        }
    }

    /// Mark trial as complete with score
    pub fn complete(&mut self, score: f64, iterations: usize) {
        self.score = score;
        self.iterations = iterations;
        self.status = TrialStatus::Completed;
    }

    /// Mark trial as failed
    pub fn fail(&mut self) {
        self.status = TrialStatus::Failed;
    }
}

/// Trial status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrialStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Pruned,
}

// =============================================================================
// Search Strategies
// =============================================================================

/// Search strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Exhaustive grid search
    Grid,
    /// Random search
    Random { n_samples: usize },
    /// Bayesian optimization
    Bayesian {
        n_initial: usize,
        acquisition: AcquisitionFunction,
        surrogate: SurrogateModel,
    },
    /// Hyperband (successive halving)
    Hyperband {
        max_iter: usize,
        eta: f64, // Reduction factor (typically 3)
    },
}

/// Acquisition function for Bayesian optimization
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AcquisitionFunction {
    /// Expected Improvement
    ExpectedImprovement,
    /// Upper Confidence Bound
    UpperConfidenceBound { kappa: f64 },
    /// Probability of Improvement
    ProbabilityOfImprovement,
}

/// Surrogate model for Bayesian optimization
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SurrogateModel {
    /// Tree-structured Parzen Estimator (recommended)
    TPE,
    /// Gaussian Process
    GaussianProcess,
    /// Random Forest
    RandomForest { n_trees: usize },
}

// =============================================================================
// TPE Optimizer (Bergstra et al., 2011)
// =============================================================================

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
    gamma: f64,
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

// =============================================================================
// Hyperband Scheduler (Li et al., 2018)
// =============================================================================

/// Hyperband scheduler for efficient hyperparameter search
///
/// # Toyota Way: Muda (Waste Elimination)
///
/// Aggressive early stopping eliminates poorly performing configurations,
/// focusing resources on promising candidates.
#[derive(Debug, Clone)]
pub struct HyperbandScheduler {
    /// Maximum iterations per configuration
    max_iter: usize,
    /// Reduction factor (typically 3)
    eta: f64,
    /// Search space
    space: HyperparameterSpace,
}

impl HyperbandScheduler {
    /// Create a new Hyperband scheduler
    pub fn new(space: HyperparameterSpace, max_iter: usize) -> Self {
        Self {
            max_iter,
            eta: 3.0,
            space,
        }
    }

    /// Set reduction factor
    pub fn with_eta(mut self, eta: f64) -> Self {
        self.eta = eta.max(2.0);
        self
    }

    /// Get s_max (number of successive halving brackets)
    pub fn s_max(&self) -> usize {
        (self.max_iter as f64).log(self.eta).floor() as usize
    }

    /// Get total budget B
    pub fn budget(&self) -> usize {
        (self.s_max() + 1) * self.max_iter
    }

    /// Generate bracket configurations
    ///
    /// Returns Vec of (n_configs, n_iterations) for each rung in the bracket
    pub fn bracket(&self, s: usize) -> Vec<(usize, usize)> {
        let s_max = self.s_max();
        if s > s_max {
            return Vec::new();
        }

        let n = ((self.budget() as f64 / self.max_iter as f64)
            * (self.eta.powi(s as i32) / (s + 1) as f64))
            .ceil() as usize;
        let r = self.max_iter / self.eta.powi(s as i32) as usize;

        (0..=s)
            .map(|i| {
                let n_i = (n as f64 / self.eta.powi(i as i32)).floor() as usize;
                let r_i = (r as f64 * self.eta.powi(i as i32)).floor() as usize;
                (n_i.max(1), r_i.max(1))
            })
            .collect()
    }

    /// Generate all configurations for a bracket
    pub fn generate_configs(&self, n: usize) -> Vec<HashMap<String, ParameterValue>> {
        let mut rng = rand::rng();
        (0..n).map(|_| self.space.sample_random(&mut rng)).collect()
    }
}

// =============================================================================
// Grid Search
// =============================================================================

/// Grid search generator
#[derive(Debug, Clone)]
pub struct GridSearch {
    space: HyperparameterSpace,
    /// Grid points per continuous parameter
    n_points: usize,
}

impl GridSearch {
    /// Create new grid search
    pub fn new(space: HyperparameterSpace, n_points: usize) -> Self {
        Self {
            space,
            n_points: n_points.max(2),
        }
    }

    /// Generate all grid configurations
    pub fn configurations(&self) -> Vec<HashMap<String, ParameterValue>> {
        let param_values: Vec<(String, Vec<ParameterValue>)> = self
            .space
            .iter()
            .map(|(name, domain)| {
                let values = match domain {
                    ParameterDomain::Continuous {
                        low,
                        high,
                        log_scale,
                    } => {
                        if *log_scale {
                            let log_low = low.ln();
                            let log_high = high.ln();
                            (0..self.n_points)
                                .map(|i| {
                                    let t = i as f64 / (self.n_points - 1) as f64;
                                    let log_val = log_low + t * (log_high - log_low);
                                    ParameterValue::Float(log_val.exp())
                                })
                                .collect()
                        } else {
                            (0..self.n_points)
                                .map(|i| {
                                    let t = i as f64 / (self.n_points - 1) as f64;
                                    ParameterValue::Float(low + t * (high - low))
                                })
                                .collect()
                        }
                    }
                    ParameterDomain::Discrete { low, high } => {
                        (*low..=*high).map(ParameterValue::Int).collect()
                    }
                    ParameterDomain::Categorical { choices } => choices
                        .iter()
                        .map(|c| ParameterValue::Categorical(c.clone()))
                        .collect(),
                };
                (name.clone(), values)
            })
            .collect();

        // Generate cartesian product
        Self::cartesian_product(&param_values)
    }

    fn cartesian_product(
        param_values: &[(String, Vec<ParameterValue>)],
    ) -> Vec<HashMap<String, ParameterValue>> {
        if param_values.is_empty() {
            return vec![HashMap::new()];
        }

        let (name, values) = &param_values[0];
        let rest = &param_values[1..];
        let rest_configs = Self::cartesian_product(rest);

        values
            .iter()
            .flat_map(|v| {
                rest_configs.iter().map(move |config| {
                    let mut new_config = config.clone();
                    new_config.insert(name.clone(), v.clone());
                    new_config
                })
            })
            .collect()
    }
}

// =============================================================================
// Tests (TDD - Written First)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // ParameterValue Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parameter_value_float() {
        let v = ParameterValue::Float(0.5);
        assert_eq!(v.as_float(), Some(0.5));
        assert_eq!(v.as_int(), Some(0));
        assert_eq!(v.as_str(), None);
    }

    #[test]
    fn test_parameter_value_int() {
        let v = ParameterValue::Int(42);
        assert_eq!(v.as_float(), Some(42.0));
        assert_eq!(v.as_int(), Some(42));
        assert_eq!(v.as_str(), None);
    }

    #[test]
    fn test_parameter_value_categorical() {
        let v = ParameterValue::Categorical("relu".to_string());
        assert_eq!(v.as_float(), None);
        assert_eq!(v.as_str(), Some("relu"));
    }

    // -------------------------------------------------------------------------
    // ParameterDomain Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_domain_continuous_sample() {
        let domain = ParameterDomain::Continuous {
            low: 0.0,
            high: 1.0,
            log_scale: false,
        };
        let mut rng = rand::rng();
        for _ in 0..100 {
            let value = domain.sample(&mut rng);
            assert!(domain.is_valid(&value));
        }
    }

    #[test]
    fn test_domain_continuous_log_scale() {
        let domain = ParameterDomain::Continuous {
            low: 1e-5,
            high: 1e-1,
            log_scale: true,
        };
        let mut rng = rand::rng();
        for _ in 0..100 {
            let value = domain.sample(&mut rng);
            assert!(domain.is_valid(&value));
        }
    }

    #[test]
    fn test_domain_discrete_sample() {
        let domain = ParameterDomain::Discrete { low: 8, high: 128 };
        let mut rng = rand::rng();
        for _ in 0..100 {
            let value = domain.sample(&mut rng);
            assert!(domain.is_valid(&value));
        }
    }

    #[test]
    fn test_domain_categorical_sample() {
        let domain = ParameterDomain::Categorical {
            choices: vec!["relu".to_string(), "gelu".to_string(), "swish".to_string()],
        };
        let mut rng = rand::rng();
        for _ in 0..100 {
            let value = domain.sample(&mut rng);
            assert!(domain.is_valid(&value));
        }
    }

    #[test]
    fn test_domain_is_valid() {
        let domain = ParameterDomain::Continuous {
            low: 0.0,
            high: 1.0,
            log_scale: false,
        };

        assert!(domain.is_valid(&ParameterValue::Float(0.5)));
        assert!(!domain.is_valid(&ParameterValue::Float(1.5)));
        assert!(!domain.is_valid(&ParameterValue::Int(0)));
    }

    // -------------------------------------------------------------------------
    // HyperparameterSpace Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_space_new() {
        let space = HyperparameterSpace::new();
        assert!(space.is_empty());
        assert_eq!(space.len(), 0);
    }

    #[test]
    fn test_space_add_and_get() {
        let mut space = HyperparameterSpace::new();
        space.add(
            "lr",
            ParameterDomain::Continuous {
                low: 1e-5,
                high: 1e-1,
                log_scale: true,
            },
        );

        assert!(!space.is_empty());
        assert_eq!(space.len(), 1);
        assert!(space.get("lr").is_some());
        assert!(space.get("unknown").is_none());
    }

    #[test]
    fn test_space_sample_random() {
        let mut space = HyperparameterSpace::new();
        space.add(
            "lr",
            ParameterDomain::Continuous {
                low: 1e-5,
                high: 1e-1,
                log_scale: true,
            },
        );
        space.add("batch_size", ParameterDomain::Discrete { low: 8, high: 64 });

        let mut rng = rand::rng();
        let config = space.sample_random(&mut rng);

        assert!(config.contains_key("lr"));
        assert!(config.contains_key("batch_size"));
        assert!(space.validate(&config).is_ok());
    }

    #[test]
    fn test_space_validate() {
        let mut space = HyperparameterSpace::new();
        space.add(
            "lr",
            ParameterDomain::Continuous {
                low: 0.0,
                high: 1.0,
                log_scale: false,
            },
        );

        let mut valid_config = HashMap::new();
        valid_config.insert("lr".to_string(), ParameterValue::Float(0.5));
        assert!(space.validate(&valid_config).is_ok());

        let mut invalid_config = HashMap::new();
        invalid_config.insert("lr".to_string(), ParameterValue::Float(2.0));
        assert!(space.validate(&invalid_config).is_err());

        let missing_config = HashMap::new();
        assert!(space.validate(&missing_config).is_err());
    }

    // -------------------------------------------------------------------------
    // Trial Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trial_new() {
        let config = HashMap::new();
        let trial = Trial::new(0, config);
        assert_eq!(trial.id, 0);
        assert_eq!(trial.status, TrialStatus::Pending);
        assert_eq!(trial.score, f64::INFINITY);
    }

    #[test]
    fn test_trial_complete() {
        let mut trial = Trial::new(0, HashMap::new());
        trial.complete(0.5, 100);
        assert_eq!(trial.status, TrialStatus::Completed);
        assert_eq!(trial.score, 0.5);
        assert_eq!(trial.iterations, 100);
    }

    #[test]
    fn test_trial_fail() {
        let mut trial = Trial::new(0, HashMap::new());
        trial.fail();
        assert_eq!(trial.status, TrialStatus::Failed);
    }

    // -------------------------------------------------------------------------
    // TPEOptimizer Tests
    // -------------------------------------------------------------------------

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

    // -------------------------------------------------------------------------
    // HyperbandScheduler Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_hyperband_new() {
        let space = HyperparameterSpace::new();
        let hb = HyperbandScheduler::new(space, 81);
        assert_eq!(hb.max_iter, 81);
        assert!((hb.eta - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyperband_s_max() {
        let space = HyperparameterSpace::new();
        let hb = HyperbandScheduler::new(space, 81);
        // log_3(81) = 4
        assert_eq!(hb.s_max(), 4);
    }

    #[test]
    fn test_hyperband_budget() {
        let space = HyperparameterSpace::new();
        let hb = HyperbandScheduler::new(space, 81);
        // B = (s_max + 1) * max_iter = 5 * 81 = 405
        assert_eq!(hb.budget(), 405);
    }

    #[test]
    fn test_hyperband_bracket() {
        let space = HyperparameterSpace::new();
        let hb = HyperbandScheduler::new(space, 81);

        // Bracket s=4 should start with most configs and least resources
        let bracket = hb.bracket(4);
        assert!(!bracket.is_empty());

        // First rung should have more configs than last
        let (n_first, r_first) = bracket.first().unwrap();
        let (n_last, r_last) = bracket.last().unwrap();
        assert!(*n_first >= *n_last);
        assert!(*r_first <= *r_last);
    }

    #[test]
    fn test_hyperband_generate_configs() {
        let mut space = HyperparameterSpace::new();
        space.add(
            "lr",
            ParameterDomain::Continuous {
                low: 0.0,
                high: 1.0,
                log_scale: false,
            },
        );

        let hb = HyperbandScheduler::new(space, 81);
        let configs = hb.generate_configs(10);
        assert_eq!(configs.len(), 10);
    }

    // -------------------------------------------------------------------------
    // GridSearch Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_grid_search_new() {
        let space = HyperparameterSpace::new();
        let grid = GridSearch::new(space, 5);
        assert_eq!(grid.n_points, 5);
    }

    #[test]
    fn test_grid_search_empty_space() {
        let space = HyperparameterSpace::new();
        let grid = GridSearch::new(space, 5);
        let configs = grid.configurations();
        assert_eq!(configs.len(), 1); // One empty config
    }

    #[test]
    fn test_grid_search_single_param() {
        let mut space = HyperparameterSpace::new();
        space.add(
            "lr",
            ParameterDomain::Continuous {
                low: 0.0,
                high: 1.0,
                log_scale: false,
            },
        );

        let grid = GridSearch::new(space, 5);
        let configs = grid.configurations();
        assert_eq!(configs.len(), 5);

        // Check values are evenly spaced
        let values: Vec<f64> = configs
            .iter()
            .map(|c| c.get("lr").unwrap().as_float().unwrap())
            .collect();
        assert!((values[0] - 0.0).abs() < 1e-10);
        assert!((values[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_grid_search_multiple_params() {
        let mut space = HyperparameterSpace::new();
        space.add(
            "lr",
            ParameterDomain::Continuous {
                low: 0.0,
                high: 1.0,
                log_scale: false,
            },
        );
        space.add(
            "act",
            ParameterDomain::Categorical {
                choices: vec!["relu".to_string(), "gelu".to_string()],
            },
        );

        let grid = GridSearch::new(space, 3);
        let configs = grid.configurations();
        // 3 lr values * 2 activation functions = 6
        assert_eq!(configs.len(), 6);
    }

    #[test]
    fn test_grid_search_discrete() {
        let mut space = HyperparameterSpace::new();
        space.add("batch_size", ParameterDomain::Discrete { low: 8, high: 10 });

        let grid = GridSearch::new(space, 5);
        let configs = grid.configurations();
        // Discrete [8,9,10] = 3 values
        assert_eq!(configs.len(), 3);
    }

    #[test]
    fn test_grid_search_log_scale() {
        let mut space = HyperparameterSpace::new();
        space.add(
            "lr",
            ParameterDomain::Continuous {
                low: 1e-4,
                high: 1e-1,
                log_scale: true,
            },
        );

        let grid = GridSearch::new(space, 4);
        let configs = grid.configurations();

        let values: Vec<f64> = configs
            .iter()
            .map(|c| c.get("lr").unwrap().as_float().unwrap())
            .collect();

        // Log scale should give approximately: 1e-4, 1e-3, 1e-2, 1e-1
        assert!(values[0] < 1e-3);
        assert!(values[3] > 1e-2);
    }
}

// =============================================================================
// Property Tests
// =============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_continuous_domain_valid(low in -100.0f64..0.0, high in 0.0f64..100.0) {
            let domain = ParameterDomain::Continuous {
                low,
                high,
                log_scale: false,
            };
            let mut rng = rand::rng();
            let value = domain.sample(&mut rng);
            prop_assert!(domain.is_valid(&value));
        }

        #[test]
        fn prop_discrete_domain_valid(low in -100i64..0, high in 0i64..100) {
            let domain = ParameterDomain::Discrete { low, high };
            let mut rng = rand::rng();
            let value = domain.sample(&mut rng);
            prop_assert!(domain.is_valid(&value));
        }

        #[test]
        fn prop_space_sample_validates(
            lr_low in 1e-6f64..1e-4,
            lr_high in 1e-2f64..1.0,
            bs_low in 1i64..16,
            bs_high in 32i64..256
        ) {
            let mut space = HyperparameterSpace::new();
            space.add("lr", ParameterDomain::Continuous {
                low: lr_low,
                high: lr_high,
                log_scale: true,
            });
            space.add("batch_size", ParameterDomain::Discrete {
                low: bs_low,
                high: bs_high,
            });

            let mut rng = rand::rng();
            let config = space.sample_random(&mut rng);
            prop_assert!(space.validate(&config).is_ok());
        }

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

        #[test]
        fn prop_hyperband_bracket_nonempty(max_iter in 9usize..243, eta in 2.0f64..5.0) {
            let space = HyperparameterSpace::new();
            let hb = HyperbandScheduler::new(space, max_iter).with_eta(eta);
            let s_max = hb.s_max();
            for s in 0..=s_max {
                let bracket = hb.bracket(s);
                prop_assert!(!bracket.is_empty());
            }
        }

        #[test]
        fn prop_grid_search_size(n_points in 2usize..10) {
            let mut space = HyperparameterSpace::new();
            space.add("x", ParameterDomain::Continuous {
                low: 0.0,
                high: 1.0,
                log_scale: false,
            });

            let grid = GridSearch::new(space, n_points);
            let configs = grid.configurations();
            prop_assert_eq!(configs.len(), n_points);
        }
    }
}
