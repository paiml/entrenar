//! Core HPO types

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::error::{HPOError, Result};

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
    // Additional Space Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_space_iter() {
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

        let param_names: Vec<_> = space.iter().map(|(name, _)| name.clone()).collect();
        assert_eq!(param_names.len(), 2);
        assert!(param_names.contains(&"lr".to_string()));
        assert!(param_names.contains(&"batch_size".to_string()));
    }

    // -------------------------------------------------------------------------
    // Domain Validation Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_domain_is_valid_type_mismatch() {
        let domain = ParameterDomain::Discrete { low: 0, high: 10 };
        // Float value for discrete domain
        assert!(!domain.is_valid(&ParameterValue::Float(5.0)));

        let domain = ParameterDomain::Categorical {
            choices: vec!["a".to_string()],
        };
        // Int value for categorical domain
        assert!(!domain.is_valid(&ParameterValue::Int(0)));
    }

    // -------------------------------------------------------------------------
    // Serde Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parameter_value_serde() {
        let v = ParameterValue::Float(0.5);
        let json = serde_json::to_string(&v).unwrap();
        let parsed: ParameterValue = serde_json::from_str(&json).unwrap();
        assert_eq!(v, parsed);

        let v = ParameterValue::Int(42);
        let json = serde_json::to_string(&v).unwrap();
        let parsed: ParameterValue = serde_json::from_str(&json).unwrap();
        assert_eq!(v, parsed);

        let v = ParameterValue::Categorical("relu".to_string());
        let json = serde_json::to_string(&v).unwrap();
        let parsed: ParameterValue = serde_json::from_str(&json).unwrap();
        assert_eq!(v, parsed);
    }

    #[test]
    fn test_parameter_domain_serde() {
        let domain = ParameterDomain::Continuous {
            low: 0.0,
            high: 1.0,
            log_scale: true,
        };
        let json = serde_json::to_string(&domain).unwrap();
        let parsed: ParameterDomain = serde_json::from_str(&json).unwrap();
        match parsed {
            ParameterDomain::Continuous { log_scale, .. } => assert!(log_scale),
            _ => panic!("Wrong domain type"),
        }

        let domain = ParameterDomain::Discrete { low: 8, high: 128 };
        let json = serde_json::to_string(&domain).unwrap();
        let _parsed: ParameterDomain = serde_json::from_str(&json).unwrap();

        let domain = ParameterDomain::Categorical {
            choices: vec!["a".to_string(), "b".to_string()],
        };
        let json = serde_json::to_string(&domain).unwrap();
        let _parsed: ParameterDomain = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_hyperparameter_space_serde() {
        let mut space = HyperparameterSpace::new();
        space.add(
            "lr",
            ParameterDomain::Continuous {
                low: 0.0,
                high: 1.0,
                log_scale: false,
            },
        );

        let json = serde_json::to_string(&space).unwrap();
        let parsed: HyperparameterSpace = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn test_trial_serde() {
        let mut config = HashMap::new();
        config.insert("lr".to_string(), ParameterValue::Float(0.01));
        let mut trial = Trial::new(0, config);
        trial.complete(0.5, 100);

        let json = serde_json::to_string(&trial).unwrap();
        let parsed: Trial = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, 0);
        assert_eq!(parsed.score, 0.5);
        assert_eq!(parsed.status, TrialStatus::Completed);
    }

    #[test]
    fn test_trial_status_serde() {
        for status in [
            TrialStatus::Pending,
            TrialStatus::Running,
            TrialStatus::Completed,
            TrialStatus::Failed,
            TrialStatus::Pruned,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let parsed: TrialStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, parsed);
        }
    }

    #[test]
    fn test_search_strategy_serde() {
        let strategy = SearchStrategy::Grid;
        let json = serde_json::to_string(&strategy).unwrap();
        let _parsed: SearchStrategy = serde_json::from_str(&json).unwrap();

        let strategy = SearchStrategy::Random { n_samples: 100 };
        let json = serde_json::to_string(&strategy).unwrap();
        let _parsed: SearchStrategy = serde_json::from_str(&json).unwrap();

        let strategy = SearchStrategy::Bayesian {
            n_initial: 10,
            acquisition: AcquisitionFunction::ExpectedImprovement,
            surrogate: SurrogateModel::TPE,
        };
        let json = serde_json::to_string(&strategy).unwrap();
        let _parsed: SearchStrategy = serde_json::from_str(&json).unwrap();

        let strategy = SearchStrategy::Hyperband {
            max_iter: 81,
            eta: 3.0,
        };
        let json = serde_json::to_string(&strategy).unwrap();
        let _parsed: SearchStrategy = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_acquisition_function_serde() {
        for acq in [
            AcquisitionFunction::ExpectedImprovement,
            AcquisitionFunction::UpperConfidenceBound { kappa: 2.576 },
            AcquisitionFunction::ProbabilityOfImprovement,
        ] {
            let json = serde_json::to_string(&acq).unwrap();
            let _parsed: AcquisitionFunction = serde_json::from_str(&json).unwrap();
        }
    }

    #[test]
    fn test_surrogate_model_serde() {
        for surrogate in [
            SurrogateModel::TPE,
            SurrogateModel::GaussianProcess,
            SurrogateModel::RandomForest { n_trees: 100 },
        ] {
            let json = serde_json::to_string(&surrogate).unwrap();
            let _parsed: SurrogateModel = serde_json::from_str(&json).unwrap();
        }
    }
}

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
    }
}
