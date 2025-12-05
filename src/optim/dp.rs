//! Differential Privacy Module (MLOPS-015)
//!
//! DP-SGD implementation following Abadi et al. (2016) for privacy-preserving training.
//!
//! # Toyota Way: 自働化 (Jidoka)
//!
//! Built-in privacy protection stops data leakage. The privacy budget acts as
//! an Andon system, halting training when privacy guarantees would be violated.
//!
//! # Example
//!
//! ```ignore
//! use entrenar::optim::dp::{DpSgdConfig, DpSgd, PrivacyBudget};
//! use entrenar::optim::SGD;
//!
//! let config = DpSgdConfig::new()
//!     .with_max_grad_norm(1.0)
//!     .with_noise_multiplier(1.1)
//!     .with_budget(PrivacyBudget::new(8.0, 1e-5));
//!
//! let dp_sgd = DpSgd::new(SGD::new(0.01), config);
//! ```
//!
//! # References
//!
//! [3] Abadi et al. (2016) - Deep Learning with Differential Privacy
//! [4] Mironov (2017) - Renyi Differential Privacy

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use thiserror::Error;

// =============================================================================
// Core Types
// =============================================================================

/// DP errors
#[derive(Debug, Error)]
pub enum DpError {
    #[error("Privacy budget exhausted: spent {spent:.4} > allowed {budget:.4}")]
    BudgetExhausted { spent: f64, budget: f64 },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Gradient computation failed: {0}")]
    GradientError(String),

    #[error("DP error: {0}")]
    Internal(String),
}

/// Result type for DP operations
pub type Result<T> = std::result::Result<T, DpError>;

/// Privacy budget (epsilon, delta)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PrivacyBudget {
    /// Privacy loss parameter epsilon (smaller = more private)
    pub epsilon: f64,
    /// Probability of privacy breach delta (smaller = more private)
    pub delta: f64,
}

impl PrivacyBudget {
    /// Create a new privacy budget
    pub fn new(epsilon: f64, delta: f64) -> Self {
        Self { epsilon, delta }
    }

    /// Check if the budget allows the given epsilon
    pub fn allows(&self, spent: f64) -> bool {
        spent <= self.epsilon
    }

    /// Get remaining budget
    pub fn remaining(&self, spent: f64) -> f64 {
        (self.epsilon - spent).max(0.0)
    }
}

impl Default for PrivacyBudget {
    fn default() -> Self {
        // Commonly used default: (8.0, 1e-5)
        Self {
            epsilon: 8.0,
            delta: 1e-5,
        }
    }
}

/// RDP (Renyi Differential Privacy) accountant
///
/// Provides tighter privacy bounds than basic composition.
/// Based on Mironov (2017).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdpAccountant {
    /// RDP orders to track
    orders: Vec<f64>,
    /// Accumulated RDP values for each order
    rdp: Vec<f64>,
    /// Number of steps taken
    steps: usize,
}

impl RdpAccountant {
    /// Create a new RDP accountant
    pub fn new() -> Self {
        // Standard orders for RDP accounting
        let orders: Vec<f64> = (2..=256).map(f64::from).collect();
        let rdp = vec![0.0; orders.len()];
        Self {
            orders,
            rdp,
            steps: 0,
        }
    }

    /// Create with custom orders
    pub fn with_orders(orders: Vec<f64>) -> Self {
        let rdp = vec![0.0; orders.len()];
        Self {
            orders,
            rdp,
            steps: 0,
        }
    }

    /// Record a training step
    pub fn step(&mut self, noise_multiplier: f64, sample_rate: f64) {
        for (i, &alpha) in self.orders.iter().enumerate() {
            let rdp_step = compute_rdp_gaussian(noise_multiplier, sample_rate, alpha);
            self.rdp[i] += rdp_step;
        }
        self.steps += 1;
    }

    /// Get privacy spent as (epsilon, delta)
    pub fn get_privacy_spent(&self, delta: f64) -> (f64, f64) {
        rdp_to_dp(&self.orders, &self.rdp, delta)
    }

    /// Get number of steps
    pub fn n_steps(&self) -> usize {
        self.steps
    }

    /// Reset the accountant
    pub fn reset(&mut self) {
        for r in &mut self.rdp {
            *r = 0.0;
        }
        self.steps = 0;
    }
}

impl Default for RdpAccountant {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute RDP of Gaussian mechanism for subsampled data
fn compute_rdp_gaussian(noise_multiplier: f64, sample_rate: f64, alpha: f64) -> f64 {
    if noise_multiplier <= 0.0 || sample_rate <= 0.0 {
        return f64::INFINITY;
    }

    // RDP of Gaussian mechanism: alpha / (2 * sigma^2)
    // For subsampled mechanism, we use Poisson subsampling bound
    let sigma = noise_multiplier;

    if sample_rate >= 1.0 {
        // Full batch
        alpha / (2.0 * sigma.powi(2))
    } else {
        // Subsampled: upper bound using privacy amplification
        // Simplified bound: q^2 * alpha / (2 * sigma^2)
        // where q is the sampling rate
        let q = sample_rate;

        // More accurate bound using log1p for numerical stability
        if alpha <= 1.0 {
            return f64::INFINITY;
        }

        // Approximate subsampled RDP
        let log_a = (alpha - 1.0)
            * ((alpha * q.powi(2)) / (2.0 * sigma.powi(2)))
                .min(1.0)
                .ln_1p();
        log_a / (alpha - 1.0)
    }
}

/// Convert RDP to (epsilon, delta)-DP
fn rdp_to_dp(orders: &[f64], rdp: &[f64], delta: f64) -> (f64, f64) {
    if delta <= 0.0 || orders.is_empty() {
        return (f64::INFINITY, delta);
    }

    let log_delta = delta.ln();

    // Find optimal order
    let mut min_epsilon = f64::INFINITY;
    for (&alpha, &rdp_alpha) in orders.iter().zip(rdp.iter()) {
        if alpha <= 1.0 {
            continue;
        }
        // epsilon = rdp_alpha - (log(delta) + log(alpha - 1)) / (alpha - 1) + log(alpha / (alpha - 1))
        let epsilon = rdp_alpha + (1.0 / (alpha - 1.0)) * ((alpha - 1.0) / alpha).ln()
            - (log_delta + (alpha - 1.0).ln()) / (alpha - 1.0);

        if epsilon < min_epsilon && epsilon >= 0.0 {
            min_epsilon = epsilon;
        }
    }

    (min_epsilon, delta)
}

// =============================================================================
// Gradient Operations
// =============================================================================

/// Clip gradient to max norm (per-sample)
pub fn clip_gradient(grad: &[f64], max_norm: f64) -> Vec<f64> {
    let norm: f64 = grad.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

    if norm > max_norm {
        let scale = max_norm / norm;
        grad.iter().map(|x| x * scale).collect()
    } else {
        grad.to_vec()
    }
}

/// Compute L2 norm of gradient
pub fn grad_norm(grad: &[f64]) -> f64 {
    grad.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
}

/// Add Gaussian noise to gradient
pub fn add_gaussian_noise<R: Rng>(grad: &[f64], std_dev: f64, rng: &mut R) -> Vec<f64> {
    grad.iter()
        .map(|&x| {
            // Box-Muller transform for Gaussian noise
            let u1: f64 = rng.random::<f64>().max(1e-10);
            let u2: f64 = rng.random::<f64>();
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos() * std_dev;
            x + noise
        })
        .collect()
}

// =============================================================================
// DP-SGD Configuration
// =============================================================================

/// Configuration for DP-SGD
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpSgdConfig {
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f64,
    /// Noise multiplier (sigma = noise_multiplier * max_grad_norm)
    pub noise_multiplier: f64,
    /// Privacy budget
    pub budget: PrivacyBudget,
    /// Sampling rate (batch_size / dataset_size)
    pub sample_rate: f64,
    /// Whether to stop training when budget is exhausted
    pub strict_budget: bool,
}

impl DpSgdConfig {
    /// Create a new DP-SGD configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum gradient norm
    pub fn with_max_grad_norm(mut self, norm: f64) -> Self {
        self.max_grad_norm = norm;
        self
    }

    /// Set noise multiplier
    pub fn with_noise_multiplier(mut self, multiplier: f64) -> Self {
        self.noise_multiplier = multiplier.max(0.0);
        self
    }

    /// Set privacy budget
    pub fn with_budget(mut self, budget: PrivacyBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Set sampling rate
    pub fn with_sample_rate(mut self, rate: f64) -> Self {
        self.sample_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set strict budget enforcement
    pub fn with_strict_budget(mut self, strict: bool) -> Self {
        self.strict_budget = strict;
        self
    }

    /// Compute noise standard deviation
    pub fn noise_std(&self) -> f64 {
        self.noise_multiplier * self.max_grad_norm
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.max_grad_norm <= 0.0 {
            return Err(DpError::InvalidConfig(
                "max_grad_norm must be positive".to_string(),
            ));
        }
        if self.noise_multiplier < 0.0 {
            return Err(DpError::InvalidConfig(
                "noise_multiplier must be non-negative".to_string(),
            ));
        }
        if self.budget.epsilon <= 0.0 {
            return Err(DpError::InvalidConfig(
                "epsilon must be positive".to_string(),
            ));
        }
        if self.budget.delta <= 0.0 || self.budget.delta >= 1.0 {
            return Err(DpError::InvalidConfig(
                "delta must be in (0, 1)".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for DpSgdConfig {
    fn default() -> Self {
        Self {
            max_grad_norm: 1.0,
            noise_multiplier: 1.1,
            budget: PrivacyBudget::default(),
            sample_rate: 0.01, // 1% batch size
            strict_budget: true,
        }
    }
}

// =============================================================================
// DP-SGD Wrapper
// =============================================================================

/// Differentially private SGD wrapper
///
/// Wraps any optimizer with DP guarantees by:
/// 1. Per-sample gradient clipping
/// 2. Adding calibrated Gaussian noise
/// 3. Privacy accounting
#[derive(Debug, Clone)]
pub struct DpSgd {
    /// DP configuration
    config: DpSgdConfig,
    /// Privacy accountant
    accountant: RdpAccountant,
    /// Learning rate
    learning_rate: f64,
}

impl DpSgd {
    /// Create a new DP-SGD optimizer
    pub fn new(learning_rate: f64, config: DpSgdConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            accountant: RdpAccountant::new(),
            learning_rate,
        })
    }

    /// Get current privacy spent
    pub fn privacy_spent(&self) -> (f64, f64) {
        self.accountant.get_privacy_spent(self.config.budget.delta)
    }

    /// Get current epsilon
    pub fn current_epsilon(&self) -> f64 {
        self.privacy_spent().0
    }

    /// Get remaining budget
    pub fn remaining_budget(&self) -> f64 {
        self.config.budget.remaining(self.current_epsilon())
    }

    /// Check if budget is exhausted
    pub fn is_budget_exhausted(&self) -> bool {
        !self.config.budget.allows(self.current_epsilon())
    }

    /// Get number of training steps
    pub fn n_steps(&self) -> usize {
        self.accountant.n_steps()
    }

    /// Get configuration
    pub fn config(&self) -> &DpSgdConfig {
        &self.config
    }

    /// Process per-sample gradients with DP mechanism
    ///
    /// Returns the privatized aggregated gradient
    pub fn privatize_gradients(&mut self, per_sample_grads: &[Vec<f64>]) -> Result<Vec<f64>> {
        if per_sample_grads.is_empty() {
            return Err(DpError::GradientError("No gradients provided".to_string()));
        }

        // Check budget
        if self.config.strict_budget && self.is_budget_exhausted() {
            return Err(DpError::BudgetExhausted {
                spent: self.current_epsilon(),
                budget: self.config.budget.epsilon,
            });
        }

        let n_samples = per_sample_grads.len();
        let grad_dim = per_sample_grads[0].len();

        // Step 1: Clip each per-sample gradient
        let clipped: Vec<Vec<f64>> = per_sample_grads
            .iter()
            .map(|g| clip_gradient(g, self.config.max_grad_norm))
            .collect();

        // Step 2: Average clipped gradients
        let mut averaged = vec![0.0; grad_dim];
        for g in &clipped {
            for (i, &val) in g.iter().enumerate() {
                averaged[i] += val / n_samples as f64;
            }
        }

        // Step 3: Add Gaussian noise
        let mut rng = rand::rng();
        let noise_std = self.config.noise_std() / n_samples as f64;
        let noised = add_gaussian_noise(&averaged, noise_std, &mut rng);

        // Step 4: Update privacy accounting
        self.accountant
            .step(self.config.noise_multiplier, self.config.sample_rate);

        Ok(noised)
    }

    /// Apply gradient update to parameters
    pub fn apply_update(&self, params: &mut [f64], grad: &[f64]) {
        for (p, g) in params.iter_mut().zip(grad.iter()) {
            *p -= self.learning_rate * g;
        }
    }

    /// Full DP-SGD step: privatize gradients and update parameters
    pub fn step(
        &mut self,
        params: &mut [f64],
        per_sample_grads: &[Vec<f64>],
    ) -> Result<(f64, f64)> {
        let grad = self.privatize_gradients(per_sample_grads)?;
        self.apply_update(params, &grad);
        Ok(self.privacy_spent())
    }

    /// Reset privacy accountant
    pub fn reset(&mut self) {
        self.accountant.reset();
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Estimate required noise multiplier for target epsilon
///
/// Given target (epsilon, delta), dataset size, batch size, and epochs,
/// estimate the noise multiplier needed.
pub fn estimate_noise_multiplier(
    target_epsilon: f64,
    delta: f64,
    dataset_size: usize,
    batch_size: usize,
    epochs: usize,
) -> f64 {
    if target_epsilon <= 0.0 || delta <= 0.0 {
        return f64::INFINITY;
    }

    let sample_rate = batch_size as f64 / dataset_size as f64;
    let steps = (epochs * dataset_size) / batch_size;

    // Binary search for noise multiplier
    let mut low = 0.1;
    let mut high = 100.0;

    for _ in 0..100 {
        let mid = f64::midpoint(low, high);

        // Simulate privacy accounting
        let mut accountant = RdpAccountant::new();
        for _ in 0..steps {
            accountant.step(mid, sample_rate);
        }

        let (epsilon, _) = accountant.get_privacy_spent(delta);

        if epsilon < target_epsilon {
            high = mid;
        } else {
            low = mid;
        }

        if (high - low) < 0.01 {
            break;
        }
    }

    high * 1.1 // Add 10% safety margin
}

/// Privacy cost per step for given parameters
pub fn privacy_cost_per_step(noise_multiplier: f64, sample_rate: f64, delta: f64) -> f64 {
    let mut accountant = RdpAccountant::new();
    accountant.step(noise_multiplier, sample_rate);
    accountant.get_privacy_spent(delta).0
}

// =============================================================================
// Tests (TDD - Written First)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // PrivacyBudget Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_privacy_budget_new() {
        let budget = PrivacyBudget::new(8.0, 1e-5);
        assert_eq!(budget.epsilon, 8.0);
        assert_eq!(budget.delta, 1e-5);
    }

    #[test]
    fn test_privacy_budget_default() {
        let budget = PrivacyBudget::default();
        assert_eq!(budget.epsilon, 8.0);
        assert_eq!(budget.delta, 1e-5);
    }

    #[test]
    fn test_privacy_budget_allows() {
        let budget = PrivacyBudget::new(8.0, 1e-5);
        assert!(budget.allows(5.0));
        assert!(budget.allows(8.0));
        assert!(!budget.allows(10.0));
    }

    #[test]
    fn test_privacy_budget_remaining() {
        let budget = PrivacyBudget::new(8.0, 1e-5);
        assert!((budget.remaining(3.0) - 5.0).abs() < 1e-10);
        assert!((budget.remaining(10.0) - 0.0).abs() < 1e-10);
    }

    // -------------------------------------------------------------------------
    // RdpAccountant Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_rdp_accountant_new() {
        let accountant = RdpAccountant::new();
        assert_eq!(accountant.n_steps(), 0);
        // Initially no privacy has been spent (RDP values are 0)
        assert_eq!(accountant.rdp.iter().sum::<f64>(), 0.0);
    }

    #[test]
    fn test_rdp_accountant_step() {
        let mut accountant = RdpAccountant::new();
        accountant.step(1.0, 0.01);
        assert_eq!(accountant.n_steps(), 1);

        let (epsilon, delta) = accountant.get_privacy_spent(1e-5);
        assert!(epsilon > 0.0);
        assert_eq!(delta, 1e-5);
    }

    #[test]
    fn test_rdp_accountant_accumulates() {
        let mut accountant = RdpAccountant::new();
        accountant.step(1.0, 0.01);
        let (e1, _) = accountant.get_privacy_spent(1e-5);

        accountant.step(1.0, 0.01);
        let (e2, _) = accountant.get_privacy_spent(1e-5);

        // Privacy loss should increase
        assert!(e2 > e1);
    }

    #[test]
    fn test_rdp_accountant_reset() {
        let mut accountant = RdpAccountant::new();
        accountant.step(1.0, 0.01);
        accountant.step(1.0, 0.01);
        assert_eq!(accountant.n_steps(), 2);

        accountant.reset();
        assert_eq!(accountant.n_steps(), 0);
    }

    #[test]
    fn test_rdp_higher_noise_lower_epsilon() {
        let mut acc1 = RdpAccountant::new();
        let mut acc2 = RdpAccountant::new();

        // Same number of steps
        for _ in 0..100 {
            acc1.step(1.0, 0.01); // Lower noise
            acc2.step(2.0, 0.01); // Higher noise
        }

        let (e1, _) = acc1.get_privacy_spent(1e-5);
        let (e2, _) = acc2.get_privacy_spent(1e-5);

        // Higher noise should give lower epsilon
        assert!(e2 < e1);
    }

    // -------------------------------------------------------------------------
    // Gradient Operations Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_clip_gradient_within_norm() {
        let grad = vec![0.3, 0.4, 0.0];
        let clipped = clip_gradient(&grad, 1.0);
        // Norm is 0.5, within limit
        assert!((clipped[0] - 0.3).abs() < 1e-10);
        assert!((clipped[1] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_clip_gradient_exceeds_norm() {
        let grad = vec![3.0, 4.0, 0.0];
        let clipped = clip_gradient(&grad, 1.0);
        // Norm is 5.0, should be clipped to 1.0
        let norm = grad_norm(&clipped);
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip_gradient_preserves_direction() {
        let grad = vec![3.0, 4.0, 0.0];
        let clipped = clip_gradient(&grad, 1.0);
        // Direction should be preserved
        let ratio = clipped[0] / clipped[1];
        assert!((ratio - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_grad_norm() {
        let grad = vec![3.0, 4.0];
        assert!((grad_norm(&grad) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_gaussian_noise() {
        let grad = vec![1.0, 2.0, 3.0];
        let mut rng = rand::rng();
        let noised = add_gaussian_noise(&grad, 0.1, &mut rng);

        // Should have same length
        assert_eq!(noised.len(), 3);

        // Should be different from original (with high probability)
        let diff: f64 = grad
            .iter()
            .zip(noised.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0);
    }

    // -------------------------------------------------------------------------
    // DpSgdConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dp_config_new() {
        let config = DpSgdConfig::new();
        assert!((config.max_grad_norm - 1.0).abs() < 1e-10);
        assert!((config.noise_multiplier - 1.1).abs() < 1e-10);
    }

    #[test]
    fn test_dp_config_builder() {
        let config = DpSgdConfig::new()
            .with_max_grad_norm(2.0)
            .with_noise_multiplier(1.5)
            .with_budget(PrivacyBudget::new(4.0, 1e-6))
            .with_sample_rate(0.05);

        assert!((config.max_grad_norm - 2.0).abs() < 1e-10);
        assert!((config.noise_multiplier - 1.5).abs() < 1e-10);
        assert!((config.budget.epsilon - 4.0).abs() < 1e-10);
        assert!((config.sample_rate - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_dp_config_validate_valid() {
        let config = DpSgdConfig::new();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_dp_config_validate_invalid_norm() {
        let config = DpSgdConfig::new().with_max_grad_norm(-1.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dp_config_validate_invalid_epsilon() {
        let config = DpSgdConfig::new().with_budget(PrivacyBudget::new(-1.0, 1e-5));
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dp_config_noise_std() {
        let config = DpSgdConfig::new()
            .with_max_grad_norm(2.0)
            .with_noise_multiplier(1.5);
        assert!((config.noise_std() - 3.0).abs() < 1e-10);
    }

    // -------------------------------------------------------------------------
    // DpSgd Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dp_sgd_new() {
        let config = DpSgdConfig::new();
        let dp_sgd = DpSgd::new(0.01, config).unwrap();
        assert_eq!(dp_sgd.n_steps(), 0);
        assert!(!dp_sgd.is_budget_exhausted());
    }

    #[test]
    fn test_dp_sgd_privatize_gradients() {
        let config = DpSgdConfig::new()
            .with_max_grad_norm(1.0)
            .with_noise_multiplier(0.1);
        let mut dp_sgd = DpSgd::new(0.01, config).unwrap();

        let grads = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.2, 0.3, 0.1],
            vec![0.3, 0.1, 0.2],
        ];

        let result = dp_sgd.privatize_gradients(&grads);
        assert!(result.is_ok());

        let private_grad = result.unwrap();
        assert_eq!(private_grad.len(), 3);
        assert_eq!(dp_sgd.n_steps(), 1);
    }

    #[test]
    fn test_dp_sgd_step() {
        let config = DpSgdConfig::new()
            .with_max_grad_norm(1.0)
            .with_noise_multiplier(0.1);
        let mut dp_sgd = DpSgd::new(0.1, config).unwrap();

        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![vec![0.1, 0.1, 0.1], vec![0.1, 0.1, 0.1]];

        let result = dp_sgd.step(&mut params, &grads);
        assert!(result.is_ok());

        // Params should have changed
        let diff: f64 = params
            .iter()
            .zip(&[1.0, 2.0, 3.0])
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_dp_sgd_privacy_accumulates() {
        let config = DpSgdConfig::new();
        let mut dp_sgd = DpSgd::new(0.01, config).unwrap();

        let grads = vec![vec![0.1, 0.2, 0.3]];

        let (e1, _) = dp_sgd.privacy_spent();

        dp_sgd.privatize_gradients(&grads).unwrap();
        let (e2, _) = dp_sgd.privacy_spent();

        dp_sgd.privatize_gradients(&grads).unwrap();
        let (e3, _) = dp_sgd.privacy_spent();

        // Privacy loss should increase monotonically
        assert!(e3 > e2);
        assert!(e2 > e1);
    }

    #[test]
    fn test_dp_sgd_budget_exhaustion() {
        // Use very tight budget
        let config = DpSgdConfig::new()
            .with_budget(PrivacyBudget::new(0.1, 1e-5))
            .with_noise_multiplier(0.1)
            .with_strict_budget(true);
        let mut dp_sgd = DpSgd::new(0.01, config).unwrap();

        let grads = vec![vec![0.1, 0.2, 0.3]];

        // Run until budget exhausted
        let mut exhausted = false;
        for _ in 0..1000 {
            match dp_sgd.privatize_gradients(&grads) {
                Err(DpError::BudgetExhausted { .. }) => {
                    exhausted = true;
                    break;
                }
                Ok(_) => {}
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }

        assert!(exhausted);
    }

    #[test]
    fn test_dp_sgd_reset() {
        let config = DpSgdConfig::new();
        let mut dp_sgd = DpSgd::new(0.01, config).unwrap();

        let grads = vec![vec![0.1, 0.2, 0.3]];
        dp_sgd.privatize_gradients(&grads).unwrap();
        dp_sgd.privatize_gradients(&grads).unwrap();

        assert!(dp_sgd.n_steps() > 0);

        dp_sgd.reset();
        assert_eq!(dp_sgd.n_steps(), 0);
    }

    // -------------------------------------------------------------------------
    // Utility Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_estimate_noise_multiplier() {
        let noise = estimate_noise_multiplier(
            8.0,   // target epsilon
            1e-5,  // delta
            60000, // dataset size (e.g., MNIST)
            256,   // batch size
            10,    // epochs
        );

        // Should return a reasonable value
        assert!(noise > 0.0);
        assert!(noise < 100.0);
    }

    #[test]
    fn test_privacy_cost_per_step() {
        let cost = privacy_cost_per_step(1.0, 0.01, 1e-5);
        assert!(cost > 0.0);
        assert!(cost < 1.0); // Single step shouldn't use much budget
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
        fn prop_clip_gradient_norm_bounded(
            grad in prop::collection::vec(-100.0f64..100.0, 1..50),
            max_norm in 0.1f64..10.0
        ) {
            let clipped = clip_gradient(&grad, max_norm);
            let norm = grad_norm(&clipped);
            prop_assert!(norm <= max_norm + 1e-10);
        }

        #[test]
        fn prop_privacy_budget_remaining_non_negative(
            epsilon in 0.1f64..10.0,
            delta in 1e-8f64..1e-3,
            spent in 0.0f64..20.0
        ) {
            let budget = PrivacyBudget::new(epsilon, delta);
            prop_assert!(budget.remaining(spent) >= 0.0);
        }

        #[test]
        fn prop_rdp_accountant_monotonic(
            noise_mult in 0.1f64..10.0,
            sample_rate in 0.001f64..0.1,
            steps in 1usize..50
        ) {
            let mut accountant = RdpAccountant::new();
            let mut prev_epsilon = 0.0f64;

            for _ in 0..steps {
                accountant.step(noise_mult, sample_rate);
                let (epsilon, _) = accountant.get_privacy_spent(1e-5);
                prop_assert!(epsilon >= prev_epsilon);
                prev_epsilon = epsilon;
            }
        }

        #[test]
        fn prop_higher_noise_lower_privacy_cost(
            noise1 in 0.1f64..5.0,
            sample_rate in 0.001f64..0.1
        ) {
            let noise2 = noise1 * 2.0; // Higher noise

            let cost1 = privacy_cost_per_step(noise1, sample_rate, 1e-5);
            let cost2 = privacy_cost_per_step(noise2, sample_rate, 1e-5);

            // Higher noise should have lower or equal privacy cost
            prop_assert!(cost2 <= cost1 + 0.01);
        }

        #[test]
        fn prop_gradient_privatization_preserves_dimension(
            n_samples in 1usize..10,
            dim in 1usize..100
        ) {
            let config = DpSgdConfig::new()
                .with_noise_multiplier(0.1)
                .with_strict_budget(false);
            let mut dp_sgd = DpSgd::new(0.01, config).unwrap();

            let grads: Vec<Vec<f64>> = (0..n_samples)
                .map(|_| vec![0.1; dim])
                .collect();

            let result = dp_sgd.privatize_gradients(&grads).unwrap();
            prop_assert_eq!(result.len(), dim);
        }
    }
}
