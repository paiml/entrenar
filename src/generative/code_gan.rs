//! Generative Adversarial Network for Code Generation
//!
//! Implements a GAN architecture for generating valid Rust AST candidates:
//! - Generator: Maps latent vectors to Rust AST token sequences
//! - Discriminator: Classifies code as real (valid) or fake (invalid)
//!
//! # Architecture
//!
//! ```text
//! Latent Vector z ─┬─► Generator ─► AST Tokens ─┬─► Discriminator ─► Valid/Invalid
//!                  │                            │
//!                  │   Real AST Samples ────────┘
//!                  │
//!                  └── (sampled from N(0, I))
//! ```
//!
//! # Example
//!
//! ```rust
//! use entrenar::generative::{CodeGan, CodeGanConfig};
//!
//! let config = CodeGanConfig::default();
//! let mut gan = CodeGan::new(config);
//!
//! // Training loop would alternate between generator and discriminator updates
//! ```

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Type alias for discriminator weights structure
type DiscriminatorWeights = (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>);

/// Latent code representation (vector in latent space)
#[derive(Debug, Clone, PartialEq)]
pub struct LatentCode {
    /// The latent vector
    pub vector: Vec<f32>,
}

impl LatentCode {
    /// Create a new latent code from a vector
    #[must_use]
    pub fn new(vector: Vec<f32>) -> Self {
        Self { vector }
    }

    /// Sample from standard normal distribution using Box-Muller transform
    pub fn sample<R: Rng>(rng: &mut R, dim: usize) -> Self {
        let vector: Vec<f32> = (0..dim)
            .map(|_| {
                // Box-Muller transform for standard normal
                let u1: f64 = rng.random::<f64>().max(1e-10);
                let u2: f64 = rng.random::<f64>();
                ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
            })
            .collect();
        Self { vector }
    }

    /// Dimension of the latent code
    #[must_use]
    pub fn dim(&self) -> usize {
        self.vector.len()
    }

    /// Linear interpolation between two latent codes
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        assert_eq!(self.dim(), other.dim(), "Latent dimensions must match");
        let vector = self
            .vector
            .iter()
            .zip(&other.vector)
            .map(|(a, b)| a * (1.0 - t) + b * t)
            .collect();
        Self { vector }
    }

    /// Spherical linear interpolation between two latent codes
    #[must_use]
    pub fn slerp(&self, other: &Self, t: f32) -> Self {
        assert_eq!(self.dim(), other.dim(), "Latent dimensions must match");

        let norm_self: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_other: f32 = other.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Fall back to lerp if either vector has near-zero norm
        if norm_self < 1e-10 || norm_other < 1e-10 {
            return self.lerp(other, t);
        }

        let dot: f32 = self
            .vector
            .iter()
            .zip(&other.vector)
            .map(|(a, b)| a * b)
            .sum();

        let cos_omega = (dot / (norm_self * norm_other)).clamp(-1.0, 1.0);
        let omega = cos_omega.acos();

        if omega.abs() < 1e-6 {
            return self.lerp(other, t);
        }

        let sin_omega = omega.sin();
        let factor_self = ((1.0 - t) * omega).sin() / sin_omega;
        let factor_other = (t * omega).sin() / sin_omega;

        let vector = self
            .vector
            .iter()
            .zip(&other.vector)
            .map(|(a, b)| a * factor_self + b * factor_other)
            .collect();

        Self { vector }
    }

    /// Compute L2 norm
    #[must_use]
    pub fn norm(&self) -> f32 {
        self.vector.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize to unit length
    #[must_use]
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n < 1e-10 {
            return self.clone();
        }
        let vector = self.vector.iter().map(|x| x / n).collect();
        Self { vector }
    }
}

/// Configuration for the Generator network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorConfig {
    /// Dimension of the latent space
    pub latent_dim: usize,
    /// Hidden layer sizes
    pub hidden_dims: Vec<usize>,
    /// Output vocabulary size (number of AST token types)
    pub vocab_size: usize,
    /// Maximum sequence length to generate
    pub max_seq_len: usize,
    /// Dropout rate during training
    pub dropout: f32,
    /// Use batch normalization
    pub batch_norm: bool,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            latent_dim: 128,
            hidden_dims: vec![256, 512, 256],
            vocab_size: 1000,
            max_seq_len: 256,
            dropout: 0.1,
            batch_norm: true,
        }
    }
}

/// Generator network: maps latent vectors to AST token sequences
#[derive(Debug)]
pub struct Generator {
    /// Configuration
    pub config: GeneratorConfig,
    /// Weights for each layer (simplified representation)
    weights: Vec<Vec<Vec<f32>>>,
    /// Biases for each layer
    biases: Vec<Vec<f32>>,
}

impl Generator {
    /// Create a new generator with random initialization
    pub fn new(config: GeneratorConfig) -> Self {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::from_os_rng();
        let (weights, biases) = Self::init_weights(&config, &mut rng);
        Self {
            config,
            weights,
            biases,
        }
    }

    /// Create a new generator with a seed for reproducibility
    pub fn with_seed(config: GeneratorConfig, seed: u64) -> Self {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let (weights, biases) = Self::init_weights(&config, &mut rng);
        Self {
            config,
            weights,
            biases,
        }
    }

    fn init_weights<R: Rng>(
        config: &GeneratorConfig,
        rng: &mut R,
    ) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>) {
        let mut dims = vec![config.latent_dim];
        dims.extend(&config.hidden_dims);
        dims.push(config.vocab_size * config.max_seq_len);

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..dims.len() - 1 {
            let input_dim = dims[i];
            let output_dim = dims[i + 1];

            // Xavier initialization using Box-Muller transform
            let std = (2.0 / (input_dim + output_dim) as f64).sqrt();

            let w: Vec<Vec<f32>> = (0..output_dim)
                .map(|_| {
                    (0..input_dim)
                        .map(|_| {
                            let u1: f64 = rng.random::<f64>().max(1e-10);
                            let u2: f64 = rng.random::<f64>();
                            let z =
                                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                            (z * std) as f32
                        })
                        .collect()
                })
                .collect();
            let b: Vec<f32> = vec![0.0; output_dim];

            weights.push(w);
            biases.push(b);
        }

        (weights, biases)
    }

    /// Generate AST tokens from a latent code
    pub fn generate(&self, latent: &LatentCode) -> Vec<u32> {
        assert_eq!(latent.dim(), self.config.latent_dim);

        // Forward pass through network
        let mut x = latent.vector.clone();

        for (w, b) in self.weights.iter().zip(&self.biases) {
            x = Self::linear_forward(&x, w, b);
            // ReLU activation (except last layer)
            if w != self.weights.last().expect("non-empty weights") {
                x = x.iter().map(|&v| v.max(0.0)).collect();
            }
        }

        // Reshape to (max_seq_len, vocab_size) and take argmax for each position
        let vocab_size = self.config.vocab_size;
        let max_seq_len = self.config.max_seq_len;

        let mut tokens = Vec::with_capacity(max_seq_len);
        for pos in 0..max_seq_len {
            let start = pos * vocab_size;
            let end = start + vocab_size;
            if end <= x.len() {
                let logits = &x[start..end];
                let max_idx = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(i, _)| i as u32);
                tokens.push(max_idx);
            }
        }

        tokens
    }

    fn linear_forward(input: &[f32], weights: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
        let output_dim = weights.len();
        let mut output = Vec::with_capacity(output_dim);

        for (i, w_row) in weights.iter().enumerate() {
            let dot: f32 = w_row.iter().zip(input).map(|(a, b)| a * b).sum();
            output.push(dot + bias[i]);
        }

        output
    }

    /// Get number of parameters
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let weight_params: usize = self.weights.iter().map(|w| w.len() * w[0].len()).sum();
        let bias_params: usize = self.biases.iter().map(Vec::len).sum();
        weight_params + bias_params
    }
}

/// Configuration for the Discriminator network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscriminatorConfig {
    /// Input vocabulary size (number of AST token types)
    pub vocab_size: usize,
    /// Maximum sequence length to process
    pub max_seq_len: usize,
    /// Embedding dimension for tokens
    pub embed_dim: usize,
    /// Hidden layer sizes
    pub hidden_dims: Vec<usize>,
    /// Dropout rate during training
    pub dropout: f32,
    /// Use spectral normalization
    pub spectral_norm: bool,
}

impl Default for DiscriminatorConfig {
    fn default() -> Self {
        Self {
            vocab_size: 1000,
            max_seq_len: 256,
            embed_dim: 64,
            hidden_dims: vec![256, 128, 64],
            dropout: 0.2,
            spectral_norm: true,
        }
    }
}

/// Discriminator network: classifies code as real or fake
#[derive(Debug)]
pub struct Discriminator {
    /// Configuration
    pub config: DiscriminatorConfig,
    /// Token embeddings
    embeddings: Vec<Vec<f32>>,
    /// Weights for each layer
    weights: Vec<Vec<Vec<f32>>>,
    /// Biases for each layer
    biases: Vec<Vec<f32>>,
}

impl Discriminator {
    /// Create a new discriminator with random initialization
    pub fn new(config: DiscriminatorConfig) -> Self {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::from_os_rng();
        let (embeddings, weights, biases) = Self::init_weights(&config, &mut rng);
        Self {
            config,
            embeddings,
            weights,
            biases,
        }
    }

    /// Create a new discriminator with a seed for reproducibility
    pub fn with_seed(config: DiscriminatorConfig, seed: u64) -> Self {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let (embeddings, weights, biases) = Self::init_weights(&config, &mut rng);
        Self {
            config,
            embeddings,
            weights,
            biases,
        }
    }

    fn init_weights<R: Rng>(config: &DiscriminatorConfig, rng: &mut R) -> DiscriminatorWeights {
        // Helper function for Box-Muller normal sampling
        let sample_normal = |rng: &mut R, std: f64| -> f32 {
            let u1: f64 = rng.random::<f64>().max(1e-10);
            let u2: f64 = rng.random::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            (z * std) as f32
        };

        // Initialize embeddings
        let embed_std = (1.0 / config.embed_dim as f64).sqrt();
        let embeddings: Vec<Vec<f32>> = (0..config.vocab_size)
            .map(|_| {
                (0..config.embed_dim)
                    .map(|_| sample_normal(rng, embed_std))
                    .collect()
            })
            .collect();

        // Initialize dense layers
        let input_dim = config.embed_dim * config.max_seq_len;
        let mut dims = vec![input_dim];
        dims.extend(&config.hidden_dims);
        dims.push(1); // Output: single logit for real/fake

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..dims.len() - 1 {
            let in_dim = dims[i];
            let out_dim = dims[i + 1];

            let std = (2.0 / (in_dim + out_dim) as f64).sqrt();

            let w: Vec<Vec<f32>> = (0..out_dim)
                .map(|_| (0..in_dim).map(|_| sample_normal(rng, std)).collect())
                .collect();
            let b: Vec<f32> = vec![0.0; out_dim];

            weights.push(w);
            biases.push(b);
        }

        (embeddings, weights, biases)
    }

    /// Discriminate: returns probability that input is real (valid code)
    pub fn discriminate(&self, tokens: &[u32]) -> f32 {
        // Pad or truncate to max_seq_len
        let mut padded = tokens.to_vec();
        padded.resize(self.config.max_seq_len, 0);

        // Embed tokens
        let mut x = Vec::with_capacity(self.config.max_seq_len * self.config.embed_dim);
        for &token in &padded {
            let token_idx = (token as usize).min(self.config.vocab_size - 1);
            x.extend(&self.embeddings[token_idx]);
        }

        // Forward pass through dense layers
        for (i, (w, b)) in self.weights.iter().zip(&self.biases).enumerate() {
            x = Self::linear_forward(&x, w, b);
            // Leaky ReLU for all but last layer
            if i < self.weights.len() - 1 {
                x = x
                    .iter()
                    .map(|&v| if v > 0.0 { v } else { 0.01 * v })
                    .collect();
            }
        }

        // Sigmoid on output
        sigmoid(x[0])
    }

    fn linear_forward(input: &[f32], weights: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
        let output_dim = weights.len();
        let mut output = Vec::with_capacity(output_dim);

        for (i, w_row) in weights.iter().enumerate() {
            let dot: f32 = w_row.iter().zip(input).map(|(a, b)| a * b).sum();
            output.push(dot + bias[i]);
        }

        output
    }

    /// Get number of parameters
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let embed_params = self.embeddings.len() * self.config.embed_dim;
        let weight_params: usize = self.weights.iter().map(|w| w.len() * w[0].len()).sum();
        let bias_params: usize = self.biases.iter().map(Vec::len).sum();
        embed_params + weight_params + bias_params
    }
}

/// Sigmoid activation function
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Configuration for the complete Code GAN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeGanConfig {
    /// Generator configuration
    pub generator: GeneratorConfig,
    /// Discriminator configuration
    pub discriminator: DiscriminatorConfig,
    /// Learning rate for generator
    pub gen_lr: f32,
    /// Learning rate for discriminator
    pub disc_lr: f32,
    /// Number of discriminator updates per generator update
    pub n_critic: usize,
    /// Gradient penalty coefficient (for WGAN-GP)
    pub gradient_penalty: f32,
    /// Label smoothing for real samples
    pub label_smoothing: f32,
    /// Batch size for training
    pub batch_size: usize,
}

impl Default for CodeGanConfig {
    fn default() -> Self {
        Self {
            generator: GeneratorConfig::default(),
            discriminator: DiscriminatorConfig::default(),
            gen_lr: 0.0002,
            disc_lr: 0.0002,
            n_critic: 5,
            gradient_penalty: 10.0,
            label_smoothing: 0.1,
            batch_size: 32,
        }
    }
}

/// Training result from a GAN update step
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Generator loss
    pub gen_loss: f32,
    /// Discriminator loss
    pub disc_loss: f32,
    /// Discriminator accuracy on real samples
    pub disc_real_acc: f32,
    /// Discriminator accuracy on fake samples
    pub disc_fake_acc: f32,
    /// Gradient penalty value
    pub gradient_penalty: f32,
}

/// Statistics from GAN training
#[derive(Debug, Clone)]
pub struct CodeGanStats {
    /// Total training steps
    pub steps: usize,
    /// Generator losses (recent history)
    pub gen_losses: VecDeque<f32>,
    /// Discriminator losses (recent history)
    pub disc_losses: VecDeque<f32>,
    /// Mode collapse score (0 = no collapse, 1 = full collapse)
    pub mode_collapse_score: f32,
    /// Number of unique tokens generated in last batch
    pub unique_tokens: usize,
}

impl Default for CodeGanStats {
    fn default() -> Self {
        Self {
            steps: 0,
            gen_losses: VecDeque::with_capacity(100),
            disc_losses: VecDeque::with_capacity(100),
            mode_collapse_score: 0.0,
            unique_tokens: 0,
        }
    }
}

/// Complete Code GAN for generating Rust AST
pub struct CodeGan {
    /// Configuration
    pub config: CodeGanConfig,
    /// Generator network
    pub generator: Generator,
    /// Discriminator network
    pub discriminator: Discriminator,
    /// Training statistics
    pub stats: CodeGanStats,
    /// Random number generator
    rng: rand::rngs::StdRng,
}

impl CodeGan {
    /// Create a new Code GAN
    pub fn new(config: CodeGanConfig) -> Self {
        use rand::SeedableRng;
        let generator = Generator::new(config.generator.clone());
        let discriminator = Discriminator::new(config.discriminator.clone());
        Self {
            config,
            generator,
            discriminator,
            stats: CodeGanStats::default(),
            rng: rand::rngs::StdRng::from_os_rng(),
        }
    }

    /// Create a new Code GAN with a seed for reproducibility
    pub fn with_seed(config: CodeGanConfig, seed: u64) -> Self {
        use rand::SeedableRng;
        let generator = Generator::with_seed(config.generator.clone(), seed);
        let discriminator = Discriminator::with_seed(config.discriminator.clone(), seed + 1);
        Self {
            config,
            generator,
            discriminator,
            stats: CodeGanStats::default(),
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    /// Sample latent codes for generation
    pub fn sample_latent(&mut self, batch_size: usize) -> Vec<LatentCode> {
        (0..batch_size)
            .map(|_| LatentCode::sample(&mut self.rng, self.config.generator.latent_dim))
            .collect()
    }

    /// Generate code from latent codes
    pub fn generate(&self, latent_codes: &[LatentCode]) -> Vec<Vec<u32>> {
        latent_codes
            .iter()
            .map(|z| self.generator.generate(z))
            .collect()
    }

    /// Generate a single code sample
    pub fn generate_one(&mut self) -> Vec<u32> {
        let z = LatentCode::sample(&mut self.rng, self.config.generator.latent_dim);
        self.generator.generate(&z)
    }

    /// Discriminate a batch of code samples
    pub fn discriminate(&self, samples: &[Vec<u32>]) -> Vec<f32> {
        samples
            .iter()
            .map(|tokens| self.discriminator.discriminate(tokens))
            .collect()
    }

    /// Compute discriminator loss (binary cross-entropy)
    pub fn discriminator_loss(&self, real_samples: &[Vec<u32>], fake_samples: &[Vec<u32>]) -> f32 {
        let real_probs = self.discriminate(real_samples);
        let fake_probs = self.discriminate(fake_samples);

        // BCE loss: -[y*log(p) + (1-y)*log(1-p)]
        // For real: y=1, for fake: y=0
        let smoothed_real = 1.0 - self.config.label_smoothing;

        let real_loss: f32 = real_probs
            .iter()
            .map(|&p| -smoothed_real * p.max(1e-7).ln())
            .sum::<f32>()
            / real_probs.len() as f32;

        let fake_loss: f32 = fake_probs
            .iter()
            .map(|&p| -(1.0 - p).max(1e-7).ln())
            .sum::<f32>()
            / fake_probs.len() as f32;

        real_loss + fake_loss
    }

    /// Compute generator loss (try to fool discriminator)
    pub fn generator_loss(&self, fake_samples: &[Vec<u32>]) -> f32 {
        let fake_probs = self.discriminate(fake_samples);

        // Generator wants discriminator to output 1 (real) for fakes
        let loss: f32 =
            fake_probs.iter().map(|&p| -p.max(1e-7).ln()).sum::<f32>() / fake_probs.len() as f32;

        loss
    }

    /// Detect mode collapse by measuring diversity of generated samples
    pub fn detect_mode_collapse(&mut self, num_samples: usize) -> f32 {
        use std::collections::HashSet;

        let latent_codes = self.sample_latent(num_samples);
        let samples = self.generate(&latent_codes);

        // Count unique token sequences
        let unique_seqs: HashSet<Vec<u32>> = samples.into_iter().collect();
        let diversity = unique_seqs.len() as f32 / num_samples as f32;

        // Also check token diversity
        let all_tokens: HashSet<u32> = unique_seqs
            .iter()
            .flat_map(|seq| seq.iter().copied())
            .collect();

        self.stats.unique_tokens = all_tokens.len();

        // Mode collapse score: 1 - diversity
        let mode_collapse_score = 1.0 - diversity;
        self.stats.mode_collapse_score = mode_collapse_score;

        mode_collapse_score
    }

    /// Interpolate between two latent codes and generate intermediate samples
    pub fn interpolate(&self, z1: &LatentCode, z2: &LatentCode, steps: usize) -> Vec<Vec<u32>> {
        (0..=steps)
            .map(|i| {
                let t = i as f32 / steps as f32;
                let z = z1.slerp(z2, t);
                self.generator.generate(&z)
            })
            .collect()
    }

    /// Get total number of parameters
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        self.generator.num_parameters() + self.discriminator.num_parameters()
    }

    /// Record training step
    pub fn record_step(&mut self, result: &TrainingResult) {
        self.stats.steps += 1;

        if self.stats.gen_losses.len() >= 100 {
            self.stats.gen_losses.pop_front();
        }
        self.stats.gen_losses.push_back(result.gen_loss);

        if self.stats.disc_losses.len() >= 100 {
            self.stats.disc_losses.pop_front();
        }
        self.stats.disc_losses.push_back(result.disc_loss);
    }

    /// Get average generator loss over recent history
    #[must_use]
    pub fn avg_gen_loss(&self) -> f32 {
        if self.stats.gen_losses.is_empty() {
            return 0.0;
        }
        self.stats.gen_losses.iter().sum::<f32>() / self.stats.gen_losses.len() as f32
    }

    /// Get average discriminator loss over recent history
    #[must_use]
    pub fn avg_disc_loss(&self) -> f32 {
        if self.stats.disc_losses.is_empty() {
            return 0.0;
        }
        self.stats.disc_losses.iter().sum::<f32>() / self.stats.disc_losses.len() as f32
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Create a small test config to avoid slow network initialization
    fn small_test_config() -> CodeGanConfig {
        CodeGanConfig {
            generator: GeneratorConfig {
                latent_dim: 16,
                hidden_dims: vec![32],
                vocab_size: 50,
                max_seq_len: 8,
                dropout: 0.0,
                batch_norm: false,
            },
            discriminator: DiscriminatorConfig {
                vocab_size: 50,
                max_seq_len: 8,
                embed_dim: 8,
                hidden_dims: vec![16],
                dropout: 0.0,
                spectral_norm: false,
            },
            ..Default::default()
        }
    }

    // ========================================
    // ENT-096: GAN core types tests
    // ========================================

    #[test]
    fn test_latent_code_creation() {
        let code = LatentCode::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(code.dim(), 3);
        assert_eq!(code.vector, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_latent_code_sample() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let code = LatentCode::sample(&mut rng, 128);
        assert_eq!(code.dim(), 128);
    }

    #[test]
    fn test_latent_code_lerp() {
        let z1 = LatentCode::new(vec![0.0, 0.0]);
        let z2 = LatentCode::new(vec![1.0, 1.0]);

        let mid = z1.lerp(&z2, 0.5);
        assert!((mid.vector[0] - 0.5).abs() < 1e-6);
        assert!((mid.vector[1] - 0.5).abs() < 1e-6);

        let start = z1.lerp(&z2, 0.0);
        assert!((start.vector[0] - 0.0).abs() < 1e-6);

        let end = z1.lerp(&z2, 1.0);
        assert!((end.vector[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_latent_code_slerp() {
        let z1 = LatentCode::new(vec![1.0, 0.0]);
        let z2 = LatentCode::new(vec![0.0, 1.0]);

        let mid = z1.slerp(&z2, 0.5);
        // At midpoint, should have roughly equal components
        assert!((mid.vector[0] - mid.vector[1]).abs() < 0.1);
    }

    #[test]
    fn test_latent_code_norm() {
        let code = LatentCode::new(vec![3.0, 4.0]);
        assert!((code.norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_latent_code_normalize() {
        let code = LatentCode::new(vec![3.0, 4.0]);
        let normalized = code.normalize();
        assert!((normalized.norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_generator_config_default() {
        let config = GeneratorConfig::default();
        assert_eq!(config.latent_dim, 128);
        assert_eq!(config.vocab_size, 1000);
        assert_eq!(config.max_seq_len, 256);
    }

    #[test]
    fn test_discriminator_config_default() {
        let config = DiscriminatorConfig::default();
        assert_eq!(config.vocab_size, 1000);
        assert_eq!(config.max_seq_len, 256);
    }

    #[test]
    fn test_code_gan_config_default() {
        let config = CodeGanConfig::default();
        assert_eq!(config.n_critic, 5);
        assert!(config.gen_lr > 0.0);
        assert!(config.disc_lr > 0.0);
    }

    // ========================================
    // ENT-097: Generator tests
    // ========================================

    #[test]
    fn test_generator_creation() {
        let config = GeneratorConfig {
            latent_dim: 32,
            hidden_dims: vec![64, 64],
            vocab_size: 100,
            max_seq_len: 10,
            dropout: 0.1,
            batch_norm: true,
        };
        let gen = Generator::with_seed(config, 42);
        assert!(gen.num_parameters() > 0);
    }

    #[test]
    fn test_generator_generate() {
        let config = GeneratorConfig {
            latent_dim: 16,
            hidden_dims: vec![32],
            vocab_size: 50,
            max_seq_len: 8,
            dropout: 0.0,
            batch_norm: false,
        };
        let gen = Generator::with_seed(config.clone(), 42);

        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let z = LatentCode::sample(&mut rng, config.latent_dim);

        let tokens = gen.generate(&z);
        assert_eq!(tokens.len(), config.max_seq_len);
        assert!(tokens.iter().all(|&t| t < config.vocab_size as u32));
    }

    #[test]
    fn test_generator_deterministic() {
        let config = GeneratorConfig {
            latent_dim: 16,
            hidden_dims: vec![32],
            vocab_size: 50,
            max_seq_len: 8,
            dropout: 0.0,
            batch_norm: false,
        };

        let gen = Generator::with_seed(config.clone(), 42);
        let z = LatentCode::new(vec![0.5; config.latent_dim]);

        let tokens1 = gen.generate(&z);
        let tokens2 = gen.generate(&z);

        assert_eq!(tokens1, tokens2);
    }

    // ========================================
    // ENT-098: Discriminator tests
    // ========================================

    #[test]
    fn test_discriminator_creation() {
        let config = DiscriminatorConfig {
            vocab_size: 100,
            max_seq_len: 10,
            embed_dim: 16,
            hidden_dims: vec![32, 16],
            dropout: 0.1,
            spectral_norm: true,
        };
        let disc = Discriminator::with_seed(config, 42);
        assert!(disc.num_parameters() > 0);
    }

    #[test]
    fn test_discriminator_output_range() {
        let config = DiscriminatorConfig {
            vocab_size: 50,
            max_seq_len: 8,
            embed_dim: 8,
            hidden_dims: vec![16],
            dropout: 0.0,
            spectral_norm: false,
        };
        let disc = Discriminator::with_seed(config, 42);

        let tokens = vec![1, 2, 3, 4, 5];
        let prob = disc.discriminate(&tokens);

        // Output should be in [0, 1] due to sigmoid
        assert!(prob >= 0.0 && prob <= 1.0);
    }

    #[test]
    fn test_discriminator_deterministic() {
        let config = DiscriminatorConfig {
            vocab_size: 50,
            max_seq_len: 8,
            embed_dim: 8,
            hidden_dims: vec![16],
            dropout: 0.0,
            spectral_norm: false,
        };
        let disc = Discriminator::with_seed(config, 42);

        let tokens = vec![1, 2, 3, 4, 5];
        let prob1 = disc.discriminate(&tokens);
        let prob2 = disc.discriminate(&tokens);

        assert!((prob1 - prob2).abs() < 1e-6);
    }

    // ========================================
    // ENT-099: Latent space interpolation tests
    // ========================================

    #[test]
    fn test_interpolation_endpoints() {
        let config = CodeGanConfig {
            generator: GeneratorConfig {
                latent_dim: 16,
                hidden_dims: vec![32],
                vocab_size: 50,
                max_seq_len: 8,
                ..Default::default()
            },
            ..Default::default()
        };
        let gan = CodeGan::with_seed(config, 42);

        let z1 = LatentCode::new(vec![0.0; 16]);
        let z2 = LatentCode::new(vec![1.0; 16]);

        let samples = gan.interpolate(&z1, &z2, 4);
        assert_eq!(samples.len(), 5); // 0, 0.25, 0.5, 0.75, 1.0

        // First should match z1 generation
        let direct_z1 = gan.generator.generate(&z1);
        assert_eq!(samples[0], direct_z1);

        // Last should match z2 generation
        let direct_z2 = gan.generator.generate(&z2);
        assert_eq!(samples[4], direct_z2);
    }

    #[test]
    fn test_slerp_maintains_norm() {
        let z1 = LatentCode::new(vec![1.0, 0.0, 0.0]).normalize();
        let z2 = LatentCode::new(vec![0.0, 1.0, 0.0]).normalize();

        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let z = z1.slerp(&z2, t);
            // SLERP should maintain approximate unit norm
            assert!((z.norm() - 1.0).abs() < 0.1);
        }
    }

    // ========================================
    // ENT-100: GAN training loop tests
    // ========================================

    #[test]
    fn test_code_gan_creation() {
        let config = small_test_config();
        let gan = CodeGan::new(config);
        assert!(gan.num_parameters() > 0);
        assert_eq!(gan.stats.steps, 0);
    }

    #[test]
    fn test_code_gan_sample_latent() {
        let config = small_test_config();
        let mut gan = CodeGan::with_seed(config.clone(), 42);

        let latents = gan.sample_latent(10);
        assert_eq!(latents.len(), 10);
        assert!(latents
            .iter()
            .all(|z| z.dim() == config.generator.latent_dim));
    }

    #[test]
    fn test_code_gan_generate() {
        let config = CodeGanConfig {
            generator: GeneratorConfig {
                latent_dim: 16,
                hidden_dims: vec![32],
                vocab_size: 50,
                max_seq_len: 8,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut gan = CodeGan::with_seed(config, 42);

        let latents = gan.sample_latent(5);
        let samples = gan.generate(&latents);

        assert_eq!(samples.len(), 5);
        assert!(samples.iter().all(|s| s.len() == 8));
    }

    #[test]
    fn test_code_gan_discriminator_loss() {
        let config = CodeGanConfig {
            generator: GeneratorConfig {
                latent_dim: 16,
                hidden_dims: vec![32],
                vocab_size: 50,
                max_seq_len: 8,
                ..Default::default()
            },
            discriminator: DiscriminatorConfig {
                vocab_size: 50,
                max_seq_len: 8,
                embed_dim: 8,
                hidden_dims: vec![16],
                ..Default::default()
            },
            ..Default::default()
        };
        let mut gan = CodeGan::with_seed(config, 42);

        let real_samples: Vec<Vec<u32>> = (0..5)
            .map(|i| (0..8).map(|j| ((i + j) % 50) as u32).collect())
            .collect();

        let latents = gan.sample_latent(5);
        let fake_samples = gan.generate(&latents);

        let loss = gan.discriminator_loss(&real_samples, &fake_samples);
        assert!(loss >= 0.0);
        assert!(!loss.is_nan());
    }

    #[test]
    fn test_code_gan_generator_loss() {
        let config = CodeGanConfig {
            generator: GeneratorConfig {
                latent_dim: 16,
                hidden_dims: vec![32],
                vocab_size: 50,
                max_seq_len: 8,
                ..Default::default()
            },
            discriminator: DiscriminatorConfig {
                vocab_size: 50,
                max_seq_len: 8,
                embed_dim: 8,
                hidden_dims: vec![16],
                ..Default::default()
            },
            ..Default::default()
        };
        let mut gan = CodeGan::with_seed(config, 42);

        let latents = gan.sample_latent(5);
        let fake_samples = gan.generate(&latents);

        let loss = gan.generator_loss(&fake_samples);
        assert!(loss >= 0.0);
        assert!(!loss.is_nan());
    }

    #[test]
    fn test_record_step() {
        let config = small_test_config();
        let mut gan = CodeGan::new(config);

        let result = TrainingResult {
            gen_loss: 0.5,
            disc_loss: 0.3,
            disc_real_acc: 0.8,
            disc_fake_acc: 0.7,
            gradient_penalty: 0.1,
        };

        gan.record_step(&result);
        assert_eq!(gan.stats.steps, 1);
        assert_eq!(gan.stats.gen_losses.len(), 1);
        assert_eq!(gan.stats.disc_losses.len(), 1);
    }

    // ========================================
    // ENT-101: Property tests
    // ========================================

    proptest! {
        #[test]
        fn test_latent_lerp_bounds(t in 0.0f32..=1.0) {
            let z1 = LatentCode::new(vec![0.0, 0.0, 0.0]);
            let z2 = LatentCode::new(vec![1.0, 1.0, 1.0]);

            let result = z1.lerp(&z2, t);

            for v in &result.vector {
                prop_assert!(*v >= 0.0 && *v <= 1.0);
            }
        }

        #[test]
        fn test_latent_norm_non_negative(values in prop::collection::vec(-10.0f32..10.0, 1..100)) {
            let code = LatentCode::new(values);
            prop_assert!(code.norm() >= 0.0);
        }

        #[test]
        fn test_normalize_unit_length(values in prop::collection::vec(-10.0f32..10.0, 1..100)) {
            let code = LatentCode::new(values);
            if code.norm() > 1e-6 {
                let normalized = code.normalize();
                prop_assert!((normalized.norm() - 1.0).abs() < 1e-5);
            }
        }

        #[test]
        fn test_discriminator_output_bounds(tokens in prop::collection::vec(0u32..50, 1..10)) {
            let config = DiscriminatorConfig {
                vocab_size: 50,
                max_seq_len: 10,
                embed_dim: 8,
                hidden_dims: vec![16],
                dropout: 0.0,
                spectral_norm: false,
            };
            let disc = Discriminator::with_seed(config, 42);

            let prob = disc.discriminate(&tokens);
            prop_assert!(prob >= 0.0 && prob <= 1.0);
        }

        #[test]
        fn test_generator_output_valid_tokens(seed in 0u64..10000) {
            let config = GeneratorConfig {
                latent_dim: 16,
                hidden_dims: vec![32],
                vocab_size: 50,
                max_seq_len: 8,
                dropout: 0.0,
                batch_norm: false,
            };
            let gen = Generator::with_seed(config.clone(), seed);

            use rand::SeedableRng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let z = LatentCode::sample(&mut rng, config.latent_dim);

            let tokens = gen.generate(&z);
            prop_assert!(tokens.iter().all(|&t| t < 50));
        }

        #[test]
        fn test_loss_non_negative(
            real_vals in prop::collection::vec(prop::collection::vec(0u32..50, 8..9), 1..5),
            fake_vals in prop::collection::vec(prop::collection::vec(0u32..50, 8..9), 1..5),
        ) {
            let config = CodeGanConfig {
                generator: GeneratorConfig {
                    latent_dim: 16,
                    hidden_dims: vec![32],
                    vocab_size: 50,
                    max_seq_len: 8,
                    ..Default::default()
                },
                discriminator: DiscriminatorConfig {
                    vocab_size: 50,
                    max_seq_len: 8,
                    embed_dim: 8,
                    hidden_dims: vec![16],
                    ..Default::default()
                },
                ..Default::default()
            };
            let gan = CodeGan::with_seed(config, 42);

            let disc_loss = gan.discriminator_loss(&real_vals, &fake_vals);
            let gen_loss = gan.generator_loss(&fake_vals);

            prop_assert!(disc_loss >= 0.0);
            prop_assert!(gen_loss >= 0.0);
        }

        #[test]
        fn test_mode_collapse_detection(num_samples in 10usize..50) {
            let config = small_test_config();
            let mut gan = CodeGan::with_seed(config, 42);

            let score = gan.detect_mode_collapse(num_samples);
            prop_assert!(score >= 0.0 && score <= 1.0);
        }

        #[test]
        fn test_interpolation_length(steps in 1usize..20) {
            let config = small_test_config();
            let gan = CodeGan::with_seed(config, 42);

            let z1 = LatentCode::new(vec![0.0; 16]);
            let z2 = LatentCode::new(vec![1.0; 16]);

            let samples = gan.interpolate(&z1, &z2, steps);
            prop_assert_eq!(samples.len(), steps + 1);
        }
    }

    // ========================================
    // Additional unit tests
    // ========================================

    #[test]
    fn test_sigmoid_function() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_code_gan_stats_default() {
        let stats = CodeGanStats::default();
        assert_eq!(stats.steps, 0);
        assert!(stats.gen_losses.is_empty());
        assert!(stats.disc_losses.is_empty());
        assert_eq!(stats.mode_collapse_score, 0.0);
    }

    #[test]
    fn test_avg_loss_empty() {
        let config = small_test_config();
        let gan = CodeGan::new(config);
        assert_eq!(gan.avg_gen_loss(), 0.0);
        assert_eq!(gan.avg_disc_loss(), 0.0);
    }

    #[test]
    fn test_avg_loss_with_history() {
        let config = small_test_config();
        let mut gan = CodeGan::new(config);

        for i in 0..10 {
            let result = TrainingResult {
                gen_loss: i as f32,
                disc_loss: i as f32 * 2.0,
                disc_real_acc: 0.8,
                disc_fake_acc: 0.7,
                gradient_penalty: 0.1,
            };
            gan.record_step(&result);
        }

        // Average of 0,1,2,...,9 = 4.5
        assert!((gan.avg_gen_loss() - 4.5).abs() < 1e-6);
        assert!((gan.avg_disc_loss() - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_generate_one() {
        let config = CodeGanConfig {
            generator: GeneratorConfig {
                latent_dim: 16,
                hidden_dims: vec![32],
                vocab_size: 50,
                max_seq_len: 8,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut gan = CodeGan::with_seed(config, 42);

        let tokens = gan.generate_one();
        assert_eq!(tokens.len(), 8);
    }

    #[test]
    fn test_history_size_limit() {
        let config = small_test_config();
        let mut gan = CodeGan::new(config);

        // Add more than 100 steps
        for i in 0..150 {
            let result = TrainingResult {
                gen_loss: i as f32,
                disc_loss: i as f32,
                disc_real_acc: 0.8,
                disc_fake_acc: 0.7,
                gradient_penalty: 0.1,
            };
            gan.record_step(&result);
        }

        // Should be capped at 100
        assert_eq!(gan.stats.gen_losses.len(), 100);
        assert_eq!(gan.stats.disc_losses.len(), 100);
    }
}
