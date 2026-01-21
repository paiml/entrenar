//! Code GAN main struct and training logic.

use std::collections::VecDeque;

use super::config::CodeGanConfig;
use super::discriminator::Discriminator;
use super::generator::Generator;
use super::latent::LatentCode;

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

/// Create a small test config to avoid slow network initialization
#[cfg(test)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

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

    proptest! {
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
            prop_assert!((0.0..=1.0).contains(&score));
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
}
