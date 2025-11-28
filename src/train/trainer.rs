//! Trainer abstraction for training loops

use super::{Batch, LossFn, MetricsTracker, TrainConfig};
use crate::optim::{clip_grad_norm, Optimizer};
use crate::Tensor;

/// High-level trainer that orchestrates the training loop
///
/// # Example
///
/// ```no_run
/// use entrenar::train::{Trainer, TrainConfig, Batch, MSELoss};
/// use entrenar::optim::Adam;
/// use entrenar::Tensor;
///
/// // Setup
/// let params = vec![Tensor::zeros(10, true)];
/// let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
/// let config = TrainConfig::default();
///
/// let mut trainer = Trainer::new(params, Box::new(optimizer), config);
/// trainer.set_loss(Box::new(MSELoss));
///
/// // Training
/// // let batch = Batch::new(inputs, targets);
/// // let loss = trainer.train_step(&batch);
/// ```
pub struct Trainer {
    /// Model parameters
    params: Vec<Tensor>,

    /// Optimizer
    optimizer: Box<dyn Optimizer>,

    /// Loss function
    loss_fn: Option<Box<dyn LossFn>>,

    /// Training configuration
    config: TrainConfig,

    /// Metrics tracker
    pub metrics: MetricsTracker,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(params: Vec<Tensor>, optimizer: Box<dyn Optimizer>, config: TrainConfig) -> Self {
        Self {
            params,
            optimizer,
            loss_fn: None,
            config,
            metrics: MetricsTracker::new(),
        }
    }

    /// Set the loss function
    pub fn set_loss(&mut self, loss_fn: Box<dyn LossFn>) {
        self.loss_fn = Some(loss_fn);
    }

    /// Get current learning rate
    pub fn lr(&self) -> f32 {
        self.optimizer.lr()
    }

    /// Set learning rate
    pub fn set_lr(&mut self, lr: f32) {
        self.optimizer.set_lr(lr);
    }

    /// Perform a single training step
    ///
    /// # Arguments
    ///
    /// * `batch` - Training batch with inputs and targets
    /// * `forward_fn` - Closure that computes predictions from inputs
    ///
    /// # Returns
    ///
    /// Scalar loss value for this batch
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use entrenar::train::{Trainer, Batch};
    /// # use entrenar::Tensor;
    /// # let mut trainer: Trainer = todo!();
    /// # let batch: Batch = todo!();
    /// let loss = trainer.train_step(&batch, |inputs| {
    ///     // Forward pass: compute predictions
    ///     inputs.clone() // Simplified example
    /// });
    /// ```
    pub fn train_step<F>(&mut self, batch: &Batch, forward_fn: F) -> f32
    where
        F: FnOnce(&Tensor) -> Tensor,
    {
        assert!(
            self.loss_fn.is_some(),
            "Loss function must be set before training"
        );

        // Zero gradients
        self.optimizer.zero_grad(&mut self.params);

        // Forward pass
        let predictions = forward_fn(&batch.inputs);

        // Compute loss
        let loss = self
            .loss_fn
            .as_ref()
            .unwrap()
            .forward(&predictions, &batch.targets);

        let loss_val = loss.data()[0];

        // Backward pass
        if let Some(backward_op) = loss.backward_op() {
            backward_op.backward();
        }

        // Gradient clipping
        if let Some(max_norm) = self.config.max_grad_norm {
            clip_grad_norm(&mut self.params, max_norm);
        }

        // Optimizer step
        self.optimizer.step(&mut self.params);

        // Update metrics
        self.metrics.increment_step();

        loss_val
    }

    /// Train for one epoch
    ///
    /// # Arguments
    ///
    /// * `batches` - Iterator over training batches
    /// * `forward_fn` - Closure that computes predictions from inputs
    ///
    /// # Returns
    ///
    /// Average loss over the epoch
    pub fn train_epoch<F, I>(&mut self, batches: I, forward_fn: F) -> f32
    where
        F: Fn(&Tensor) -> Tensor,
        I: IntoIterator<Item = Batch>,
    {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for (i, batch) in batches.into_iter().enumerate() {
            let loss = self.train_step(&batch, &forward_fn);
            total_loss += loss;
            num_batches += 1;

            // Log progress
            if (i + 1) % self.config.log_interval == 0 {
                let avg_loss = total_loss / num_batches as f32;
                println!(
                    "Epoch {}, Step {}: loss={:.4}, lr={:.6}",
                    self.metrics.epoch,
                    i + 1,
                    avg_loss,
                    self.lr()
                );
            }
        }

        let avg_loss = if num_batches > 0 {
            total_loss / num_batches as f32
        } else {
            0.0
        };

        // Record epoch metrics
        self.metrics.record_epoch(avg_loss, self.lr());

        avg_loss
    }

    /// Get reference to model parameters
    pub fn params(&self) -> &[Tensor] {
        &self.params
    }

    /// Get mutable reference to model parameters
    pub fn params_mut(&mut self) -> &mut [Tensor] {
        &mut self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::Adam;
    use crate::train::MSELoss;

    #[test]
    fn test_trainer_creation() {
        let params = vec![Tensor::zeros(10, true)];
        let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let trainer = Trainer::new(params, Box::new(optimizer), config);

        assert_eq!(trainer.params().len(), 1);
        assert_eq!(trainer.lr(), 0.001);
    }

    #[test]
    fn test_train_step() {
        let params = vec![Tensor::from_vec(vec![1.0, 2.0, 3.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));

        // Create a simple batch
        let inputs = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        let targets = Tensor::from_vec(vec![2.0, 3.0, 4.0], false);
        let batch = Batch::new(inputs, targets);

        // Train step (identity function)
        let loss = trainer.train_step(&batch, |x| x.clone());

        // Loss should be positive (predictions != targets)
        assert!(loss > 0.0);
        assert!(loss.is_finite());
        assert_eq!(trainer.metrics.steps, 1);
    }

    #[test]
    fn test_train_epoch() {
        let params = vec![Tensor::from_vec(vec![1.0, 2.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100); // Disable logging

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));

        // Create multiple batches
        let batches = vec![
            Batch::new(
                Tensor::from_vec(vec![1.0, 2.0], false),
                Tensor::from_vec(vec![2.0, 3.0], false),
            ),
            Batch::new(
                Tensor::from_vec(vec![2.0, 3.0], false),
                Tensor::from_vec(vec![3.0, 4.0], false),
            ),
        ];

        let avg_loss = trainer.train_epoch(batches, |x| x.clone());

        assert!(avg_loss > 0.0);
        assert_eq!(trainer.metrics.epoch, 1);
        assert_eq!(trainer.metrics.steps, 2);
    }

    #[test]
    #[should_panic(expected = "Loss function must be set")]
    fn test_train_step_without_loss() {
        let params = vec![Tensor::zeros(10, true)];
        let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);

        let batch = Batch::new(Tensor::zeros(10, false), Tensor::zeros(10, false));

        trainer.train_step(&batch, |x| x.clone());
    }
}
