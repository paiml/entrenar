//! Core Trainer struct and basic methods

use crate::optim::Optimizer;
use crate::train::callback::{CallbackContext, CallbackManager, TrainerCallback};
use crate::train::{LossFn, MetricsTracker, TrainConfig};
use crate::Tensor;
use std::time::Instant;

/// High-level trainer that orchestrates the training loop
///
/// # Example
///
/// ```no_run
/// use entrenar::train::{Trainer, TrainConfig, Batch, MSELoss, EarlyStopping};
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
/// trainer.add_callback(EarlyStopping::new(5, 0.001));
///
/// // Training with callbacks
/// // let result = trainer.train(10, || batches.clone(), |x| x.clone());
/// ```
pub struct Trainer {
    /// Model parameters
    pub(crate) params: Vec<Tensor>,

    /// Optimizer
    pub(crate) optimizer: Box<dyn Optimizer>,

    /// Loss function
    pub(crate) loss_fn: Option<Box<dyn LossFn>>,

    /// Training configuration
    pub(crate) config: TrainConfig,

    /// Metrics tracker
    pub metrics: MetricsTracker,

    /// Callback manager
    pub(crate) callbacks: CallbackManager,

    /// Best loss achieved during training
    pub(crate) best_loss: Option<f32>,

    /// Training start time
    pub(crate) start_time: Option<Instant>,
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
            callbacks: CallbackManager::new(),
            best_loss: None,
            start_time: None,
        }
    }

    /// Set the loss function
    pub fn set_loss(&mut self, loss_fn: Box<dyn LossFn>) {
        self.loss_fn = Some(loss_fn);
    }

    /// Add a callback to the trainer
    pub fn add_callback<C: TrainerCallback + 'static>(&mut self, callback: C) {
        self.callbacks.add(callback);
    }

    /// Get current learning rate
    pub fn lr(&self) -> f32 {
        self.optimizer.lr()
    }

    /// Set learning rate
    pub fn set_lr(&mut self, lr: f32) {
        self.optimizer.set_lr(lr);
    }

    /// Get reference to model parameters
    pub fn params(&self) -> &[Tensor] {
        &self.params
    }

    /// Get mutable reference to model parameters
    pub fn params_mut(&mut self) -> &mut [Tensor] {
        &mut self.params
    }

    /// Get reference to callback manager
    pub fn callbacks(&self) -> &CallbackManager {
        &self.callbacks
    }

    /// Get mutable reference to callback manager
    pub fn callbacks_mut(&mut self) -> &mut CallbackManager {
        &mut self.callbacks
    }

    /// Build callback context from current state
    pub(crate) fn build_context(
        &self,
        epoch: usize,
        max_epochs: usize,
        step: usize,
        steps_per_epoch: usize,
        loss: f32,
        val_loss: Option<f32>,
    ) -> CallbackContext {
        CallbackContext {
            epoch,
            max_epochs,
            step,
            steps_per_epoch,
            global_step: self.metrics.steps,
            loss,
            lr: self.lr(),
            best_loss: self.best_loss,
            val_loss,
            elapsed_secs: self.start_time.map_or(0.0, |t| t.elapsed().as_secs_f64()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::Adam;

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
    fn test_set_lr() {
        let params = vec![Tensor::zeros(10, true)];
        let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        assert_eq!(trainer.lr(), 0.001);

        trainer.set_lr(0.01);
        assert_eq!(trainer.lr(), 0.01);
    }

    #[test]
    fn test_params_mut() {
        let params = vec![Tensor::from_vec(vec![1.0, 2.0], true)];
        let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        let params = trainer.params_mut();
        assert_eq!(params.len(), 1);
        // Params should be mutable
        params[0] = Tensor::from_vec(vec![3.0, 4.0], true);
        assert_eq!(trainer.params()[0].data()[0], 3.0);
    }

    #[test]
    fn test_add_callback() {
        use crate::train::ProgressCallback;

        let params = vec![Tensor::zeros(10, true)];
        let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.add_callback(ProgressCallback::new(5));

        // Verify callback was added
        assert!(!trainer.callbacks().is_empty());
    }

    #[test]
    fn test_callbacks_mut() {
        use crate::train::ProgressCallback;

        let params = vec![Tensor::zeros(10, true)];
        let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        assert!(trainer.callbacks().is_empty());

        // Add callback via mutable ref
        trainer.callbacks_mut();
        trainer.add_callback(ProgressCallback::new(10));
        assert!(!trainer.callbacks().is_empty());
    }
}
