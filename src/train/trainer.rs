//! Trainer abstraction for training loops

use super::callback::{CallbackAction, CallbackContext, CallbackManager, TrainerCallback};
use super::{Batch, LossFn, MetricsTracker, TrainConfig};
use crate::optim::{clip_grad_norm, Optimizer};
use crate::Tensor;
use std::time::Instant;

/// Result of a training run
#[derive(Debug, Clone)]
pub struct TrainResult {
    /// Final epoch reached
    pub final_epoch: usize,
    /// Final training loss
    pub final_loss: f32,
    /// Best loss achieved
    pub best_loss: f32,
    /// Whether training was stopped early
    pub stopped_early: bool,
    /// Total training time in seconds
    pub elapsed_secs: f64,
}

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
    params: Vec<Tensor>,

    /// Optimizer
    optimizer: Box<dyn Optimizer>,

    /// Loss function
    loss_fn: Option<Box<dyn LossFn>>,

    /// Training configuration
    config: TrainConfig,

    /// Metrics tracker
    pub metrics: MetricsTracker,

    /// Callback manager
    callbacks: CallbackManager,

    /// Best loss achieved during training
    best_loss: Option<f32>,

    /// Training start time
    start_time: Option<Instant>,
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

    /// Build callback context from current state
    fn build_context(
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
            elapsed_secs: self
                .start_time
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(0.0),
        }
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

    /// Perform forward and backward pass without optimizer step (for gradient accumulation)
    ///
    /// This is used internally for gradient accumulation. Gradients accumulate
    /// across calls until zero_grad is called.
    fn accumulate_gradients<F>(&mut self, batch: &Batch, forward_fn: F) -> f32
    where
        F: FnOnce(&Tensor) -> Tensor,
    {
        assert!(
            self.loss_fn.is_some(),
            "Loss function must be set before training"
        );

        // Forward pass
        let predictions = forward_fn(&batch.inputs);

        // Compute loss
        let loss = self
            .loss_fn
            .as_ref()
            .unwrap()
            .forward(&predictions, &batch.targets);

        let loss_val = loss.data()[0];

        // Backward pass (gradients accumulate)
        if let Some(backward_op) = loss.backward_op() {
            backward_op.backward();
        }

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

    /// Train for multiple epochs with full callback support
    ///
    /// # Arguments
    ///
    /// * `max_epochs` - Maximum number of epochs to train
    /// * `batch_fn` - Function that returns batches for each epoch
    /// * `forward_fn` - Closure that computes predictions from inputs
    ///
    /// # Returns
    ///
    /// TrainResult with final metrics
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use entrenar::train::{Trainer, Batch, EarlyStopping};
    /// # use entrenar::Tensor;
    /// # let mut trainer: Trainer = todo!();
    /// # let batches: Vec<Batch> = vec![];
    /// trainer.add_callback(EarlyStopping::new(5, 0.001));
    ///
    /// let result = trainer.train(100, || batches.clone(), |x| x.clone());
    /// println!("Trained {} epochs, final loss: {:.4}", result.final_epoch, result.final_loss);
    /// ```
    pub fn train<F, B, I>(&mut self, max_epochs: usize, batch_fn: B, forward_fn: F) -> TrainResult
    where
        F: Fn(&Tensor) -> Tensor,
        B: Fn() -> I,
        I: IntoIterator<Item = Batch>,
    {
        self.start_time = Some(Instant::now());
        self.best_loss = None;
        let mut stopped_early = false;
        let mut final_loss = 0.0;

        // Fire train_begin
        let ctx = self.build_context(0, max_epochs, 0, 0, 0.0, None);
        if self.callbacks.on_train_begin(&ctx) == CallbackAction::Stop {
            return TrainResult {
                final_epoch: 0,
                final_loss: 0.0,
                best_loss: 0.0,
                stopped_early: true,
                elapsed_secs: self.start_time.unwrap().elapsed().as_secs_f64(),
            };
        }

        for epoch in 0..max_epochs {
            // Fire epoch_begin
            let ctx = self.build_context(epoch, max_epochs, 0, 0, final_loss, None);
            match self.callbacks.on_epoch_begin(&ctx) {
                CallbackAction::Stop => {
                    stopped_early = true;
                    break;
                }
                CallbackAction::SkipEpoch => continue,
                CallbackAction::Continue => {}
            }

            // Collect batches and count them
            let batches: Vec<Batch> = batch_fn().into_iter().collect();
            let steps_per_epoch = batches.len();

            // Train epoch with step callbacks and gradient accumulation
            let mut total_loss = 0.0;
            let mut num_batches = 0;
            let accum_steps = self.config.gradient_accumulation_steps.max(1);

            for (step, batch) in batches.into_iter().enumerate() {
                // Fire step_begin
                let ctx =
                    self.build_context(epoch, max_epochs, step, steps_per_epoch, final_loss, None);
                if self.callbacks.on_step_begin(&ctx) == CallbackAction::Stop {
                    stopped_early = true;
                    break;
                }

                // Zero gradients at start of accumulation window
                if step % accum_steps == 0 {
                    self.optimizer.zero_grad(&mut self.params);
                }

                // Accumulate gradients
                let loss = self.accumulate_gradients(&batch, &forward_fn);
                total_loss += loss;
                num_batches += 1;

                // Optimizer step at end of accumulation window (or last batch)
                let is_accum_boundary = (step + 1) % accum_steps == 0;
                let is_last_batch = step + 1 == steps_per_epoch;
                if is_accum_boundary || is_last_batch {
                    // Gradient clipping
                    if let Some(max_norm) = self.config.max_grad_norm {
                        clip_grad_norm(&mut self.params, max_norm);
                    }
                    // Optimizer step
                    self.optimizer.step(&mut self.params);
                }

                // Update metrics
                self.metrics.increment_step();

                // Fire step_end
                let ctx = self.build_context(epoch, max_epochs, step, steps_per_epoch, loss, None);
                if self.callbacks.on_step_end(&ctx) == CallbackAction::Stop {
                    stopped_early = true;
                    break;
                }
            }

            if stopped_early {
                break;
            }

            // Calculate epoch loss
            let avg_loss = if num_batches > 0 {
                total_loss / num_batches as f32
            } else {
                0.0
            };
            final_loss = avg_loss;

            // Update best loss
            if self.best_loss.is_none() || avg_loss < self.best_loss.unwrap() {
                self.best_loss = Some(avg_loss);
            }

            // Record epoch metrics
            self.metrics.record_epoch(avg_loss, self.lr());

            // Fire epoch_end
            let ctx = self.build_context(
                epoch,
                max_epochs,
                steps_per_epoch,
                steps_per_epoch,
                avg_loss,
                None,
            );
            if self.callbacks.on_epoch_end(&ctx) == CallbackAction::Stop {
                stopped_early = true;
                break;
            }
        }

        // Fire train_end
        let ctx = self.build_context(self.metrics.epoch, max_epochs, 0, 0, final_loss, None);
        self.callbacks.on_train_end(&ctx);

        TrainResult {
            final_epoch: self.metrics.epoch,
            final_loss,
            best_loss: self.best_loss.unwrap_or(final_loss),
            stopped_early,
            elapsed_secs: self.start_time.unwrap().elapsed().as_secs_f64(),
        }
    }

    /// Get reference to callback manager
    pub fn callbacks(&self) -> &CallbackManager {
        &self.callbacks
    }

    /// Get mutable reference to callback manager
    pub fn callbacks_mut(&mut self) -> &mut CallbackManager {
        &mut self.callbacks
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

    #[test]
    fn test_train_with_callbacks() {
        use crate::train::EarlyStopping;

        let params = vec![Tensor::from_vec(vec![1.0, 2.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));
        trainer.add_callback(EarlyStopping::new(2, 0.0001));

        // Batches that produce constant loss (will trigger early stopping)
        let batches = vec![
            Batch::new(
                Tensor::from_vec(vec![1.0, 2.0], false),
                Tensor::from_vec(vec![2.0, 3.0], false),
            ),
            Batch::new(
                Tensor::from_vec(vec![1.0, 2.0], false),
                Tensor::from_vec(vec![2.0, 3.0], false),
            ),
        ];

        let result = trainer.train(10, || batches.clone(), |x| x.clone());

        // Should stop early due to no improvement
        assert!(result.stopped_early);
        assert!(result.final_epoch < 10);
        assert!(result.elapsed_secs > 0.0);
        assert!(result.best_loss > 0.0);
    }

    #[test]
    fn test_train_runs_all_epochs() {
        let params = vec![Tensor::from_vec(vec![1.0, 2.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));
        // No callbacks - should run all epochs

        let batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0, 2.0], false),
            Tensor::from_vec(vec![2.0, 3.0], false),
        )];

        let result = trainer.train(3, || batches.clone(), |x| x.clone());

        assert!(!result.stopped_early);
        assert_eq!(result.final_epoch, 3);
    }

    #[test]
    fn test_train_result_fields() {
        let params = vec![Tensor::from_vec(vec![1.0], true)];
        let optimizer = Adam::new(0.1, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));

        let batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0], false),
            Tensor::from_vec(vec![2.0], false),
        )];

        let result = trainer.train(2, || batches.clone(), |x| x.clone());

        // Verify all fields are populated
        assert!(result.final_loss.is_finite());
        assert!(result.best_loss.is_finite());
        assert!(
            result.best_loss <= result.final_loss
                || (result.best_loss - result.final_loss).abs() < 0.001
        );
        assert!(result.elapsed_secs >= 0.0);
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
    fn test_gradient_accumulation() {
        let params = vec![Tensor::from_vec(vec![1.0, 2.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new()
            .with_log_interval(100)
            .with_gradient_accumulation(2);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));

        // Create 4 batches - with accum_steps=2, we get 2 optimizer steps per epoch
        let batches = vec![
            Batch::new(
                Tensor::from_vec(vec![1.0, 2.0], false),
                Tensor::from_vec(vec![2.0, 3.0], false),
            ),
            Batch::new(
                Tensor::from_vec(vec![1.0, 2.0], false),
                Tensor::from_vec(vec![2.0, 3.0], false),
            ),
            Batch::new(
                Tensor::from_vec(vec![1.0, 2.0], false),
                Tensor::from_vec(vec![2.0, 3.0], false),
            ),
            Batch::new(
                Tensor::from_vec(vec![1.0, 2.0], false),
                Tensor::from_vec(vec![2.0, 3.0], false),
            ),
        ];

        let result = trainer.train(1, || batches.clone(), |x| x.clone());

        // Should complete successfully
        assert!(!result.stopped_early);
        assert_eq!(result.final_epoch, 1);
        assert!(result.final_loss.is_finite());
        // 4 batches = 4 steps
        assert_eq!(trainer.metrics.steps, 4);
    }

    #[test]
    fn test_gradient_accumulation_partial_window() {
        let params = vec![Tensor::from_vec(vec![1.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new()
            .with_log_interval(100)
            .with_gradient_accumulation(3);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));

        // Create 5 batches with accum_steps=3
        // Optimizer steps at: batch 2 (0,1,2), batch 4 (3,4 - partial)
        let batches = vec![
            Batch::new(
                Tensor::from_vec(vec![1.0], false),
                Tensor::from_vec(vec![2.0], false),
            ),
            Batch::new(
                Tensor::from_vec(vec![1.0], false),
                Tensor::from_vec(vec![2.0], false),
            ),
            Batch::new(
                Tensor::from_vec(vec![1.0], false),
                Tensor::from_vec(vec![2.0], false),
            ),
            Batch::new(
                Tensor::from_vec(vec![1.0], false),
                Tensor::from_vec(vec![2.0], false),
            ),
            Batch::new(
                Tensor::from_vec(vec![1.0], false),
                Tensor::from_vec(vec![2.0], false),
            ),
        ];

        let result = trainer.train(1, || batches.clone(), |x| x.clone());

        assert!(!result.stopped_early);
        assert_eq!(trainer.metrics.steps, 5);
        assert!(result.final_loss.is_finite());
    }
}
