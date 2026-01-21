//! Multi-epoch training loops

use super::core::Trainer;
use super::result::TrainResult;
use crate::optim::clip_grad_norm;
use crate::train::callback::CallbackAction;
use crate::train::Batch;
use crate::Tensor;
use std::time::Instant;

impl Trainer {
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

    /// Train for multiple epochs with validation after each epoch
    ///
    /// This method runs training and validation each epoch, passing validation
    /// loss to callbacks for proper early stopping and checkpointing.
    ///
    /// # Arguments
    ///
    /// * `max_epochs` - Maximum number of epochs to train
    /// * `train_fn` - Function that returns training batches for each epoch
    /// * `val_fn` - Function that returns validation batches for each epoch
    /// * `forward_fn` - Closure that computes predictions from inputs
    ///
    /// # Returns
    ///
    /// TrainResult with final metrics including best validation loss
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use entrenar::train::{Trainer, Batch, EarlyStopping};
    /// # use entrenar::Tensor;
    /// # let mut trainer: Trainer = todo!();
    /// # let train_batches: Vec<Batch> = vec![];
    /// # let val_batches: Vec<Batch> = vec![];
    /// trainer.add_callback(EarlyStopping::new(5, 0.001).monitor_validation());
    ///
    /// let result = trainer.train_with_val(
    ///     100,
    ///     || train_batches.clone(),
    ///     || val_batches.clone(),
    ///     |x| x.clone()
    /// );
    /// println!("Best val loss: {:.4}", result.best_loss);
    /// ```
    pub fn train_with_val<F, BT, BV, IT, IV>(
        &mut self,
        max_epochs: usize,
        train_fn: BT,
        val_fn: BV,
        forward_fn: F,
    ) -> TrainResult
    where
        F: Fn(&Tensor) -> Tensor,
        BT: Fn() -> IT,
        BV: Fn() -> IV,
        IT: IntoIterator<Item = Batch>,
        IV: IntoIterator<Item = Batch>,
    {
        self.start_time = Some(Instant::now());
        self.best_loss = None;
        let mut stopped_early = false;
        let mut final_loss = 0.0;
        let mut best_val_loss: Option<f32> = None;

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

            // Training phase
            let train_batches: Vec<Batch> = train_fn().into_iter().collect();
            let steps_per_epoch = train_batches.len();
            let mut total_loss = 0.0;
            let mut num_batches = 0;
            let accum_steps = self.config.gradient_accumulation_steps.max(1);

            for (step, batch) in train_batches.into_iter().enumerate() {
                let ctx =
                    self.build_context(epoch, max_epochs, step, steps_per_epoch, final_loss, None);
                if self.callbacks.on_step_begin(&ctx) == CallbackAction::Stop {
                    stopped_early = true;
                    break;
                }

                if step % accum_steps == 0 {
                    self.optimizer.zero_grad(&mut self.params);
                }

                let loss = self.accumulate_gradients(&batch, &forward_fn);
                total_loss += loss;
                num_batches += 1;

                let is_accum_boundary = (step + 1) % accum_steps == 0;
                let is_last_batch = step + 1 == steps_per_epoch;
                if is_accum_boundary || is_last_batch {
                    if let Some(max_norm) = self.config.max_grad_norm {
                        clip_grad_norm(&mut self.params, max_norm);
                    }
                    self.optimizer.step(&mut self.params);
                }

                self.metrics.increment_step();

                let ctx = self.build_context(epoch, max_epochs, step, steps_per_epoch, loss, None);
                if self.callbacks.on_step_end(&ctx) == CallbackAction::Stop {
                    stopped_early = true;
                    break;
                }
            }

            if stopped_early {
                break;
            }

            // Calculate training loss
            let avg_train_loss = if num_batches > 0 {
                total_loss / num_batches as f32
            } else {
                0.0
            };
            final_loss = avg_train_loss;

            // Validation phase
            let val_batches: Vec<Batch> = val_fn().into_iter().collect();
            let val_loss = if val_batches.is_empty() {
                None
            } else {
                let mut val_total = 0.0;
                let mut val_count = 0;
                for batch in val_batches {
                    let predictions = forward_fn(&batch.inputs);
                    let loss = self
                        .loss_fn
                        .as_ref()
                        .unwrap()
                        .forward(&predictions, &batch.targets);
                    val_total += loss.data()[0];
                    val_count += 1;
                }
                let val_avg = if val_count > 0 {
                    val_total / val_count as f32
                } else {
                    0.0
                };
                self.metrics.record_val_loss(val_avg);
                Some(val_avg)
            };

            // Update best loss (prefer val_loss if available)
            let monitored_loss = val_loss.unwrap_or(avg_train_loss);
            if best_val_loss.is_none() || monitored_loss < best_val_loss.unwrap() {
                best_val_loss = Some(monitored_loss);
            }
            if self.best_loss.is_none() || avg_train_loss < self.best_loss.unwrap() {
                self.best_loss = Some(avg_train_loss);
            }

            // Record epoch metrics
            self.metrics.record_epoch(avg_train_loss, self.lr());

            // Fire epoch_end with val_loss
            let ctx = self.build_context(
                epoch,
                max_epochs,
                steps_per_epoch,
                steps_per_epoch,
                avg_train_loss,
                val_loss,
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
            best_loss: best_val_loss.unwrap_or(self.best_loss.unwrap_or(final_loss)),
            stopped_early,
            elapsed_secs: self.start_time.unwrap().elapsed().as_secs_f64(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::optim::Adam;
    use crate::train::callback::{CallbackAction, CallbackContext, TrainerCallback};
    use crate::train::{Batch, EarlyStopping, MSELoss, TrainConfig, Trainer};
    use crate::Tensor;

    #[test]
    fn test_train_with_callbacks() {
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

        let result = trainer.train(10, || batches.clone(), std::clone::Clone::clone);

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

        let result = trainer.train(3, || batches.clone(), std::clone::Clone::clone);

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

        let result = trainer.train(2, || batches.clone(), std::clone::Clone::clone);

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

        let result = trainer.train(1, || batches.clone(), std::clone::Clone::clone);

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

        let result = trainer.train(1, || batches.clone(), std::clone::Clone::clone);

        assert!(!result.stopped_early);
        assert_eq!(trainer.metrics.steps, 5);
        assert!(result.final_loss.is_finite());
    }

    #[test]
    fn test_early_stopping_monitor_validation() {
        let params = vec![Tensor::from_vec(vec![1.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));
        trainer.add_callback(EarlyStopping::new(2, 0.0001).monitor_validation());

        let batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0], false),
            Tensor::from_vec(vec![2.0], false),
        )];

        let result = trainer.train(10, || batches.clone(), std::clone::Clone::clone);

        // Should run and eventually stop (training loss is used as fallback)
        assert!(result.final_loss.is_finite());
    }

    #[test]
    fn test_train_with_val() {
        let params = vec![Tensor::from_vec(vec![1.0, 2.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));

        let train_batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0, 2.0], false),
            Tensor::from_vec(vec![2.0, 3.0], false),
        )];

        let val_batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0, 2.0], false),
            Tensor::from_vec(vec![2.5, 3.5], false),
        )];

        let result = trainer.train_with_val(
            3,
            || train_batches.clone(),
            || val_batches.clone(),
            std::clone::Clone::clone,
        );

        assert!(!result.stopped_early);
        assert_eq!(result.final_epoch, 3);
        assert!(result.final_loss.is_finite());
        assert!(result.best_loss.is_finite());
        // Val losses should be tracked
        assert_eq!(trainer.metrics.val_losses.len(), 3);
    }

    #[test]
    fn test_train_with_val_early_stopping() {
        let params = vec![Tensor::from_vec(vec![1.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));
        trainer.add_callback(EarlyStopping::new(2, 0.0001).monitor_validation());

        let train_batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0], false),
            Tensor::from_vec(vec![2.0], false),
        )];

        // Validation with same targets - constant val loss triggers early stopping
        let val_batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0], false),
            Tensor::from_vec(vec![2.0], false),
        )];

        let result = trainer.train_with_val(
            100,
            || train_batches.clone(),
            || val_batches.clone(),
            std::clone::Clone::clone,
        );

        // Should stop early due to no val improvement
        assert!(result.stopped_early);
        assert!(result.final_epoch < 100);
    }

    #[test]
    fn test_train_with_val_empty_validation() {
        let params = vec![Tensor::from_vec(vec![1.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));

        let train_batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0], false),
            Tensor::from_vec(vec![2.0], false),
        )];

        // Empty validation - should still work
        let val_batches: Vec<Batch> = vec![];

        let result = trainer.train_with_val(
            2,
            || train_batches.clone(),
            || val_batches.clone(),
            std::clone::Clone::clone,
        );

        assert!(!result.stopped_early);
        assert_eq!(result.final_epoch, 2);
        // No val losses recorded
        assert_eq!(trainer.metrics.val_losses.len(), 0);
    }

    #[test]
    fn test_train_stops_at_train_begin() {
        struct StopAtBeginCallback;
        impl TrainerCallback for StopAtBeginCallback {
            fn on_train_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "StopAtBegin"
            }
        }

        let params = vec![Tensor::from_vec(vec![1.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));
        trainer.add_callback(StopAtBeginCallback);

        let batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0], false),
            Tensor::from_vec(vec![2.0], false),
        )];

        let result = trainer.train(10, || batches.clone(), std::clone::Clone::clone);
        assert!(result.stopped_early);
        assert_eq!(result.final_epoch, 0);
    }

    #[test]
    fn test_train_with_epoch_skip() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct SkipFirstEpochCallback {
            skipped: Arc<AtomicUsize>,
        }
        impl TrainerCallback for SkipFirstEpochCallback {
            fn on_epoch_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
                if ctx.epoch == 0 {
                    self.skipped.fetch_add(1, Ordering::SeqCst);
                    CallbackAction::SkipEpoch
                } else {
                    CallbackAction::Continue
                }
            }
            fn name(&self) -> &'static str {
                "SkipFirstEpoch"
            }
        }

        let skipped = Arc::new(AtomicUsize::new(0));
        let params = vec![Tensor::from_vec(vec![1.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));
        trainer.add_callback(SkipFirstEpochCallback {
            skipped: skipped.clone(),
        });

        let batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0], false),
            Tensor::from_vec(vec![2.0], false),
        )];

        let result = trainer.train(3, || batches.clone(), std::clone::Clone::clone);
        assert!(!result.stopped_early);
        assert_eq!(skipped.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_train_with_step_begin_stop() {
        struct StopAtStepBeginCallback;
        impl TrainerCallback for StopAtStepBeginCallback {
            fn on_step_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
                if ctx.step >= 1 {
                    CallbackAction::Stop
                } else {
                    CallbackAction::Continue
                }
            }
            fn name(&self) -> &'static str {
                "StopAtStepBegin"
            }
        }

        let params = vec![Tensor::from_vec(vec![1.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));
        trainer.add_callback(StopAtStepBeginCallback);

        let batches = vec![
            Batch::new(
                Tensor::from_vec(vec![1.0], false),
                Tensor::from_vec(vec![2.0], false),
            ),
            Batch::new(
                Tensor::from_vec(vec![1.0], false),
                Tensor::from_vec(vec![2.0], false),
            ),
        ];

        let result = trainer.train(10, || batches.clone(), std::clone::Clone::clone);
        assert!(result.stopped_early);
    }

    #[test]
    fn test_train_with_step_end_stop() {
        struct StopAtStepEndCallback;
        impl TrainerCallback for StopAtStepEndCallback {
            fn on_step_end(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "StopAtStepEnd"
            }
        }

        let params = vec![Tensor::from_vec(vec![1.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));
        trainer.add_callback(StopAtStepEndCallback);

        let batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0], false),
            Tensor::from_vec(vec![2.0], false),
        )];

        let result = trainer.train(10, || batches.clone(), std::clone::Clone::clone);
        assert!(result.stopped_early);
        // Only one step executed
        assert_eq!(trainer.metrics.steps, 1);
    }

    #[test]
    fn test_train_with_grad_clipping() {
        let params = vec![Tensor::from_vec(vec![1.0, 2.0], true)];
        let optimizer = Adam::new(0.1, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new()
            .with_log_interval(100)
            .with_grad_clip(1.0);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));

        let batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0, 2.0], false),
            Tensor::from_vec(vec![100.0, 200.0], false), // Large targets for big gradients
        )];

        let result = trainer.train(2, || batches.clone(), std::clone::Clone::clone);
        assert!(!result.stopped_early);
        assert!(result.final_loss.is_finite());
    }
}
