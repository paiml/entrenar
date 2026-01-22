//! Training loop with validation support

use crate::optim::clip_grad_norm;
use crate::train::callback::CallbackAction;
use crate::train::trainer::core::Trainer;
use crate::train::trainer::result::TrainResult;
use crate::train::Batch;
use crate::Tensor;
use std::time::Instant;

impl Trainer {
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
