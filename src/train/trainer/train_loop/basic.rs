//! Basic training loop without validation

use crate::optim::clip_grad_norm;
use crate::train::callback::CallbackAction;
use crate::train::trainer::core::Trainer;
use crate::train::trainer::result::TrainResult;
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
                elapsed_secs: self.start_time.map_or(0.0, |t| t.elapsed().as_secs_f64()),
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
            if self.best_loss.is_none_or(|bl| avg_loss < bl) {
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
            elapsed_secs: self.start_time.map_or(0.0, |t| t.elapsed().as_secs_f64()),
        }
    }
}
