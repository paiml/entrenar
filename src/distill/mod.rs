//! Knowledge Distillation
//!
//! This module implements various knowledge distillation techniques for training
//! smaller student models from larger teacher models.
//!
//! ## Features
//!
//! - **Temperature-scaled KL divergence**: Standard distillation loss with soft targets
//! - **Multi-teacher ensemble**: Distill from multiple teachers simultaneously
//! - **Progressive distillation**: Layer-wise distillation for intermediate representations
//!
//! ## Example
//!
//! ```no_run
//! use entrenar::distill::DistillationLoss;
//!
//! let loss_fn = DistillationLoss::new(3.0, 0.5);
//! let loss = loss_fn.forward(&student_logits, &teacher_logits, &labels);
//! ```

mod loss;
mod ensemble;
mod progressive;

#[cfg(test)]
mod tests;

pub use loss::DistillationLoss;
pub use ensemble::EnsembleDistiller;
pub use progressive::ProgressiveDistiller;
