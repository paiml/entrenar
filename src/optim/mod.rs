//! Optimizers for training neural networks

mod adam;
mod adamw;
mod clip;
mod convergence_tests;
mod optimizer;
mod scheduler;
mod sgd;
mod simd;

pub use adam::Adam;
pub use adamw::AdamW;
pub use clip::clip_grad_norm;
pub use optimizer::Optimizer;
pub use scheduler::{CosineAnnealingLR, LRScheduler};
pub use sgd::SGD;
pub use simd::{simd_adam_update, simd_adamw_update, simd_axpy};
