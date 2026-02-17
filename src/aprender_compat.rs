//! Aprender Compatibility Layer
//!
//! Re-exports from the `aprender` crate for users who need direct access to
//! aprender's ML primitives without adding a separate dependency.
//!
//! ## Architecture Boundary
//!
//! **Entrenar** owns training orchestration (autograd, optimizers, LoRA, training loop).
//! **Aprender** owns ML primitives (loss functions, metrics, pruning algorithms, HF Hub client).
//!
//! Entrenar delegates to aprender internally (e.g., regression metrics) and re-exports
//! aprender's APIs here for convenience.
//!
//! ## Loss Functions
//!
//! Aprender provides standalone loss functions that operate on `Vector<f32>`.
//! For training with autograd backward passes, use entrenar's `train::LossFn` trait instead.
//!
//! ```rust,no_run
//! use entrenar::aprender_compat::loss;
//! use entrenar::aprender_compat::primitives::Vector;
//!
//! let y_pred = Vector::from_slice(&[0.9, 0.1, 0.8]);
//! let y_true = Vector::from_slice(&[1.0, 0.0, 1.0]);
//! let error = loss::mse_loss(&y_pred, &y_true);
//! ```
//!
//! ## Metrics
//!
//! Aprender provides standalone metric functions. Entrenar's `train::Metric` trait
//! wraps these for integration with the training loop.
//!
//! ```rust,no_run
//! use entrenar::aprender_compat::metrics;
//! use entrenar::aprender_compat::primitives::Vector;
//!
//! let y_pred = Vector::from_slice(&[1.1, 2.0, 3.2]);
//! let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
//! let r2 = metrics::r_squared(&y_pred, &y_true);
//! ```
//!
//! ## Pruning
//!
//! Aprender provides low-level pruning algorithms (magnitude, WANDA, SparseGPT).
//! Entrenar's `prune` module wraps these with training-loop integration.

/// Re-export aprender's loss functions
pub mod loss {
    pub use aprender::loss::{
        dice_loss, focal_loss, hinge_loss, huber_loss, info_nce_loss, kl_divergence, mae_loss,
        mse_loss, squared_hinge_loss, triplet_loss, wasserstein_discriminator_loss,
        wasserstein_generator_loss, wasserstein_loss,
    };

    // Loss trait and struct implementations
    pub use aprender::loss::{
        CTCLoss, DiceLoss, FocalLoss, HingeLoss, HuberLoss, InfoNCELoss, Loss, MAELoss, MSELoss,
        TripletLoss, WassersteinLoss,
    };
}

/// Re-export aprender's metrics
pub mod metrics {
    pub use aprender::metrics::{mae, mse, r_squared, rmse};

    // Classification metrics
    pub use aprender::metrics::classification;

    // Ranking metrics
    pub use aprender::metrics::ranking;
}

/// Re-export aprender's pruning primitives
pub mod pruning {
    pub use aprender::pruning::{
        generate_block_mask, generate_column_mask, generate_nm_mask, generate_row_mask,
        generate_unstructured_mask, sparsify, Importance, MagnitudeImportance, MagnitudePruner,
        Pruner, PruningError, PruningResult, SparseGPTImportance, SparseTensor, SparsityMask,
        SparsityPattern, WandaImportance, WandaPruner,
    };
}

/// Re-export aprender's primitive types
pub mod primitives {
    pub use aprender::primitives::{Matrix, Vector};
}
