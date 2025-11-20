//! Model I/O - Loading and saving models
//!
//! Provides functionality to save and load trained models, supporting
//! multiple serialization formats.

mod save;
mod load;
mod format;
mod model;

#[cfg(test)]
mod tests;

pub use save::save_model;
pub use load::load_model;
pub use format::{ModelFormat, SaveConfig};
pub use model::{Model, ModelMetadata, ParameterInfo};
