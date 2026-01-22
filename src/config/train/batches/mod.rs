//! Training batch loading from various data formats

mod loader;
mod rebatch;

#[cfg(not(target_arch = "wasm32"))]
mod json;
#[cfg(not(target_arch = "wasm32"))]
mod parquet;

#[cfg(test)]
mod tests;

pub use loader::load_training_batches;
pub use rebatch::rebatch;

#[cfg(not(target_arch = "wasm32"))]
pub use json::load_json_batches;
#[cfg(not(target_arch = "wasm32"))]
pub use parquet::load_parquet_batches;
