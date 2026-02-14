//! Generative AI evaluation metrics
//!
//! Metrics for evaluating generative models across domains:
//! - **ASR**: Word Error Rate (WER), Real-Time Factor inverse (RTFx)
//! - **Text Generation**: BLEU, ROUGE (1/2/L), Perplexity
//! - **Code Generation**: pass@k
//! - **Retrieval**: NDCG@k

pub mod asr;
pub mod code;
pub mod retrieval;
pub mod text_gen;

#[cfg(test)]
mod tests;

// Re-exports
pub use asr::{real_time_factor_inverse, word_error_rate};
pub use code::pass_at_k;
pub use retrieval::ndcg_at_k;
pub use text_gen::{bleu_score, perplexity, rouge_l, rouge_n};
