//! LLM Evaluation Metrics Module (#71)
//!
//! Direct observation of LLM behavior through comprehensive metrics tracking.
//!
//! # Toyota Way: 現地現物 (Genchi Genbutsu)
//!
//! "Go and see" - Direct observation of LLM behavior through metrics enables
//! data-driven decisions about prompt engineering and model selection.
//!
//! # Example
//!
//! ```ignore
//! use entrenar::monitor::llm::{LLMMetrics, PromptVersion, EvalResult, InMemoryLLMEvaluator};
//!
//! let mut evaluator = InMemoryLLMEvaluator::new();
//!
//! // Track prompt version
//! let prompt = PromptVersion::new("Summarize: {text}", vec!["text".to_string()]);
//! evaluator.track_prompt("run-1", &prompt)?;
//!
//! // Log LLM call metrics
//! let metrics = LLMMetrics::new("gpt-4")
//!     .with_tokens(100, 50)
//!     .with_latency(1500.0);
//! evaluator.log_llm_call("run-1", metrics)?;
//!
//! // Evaluate response quality
//! let result = evaluator.evaluate_response("What is 2+2?", "4", Some("4"))?;
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use thiserror::Error;

// =============================================================================
// Core Types
// =============================================================================

/// LLM evaluation errors
#[derive(Debug, Error)]
pub enum LLMError {
    #[error("Run not found: {0}")]
    RunNotFound(String),

    #[error("Prompt not found: {0}")]
    PromptNotFound(String),

    #[error("Evaluation failed: {0}")]
    EvaluationFailed(String),

    #[error("Invalid metric: {0}")]
    InvalidMetric(String),

    #[error("LLM error: {0}")]
    Internal(String),
}

/// Result type for LLM operations
pub type Result<T> = std::result::Result<T, LLMError>;

/// Prompt identifier (content-addressable)
pub type PromptId = String;

/// LLM call metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMMetrics {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,
    /// Number of tokens in the completion
    pub completion_tokens: u32,
    /// Total tokens (prompt + completion)
    pub total_tokens: u32,
    /// Time to first token in milliseconds
    pub time_to_first_token_ms: f64,
    /// Tokens generated per second
    pub tokens_per_second: f64,
    /// Total latency in milliseconds
    pub latency_ms: f64,
    /// Estimated cost in USD (if known)
    pub cost_usd: Option<f64>,
    /// Model name (e.g., "gpt-4", "claude-3-opus")
    pub model_name: String,
    /// Timestamp of the call
    pub timestamp: DateTime<Utc>,
    /// Optional request ID
    pub request_id: Option<String>,
    /// Optional tags
    pub tags: HashMap<String, String>,
}

impl LLMMetrics {
    /// Create new LLM metrics with model name
    pub fn new(model_name: &str) -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            time_to_first_token_ms: 0.0,
            tokens_per_second: 0.0,
            latency_ms: 0.0,
            cost_usd: None,
            model_name: model_name.to_string(),
            timestamp: Utc::now(),
            request_id: None,
            tags: HashMap::new(),
        }
    }

    /// Set token counts
    pub fn with_tokens(mut self, prompt: u32, completion: u32) -> Self {
        self.prompt_tokens = prompt;
        self.completion_tokens = completion;
        self.total_tokens = prompt + completion;
        self
    }

    /// Set latency
    pub fn with_latency(mut self, latency_ms: f64) -> Self {
        self.latency_ms = latency_ms;
        if latency_ms > 0.0 && self.completion_tokens > 0 {
            self.tokens_per_second = f64::from(self.completion_tokens) / (latency_ms / 1000.0);
        }
        self
    }

    /// Set time to first token
    pub fn with_ttft(mut self, ttft_ms: f64) -> Self {
        self.time_to_first_token_ms = ttft_ms;
        self
    }

    /// Set cost
    pub fn with_cost(mut self, cost_usd: f64) -> Self {
        self.cost_usd = Some(cost_usd);
        self
    }

    /// Set request ID
    pub fn with_request_id(mut self, id: &str) -> Self {
        self.request_id = Some(id.to_string());
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, key: &str, value: &str) -> Self {
        self.tags.insert(key.to_string(), value.to_string());
        self
    }

    /// Calculate cost based on model pricing (approximate)
    pub fn estimate_cost(&self) -> f64 {
        // Approximate pricing per 1K tokens (as of late 2024)
        let (prompt_price, completion_price) = match self.model_name.as_str() {
            m if m.contains("gpt-4-turbo") => (0.01, 0.03),
            m if m.contains("gpt-4") => (0.03, 0.06),
            m if m.contains("gpt-3.5") => (0.0005, 0.0015),
            m if m.contains("claude-3-opus") => (0.015, 0.075),
            m if m.contains("claude-3-sonnet") => (0.003, 0.015),
            m if m.contains("claude-3-haiku") => (0.00025, 0.00125),
            _ => (0.001, 0.002), // Default conservative estimate
        };

        let prompt_cost = (f64::from(self.prompt_tokens) / 1000.0) * prompt_price;
        let completion_cost = (f64::from(self.completion_tokens) / 1000.0) * completion_price;
        prompt_cost + completion_cost
    }
}

/// Prompt version with content-addressable ID
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptVersion {
    /// Content-addressable ID (SHA-256 of template)
    pub id: PromptId,
    /// Prompt template (with {variable} placeholders)
    pub template: String,
    /// Variable names in the template
    pub variables: Vec<String>,
    /// Version number (monotonically increasing per template family)
    pub version: u32,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// SHA-256 hash of the template
    pub sha256: String,
    /// Optional description
    pub description: Option<String>,
    /// Optional tags
    pub tags: HashMap<String, String>,
}

impl PromptVersion {
    /// Create a new prompt version
    pub fn new(template: &str, variables: Vec<String>) -> Self {
        let sha256 = Self::compute_hash(template);
        let id = sha256[..16].to_string(); // Short ID from hash

        Self {
            id,
            template: template.to_string(),
            variables,
            version: 1,
            created_at: Utc::now(),
            sha256,
            description: None,
            tags: HashMap::new(),
        }
    }

    /// Create with specific version number
    pub fn with_version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    /// Add description
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Add tag
    pub fn with_tag(mut self, key: &str, value: &str) -> Self {
        self.tags.insert(key.to_string(), value.to_string());
        self
    }

    /// Compute SHA-256 hash of template
    fn compute_hash(template: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(template.as_bytes());
        let result = hasher.finalize();
        hex::encode(result)
    }

    /// Render template with variables
    pub fn render(&self, vars: &HashMap<String, String>) -> Result<String> {
        let mut result = self.template.clone();
        for var in &self.variables {
            let placeholder = format!("{{{var}}}");
            if let Some(value) = vars.get(var) {
                result = result.replace(&placeholder, value);
            } else {
                return Err(LLMError::EvaluationFailed(format!(
                    "Missing variable: {var}"
                )));
            }
        }
        Ok(result)
    }

    /// Extract variables from template
    pub fn extract_variables(template: &str) -> Vec<String> {
        let mut vars = Vec::new();
        let mut in_var = false;
        let mut current = String::new();

        for c in template.chars() {
            match c {
                '{' => {
                    in_var = true;
                    current.clear();
                }
                '}' if in_var => {
                    if !current.is_empty() && !vars.contains(&current) {
                        vars.push(current.clone());
                    }
                    in_var = false;
                }
                _ if in_var => {
                    current.push(c);
                }
                _ => {}
            }
        }
        vars
    }
}

/// Evaluation result scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    /// Relevance to the query (0-1, higher is better)
    pub relevance: f64,
    /// Logical consistency (0-1, higher is better)
    pub coherence: f64,
    /// Factual accuracy (0-1, higher is better)
    pub groundedness: f64,
    /// Potential harm score (0-1, lower is better)
    pub harmfulness: f64,
    /// Optional detailed scores
    pub details: HashMap<String, f64>,
    /// Overall score (weighted average)
    pub overall: f64,
}

impl EvalResult {
    /// Create a new evaluation result
    pub fn new(relevance: f64, coherence: f64, groundedness: f64, harmfulness: f64) -> Self {
        let overall = Self::compute_overall(relevance, coherence, groundedness, harmfulness);
        Self {
            relevance: relevance.clamp(0.0, 1.0),
            coherence: coherence.clamp(0.0, 1.0),
            groundedness: groundedness.clamp(0.0, 1.0),
            harmfulness: harmfulness.clamp(0.0, 1.0),
            details: HashMap::new(),
            overall,
        }
    }

    /// Add a detail score
    pub fn with_detail(mut self, name: &str, score: f64) -> Self {
        self.details.insert(name.to_string(), score.clamp(0.0, 1.0));
        self
    }

    /// Compute overall score
    fn compute_overall(relevance: f64, coherence: f64, groundedness: f64, harmfulness: f64) -> f64 {
        // Weighted average (harmfulness is inverted since lower is better)
        let weights = [0.3, 0.2, 0.3, 0.2]; // relevance, coherence, groundedness, safety
        let scores = [
            relevance,
            coherence,
            groundedness,
            1.0 - harmfulness, // Invert harmfulness
        ];

        let weighted_sum: f64 = weights.iter().zip(scores.iter()).map(|(w, s)| w * s).sum();
        weighted_sum.clamp(0.0, 1.0)
    }

    /// Check if result passes quality threshold
    pub fn passes_threshold(&self, min_overall: f64, max_harmfulness: f64) -> bool {
        self.overall >= min_overall && self.harmfulness <= max_harmfulness
    }
}

impl Default for EvalResult {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
}

// =============================================================================
// LLM Evaluator Trait
// =============================================================================

/// Trait for LLM evaluation systems
pub trait LLMEvaluator: Send + Sync {
    /// Evaluate response quality
    fn evaluate_response(
        &self,
        prompt: &str,
        response: &str,
        reference: Option<&str>,
    ) -> Result<EvalResult>;

    /// Log LLM call metrics
    fn log_llm_call(&mut self, run_id: &str, metrics: LLMMetrics) -> Result<()>;

    /// Track prompt version
    fn track_prompt(&mut self, run_id: &str, prompt: &PromptVersion) -> Result<()>;

    /// Get metrics for a run
    fn get_metrics(&self, run_id: &str) -> Result<Vec<LLMMetrics>>;

    /// Get prompts for a run
    fn get_prompts(&self, run_id: &str) -> Result<Vec<PromptVersion>>;

    /// Get aggregate statistics
    fn get_stats(&self, run_id: &str) -> Result<LLMStats>;
}

/// Aggregate LLM statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LLMStats {
    /// Number of calls
    pub n_calls: usize,
    /// Total tokens used
    pub total_tokens: u64,
    /// Total prompt tokens
    pub total_prompt_tokens: u64,
    /// Total completion tokens
    pub total_completion_tokens: u64,
    /// Total estimated cost
    pub total_cost: f64,
    /// Average latency
    pub avg_latency_ms: f64,
    /// Average tokens per second
    pub avg_tokens_per_second: f64,
    /// P50 latency
    pub p50_latency_ms: f64,
    /// P95 latency
    pub p95_latency_ms: f64,
    /// P99 latency
    pub p99_latency_ms: f64,
}

impl LLMStats {
    /// Compute stats from metrics
    pub fn from_metrics(metrics: &[LLMMetrics]) -> Self {
        if metrics.is_empty() {
            return Self::default();
        }

        let n = metrics.len();
        let total_tokens: u64 = metrics.iter().map(|m| u64::from(m.total_tokens)).sum();
        let total_prompt: u64 = metrics.iter().map(|m| u64::from(m.prompt_tokens)).sum();
        let total_completion: u64 = metrics.iter().map(|m| u64::from(m.completion_tokens)).sum();
        let total_cost: f64 = metrics
            .iter()
            .map(|m| m.cost_usd.unwrap_or_else(|| m.estimate_cost()))
            .sum();

        let avg_latency: f64 = metrics.iter().map(|m| m.latency_ms).sum::<f64>() / n as f64;
        let avg_tps: f64 = metrics.iter().map(|m| m.tokens_per_second).sum::<f64>() / n as f64;

        // Compute percentiles
        let mut latencies: Vec<f64> = metrics.iter().map(|m| m.latency_ms).collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50 = percentile(&latencies, 50.0);
        let p95 = percentile(&latencies, 95.0);
        let p99 = percentile(&latencies, 99.0);

        Self {
            n_calls: n,
            total_tokens,
            total_prompt_tokens: total_prompt,
            total_completion_tokens: total_completion,
            total_cost,
            avg_latency_ms: avg_latency,
            avg_tokens_per_second: avg_tps,
            p50_latency_ms: p50,
            p95_latency_ms: p95,
            p99_latency_ms: p99,
        }
    }
}

/// Compute percentile from sorted array
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// =============================================================================
// In-Memory Evaluator Implementation
// =============================================================================

/// In-memory LLM evaluator for testing
#[derive(Debug, Default)]
pub struct InMemoryLLMEvaluator {
    /// Metrics by run ID
    metrics: Arc<RwLock<HashMap<String, Vec<LLMMetrics>>>>,
    /// Prompts by run ID
    prompts: Arc<RwLock<HashMap<String, Vec<PromptVersion>>>>,
}

impl InMemoryLLMEvaluator {
    /// Create a new in-memory evaluator
    pub fn new() -> Self {
        Self::default()
    }
}

impl LLMEvaluator for InMemoryLLMEvaluator {
    fn evaluate_response(
        &self,
        prompt: &str,
        response: &str,
        reference: Option<&str>,
    ) -> Result<EvalResult> {
        // Simple heuristic evaluation (production would use a model)
        let relevance = compute_relevance(prompt, response);
        let coherence = compute_coherence(response);
        let groundedness = if let Some(ref_text) = reference {
            compute_groundedness(response, ref_text)
        } else {
            0.5 // Unknown without reference
        };
        let harmfulness = compute_harmfulness(response);

        Ok(EvalResult::new(
            relevance,
            coherence,
            groundedness,
            harmfulness,
        ))
    }

    fn log_llm_call(&mut self, run_id: &str, metrics: LLMMetrics) -> Result<()> {
        let mut store = self
            .metrics
            .write()
            .map_err(|e| LLMError::Internal(format!("Lock error: {e}")))?;

        store.entry(run_id.to_string()).or_default().push(metrics);

        Ok(())
    }

    fn track_prompt(&mut self, run_id: &str, prompt: &PromptVersion) -> Result<()> {
        let mut store = self
            .prompts
            .write()
            .map_err(|e| LLMError::Internal(format!("Lock error: {e}")))?;

        store
            .entry(run_id.to_string())
            .or_default()
            .push(prompt.clone());

        Ok(())
    }

    fn get_metrics(&self, run_id: &str) -> Result<Vec<LLMMetrics>> {
        let store = self
            .metrics
            .read()
            .map_err(|e| LLMError::Internal(format!("Lock error: {e}")))?;

        store
            .get(run_id)
            .cloned()
            .ok_or_else(|| LLMError::RunNotFound(run_id.to_string()))
    }

    fn get_prompts(&self, run_id: &str) -> Result<Vec<PromptVersion>> {
        let store = self
            .prompts
            .read()
            .map_err(|e| LLMError::Internal(format!("Lock error: {e}")))?;

        store
            .get(run_id)
            .cloned()
            .ok_or_else(|| LLMError::RunNotFound(run_id.to_string()))
    }

    fn get_stats(&self, run_id: &str) -> Result<LLMStats> {
        let metrics = self.get_metrics(run_id)?;
        Ok(LLMStats::from_metrics(&metrics))
    }
}

// =============================================================================
// Heuristic Evaluation Functions
// =============================================================================

/// Compute relevance score (word overlap heuristic)
fn compute_relevance(prompt: &str, response: &str) -> f64 {
    let prompt_lower = prompt.to_lowercase();
    let response_lower = response.to_lowercase();
    let prompt_words: std::collections::HashSet<&str> = prompt_lower.split_whitespace().collect();
    let response_words: std::collections::HashSet<&str> =
        response_lower.split_whitespace().collect();

    if prompt_words.is_empty() {
        return 0.5;
    }

    let overlap = prompt_words.intersection(&response_words).count();
    let jaccard = overlap as f64 / (prompt_words.len() + response_words.len() - overlap) as f64;

    // Scale to [0.3, 1.0] range (some overlap is expected)
    (0.3 + jaccard * 0.7).min(1.0)
}

/// Compute coherence score (sentence structure heuristic)
fn compute_coherence(response: &str) -> f64 {
    if response.is_empty() {
        return 0.0;
    }

    let sentences: Vec<&str> = response
        .split(['.', '!', '?'])
        .filter(|s| !s.trim().is_empty())
        .collect();

    if sentences.is_empty() {
        return 0.3; // No clear sentences
    }

    // Check for basic sentence structure
    let mut score: f64 = 0.7;

    // Bonus for multiple sentences
    if sentences.len() > 1 {
        score += 0.1;
    }

    // Bonus for reasonable length
    let avg_words = response.split_whitespace().count() as f64 / sentences.len() as f64;
    if (5.0..=30.0).contains(&avg_words) {
        score += 0.1;
    }

    // Penalty for all caps
    if response
        .chars()
        .filter(|c| c.is_alphabetic())
        .all(char::is_uppercase)
    {
        score -= 0.3;
    }

    score.clamp(0.0, 1.0)
}

/// Compute groundedness score (overlap with reference)
fn compute_groundedness(response: &str, reference: &str) -> f64 {
    let ref_lower = reference.to_lowercase();
    let resp_lower = response.to_lowercase();
    let ref_words: std::collections::HashSet<&str> = ref_lower.split_whitespace().collect();
    let resp_words: std::collections::HashSet<&str> = resp_lower.split_whitespace().collect();

    if ref_words.is_empty() {
        return 0.5;
    }

    // How much of the response is grounded in the reference
    let grounded = resp_words.intersection(&ref_words).count() as f64;
    let coverage = grounded / resp_words.len().max(1) as f64;

    (0.2 + coverage * 0.8).min(1.0)
}

/// Compute harmfulness score (keyword-based heuristic)
fn compute_harmfulness(response: &str) -> f64 {
    let response_lower = response.to_lowercase();

    // Simple keyword detection (production would use a classifier)
    let harmful_patterns = [
        "kill",
        "harm",
        "attack",
        "bomb",
        "weapon",
        "hate",
        "racist",
        "illegal",
        "drugs",
        "exploit",
        "hack into",
        "steal",
    ];

    let matches = harmful_patterns
        .iter()
        .filter(|p| response_lower.contains(*p))
        .count() as f64;

    // Scale: 0 matches = 0.0, 3+ matches = 1.0
    (matches / 3.0).min(1.0)
}

// =============================================================================
// Tests (TDD - Written First)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // LLMMetrics Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_llm_metrics_new() {
        let metrics = LLMMetrics::new("gpt-4");
        assert_eq!(metrics.model_name, "gpt-4");
        assert_eq!(metrics.total_tokens, 0);
    }

    #[test]
    fn test_llm_metrics_with_tokens() {
        let metrics = LLMMetrics::new("gpt-4").with_tokens(100, 50);
        assert_eq!(metrics.prompt_tokens, 100);
        assert_eq!(metrics.completion_tokens, 50);
        assert_eq!(metrics.total_tokens, 150);
    }

    #[test]
    fn test_llm_metrics_with_latency() {
        let metrics = LLMMetrics::new("gpt-4")
            .with_tokens(100, 50)
            .with_latency(1000.0);
        assert_eq!(metrics.latency_ms, 1000.0);
        assert!((metrics.tokens_per_second - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_llm_metrics_estimate_cost() {
        let metrics = LLMMetrics::new("gpt-4").with_tokens(1000, 500);
        let cost = metrics.estimate_cost();
        // gpt-4: $0.03/1K prompt + $0.06/1K completion
        // = 0.03 + 0.03 = 0.06
        assert!((cost - 0.06).abs() < 0.001);
    }

    #[test]
    fn test_llm_metrics_with_tag() {
        let metrics = LLMMetrics::new("gpt-4").with_tag("purpose", "summarization");
        assert_eq!(
            metrics.tags.get("purpose"),
            Some(&"summarization".to_string())
        );
    }

    // -------------------------------------------------------------------------
    // PromptVersion Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_prompt_version_new() {
        let prompt = PromptVersion::new("Hello {name}!", vec!["name".to_string()]);
        assert!(!prompt.id.is_empty());
        assert_eq!(prompt.template, "Hello {name}!");
        assert_eq!(prompt.variables, vec!["name"]);
        assert_eq!(prompt.version, 1);
    }

    #[test]
    fn test_prompt_version_hash_deterministic() {
        let p1 = PromptVersion::new("Test template", vec![]);
        let p2 = PromptVersion::new("Test template", vec![]);
        assert_eq!(p1.sha256, p2.sha256);
        assert_eq!(p1.id, p2.id);
    }

    #[test]
    fn test_prompt_version_hash_different() {
        let p1 = PromptVersion::new("Template A", vec![]);
        let p2 = PromptVersion::new("Template B", vec![]);
        assert_ne!(p1.sha256, p2.sha256);
    }

    #[test]
    fn test_prompt_version_render() {
        let prompt = PromptVersion::new(
            "Hello {name}! You are {age} years old.",
            vec!["name".to_string(), "age".to_string()],
        );

        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "Alice".to_string());
        vars.insert("age".to_string(), "30".to_string());

        let rendered = prompt.render(&vars).unwrap();
        assert_eq!(rendered, "Hello Alice! You are 30 years old.");
    }

    #[test]
    fn test_prompt_version_render_missing_var() {
        let prompt = PromptVersion::new("Hello {name}!", vec!["name".to_string()]);
        let vars = HashMap::new();
        let result = prompt.render(&vars);
        assert!(result.is_err());
    }

    #[test]
    fn test_prompt_version_extract_variables() {
        let vars = PromptVersion::extract_variables("Hello {name}, your ID is {id}.");
        assert_eq!(vars, vec!["name", "id"]);
    }

    #[test]
    fn test_prompt_version_with_version() {
        let prompt = PromptVersion::new("Test", vec![]).with_version(5);
        assert_eq!(prompt.version, 5);
    }

    // -------------------------------------------------------------------------
    // EvalResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_eval_result_new() {
        let result = EvalResult::new(0.8, 0.9, 0.7, 0.1);
        assert_eq!(result.relevance, 0.8);
        assert_eq!(result.coherence, 0.9);
        assert_eq!(result.groundedness, 0.7);
        assert_eq!(result.harmfulness, 0.1);
    }

    #[test]
    fn test_eval_result_clamped() {
        let result = EvalResult::new(1.5, -0.1, 0.5, 2.0);
        assert_eq!(result.relevance, 1.0);
        assert_eq!(result.coherence, 0.0);
        assert_eq!(result.harmfulness, 1.0);
    }

    #[test]
    fn test_eval_result_overall() {
        let result = EvalResult::new(1.0, 1.0, 1.0, 0.0);
        // Perfect scores: overall should be 1.0
        assert!((result.overall - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_eval_result_passes_threshold() {
        let good = EvalResult::new(0.9, 0.9, 0.9, 0.1);
        let bad = EvalResult::new(0.3, 0.3, 0.3, 0.8);

        assert!(good.passes_threshold(0.7, 0.3));
        assert!(!bad.passes_threshold(0.7, 0.3));
    }

    #[test]
    fn test_eval_result_with_detail() {
        let result = EvalResult::new(0.8, 0.9, 0.7, 0.1).with_detail("fluency", 0.95);
        assert_eq!(result.details.get("fluency"), Some(&0.95));
    }

    // -------------------------------------------------------------------------
    // LLMStats Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_llm_stats_empty() {
        let stats = LLMStats::from_metrics(&[]);
        assert_eq!(stats.n_calls, 0);
        assert_eq!(stats.total_tokens, 0);
    }

    #[test]
    fn test_llm_stats_single() {
        let metrics = vec![LLMMetrics::new("gpt-4")
            .with_tokens(100, 50)
            .with_latency(1000.0)];
        let stats = LLMStats::from_metrics(&metrics);
        assert_eq!(stats.n_calls, 1);
        assert_eq!(stats.total_tokens, 150);
        assert_eq!(stats.avg_latency_ms, 1000.0);
    }

    #[test]
    fn test_llm_stats_multiple() {
        let metrics = vec![
            LLMMetrics::new("gpt-4")
                .with_tokens(100, 50)
                .with_latency(1000.0),
            LLMMetrics::new("gpt-4")
                .with_tokens(200, 100)
                .with_latency(2000.0),
        ];
        let stats = LLMStats::from_metrics(&metrics);
        assert_eq!(stats.n_calls, 2);
        assert_eq!(stats.total_tokens, 450);
        assert_eq!(stats.avg_latency_ms, 1500.0);
    }

    // -------------------------------------------------------------------------
    // InMemoryLLMEvaluator Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_evaluator_new() {
        let evaluator = InMemoryLLMEvaluator::new();
        let result = evaluator.get_metrics("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_evaluator_log_metrics() {
        let mut evaluator = InMemoryLLMEvaluator::new();
        let metrics = LLMMetrics::new("gpt-4").with_tokens(100, 50);

        evaluator.log_llm_call("run-1", metrics).unwrap();

        let retrieved = evaluator.get_metrics("run-1").unwrap();
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].total_tokens, 150);
    }

    #[test]
    fn test_evaluator_track_prompt() {
        let mut evaluator = InMemoryLLMEvaluator::new();
        let prompt = PromptVersion::new("Test prompt", vec![]);

        evaluator.track_prompt("run-1", &prompt).unwrap();

        let retrieved = evaluator.get_prompts("run-1").unwrap();
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].template, "Test prompt");
    }

    #[test]
    fn test_evaluator_evaluate_response() {
        let evaluator = InMemoryLLMEvaluator::new();

        let result = evaluator
            .evaluate_response(
                "What is the capital of France?",
                "The capital of France is Paris.",
                Some("Paris is the capital of France"),
            )
            .unwrap();

        assert!(result.relevance > 0.3);
        assert!(result.coherence > 0.5);
        assert!(result.groundedness > 0.5);
        assert!(result.harmfulness < 0.5);
    }

    #[test]
    fn test_evaluator_get_stats() {
        let mut evaluator = InMemoryLLMEvaluator::new();

        evaluator
            .log_llm_call(
                "run-1",
                LLMMetrics::new("gpt-4")
                    .with_tokens(100, 50)
                    .with_latency(500.0),
            )
            .unwrap();
        evaluator
            .log_llm_call(
                "run-1",
                LLMMetrics::new("gpt-4")
                    .with_tokens(200, 100)
                    .with_latency(1500.0),
            )
            .unwrap();

        let stats = evaluator.get_stats("run-1").unwrap();
        assert_eq!(stats.n_calls, 2);
        assert_eq!(stats.total_tokens, 450);
    }

    // -------------------------------------------------------------------------
    // Heuristic Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compute_relevance() {
        let high = compute_relevance("capital France", "The capital of France is Paris");
        let low = compute_relevance("weather today", "The capital of France is Paris");
        assert!(high > low);
    }

    #[test]
    fn test_compute_coherence() {
        let good = compute_coherence("This is a well-formed sentence. It has good structure.");
        let bad = compute_coherence("BAD ALL CAPS TEXT");
        assert!(good > bad);
    }

    #[test]
    fn test_compute_groundedness() {
        let grounded =
            compute_groundedness("Paris is the capital", "Paris is the capital of France");
        let ungrounded = compute_groundedness("Tokyo is in Asia", "Paris is the capital of France");
        assert!(grounded > ungrounded);
    }

    #[test]
    fn test_compute_harmfulness() {
        let safe = compute_harmfulness("The weather is nice today.");
        let harmful = compute_harmfulness("How to hack into systems and steal data.");
        assert!(safe < harmful);
    }
}

// =============================================================================
// Property Tests
// =============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_eval_result_overall_bounded(
            relevance in 0.0f64..1.0,
            coherence in 0.0f64..1.0,
            groundedness in 0.0f64..1.0,
            harmfulness in 0.0f64..1.0
        ) {
            let result = EvalResult::new(relevance, coherence, groundedness, harmfulness);
            prop_assert!(result.overall >= 0.0);
            prop_assert!(result.overall <= 1.0);
        }

        #[test]
        fn prop_llm_metrics_tokens_sum(prompt in 0u32..10000, completion in 0u32..10000) {
            let metrics = LLMMetrics::new("test").with_tokens(prompt, completion);
            prop_assert_eq!(metrics.total_tokens, prompt + completion);
        }

        #[test]
        fn prop_prompt_hash_deterministic(template in "[a-zA-Z0-9 ]{1,100}") {
            let p1 = PromptVersion::new(&template, vec![]);
            let p2 = PromptVersion::new(&template, vec![]);
            prop_assert_eq!(p1.sha256, p2.sha256);
        }

        #[test]
        fn prop_llm_stats_tokens_consistent(
            calls in prop::collection::vec(
                (0u32..1000, 0u32..1000),
                1..10
            )
        ) {
            let metrics: Vec<LLMMetrics> = calls
                .iter()
                .map(|(p, c)| LLMMetrics::new("test").with_tokens(*p, *c))
                .collect();

            let stats = LLMStats::from_metrics(&metrics);
            let expected_total: u64 = calls.iter().map(|(p, c)| (*p + *c) as u64).sum();
            prop_assert_eq!(stats.total_tokens, expected_total);
        }

        #[test]
        fn prop_percentile_bounded(values in prop::collection::vec(0.0f64..1000.0, 1..100)) {
            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let p50 = percentile(&sorted, 50.0);
            let min = sorted.first().unwrap();
            let max = sorted.last().unwrap();

            prop_assert!(p50 >= *min);
            prop_assert!(p50 <= *max);
        }
    }
}
