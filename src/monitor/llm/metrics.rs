//! LLM call metrics tracking.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
