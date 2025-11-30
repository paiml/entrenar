//! Batuta GPU Pricing and Queue Integration (ENT-030, ENT-031)
//!
//! Provides integration with Batuta for:
//! - Real-time GPU hourly rates
//! - Queue depth monitoring
//! - ETA adjustments based on queue state
//!
//! Falls back to static pricing when Batuta is unavailable.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Errors that can occur when interacting with Batuta.
#[derive(Debug, thiserror::Error)]
pub enum BatutaError {
    /// Batuta service is unavailable
    #[error("Batuta service unavailable: {0}")]
    ServiceUnavailable(String),

    /// Invalid GPU type requested
    #[error("Unknown GPU type: {0}")]
    UnknownGpuType(String),

    /// Network or connection error
    #[error("Connection error: {0}")]
    ConnectionError(String),

    /// Response parsing error
    #[error("Invalid response: {0}")]
    InvalidResponse(String),
}

/// GPU pricing information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuPricing {
    /// GPU type identifier (e.g., "a100-80gb", "v100", "t4")
    pub gpu_type: String,
    /// Hourly rate in USD
    pub hourly_rate: f64,
    /// GPU memory in GB
    pub memory_gb: u32,
    /// Whether this is spot/preemptible pricing
    pub is_spot: bool,
    /// Provider name (e.g., "aws", "gcp", "azure")
    pub provider: String,
    /// Region identifier
    pub region: String,
}

impl GpuPricing {
    /// Create a new GPU pricing entry.
    pub fn new(gpu_type: impl Into<String>, hourly_rate: f64, memory_gb: u32) -> Self {
        Self {
            gpu_type: gpu_type.into(),
            hourly_rate,
            memory_gb,
            is_spot: false,
            provider: "unknown".to_string(),
            region: "unknown".to_string(),
        }
    }

    /// Set spot pricing flag.
    pub fn with_spot(mut self, is_spot: bool) -> Self {
        self.is_spot = is_spot;
        self
    }

    /// Set provider.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    /// Set region.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self
    }
}

/// Queue state information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueueState {
    /// Current queue depth (number of waiting jobs)
    pub queue_depth: u32,
    /// Average wait time in seconds
    pub avg_wait_seconds: u64,
    /// Number of available GPUs
    pub available_gpus: u32,
    /// Total GPUs in the pool
    pub total_gpus: u32,
    /// Estimated time until next available slot (seconds)
    pub eta_seconds: Option<u64>,
}

impl QueueState {
    /// Create a new queue state.
    pub fn new(queue_depth: u32, available_gpus: u32, total_gpus: u32) -> Self {
        Self {
            queue_depth,
            avg_wait_seconds: 0,
            available_gpus,
            total_gpus,
            eta_seconds: None,
        }
    }

    /// Set average wait time.
    pub fn with_avg_wait(mut self, seconds: u64) -> Self {
        self.avg_wait_seconds = seconds;
        self
    }

    /// Set ETA to next available slot.
    pub fn with_eta(mut self, seconds: u64) -> Self {
        self.eta_seconds = Some(seconds);
        self
    }

    /// Check if GPUs are immediately available.
    pub fn is_available(&self) -> bool {
        self.available_gpus > 0
    }

    /// Calculate queue utilization (0.0 to 1.0).
    pub fn utilization(&self) -> f64 {
        if self.total_gpus == 0 {
            return 1.0;
        }
        1.0 - (f64::from(self.available_gpus) / f64::from(self.total_gpus))
    }
}

/// Fallback pricing when Batuta is unavailable.
///
/// Uses conservative estimates based on typical cloud provider pricing.
#[derive(Debug, Clone)]
pub struct FallbackPricing {
    /// Default pricing for known GPU types
    pricing: Vec<GpuPricing>,
}

impl Default for FallbackPricing {
    fn default() -> Self {
        Self::new()
    }
}

impl FallbackPricing {
    /// Create fallback pricing with typical cloud rates.
    pub fn new() -> Self {
        Self {
            pricing: vec![
                GpuPricing::new("a100-80gb", 3.00, 80)
                    .with_provider("generic")
                    .with_region("us-east-1"),
                GpuPricing::new("a100-40gb", 2.50, 40)
                    .with_provider("generic")
                    .with_region("us-east-1"),
                GpuPricing::new("v100", 2.00, 16)
                    .with_provider("generic")
                    .with_region("us-east-1"),
                GpuPricing::new("t4", 0.50, 16)
                    .with_provider("generic")
                    .with_region("us-east-1"),
                GpuPricing::new("l4", 0.75, 24)
                    .with_provider("generic")
                    .with_region("us-east-1"),
                GpuPricing::new("a10g", 1.00, 24)
                    .with_provider("generic")
                    .with_region("us-east-1"),
                GpuPricing::new("h100-80gb", 4.50, 80)
                    .with_provider("generic")
                    .with_region("us-east-1"),
            ],
        }
    }

    /// Get pricing for a GPU type.
    pub fn get_rate(&self, gpu_type: &str) -> Option<&GpuPricing> {
        let normalized = gpu_type.to_lowercase().replace(['-', '_'], "");
        self.pricing.iter().find(|p| {
            let p_normalized = p.gpu_type.to_lowercase().replace(['-', '_'], "");
            p_normalized == normalized
        })
    }

    /// Get all available pricing.
    pub fn all_pricing(&self) -> &[GpuPricing] {
        &self.pricing
    }

    /// Add or update pricing for a GPU type.
    pub fn set_rate(&mut self, pricing: GpuPricing) {
        let normalized = pricing.gpu_type.to_lowercase().replace(['-', '_'], "");
        if let Some(existing) = self.pricing.iter_mut().find(|p| {
            let p_normalized = p.gpu_type.to_lowercase().replace(['-', '_'], "");
            p_normalized == normalized
        }) {
            *existing = pricing;
        } else {
            self.pricing.push(pricing);
        }
    }
}

/// Client for interacting with Batuta pricing and queue services.
#[derive(Debug, Clone)]
pub struct BatutaClient {
    /// Base URL for Batuta API (None if using fallback only)
    base_url: Option<String>,
    /// Fallback pricing when Batuta is unavailable
    fallback: FallbackPricing,
    /// Connection timeout
    timeout: Duration,
    /// Whether Batuta service is available
    service_available: bool,
}

impl Default for BatutaClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BatutaClient {
    /// Create a new Batuta client with fallback pricing only.
    pub fn new() -> Self {
        Self {
            base_url: None,
            fallback: FallbackPricing::new(),
            timeout: Duration::from_secs(5),
            service_available: false,
        }
    }

    /// Create a client connected to a Batuta instance.
    pub fn with_url(url: impl Into<String>) -> Self {
        Self {
            base_url: Some(url.into()),
            fallback: FallbackPricing::new(),
            timeout: Duration::from_secs(5),
            service_available: true,
        }
    }

    /// Set connection timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set custom fallback pricing.
    pub fn with_fallback(mut self, fallback: FallbackPricing) -> Self {
        self.fallback = fallback;
        self
    }

    /// Check if connected to a live Batuta service.
    pub fn is_connected(&self) -> bool {
        self.base_url.is_some() && self.service_available
    }

    /// Get hourly rate for a GPU type.
    ///
    /// Returns live pricing from Batuta if available, otherwise fallback pricing.
    pub fn get_hourly_rate(&self, gpu_type: &str) -> Result<GpuPricing, BatutaError> {
        // If we have a live connection, try to fetch from Batuta
        if let Some(_url) = &self.base_url {
            // In a real implementation, this would make an HTTP request
            // For now, we simulate by returning fallback
            // TODO: Implement actual HTTP client when Batuta API is finalized
        }

        // Use fallback pricing
        self.fallback
            .get_rate(gpu_type)
            .cloned()
            .ok_or_else(|| BatutaError::UnknownGpuType(gpu_type.to_string()))
    }

    /// Get current queue depth.
    ///
    /// Returns queue state from Batuta if available, otherwise returns
    /// an optimistic default (no queue).
    pub fn get_queue_depth(&self, gpu_type: &str) -> Result<QueueState, BatutaError> {
        // Validate GPU type exists
        if self.fallback.get_rate(gpu_type).is_none() {
            return Err(BatutaError::UnknownGpuType(gpu_type.to_string()));
        }

        // If we have a live connection, try to fetch from Batuta
        if let Some(_url) = &self.base_url {
            // In a real implementation, this would make an HTTP request
            // TODO: Implement actual HTTP client when Batuta API is finalized
        }

        // Return optimistic default (no queue)
        Ok(QueueState::new(0, 4, 4))
    }

    /// Get both pricing and queue state in one call.
    pub fn get_status(&self, gpu_type: &str) -> Result<(GpuPricing, QueueState), BatutaError> {
        let pricing = self.get_hourly_rate(gpu_type)?;
        let queue = self.get_queue_depth(gpu_type)?;
        Ok((pricing, queue))
    }

    /// Estimate total cost for a job.
    pub fn estimate_cost(&self, gpu_type: &str, hours: f64) -> Result<f64, BatutaError> {
        let pricing = self.get_hourly_rate(gpu_type)?;
        Ok(pricing.hourly_rate * hours)
    }

    /// Get the cheapest GPU that meets memory requirements.
    pub fn cheapest_gpu(&self, min_memory_gb: u32) -> Option<&GpuPricing> {
        self.fallback
            .all_pricing()
            .iter()
            .filter(|p| p.memory_gb >= min_memory_gb)
            .min_by(|a, b| {
                a.hourly_rate
                    .partial_cmp(&b.hourly_rate)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

/// Adjust estimated completion time based on queue state.
///
/// Takes a base ETA and adjusts it based on:
/// - Current queue depth
/// - Average wait time
/// - Queue utilization
pub fn adjust_eta(base_eta_seconds: u64, queue_state: &QueueState) -> Duration {
    let mut adjusted = base_eta_seconds;

    // Add queue wait time if not immediately available
    if !queue_state.is_available() {
        // Estimate wait based on queue depth and average wait time
        let queue_wait = if queue_state.avg_wait_seconds > 0 {
            u64::from(queue_state.queue_depth) * queue_state.avg_wait_seconds
        } else {
            // Default: 5 minutes per queued job
            u64::from(queue_state.queue_depth) * 300
        };
        adjusted += queue_wait;
    }

    // Add ETA from queue state if available
    if let Some(eta) = queue_state.eta_seconds {
        adjusted = adjusted.max(eta);
    }

    // Apply utilization multiplier (high utilization = longer times)
    let utilization = queue_state.utilization();
    if utilization > 0.8 {
        let multiplier = 1.0 + (utilization - 0.8) * 2.0; // Up to 40% increase at 100% util
        adjusted = (adjusted as f64 * multiplier) as u64;
    }

    Duration::from_secs(adjusted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_pricing_creation() {
        let pricing = GpuPricing::new("a100-80gb", 3.00, 80)
            .with_spot(true)
            .with_provider("aws")
            .with_region("us-west-2");

        assert_eq!(pricing.gpu_type, "a100-80gb");
        assert!((pricing.hourly_rate - 3.00).abs() < f64::EPSILON);
        assert_eq!(pricing.memory_gb, 80);
        assert!(pricing.is_spot);
        assert_eq!(pricing.provider, "aws");
        assert_eq!(pricing.region, "us-west-2");
    }

    #[test]
    fn test_queue_state_creation() {
        let queue = QueueState::new(5, 2, 8).with_avg_wait(300).with_eta(600);

        assert_eq!(queue.queue_depth, 5);
        assert_eq!(queue.available_gpus, 2);
        assert_eq!(queue.total_gpus, 8);
        assert_eq!(queue.avg_wait_seconds, 300);
        assert_eq!(queue.eta_seconds, Some(600));
    }

    #[test]
    fn test_queue_state_is_available() {
        let available = QueueState::new(0, 2, 8);
        assert!(available.is_available());

        let unavailable = QueueState::new(5, 0, 8);
        assert!(!unavailable.is_available());
    }

    #[test]
    fn test_queue_state_utilization() {
        let empty = QueueState::new(0, 8, 8);
        assert!((empty.utilization() - 0.0).abs() < f64::EPSILON);

        let half = QueueState::new(0, 4, 8);
        assert!((half.utilization() - 0.5).abs() < f64::EPSILON);

        let full = QueueState::new(10, 0, 8);
        assert!((full.utilization() - 1.0).abs() < f64::EPSILON);

        let zero_total = QueueState::new(0, 0, 0);
        assert!((zero_total.utilization() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fallback_pricing_default() {
        let fallback = FallbackPricing::new();

        assert!(fallback.get_rate("a100-80gb").is_some());
        assert!(fallback.get_rate("v100").is_some());
        assert!(fallback.get_rate("t4").is_some());
        assert!(fallback.get_rate("unknown-gpu").is_none());
    }

    #[test]
    fn test_fallback_pricing_case_insensitive() {
        let fallback = FallbackPricing::new();

        // Should match regardless of case and separators
        assert!(fallback.get_rate("A100-80GB").is_some());
        assert!(fallback.get_rate("a100_80gb").is_some());
        assert!(fallback.get_rate("V100").is_some());
    }

    #[test]
    fn test_fallback_pricing_set_rate() {
        let mut fallback = FallbackPricing::new();

        // Update existing
        let old_rate = fallback.get_rate("t4").unwrap().hourly_rate;
        fallback.set_rate(GpuPricing::new("t4", 0.35, 16));
        let new_rate = fallback.get_rate("t4").unwrap().hourly_rate;
        assert!((old_rate - 0.50).abs() < f64::EPSILON);
        assert!((new_rate - 0.35).abs() < f64::EPSILON);

        // Add new
        fallback.set_rate(GpuPricing::new("rtx-4090", 0.80, 24));
        assert!(fallback.get_rate("rtx-4090").is_some());
    }

    #[test]
    fn test_batuta_client_new() {
        let client = BatutaClient::new();
        assert!(!client.is_connected());

        let pricing = client.get_hourly_rate("a100-80gb").unwrap();
        assert!((pricing.hourly_rate - 3.00).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batuta_client_with_url() {
        let client = BatutaClient::with_url("http://batuta.local:8080");
        assert!(client.is_connected());
    }

    #[test]
    fn test_batuta_client_unknown_gpu() {
        let client = BatutaClient::new();
        let result = client.get_hourly_rate("nonexistent-gpu");
        assert!(matches!(result, Err(BatutaError::UnknownGpuType(_))));
    }

    #[test]
    fn test_batuta_client_get_queue_depth() {
        let client = BatutaClient::new();
        let queue = client.get_queue_depth("a100-80gb").unwrap();

        // Default optimistic queue state
        assert_eq!(queue.queue_depth, 0);
        assert!(queue.is_available());
    }

    #[test]
    fn test_batuta_client_get_status() {
        let client = BatutaClient::new();
        let (pricing, queue) = client.get_status("v100").unwrap();

        assert_eq!(pricing.gpu_type, "v100");
        assert!((pricing.hourly_rate - 2.00).abs() < f64::EPSILON);
        assert!(queue.is_available());
    }

    #[test]
    fn test_batuta_client_estimate_cost() {
        let client = BatutaClient::new();
        let cost = client.estimate_cost("a100-80gb", 10.0).unwrap();
        assert!((cost - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batuta_client_cheapest_gpu() {
        let client = BatutaClient::new();

        // Should return T4 (cheapest with 16GB)
        let cheapest_16 = client.cheapest_gpu(16).unwrap();
        assert_eq!(cheapest_16.gpu_type, "t4");

        // Should return something with 80GB
        let cheapest_80 = client.cheapest_gpu(80).unwrap();
        assert!(cheapest_80.memory_gb >= 80);

        // Should return None for impossible requirements
        let impossible = client.cheapest_gpu(1000);
        assert!(impossible.is_none());
    }

    #[test]
    fn test_adjust_eta_available() {
        let queue = QueueState::new(0, 4, 8);
        let adjusted = adjust_eta(3600, &queue);
        assert_eq!(adjusted.as_secs(), 3600); // No adjustment
    }

    #[test]
    fn test_adjust_eta_queue_wait() {
        let queue = QueueState::new(3, 0, 8).with_avg_wait(300);
        let adjusted = adjust_eta(3600, &queue);

        // 3600 + (3 jobs * 300s avg wait) = 4500s
        assert!(adjusted.as_secs() >= 4500);
    }

    #[test]
    fn test_adjust_eta_high_utilization() {
        let queue = QueueState::new(0, 1, 8); // 87.5% utilization
        let base = 3600;
        let adjusted = adjust_eta(base, &queue);

        // Should be increased due to high utilization
        assert!(adjusted.as_secs() > base);
    }

    #[test]
    fn test_adjust_eta_with_queue_eta() {
        let queue = QueueState::new(0, 4, 8).with_eta(7200);
        let adjusted = adjust_eta(3600, &queue);

        // Should use queue ETA since it's higher
        assert!(adjusted.as_secs() >= 7200);
    }
}
