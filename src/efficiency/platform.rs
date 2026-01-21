//! Platform Efficiency (ENT-012)
//!
//! Provides platform-specific efficiency metrics for server and edge deployments.

use serde::{Deserialize, Serialize};

/// Server deployment efficiency metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerEfficiency {
    /// Training/inference throughput in samples per second
    pub throughput_samples_per_sec: f64,
    /// GPU utilization percentage (0-100)
    pub gpu_utilization_percent: f64,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f64,
}

impl ServerEfficiency {
    /// Create new server efficiency metrics
    pub fn new(
        throughput_samples_per_sec: f64,
        gpu_utilization_percent: f64,
        memory_bandwidth_gbps: f64,
    ) -> Self {
        Self {
            throughput_samples_per_sec,
            gpu_utilization_percent: gpu_utilization_percent.clamp(0.0, 100.0),
            memory_bandwidth_gbps,
        }
    }

    /// Check if GPU is being efficiently utilized (>70%)
    pub fn is_gpu_efficient(&self) -> bool {
        self.gpu_utilization_percent >= 70.0
    }

    /// Check if GPU is underutilized (<50%)
    pub fn is_gpu_underutilized(&self) -> bool {
        self.gpu_utilization_percent < 50.0
    }

    /// Calculate throughput efficiency (samples per second per % GPU util)
    pub fn throughput_efficiency(&self) -> f64 {
        if self.gpu_utilization_percent > 0.0 {
            self.throughput_samples_per_sec / self.gpu_utilization_percent
        } else {
            0.0
        }
    }

    /// Estimate maximum throughput at 100% utilization
    pub fn estimated_max_throughput(&self) -> f64 {
        if self.gpu_utilization_percent > 0.0 {
            self.throughput_samples_per_sec * (100.0 / self.gpu_utilization_percent)
        } else {
            0.0
        }
    }

    /// Check if memory bandwidth might be a bottleneck
    pub fn memory_bound(&self, expected_bandwidth_gbps: f64) -> bool {
        self.memory_bandwidth_gbps > expected_bandwidth_gbps * 0.9
    }
}

impl Default for ServerEfficiency {
    fn default() -> Self {
        Self {
            throughput_samples_per_sec: 0.0,
            gpu_utilization_percent: 0.0,
            memory_bandwidth_gbps: 0.0,
        }
    }
}

/// Edge deployment efficiency metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeEfficiency {
    /// Binary/model size in bytes
    pub binary_size_bytes: u64,
    /// Startup/initialization latency in milliseconds
    pub startup_latency_ms: u64,
    /// 99th percentile inference latency in milliseconds
    pub inference_latency_p99_ms: f64,
    /// Runtime memory footprint in bytes
    pub memory_footprint_bytes: u64,
}

impl EdgeEfficiency {
    /// Create new edge efficiency metrics
    pub fn new(
        binary_size_bytes: u64,
        startup_latency_ms: u64,
        inference_latency_p99_ms: f64,
        memory_footprint_bytes: u64,
    ) -> Self {
        Self {
            binary_size_bytes,
            startup_latency_ms,
            inference_latency_p99_ms,
            memory_footprint_bytes,
        }
    }

    /// Check if deployment meets WASM budget constraints
    pub fn meets_wasm_budget(&self, budget: &WasmBudget) -> bool {
        self.binary_size_bytes <= budget.max_binary_size
            && self.startup_latency_ms <= budget.max_startup_ms
            && self.memory_footprint_bytes <= budget.max_memory_bytes
    }

    /// Get binary size in MB
    pub fn binary_size_mb(&self) -> f64 {
        self.binary_size_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get memory footprint in MB
    pub fn memory_footprint_mb(&self) -> f64 {
        self.memory_footprint_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Calculate maximum throughput based on latency
    pub fn max_throughput_per_sec(&self) -> f64 {
        if self.inference_latency_p99_ms > 0.0 {
            1000.0 / self.inference_latency_p99_ms
        } else {
            0.0
        }
    }

    /// Check which budget constraints are violated
    pub fn budget_violations(&self, budget: &WasmBudget) -> Vec<BudgetViolation> {
        let mut violations = Vec::new();

        if self.binary_size_bytes > budget.max_binary_size {
            violations.push(BudgetViolation::BinarySize {
                actual: self.binary_size_bytes,
                limit: budget.max_binary_size,
            });
        }

        if self.startup_latency_ms > budget.max_startup_ms {
            violations.push(BudgetViolation::StartupLatency {
                actual: self.startup_latency_ms,
                limit: budget.max_startup_ms,
            });
        }

        if self.memory_footprint_bytes > budget.max_memory_bytes {
            violations.push(BudgetViolation::MemoryFootprint {
                actual: self.memory_footprint_bytes,
                limit: budget.max_memory_bytes,
            });
        }

        violations
    }
}

impl Default for EdgeEfficiency {
    fn default() -> Self {
        Self {
            binary_size_bytes: 0,
            startup_latency_ms: 0,
            inference_latency_p99_ms: 0.0,
            memory_footprint_bytes: 0,
        }
    }
}

/// WASM deployment budget constraints
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WasmBudget {
    /// Maximum binary size in bytes (default: 5 MB)
    pub max_binary_size: u64,
    /// Maximum startup latency in milliseconds (default: 500 ms)
    pub max_startup_ms: u64,
    /// Maximum memory usage in bytes (default: 256 MB)
    pub max_memory_bytes: u64,
}

impl WasmBudget {
    /// Create new WASM budget
    pub fn new(max_binary_size: u64, max_startup_ms: u64, max_memory_bytes: u64) -> Self {
        Self {
            max_binary_size,
            max_startup_ms,
            max_memory_bytes,
        }
    }

    /// Create a strict budget for mobile web
    pub fn mobile() -> Self {
        Self {
            max_binary_size: 2 * 1024 * 1024,    // 2 MB
            max_startup_ms: 200,                 // 200 ms
            max_memory_bytes: 128 * 1024 * 1024, // 128 MB
        }
    }

    /// Create a standard budget for desktop web
    pub fn desktop() -> Self {
        Self {
            max_binary_size: 10 * 1024 * 1024,   // 10 MB
            max_startup_ms: 1000,                // 1 second
            max_memory_bytes: 512 * 1024 * 1024, // 512 MB
        }
    }

    /// Create a relaxed budget for embedded/IoT
    pub fn embedded() -> Self {
        Self {
            max_binary_size: 1024 * 1024,       // 1 MB
            max_startup_ms: 100,                // 100 ms
            max_memory_bytes: 64 * 1024 * 1024, // 64 MB
        }
    }

    /// Get budget sizes in MB
    pub fn sizes_mb(&self) -> (f64, f64) {
        (
            self.max_binary_size as f64 / (1024.0 * 1024.0),
            self.max_memory_bytes as f64 / (1024.0 * 1024.0),
        )
    }
}

impl Default for WasmBudget {
    fn default() -> Self {
        Self {
            max_binary_size: 5 * 1024 * 1024,    // 5 MB
            max_startup_ms: 500,                 // 500 ms
            max_memory_bytes: 256 * 1024 * 1024, // 256 MB
        }
    }
}

/// Budget violation details
#[derive(Debug, Clone, PartialEq)]
pub enum BudgetViolation {
    /// Binary size exceeds limit
    BinarySize { actual: u64, limit: u64 },
    /// Startup latency exceeds limit
    StartupLatency { actual: u64, limit: u64 },
    /// Memory footprint exceeds limit
    MemoryFootprint { actual: u64, limit: u64 },
}

impl std::fmt::Display for BudgetViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BinarySize { actual, limit } => {
                write!(
                    f,
                    "Binary size {} MB exceeds {} MB limit",
                    *actual as f64 / (1024.0 * 1024.0),
                    *limit as f64 / (1024.0 * 1024.0)
                )
            }
            Self::StartupLatency { actual, limit } => {
                write!(f, "Startup latency {actual} ms exceeds {limit} ms limit")
            }
            Self::MemoryFootprint { actual, limit } => {
                write!(
                    f,
                    "Memory footprint {} MB exceeds {} MB limit",
                    *actual as f64 / (1024.0 * 1024.0),
                    *limit as f64 / (1024.0 * 1024.0)
                )
            }
        }
    }
}

/// Platform efficiency abstraction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PlatformEfficiency {
    /// Server/cloud deployment
    Server(ServerEfficiency),
    /// Edge/embedded deployment
    Edge(EdgeEfficiency),
}

impl PlatformEfficiency {
    /// Check if this is a server deployment
    pub fn is_server(&self) -> bool {
        matches!(self, Self::Server(_))
    }

    /// Check if this is an edge deployment
    pub fn is_edge(&self) -> bool {
        matches!(self, Self::Edge(_))
    }

    /// Get throughput in samples per second
    pub fn throughput(&self) -> f64 {
        match self {
            Self::Server(s) => s.throughput_samples_per_sec,
            Self::Edge(e) => e.max_throughput_per_sec(),
        }
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> u64 {
        match self {
            Self::Server(_) => 0, // Server typically has abundant memory
            Self::Edge(e) => e.memory_footprint_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_efficiency_new() {
        let eff = ServerEfficiency::new(1000.0, 85.0, 500.0);

        assert!((eff.throughput_samples_per_sec - 1000.0).abs() < f64::EPSILON);
        assert!((eff.gpu_utilization_percent - 85.0).abs() < f64::EPSILON);
        assert!((eff.memory_bandwidth_gbps - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_server_efficiency_clamps_utilization() {
        let eff = ServerEfficiency::new(1000.0, 150.0, 500.0);
        assert!((eff.gpu_utilization_percent - 100.0).abs() < f64::EPSILON);

        let eff = ServerEfficiency::new(1000.0, -10.0, 500.0);
        assert!((eff.gpu_utilization_percent - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_server_efficiency_gpu_status() {
        let efficient = ServerEfficiency::new(1000.0, 80.0, 500.0);
        assert!(efficient.is_gpu_efficient());
        assert!(!efficient.is_gpu_underutilized());

        let underutilized = ServerEfficiency::new(1000.0, 40.0, 500.0);
        assert!(!underutilized.is_gpu_efficient());
        assert!(underutilized.is_gpu_underutilized());
    }

    #[test]
    fn test_server_efficiency_throughput_efficiency() {
        let eff = ServerEfficiency::new(800.0, 80.0, 500.0);
        assert!((eff.throughput_efficiency() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_server_efficiency_estimated_max() {
        let eff = ServerEfficiency::new(500.0, 50.0, 500.0);
        assert!((eff.estimated_max_throughput() - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_server_efficiency_memory_bound() {
        let eff = ServerEfficiency::new(1000.0, 80.0, 950.0);
        assert!(eff.memory_bound(1000.0)); // 950 > 1000 * 0.9 = 900

        let eff = ServerEfficiency::new(1000.0, 80.0, 500.0);
        assert!(!eff.memory_bound(1000.0)); // 500 < 900
    }

    #[test]
    fn test_edge_efficiency_new() {
        let eff = EdgeEfficiency::new(
            5 * 1024 * 1024,   // 5 MB
            100,               // 100 ms
            10.0,              // 10 ms p99
            128 * 1024 * 1024, // 128 MB
        );

        assert_eq!(eff.binary_size_bytes, 5 * 1024 * 1024);
        assert_eq!(eff.startup_latency_ms, 100);
        assert!((eff.inference_latency_p99_ms - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_edge_efficiency_size_conversions() {
        let eff = EdgeEfficiency::new(
            10 * 1024 * 1024, // 10 MB
            100,
            10.0,
            256 * 1024 * 1024, // 256 MB
        );

        assert!((eff.binary_size_mb() - 10.0).abs() < 0.01);
        assert!((eff.memory_footprint_mb() - 256.0).abs() < 0.01);
    }

    #[test]
    fn test_edge_efficiency_max_throughput() {
        let eff = EdgeEfficiency::new(0, 0, 10.0, 0); // 10 ms latency
        assert!((eff.max_throughput_per_sec() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_edge_efficiency_meets_budget() {
        let eff = EdgeEfficiency::new(
            4 * 1024 * 1024, // 4 MB
            400,             // 400 ms
            10.0,
            200 * 1024 * 1024, // 200 MB
        );

        let budget = WasmBudget::default(); // 5 MB, 500 ms, 256 MB
        assert!(eff.meets_wasm_budget(&budget));
    }

    #[test]
    fn test_edge_efficiency_violates_budget() {
        let eff = EdgeEfficiency::new(
            10 * 1024 * 1024, // 10 MB - exceeds default 5 MB
            400,
            10.0,
            200 * 1024 * 1024,
        );

        let budget = WasmBudget::default();
        assert!(!eff.meets_wasm_budget(&budget));

        let violations = eff.budget_violations(&budget);
        assert_eq!(violations.len(), 1);
        assert!(matches!(violations[0], BudgetViolation::BinarySize { .. }));
    }

    #[test]
    fn test_edge_efficiency_multiple_violations() {
        let eff = EdgeEfficiency::new(
            10 * 1024 * 1024, // 10 MB - exceeds 5 MB
            1000,             // 1000 ms - exceeds 500 ms
            10.0,
            300 * 1024 * 1024, // 300 MB - exceeds 256 MB
        );

        let budget = WasmBudget::default();
        let violations = eff.budget_violations(&budget);
        assert_eq!(violations.len(), 3);
    }

    #[test]
    fn test_wasm_budget_presets() {
        let mobile = WasmBudget::mobile();
        let desktop = WasmBudget::desktop();
        let embedded = WasmBudget::embedded();

        // Mobile is strictest
        assert!(mobile.max_binary_size < desktop.max_binary_size);
        assert!(mobile.max_memory_bytes < desktop.max_memory_bytes);

        // Embedded is most constrained
        assert!(embedded.max_binary_size < mobile.max_binary_size);
        assert!(embedded.max_memory_bytes < mobile.max_memory_bytes);
    }

    #[test]
    fn test_wasm_budget_default() {
        let budget = WasmBudget::default();
        assert_eq!(budget.max_binary_size, 5 * 1024 * 1024);
        assert_eq!(budget.max_startup_ms, 500);
        assert_eq!(budget.max_memory_bytes, 256 * 1024 * 1024);
    }

    #[test]
    fn test_wasm_budget_sizes_mb() {
        let budget = WasmBudget::default();
        let (binary_mb, memory_mb) = budget.sizes_mb();
        assert!((binary_mb - 5.0).abs() < 0.01);
        assert!((memory_mb - 256.0).abs() < 0.01);
    }

    #[test]
    fn test_budget_violation_display() {
        let violation = BudgetViolation::BinarySize {
            actual: 10 * 1024 * 1024,
            limit: 5 * 1024 * 1024,
        };
        let msg = format!("{violation}");
        assert!(msg.contains("10"));
        assert!(msg.contains('5'));
        assert!(msg.contains("MB"));
    }

    #[test]
    fn test_platform_efficiency_server() {
        let server = PlatformEfficiency::Server(ServerEfficiency::new(1000.0, 80.0, 500.0));

        assert!(server.is_server());
        assert!(!server.is_edge());
        assert!((server.throughput() - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_platform_efficiency_edge() {
        let edge = PlatformEfficiency::Edge(EdgeEfficiency::new(0, 0, 10.0, 128 * 1024 * 1024));

        assert!(!edge.is_server());
        assert!(edge.is_edge());
        assert!((edge.throughput() - 100.0).abs() < 0.01);
        assert_eq!(edge.memory_bytes(), 128 * 1024 * 1024);
    }

    #[test]
    fn test_server_efficiency_serialization() {
        let eff = ServerEfficiency::new(1000.0, 85.0, 500.0);
        let json = serde_json::to_string(&eff).unwrap();
        let parsed: ServerEfficiency = serde_json::from_str(&json).unwrap();

        assert!(
            (parsed.throughput_samples_per_sec - eff.throughput_samples_per_sec).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_edge_efficiency_serialization() {
        let eff = EdgeEfficiency::new(5 * 1024 * 1024, 100, 10.0, 128 * 1024 * 1024);
        let json = serde_json::to_string(&eff).unwrap();
        let parsed: EdgeEfficiency = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.binary_size_bytes, eff.binary_size_bytes);
    }

    #[test]
    fn test_server_efficiency_default() {
        let eff = ServerEfficiency::default();
        assert!((eff.throughput_samples_per_sec - 0.0).abs() < f64::EPSILON);
        assert!((eff.gpu_utilization_percent - 0.0).abs() < f64::EPSILON);
        assert!((eff.memory_bandwidth_gbps - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_server_efficiency_zero_utilization_throughput_efficiency() {
        let eff = ServerEfficiency::new(1000.0, 0.0, 500.0);
        assert!((eff.throughput_efficiency() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_server_efficiency_zero_utilization_estimated_max() {
        let eff = ServerEfficiency::new(1000.0, 0.0, 500.0);
        assert!((eff.estimated_max_throughput() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_edge_efficiency_default() {
        let eff = EdgeEfficiency::default();
        assert_eq!(eff.binary_size_bytes, 0);
        assert_eq!(eff.startup_latency_ms, 0);
        assert!((eff.inference_latency_p99_ms - 0.0).abs() < f64::EPSILON);
        assert_eq!(eff.memory_footprint_bytes, 0);
    }

    #[test]
    fn test_edge_efficiency_zero_latency_max_throughput() {
        let eff = EdgeEfficiency::new(0, 0, 0.0, 0);
        assert!((eff.max_throughput_per_sec() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_budget_violation_display_startup_latency() {
        let violation = BudgetViolation::StartupLatency {
            actual: 1000,
            limit: 500,
        };
        let msg = format!("{violation}");
        assert!(msg.contains("1000"));
        assert!(msg.contains("500"));
        assert!(msg.contains("ms"));
    }

    #[test]
    fn test_budget_violation_display_memory_footprint() {
        let violation = BudgetViolation::MemoryFootprint {
            actual: 512 * 1024 * 1024,
            limit: 256 * 1024 * 1024,
        };
        let msg = format!("{violation}");
        assert!(msg.contains("512"));
        assert!(msg.contains("256"));
        assert!(msg.contains("MB"));
    }

    #[test]
    fn test_wasm_budget_new() {
        let budget = WasmBudget::new(1024, 100, 4096);
        assert_eq!(budget.max_binary_size, 1024);
        assert_eq!(budget.max_startup_ms, 100);
        assert_eq!(budget.max_memory_bytes, 4096);
    }

    #[test]
    fn test_platform_efficiency_server_memory_bytes() {
        let server = PlatformEfficiency::Server(ServerEfficiency::new(1000.0, 80.0, 500.0));
        assert_eq!(server.memory_bytes(), 0);
    }

    #[test]
    fn test_platform_efficiency_serde() {
        let server = PlatformEfficiency::Server(ServerEfficiency::new(1000.0, 80.0, 500.0));
        let json = serde_json::to_string(&server).unwrap();
        let parsed: PlatformEfficiency = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_server());

        let edge = PlatformEfficiency::Edge(EdgeEfficiency::new(1024, 100, 10.0, 4096));
        let json = serde_json::to_string(&edge).unwrap();
        let parsed: PlatformEfficiency = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_edge());
    }

    #[test]
    fn test_wasm_budget_serde() {
        let budget = WasmBudget::mobile();
        let json = serde_json::to_string(&budget).unwrap();
        let parsed: WasmBudget = serde_json::from_str(&json).unwrap();
        assert_eq!(budget.max_binary_size, parsed.max_binary_size);
    }

    #[test]
    fn test_budget_violation_equality() {
        let v1 = BudgetViolation::BinarySize {
            actual: 10,
            limit: 5,
        };
        let v2 = BudgetViolation::BinarySize {
            actual: 10,
            limit: 5,
        };
        assert_eq!(v1, v2);

        let v3 = BudgetViolation::StartupLatency {
            actual: 10,
            limit: 5,
        };
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_server_efficiency_boundary_utilization() {
        // Test exactly at the 70% efficiency boundary
        let eff_70 = ServerEfficiency::new(1000.0, 70.0, 500.0);
        assert!(eff_70.is_gpu_efficient());
        assert!(!eff_70.is_gpu_underutilized());

        // Test exactly at the 50% underutilization boundary
        let eff_50 = ServerEfficiency::new(1000.0, 50.0, 500.0);
        assert!(!eff_50.is_gpu_efficient());
        assert!(!eff_50.is_gpu_underutilized());

        // Test just below 50%
        let eff_49 = ServerEfficiency::new(1000.0, 49.9, 500.0);
        assert!(eff_49.is_gpu_underutilized());
    }

    #[test]
    fn test_server_efficiency_clone() {
        let eff = ServerEfficiency::new(1000.0, 80.0, 500.0);
        let cloned = eff.clone();
        assert_eq!(eff, cloned);
    }

    #[test]
    fn test_edge_efficiency_clone() {
        let eff = EdgeEfficiency::new(1024, 100, 10.0, 4096);
        let cloned = eff.clone();
        assert_eq!(eff, cloned);
    }

    #[test]
    fn test_wasm_budget_clone() {
        let budget = WasmBudget::desktop();
        let cloned = budget.clone();
        assert_eq!(budget, cloned);
    }

    #[test]
    fn test_platform_efficiency_clone() {
        let server = PlatformEfficiency::Server(ServerEfficiency::new(1000.0, 80.0, 500.0));
        let cloned = server.clone();
        assert_eq!(server, cloned);
    }

    #[test]
    fn test_budget_violation_clone() {
        let violation = BudgetViolation::BinarySize {
            actual: 10,
            limit: 5,
        };
        let cloned = violation.clone();
        assert_eq!(violation, cloned);
    }
}
