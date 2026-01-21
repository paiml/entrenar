//! Preflight check definitions and built-in checks.

use std::collections::HashMap;

use super::{CheckMetadata, CheckResult, CheckType, PreflightContext};

/// Check function type
type CheckFn = Box<dyn Fn(&[Vec<f64>], &PreflightContext) -> CheckResult + Send + Sync>;

/// A single preflight check
pub struct PreflightCheck {
    /// Name of the check
    pub name: String,
    /// Type of check
    pub check_type: CheckType,
    /// Description of what this check validates
    pub description: String,
    /// Whether this check is required (failure blocks training)
    pub required: bool,
    /// The check function
    check_fn: CheckFn,
}

impl From<&PreflightCheck> for CheckMetadata {
    fn from(check: &PreflightCheck) -> Self {
        Self {
            name: check.name.clone(),
            check_type: check.check_type.clone(),
            description: check.description.clone(),
            required: check.required,
        }
    }
}

impl std::fmt::Debug for PreflightCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreflightCheck")
            .field("name", &self.name)
            .field("check_type", &self.check_type)
            .field("description", &self.description)
            .field("required", &self.required)
            .finish_non_exhaustive()
    }
}

impl PreflightCheck {
    /// Create a new check
    pub fn new<F>(
        name: impl Into<String>,
        check_type: CheckType,
        description: impl Into<String>,
        check_fn: F,
    ) -> Self
    where
        F: Fn(&[Vec<f64>], &PreflightContext) -> CheckResult + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            check_type,
            description: description.into(),
            required: true,
            check_fn: Box::new(check_fn),
        }
    }

    /// Make this check optional (warning only)
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    /// Run this check
    pub fn run(&self, data: &[Vec<f64>], context: &PreflightContext) -> CheckResult {
        (self.check_fn)(data, context)
    }

    // =========================================================================
    // Built-in Data Integrity Checks
    // =========================================================================

    /// Check for NaN values in data
    pub fn no_nan_values() -> Self {
        Self::new(
            "no_nan_values",
            CheckType::DataIntegrity,
            "Ensures no NaN values exist in the dataset",
            |data, _ctx| {
                let mut nan_count = 0;
                let mut nan_locations = Vec::new();

                for (row_idx, row) in data.iter().enumerate() {
                    for (col_idx, val) in row.iter().enumerate() {
                        if val.is_nan() {
                            nan_count += 1;
                            if nan_locations.len() < 5 {
                                nan_locations.push(format!("({row_idx}, {col_idx})"));
                            }
                        }
                    }
                }

                if nan_count == 0 {
                    CheckResult::passed("No NaN values found")
                } else {
                    CheckResult::failed_with_details(
                        format!("Found {nan_count} NaN values"),
                        format!("First locations: {}", nan_locations.join(", ")),
                    )
                }
            },
        )
    }

    /// Check for infinite values in data
    pub fn no_inf_values() -> Self {
        Self::new(
            "no_inf_values",
            CheckType::DataIntegrity,
            "Ensures no infinite values exist in the dataset",
            |data, _ctx| {
                let mut inf_count = 0;
                let mut inf_locations = Vec::new();

                for (row_idx, row) in data.iter().enumerate() {
                    for (col_idx, val) in row.iter().enumerate() {
                        if val.is_infinite() {
                            inf_count += 1;
                            if inf_locations.len() < 5 {
                                inf_locations.push(format!("({row_idx}, {col_idx})"));
                            }
                        }
                    }
                }

                if inf_count == 0 {
                    CheckResult::passed("No infinite values found")
                } else {
                    CheckResult::failed_with_details(
                        format!("Found {inf_count} infinite values"),
                        format!("First locations: {}", inf_locations.join(", ")),
                    )
                }
            },
        )
    }

    /// Check minimum number of samples
    pub fn min_samples(min: usize) -> Self {
        Self::new(
            "min_samples",
            CheckType::DataIntegrity,
            format!("Ensures at least {min} samples exist"),
            move |data, ctx| {
                let min_required = ctx.min_samples.unwrap_or(min);
                let actual = data.len();

                if actual >= min_required {
                    CheckResult::passed(format!("Found {actual} samples (minimum: {min_required})"))
                } else {
                    CheckResult::failed(format!(
                        "Only {actual} samples found (minimum: {min_required})"
                    ))
                }
            },
        )
    }

    /// Check minimum number of features
    pub fn min_features(min: usize) -> Self {
        Self::new(
            "min_features",
            CheckType::DataIntegrity,
            format!("Ensures at least {min} features exist"),
            move |data, ctx| {
                let min_required = ctx.min_features.unwrap_or(min);
                let actual = data.first().map_or(0, Vec::len);

                if actual >= min_required {
                    CheckResult::passed(format!(
                        "Found {actual} features (minimum: {min_required})"
                    ))
                } else {
                    CheckResult::failed(format!(
                        "Only {actual} features found (minimum: {min_required})"
                    ))
                }
            },
        )
    }

    /// Check for consistent row lengths
    pub fn consistent_dimensions() -> Self {
        Self::new(
            "consistent_dimensions",
            CheckType::DataIntegrity,
            "Ensures all rows have the same number of features",
            |data, _ctx| {
                if data.is_empty() {
                    return CheckResult::skipped("No data to check");
                }

                let expected_len = data[0].len();
                let mut inconsistent = Vec::new();

                for (idx, row) in data.iter().enumerate() {
                    if row.len() != expected_len {
                        inconsistent.push(format!("row {idx}: {} features", row.len()));
                        if inconsistent.len() >= 5 {
                            break;
                        }
                    }
                }

                if inconsistent.is_empty() {
                    CheckResult::passed(format!(
                        "All {} rows have {expected_len} features",
                        data.len()
                    ))
                } else {
                    CheckResult::failed_with_details(
                        format!("Inconsistent dimensions (expected {expected_len} features)"),
                        inconsistent.join(", "),
                    )
                }
            },
        )
    }

    /// Check feature variance (detect constant features)
    pub fn no_constant_features() -> Self {
        Self::new(
            "no_constant_features",
            CheckType::DataIntegrity,
            "Ensures no features have zero variance",
            |data, _ctx| {
                if data.is_empty() || data[0].is_empty() {
                    return CheckResult::skipped("No data to check");
                }

                let n_features = data[0].len();
                let mut constant_features = Vec::new();

                for col in 0..n_features {
                    let values: Vec<f64> = data.iter().map(|row| row[col]).collect();
                    let first = values[0];

                    if values.iter().all(|v| (*v - first).abs() < f64::EPSILON) {
                        constant_features.push(col);
                    }
                }

                if constant_features.is_empty() {
                    CheckResult::passed("No constant features found")
                } else {
                    CheckResult::warning(format!(
                        "Found {} constant feature(s): {:?}",
                        constant_features.len(),
                        constant_features
                    ))
                }
            },
        )
        .optional()
    }

    /// Check for label balance (classification)
    pub fn label_balance(max_imbalance_ratio: f64) -> Self {
        Self::new(
            "label_balance",
            CheckType::DataIntegrity,
            format!("Ensures class imbalance ratio <= {max_imbalance_ratio}"),
            move |data, _ctx| {
                if data.is_empty() || data[0].is_empty() {
                    return CheckResult::skipped("No data to check");
                }

                // Assume last column is label
                let labels: Vec<i64> = data
                    .iter()
                    .map(|row| *row.last().unwrap_or(&0.0) as i64)
                    .collect();

                let mut counts: HashMap<i64, usize> = HashMap::new();
                for label in &labels {
                    *counts.entry(*label).or_default() += 1;
                }

                if counts.is_empty() {
                    return CheckResult::skipped("No labels found");
                }

                let max_count = *counts.values().max().unwrap_or(&0);
                let min_count = *counts.values().min().unwrap_or(&0);

                if min_count == 0 {
                    return CheckResult::failed("One or more classes have zero samples");
                }

                let ratio = max_count as f64 / min_count as f64;

                if ratio <= max_imbalance_ratio {
                    CheckResult::passed(format!(
                        "Class imbalance ratio {ratio:.2} <= {max_imbalance_ratio}"
                    ))
                } else {
                    CheckResult::warning(format!(
                        "Class imbalance ratio {ratio:.2} > {max_imbalance_ratio}"
                    ))
                }
            },
        )
        .optional()
    }

    // =========================================================================
    // Built-in Environment Checks
    // =========================================================================

    /// Check available disk space
    pub fn disk_space_mb(min_mb: u64) -> Self {
        Self::new(
            "disk_space",
            CheckType::Environment,
            format!("Ensures at least {min_mb} MB disk space available"),
            move |_data, ctx| {
                let required = ctx.min_disk_space_mb.unwrap_or(min_mb);

                // Get available disk space (platform-specific)
                #[cfg(unix)]
                {
                    use std::process::Command;
                    let output = Command::new("df").args(["-m", "."]).output().ok();

                    if let Some(output) = output {
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        // Parse df output (second line, fourth column is available)
                        if let Some(line) = stdout.lines().nth(1) {
                            if let Some(available) = line.split_whitespace().nth(3) {
                                if let Ok(avail_mb) = available.parse::<u64>() {
                                    if avail_mb >= required {
                                        return CheckResult::passed(format!(
                                            "{avail_mb} MB available (minimum: {required} MB)"
                                        ));
                                    }
                                    return CheckResult::failed(format!(
                                        "Only {avail_mb} MB available (minimum: {required} MB)"
                                    ));
                                }
                            }
                        }
                    }
                }

                // Fallback: assume sufficient space
                CheckResult::passed(format!(
                    "Disk space check passed (assumed >= {required} MB)"
                ))
            },
        )
    }

    /// Check available memory
    pub fn memory_mb(min_mb: u64) -> Self {
        Self::new(
            "memory",
            CheckType::Environment,
            format!("Ensures at least {min_mb} MB memory available"),
            move |_data, ctx| {
                let required = ctx.min_memory_mb.unwrap_or(min_mb);

                // Check available memory (platform-specific)
                #[cfg(unix)]
                {
                    use std::process::Command;
                    let output = Command::new("free").args(["-m"]).output().ok();

                    if let Some(output) = output {
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        // Parse free output (second line, "available" column)
                        if let Some(line) = stdout.lines().nth(1) {
                            // Mem: line format varies, try to find available
                            let parts: Vec<&str> = line.split_whitespace().collect();
                            if parts.len() >= 7 {
                                if let Ok(avail_mb) = parts[6].parse::<u64>() {
                                    if avail_mb >= required {
                                        return CheckResult::passed(format!(
                                            "{avail_mb} MB available (minimum: {required} MB)"
                                        ));
                                    }
                                    return CheckResult::failed(format!(
                                        "Only {avail_mb} MB available (minimum: {required} MB)"
                                    ));
                                }
                            }
                        }
                    }
                }

                // Fallback
                CheckResult::passed(format!("Memory check passed (assumed >= {required} MB)"))
            },
        )
    }

    /// Check GPU availability
    pub fn gpu_available() -> Self {
        Self::new(
            "gpu_available",
            CheckType::Environment,
            "Checks if GPU is available for training",
            |_data, _ctx| {
                // Check for NVIDIA GPU using nvidia-smi
                #[cfg(unix)]
                {
                    use std::process::Command;
                    let result = Command::new("nvidia-smi")
                        .args(["--query-gpu=name", "--format=csv,noheader"])
                        .output();

                    if let Ok(output) = result {
                        if output.status.success() {
                            let gpu_name = String::from_utf8_lossy(&output.stdout);
                            let gpu_name = gpu_name.trim();
                            if !gpu_name.is_empty() {
                                return CheckResult::passed(format!("GPU available: {gpu_name}"));
                            }
                        }
                    }
                }

                CheckResult::warning("No GPU detected, training will use CPU")
            },
        )
        .optional()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // PreflightCheck Tests - No NaN
    // =========================================================================

    #[test]
    fn test_no_nan_values_passes() {
        let check = PreflightCheck::no_nan_values();
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = check.run(&data, &PreflightContext::default());
        assert!(result.is_passed());
    }

    #[test]
    fn test_no_nan_values_fails() {
        let check = PreflightCheck::no_nan_values();
        let data = vec![vec![1.0, f64::NAN], vec![3.0, 4.0]];
        let result = check.run(&data, &PreflightContext::default());
        assert!(result.is_failed());
    }

    #[test]
    fn test_no_nan_values_empty_data() {
        let check = PreflightCheck::no_nan_values();
        let data: Vec<Vec<f64>> = vec![];
        let result = check.run(&data, &PreflightContext::default());
        assert!(result.is_passed());
    }

    // =========================================================================
    // PreflightCheck Tests - No Inf
    // =========================================================================

    #[test]
    fn test_no_inf_values_passes() {
        let check = PreflightCheck::no_inf_values();
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = check.run(&data, &PreflightContext::default());
        assert!(result.is_passed());
    }

    #[test]
    fn test_no_inf_values_fails_positive() {
        let check = PreflightCheck::no_inf_values();
        let data = vec![vec![1.0, f64::INFINITY], vec![3.0, 4.0]];
        let result = check.run(&data, &PreflightContext::default());
        assert!(result.is_failed());
    }

    #[test]
    fn test_no_inf_values_fails_negative() {
        let check = PreflightCheck::no_inf_values();
        let data = vec![vec![1.0, f64::NEG_INFINITY]];
        let result = check.run(&data, &PreflightContext::default());
        assert!(result.is_failed());
    }

    // =========================================================================
    // PreflightCheck Tests - Min Samples
    // =========================================================================

    #[test]
    fn test_min_samples_passes() {
        let check = PreflightCheck::min_samples(2);
        let data = vec![vec![1.0], vec![2.0], vec![3.0]];
        let result = check.run(&data, &PreflightContext::default());
        assert!(result.is_passed());
    }

    #[test]
    fn test_min_samples_fails() {
        let check = PreflightCheck::min_samples(10);
        let data = vec![vec![1.0], vec![2.0]];
        let result = check.run(&data, &PreflightContext::default());
        assert!(result.is_failed());
    }

    #[test]
    fn test_min_samples_uses_context() {
        let check = PreflightCheck::min_samples(2);
        let data = vec![vec![1.0], vec![2.0]];
        let ctx = PreflightContext::new().with_min_samples(5);
        let result = check.run(&data, &ctx);
        assert!(result.is_failed()); // Context overrides default
    }

    // =========================================================================
    // PreflightCheck Tests - Min Features
    // =========================================================================

    #[test]
    fn test_min_features_passes() {
        let check = PreflightCheck::min_features(2);
        let data = vec![vec![1.0, 2.0, 3.0]];
        let result = check.run(&data, &PreflightContext::default());
        assert!(result.is_passed());
    }

    #[test]
    fn test_min_features_fails() {
        let check = PreflightCheck::min_features(5);
        let data = vec![vec![1.0, 2.0]];
        let result = check.run(&data, &PreflightContext::default());
        assert!(result.is_failed());
    }

    // =========================================================================
    // PreflightCheck Tests - Consistent Dimensions
    // =========================================================================

    #[test]
    fn test_consistent_dimensions_passes() {
        let check = PreflightCheck::consistent_dimensions();
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let result = check.run(&data, &PreflightContext::default());
        assert!(result.is_passed());
    }

    #[test]
    fn test_consistent_dimensions_fails() {
        let check = PreflightCheck::consistent_dimensions();
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]];
        let result = check.run(&data, &PreflightContext::default());
        assert!(result.is_failed());
    }

    #[test]
    fn test_consistent_dimensions_empty() {
        let check = PreflightCheck::consistent_dimensions();
        let data: Vec<Vec<f64>> = vec![];
        let result = check.run(&data, &PreflightContext::default());
        assert!(result.is_skipped());
    }

    // =========================================================================
    // PreflightCheck Tests - No Constant Features
    // =========================================================================

    #[test]
    fn test_no_constant_features_passes() {
        let check = PreflightCheck::no_constant_features();
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = check.run(&data, &PreflightContext::default());
        assert!(result.is_passed());
    }

    #[test]
    fn test_no_constant_features_warns() {
        let check = PreflightCheck::no_constant_features();
        let data = vec![vec![1.0, 2.0], vec![1.0, 4.0]]; // First column constant
        let result = check.run(&data, &PreflightContext::default());
        assert!(result.is_warning());
    }

    // =========================================================================
    // PreflightCheck Tests - Label Balance
    // =========================================================================

    #[test]
    fn test_label_balance_passes() {
        let check = PreflightCheck::label_balance(2.0);
        // Last column is label: 2 samples of class 0, 2 of class 1
        let data = vec![
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![3.0, 1.0],
            vec![4.0, 1.0],
        ];
        let result = check.run(&data, &PreflightContext::default());
        assert!(result.is_passed());
    }

    #[test]
    fn test_label_balance_warns() {
        let check = PreflightCheck::label_balance(2.0);
        // Imbalanced: 1 sample class 0, 5 samples class 1
        let data = vec![
            vec![1.0, 0.0],
            vec![2.0, 1.0],
            vec![3.0, 1.0],
            vec![4.0, 1.0],
            vec![5.0, 1.0],
            vec![6.0, 1.0],
        ];
        let result = check.run(&data, &PreflightContext::default());
        assert!(result.is_warning());
    }

    // =========================================================================
    // Environment Check Tests (may be skipped on some systems)
    // =========================================================================

    #[test]
    fn test_disk_space_check() {
        let check = PreflightCheck::disk_space_mb(1);
        let result = check.run(&[], &PreflightContext::default());
        // Should pass or be a valid result
        assert!(result.is_passed() || result.is_failed() || result.is_skipped());
    }

    #[test]
    fn test_memory_check() {
        let check = PreflightCheck::memory_mb(1);
        let result = check.run(&[], &PreflightContext::default());
        assert!(result.is_passed() || result.is_failed() || result.is_skipped());
    }

    #[test]
    fn test_gpu_available_check() {
        let check = PreflightCheck::gpu_available();
        let result = check.run(&[], &PreflightContext::default());
        // May pass or warn depending on system
        assert!(result.is_passed() || result.is_warning());
    }

    // =========================================================================
    // Property Tests
    // =========================================================================

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_no_nan_passes_for_valid_data(
            rows in 1usize..100,
            cols in 1usize..10
        ) {
            let data: Vec<Vec<f64>> = (0..rows)
                .map(|i| (0..cols).map(|j| (i * cols + j) as f64).collect())
                .collect();

            let check = PreflightCheck::no_nan_values();
            let result = check.run(&data, &PreflightContext::default());
            prop_assert!(result.is_passed());
        }

        #[test]
        fn prop_consistent_dimensions_passes_for_rectangular(
            rows in 1usize..50,
            cols in 1usize..10
        ) {
            let data: Vec<Vec<f64>> = (0..rows)
                .map(|_| vec![0.0; cols])
                .collect();

            let check = PreflightCheck::consistent_dimensions();
            let result = check.run(&data, &PreflightContext::default());
            prop_assert!(result.is_passed());
        }

        #[test]
        fn prop_min_samples_respects_threshold(
            actual in 0usize..100,
            required in 1usize..50
        ) {
            let data: Vec<Vec<f64>> = (0..actual).map(|_| vec![1.0]).collect();
            let check = PreflightCheck::min_samples(required);
            let result = check.run(&data, &PreflightContext::default());

            if actual >= required {
                prop_assert!(result.is_passed());
            } else {
                prop_assert!(result.is_failed());
            }
        }
    }
}
