//! ENT-034: Auto-feature type inference from data
//!
//! Automatically infers feature types from training data by analyzing column statistics.
//! Supports: numeric, categorical, text, datetime, embedding types.

use std::collections::HashMap;
use std::path::Path;

/// Inferred feature type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureType {
    /// Continuous numeric values (float32/float64)
    Numeric,
    /// Discrete categories with limited cardinality
    Categorical,
    /// Free-form text requiring tokenization
    Text,
    /// Timestamp/datetime values
    DateTime,
    /// Pre-computed embedding vectors
    Embedding,
    /// Binary classification target
    BinaryTarget,
    /// Multi-class classification target
    MultiClassTarget,
    /// Regression target
    RegressionTarget,
    /// Sequence of tokens (for language models)
    TokenSequence,
    /// Unknown/ambiguous type
    Unknown,
}

impl std::fmt::Display for FeatureType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Numeric => write!(f, "numeric"),
            Self::Categorical => write!(f, "categorical"),
            Self::Text => write!(f, "text"),
            Self::DateTime => write!(f, "datetime"),
            Self::Embedding => write!(f, "embedding"),
            Self::BinaryTarget => write!(f, "binary_target"),
            Self::MultiClassTarget => write!(f, "multiclass_target"),
            Self::RegressionTarget => write!(f, "regression_target"),
            Self::TokenSequence => write!(f, "token_sequence"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Statistics about a column used for type inference
#[derive(Debug, Clone, Default)]
pub struct ColumnStats {
    /// Column name
    pub name: String,
    /// Number of rows
    pub count: usize,
    /// Number of unique values
    pub unique_count: usize,
    /// Number of null/missing values
    pub null_count: usize,
    /// Whether all values are integers
    pub all_integers: bool,
    /// Whether all values are numeric
    pub all_numeric: bool,
    /// Minimum string length (if text)
    pub min_str_len: Option<usize>,
    /// Maximum string length (if text)
    pub max_str_len: Option<usize>,
    /// Average string length (if text)
    pub avg_str_len: Option<f32>,
    /// Whether values look like timestamps
    pub looks_like_datetime: bool,
    /// Whether values are arrays/lists
    pub is_array: bool,
    /// Array element count (if array)
    pub array_len: Option<usize>,
    /// Sample values for heuristic analysis
    pub sample_values: Vec<String>,
}

impl ColumnStats {
    /// Create stats for a column
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Cardinality ratio: unique_count / count
    pub fn cardinality_ratio(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.unique_count as f32 / self.count as f32
        }
    }

    /// Null ratio: null_count / count
    pub fn null_ratio(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.null_count as f32 / self.count as f32
        }
    }
}

/// Configuration for type inference
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Maximum cardinality ratio to consider categorical (default: 0.05)
    pub categorical_threshold: f32,
    /// Minimum average string length to consider text (default: 20)
    pub text_min_avg_len: f32,
    /// Column names that should be treated as targets
    pub target_columns: Vec<String>,
    /// Column names to exclude from inference
    pub exclude_columns: Vec<String>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            categorical_threshold: 0.05,
            text_min_avg_len: 20.0,
            target_columns: vec!["label".to_string(), "target".to_string(), "y".to_string()],
            exclude_columns: vec![],
        }
    }
}

/// Inferred schema for a dataset
#[derive(Debug, Clone, Default)]
pub struct InferredSchema {
    /// Feature name -> inferred type
    pub features: HashMap<String, FeatureType>,
    /// Column statistics used for inference
    pub stats: HashMap<String, ColumnStats>,
}

impl InferredSchema {
    /// Get features of a specific type
    pub fn features_of_type(&self, feature_type: FeatureType) -> Vec<&str> {
        self.features
            .iter()
            .filter(|(_, &t)| t == feature_type)
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get target columns
    pub fn targets(&self) -> Vec<&str> {
        self.features
            .iter()
            .filter(|(_, &t)| {
                matches!(
                    t,
                    FeatureType::BinaryTarget
                        | FeatureType::MultiClassTarget
                        | FeatureType::RegressionTarget
                )
            })
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get input feature columns (non-targets)
    pub fn inputs(&self) -> Vec<&str> {
        self.features
            .iter()
            .filter(|(_, &t)| {
                !matches!(
                    t,
                    FeatureType::BinaryTarget
                        | FeatureType::MultiClassTarget
                        | FeatureType::RegressionTarget
                )
            })
            .map(|(name, _)| name.as_str())
            .collect()
    }
}

/// Infer feature type from column statistics
pub fn infer_type(stats: &ColumnStats, config: &InferenceConfig) -> FeatureType {
    // Check if this is a target column
    // Match exact name or common suffixes like _label, _target
    let name_lower = stats.name.to_lowercase();
    let is_target = config.target_columns.iter().any(|t| {
        let t_lower = t.to_lowercase();
        name_lower == t_lower
            || name_lower.ends_with(&format!("_{}", t_lower))
            || name_lower.starts_with(&format!("{}_", t_lower))
    });

    // Skip excluded columns
    if config.exclude_columns.contains(&stats.name) {
        return FeatureType::Unknown;
    }

    // Check for embedding (fixed-size array of floats)
    if stats.is_array && stats.array_len.is_some() {
        return FeatureType::Embedding;
    }

    // Check for datetime
    if stats.looks_like_datetime {
        return FeatureType::DateTime;
    }

    // Check for numeric types
    if stats.all_numeric {
        // Target column inference
        if is_target {
            if stats.all_integers && stats.unique_count == 2 {
                return FeatureType::BinaryTarget;
            } else if stats.all_integers && stats.unique_count <= 100 {
                return FeatureType::MultiClassTarget;
            } else {
                return FeatureType::RegressionTarget;
            }
        }

        // Low cardinality integers -> categorical
        if stats.all_integers && stats.cardinality_ratio() < config.categorical_threshold {
            return FeatureType::Categorical;
        }

        return FeatureType::Numeric;
    }

    // String-based inference
    if let Some(avg_len) = stats.avg_str_len {
        // Long strings -> text
        if avg_len >= config.text_min_avg_len {
            return FeatureType::Text;
        }

        // Short strings with low cardinality -> categorical
        if stats.cardinality_ratio() < config.categorical_threshold {
            return FeatureType::Categorical;
        }

        // Token sequences (space-separated tokens)
        if stats
            .sample_values
            .iter()
            .any(|s| s.split_whitespace().count() > 5)
        {
            return FeatureType::TokenSequence;
        }

        return FeatureType::Text;
    }

    FeatureType::Unknown
}

/// Infer schema from column statistics
pub fn infer_schema(stats: Vec<ColumnStats>, config: &InferenceConfig) -> InferredSchema {
    let mut schema = InferredSchema::default();

    for col_stats in stats {
        let feature_type = infer_type(&col_stats, config);
        schema
            .features
            .insert(col_stats.name.clone(), feature_type);
        schema.stats.insert(col_stats.name.clone(), col_stats);
    }

    schema
}

/// Collect statistics from sample values (simplified in-memory analysis)
pub fn collect_stats_from_samples(
    name: &str,
    values: &[Option<&str>],
) -> ColumnStats {
    let mut stats = ColumnStats::new(name);
    stats.count = values.len();

    let mut unique: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut total_len = 0usize;
    let mut min_len = usize::MAX;
    let mut max_len = 0usize;
    let mut all_numeric = true;
    let mut all_integers = true;
    let mut datetime_count = 0usize;

    for val in values {
        match val {
            Some(s) => {
                unique.insert(s);
                let len = s.len();
                total_len += len;
                min_len = min_len.min(len);
                max_len = max_len.max(len);

                // Check if numeric
                if s.parse::<f64>().is_err() {
                    all_numeric = false;
                    all_integers = false;
                } else if s.parse::<i64>().is_err() {
                    all_integers = false;
                }

                // Simple datetime heuristic
                if s.contains('-') && s.len() >= 10 && s.len() <= 30 {
                    if s.chars().filter(|c| c.is_ascii_digit()).count() >= 8 {
                        datetime_count += 1;
                    }
                }

                if stats.sample_values.len() < 10 {
                    stats.sample_values.push(s.to_string());
                }
            }
            None => {
                stats.null_count += 1;
            }
        }
    }

    stats.unique_count = unique.len();
    stats.all_numeric = all_numeric && stats.null_count < stats.count;
    stats.all_integers = all_integers && stats.null_count < stats.count;

    let non_null = stats.count - stats.null_count;
    if non_null > 0 {
        stats.min_str_len = Some(min_len);
        stats.max_str_len = Some(max_len);
        stats.avg_str_len = Some(total_len as f32 / non_null as f32);
    }

    // Consider datetime if >50% look like timestamps
    stats.looks_like_datetime = non_null > 0 && datetime_count as f32 / non_null as f32 > 0.5;

    stats
}

/// Placeholder: Load stats from Parquet file
/// Real implementation would use arrow-rs/parquet crate
pub fn infer_schema_from_path(
    _path: &Path,
    _config: &InferenceConfig,
) -> Result<InferredSchema, std::io::Error> {
    // In a real implementation, this would:
    // 1. Open the Parquet file
    // 2. Read schema metadata
    // 3. Sample rows for statistics
    // 4. Call infer_schema()

    // For now, return empty schema
    Ok(InferredSchema::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn make_stats(
        name: &str,
        count: usize,
        unique: usize,
        all_numeric: bool,
        all_int: bool,
    ) -> ColumnStats {
        ColumnStats {
            name: name.to_string(),
            count,
            unique_count: unique,
            all_numeric,
            all_integers: all_int,
            ..Default::default()
        }
    }

    // ============================================================
    // Unit Tests
    // ============================================================

    #[test]
    fn test_infer_numeric() {
        let stats = make_stats("price", 1000, 500, true, false);
        let config = InferenceConfig::default();
        assert_eq!(infer_type(&stats, &config), FeatureType::Numeric);
    }

    #[test]
    fn test_infer_categorical_low_cardinality() {
        // Use column name that won't match target heuristics
        let stats = make_stats("status_code", 1000, 10, true, true);
        let config = InferenceConfig::default();
        assert_eq!(infer_type(&stats, &config), FeatureType::Categorical);
    }

    #[test]
    fn test_infer_binary_target() {
        let stats = make_stats("label", 1000, 2, true, true);
        let config = InferenceConfig::default();
        assert_eq!(infer_type(&stats, &config), FeatureType::BinaryTarget);
    }

    #[test]
    fn test_infer_multiclass_target() {
        let stats = make_stats("target", 1000, 10, true, true);
        let config = InferenceConfig::default();
        assert_eq!(infer_type(&stats, &config), FeatureType::MultiClassTarget);
    }

    #[test]
    fn test_infer_regression_target() {
        let stats = make_stats("y", 1000, 800, true, false);
        let config = InferenceConfig::default();
        assert_eq!(infer_type(&stats, &config), FeatureType::RegressionTarget);
    }

    #[test]
    fn test_infer_text() {
        let mut stats = make_stats("description", 1000, 900, false, false);
        stats.avg_str_len = Some(100.0);
        let config = InferenceConfig::default();
        assert_eq!(infer_type(&stats, &config), FeatureType::Text);
    }

    #[test]
    fn test_infer_categorical_string() {
        let mut stats = make_stats("status", 1000, 5, false, false);
        stats.avg_str_len = Some(8.0);
        let config = InferenceConfig::default();
        assert_eq!(infer_type(&stats, &config), FeatureType::Categorical);
    }

    #[test]
    fn test_infer_datetime() {
        let mut stats = make_stats("created_at", 1000, 1000, false, false);
        stats.looks_like_datetime = true;
        let config = InferenceConfig::default();
        assert_eq!(infer_type(&stats, &config), FeatureType::DateTime);
    }

    #[test]
    fn test_infer_embedding() {
        let mut stats = make_stats("embedding", 1000, 1000, true, false);
        stats.is_array = true;
        stats.array_len = Some(768);
        let config = InferenceConfig::default();
        assert_eq!(infer_type(&stats, &config), FeatureType::Embedding);
    }

    #[test]
    fn test_infer_schema() {
        let stats = vec![
            make_stats("id", 1000, 1000, true, true),
            make_stats("label", 1000, 2, true, true),
            make_stats("price", 1000, 500, true, false),
        ];
        let config = InferenceConfig::default();
        let schema = infer_schema(stats, &config);

        assert_eq!(schema.features.len(), 3);
        assert_eq!(schema.features["label"], FeatureType::BinaryTarget);
        assert_eq!(schema.features["price"], FeatureType::Numeric);
    }

    #[test]
    fn test_schema_targets() {
        let stats = vec![
            make_stats("x1", 100, 50, true, false),
            make_stats("x2", 100, 50, true, false),
            make_stats("y", 100, 80, true, false),
        ];
        let config = InferenceConfig::default();
        let schema = infer_schema(stats, &config);

        let targets = schema.targets();
        assert_eq!(targets.len(), 1);
        assert!(targets.contains(&"y"));
    }

    #[test]
    fn test_schema_inputs() {
        let stats = vec![
            make_stats("x1", 100, 50, true, false),
            make_stats("x2", 100, 50, true, false),
            make_stats("y", 100, 80, true, false),
        ];
        let config = InferenceConfig::default();
        let schema = infer_schema(stats, &config);

        let inputs = schema.inputs();
        assert_eq!(inputs.len(), 2);
        assert!(inputs.contains(&"x1"));
        assert!(inputs.contains(&"x2"));
    }

    #[test]
    fn test_collect_stats_numeric() {
        let values: Vec<Option<&str>> = vec![
            Some("1.5"),
            Some("2.3"),
            Some("3.7"),
            None,
            Some("4.1"),
        ];
        let stats = collect_stats_from_samples("price", &values);

        assert_eq!(stats.count, 5);
        assert_eq!(stats.null_count, 1);
        assert!(stats.all_numeric);
        assert!(!stats.all_integers);
    }

    #[test]
    fn test_collect_stats_integers() {
        let values: Vec<Option<&str>> = vec![
            Some("1"),
            Some("2"),
            Some("3"),
            Some("4"),
            Some("5"),
        ];
        let stats = collect_stats_from_samples("count", &values);

        assert!(stats.all_numeric);
        assert!(stats.all_integers);
        assert_eq!(stats.unique_count, 5);
    }

    #[test]
    fn test_collect_stats_datetime() {
        let values: Vec<Option<&str>> = vec![
            Some("2024-01-15"),
            Some("2024-02-20"),
            Some("2024-03-25"),
        ];
        let stats = collect_stats_from_samples("date", &values);

        assert!(stats.looks_like_datetime);
    }

    #[test]
    fn test_cardinality_ratio() {
        let stats = ColumnStats {
            count: 1000,
            unique_count: 50,
            ..Default::default()
        };
        assert!((stats.cardinality_ratio() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_null_ratio() {
        let stats = ColumnStats {
            count: 100,
            null_count: 10,
            ..Default::default()
        };
        assert!((stats.null_ratio() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_feature_type_display() {
        assert_eq!(format!("{}", FeatureType::Numeric), "numeric");
        assert_eq!(format!("{}", FeatureType::Categorical), "categorical");
        assert_eq!(format!("{}", FeatureType::BinaryTarget), "binary_target");
    }

    // ============================================================
    // Property Tests
    // ============================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_cardinality_ratio_bounded(
            count in 1usize..10000,
            unique in 1usize..10000
        ) {
            let unique = unique.min(count);
            let stats = ColumnStats {
                count,
                unique_count: unique,
                ..Default::default()
            };
            let ratio = stats.cardinality_ratio();
            prop_assert!(ratio >= 0.0 && ratio <= 1.0);
        }

        #[test]
        fn prop_null_ratio_bounded(
            count in 1usize..10000,
            null_count in 0usize..10000
        ) {
            let null_count = null_count.min(count);
            let stats = ColumnStats {
                count,
                null_count,
                ..Default::default()
            };
            let ratio = stats.null_ratio();
            prop_assert!(ratio >= 0.0 && ratio <= 1.0);
        }

        #[test]
        fn prop_numeric_low_cardinality_is_categorical(
            count in 1000usize..10000,
            unique in 2usize..20
        ) {
            // Ensure cardinality ratio < 0.05 threshold
            let unique = unique.min((count as f32 * 0.04) as usize).max(2);
            let stats = make_stats("feature_col", count, unique, true, true);
            // Use empty target columns to avoid accidental target matching
            let config = InferenceConfig {
                target_columns: vec![],
                ..Default::default()
            };
            let inferred = infer_type(&stats, &config);
            // Low cardinality integers should be categorical
            prop_assert_eq!(inferred, FeatureType::Categorical);
        }

        #[test]
        fn prop_high_cardinality_numeric_stays_numeric(
            count in 1000usize..10000,
            unique_ratio in 0.5f32..1.0
        ) {
            let unique = (count as f32 * unique_ratio) as usize;
            let stats = make_stats("col", count, unique, true, false);
            let config = InferenceConfig::default();
            let inferred = infer_type(&stats, &config);
            prop_assert_eq!(inferred, FeatureType::Numeric);
        }

        #[test]
        fn prop_binary_target_detected(
            count in 100usize..10000
        ) {
            let stats = make_stats("label", count, 2, true, true);
            let config = InferenceConfig::default();
            let inferred = infer_type(&stats, &config);
            prop_assert_eq!(inferred, FeatureType::BinaryTarget);
        }

        #[test]
        fn prop_multiclass_target_detected(
            count in 100usize..10000,
            classes in 3usize..50
        ) {
            let stats = make_stats("target", count, classes, true, true);
            let config = InferenceConfig::default();
            let inferred = infer_type(&stats, &config);
            prop_assert_eq!(inferred, FeatureType::MultiClassTarget);
        }

        #[test]
        fn prop_text_detected_by_length(
            count in 100usize..1000,
            avg_len in 50.0f32..500.0
        ) {
            let mut stats = make_stats("description", count, count, false, false);
            stats.avg_str_len = Some(avg_len);
            let config = InferenceConfig::default();
            let inferred = infer_type(&stats, &config);
            prop_assert_eq!(inferred, FeatureType::Text);
        }

        #[test]
        fn prop_embedding_detected(
            count in 100usize..1000,
            embed_dim in 64usize..2048
        ) {
            let mut stats = make_stats("embedding", count, count, true, false);
            stats.is_array = true;
            stats.array_len = Some(embed_dim);
            let config = InferenceConfig::default();
            let inferred = infer_type(&stats, &config);
            prop_assert_eq!(inferred, FeatureType::Embedding);
        }

        #[test]
        fn prop_datetime_detected(count in 100usize..1000) {
            let mut stats = make_stats("timestamp", count, count, false, false);
            stats.looks_like_datetime = true;
            let config = InferenceConfig::default();
            let inferred = infer_type(&stats, &config);
            prop_assert_eq!(inferred, FeatureType::DateTime);
        }

        #[test]
        fn prop_schema_preserves_all_columns(
            num_cols in 1usize..20
        ) {
            let stats: Vec<ColumnStats> = (0..num_cols)
                .map(|i| make_stats(&format!("col_{}", i), 100, 50, true, false))
                .collect();
            let config = InferenceConfig::default();
            let schema = infer_schema(stats, &config);
            prop_assert_eq!(schema.features.len(), num_cols);
        }

        #[test]
        fn prop_targets_and_inputs_partition(
            num_features in 1usize..10,
            num_targets in 0usize..3
        ) {
            let mut stats: Vec<ColumnStats> = (0..num_features)
                .map(|i| make_stats(&format!("x{}", i), 100, 50, true, false))
                .collect();

            // Add targets
            for i in 0..num_targets {
                stats.push(make_stats(&format!("y{}", i), 100, 80, true, false));
            }

            let mut config = InferenceConfig::default();
            config.target_columns = (0..num_targets).map(|i| format!("y{}", i)).collect();

            let schema = infer_schema(stats, &config);

            let inputs = schema.inputs();
            let targets = schema.targets();

            // Inputs and targets should partition all columns
            prop_assert_eq!(inputs.len() + targets.len(), num_features + num_targets);
        }
    }
}
