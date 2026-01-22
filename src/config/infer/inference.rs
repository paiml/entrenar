//! Type inference functions

use std::collections::HashSet;
use std::path::Path;

use super::config::InferenceConfig;
use super::schema::InferredSchema;
use super::stats::ColumnStats;
use super::types::FeatureType;

/// Infer feature type from column statistics
pub fn infer_type(stats: &ColumnStats, config: &InferenceConfig) -> FeatureType {
    // Check if this is a target column
    // Match exact name or common suffixes like _label, _target
    let name_lower = stats.name.to_lowercase();
    let is_target = config.target_columns.iter().any(|t| {
        let t_lower = t.to_lowercase();
        name_lower == t_lower
            || name_lower.ends_with(&format!("_{t_lower}"))
            || name_lower.starts_with(&format!("{t_lower}_"))
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
            }
            return FeatureType::RegressionTarget;
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
        schema.features.insert(col_stats.name.clone(), feature_type);
        schema.stats.insert(col_stats.name.clone(), col_stats);
    }

    schema
}

/// Collect statistics from sample values (simplified in-memory analysis)
pub fn collect_stats_from_samples(name: &str, values: &[Option<&str>]) -> ColumnStats {
    let mut stats = ColumnStats::new(name);
    stats.count = values.len();

    let mut unique: HashSet<&str> = HashSet::new();
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
                if s.contains('-')
                    && s.len() >= 10
                    && s.len() <= 30
                    && s.chars().filter(char::is_ascii_digit).count() >= 8
                {
                    datetime_count += 1;
                }

                if stats.sample_values.len() < 10 {
                    stats.sample_values.push((*s).to_string());
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
