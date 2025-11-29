//! Decision pattern storage with hybrid retrieval
//!
//! Uses trueno-rag for BM25 lexical search combined with dense embeddings
//! and Reciprocal Rank Fusion (RRF) for optimal fix suggestions.
//!
//! # References
//! - Lewis et al. (2020): Retrieval-Augmented Generation
//! - Cormack et al. (2009): Reciprocal Rank Fusion

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use trueno_rag::{
    chunk::FixedSizeChunker, embed::MockEmbedder, fusion::FusionStrategy,
    pipeline::RagPipelineBuilder, rerank::NoOpReranker, Document, RagPipeline,
};

/// Unique identifier for a chunk in the pattern store
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkId(pub uuid::Uuid);

impl ChunkId {
    /// Create a new random chunk ID
    #[must_use]
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl Default for ChunkId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ChunkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A pattern representing a successful fix for an error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixPattern {
    /// Unique identifier for this pattern
    pub id: ChunkId,
    /// The error code this pattern fixes (e.g., "E0308", "E0382")
    pub error_code: String,
    /// Sequence of decisions that led to this fix
    pub decision_sequence: Vec<String>,
    /// The actual fix diff (unified diff format)
    pub fix_diff: String,
    /// Number of times this pattern was successfully applied
    pub success_count: u32,
    /// Number of times this pattern was attempted
    pub attempt_count: u32,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl FixPattern {
    /// Create a new fix pattern
    #[must_use]
    pub fn new(error_code: impl Into<String>, fix_diff: impl Into<String>) -> Self {
        Self {
            id: ChunkId::new(),
            error_code: error_code.into(),
            decision_sequence: Vec::new(),
            fix_diff: fix_diff.into(),
            success_count: 0,
            attempt_count: 0,
            metadata: HashMap::new(),
        }
    }

    /// Add a decision to the sequence
    #[must_use]
    pub fn with_decision(mut self, decision: impl Into<String>) -> Self {
        self.decision_sequence.push(decision.into());
        self
    }

    /// Add multiple decisions to the sequence
    #[must_use]
    pub fn with_decisions(mut self, decisions: Vec<String>) -> Self {
        self.decision_sequence.extend(decisions);
        self
    }

    /// Record a successful application
    pub fn record_success(&mut self) {
        self.success_count += 1;
        self.attempt_count += 1;
    }

    /// Record a failed application
    pub fn record_failure(&mut self) {
        self.attempt_count += 1;
    }

    /// Calculate success rate
    #[must_use]
    pub fn success_rate(&self) -> f32 {
        if self.attempt_count == 0 {
            0.0
        } else {
            self.success_count as f32 / self.attempt_count as f32
        }
    }

    /// Convert to searchable text for indexing
    #[must_use]
    pub fn to_searchable_text(&self) -> String {
        let decisions = self.decision_sequence.join(" ");
        format!("{} {} {}", self.error_code, decisions, self.fix_diff)
    }
}

/// A suggested fix from the pattern store
#[derive(Debug, Clone)]
pub struct FixSuggestion {
    /// The fix pattern
    pub pattern: FixPattern,
    /// Retrieval score from the RAG pipeline
    pub score: f32,
    /// Rank in the result set
    pub rank: usize,
}

impl FixSuggestion {
    /// Create a new fix suggestion
    #[must_use]
    pub fn new(pattern: FixPattern, score: f32, rank: usize) -> Self {
        Self {
            pattern,
            score,
            rank,
        }
    }

    /// Get the weighted score (retrieval score * success rate)
    #[must_use]
    pub fn weighted_score(&self) -> f32 {
        self.score * (0.5 + 0.5 * self.pattern.success_rate())
    }
}

/// Configuration for the pattern store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternStoreConfig {
    /// Chunk size for the chunker (default: 256)
    pub chunk_size: usize,
    /// Embedding dimension (default: 384)
    pub embedding_dim: usize,
    /// RRF k constant (default: 60.0)
    pub rrf_k: f32,
}

impl Default for PatternStoreConfig {
    fn default() -> Self {
        Self {
            chunk_size: 256,
            embedding_dim: 384,
            rrf_k: 60.0,
        }
    }
}

/// Store for decision patterns with hybrid retrieval
///
/// Uses trueno-rag for BM25 + dense embedding retrieval with RRF fusion.
///
/// # Example
///
/// ```ignore
/// use entrenar::citl::{DecisionPatternStore, FixPattern};
///
/// let mut store = DecisionPatternStore::new()?;
///
/// // Index a fix pattern
/// let pattern = FixPattern::new("E0308", "- let x: i32 = \"hello\";\n+ let x: &str = \"hello\";")
///     .with_decision("type_mismatch_detected")
///     .with_decision("infer_correct_type");
/// store.index_fix(pattern)?;
///
/// // Get fix suggestions
/// let suggestions = store.suggest_fix("E0308", &["type_mismatch"], 5)?;
/// ```
pub struct DecisionPatternStore {
    /// RAG pipeline for hybrid retrieval
    pipeline: RagPipeline<MockEmbedder, NoOpReranker>,
    /// Pattern storage indexed by chunk ID
    patterns: HashMap<ChunkId, FixPattern>,
    /// Error code index for fast filtering
    error_index: HashMap<String, Vec<ChunkId>>,
    /// Configuration
    config: PatternStoreConfig,
}

impl DecisionPatternStore {
    /// Create a new pattern store with default configuration
    pub fn new() -> Result<Self, crate::Error> {
        Self::with_config(PatternStoreConfig::default())
    }

    /// Create a new pattern store with custom configuration
    pub fn with_config(config: PatternStoreConfig) -> Result<Self, crate::Error> {
        let pipeline = RagPipelineBuilder::new()
            .chunker(FixedSizeChunker::new(
                config.chunk_size,
                config.chunk_size / 8,
            ))
            .embedder(MockEmbedder::new(config.embedding_dim))
            .reranker(NoOpReranker::new())
            .fusion(FusionStrategy::RRF { k: config.rrf_k })
            .build()
            .map_err(|e| crate::Error::ConfigError(format!("RAG pipeline error: {e}")))?;

        Ok(Self {
            pipeline,
            patterns: HashMap::new(),
            error_index: HashMap::new(),
            config,
        })
    }

    /// Index a fix pattern for later retrieval
    pub fn index_fix(&mut self, pattern: FixPattern) -> Result<(), crate::Error> {
        let chunk_id = pattern.id;
        let error_code = pattern.error_code.clone();

        // Create searchable document
        let doc = Document::new(pattern.to_searchable_text())
            .with_title(format!("Fix for {}", pattern.error_code));

        // Index in RAG pipeline
        self.pipeline
            .index_document(&doc)
            .map_err(|e| crate::Error::ConfigError(format!("Indexing error: {e}")))?;

        // Update error index
        self.error_index
            .entry(error_code)
            .or_default()
            .push(chunk_id);

        // Store pattern
        self.patterns.insert(chunk_id, pattern);

        Ok(())
    }

    /// Suggest fixes for a given error code and decision context
    ///
    /// # Arguments
    ///
    /// * `error_code` - The error code to find fixes for
    /// * `decision_context` - Recent decisions that led to the error
    /// * `k` - Maximum number of suggestions to return
    ///
    /// # Returns
    ///
    /// Vector of fix suggestions ranked by relevance
    pub fn suggest_fix(
        &self,
        error_code: &str,
        decision_context: &[String],
        k: usize,
    ) -> Result<Vec<FixSuggestion>, crate::Error> {
        // Build query from error code and decision context
        let context_str = decision_context.join(" ");
        let query = format!("{error_code} {context_str}");

        // Retrieve from RAG pipeline
        let results = self
            .pipeline
            .query(&query, k * 2) // Over-fetch for filtering
            .map_err(|e| crate::Error::ConfigError(format!("Query error: {e}")))?;

        // Filter by error code if we have patterns for it
        let relevant_patterns: Vec<_> = if let Some(pattern_ids) = self.error_index.get(error_code)
        {
            pattern_ids
                .iter()
                .filter_map(|id| self.patterns.get(id))
                .collect()
        } else {
            // Return any patterns if no exact error code match
            self.patterns.values().collect()
        };

        // Match RAG results with our patterns (by content similarity)
        let mut suggestions: Vec<FixSuggestion> = Vec::new();

        for (rank, result) in results.iter().enumerate() {
            // Find matching pattern by comparing content
            for pattern in &relevant_patterns {
                let pattern_text = pattern.to_searchable_text();
                if result.chunk.content.contains(&pattern.error_code)
                    || pattern_text.contains(&result.chunk.content)
                {
                    suggestions.push(FixSuggestion::new(
                        (*pattern).clone(),
                        result.best_score(),
                        rank,
                    ));
                    break;
                }
            }
        }

        // If no RAG matches, fall back to error index
        if suggestions.is_empty() && !relevant_patterns.is_empty() {
            for (rank, pattern) in relevant_patterns.iter().take(k).enumerate() {
                suggestions.push(FixSuggestion::new(
                    (*pattern).clone(),
                    1.0 - (rank as f32 * 0.1),
                    rank,
                ));
            }
        }

        // Sort by weighted score and limit
        suggestions.sort_by(|a, b| {
            b.weighted_score()
                .partial_cmp(&a.weighted_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        suggestions.truncate(k);

        // Re-assign ranks after sorting
        for (rank, suggestion) in suggestions.iter_mut().enumerate() {
            suggestion.rank = rank;
        }

        Ok(suggestions)
    }

    /// Get the number of indexed patterns
    #[must_use]
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Check if the store is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Get a pattern by ID
    #[must_use]
    pub fn get(&self, id: &ChunkId) -> Option<&FixPattern> {
        self.patterns.get(id)
    }

    /// Get a mutable pattern by ID
    pub fn get_mut(&mut self, id: &ChunkId) -> Option<&mut FixPattern> {
        self.patterns.get_mut(id)
    }

    /// Update a pattern's success/failure count
    pub fn record_outcome(&mut self, id: &ChunkId, success: bool) {
        if let Some(pattern) = self.patterns.get_mut(id) {
            if success {
                pattern.record_success();
            } else {
                pattern.record_failure();
            }
        }
    }

    /// Get all patterns for an error code
    #[must_use]
    pub fn patterns_for_error(&self, error_code: &str) -> Vec<&FixPattern> {
        self.error_index
            .get(error_code)
            .map(|ids| ids.iter().filter_map(|id| self.patterns.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &PatternStoreConfig {
        &self.config
    }

    /// Export all patterns to JSON
    pub fn export_json(&self) -> Result<String, crate::Error> {
        let patterns: Vec<_> = self.patterns.values().collect();
        serde_json::to_string_pretty(&patterns)
            .map_err(|e| crate::Error::Serialization(format!("JSON export error: {e}")))
    }

    /// Import patterns from JSON
    pub fn import_json(&mut self, json: &str) -> Result<usize, crate::Error> {
        let patterns: Vec<FixPattern> = serde_json::from_str(json)
            .map_err(|e| crate::Error::Serialization(format!("JSON import error: {e}")))?;

        let count = patterns.len();
        for pattern in patterns {
            self.index_fix(pattern)?;
        }

        Ok(count)
    }

    /// Save patterns to .apr format (aprender model format)
    ///
    /// Uses `ModelType::Custom` with compressed MessagePack serialization.
    /// The .apr format provides:
    /// - CRC32 checksum (integrity)
    /// - Optional zstd compression
    /// - Compatible with aprender ecosystem
    ///
    /// # Example
    ///
    /// ```ignore
    /// use entrenar::citl::DecisionPatternStore;
    ///
    /// let store = DecisionPatternStore::new()?;
    /// // ... index patterns ...
    /// store.save_apr("decision_patterns.apr")?;
    /// ```
    pub fn save_apr(&self, path: impl AsRef<Path>) -> Result<(), crate::Error> {
        use aprender::format::{save, Compression, ModelType, SaveOptions};

        // Collect patterns into serializable wrapper
        let patterns: Vec<FixPattern> = self.patterns.values().cloned().collect();
        let wrapper = PatternStoreData {
            version: 1,
            config: self.config.clone(),
            patterns,
        };

        save(
            &wrapper,
            ModelType::Custom,
            path,
            SaveOptions::default().with_compression(Compression::ZstdDefault),
        )
        .map_err(|e| crate::Error::Serialization(format!("APR save error: {e}")))
    }

    /// Load patterns from .apr format
    ///
    /// Restores patterns and rebuilds the RAG index.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use entrenar::citl::DecisionPatternStore;
    ///
    /// let store = DecisionPatternStore::load_apr("decision_patterns.apr")?;
    /// let suggestions = store.suggest_fix("E0308", &["type_mismatch".into()], 5)?;
    /// ```
    pub fn load_apr(path: impl AsRef<Path>) -> Result<Self, crate::Error> {
        use aprender::format::{load, ModelType};

        let wrapper: PatternStoreData = load(path, ModelType::Custom)
            .map_err(|e| crate::Error::Serialization(format!("APR load error: {e}")))?;

        // Rebuild store with loaded config
        let mut store = Self::with_config(wrapper.config)?;

        // Re-index all patterns
        for pattern in wrapper.patterns {
            store.index_fix(pattern)?;
        }

        Ok(store)
    }
}

/// Serializable wrapper for pattern store data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PatternStoreData {
    /// Format version for future compatibility
    version: u32,
    /// Store configuration
    config: PatternStoreConfig,
    /// All indexed patterns
    patterns: Vec<FixPattern>,
}

impl std::fmt::Debug for DecisionPatternStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionPatternStore")
            .field("pattern_count", &self.patterns.len())
            .field("error_codes", &self.error_index.keys().collect::<Vec<_>>())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============ ChunkId Tests ============

    #[test]
    fn test_chunk_id_unique() {
        let id1 = ChunkId::new();
        let id2 = ChunkId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_chunk_id_display() {
        let id = ChunkId::new();
        let display = format!("{id}");
        assert!(!display.is_empty());
        assert!(display.contains('-')); // UUID format
    }

    #[test]
    fn test_chunk_id_serialization() {
        let id = ChunkId::new();
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: ChunkId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }

    // ============ FixPattern Tests ============

    #[test]
    fn test_fix_pattern_new() {
        let pattern = FixPattern::new("E0308", "- old\n+ new");
        assert_eq!(pattern.error_code, "E0308");
        assert_eq!(pattern.fix_diff, "- old\n+ new");
        assert!(pattern.decision_sequence.is_empty());
        assert_eq!(pattern.success_count, 0);
        assert_eq!(pattern.attempt_count, 0);
    }

    #[test]
    fn test_fix_pattern_with_decision() {
        let pattern = FixPattern::new("E0308", "diff")
            .with_decision("detect_mismatch")
            .with_decision("suggest_fix");

        assert_eq!(pattern.decision_sequence.len(), 2);
        assert_eq!(pattern.decision_sequence[0], "detect_mismatch");
        assert_eq!(pattern.decision_sequence[1], "suggest_fix");
    }

    #[test]
    fn test_fix_pattern_with_decisions() {
        let decisions = vec!["step1".to_string(), "step2".to_string()];
        let pattern = FixPattern::new("E0308", "diff").with_decisions(decisions);

        assert_eq!(pattern.decision_sequence.len(), 2);
    }

    #[test]
    fn test_fix_pattern_record_success() {
        let mut pattern = FixPattern::new("E0308", "diff");
        pattern.record_success();
        pattern.record_success();

        assert_eq!(pattern.success_count, 2);
        assert_eq!(pattern.attempt_count, 2);
    }

    #[test]
    fn test_fix_pattern_record_failure() {
        let mut pattern = FixPattern::new("E0308", "diff");
        pattern.record_success();
        pattern.record_failure();

        assert_eq!(pattern.success_count, 1);
        assert_eq!(pattern.attempt_count, 2);
    }

    #[test]
    fn test_fix_pattern_success_rate() {
        let mut pattern = FixPattern::new("E0308", "diff");
        assert_eq!(pattern.success_rate(), 0.0);

        pattern.record_success();
        pattern.record_success();
        pattern.record_failure();

        assert!((pattern.success_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_fix_pattern_to_searchable_text() {
        let pattern = FixPattern::new("E0308", "- i32\n+ &str").with_decision("type_mismatch");

        let text = pattern.to_searchable_text();
        assert!(text.contains("E0308"));
        assert!(text.contains("type_mismatch"));
        assert!(text.contains("- i32"));
    }

    #[test]
    fn test_fix_pattern_serialization() {
        let pattern = FixPattern::new("E0308", "diff").with_decision("step1");

        let json = serde_json::to_string(&pattern).unwrap();
        let deserialized: FixPattern = serde_json::from_str(&json).unwrap();

        assert_eq!(pattern.error_code, deserialized.error_code);
        assert_eq!(pattern.fix_diff, deserialized.fix_diff);
        assert_eq!(pattern.decision_sequence, deserialized.decision_sequence);
    }

    // ============ FixSuggestion Tests ============

    #[test]
    fn test_fix_suggestion_new() {
        let pattern = FixPattern::new("E0308", "diff");
        let suggestion = FixSuggestion::new(pattern, 0.85, 0);

        assert_eq!(suggestion.score, 0.85);
        assert_eq!(suggestion.rank, 0);
    }

    #[test]
    fn test_fix_suggestion_weighted_score() {
        let mut pattern = FixPattern::new("E0308", "diff");
        pattern.record_success();
        pattern.record_success();

        let suggestion = FixSuggestion::new(pattern, 0.8, 0);
        // weighted = 0.8 * (0.5 + 0.5 * 1.0) = 0.8 * 1.0 = 0.8
        assert!((suggestion.weighted_score() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_fix_suggestion_weighted_score_partial_success() {
        let mut pattern = FixPattern::new("E0308", "diff");
        pattern.record_success();
        pattern.record_failure();

        let suggestion = FixSuggestion::new(pattern, 1.0, 0);
        // success_rate = 0.5, weighted = 1.0 * (0.5 + 0.5 * 0.5) = 1.0 * 0.75 = 0.75
        assert!((suggestion.weighted_score() - 0.75).abs() < 0.01);
    }

    // ============ PatternStoreConfig Tests ============

    #[test]
    fn test_pattern_store_config_default() {
        let config = PatternStoreConfig::default();
        assert_eq!(config.chunk_size, 256);
        assert_eq!(config.embedding_dim, 384);
        assert!((config.rrf_k - 60.0).abs() < 0.01);
    }

    // ============ DecisionPatternStore Tests ============

    #[test]
    fn test_pattern_store_new() {
        let store = DecisionPatternStore::new().unwrap();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_pattern_store_with_config() {
        let config = PatternStoreConfig {
            chunk_size: 512,
            embedding_dim: 768,
            rrf_k: 30.0,
        };
        let store = DecisionPatternStore::with_config(config.clone()).unwrap();
        assert_eq!(store.config().chunk_size, 512);
        assert_eq!(store.config().embedding_dim, 768);
    }

    #[test]
    fn test_pattern_store_index_fix() {
        let mut store = DecisionPatternStore::new().unwrap();

        let pattern = FixPattern::new("E0308", "- i32\n+ &str").with_decision("type_mismatch");

        store.index_fix(pattern).unwrap();

        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_pattern_store_index_multiple() {
        let mut store = DecisionPatternStore::new().unwrap();

        store.index_fix(FixPattern::new("E0308", "fix1")).unwrap();
        store.index_fix(FixPattern::new("E0308", "fix2")).unwrap();
        store.index_fix(FixPattern::new("E0382", "fix3")).unwrap();

        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_pattern_store_get() {
        let mut store = DecisionPatternStore::new().unwrap();

        let pattern = FixPattern::new("E0308", "diff");
        let id = pattern.id;

        store.index_fix(pattern).unwrap();

        let retrieved = store.get(&id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().error_code, "E0308");
    }

    #[test]
    fn test_pattern_store_get_mut() {
        let mut store = DecisionPatternStore::new().unwrap();

        let pattern = FixPattern::new("E0308", "diff");
        let id = pattern.id;

        store.index_fix(pattern).unwrap();

        if let Some(p) = store.get_mut(&id) {
            p.record_success();
        }

        assert_eq!(store.get(&id).unwrap().success_count, 1);
    }

    #[test]
    fn test_pattern_store_record_outcome() {
        let mut store = DecisionPatternStore::new().unwrap();

        let pattern = FixPattern::new("E0308", "diff");
        let id = pattern.id;

        store.index_fix(pattern).unwrap();
        store.record_outcome(&id, true);
        store.record_outcome(&id, false);

        let p = store.get(&id).unwrap();
        assert_eq!(p.success_count, 1);
        assert_eq!(p.attempt_count, 2);
    }

    #[test]
    fn test_pattern_store_patterns_for_error() {
        let mut store = DecisionPatternStore::new().unwrap();

        store.index_fix(FixPattern::new("E0308", "fix1")).unwrap();
        store.index_fix(FixPattern::new("E0308", "fix2")).unwrap();
        store.index_fix(FixPattern::new("E0382", "fix3")).unwrap();

        let e0308_patterns = store.patterns_for_error("E0308");
        assert_eq!(e0308_patterns.len(), 2);

        let e0382_patterns = store.patterns_for_error("E0382");
        assert_eq!(e0382_patterns.len(), 1);

        let e0000_patterns = store.patterns_for_error("E0000");
        assert!(e0000_patterns.is_empty());
    }

    #[test]
    fn test_pattern_store_suggest_fix() {
        let mut store = DecisionPatternStore::new().unwrap();

        let pattern = FixPattern::new(
            "E0308",
            "- let x: i32 = \"hello\";\n+ let x: &str = \"hello\";",
        )
        .with_decision("type_mismatch_detected")
        .with_decision("infer_correct_type");

        store.index_fix(pattern).unwrap();

        let context = vec!["type_mismatch".to_string()];
        let suggestions = store.suggest_fix("E0308", &context, 5).unwrap();

        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0].pattern.error_code, "E0308");
    }

    #[test]
    fn test_pattern_store_suggest_fix_empty() {
        let store = DecisionPatternStore::new().unwrap();

        let suggestions = store.suggest_fix("E0308", &[], 5).unwrap();
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_pattern_store_suggest_fix_ranking() {
        let mut store = DecisionPatternStore::new().unwrap();

        // Pattern with high success rate
        let mut pattern1 = FixPattern::new("E0308", "fix1 high success");
        pattern1.record_success();
        pattern1.record_success();
        store.index_fix(pattern1).unwrap();

        // Pattern with low success rate
        let mut pattern2 = FixPattern::new("E0308", "fix2 low success");
        pattern2.record_failure();
        pattern2.record_failure();
        store.index_fix(pattern2).unwrap();

        let suggestions = store.suggest_fix("E0308", &[], 5).unwrap();

        // Higher success rate should be ranked first
        assert!(!suggestions.is_empty());
        // The first suggestion should have higher weighted score
        if suggestions.len() >= 2 {
            assert!(suggestions[0].weighted_score() >= suggestions[1].weighted_score());
        }
    }

    #[test]
    fn test_pattern_store_export_import_json() {
        let mut store = DecisionPatternStore::new().unwrap();

        store.index_fix(FixPattern::new("E0308", "fix1")).unwrap();
        store.index_fix(FixPattern::new("E0382", "fix2")).unwrap();

        let json = store.export_json().unwrap();

        let mut new_store = DecisionPatternStore::new().unwrap();
        let count = new_store.import_json(&json).unwrap();

        assert_eq!(count, 2);
        assert_eq!(new_store.len(), 2);
    }

    #[test]
    fn test_pattern_store_debug() {
        let mut store = DecisionPatternStore::new().unwrap();
        store.index_fix(FixPattern::new("E0308", "fix")).unwrap();

        let debug = format!("{store:?}");
        assert!(debug.contains("DecisionPatternStore"));
        assert!(debug.contains("pattern_count"));
    }

    // ============ Property Tests ============

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_fix_pattern_success_rate_bounded(
            successes in 0u32..100,
            failures in 0u32..100
        ) {
            let mut pattern = FixPattern::new("E0308", "diff");
            for _ in 0..successes {
                pattern.record_success();
            }
            for _ in 0..failures {
                pattern.record_failure();
            }

            let rate = pattern.success_rate();
            prop_assert!(rate >= 0.0);
            prop_assert!(rate <= 1.0);
        }

        #[test]
        fn prop_fix_pattern_searchable_contains_error_code(
            error_code in "[A-Z][0-9]{4}"
        ) {
            let pattern = FixPattern::new(&error_code, "diff");
            let text = pattern.to_searchable_text();
            prop_assert!(text.contains(&error_code));
        }

        #[test]
        fn prop_chunk_id_serialization_roundtrip(n in 0u128..1000) {
            let id = ChunkId(uuid::Uuid::from_u128(n));
            let json = serde_json::to_string(&id).unwrap();
            let deserialized: ChunkId = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(id, deserialized);
        }

        #[test]
        fn prop_suggestion_weighted_score_positive(
            score in 0.0f32..1.0,
            successes in 0u32..10,
            attempts in 1u32..20
        ) {
            let mut pattern = FixPattern::new("E0308", "diff");
            let actual_attempts = attempts.max(successes);
            for _ in 0..successes {
                pattern.record_success();
            }
            for _ in 0..(actual_attempts - successes) {
                pattern.record_failure();
            }

            let suggestion = FixSuggestion::new(pattern, score, 0);
            prop_assert!(suggestion.weighted_score() >= 0.0);
        }
    }

    // ============ APR Persistence Tests ============

    #[test]
    fn test_pattern_store_save_load_apr() {
        let mut store = DecisionPatternStore::new().unwrap();

        // Add patterns with various states
        let mut pattern1 = FixPattern::new("E0308", "- i32\n+ &str")
            .with_decision("type_mismatch")
            .with_decision("infer_string");
        pattern1.record_success();
        pattern1.record_success();
        store.index_fix(pattern1).unwrap();

        let pattern2 = FixPattern::new("E0382", "- x\n+ x.clone()")
            .with_decision("borrow_after_move");
        store.index_fix(pattern2).unwrap();

        // Save to temp file
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("patterns.apr");

        store.save_apr(&path).unwrap();
        assert!(path.exists());

        // Load and verify
        let loaded = DecisionPatternStore::load_apr(&path).unwrap();
        assert_eq!(loaded.len(), 2);

        // Verify patterns for each error code
        let e0308_patterns = loaded.patterns_for_error("E0308");
        assert_eq!(e0308_patterns.len(), 1);
        assert_eq!(e0308_patterns[0].success_count, 2);
        assert_eq!(e0308_patterns[0].decision_sequence.len(), 2);

        let e0382_patterns = loaded.patterns_for_error("E0382");
        assert_eq!(e0382_patterns.len(), 1);
    }

    #[test]
    fn test_pattern_store_apr_config_preserved() {
        let config = PatternStoreConfig {
            chunk_size: 512,
            embedding_dim: 768,
            rrf_k: 30.0,
        };
        let mut store = DecisionPatternStore::with_config(config).unwrap();
        store.index_fix(FixPattern::new("E0308", "fix")).unwrap();

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("patterns.apr");

        store.save_apr(&path).unwrap();
        let loaded = DecisionPatternStore::load_apr(&path).unwrap();

        assert_eq!(loaded.config().chunk_size, 512);
        assert_eq!(loaded.config().embedding_dim, 768);
        assert!((loaded.config().rrf_k - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_pattern_store_apr_empty() {
        let store = DecisionPatternStore::new().unwrap();

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("empty.apr");

        store.save_apr(&path).unwrap();
        let loaded = DecisionPatternStore::load_apr(&path).unwrap();

        assert!(loaded.is_empty());
    }

    #[test]
    fn test_pattern_store_apr_suggest_after_load() {
        let mut store = DecisionPatternStore::new().unwrap();

        let pattern = FixPattern::new("E0308", "type fix")
            .with_decision("detect_mismatch");
        store.index_fix(pattern).unwrap();

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("patterns.apr");

        store.save_apr(&path).unwrap();
        let loaded = DecisionPatternStore::load_apr(&path).unwrap();

        // RAG index should be rebuilt and queryable
        let suggestions = loaded
            .suggest_fix("E0308", &["detect_mismatch".into()], 5)
            .unwrap();
        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0].pattern.error_code, "E0308");
    }

    proptest! {
        #[test]
        fn prop_apr_roundtrip_preserves_patterns(
            error_codes in proptest::collection::vec("[A-Z][0-9]{4}", 1..5),
            successes in proptest::collection::vec(0u32..10, 1..5)
        ) {
            let mut store = DecisionPatternStore::new().unwrap();

            for (i, code) in error_codes.iter().enumerate() {
                let mut pattern = FixPattern::new(code, format!("fix{i}"));
                let success_count = successes.get(i).copied().unwrap_or(0);
                for _ in 0..success_count {
                    pattern.record_success();
                }
                store.index_fix(pattern).unwrap();
            }

            let temp_dir = tempfile::tempdir().unwrap();
            let path = temp_dir.path().join("patterns.apr");

            store.save_apr(&path).unwrap();
            let loaded = DecisionPatternStore::load_apr(&path).unwrap();

            prop_assert_eq!(store.len(), loaded.len());

            for code in &error_codes {
                let orig = store.patterns_for_error(code);
                let load = loaded.patterns_for_error(code);
                prop_assert_eq!(orig.len(), load.len());
            }
        }
    }
}
