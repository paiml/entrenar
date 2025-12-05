//! Subword Tokenization Module (#26)
//!
//! Just-in-Time tokenization for training pipelines with BPE and WordPiece support.
//!
//! # Toyota Principle: Just-in-Time (ジャスト・イン・タイム)
//!
//! Tokenize on demand during training, not upfront - reducing memory footprint
//! and enabling dynamic vocabulary adaptation.
//!
//! # Example
//!
//! ```
//! use entrenar::tokenizer::{BPETokenizer, Tokenizer, TokenizerConfig};
//!
//! // Create a BPE tokenizer
//! let config = TokenizerConfig::bpe().with_vocab_size(1000);
//! let mut tokenizer = BPETokenizer::new(config);
//!
//! // Train on corpus
//! let corpus = vec!["hello world", "hello there"];
//! tokenizer.train(&corpus).unwrap();
//!
//! // Tokenize text
//! let tokens = tokenizer.encode("hello world").unwrap();
//! let decoded = tokenizer.decode(&tokens).unwrap();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Tokenizer errors
#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("Vocabulary not trained")]
    NotTrained,

    #[error("Unknown token: {0}")]
    UnknownToken(String),

    #[error("Invalid token ID: {0}")]
    InvalidTokenId(u32),

    #[error("Training error: {0}")]
    Training(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Result type for tokenizer operations
pub type Result<T> = std::result::Result<T, TokenizerError>;

/// Token ID type
pub type TokenId = u32;

/// Special tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    /// Unknown token
    pub unk: String,
    /// Beginning of sequence
    pub bos: String,
    /// End of sequence
    pub eos: String,
    /// Padding token
    pub pad: String,
    /// Mask token (for MLM)
    pub mask: String,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            unk: "<unk>".to_string(),
            bos: "<s>".to_string(),
            eos: "</s>".to_string(),
            pad: "<pad>".to_string(),
            mask: "<mask>".to_string(),
        }
    }
}

/// Tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Target vocabulary size
    pub vocab_size: usize,
    /// Minimum token frequency for training
    pub min_frequency: usize,
    /// Special tokens
    pub special_tokens: SpecialTokens,
    /// Whether to lowercase input
    pub lowercase: bool,
    /// Tokenizer type
    pub tokenizer_type: TokenizerType,
}

/// Tokenizer type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenizerType {
    /// Byte Pair Encoding
    BPE,
    /// WordPiece (BERT-style)
    WordPiece,
    /// Character-level
    Char,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            min_frequency: 2,
            special_tokens: SpecialTokens::default(),
            lowercase: false,
            tokenizer_type: TokenizerType::BPE,
        }
    }
}

impl TokenizerConfig {
    /// Create a BPE tokenizer config
    pub fn bpe() -> Self {
        Self {
            tokenizer_type: TokenizerType::BPE,
            ..Default::default()
        }
    }

    /// Create a WordPiece tokenizer config
    pub fn wordpiece() -> Self {
        Self {
            tokenizer_type: TokenizerType::WordPiece,
            ..Default::default()
        }
    }

    /// Create a character-level tokenizer config
    pub fn char() -> Self {
        Self {
            tokenizer_type: TokenizerType::Char,
            vocab_size: 256,
            ..Default::default()
        }
    }

    /// Set vocabulary size
    pub fn with_vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = size;
        self
    }

    /// Set minimum frequency
    pub fn with_min_frequency(mut self, freq: usize) -> Self {
        self.min_frequency = freq;
        self
    }

    /// Enable lowercase preprocessing
    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }
}

/// Tokenizer trait
pub trait Tokenizer: Send + Sync {
    /// Train the tokenizer on a corpus
    fn train(&mut self, corpus: &[&str]) -> Result<()>;

    /// Encode text to token IDs
    fn encode(&self, text: &str) -> Result<Vec<TokenId>>;

    /// Decode token IDs to text
    fn decode(&self, ids: &[TokenId]) -> Result<String>;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Check if tokenizer is trained
    fn is_trained(&self) -> bool;

    /// Get token for ID
    fn id_to_token(&self, id: TokenId) -> Option<&str>;

    /// Get ID for token
    fn token_to_id(&self, token: &str) -> Option<TokenId>;
}

/// BPE (Byte Pair Encoding) tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPETokenizer {
    config: TokenizerConfig,
    /// Token to ID mapping
    vocab: HashMap<String, TokenId>,
    /// ID to token mapping
    id_to_token_map: HashMap<TokenId, String>,
    /// Merge rules (pair -> merged token)
    merges: Vec<(String, String)>,
    /// Whether the tokenizer is trained
    trained: bool,
}

impl BPETokenizer {
    /// Create a new BPE tokenizer
    pub fn new(config: TokenizerConfig) -> Self {
        Self {
            config,
            vocab: HashMap::new(),
            id_to_token_map: HashMap::new(),
            merges: Vec::new(),
            trained: false,
        }
    }

    /// Initialize vocabulary with special tokens and bytes
    fn init_vocab(&mut self) {
        let mut id: TokenId = 0;

        // Add special tokens
        let special = [
            &self.config.special_tokens.unk,
            &self.config.special_tokens.bos,
            &self.config.special_tokens.eos,
            &self.config.special_tokens.pad,
            &self.config.special_tokens.mask,
        ];

        for token in special {
            self.vocab.insert(token.clone(), id);
            self.id_to_token_map.insert(id, token.clone());
            id += 1;
        }

        // Add all single bytes as base vocabulary
        for byte in 0..=255u8 {
            let token = format!("{byte:02x}");
            if !self.vocab.contains_key(&token) {
                self.vocab.insert(token.clone(), id);
                self.id_to_token_map.insert(id, token);
                id += 1;
            }
        }
    }

    /// Get pair frequencies from tokenized corpus
    fn get_pair_freqs(&self, tokenized: &[Vec<String>]) -> HashMap<(String, String), usize> {
        let mut freqs = HashMap::new();

        for tokens in tokenized {
            for pair in tokens.windows(2) {
                let key = (pair[0].clone(), pair[1].clone());
                *freqs.entry(key).or_insert(0) += 1;
            }
        }

        freqs
    }

    /// Merge the most frequent pair
    fn merge_pair(&self, tokenized: &mut [Vec<String>], pair: &(String, String), merged: &str) {
        for tokens in tokenized.iter_mut() {
            let mut i = 0;
            while i < tokens.len().saturating_sub(1) {
                if tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                    tokens[i] = merged.to_string();
                    tokens.remove(i + 1);
                }
                i += 1;
            }
        }
    }

    /// Tokenize text to bytes (initial tokenization)
    fn to_bytes(&self, text: &str) -> Vec<String> {
        text.as_bytes().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Apply all learned merges
    fn apply_merges(&self, mut tokens: Vec<String>) -> Vec<String> {
        for (a, b) in &self.merges {
            let merged = format!("{a}{b}");
            let mut i = 0;
            while i < tokens.len().saturating_sub(1) {
                if &tokens[i] == a && &tokens[i + 1] == b {
                    tokens[i] = merged.clone();
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }
        tokens
    }

    /// Save tokenizer to file
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| TokenizerError::Serialization(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load tokenizer from file
    pub fn load(path: &str) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(|e| TokenizerError::Serialization(e.to_string()))
    }
}

impl Tokenizer for BPETokenizer {
    fn train(&mut self, corpus: &[&str]) -> Result<()> {
        self.init_vocab();

        // Tokenize corpus to bytes
        let mut tokenized: Vec<Vec<String>> = corpus
            .iter()
            .map(|text| {
                let t = if self.config.lowercase {
                    text.to_lowercase()
                } else {
                    text.to_string()
                };
                self.to_bytes(&t)
            })
            .collect();

        // Learn merges until we reach target vocab size
        let target = self.config.vocab_size;
        while self.vocab.len() < target {
            let freqs = self.get_pair_freqs(&tokenized);

            // Find most frequent pair
            let best = freqs
                .iter()
                .filter(|(_, &count)| count >= self.config.min_frequency)
                .max_by_key(|(_, count)| *count);

            match best {
                Some((pair, _)) => {
                    let merged = format!("{}{}", pair.0, pair.1);

                    // Add to vocabulary
                    let id = self.vocab.len() as TokenId;
                    self.vocab.insert(merged.clone(), id);
                    self.id_to_token_map.insert(id, merged.clone());

                    // Record merge
                    self.merges.push(pair.clone());

                    // Apply merge
                    self.merge_pair(&mut tokenized, pair, &merged);
                }
                None => break, // No more pairs meet frequency threshold
            }
        }

        self.trained = true;
        Ok(())
    }

    fn encode(&self, text: &str) -> Result<Vec<TokenId>> {
        if !self.trained {
            return Err(TokenizerError::NotTrained);
        }

        let processed = if self.config.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        let tokens = self.to_bytes(&processed);
        let tokens = self.apply_merges(tokens);

        let unk_id = *self.vocab.get(&self.config.special_tokens.unk).unwrap();

        let ids: Vec<TokenId> = tokens
            .iter()
            .map(|t| *self.vocab.get(t).unwrap_or(&unk_id))
            .collect();

        Ok(ids)
    }

    fn decode(&self, ids: &[TokenId]) -> Result<String> {
        if !self.trained {
            return Err(TokenizerError::NotTrained);
        }

        let mut hex_string = String::new();

        for &id in ids {
            if let Some(token) = self.id_to_token_map.get(&id) {
                // Skip special tokens
                if token.starts_with('<') && token.ends_with('>') {
                    continue;
                }
                hex_string.push_str(token);
            }
        }

        // Convert hex string back to bytes
        let bytes: Vec<u8> = (0..hex_string.len())
            .step_by(2)
            .filter_map(|i| {
                if i + 2 <= hex_string.len() {
                    u8::from_str_radix(&hex_string[i..i + 2], 16).ok()
                } else {
                    None
                }
            })
            .collect();

        String::from_utf8(bytes).map_err(|e| TokenizerError::Training(e.to_string()))
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn id_to_token(&self, id: TokenId) -> Option<&str> {
        self.id_to_token_map.get(&id).map(String::as_str)
    }

    fn token_to_id(&self, token: &str) -> Option<TokenId> {
        self.vocab.get(token).copied()
    }
}

/// Character-level tokenizer (simple baseline)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharTokenizer {
    config: TokenizerConfig,
    vocab: HashMap<char, TokenId>,
    id_to_char: HashMap<TokenId, char>,
    trained: bool,
}

impl CharTokenizer {
    /// Create a new character tokenizer
    pub fn new(config: TokenizerConfig) -> Self {
        Self {
            config,
            vocab: HashMap::new(),
            id_to_char: HashMap::new(),
            trained: false,
        }
    }
}

impl Tokenizer for CharTokenizer {
    fn train(&mut self, corpus: &[&str]) -> Result<()> {
        let mut id: TokenId = 0;

        // Count character frequencies
        let mut char_counts: HashMap<char, usize> = HashMap::new();
        for text in corpus {
            let processed = if self.config.lowercase {
                text.to_lowercase()
            } else {
                text.to_string()
            };
            for c in processed.chars() {
                *char_counts.entry(c).or_insert(0) += 1;
            }
        }

        // Sort by frequency and take top vocab_size
        let mut chars: Vec<_> = char_counts.into_iter().collect();
        chars.sort_by(|a, b| b.1.cmp(&a.1));

        for (c, count) in chars.into_iter().take(self.config.vocab_size) {
            if count >= self.config.min_frequency {
                self.vocab.insert(c, id);
                self.id_to_char.insert(id, c);
                id += 1;
            }
        }

        self.trained = true;
        Ok(())
    }

    fn encode(&self, text: &str) -> Result<Vec<TokenId>> {
        if !self.trained {
            return Err(TokenizerError::NotTrained);
        }

        let processed = if self.config.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        let mut ids = Vec::new();
        for c in processed.chars() {
            if let Some(&id) = self.vocab.get(&c) {
                ids.push(id);
            }
            // Unknown characters are skipped
        }

        Ok(ids)
    }

    fn decode(&self, ids: &[TokenId]) -> Result<String> {
        if !self.trained {
            return Err(TokenizerError::NotTrained);
        }

        let mut result = String::new();
        for &id in ids {
            if let Some(&c) = self.id_to_char.get(&id) {
                result.push(c);
            }
        }

        Ok(result)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn id_to_token(&self, _id: TokenId) -> Option<&str> {
        // Characters are not stored as strings
        None
    }

    fn token_to_id(&self, token: &str) -> Option<TokenId> {
        if token.len() == 1 {
            self.vocab.get(&token.chars().next().unwrap()).copied()
        } else {
            None
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_config_default() {
        let config = TokenizerConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.tokenizer_type, TokenizerType::BPE);
    }

    #[test]
    fn test_tokenizer_config_bpe() {
        let config = TokenizerConfig::bpe().with_vocab_size(1000);
        assert_eq!(config.vocab_size, 1000);
        assert_eq!(config.tokenizer_type, TokenizerType::BPE);
    }

    #[test]
    fn test_tokenizer_config_wordpiece() {
        let config = TokenizerConfig::wordpiece();
        assert_eq!(config.tokenizer_type, TokenizerType::WordPiece);
    }

    #[test]
    fn test_tokenizer_config_char() {
        let config = TokenizerConfig::char();
        assert_eq!(config.tokenizer_type, TokenizerType::Char);
        assert_eq!(config.vocab_size, 256);
    }

    #[test]
    fn test_special_tokens_default() {
        let special = SpecialTokens::default();
        assert_eq!(special.unk, "<unk>");
        assert_eq!(special.bos, "<s>");
        assert_eq!(special.eos, "</s>");
    }

    // -------------------------------------------------------------------------
    // BPE Tokenizer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bpe_new() {
        let config = TokenizerConfig::bpe();
        let tokenizer = BPETokenizer::new(config);
        assert!(!tokenizer.is_trained());
    }

    #[test]
    fn test_bpe_train() {
        let config = TokenizerConfig::bpe()
            .with_vocab_size(300)
            .with_min_frequency(1);
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["hello hello", "hello world", "world hello"];
        tokenizer.train(&corpus).unwrap();

        assert!(tokenizer.is_trained());
        assert!(tokenizer.vocab_size() > 256); // Base bytes + some merges
    }

    #[test]
    fn test_bpe_encode_not_trained() {
        let config = TokenizerConfig::bpe();
        let tokenizer = BPETokenizer::new(config);

        let result = tokenizer.encode("hello");
        assert!(result.is_err());
    }

    #[test]
    fn test_bpe_encode_decode() {
        let config = TokenizerConfig::bpe()
            .with_vocab_size(300)
            .with_min_frequency(1);
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["hello world", "hello there"];
        tokenizer.train(&corpus).unwrap();

        let text = "hello";
        let encoded = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&encoded).unwrap();

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_bpe_lowercase() {
        let config = TokenizerConfig::bpe()
            .with_vocab_size(300)
            .with_min_frequency(1)
            .with_lowercase(true);
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["Hello World"];
        tokenizer.train(&corpus).unwrap();

        let encoded = tokenizer.encode("HELLO").unwrap();
        let decoded = tokenizer.decode(&encoded).unwrap();

        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_bpe_id_to_token() {
        let config = TokenizerConfig::bpe()
            .with_vocab_size(300)
            .with_min_frequency(1);
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["test"];
        tokenizer.train(&corpus).unwrap();

        // ID 0 should be <unk>
        assert_eq!(tokenizer.id_to_token(0), Some("<unk>"));
    }

    #[test]
    fn test_bpe_token_to_id() {
        let config = TokenizerConfig::bpe()
            .with_vocab_size(300)
            .with_min_frequency(1);
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["test"];
        tokenizer.train(&corpus).unwrap();

        assert_eq!(tokenizer.token_to_id("<unk>"), Some(0));
    }

    // -------------------------------------------------------------------------
    // Character Tokenizer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_char_new() {
        let config = TokenizerConfig::char();
        let tokenizer = CharTokenizer::new(config);
        assert!(!tokenizer.is_trained());
    }

    #[test]
    fn test_char_train() {
        let config = TokenizerConfig::char().with_min_frequency(1);
        let mut tokenizer = CharTokenizer::new(config);

        let corpus = vec!["hello", "world"];
        tokenizer.train(&corpus).unwrap();

        assert!(tokenizer.is_trained());
        // h, e, l, o, w, r, d = 7 unique chars
        assert_eq!(tokenizer.vocab_size(), 7);
    }

    #[test]
    fn test_char_encode_decode() {
        let config = TokenizerConfig::char().with_min_frequency(1);
        let mut tokenizer = CharTokenizer::new(config);

        let corpus = vec!["hello"];
        tokenizer.train(&corpus).unwrap();

        let text = "hello";
        let encoded = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&encoded).unwrap();

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_char_unknown_chars() {
        let config = TokenizerConfig::char().with_min_frequency(1);
        let mut tokenizer = CharTokenizer::new(config);

        let corpus = vec!["abc"];
        tokenizer.train(&corpus).unwrap();

        // 'x' is not in vocabulary, should be skipped
        let encoded = tokenizer.encode("axbc").unwrap();
        let decoded = tokenizer.decode(&encoded).unwrap();

        assert_eq!(decoded, "abc");
    }

    #[test]
    fn test_char_lowercase() {
        let config = TokenizerConfig::char()
            .with_min_frequency(1)
            .with_lowercase(true);
        let mut tokenizer = CharTokenizer::new(config);

        let corpus = vec!["Hello"];
        tokenizer.train(&corpus).unwrap();

        let encoded = tokenizer.encode("HELLO").unwrap();
        let decoded = tokenizer.decode(&encoded).unwrap();

        assert_eq!(decoded, "hello");
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
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_char_roundtrip(text in "[a-z]{1,20}") {
            let config = TokenizerConfig::char().with_min_frequency(1);
            let mut tokenizer = CharTokenizer::new(config);
            tokenizer.train(&[&text]).unwrap();

            let encoded = tokenizer.encode(&text).unwrap();
            let decoded = tokenizer.decode(&encoded).unwrap();

            prop_assert_eq!(decoded, text);
        }

        #[test]
        fn prop_bpe_encode_produces_valid_ids(text in "[a-zA-Z ]{1,20}") {
            let config = TokenizerConfig::bpe()
                .with_vocab_size(300)
                .with_min_frequency(1);
            let mut tokenizer = BPETokenizer::new(config);
            tokenizer.train(&[&text]).unwrap();

            let encoded = tokenizer.encode(&text).unwrap();

            for id in encoded {
                prop_assert!(tokenizer.id_to_token(id).is_some());
            }
        }

        #[test]
        fn prop_vocab_size_bounded(target_size in 261usize..500) {
            let config = TokenizerConfig::bpe()
                .with_vocab_size(target_size)
                .with_min_frequency(1);
            let mut tokenizer = BPETokenizer::new(config);

            let corpus = vec!["hello world hello world test test"];
            tokenizer.train(&corpus).unwrap();

            prop_assert!(tokenizer.vocab_size() <= target_size);
        }

        #[test]
        fn prop_char_vocab_size_matches_unique_chars(text in "[a-z]{5,30}") {
            let config = TokenizerConfig::char()
                .with_min_frequency(1)
                .with_vocab_size(256);
            let mut tokenizer = CharTokenizer::new(config);
            tokenizer.train(&[&text]).unwrap();

            let unique_chars: std::collections::HashSet<char> = text.chars().collect();
            prop_assert_eq!(tokenizer.vocab_size(), unique_chars.len());
        }
    }
}
