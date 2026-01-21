//! Trace Collectors (ENT-105, ENT-106, ENT-107)
//!
//! Strategies for collecting decision traces:
//! - RingCollector: Stack-allocated, <100ns, for games/drones
//! - StreamCollector: Write-through, <1µs, for persistent logging
//! - HashChainCollector: SHA-256 chain, <10µs, for safety-critical

use super::path::DecisionPath;
use super::trace::DecisionTrace;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Write;

/// Strategy for collecting decision traces
pub trait TraceCollector<P: DecisionPath>: Send + Sync {
    /// Record a decision trace
    fn record(&mut self, trace: DecisionTrace<P>);

    /// Flush any buffered traces
    fn flush(&mut self) -> std::io::Result<()>;

    /// Number of traces recorded
    fn len(&self) -> usize;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// =============================================================================
// RingCollector - Vec-based ring buffer for real-time
// =============================================================================

/// Ring buffer collector with fixed capacity
///
/// Target: <100ns per trace
///
/// # Features
/// - O(1) push operation
/// - Overwrites oldest entries when full
/// - No unsafe code
///
/// # Example
///
/// ```ignore
/// use entrenar::monitor::inference::{RingCollector, LinearPath};
///
/// let mut collector = RingCollector::<LinearPath, 64>::new();
/// collector.record(trace);
/// let recent = collector.recent(10);
/// ```
pub struct RingCollector<P: DecisionPath, const N: usize> {
    buffer: Vec<DecisionTrace<P>>,
    head: usize,
}

impl<P: DecisionPath, const N: usize> RingCollector<P, N> {
    /// Create a new ring collector
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(N),
            head: 0,
        }
    }

    /// Get the most recent n traces (or all if n > count)
    pub fn recent(&self, n: usize) -> Vec<&DecisionTrace<P>> {
        let take = n.min(self.buffer.len());
        let mut result = Vec::with_capacity(take);

        for i in 0..take {
            let idx = if self.buffer.len() < N {
                // Not yet wrapped
                self.buffer.len() - 1 - i
            } else {
                // Wrapped: head points to next write, so head-1 is most recent
                (self.head + N - 1 - i) % N
            };
            result.push(&self.buffer[idx]);
        }

        result
    }

    /// Get all traces in order (oldest first)
    pub fn all(&self) -> Vec<&DecisionTrace<P>> {
        let mut result = Vec::with_capacity(self.buffer.len());

        if self.buffer.is_empty() {
            return result;
        }

        if self.buffer.len() < N {
            // Not yet wrapped - just iterate in order
            for trace in &self.buffer {
                result.push(trace);
            }
        } else {
            // Wrapped: head is the oldest
            for i in 0..N {
                let idx = (self.head + i) % N;
                result.push(&self.buffer[idx]);
            }
        }

        result
    }

    /// Get the last trace if any
    pub fn last(&self) -> Option<&DecisionTrace<P>> {
        if self.buffer.is_empty() {
            return None;
        }
        if self.buffer.len() < N {
            self.buffer.last()
        } else {
            let idx = (self.head + N - 1) % N;
            Some(&self.buffer[idx])
        }
    }

    /// Clear all traces
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.head = 0;
    }

    /// Capacity of the ring buffer
    pub const fn capacity(&self) -> usize {
        N
    }
}

impl<P: DecisionPath, const N: usize> TraceCollector<P> for RingCollector<P, N> {
    fn record(&mut self, trace: DecisionTrace<P>) {
        if self.buffer.len() < N {
            // Buffer not yet full, just push
            self.buffer.push(trace);
        } else {
            // Buffer full, overwrite oldest
            self.buffer[self.head] = trace;
            self.head = (self.head + 1) % N;
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        // Ring buffer doesn't need flushing
        Ok(())
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }
}

impl<P: DecisionPath, const N: usize> Default for RingCollector<P, N> {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// StreamCollector - Write-through for persistent logging
// =============================================================================

/// Trace format for serialization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamFormat {
    /// Binary format (compact, fast)
    Binary,
    /// JSON format (human-readable)
    Json,
    /// JSON Lines (one JSON object per line)
    JsonLines,
}

/// Stream collector for persistent logging
///
/// Target: <1µs per trace
///
/// # Features
/// - Write-through to any `Write` impl
/// - Supports binary and JSON formats
/// - Buffered writes for efficiency
///
/// # Example
///
/// ```ignore
/// use entrenar::monitor::inference::{StreamCollector, LinearPath, StreamFormat};
/// use std::fs::File;
///
/// let file = File::create("traces.jsonl")?;
/// let mut collector = StreamCollector::<LinearPath, _>::new(file, StreamFormat::JsonLines);
/// collector.record(trace);
/// collector.flush()?;
/// ```
pub struct StreamCollector<P: DecisionPath, W: Write + Send> {
    writer: W,
    format: StreamFormat,
    buffer: Vec<DecisionTrace<P>>,
    flush_threshold: usize,
    count: usize,
}

impl<P: DecisionPath + Serialize, W: Write + Send + Sync> StreamCollector<P, W> {
    /// Create a new stream collector
    pub fn new(writer: W, format: StreamFormat) -> Self {
        Self {
            writer,
            format,
            buffer: Vec::with_capacity(100),
            flush_threshold: 100,
            count: 0,
        }
    }

    /// Set the flush threshold (number of traces before auto-flush)
    pub fn with_flush_threshold(mut self, threshold: usize) -> Self {
        self.flush_threshold = threshold;
        self
    }

    /// Get reference to the underlying writer
    pub fn writer(&self) -> &W {
        &self.writer
    }

    /// Get mutable reference to the underlying writer
    pub fn writer_mut(&mut self) -> &mut W {
        &mut self.writer
    }

    /// Write a single trace
    fn write_trace(&mut self, trace: &DecisionTrace<P>) -> std::io::Result<()> {
        match self.format {
            StreamFormat::Binary => {
                let bytes = trace.to_bytes();
                // Write length prefix
                self.writer.write_all(&(bytes.len() as u32).to_le_bytes())?;
                self.writer.write_all(&bytes)?;
            }
            StreamFormat::Json => {
                serde_json::to_writer(&mut self.writer, trace)?;
            }
            StreamFormat::JsonLines => {
                serde_json::to_writer(&mut self.writer, trace)?;
                self.writer.write_all(b"\n")?;
            }
        }
        Ok(())
    }
}

impl<P: DecisionPath + Serialize, W: Write + Send + Sync> TraceCollector<P>
    for StreamCollector<P, W>
{
    fn record(&mut self, trace: DecisionTrace<P>) {
        self.buffer.push(trace);
        self.count += 1;

        if self.buffer.len() >= self.flush_threshold {
            let _ = self.flush();
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let traces: Vec<_> = self.buffer.drain(..).collect();
        for trace in traces {
            self.write_trace(&trace)?;
        }
        self.writer.flush()
    }

    fn len(&self) -> usize {
        self.count
    }
}

// =============================================================================
// HashChainCollector - Tamper-evident for safety-critical
// =============================================================================

/// A single entry in the hash chain
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChainEntry<P: DecisionPath> {
    /// Monotonic sequence number
    pub sequence: u64,
    /// Hash of previous entry (SHA-256)
    pub prev_hash: [u8; 32],
    /// The decision trace
    pub trace: DecisionTrace<P>,
    /// SHA-256(sequence || prev_hash || trace)
    pub hash: [u8; 32],
}

/// Chain verification result
#[derive(Debug, Clone)]
pub struct ChainVerification {
    /// Whether the chain is valid
    pub valid: bool,
    /// Number of entries verified
    pub entries_verified: usize,
    /// First broken link index (if any)
    pub first_break: Option<usize>,
    /// Error message (if any)
    pub error: Option<String>,
}

/// Hash-chain collector for safety-critical systems
///
/// Target: <10µs per entry
///
/// # Features
/// - SHA-256 hash chain for tamper evidence
/// - Chain verification on load
/// - Append-only storage
///
/// # Example
///
/// ```ignore
/// use entrenar::monitor::inference::{HashChainCollector, LinearPath};
///
/// let mut collector = HashChainCollector::<LinearPath>::new();
/// collector.record(trace);
///
/// let verification = collector.verify_chain();
/// assert!(verification.valid);
/// ```
pub struct HashChainCollector<P: DecisionPath> {
    entries: Vec<ChainEntry<P>>,
    prev_hash: [u8; 32],
    sequence: u64,
}

impl<P: DecisionPath + Serialize> HashChainCollector<P> {
    /// Create a new hash-chain collector
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            prev_hash: [0u8; 32], // Genesis block has zero hash
            sequence: 0,
        }
    }

    /// Compute hash of an entry
    fn compute_hash(sequence: u64, prev_hash: &[u8; 32], trace: &DecisionTrace<P>) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(sequence.to_le_bytes());
        hasher.update(prev_hash);
        hasher.update(trace.to_bytes());
        hasher.finalize().into()
    }

    /// Verify the entire chain
    pub fn verify_chain(&self) -> ChainVerification {
        if self.entries.is_empty() {
            return ChainVerification {
                valid: true,
                entries_verified: 0,
                first_break: None,
                error: None,
            };
        }

        let mut prev_hash = [0u8; 32]; // Genesis

        for (i, entry) in self.entries.iter().enumerate() {
            // Check sequence
            if entry.sequence != i as u64 {
                return ChainVerification {
                    valid: false,
                    entries_verified: i,
                    first_break: Some(i),
                    error: Some(format!(
                        "Sequence mismatch at index {}: expected {}, got {}",
                        i, i, entry.sequence
                    )),
                };
            }

            // Check prev_hash
            if entry.prev_hash != prev_hash {
                return ChainVerification {
                    valid: false,
                    entries_verified: i,
                    first_break: Some(i),
                    error: Some(format!("Previous hash mismatch at index {i}")),
                };
            }

            // Verify hash
            let computed_hash = Self::compute_hash(entry.sequence, &prev_hash, &entry.trace);
            if entry.hash != computed_hash {
                return ChainVerification {
                    valid: false,
                    entries_verified: i,
                    first_break: Some(i),
                    error: Some(format!("Hash mismatch at index {i}")),
                };
            }

            prev_hash = entry.hash;
        }

        ChainVerification {
            valid: true,
            entries_verified: self.entries.len(),
            first_break: None,
            error: None,
        }
    }

    /// Get all entries
    pub fn entries(&self) -> &[ChainEntry<P>] {
        &self.entries
    }

    /// Get entry by sequence number
    pub fn get(&self, sequence: u64) -> Option<&ChainEntry<P>> {
        self.entries.get(sequence as usize)
    }

    /// Get the latest hash
    pub fn latest_hash(&self) -> [u8; 32] {
        self.entries.last().map_or([0u8; 32], |e| e.hash)
    }

    /// Export chain to JSON
    pub fn to_json(&self) -> serde_json::Result<String>
    where
        P: Serialize,
    {
        serde_json::to_string_pretty(&self.entries)
    }
}

impl<P: DecisionPath + Serialize> TraceCollector<P> for HashChainCollector<P> {
    fn record(&mut self, trace: DecisionTrace<P>) {
        let hash = Self::compute_hash(self.sequence, &self.prev_hash, &trace);

        let entry = ChainEntry {
            sequence: self.sequence,
            prev_hash: self.prev_hash,
            trace,
            hash,
        };

        self.prev_hash = hash;
        self.sequence += 1;
        self.entries.push(entry);
    }

    fn flush(&mut self) -> std::io::Result<()> {
        // Hash chain is always "flushed" (in memory)
        Ok(())
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

impl<P: DecisionPath + Serialize> Default for HashChainCollector<P> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::inference::path::LinearPath;

    fn make_test_trace(seq: u64) -> DecisionTrace<LinearPath> {
        let path = LinearPath::new(vec![0.5, -0.3], 0.1, 0.5, 0.87);
        DecisionTrace::new(seq * 1000, seq, seq, path, 0.87, 100)
    }

    // ==========================================================================
    // RingCollector tests
    // ==========================================================================

    #[test]
    fn test_ring_collector_new() {
        let collector = RingCollector::<LinearPath, 64>::new();
        assert_eq!(collector.len(), 0);
        assert!(collector.is_empty());
        assert_eq!(collector.capacity(), 64);
    }

    #[test]
    fn test_ring_collector_record() {
        let mut collector = RingCollector::<LinearPath, 64>::new();
        collector.record(make_test_trace(0));
        assert_eq!(collector.len(), 1);
        assert!(!collector.is_empty());
    }

    #[test]
    fn test_ring_collector_wraparound() {
        let mut collector = RingCollector::<LinearPath, 4>::new();

        // Fill beyond capacity
        for i in 0..6 {
            collector.record(make_test_trace(i));
        }

        assert_eq!(collector.len(), 4);

        // Check we have the last 4 entries
        let all = collector.all();
        assert_eq!(all.len(), 4);
        assert_eq!(all[0].sequence, 2);
        assert_eq!(all[1].sequence, 3);
        assert_eq!(all[2].sequence, 4);
        assert_eq!(all[3].sequence, 5);
    }

    #[test]
    fn test_ring_collector_recent() {
        let mut collector = RingCollector::<LinearPath, 8>::new();

        for i in 0..5 {
            collector.record(make_test_trace(i));
        }

        let recent = collector.recent(3);
        assert_eq!(recent.len(), 3);
        // Most recent first
        assert_eq!(recent[0].sequence, 4);
        assert_eq!(recent[1].sequence, 3);
        assert_eq!(recent[2].sequence, 2);
    }

    #[test]
    fn test_ring_collector_last() {
        let mut collector = RingCollector::<LinearPath, 4>::new();

        assert!(collector.last().is_none());

        collector.record(make_test_trace(42));
        assert_eq!(collector.last().unwrap().sequence, 42);

        collector.record(make_test_trace(43));
        assert_eq!(collector.last().unwrap().sequence, 43);
    }

    #[test]
    fn test_ring_collector_clear() {
        let mut collector = RingCollector::<LinearPath, 4>::new();

        for i in 0..3 {
            collector.record(make_test_trace(i));
        }

        collector.clear();
        assert_eq!(collector.len(), 0);
        assert!(collector.is_empty());
    }

    #[test]
    fn test_ring_collector_flush() {
        let mut collector = RingCollector::<LinearPath, 4>::new();
        assert!(collector.flush().is_ok());
    }

    // ==========================================================================
    // StreamCollector tests
    // ==========================================================================

    #[test]
    fn test_stream_collector_json_lines() {
        let mut buffer = Vec::new();
        {
            let mut collector =
                StreamCollector::<LinearPath, _>::new(&mut buffer, StreamFormat::JsonLines)
                    .with_flush_threshold(10);

            collector.record(make_test_trace(0));
            collector.record(make_test_trace(1));
            collector.flush().unwrap();
        }

        let output = String::from_utf8(buffer).unwrap();
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 2);

        // Each line should be valid JSON
        for line in lines {
            assert!(serde_json::from_str::<serde_json::Value>(line).is_ok());
        }
    }

    #[test]
    fn test_stream_collector_binary() {
        let mut buffer = Vec::new();
        {
            let mut collector =
                StreamCollector::<LinearPath, _>::new(&mut buffer, StreamFormat::Binary);

            collector.record(make_test_trace(0));
            collector.flush().unwrap();
        }

        // Should have length prefix + trace bytes
        assert!(buffer.len() > 4);

        // Verify length prefix
        let len = u32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]) as usize;
        assert_eq!(buffer.len(), 4 + len);
    }

    #[test]
    fn test_stream_collector_auto_flush() {
        use std::sync::{Arc, Mutex};

        let buffer = Arc::new(Mutex::new(Vec::new()));
        {
            let buf_clone = Arc::clone(&buffer);
            let mut collector = StreamCollector::<LinearPath, _>::new(
                SyncWriter(buf_clone),
                StreamFormat::JsonLines,
            )
            .with_flush_threshold(2);

            collector.record(make_test_trace(0));
            // Should not have written yet (buffered)
            assert!(buffer.lock().unwrap().is_empty());

            collector.record(make_test_trace(1));
            // Should have auto-flushed at threshold
        }

        // After drop, should have data
        assert!(!buffer.lock().unwrap().is_empty());
    }

    /// Wrapper to make Arc<Mutex<Vec<u8>>> implement Write + Sync
    struct SyncWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl Write for SyncWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().write(buf)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.0.lock().unwrap().flush()
        }
    }

    #[test]
    fn test_stream_collector_len() {
        let buffer = Vec::new();
        let mut collector = StreamCollector::<LinearPath, _>::new(buffer, StreamFormat::JsonLines);

        assert_eq!(collector.len(), 0);
        collector.record(make_test_trace(0));
        assert_eq!(collector.len(), 1);
        collector.record(make_test_trace(1));
        assert_eq!(collector.len(), 2);
    }

    // ==========================================================================
    // HashChainCollector tests
    // ==========================================================================

    #[test]
    fn test_hash_chain_new() {
        let collector = HashChainCollector::<LinearPath>::new();
        assert_eq!(collector.len(), 0);
        assert!(collector.is_empty());
        assert_eq!(collector.latest_hash(), [0u8; 32]);
    }

    #[test]
    fn test_hash_chain_record() {
        let mut collector = HashChainCollector::<LinearPath>::new();

        collector.record(make_test_trace(0));
        assert_eq!(collector.len(), 1);
        assert_ne!(collector.latest_hash(), [0u8; 32]);

        let entry = collector.get(0).unwrap();
        assert_eq!(entry.sequence, 0);
        assert_eq!(entry.prev_hash, [0u8; 32]); // Genesis
    }

    #[test]
    fn test_hash_chain_linking() {
        let mut collector = HashChainCollector::<LinearPath>::new();

        collector.record(make_test_trace(0));
        let first_hash = collector.latest_hash();

        collector.record(make_test_trace(1));

        let second_entry = collector.get(1).unwrap();
        assert_eq!(second_entry.prev_hash, first_hash);
    }

    #[test]
    fn test_hash_chain_verify_empty() {
        let collector = HashChainCollector::<LinearPath>::new();
        let verification = collector.verify_chain();
        assert!(verification.valid);
        assert_eq!(verification.entries_verified, 0);
    }

    #[test]
    fn test_hash_chain_verify_valid() {
        let mut collector = HashChainCollector::<LinearPath>::new();

        for i in 0..10 {
            collector.record(make_test_trace(i));
        }

        let verification = collector.verify_chain();
        assert!(verification.valid);
        assert_eq!(verification.entries_verified, 10);
        assert!(verification.first_break.is_none());
    }

    #[test]
    fn test_hash_chain_detect_tampering() {
        let mut collector = HashChainCollector::<LinearPath>::new();

        for i in 0..5 {
            collector.record(make_test_trace(i));
        }

        // Tamper with an entry
        if let Some(entry) = collector.entries.get_mut(2) {
            entry.trace.output = 999.0; // Change the output
        }

        let verification = collector.verify_chain();
        assert!(!verification.valid);
        assert_eq!(verification.first_break, Some(2));
        assert!(verification.error.is_some());
    }

    #[test]
    fn test_hash_chain_detect_sequence_tampering() {
        let mut collector = HashChainCollector::<LinearPath>::new();

        for i in 0..3 {
            collector.record(make_test_trace(i));
        }

        // Tamper with sequence
        if let Some(entry) = collector.entries.get_mut(1) {
            entry.sequence = 999;
        }

        let verification = collector.verify_chain();
        assert!(!verification.valid);
        assert_eq!(verification.first_break, Some(1));
    }

    #[test]
    fn test_hash_chain_to_json() {
        let mut collector = HashChainCollector::<LinearPath>::new();
        collector.record(make_test_trace(0));

        let json = collector.to_json().unwrap();
        assert!(json.contains("sequence"));
        assert!(json.contains("prev_hash"));
        assert!(json.contains("hash"));
    }

    #[test]
    fn test_hash_chain_flush() {
        let mut collector = HashChainCollector::<LinearPath>::new();
        assert!(collector.flush().is_ok());
    }

    #[test]
    fn test_hash_chain_entries() {
        let mut collector = HashChainCollector::<LinearPath>::new();

        for i in 0..3 {
            collector.record(make_test_trace(i));
        }

        let entries = collector.entries();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].sequence, 0);
        assert_eq!(entries[1].sequence, 1);
        assert_eq!(entries[2].sequence, 2);
    }
}
