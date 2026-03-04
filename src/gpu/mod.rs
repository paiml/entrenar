//! GPU resource management: VRAM ledger, guard, and wait queue.
//!
//! Implements GPU sharing spec Phase 1:
//! - **Ledger**: flock-based VRAM reservation with lease expiry (GPU-SHARE-001)
//! - **Guard**: Pre-allocation VRAM check + post-init tracking (GPU-SHARE-002)
//! - **Wait**: Polling wait queue with timeout (GPU-SHARE-003)
//! - **Profiler**: Brick-phase profiler for GPU sharing ops (GPU-SHARE-005)

pub mod cluster;
pub mod coordinator;
pub mod error;
pub mod guard;
pub mod ledger;
pub mod placement;
pub mod profiler;
pub mod wait;
