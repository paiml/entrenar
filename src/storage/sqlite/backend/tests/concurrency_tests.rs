//! Thread safety tests.

use std::thread;

use crate::storage::sqlite::backend::SqliteBackend;
use crate::storage::{ExperimentStorage, MetricPoint};

#[test]
fn test_concurrent_access() {
    let mut backend = SqliteBackend::open_in_memory().expect("operation should succeed");
    let exp_id = backend.create_experiment("test-exp", None).expect("operation should succeed");
    let run_id = backend.create_run(&exp_id).expect("operation should succeed");

    // Clone state for threads
    let state = backend.state.clone();

    let handles: Vec<_> = (0..10)
        .map(|i| {
            let state = state.clone();
            let run_id = run_id.clone();
            thread::spawn(move || {
                let mut s = state.write().expect("file write should succeed");
                let point = MetricPoint::new(i, i as f64 * 0.1);
                s.metrics
                    .entry(run_id.clone())
                    .or_default()
                    .entry("loss".to_string())
                    .or_default()
                    .push(point);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread join should succeed");
    }

    let metrics = backend.get_metrics(&run_id, "loss").expect("operation should succeed");
    assert_eq!(metrics.len(), 10);
}
