//! SLERP commutativity tests

use super::helpers::{make_model, models_approx_equal};
use crate::merge::{slerp_merge, SlerpConfig};

#[test]
fn slerp_commutativity_basic() {
    // slerp(A, B, t) = slerp(B, A, 1-t)
    let m1 = make_model(vec![1.0, 2.0, 3.0]);
    let m2 = make_model(vec![4.0, 5.0, 6.0]);

    let t = 0.3;
    let c1 = SlerpConfig::new(t).unwrap();
    let c2 = SlerpConfig::new(1.0 - t).unwrap();

    let r1 = slerp_merge(&m1, &m2, &c1).unwrap();
    let r2 = slerp_merge(&m2, &m1, &c2).unwrap();

    assert!(
        models_approx_equal(&r1, &r2, 1e-4),
        "SLERP should be commutative: slerp(A,B,t) = slerp(B,A,1-t)"
    );
}

#[test]
fn slerp_self_merge_identity() {
    // slerp(A, A, t) = A for any t
    let m = make_model(vec![1.0, 2.0, 3.0, 4.0]);

    for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let config = SlerpConfig::new(t).unwrap();
        let result = slerp_merge(&m, &m, &config).unwrap();
        assert!(models_approx_equal(&result, &m, 1e-5), "slerp(A, A, {t}) should equal A");
    }
}

#[test]
fn slerp_midpoint_symmetry() {
    // slerp(A, B, 0.5) should equal slerp(B, A, 0.5)
    let m1 = make_model(vec![1.0, 0.0, 0.0]);
    let m2 = make_model(vec![0.0, 1.0, 0.0]);

    let config = SlerpConfig::new(0.5).unwrap();

    let r1 = slerp_merge(&m1, &m2, &config).unwrap();
    let r2 = slerp_merge(&m2, &m1, &config).unwrap();

    assert!(models_approx_equal(&r1, &r2, 1e-5), "SLERP at t=0.5 should be symmetric");
}
