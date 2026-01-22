//! Test helper functions for commutativity tests

use crate::autograd::Tensor;
use crate::merge::Model;
use std::collections::HashMap;

/// Create a model with single parameter
pub fn make_model(values: Vec<f32>) -> Model {
    let mut m = HashMap::new();
    m.insert("w".to_string(), Tensor::from_vec(values, false));
    m
}

/// Create a model with multiple parameters
pub fn make_multi_param_model(params: Vec<(&str, Vec<f32>)>) -> Model {
    let mut m = HashMap::new();
    for (name, values) in params {
        m.insert(name.to_string(), Tensor::from_vec(values, false));
    }
    m
}

/// Compare two models for approximate equality
pub fn models_approx_equal(m1: &Model, m2: &Model, tolerance: f32) -> bool {
    if m1.len() != m2.len() {
        return false;
    }
    for (name, t1) in m1 {
        if let Some(t2) = m2.get(name) {
            let d1 = t1.data();
            let d2 = t2.data();
            if d1.len() != d2.len() {
                return false;
            }
            for (a, b) in d1.iter().zip(d2.iter()) {
                if (a - b).abs() > tolerance {
                    return false;
                }
            }
        } else {
            return false;
        }
    }
    true
}
