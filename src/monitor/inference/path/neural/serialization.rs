//! Serialization and DecisionPath trait implementation for NeuralPath

use super::{DecisionPath, NeuralPath, PathError};

impl DecisionPath for NeuralPath {
    fn explain(&self) -> String {
        let mut explanation = format!(
            "Neural Network Prediction: {:.4} (confidence: {:.1}%)\n",
            self.prediction,
            self.confidence * 100.0
        );

        explanation.push_str("\nTop salient input features (by gradient):\n");
        for (idx, grad) in self.top_salient_features(5) {
            let sign = if grad >= 0.0 { "+" } else { "" };
            explanation.push_str(&format!("  input[{idx}]: {sign}{grad:.6}\n"));
        }

        if let Some(ig) = &self.integrated_gradients {
            explanation.push_str("\nIntegrated gradients available (");
            let len = ig.len();
            explanation.push_str(&format!("{len} features)\n"));
        }

        if self.attention_weights.is_some() {
            explanation.push_str("\nAttention weights available\n");
        }

        explanation
    }

    fn feature_contributions(&self) -> &[f32] {
        self.integrated_gradients
            .as_deref()
            .unwrap_or(&self.input_gradient)
    }

    fn confidence(&self) -> f32 {
        self.confidence
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(1); // version

        // Input gradient
        bytes.extend_from_slice(&(self.input_gradient.len() as u32).to_le_bytes());
        for g in &self.input_gradient {
            bytes.extend_from_slice(&g.to_le_bytes());
        }

        // Prediction and confidence
        bytes.extend_from_slice(&self.prediction.to_le_bytes());
        bytes.extend_from_slice(&self.confidence.to_le_bytes());

        // Activations
        let has_activations = self.activations.is_some();
        bytes.push(u8::from(has_activations));
        if let Some(activations) = &self.activations {
            bytes.extend_from_slice(&(activations.len() as u32).to_le_bytes());
            for layer in activations {
                bytes.extend_from_slice(&(layer.len() as u32).to_le_bytes());
                for a in layer {
                    bytes.extend_from_slice(&a.to_le_bytes());
                }
            }
        }

        // Attention weights
        let has_attention = self.attention_weights.is_some();
        bytes.push(u8::from(has_attention));
        if let Some(attention) = &self.attention_weights {
            bytes.extend_from_slice(&(attention.len() as u32).to_le_bytes());
            for layer in attention {
                bytes.extend_from_slice(&(layer.len() as u32).to_le_bytes());
                for a in layer {
                    bytes.extend_from_slice(&a.to_le_bytes());
                }
            }
        }

        // Integrated gradients
        let has_ig = self.integrated_gradients.is_some();
        bytes.push(u8::from(has_ig));
        if let Some(ig) = &self.integrated_gradients {
            bytes.extend_from_slice(&(ig.len() as u32).to_le_bytes());
            for g in ig {
                bytes.extend_from_slice(&g.to_le_bytes());
            }
        }

        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, PathError> {
        if bytes.len() < 5 {
            return Err(PathError::InsufficientData {
                expected: 5,
                actual: bytes.len(),
            });
        }

        let version = bytes[0];
        if version != 1 {
            return Err(PathError::VersionMismatch {
                expected: 1,
                actual: version,
            });
        }

        let mut offset = 1;

        // Input gradient
        let n_grad = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        let mut input_gradient = Vec::with_capacity(n_grad);
        for _ in 0..n_grad {
            if offset + 4 > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + 4,
                    actual: bytes.len(),
                });
            }
            let g = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            offset += 4;
            input_gradient.push(g);
        }

        // Prediction and confidence
        if offset + 8 > bytes.len() {
            return Err(PathError::InsufficientData {
                expected: offset + 8,
                actual: bytes.len(),
            });
        }
        let prediction = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        offset += 4;

        let confidence = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        offset += 4;

        // Activations
        if offset + 1 > bytes.len() {
            return Err(PathError::InsufficientData {
                expected: offset + 1,
                actual: bytes.len(),
            });
        }
        let has_activations = bytes[offset] != 0;
        offset += 1;

        let activations = if has_activations {
            if offset + 4 > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + 4,
                    actual: bytes.len(),
                });
            }
            let n_layers = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            offset += 4;

            let mut layers = Vec::with_capacity(n_layers);
            for _ in 0..n_layers {
                if offset + 4 > bytes.len() {
                    return Err(PathError::InsufficientData {
                        expected: offset + 4,
                        actual: bytes.len(),
                    });
                }
                let layer_len = u32::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]) as usize;
                offset += 4;

                let mut layer = Vec::with_capacity(layer_len);
                for _ in 0..layer_len {
                    if offset + 4 > bytes.len() {
                        return Err(PathError::InsufficientData {
                            expected: offset + 4,
                            actual: bytes.len(),
                        });
                    }
                    let a = f32::from_le_bytes([
                        bytes[offset],
                        bytes[offset + 1],
                        bytes[offset + 2],
                        bytes[offset + 3],
                    ]);
                    offset += 4;
                    layer.push(a);
                }
                layers.push(layer);
            }
            Some(layers)
        } else {
            None
        };

        // Attention weights (similar pattern)
        if offset + 1 > bytes.len() {
            return Err(PathError::InsufficientData {
                expected: offset + 1,
                actual: bytes.len(),
            });
        }
        let has_attention = bytes[offset] != 0;
        offset += 1;

        let attention_weights = if has_attention {
            if offset + 4 > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + 4,
                    actual: bytes.len(),
                });
            }
            let n_layers = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            offset += 4;

            let mut layers = Vec::with_capacity(n_layers);
            for _ in 0..n_layers {
                if offset + 4 > bytes.len() {
                    return Err(PathError::InsufficientData {
                        expected: offset + 4,
                        actual: bytes.len(),
                    });
                }
                let layer_len = u32::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]) as usize;
                offset += 4;

                let mut layer = Vec::with_capacity(layer_len);
                for _ in 0..layer_len {
                    if offset + 4 > bytes.len() {
                        return Err(PathError::InsufficientData {
                            expected: offset + 4,
                            actual: bytes.len(),
                        });
                    }
                    let a = f32::from_le_bytes([
                        bytes[offset],
                        bytes[offset + 1],
                        bytes[offset + 2],
                        bytes[offset + 3],
                    ]);
                    offset += 4;
                    layer.push(a);
                }
                layers.push(layer);
            }
            Some(layers)
        } else {
            None
        };

        // Integrated gradients
        if offset + 1 > bytes.len() {
            return Err(PathError::InsufficientData {
                expected: offset + 1,
                actual: bytes.len(),
            });
        }
        let has_ig = bytes[offset] != 0;
        offset += 1;

        let integrated_gradients = if has_ig {
            if offset + 4 > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + 4,
                    actual: bytes.len(),
                });
            }
            let n_ig = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            offset += 4;

            let mut ig = Vec::with_capacity(n_ig);
            for _ in 0..n_ig {
                if offset + 4 > bytes.len() {
                    return Err(PathError::InsufficientData {
                        expected: offset + 4,
                        actual: bytes.len(),
                    });
                }
                let g = f32::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]);
                offset += 4;
                ig.push(g);
            }
            Some(ig)
        } else {
            None
        };

        Ok(Self {
            input_gradient,
            activations,
            attention_weights,
            integrated_gradients,
            prediction,
            confidence,
        })
    }
}
