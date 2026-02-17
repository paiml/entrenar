//! Serialization and DecisionPath trait implementation for NeuralPath

use super::{DecisionPath, NeuralPath, PathError};

/// Stateful byte reader that tracks offset and validates bounds.
struct ByteReader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> ByteReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    /// Read a single byte, advancing the offset.
    fn read_u8(&mut self) -> Result<u8, PathError> {
        self.ensure_available(1)?;
        let val = self.data[self.offset];
        self.offset += 1;
        Ok(val)
    }

    /// Read a little-endian u32, advancing the offset.
    fn read_u32(&mut self) -> Result<u32, PathError> {
        self.ensure_available(4)?;
        let o = self.offset;
        let val = u32::from_le_bytes([
            self.data[o],
            self.data[o + 1],
            self.data[o + 2],
            self.data[o + 3],
        ]);
        self.offset += 4;
        Ok(val)
    }

    /// Read a little-endian f32, advancing the offset.
    fn read_f32(&mut self) -> Result<f32, PathError> {
        self.ensure_available(4)?;
        let o = self.offset;
        let val = f32::from_le_bytes([
            self.data[o],
            self.data[o + 1],
            self.data[o + 2],
            self.data[o + 3],
        ]);
        self.offset += 4;
        Ok(val)
    }

    /// Read a length-prefixed `Vec<f32>`.
    fn read_f32_vec(&mut self) -> Result<Vec<f32>, PathError> {
        let len = self.read_u32()? as usize;
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(self.read_f32()?);
        }
        Ok(vec)
    }

    /// Read an optional field: 1-byte flag, then the value if present.
    fn read_optional<T>(
        &mut self,
        read_value: impl FnOnce(&mut Self) -> Result<T, PathError>,
    ) -> Result<Option<T>, PathError> {
        let present = self.read_u8()? != 0;
        if present {
            Ok(Some(read_value(self)?))
        } else {
            Ok(None)
        }
    }

    /// Read a length-prefixed `Vec<Vec<f32>>` (nested layers).
    fn read_nested_f32_vecs(&mut self) -> Result<Vec<Vec<f32>>, PathError> {
        let n_layers = self.read_u32()? as usize;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(self.read_f32_vec()?);
        }
        Ok(layers)
    }

    fn ensure_available(&self, needed: usize) -> Result<(), PathError> {
        if self.offset + needed > self.data.len() {
            return Err(PathError::InsufficientData {
                expected: self.offset + needed,
                actual: self.data.len(),
            });
        }
        Ok(())
    }
}

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

        let mut reader = ByteReader::new(bytes);

        let version = reader.read_u8()?;
        if version != 1 {
            return Err(PathError::VersionMismatch {
                expected: 1,
                actual: version,
            });
        }

        let input_gradient = reader.read_f32_vec()?;
        let prediction = reader.read_f32()?;
        let confidence = reader.read_f32()?;
        let activations = reader.read_optional(ByteReader::read_nested_f32_vecs)?;
        let attention_weights = reader.read_optional(ByteReader::read_nested_f32_vecs)?;
        let integrated_gradients = reader.read_optional(ByteReader::read_f32_vec)?;

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
