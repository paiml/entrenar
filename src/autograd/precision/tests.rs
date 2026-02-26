//! Tests for mixed-precision training utilities.

#[cfg(test)]
mod tests {
    use crate::autograd::precision::{
        bf16_to_f32, estimate_memory_savings, f32_to_bf16, f32_to_fp16, fp16_to_f32, GradScaler,
        MixedPrecisionConfig, Precision,
    };

    #[test]
    fn test_precision_size_bytes() {
        assert_eq!(Precision::Fp32.size_bytes(), 4);
        assert_eq!(Precision::Fp16.size_bytes(), 2);
        assert_eq!(Precision::Bf16.size_bytes(), 2);
    }

    #[test]
    fn test_precision_name() {
        assert_eq!(Precision::Fp32.name(), "fp32");
        assert_eq!(Precision::Fp16.name(), "fp16");
        assert_eq!(Precision::Bf16.name(), "bf16");
    }

    #[test]
    fn test_precision_is_reduced() {
        assert!(!Precision::Fp32.is_reduced());
        assert!(Precision::Fp16.is_reduced());
        assert!(Precision::Bf16.is_reduced());
    }

    #[test]
    fn test_precision_memory_multiplier() {
        assert_eq!(Precision::Fp32.memory_multiplier(), 1.0);
        assert_eq!(Precision::Fp16.memory_multiplier(), 0.5);
    }

    #[test]
    fn test_precision_display() {
        assert_eq!(format!("{}", Precision::Bf16), "bf16");
    }

    #[test]
    fn test_precision_default() {
        assert_eq!(Precision::default(), Precision::Fp32);
    }

    #[test]
    fn test_mixed_precision_config_fp32() {
        let config = MixedPrecisionConfig::fp32();
        assert!(!config.is_mixed());
        assert_eq!(config.compute_precision, Precision::Fp32);
    }

    #[test]
    fn test_mixed_precision_config_fp16() {
        let config = MixedPrecisionConfig::fp16();
        assert!(config.is_mixed());
        assert_eq!(config.compute_precision, Precision::Fp16);
        assert!(config.dynamic_scaling);
        assert_eq!(config.initial_scale, 65536.0);
    }

    #[test]
    fn test_mixed_precision_config_bf16() {
        let config = MixedPrecisionConfig::bf16();
        assert!(config.is_mixed());
        assert_eq!(config.compute_precision, Precision::Bf16);
        assert!(!config.dynamic_scaling); // bf16 typically doesn't need scaling
    }

    #[test]
    fn test_mixed_precision_config_builders() {
        let config =
            MixedPrecisionConfig::fp16().with_initial_scale(1024.0).with_dynamic_scaling(false);
        assert_eq!(config.initial_scale, 1024.0);
        assert!(!config.dynamic_scaling);
    }

    #[test]
    fn test_grad_scaler_new() {
        let scaler = GradScaler::new(65536.0);
        assert_eq!(scaler.scale(), 65536.0);
        assert!(scaler.is_dynamic());
    }

    #[test]
    fn test_grad_scaler_from_config() {
        let config = MixedPrecisionConfig::fp16();
        let scaler = GradScaler::from_config(&config);
        assert_eq!(scaler.scale(), config.initial_scale);
    }

    #[test]
    fn test_grad_scaler_scale_loss() {
        let scaler = GradScaler::new(1000.0);
        assert_eq!(scaler.scale_loss(0.001), 1.0);
    }

    #[test]
    fn test_grad_scaler_unscale_grad() {
        let scaler = GradScaler::new(1000.0);
        assert_eq!(scaler.unscale_grad(1000.0), 1.0);
    }

    #[test]
    fn test_grad_scaler_unscale_and_check_valid() {
        let scaler = GradScaler::new(100.0);
        let mut grads = vec![100.0, 200.0, 300.0];
        let valid = scaler.unscale_and_check(&mut grads);

        assert!(valid);
        assert_eq!(grads[0], 1.0);
        assert_eq!(grads[1], 2.0);
        assert_eq!(grads[2], 3.0);
    }

    #[test]
    fn test_grad_scaler_unscale_and_check_overflow() {
        let scaler = GradScaler::new(100.0);
        let mut grads = vec![100.0, f32::INFINITY, 300.0];
        let valid = scaler.unscale_and_check(&mut grads);

        assert!(!valid);
    }

    #[test]
    fn test_grad_scaler_update_success() {
        let mut scaler = GradScaler::new(1000.0);
        scaler.growth_interval = 2; // Fast growth for testing

        scaler.update(true);
        scaler.update(true);

        // After growth_interval successful steps, scale should grow
        assert!(scaler.scale() > 1000.0);
        assert_eq!(scaler.successful_steps(), 2);
    }

    #[test]
    fn test_grad_scaler_update_overflow() {
        let mut scaler = GradScaler::new(1000.0);

        scaler.update(false); // Overflow

        assert!(scaler.scale() < 1000.0);
        assert_eq!(scaler.overflow_count(), 1);
    }

    #[test]
    fn test_grad_scaler_scale_floor() {
        let mut scaler = GradScaler::new(1.0);

        scaler.update(false); // Overflow

        // Scale should not go below 1.0
        assert!(scaler.scale() >= 1.0);
    }

    #[test]
    fn test_grad_scaler_dynamic_disabled() {
        let mut scaler = GradScaler::new(1000.0);
        scaler.set_dynamic(false);

        scaler.update(false);

        // Scale should not change when dynamic is disabled
        assert_eq!(scaler.scale(), 1000.0);
    }

    #[test]
    fn test_f32_to_bf16_roundtrip() {
        let values = vec![0.0, 1.0, -1.0, 0.5, 100.0, -0.001];
        for &val in &values {
            let bf16 = f32_to_bf16(val);
            let back = bf16_to_f32(bf16);
            // BF16 has limited precision, so we check approximate equality
            if val.abs() > 1e-6 {
                let rel_err = (back - val).abs() / val.abs();
                assert!(rel_err < 0.01, "BF16 roundtrip error too large for {val}");
            }
        }
    }

    #[test]
    fn test_f32_to_fp16_roundtrip() {
        let values = vec![0.0, 1.0, -1.0, 0.5, 100.0];
        for &val in &values {
            let fp16 = f32_to_fp16(val);
            let back = fp16_to_f32(fp16);
            // FP16 has limited precision
            if val.abs() > 1e-4 {
                let rel_err = (back - val).abs() / val.abs();
                assert!(rel_err < 0.01, "FP16 roundtrip error too large for {val}");
            }
        }
    }

    #[test]
    fn test_bf16_special_values() {
        // Zero
        let zero = f32_to_bf16(0.0);
        assert_eq!(bf16_to_f32(zero), 0.0);

        // Negative zero
        let neg_zero = f32_to_bf16(-0.0);
        assert_eq!(bf16_to_f32(neg_zero), -0.0);
    }

    #[test]
    fn test_fp16_infinity() {
        let inf = f32_to_fp16(f32::INFINITY);
        let back = fp16_to_f32(inf);
        assert!(back.is_infinite() && back > 0.0);

        let neg_inf = f32_to_fp16(f32::NEG_INFINITY);
        let back_neg = fp16_to_f32(neg_inf);
        assert!(back_neg.is_infinite() && back_neg < 0.0);
    }

    #[test]
    fn test_estimate_memory_savings() {
        let (fp32, mixed, savings) =
            estimate_memory_savings(1_000_000, 8, 512, 4096, Precision::Bf16);

        assert!(mixed < fp32);
        assert!(savings > 0.0);
        assert!(savings < 1.0);
    }

    #[test]
    fn test_memory_savings_no_reduction_for_fp32() {
        let (fp32, mixed, savings) =
            estimate_memory_savings(1_000_000, 8, 512, 4096, Precision::Fp32);

        assert_eq!(fp32, mixed);
        assert_eq!(savings, 0.0);
    }

    #[test]
    fn test_grad_scaler_default() {
        let scaler = GradScaler::default();
        assert_eq!(scaler.scale(), 65536.0);
    }

    #[test]
    fn test_mixed_precision_config_default() {
        let config = MixedPrecisionConfig::default();
        assert!(!config.is_mixed());
    }
}
