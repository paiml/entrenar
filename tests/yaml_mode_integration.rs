//! YAML Mode Integration Tests
//!
//! Tests for the YAML Mode QA Epic (100 scenarios).
//! Each test verifies that the YAML config files parse correctly
//! and meet the acceptance criteria from the QA specification.

use entrenar::yaml_mode::{validate_manifest, TrainingManifest};
use std::fs;
use std::path::Path;

/// Helper to parse YAML example files
fn parse_yaml_file(filename: &str) -> TrainingManifest {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/yaml")
        .join(filename);
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", filename, e));
    serde_yaml::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {}", filename, e))
}

/// Helper to validate a manifest passes the 25-point checklist basics
fn validate_qa_basics(manifest: &TrainingManifest, name: &str) {
    // Basic QA checks (subset of 25-point checklist)
    assert!(!manifest.name.is_empty(), "{}: name must be non-empty", name);
    assert_eq!(manifest.entrenar, "1.0", "{}: version must be 1.0", name);
    assert!(
        !manifest.version.is_empty(),
        "{}: version must be non-empty",
        name
    );
}

// ============================================================================
// SECTION A: BASIC TRAINING & DATA (YAML-001 to YAML-010)
// ============================================================================

mod section_a_data {
    use super::*;

    /// YAML-001: MNIST Baseline (CPU)
    /// QA Focus: Verify alimentar downloads and caches correctly
    #[test]
    fn yaml_001_mnist_baseline_cpu() {
        let manifest = parse_yaml_file("mnist_cpu.yaml");
        validate_qa_basics(&manifest, "YAML-001");

        // Acceptance criteria:
        // - Data source specified
        // - CPU device
        let data = manifest.data.expect("YAML-001 must have data config");
        assert!(
            data.source.is_some(),
            "YAML-001: data source must be specified"
        );

        if let Some(model) = &manifest.model {
            assert!(
                model.device.is_none()
                    || model.device.as_deref() == Some("cpu")
                    || model.device.as_deref() == Some("auto"),
                "YAML-001: should target CPU"
            );
        }
    }

    /// YAML-003: Custom Dataset (CSV)
    /// QA Focus: Check CSV parsing robustness
    #[test]
    fn yaml_003_custom_csv() {
        let manifest = parse_yaml_file("csv_data.yaml");
        validate_qa_basics(&manifest, "YAML-003");

        let data = manifest.data.expect("YAML-003 must have data config");
        assert_eq!(
            data.format,
            Some("csv".to_string()),
            "YAML-003: format must be csv"
        );
    }

    /// YAML-004: Parquet High-Throughput
    /// QA Focus: Throughput > 10k samples/sec
    #[test]
    fn yaml_004_parquet_throughput() {
        let manifest = parse_yaml_file("parquet_data.yaml");
        validate_qa_basics(&manifest, "YAML-004");

        let data = manifest.data.expect("YAML-004 must have data config");
        assert_eq!(
            data.format,
            Some("parquet".to_string()),
            "YAML-004: format must be parquet"
        );
    }

    /// YAML-005: Deterministic Replay
    /// QA Focus: Run twice; artifacts must have identical SHA256
    #[test]
    fn yaml_005_deterministic_replay() {
        let manifest = parse_yaml_file("deterministic.yaml");
        validate_qa_basics(&manifest, "YAML-005");

        // Acceptance criteria:
        // - Seed must be set
        // - Deterministic mode enabled
        assert!(manifest.seed.is_some(), "YAML-005: seed must be set");

        if let Some(training) = &manifest.training {
            assert_eq!(
                training.deterministic,
                Some(true),
                "YAML-005: deterministic must be true"
            );
        }
    }

    /// YAML-009: Multi-Worker Dataloading
    /// QA Focus: No deadlocks or race conditions
    #[test]
    fn yaml_009_multiworker() {
        let manifest = parse_yaml_file("multiworker.yaml");
        validate_qa_basics(&manifest, "YAML-009");

        let data = manifest.data.expect("YAML-009 must have data config");
        let loader = data.loader.expect("YAML-009 must have loader config");
        assert!(
            loader.num_workers.unwrap_or(0) > 1,
            "YAML-009: num_workers must be > 1"
        );
    }
}

// ============================================================================
// SECTION B: COMPILER-IN-THE-LOOP (YAML-014, YAML-017)
// ============================================================================

mod section_b_citl {
    use super::*;

    /// YAML-014: Automated Fix Suggestion
    /// QA Focus: Top 3 suggestions contain correct fix
    #[test]
    fn yaml_014_citl_suggest() {
        let manifest = parse_yaml_file("citl_suggest.yaml");
        validate_qa_basics(&manifest, "YAML-014");

        let citl = manifest.citl.expect("YAML-014 must have citl config");
        assert!(!citl.mode.is_empty(), "YAML-014: citl mode must be set");
    }

    /// YAML-017: Cross-Crate Decision Tracking
    /// QA Focus: Graph connectivity verified
    #[test]
    fn yaml_017_citl_workspace() {
        let manifest = parse_yaml_file("citl_workspace.yaml");
        validate_qa_basics(&manifest, "YAML-017");

        let citl = manifest.citl.expect("YAML-017 must have citl config");
        assert!(
            citl.workspace.is_some(),
            "YAML-017: workspace must be specified"
        );
    }
}

// ============================================================================
// SECTION C: MODEL ARCHITECTURE (YAML-021, YAML-023, YAML-025, YAML-028)
// ============================================================================

mod section_c_model {
    use super::*;

    /// YAML-021: Llama2 7B Training (Mock)
    /// QA Focus: Graph construction valid; parameter count exact
    #[test]
    fn yaml_021_llama2_mock() {
        let manifest = parse_yaml_file("llama2_mock.yaml");
        validate_qa_basics(&manifest, "YAML-021");

        assert!(
            manifest.model.is_some(),
            "YAML-021: model config must be present"
        );
    }

    /// YAML-023: Custom Architecture via YAML
    /// QA Focus: Layer dimensions match spec
    #[test]
    fn yaml_023_custom_arch() {
        let manifest = parse_yaml_file("custom_arch.yaml");
        validate_qa_basics(&manifest, "YAML-023");

        let model = manifest.model.expect("YAML-023 must have model config");
        assert!(
            model.architecture.is_some(),
            "YAML-023: architecture must be specified"
        );
    }

    /// YAML-025: Dropout Regularization
    /// QA Focus: Training loss > Training loss (no dropout); Val loss improves
    #[test]
    fn yaml_025_dropout() {
        let manifest = parse_yaml_file("dropout.yaml");
        validate_qa_basics(&manifest, "YAML-025");
        // Dropout config is typically in the model architecture
    }

    /// YAML-028: Gradient Clipping (Jidoka)
    /// QA Focus: Max gradient norm never exceeds 1.0
    #[test]
    fn yaml_028_grad_clip() {
        let manifest = parse_yaml_file("grad_clip.yaml");
        validate_qa_basics(&manifest, "YAML-028");

        let training = manifest
            .training
            .expect("YAML-028 must have training config");
        let gradient = training.gradient.expect("YAML-028 must have gradient config");
        assert!(
            gradient.clip_norm.is_some(),
            "YAML-028: clip_norm must be set"
        );
    }
}

// ============================================================================
// SECTION D: OPTIMIZATION & EFFICIENCY (YAML-031 to YAML-040)
// ============================================================================

mod section_d_optimization {
    use super::*;

    /// YAML-031: LoRA Fine-Tuning
    /// QA Focus: Trainable params < 1% of total
    #[test]
    fn yaml_031_lora() {
        let manifest = parse_yaml_file("lora.yaml");
        validate_qa_basics(&manifest, "YAML-031");

        let lora = manifest.lora.expect("YAML-031 must have lora config");
        assert!(lora.enabled, "YAML-031: lora must be enabled");
        assert!(lora.rank > 0, "YAML-031: lora rank must be > 0");
    }

    /// YAML-032: QLoRA (4-bit)
    /// QA Focus: VRAM usage reduction > 50%
    #[test]
    fn yaml_032_qlora() {
        let manifest = parse_yaml_file("qlora.yaml");
        validate_qa_basics(&manifest, "YAML-032");

        let lora = manifest.lora.expect("YAML-032 must have lora config");
        assert!(lora.enabled, "YAML-032: lora must be enabled");
        assert!(
            lora.quantize_base.unwrap_or(false),
            "YAML-032: quantize_base must be true"
        );
        assert_eq!(
            lora.quantize_bits,
            Some(4),
            "YAML-032: quantize_bits must be 4"
        );
    }

    /// YAML-034: Model Distillation
    /// QA Focus: Student matches 90% of teacher performance
    #[test]
    fn yaml_034_distillation() {
        let manifest = parse_yaml_file("distillation.yaml");
        validate_qa_basics(&manifest, "YAML-034");

        let distill = manifest
            .distillation
            .expect("YAML-034 must have distillation config");
        assert!(
            !distill.teacher.source.is_empty(),
            "YAML-034: teacher must be specified"
        );
        assert!(
            !distill.student.source.is_empty(),
            "YAML-034: student must be specified"
        );
    }

    /// YAML-037: Gradient Accumulation
    /// QA Focus: Math equivalence to large batch size verified
    #[test]
    fn yaml_037_grad_accum() {
        let manifest = parse_yaml_file("grad_accum.yaml");
        validate_qa_basics(&manifest, "YAML-037");

        let training = manifest
            .training
            .expect("YAML-037 must have training config");
        let gradient = training
            .gradient
            .expect("YAML-037 must have gradient config");
        assert!(
            gradient.accumulation_steps.unwrap_or(1) > 1,
            "YAML-037: accumulation_steps must be > 1"
        );
    }

    /// YAML-040: Learning Rate Scheduling
    /// QA Focus: LR follows exact curve (visual verify)
    #[test]
    fn yaml_040_lr_schedule() {
        let manifest = parse_yaml_file("lr_schedule.yaml");
        validate_qa_basics(&manifest, "YAML-040");

        assert!(
            manifest.scheduler.is_some(),
            "YAML-040: scheduler must be configured"
        );
    }
}

// ============================================================================
// SECTION E: MONITORING & OBSERVABILITY (YAML-044, YAML-048)
// ============================================================================

mod section_e_monitoring {
    use super::*;

    /// YAML-044: Andon Alert System
    /// QA Focus: ANDON alert triggered in < 5 steps
    #[test]
    fn yaml_044_andon() {
        let manifest = parse_yaml_file("andon.yaml");
        validate_qa_basics(&manifest, "YAML-044");

        let monitoring = manifest
            .monitoring
            .expect("YAML-044 must have monitoring config");
        assert!(
            monitoring.alerts.is_some(),
            "YAML-044: alerts must be configured"
        );
    }

    /// YAML-048: Outlier Detection
    /// QA Focus: Z-score threshold logic works
    #[test]
    fn yaml_048_outlier() {
        let manifest = parse_yaml_file("outlier.yaml");
        validate_qa_basics(&manifest, "YAML-048");

        let inspect = manifest
            .inspect
            .expect("YAML-048 must have inspect config");
        assert!(!inspect.mode.is_empty(), "YAML-048: inspect mode must be set");
    }
}

// ============================================================================
// SECTION F: RELIABILITY & RECOVERY (YAML-051, YAML-056, YAML-058, YAML-060)
// ============================================================================

mod section_f_reliability {
    use super::*;

    /// YAML-051: Automatic Checkpointing
    /// QA Focus: Resume from step 100 works perfectly
    #[test]
    fn yaml_051_checkpoint() {
        let manifest = parse_yaml_file("checkpoint.yaml");
        validate_qa_basics(&manifest, "YAML-051");

        let training = manifest
            .training
            .expect("YAML-051 must have training config");
        assert!(
            training.checkpoint.is_some(),
            "YAML-051: checkpoint must be configured"
        );
    }

    /// YAML-056: Config Validation (Poka-Yoke)
    /// QA Focus: Invalid types caught immediately
    #[test]
    fn yaml_056_config_validate() {
        let manifest = parse_yaml_file("config_validate.yaml");
        validate_qa_basics(&manifest, "YAML-056");

        assert_eq!(
            manifest.strict_validation,
            Some(true),
            "YAML-056: strict_validation must be true"
        );
    }

    /// YAML-058: Memory Leak Check (Long Run)
    /// QA Focus: RSS memory stable over 24h (simulated)
    #[test]
    fn yaml_058_long_run() {
        let manifest = parse_yaml_file("long_run.yaml");
        validate_qa_basics(&manifest, "YAML-058");

        // Long run should have extended training
        assert!(
            manifest.training.is_some(),
            "YAML-058: training config required"
        );
    }

    /// YAML-060: Lockfile Adherence
    /// QA Focus: Run matches locked parameters exactly
    #[test]
    fn yaml_060_locked() {
        let manifest = parse_yaml_file("locked.yaml");
        validate_qa_basics(&manifest, "YAML-060");

        assert!(
            manifest.lockfile.is_some(),
            "YAML-060: lockfile must be specified"
        );
    }
}

// ============================================================================
// SECTION G: INFERENCE & DEPLOYMENT (YAML-062, YAML-067)
// ============================================================================

mod section_g_inference {
    use super::*;

    /// YAML-062: Inference Latency Benchmark
    /// QA Focus: P99 latency within spec
    #[test]
    fn yaml_062_latency() {
        let manifest = parse_yaml_file("latency.yaml");
        validate_qa_basics(&manifest, "YAML-062");

        assert!(
            manifest.benchmark.is_some(),
            "YAML-062: benchmark config required"
        );
    }

    /// YAML-067: JSON Output Mode
    /// QA Focus: Output parsable by jq
    #[test]
    fn yaml_067_json_output() {
        let manifest = parse_yaml_file("json_output.yaml");
        validate_qa_basics(&manifest, "YAML-067");

        let output = manifest.output.expect("YAML-067 must have output config");
        let report = output.report.expect("YAML-067 must have report config");
        assert_eq!(
            report.format,
            Some("json".to_string()),
            "YAML-067: report format must be json"
        );
    }
}

// ============================================================================
// SECTION H: RESEARCH & SOVEREIGN AI (YAML-073, YAML-076)
// ============================================================================

mod section_h_research {
    use super::*;

    /// YAML-073: Differential Privacy
    /// QA Focus: Noise injection verified
    #[test]
    fn yaml_073_dp() {
        let manifest = parse_yaml_file("dp.yaml");
        validate_qa_basics(&manifest, "YAML-073");

        let privacy = manifest
            .privacy
            .expect("YAML-073 must have privacy config");
        assert!(privacy.differential, "YAML-073: differential must be true");
        assert!(privacy.epsilon > 0.0, "YAML-073: epsilon must be positive");
    }

    /// YAML-076: Bias Stress Test
    /// QA Focus: Parity score > 0.9
    #[test]
    fn yaml_076_bias() {
        let manifest = parse_yaml_file("bias.yaml");
        validate_qa_basics(&manifest, "YAML-076");

        let audit = manifest.audit.expect("YAML-076 must have audit config");
        assert!(!audit.audit_type.is_empty(), "YAML-076: audit type must be set");
    }
}

// ============================================================================
// SECTION I: ECOSYSTEM INTEGRATION (YAML-083)
// ============================================================================

mod section_i_ecosystem {
    use super::*;

    /// YAML-083: Ruchy Session Resume
    /// QA Focus: Session state restored exactly
    #[test]
    fn yaml_083_session() {
        let manifest = parse_yaml_file("session.yaml");
        validate_qa_basics(&manifest, "YAML-083");

        let session = manifest
            .session
            .expect("YAML-083 must have session config");
        assert!(!session.id.is_empty(), "YAML-083: session id must be set");
    }
}

// ============================================================================
// SECTION J: MISSION CRITICAL EDGE CASES (YAML-096, YAML-098, YAML-100)
// ============================================================================

mod section_j_edge_cases {
    use super::*;

    /// YAML-096: Maximal Load Soak Test
    /// QA Focus: No OOM; scheduler handles backpressure
    #[test]
    fn yaml_096_soak() {
        let manifest = parse_yaml_file("soak.yaml");
        validate_qa_basics(&manifest, "YAML-096");

        let stress = manifest.stress.expect("YAML-096 must have stress config");
        assert!(
            stress.parallel_jobs > 0,
            "YAML-096: parallel_jobs must be set"
        );
    }

    /// YAML-098: Model Drift Critical Alert
    /// QA Focus: Emergency stop triggered
    #[test]
    fn yaml_098_drift() {
        let manifest = parse_yaml_file("drift.yaml");
        validate_qa_basics(&manifest, "YAML-098");

        let monitoring = manifest
            .monitoring
            .expect("YAML-098 must have monitoring config");
        assert!(
            monitoring.drift_detection.is_some(),
            "YAML-098: drift_detection must be configured"
        );
    }

    /// YAML-100: The "Golden Run" (Final Verification)
    /// QA Focus: ALL 25 CHECKS MUST PASS. Peer review required.
    #[test]
    fn yaml_100_golden_run() {
        let manifest = parse_yaml_file("release.yaml");
        validate_qa_basics(&manifest, "YAML-100");

        // Golden run must have all safety features enabled
        assert_eq!(
            manifest.strict_validation,
            Some(true),
            "YAML-100: strict_validation required"
        );
        assert_eq!(
            manifest.require_peer_review,
            Some(true),
            "YAML-100: peer review required"
        );
        assert!(
            manifest.signing.is_some(),
            "YAML-100: signing config required"
        );
        assert!(
            manifest.verification.is_some(),
            "YAML-100: verification config required"
        );

        // Validate manifest passes full validation
        let result = validate_manifest(&manifest);
        assert!(
            result.is_ok(),
            "YAML-100: manifest must pass validation: {:?}",
            result
        );
    }
}

// ============================================================================
// COMPREHENSIVE VALIDATION TESTS
// ============================================================================

mod validation {
    use super::*;

    /// Test that all 30 YAML files parse successfully
    #[test]
    fn all_yaml_files_parse() {
        let files = [
            "mnist_cpu.yaml",
            "csv_data.yaml",
            "parquet_data.yaml",
            "deterministic.yaml",
            "multiworker.yaml",
            "citl_suggest.yaml",
            "citl_workspace.yaml",
            "llama2_mock.yaml",
            "custom_arch.yaml",
            "dropout.yaml",
            "grad_clip.yaml",
            "lora.yaml",
            "qlora.yaml",
            "distillation.yaml",
            "grad_accum.yaml",
            "lr_schedule.yaml",
            "andon.yaml",
            "outlier.yaml",
            "checkpoint.yaml",
            "config_validate.yaml",
            "long_run.yaml",
            "locked.yaml",
            "latency.yaml",
            "json_output.yaml",
            "dp.yaml",
            "bias.yaml",
            "session.yaml",
            "soak.yaml",
            "drift.yaml",
            "release.yaml",
        ];

        for file in &files {
            let manifest = parse_yaml_file(file);
            assert!(
                !manifest.name.is_empty(),
                "{} must have a name",
                file
            );
            assert_eq!(
                manifest.entrenar, "1.0",
                "{} must have entrenar version 1.0",
                file
            );
        }
    }

    /// Test that all YAML files pass validation
    #[test]
    fn all_yaml_files_validate() {
        let files = [
            "mnist_cpu.yaml",
            "csv_data.yaml",
            "parquet_data.yaml",
            "deterministic.yaml",
            "multiworker.yaml",
            "citl_suggest.yaml",
            "citl_workspace.yaml",
            "llama2_mock.yaml",
            "custom_arch.yaml",
            "dropout.yaml",
            "grad_clip.yaml",
            "lora.yaml",
            "qlora.yaml",
            "distillation.yaml",
            "grad_accum.yaml",
            "lr_schedule.yaml",
            "andon.yaml",
            "outlier.yaml",
            "checkpoint.yaml",
            "config_validate.yaml",
            "long_run.yaml",
            "locked.yaml",
            "latency.yaml",
            "json_output.yaml",
            "dp.yaml",
            "bias.yaml",
            "session.yaml",
            "soak.yaml",
            "drift.yaml",
            "release.yaml",
        ];

        for file in &files {
            let manifest = parse_yaml_file(file);
            let result = validate_manifest(&manifest);
            assert!(
                result.is_ok(),
                "{} must pass validation: {:?}",
                file, result
            );
        }
    }
}
