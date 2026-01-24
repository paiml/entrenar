//! Popperian Falsification QA System
//!
//! Implements a 100-point scientific validation checklist based on
//! Karl Popper's philosophy of falsifiability.
//!
//! # References
//!
//! - Popper, K. (1959) "The Logic of Scientific Discovery"
//! - Popper, K. (1963) "Conjectures and Refutations"

use std::fmt;

/// Quality grade based on Popperian score
/// Ordering: F < C < B < BPlus < A < APlus (worst to best)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QAGrade {
    /// <70: Failing
    F,
    /// 70-79: Needs Improvement
    C,
    /// 80-84: Satisfactory
    B,
    /// 85-89: Good
    BPlus,
    /// 90-94: Very Good
    A,
    /// 95-100: Excellent
    APlus,
}

impl QAGrade {
    /// Create grade from score
    #[must_use]
    pub const fn from_score(score: u8) -> Self {
        match score {
            95..=100 => Self::APlus,
            90..=94 => Self::A,
            85..=89 => Self::BPlus,
            80..=84 => Self::B,
            70..=79 => Self::C,
            _ => Self::F,
        }
    }

    /// Check if grade is passing (C or better)
    #[must_use]
    pub const fn is_passing(&self) -> bool {
        !matches!(self, Self::F)
    }
}

impl fmt::Display for QAGrade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::APlus => write!(f, "A+ (Excellent)"),
            Self::A => write!(f, "A  (Very Good)"),
            Self::BPlus => write!(f, "B+ (Good)"),
            Self::B => write!(f, "B  (Satisfactory)"),
            Self::C => write!(f, "C  (Needs Improvement)"),
            Self::F => write!(f, "F  (Failing)"),
        }
    }
}

/// 100-Point Popperian Falsification Checklist
///
/// Each field represents a falsifiable hypothesis. If the test passes,
/// the hypothesis is corroborated (not proven). If it fails, the hypothesis
/// is falsified.
#[derive(Debug, Clone, Default)]
pub struct PopperianQA {
    // === REPRODUCIBILITY (20 points) ===
    // H1: Training is deterministic with fixed seed
    /// Same seed produces identical loss curves (±1e-6)
    pub r1_same_loss_curve: bool,
    /// Adapter weights match exactly across runs
    pub r2_same_final_weights: bool,
    /// Evaluation metrics are identical
    pub r3_same_eval_metrics: bool,
    /// All dependencies are version-locked
    pub r4_environment_locked: bool,

    // === COMPILATION (20 points) ===
    // H2: Generated tests are syntactically valid Rust
    /// rustfmt succeeds on generated code
    pub c1_parses_as_rust: bool,
    /// cargo check succeeds
    pub c2_type_checks: bool,
    /// No clippy warnings
    pub c3_no_unused_warnings: bool,
    /// Links correctly against target crate
    pub c4_links_correctly: bool,

    // === CORRECTNESS (20 points) ===
    // H3: Generated tests are semantically meaningful
    /// Tests pass on original implementation
    pub x1_tests_pass_on_correct: bool,
    /// Tests fail on mutated implementation
    pub x2_tests_fail_on_mutant: bool,
    /// Assertions are not trivial (not just `assert!(true)`)
    pub x3_assertions_meaningful: bool,
    /// No tautologies (not `assert_eq!(x, x)`)
    pub x4_no_tautologies: bool,

    // === COVERAGE (15 points) ===
    // H4: Tests exercise meaningful code paths
    /// Branch coverage delta ≥+5%
    pub v1_branch_coverage_delta: bool,
    /// Line coverage delta ≥+10%
    pub v2_line_coverage_delta: bool,
    /// Tests include edge cases (empty, null, max)
    pub v3_edge_cases_present: bool,

    // === EFFICIENCY (10 points) ===
    // H5: Training completes within resource bounds
    /// Peak VRAM < 8GB
    pub e1_vram_under_8gb: bool,
    /// Training completes in <4 hours
    pub e2_training_under_4hrs: bool,
    /// Inference < 1s per function
    pub e3_inference_under_1s: bool,

    // === EDGE CASES (10 points) ===
    // H6: Handles difficult inputs gracefully
    /// Handles generic functions
    pub g1_handles_generics: bool,
    /// Handles lifetime annotations
    pub g2_handles_lifetimes: bool,
    /// Handles async functions
    pub g3_handles_async: bool,
    /// Handles unsafe blocks
    pub g4_handles_unsafe: bool,
    /// Handles macro-heavy code
    pub g5_handles_macros: bool,

    // === DOCUMENTATION (5 points) ===
    // H7: Output is self-explanatory
    /// Test names describe intent (test_*)
    pub d1_test_names_descriptive: bool,
    /// Comments explain edge cases
    pub d2_comments_present: bool,
    /// Proptest strategies have clear names
    pub d3_proptest_strategies_clear: bool,
}

impl PopperianQA {
    /// Create new QA checklist with all items unchecked
    #[must_use]
    pub const fn new() -> Self {
        Self {
            r1_same_loss_curve: false,
            r2_same_final_weights: false,
            r3_same_eval_metrics: false,
            r4_environment_locked: false,
            c1_parses_as_rust: false,
            c2_type_checks: false,
            c3_no_unused_warnings: false,
            c4_links_correctly: false,
            x1_tests_pass_on_correct: false,
            x2_tests_fail_on_mutant: false,
            x3_assertions_meaningful: false,
            x4_no_tautologies: false,
            v1_branch_coverage_delta: false,
            v2_line_coverage_delta: false,
            v3_edge_cases_present: false,
            e1_vram_under_8gb: false,
            e2_training_under_4hrs: false,
            e3_inference_under_1s: false,
            g1_handles_generics: false,
            g2_handles_lifetimes: false,
            g3_handles_async: false,
            g4_handles_unsafe: false,
            g5_handles_macros: false,
            d1_test_names_descriptive: false,
            d2_comments_present: false,
            d3_proptest_strategies_clear: false,
        }
    }

    /// Calculate total score (0-100)
    #[must_use]
    pub fn score(&self) -> u8 {
        let mut score = 0u8;

        // Reproducibility (20 points)
        if self.r1_same_loss_curve {
            score += 5;
        }
        if self.r2_same_final_weights {
            score += 5;
        }
        if self.r3_same_eval_metrics {
            score += 5;
        }
        if self.r4_environment_locked {
            score += 5;
        }

        // Compilation (20 points)
        if self.c1_parses_as_rust {
            score += 5;
        }
        if self.c2_type_checks {
            score += 5;
        }
        if self.c3_no_unused_warnings {
            score += 5;
        }
        if self.c4_links_correctly {
            score += 5;
        }

        // Correctness (20 points)
        if self.x1_tests_pass_on_correct {
            score += 5;
        }
        if self.x2_tests_fail_on_mutant {
            score += 5;
        }
        if self.x3_assertions_meaningful {
            score += 5;
        }
        if self.x4_no_tautologies {
            score += 5;
        }

        // Coverage (15 points)
        if self.v1_branch_coverage_delta {
            score += 5;
        }
        if self.v2_line_coverage_delta {
            score += 5;
        }
        if self.v3_edge_cases_present {
            score += 5;
        }

        // Efficiency (10 points)
        if self.e1_vram_under_8gb {
            score += 3;
        }
        if self.e2_training_under_4hrs {
            score += 4;
        }
        if self.e3_inference_under_1s {
            score += 3;
        }

        // Edge Cases (10 points)
        if self.g1_handles_generics {
            score += 2;
        }
        if self.g2_handles_lifetimes {
            score += 2;
        }
        if self.g3_handles_async {
            score += 2;
        }
        if self.g4_handles_unsafe {
            score += 2;
        }
        if self.g5_handles_macros {
            score += 2;
        }

        // Documentation (5 points)
        if self.d1_test_names_descriptive {
            score += 2;
        }
        if self.d2_comments_present {
            score += 2;
        }
        if self.d3_proptest_strategies_clear {
            score += 1;
        }

        score
    }

    /// Get quality grade
    #[must_use]
    pub fn grade(&self) -> QAGrade {
        QAGrade::from_score(self.score())
    }

    /// Check if all reproducibility criteria pass
    #[must_use]
    pub const fn reproducibility_passed(&self) -> bool {
        self.r1_same_loss_curve
            && self.r2_same_final_weights
            && self.r3_same_eval_metrics
            && self.r4_environment_locked
    }

    /// Check if all compilation criteria pass
    #[must_use]
    pub const fn compilation_passed(&self) -> bool {
        self.c1_parses_as_rust
            && self.c2_type_checks
            && self.c3_no_unused_warnings
            && self.c4_links_correctly
    }

    /// Check if all correctness criteria pass
    #[must_use]
    pub const fn correctness_passed(&self) -> bool {
        self.x1_tests_pass_on_correct
            && self.x2_tests_fail_on_mutant
            && self.x3_assertions_meaningful
            && self.x4_no_tautologies
    }

    /// Count passed items
    #[must_use]
    pub fn passed_count(&self) -> usize {
        let bools = [
            self.r1_same_loss_curve,
            self.r2_same_final_weights,
            self.r3_same_eval_metrics,
            self.r4_environment_locked,
            self.c1_parses_as_rust,
            self.c2_type_checks,
            self.c3_no_unused_warnings,
            self.c4_links_correctly,
            self.x1_tests_pass_on_correct,
            self.x2_tests_fail_on_mutant,
            self.x3_assertions_meaningful,
            self.x4_no_tautologies,
            self.v1_branch_coverage_delta,
            self.v2_line_coverage_delta,
            self.v3_edge_cases_present,
            self.e1_vram_under_8gb,
            self.e2_training_under_4hrs,
            self.e3_inference_under_1s,
            self.g1_handles_generics,
            self.g2_handles_lifetimes,
            self.g3_handles_async,
            self.g4_handles_unsafe,
            self.g5_handles_macros,
            self.d1_test_names_descriptive,
            self.d2_comments_present,
            self.d3_proptest_strategies_clear,
        ];
        bools.iter().filter(|&&b| b).count()
    }

    /// Total number of items
    #[must_use]
    pub const fn total_items(&self) -> usize {
        26
    }

    /// Generate markdown report
    #[must_use]
    pub fn report(&self) -> String {
        let mut out = String::new();
        out.push_str("# Popperian Falsification QA Report\n\n");
        out.push_str(&format!("**Score:** {}/100\n", self.score()));
        out.push_str(&format!("**Grade:** {}\n", self.grade()));
        out.push_str(&format!(
            "**Items Passed:** {}/{}\n\n",
            self.passed_count(),
            self.total_items()
        ));

        out.push_str("## Reproducibility (20 pts)\n");
        out.push_str(&format!(
            "- [{}] R1: Same loss curve\n",
            if self.r1_same_loss_curve { "x" } else { " " }
        ));
        out.push_str(&format!(
            "- [{}] R2: Same final weights\n",
            if self.r2_same_final_weights { "x" } else { " " }
        ));
        out.push_str(&format!(
            "- [{}] R3: Same eval metrics\n",
            if self.r3_same_eval_metrics { "x" } else { " " }
        ));
        out.push_str(&format!(
            "- [{}] R4: Environment locked\n",
            if self.r4_environment_locked { "x" } else { " " }
        ));

        out.push_str("\n## Compilation (20 pts)\n");
        out.push_str(&format!(
            "- [{}] C1: Parses as Rust\n",
            if self.c1_parses_as_rust { "x" } else { " " }
        ));
        out.push_str(&format!(
            "- [{}] C2: Type checks\n",
            if self.c2_type_checks { "x" } else { " " }
        ));
        out.push_str(&format!(
            "- [{}] C3: No unused warnings\n",
            if self.c3_no_unused_warnings { "x" } else { " " }
        ));
        out.push_str(&format!(
            "- [{}] C4: Links correctly\n",
            if self.c4_links_correctly { "x" } else { " " }
        ));

        out.push_str("\n## Correctness (20 pts)\n");
        out.push_str(&format!(
            "- [{}] X1: Tests pass on correct\n",
            if self.x1_tests_pass_on_correct {
                "x"
            } else {
                " "
            }
        ));
        out.push_str(&format!(
            "- [{}] X2: Tests fail on mutant\n",
            if self.x2_tests_fail_on_mutant {
                "x"
            } else {
                " "
            }
        ));
        out.push_str(&format!(
            "- [{}] X3: Assertions meaningful\n",
            if self.x3_assertions_meaningful {
                "x"
            } else {
                " "
            }
        ));
        out.push_str(&format!(
            "- [{}] X4: No tautologies\n",
            if self.x4_no_tautologies { "x" } else { " " }
        ));

        out.push_str("\n## Coverage (15 pts)\n");
        out.push_str(&format!(
            "- [{}] V1: Branch coverage +5%\n",
            if self.v1_branch_coverage_delta {
                "x"
            } else {
                " "
            }
        ));
        out.push_str(&format!(
            "- [{}] V2: Line coverage +10%\n",
            if self.v2_line_coverage_delta {
                "x"
            } else {
                " "
            }
        ));
        out.push_str(&format!(
            "- [{}] V3: Edge cases present\n",
            if self.v3_edge_cases_present { "x" } else { " " }
        ));

        out.push_str("\n## Efficiency (10 pts)\n");
        out.push_str(&format!(
            "- [{}] E1: VRAM < 8GB\n",
            if self.e1_vram_under_8gb { "x" } else { " " }
        ));
        out.push_str(&format!(
            "- [{}] E2: Training < 4hrs\n",
            if self.e2_training_under_4hrs {
                "x"
            } else {
                " "
            }
        ));
        out.push_str(&format!(
            "- [{}] E3: Inference < 1s\n",
            if self.e3_inference_under_1s { "x" } else { " " }
        ));

        out.push_str("\n## Edge Cases (10 pts)\n");
        out.push_str(&format!(
            "- [{}] G1: Handles generics\n",
            if self.g1_handles_generics { "x" } else { " " }
        ));
        out.push_str(&format!(
            "- [{}] G2: Handles lifetimes\n",
            if self.g2_handles_lifetimes { "x" } else { " " }
        ));
        out.push_str(&format!(
            "- [{}] G3: Handles async\n",
            if self.g3_handles_async { "x" } else { " " }
        ));
        out.push_str(&format!(
            "- [{}] G4: Handles unsafe\n",
            if self.g4_handles_unsafe { "x" } else { " " }
        ));
        out.push_str(&format!(
            "- [{}] G5: Handles macros\n",
            if self.g5_handles_macros { "x" } else { " " }
        ));

        out.push_str("\n## Documentation (5 pts)\n");
        out.push_str(&format!(
            "- [{}] D1: Descriptive test names\n",
            if self.d1_test_names_descriptive {
                "x"
            } else {
                " "
            }
        ));
        out.push_str(&format!(
            "- [{}] D2: Comments present\n",
            if self.d2_comments_present { "x" } else { " " }
        ));
        out.push_str(&format!(
            "- [{}] D3: Clear proptest strategies\n",
            if self.d3_proptest_strategies_clear {
                "x"
            } else {
                " "
            }
        ));

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qa_grade_from_score() {
        assert_eq!(QAGrade::from_score(100), QAGrade::APlus);
        assert_eq!(QAGrade::from_score(95), QAGrade::APlus);
        assert_eq!(QAGrade::from_score(94), QAGrade::A);
        assert_eq!(QAGrade::from_score(90), QAGrade::A);
        assert_eq!(QAGrade::from_score(89), QAGrade::BPlus);
        assert_eq!(QAGrade::from_score(85), QAGrade::BPlus);
        assert_eq!(QAGrade::from_score(84), QAGrade::B);
        assert_eq!(QAGrade::from_score(80), QAGrade::B);
        assert_eq!(QAGrade::from_score(79), QAGrade::C);
        assert_eq!(QAGrade::from_score(70), QAGrade::C);
        assert_eq!(QAGrade::from_score(69), QAGrade::F);
        assert_eq!(QAGrade::from_score(0), QAGrade::F);
    }

    #[test]
    fn test_qa_grade_is_passing() {
        assert!(QAGrade::APlus.is_passing());
        assert!(QAGrade::A.is_passing());
        assert!(QAGrade::BPlus.is_passing());
        assert!(QAGrade::B.is_passing());
        assert!(QAGrade::C.is_passing());
        assert!(!QAGrade::F.is_passing());
    }

    #[test]
    fn test_popperian_qa_new() {
        let qa = PopperianQA::new();
        assert_eq!(qa.score(), 0);
        assert_eq!(qa.grade(), QAGrade::F);
        assert_eq!(qa.passed_count(), 0);
    }

    #[test]
    fn test_popperian_qa_full_score() {
        let qa = PopperianQA {
            r1_same_loss_curve: true,
            r2_same_final_weights: true,
            r3_same_eval_metrics: true,
            r4_environment_locked: true,
            c1_parses_as_rust: true,
            c2_type_checks: true,
            c3_no_unused_warnings: true,
            c4_links_correctly: true,
            x1_tests_pass_on_correct: true,
            x2_tests_fail_on_mutant: true,
            x3_assertions_meaningful: true,
            x4_no_tautologies: true,
            v1_branch_coverage_delta: true,
            v2_line_coverage_delta: true,
            v3_edge_cases_present: true,
            e1_vram_under_8gb: true,
            e2_training_under_4hrs: true,
            e3_inference_under_1s: true,
            g1_handles_generics: true,
            g2_handles_lifetimes: true,
            g3_handles_async: true,
            g4_handles_unsafe: true,
            g5_handles_macros: true,
            d1_test_names_descriptive: true,
            d2_comments_present: true,
            d3_proptest_strategies_clear: true,
        };
        assert_eq!(qa.score(), 100);
        assert_eq!(qa.grade(), QAGrade::APlus);
        assert_eq!(qa.passed_count(), 26);
    }

    #[test]
    fn test_popperian_qa_partial_score() {
        let mut qa = PopperianQA::new();
        // Set all reproducibility (20 pts)
        qa.r1_same_loss_curve = true;
        qa.r2_same_final_weights = true;
        qa.r3_same_eval_metrics = true;
        qa.r4_environment_locked = true;

        assert_eq!(qa.score(), 20);
        assert!(qa.reproducibility_passed());
        assert!(!qa.compilation_passed());
    }

    #[test]
    fn test_popperian_qa_category_checks() {
        let mut qa = PopperianQA::new();

        // Compilation only
        qa.c1_parses_as_rust = true;
        qa.c2_type_checks = true;
        qa.c3_no_unused_warnings = true;
        qa.c4_links_correctly = true;

        assert!(qa.compilation_passed());
        assert!(!qa.reproducibility_passed());
        assert!(!qa.correctness_passed());
    }

    #[test]
    fn test_popperian_qa_report_contains_sections() {
        let qa = PopperianQA::new();
        let report = qa.report();

        assert!(report.contains("# Popperian Falsification QA Report"));
        assert!(report.contains("## Reproducibility"));
        assert!(report.contains("## Compilation"));
        assert!(report.contains("## Correctness"));
        assert!(report.contains("## Coverage"));
        assert!(report.contains("## Efficiency"));
        assert!(report.contains("## Edge Cases"));
        assert!(report.contains("## Documentation"));
    }

    #[test]
    fn test_qa_grade_display() {
        assert_eq!(format!("{}", QAGrade::APlus), "A+ (Excellent)");
        assert_eq!(format!("{}", QAGrade::F), "F  (Failing)");
    }

    #[test]
    fn test_qa_grade_ordering() {
        assert!(QAGrade::APlus > QAGrade::A);
        assert!(QAGrade::A > QAGrade::BPlus);
        assert!(QAGrade::BPlus > QAGrade::B);
        assert!(QAGrade::B > QAGrade::C);
        assert!(QAGrade::C > QAGrade::F);
    }
}
