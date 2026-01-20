# Model Evaluation Framework Specification

**Issue:** https://github.com/paiml/aprender/issues/73
**Version:** 2.0.0
**Status:** Draft
**Target:** aprender 0.12.0 + entrenar 0.3.0
**Authors:** PAIML Team
**Last Updated:** 2026-01-21

---

## Executive Summary

This specification defines a comprehensive model evaluation framework for traditional ML, LLM fine-tuning, and knowledge distillation. It follows Toyota Production System principles for continuous improvement and includes 100-point Popperian falsification criteria to ensure scientific rigor.

---

## 1. Overview

### 1.1 Problem Statement

Current gaps in evaluation capabilities:

| Domain | Current State | Required |
|--------|--------------|----------|
| Classification | Partial | Complete with multi-class |
| Regression | Complete | Maintained |
| **Text Generation** | **None** | **BLEU, ROUGE, Perplexity, EM** |
| **Distillation** | **None** | **KL-Div, Agreement, Transfer** |
| **Structural** | **None** | **Domain-specific validation** |
| Drift Detection | Planned | Implemented |

### 1.2 Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        entrenar::eval Module                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌─────────────┐  │
│  │  metrics/     │  │  generation/  │  │ distillation/ │  │   drift/    │  │
│  │               │  │               │  │               │  │             │  │
│  │ - accuracy    │  │ - bleu        │  │ - kl_div      │  │ - ks_test   │  │
│  │ - precision   │  │ - rouge       │  │ - agreement   │  │ - chi_sq    │  │
│  │ - recall      │  │ - meteor      │  │ - transfer    │  │ - psi       │  │
│  │ - f1_score    │  │ - perplexity  │  │ - layer_sim   │  │ - callback  │  │
│  │ - confusion   │  │ - exact_match │  │ - temp_scale  │  │             │  │
│  │               │  │ - structural  │  │               │  │             │  │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘  └──────┬──────┘  │
│          │                  │                  │                  │         │
│          └──────────────────┴────────┬─────────┴──────────────────┘         │
│                                      │                                      │
│                                      ▼                                      │
│                    ┌─────────────────────────────────┐                      │
│                    │     EvalPipeline (Unified)      │                      │
│                    │  - configure()                  │                      │
│                    │  - evaluate()                   │                      │
│                    │  - compare()                    │                      │
│                    │  - report()                     │                      │
│                    └─────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Toyota Way Alignment

This specification embeds Toyota Production System (TPS) principles:

| TPS Principle | Application in Eval Framework |
|---------------|-------------------------------|
| **Kaizen** (改善) | Continuous metric improvement via A/B testing |
| **Genchi Genbutsu** (現地現物) | Go and see: inspect actual model outputs, not just scores |
| **Jidoka** (自働化) | Automation with human judgment: auto-flag anomalies for review |
| **Heijunka** (平準化) | Level evaluation load across test batches |
| **Poka-Yoke** (ポカヨケ) | Error-proofing: type-safe APIs prevent metric misuse |
| **Standardized Work** | Consistent evaluation protocols across all models |
| **Visual Management** | Dashboard-ready metrics with clear pass/fail indicators |
| **Pull System** | Evaluate on-demand, not batch-scheduled |

---

## 2. Text Generation Metrics

### 2.1 BLEU Score

**Definition:** Bilingual Evaluation Understudy measures n-gram precision between generated and reference text.

```rust
/// BLEU score implementation (Papineni et al., 2002)
///
/// BLEU = BP × exp(Σ wₙ log pₙ)
/// where BP = brevity penalty, pₙ = modified n-gram precision
pub struct BleuScore {
    /// Maximum n-gram order (default: 4)
    pub max_n: usize,
    /// Weights for each n-gram (default: uniform)
    pub weights: Vec<f64>,
    /// Smoothing method for zero counts
    pub smoothing: SmoothingMethod,
}

#[derive(Clone, Copy, Debug)]
pub enum SmoothingMethod {
    /// No smoothing (original BLEU)
    None,
    /// Add-k smoothing (Lin & Och, 2004)
    AddK(f64),
    /// Exponential smoothing (Chen & Cherry, 2014)
    Exponential,
}

impl BleuScore {
    pub fn compute(&self, hypothesis: &str, references: &[&str]) -> f64;
    pub fn corpus_bleu(&self, hypotheses: &[&str], references: &[Vec<&str>]) -> f64;
}
```

**Citation:**
> Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: a method for automatic evaluation of machine translation. *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (ACL)*, 311-318. https://doi.org/10.3115/1073083.1073135

### 2.2 ROUGE Score

**Definition:** Recall-Oriented Understudy for Gisting Evaluation measures recall of reference n-grams.

```rust
/// ROUGE score implementation (Lin, 2004)
pub struct RougeScore {
    pub variant: RougeVariant,
}

#[derive(Clone, Copy, Debug)]
pub enum RougeVariant {
    /// ROUGE-N: n-gram recall
    N(usize),
    /// ROUGE-L: Longest Common Subsequence
    L,
    /// ROUGE-W: Weighted LCS
    W { weight: f64 },
    /// ROUGE-S: Skip-bigram
    S { skip_distance: usize },
}

#[derive(Clone, Debug)]
pub struct RougeResult {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

impl RougeScore {
    pub fn compute(&self, hypothesis: &str, reference: &str) -> RougeResult;
}
```

**Citation:**
> Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries. *Text Summarization Branches Out*, 74-81. https://aclanthology.org/W04-1013/

### 2.3 METEOR Score

**Definition:** Metric for Evaluation of Translation with Explicit ORdering uses stemming, synonymy, and word order.

```rust
/// METEOR score implementation (Banerjee & Lavie, 2005)
pub struct MeteorScore {
    /// Language for stemming/synonyms
    pub language: Language,
    /// Weight parameters (α, β, γ)
    pub alpha: f64,  // precision weight
    pub beta: f64,   // recall weight
    pub gamma: f64,  // fragmentation penalty
}

impl MeteorScore {
    pub fn compute(&self, hypothesis: &str, reference: &str) -> f64;
}
```

**Citation:**
> Banerjee, S., & Lavie, A. (2005). METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. *Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization*, 65-72. https://aclanthology.org/W05-0909/

### 2.4 BERTScore

**Definition:** Computes semantic similarity using contextual embeddings.

```rust
/// BERTScore implementation (Zhang et al., 2020)
pub struct BertScore {
    /// Model for embeddings (default: microsoft/deberta-xlarge-mnli)
    pub model: String,
    /// Layer to extract embeddings from
    pub layer: i32,
    /// Use IDF weighting
    pub use_idf: bool,
}

#[derive(Clone, Debug)]
pub struct BertScoreResult {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

impl BertScore {
    pub fn compute(&self, hypothesis: &str, reference: &str) -> BertScoreResult;
}
```

**Citation:**
> Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating text generation with BERT. *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=SkeHuCVFDr

### 2.5 Perplexity

**Definition:** Exponentiated average negative log-likelihood per token.

```rust
/// Perplexity computation
///
/// PPL = exp(-1/N × Σ log P(xᵢ|x₁...xᵢ₋₁))
pub fn perplexity(logits: &Tensor, targets: &[u32], ignore_index: Option<u32>) -> f64;

/// Cross-entropy loss (related metric)
pub fn cross_entropy_loss(logits: &Tensor, targets: &[u32]) -> f64;
```

**Citation:**
> Jelinek, F., Mercer, R. L., Bahl, L. R., & Baker, J. K. (1977). Perplexity—a measure of the difficulty of speech recognition tasks. *Journal of the Acoustical Society of America*, 62(S1), S63. https://doi.org/10.1121/1.2016299

### 2.6 Exact Match

**Definition:** Binary indicator of exact string equality after normalization.

```rust
/// Exact match with configurable normalization
pub struct ExactMatch {
    /// Lowercase before comparison
    pub lowercase: bool,
    /// Strip whitespace
    pub strip: bool,
    /// Remove punctuation
    pub remove_punctuation: bool,
    /// Normalize unicode
    pub normalize_unicode: bool,
}

impl ExactMatch {
    pub fn compute(&self, hypothesis: &str, reference: &str) -> bool;
    pub fn corpus_em(&self, hypotheses: &[&str], references: &[&str]) -> f64;
}
```

**Citation:**
> Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD: 100,000+ questions for machine comprehension of text. *Proceedings of EMNLP*, 2383-2392. https://doi.org/10.18653/v1/D16-1264

### 2.7 Structural Validation (Domain-Specific)

**Definition:** Validates output conforms to expected structure (e.g., CLI help format).

```rust
/// Structural validation for CLI help generation
pub struct CliHelpValidator {
    /// Required sections
    pub required_sections: Vec<Section>,
    /// Flag format regex
    pub flag_pattern: Regex,
}

#[derive(Clone, Copy, Debug)]
pub enum Section {
    Description,
    Usage,
    Arguments,
    Options,
    Examples,
}

#[derive(Clone, Debug)]
pub struct StructuralResult {
    pub has_description: bool,
    pub has_usage: bool,
    pub has_options: bool,
    pub has_help_flag: bool,
    pub valid_flag_format: bool,
    pub score: f64,
}

impl CliHelpValidator {
    pub fn validate(&self, output: &str) -> StructuralResult;
}

/// Content accuracy: checks for hallucinated vs real flags
pub fn content_accuracy(generated: &str, reference: &str) -> ContentResult;

#[derive(Clone, Debug)]
pub struct ContentResult {
    pub precision: f64,      // real flags / generated flags
    pub recall: f64,         // found flags / reference flags
    pub hallucination_rate: f64,  // invented flags / generated flags
    pub f1: f64,
}
```

---

## 3. Knowledge Distillation Metrics

### 3.1 KL Divergence

**Definition:** Measures how student distribution diverges from teacher distribution.

```rust
/// KL Divergence for distillation
///
/// D_KL(T||S) = Σ T(x) log(T(x)/S(x))
pub fn kl_divergence(
    teacher_logits: &Tensor,
    student_logits: &Tensor,
    temperature: f64,
) -> f64;

/// Symmetric KL (Jensen-Shannon divergence)
pub fn js_divergence(
    teacher_logits: &Tensor,
    student_logits: &Tensor,
) -> f64;
```

**Citation:**
> Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*. https://arxiv.org/abs/1503.02531

### 3.2 Temperature Scaling Analysis

**Definition:** Analyzes optimal temperature for knowledge transfer.

```rust
/// Temperature scaling for distillation
pub struct TemperatureAnalysis {
    /// Temperatures to evaluate
    pub temperatures: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct TempResult {
    pub temperature: f64,
    pub kl_div: f64,
    pub student_accuracy: f64,
    pub transfer_efficiency: f64,
}

impl TemperatureAnalysis {
    pub fn analyze(
        &self,
        teacher: &dyn Model,
        student: &dyn Model,
        data: &Dataset,
    ) -> Vec<TempResult>;

    pub fn optimal_temperature(&self, results: &[TempResult]) -> f64;
}
```

**Citation:**
> Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of ICML*, 1321-1330. https://proceedings.mlr.press/v70/guo17a.html

### 3.3 Student-Teacher Agreement

**Definition:** Measures prediction agreement between student and teacher.

```rust
/// Agreement metrics for distillation
pub struct AgreementMetrics;

impl AgreementMetrics {
    /// Top-1 prediction agreement
    pub fn top1_agreement(
        teacher_preds: &[usize],
        student_preds: &[usize],
    ) -> f64;

    /// Top-k prediction overlap
    pub fn topk_agreement(
        teacher_logits: &Tensor,
        student_logits: &Tensor,
        k: usize,
    ) -> f64;

    /// Rank correlation (Spearman)
    pub fn rank_correlation(
        teacher_logits: &Tensor,
        student_logits: &Tensor,
    ) -> f64;
}
```

**Citation:**
> Stanton, S., Izmailov, P., Kirichenko, P., Alemi, A. A., & Wilson, A. G. (2021). Does knowledge distillation really work? *Advances in Neural Information Processing Systems (NeurIPS)*, 34. https://arxiv.org/abs/2106.05945

### 3.4 Layer-wise Similarity

**Definition:** Compares intermediate representations between teacher and student.

```rust
/// Layer-wise representation similarity
pub struct LayerSimilarity {
    /// Similarity metric
    pub metric: SimilarityMetric,
}

#[derive(Clone, Copy, Debug)]
pub enum SimilarityMetric {
    /// Centered Kernel Alignment
    CKA,
    /// Canonical Correlation Analysis
    CCA,
    /// Cosine similarity
    Cosine,
    /// Projection-weighted CCA (Morcos et al., 2018)
    PWCCA,
}

impl LayerSimilarity {
    pub fn compute(
        &self,
        teacher_activations: &Tensor,
        student_activations: &Tensor,
    ) -> f64;

    pub fn layer_mapping(
        &self,
        teacher: &dyn Model,
        student: &dyn Model,
        data: &Dataset,
    ) -> Vec<(usize, usize, f64)>;  // (teacher_layer, student_layer, similarity)
}
```

**Citation:**
> Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. *Proceedings of ICML*, 3519-3529. https://proceedings.mlr.press/v97/kornblith19a.html

### 3.5 Knowledge Transfer Efficiency

**Definition:** Measures how much teacher knowledge transfers to student relative to capacity.

```rust
/// Transfer efficiency metrics
pub struct TransferEfficiency;

impl TransferEfficiency {
    /// Efficiency = (student_score - baseline) / (teacher_score - baseline)
    pub fn compute(
        teacher_score: f64,
        student_score: f64,
        baseline_score: f64,
    ) -> f64;

    /// Parameter efficiency: performance per parameter
    pub fn parameter_efficiency(
        score: f64,
        param_count: usize,
    ) -> f64;

    /// Compression ratio with quality retention
    pub fn compression_quality(
        teacher_params: usize,
        student_params: usize,
        teacher_score: f64,
        student_score: f64,
    ) -> CompressionResult;
}

#[derive(Clone, Debug)]
pub struct CompressionResult {
    pub compression_ratio: f64,    // teacher_params / student_params
    pub quality_retention: f64,    // student_score / teacher_score
    pub efficiency_score: f64,     // quality_retention × compression_ratio
}
```

**Citation:**
> Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., & Liu, Q. (2020). TinyBERT: Distilling BERT for natural language understanding. *Findings of EMNLP*, 4163-4174. https://doi.org/10.18653/v1/2020.findings-emnlp.372

---

## 4. Fine-Tuning Evaluation Pipeline

### 4.1 LoRA/QLoRA Specific Metrics

```rust
/// Fine-tuning evaluation for LoRA/QLoRA
pub struct FineTuneEvaluator {
    /// Base model for comparison
    pub base_model: PathBuf,
    /// Adapter path
    pub adapter_path: PathBuf,
    /// Evaluation config
    pub config: FineTuneEvalConfig,
}

#[derive(Clone, Debug)]
pub struct FineTuneEvalConfig {
    /// Text generation metrics to compute
    pub generation_metrics: Vec<GenerationMetric>,
    /// Held-out test set
    pub test_data: PathBuf,
    /// Compare to base model (pre-fine-tune)
    pub compare_base: bool,
    /// Number of samples for evaluation
    pub num_samples: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum GenerationMetric {
    Bleu { max_n: usize },
    RougeL,
    Meteor,
    ExactMatch,
    Perplexity,
    Structural,
    ContentAccuracy,
}

#[derive(Clone, Debug)]
pub struct FineTuneResult {
    /// Per-metric scores
    pub scores: HashMap<GenerationMetric, f64>,
    /// Comparison to base model (if enabled)
    pub base_comparison: Option<BaseComparison>,
    /// Train/test gap (generalization check)
    pub train_test_gap: f64,
    /// Inference latency
    pub latency_ms: f64,
}

#[derive(Clone, Debug)]
pub struct BaseComparison {
    pub base_scores: HashMap<GenerationMetric, f64>,
    pub improvement: HashMap<GenerationMetric, f64>,
    pub degradation: HashMap<GenerationMetric, f64>,
}
```

### 4.2 Evaluation Protocol

Following Toyota Way's standardized work principle:

```rust
/// Standardized evaluation protocol
pub struct EvalProtocol {
    /// Phase 1: Structural validation
    pub structural: StructuralPhase,
    /// Phase 2: Content accuracy
    pub content: ContentPhase,
    /// Phase 3: Generation quality
    pub generation: GenerationPhase,
    /// Phase 4: Generalization check
    pub generalization: GeneralizationPhase,
}

impl EvalProtocol {
    /// Execute full evaluation protocol
    pub fn execute(&self, model: &dyn Model, test_data: &Dataset) -> ProtocolResult;

    /// Jidoka: Stop and flag if quality gate fails
    pub fn with_quality_gates(self, gates: Vec<QualityGate>) -> Self;
}

#[derive(Clone, Debug)]
pub struct QualityGate {
    pub metric: GenerationMetric,
    pub threshold: f64,
    pub action: GateAction,
}

#[derive(Clone, Copy, Debug)]
pub enum GateAction {
    /// Warn but continue
    Warn,
    /// Stop evaluation, require human review (Jidoka)
    Stop,
    /// Fail the entire evaluation
    Fail,
}
```

---

## 5. Classification Metrics (Existing, Enhanced)

### 5.1 Multi-class Support

```rust
/// Enhanced classification metrics with multi-class support
pub mod classification {
    /// Averaging strategy for multi-class metrics
    #[derive(Clone, Copy, Debug)]
    pub enum Average {
        Macro,
        Micro,
        Weighted,
        None,
    }

    pub fn accuracy(y_pred: &[usize], y_true: &[usize]) -> f64;
    pub fn precision(y_pred: &[usize], y_true: &[usize], average: Average) -> f64;
    pub fn recall(y_pred: &[usize], y_true: &[usize], average: Average) -> f64;
    pub fn f1_score(y_pred: &[usize], y_true: &[usize], average: Average) -> f64;
    pub fn confusion_matrix(y_pred: &[usize], y_true: &[usize], n_classes: usize) -> Matrix<usize>;
    pub fn classification_report(y_pred: &[usize], y_true: &[usize]) -> String;

    /// ROC-AUC for binary and multi-class
    pub fn roc_auc(y_true: &[usize], y_scores: &[f64], multi_class: MultiClass) -> f64;

    #[derive(Clone, Copy, Debug)]
    pub enum MultiClass {
        OvR,  // One-vs-Rest
        OvO,  // One-vs-One
    }
}
```

---

## 6. Drift Detection (Existing, Maintained)

```rust
/// Statistical tests for drift detection
pub mod drift {
    pub enum DriftTest {
        KS { threshold: f64 },
        ChiSquare { threshold: f64 },
        PSI { threshold: f64 },
    }

    pub struct DriftDetector { /* ... */ }

    impl DriftDetector {
        pub fn set_baseline(&mut self, data: &Matrix<f64>);
        pub fn check(&self, current: &Matrix<f64>) -> Vec<DriftResult>;
        pub fn on_drift<F>(&mut self, callback: F);
    }
}
```

---

## 7. File Structure

```
entrenar/src/
├── eval/
│   ├── mod.rs                    # Module exports
│   ├── pipeline.rs               # EvalPipeline, EvalProtocol
│   └── quality_gates.rs          # QualityGate, Jidoka implementation
├── metrics/
│   ├── mod.rs                    # Re-exports
│   ├── classification.rs         # Accuracy, F1, etc.
│   ├── regression.rs             # R², MSE, MAE, RMSE
│   └── clustering.rs             # Silhouette, inertia
├── generation/
│   ├── mod.rs                    # Re-exports
│   ├── bleu.rs                   # BLEU implementation
│   ├── rouge.rs                  # ROUGE implementation
│   ├── meteor.rs                 # METEOR implementation
│   ├── bertscore.rs              # BERTScore (feature-gated)
│   ├── perplexity.rs             # Perplexity computation
│   ├── exact_match.rs            # Exact match
│   └── structural.rs             # Domain-specific validation
├── distillation/
│   ├── mod.rs                    # Re-exports
│   ├── kl_divergence.rs          # KL/JS divergence
│   ├── temperature.rs            # Temperature scaling analysis
│   ├── agreement.rs              # Student-teacher agreement
│   ├── layer_similarity.rs       # CKA, CCA, PWCCA
│   └── transfer.rs               # Transfer efficiency
└── drift/
    ├── mod.rs                    # Re-exports
    ├── detector.rs               # DriftDetector
    └── tests.rs                  # KS, Chi-square, PSI
```

---

## 8. Feature Flags

```toml
[features]
default = ["generation"]
generation = []                    # Text generation metrics
distillation = ["generation"]      # Distillation metrics
bertscore = ["generation", "dep:candle-transformers"]  # BERTScore (requires model)
drift = []                         # Drift detection
full = ["generation", "distillation", "bertscore", "drift"]
```

---

## 9. Dependencies

| Crate | Version | Purpose | Feature |
|-------|---------|---------|---------|
| trueno | 0.13+ | Tensor operations | - |
| unicode-segmentation | 1.10 | Tokenization | generation |
| regex | 1.10 | Pattern matching | generation |
| statrs | 0.17 | Statistical tests | drift |
| candle-transformers | 0.4 | BERTScore embeddings | bertscore |

---

## 10. Implementation Tickets

| ID | Task | Hours | Priority |
|----|------|-------|----------|
| ENT-100-1 | BLEU implementation + tests | 12 | P0 |
| ENT-100-2 | ROUGE-L/ROUGE-N implementation | 8 | P0 |
| ENT-100-3 | Perplexity + cross-entropy | 4 | P0 |
| ENT-100-4 | Exact match + normalization | 4 | P0 |
| ENT-100-5 | Structural validation (CLI help) | 8 | P0 |
| ENT-100-6 | Content accuracy + hallucination | 8 | P0 |
| ENT-100-7 | KL divergence + temperature | 8 | P1 |
| ENT-100-8 | Student-teacher agreement | 6 | P1 |
| ENT-100-9 | Layer similarity (CKA) | 12 | P1 |
| ENT-100-10 | Transfer efficiency | 4 | P1 |
| ENT-100-11 | EvalProtocol + quality gates | 12 | P0 |
| ENT-100-12 | METEOR implementation | 8 | P2 |
| ENT-100-13 | BERTScore integration | 16 | P2 |
| ENT-100-14 | Property tests (1000+ cases) | 16 | P0 |
| ENT-100-15 | Documentation + examples | 12 | P0 |

**Total:** 138 hours

---

## 11. Quality Requirements

| Requirement | Target | Verification |
|-------------|--------|--------------|
| Test coverage | ≥95% | cargo llvm-cov |
| Mutation score | ≥85% | cargo mutants |
| Property tests | 1000+ iterations | proptest |
| Numerical stability | ε < 1e-6 | Gradient checking |
| API compatibility | semver | cargo semver-checks |
| Documentation | 100% public items | cargo doc --deny warnings |

---

## 12. Toyota Way Implementation Checklist

### 12.1 Continuous Improvement (Kaizen)

- [ ] Metric versioning for A/B testing
- [ ] Automated regression detection
- [ ] Feedback loop from production to evaluation
- [ ] Monthly metric review process

### 12.2 Go and See (Genchi Genbutsu)

- [ ] Sample output inspection in every eval run
- [ ] Human-readable eval reports (not just numbers)
- [ ] Random sample audit trail
- [ ] "Worst performing" sample highlighting

### 12.3 Automation with Human Touch (Jidoka)

- [ ] Quality gates with configurable thresholds
- [ ] Automatic stop on anomaly detection
- [ ] Human review queue for edge cases
- [ ] Escalation protocol for failures

### 12.4 Visual Management

- [ ] Dashboard-ready JSON output
- [ ] Pass/fail/warn color coding
- [ ] Trend visualization data
- [ ] Metric correlation matrix

### 12.5 Standardized Work

- [ ] Documented evaluation protocol
- [ ] Reproducible evaluation seeds
- [ ] Versioned test datasets
- [ ] Canonical metric implementations

---

## 13. Popperian Falsification Criteria (100 Points)

Following Karl Popper's philosophy of science, each criterion below represents a testable hypothesis that could **falsify** the evaluation framework. If any criterion fails, the framework is demonstrably inadequate.

### 13.1 BLEU Score Correctness (F001-F015)

| ID | Falsification Criterion | Test Method |
|----|------------------------|-------------|
| F001 | BLEU(identical, identical) ≠ 1.0 | Property test |
| F002 | BLEU(empty, reference) ≠ 0.0 | Unit test |
| F003 | BLEU score outside [0, 1] | Property test with random inputs |
| F004 | BLEU not symmetric in references | Multi-reference test |
| F005 | BLEU-1 ≠ unigram precision × BP | Mathematical verification |
| F006 | BLEU with smoothing < BLEU without for short texts | Known short sentence test |
| F007 | Corpus BLEU ≠ geometric mean of sentence BLEU | Corpus test |
| F008 | BLEU changes with whitespace normalization disabled | Whitespace sensitivity test |
| F009 | BLEU(A,B) + BLEU(B,A) behaves unexpectedly | Symmetry analysis |
| F010 | BLEU fails on Unicode (emoji, CJK, RTL) | Unicode corpus test |
| F011 | BLEU computation not O(n) in hypothesis length | Benchmark test |
| F012 | BLEU precision overflow on long documents | 100K token stress test |
| F013 | BLEU brevity penalty incorrect for |hyp| > |ref| | Length boundary test |
| F014 | BLEU NaN/Inf on edge cases | Fuzzing test |
| F015 | BLEU differs from sacrebleu reference >0.01 | Cross-validation test |

### 13.2 ROUGE Score Correctness (F016-F025)

| ID | Falsification Criterion | Test Method |
|----|------------------------|-------------|
| F016 | ROUGE-L(identical, identical) ≠ 1.0 | Property test |
| F017 | ROUGE-L recall > 1.0 | Property test |
| F018 | ROUGE-L precision > 1.0 | Property test |
| F019 | ROUGE-L LCS length > min(|hyp|, |ref|) | Property test |
| F020 | ROUGE-N for n > |text| not handled | Edge case test |
| F021 | ROUGE F1 ≠ harmonic mean of P and R | Mathematical verification |
| F022 | ROUGE-S skip distance negative | Input validation test |
| F023 | ROUGE not monotonic in overlap | Monotonicity test |
| F024 | ROUGE fails on empty inputs | Edge case test |
| F025 | ROUGE differs from rouge-score library >0.01 | Cross-validation test |

### 13.3 Perplexity Correctness (F026-F035)

| ID | Falsification Criterion | Test Method |
|----|------------------------|-------------|
| F026 | Perplexity ≤ 0 | Property test |
| F027 | Perplexity(uniform) ≠ vocab_size | Known distribution test |
| F028 | Perplexity increases when model improves | Correlation test |
| F029 | Perplexity(deterministic) ≠ 1.0 | Degenerate model test |
| F030 | log(perplexity) ≠ cross-entropy | Mathematical verification |
| F031 | Perplexity overflow on large vocab | 100K vocab test |
| F032 | Perplexity underflow on confident predictions | Near-zero logit test |
| F033 | Perplexity NaN when logit = -inf | Edge case test |
| F034 | Perplexity changes with padding tokens | Mask verification test |
| F035 | Perplexity not comparable across tokenizers | Documentation verification |

### 13.4 Exact Match Correctness (F036-F042)

| ID | Falsification Criterion | Test Method |
|----|------------------------|-------------|
| F036 | EM(identical, identical) ≠ true | Property test |
| F037 | EM("a", "A") with lowercase=true ≠ true | Normalization test |
| F038 | EM(" a ", "a") with strip=true ≠ true | Whitespace test |
| F039 | EM not reflexive | Property test |
| F040 | EM not symmetric | Property test |
| F041 | Corpus EM not in [0, 1] | Property test |
| F042 | EM fails on Unicode normalization | NFC/NFD test |

### 13.5 KL Divergence Correctness (F043-F052)

| ID | Falsification Criterion | Test Method |
|----|------------------------|-------------|
| F043 | KL(P, P) ≠ 0 | Property test |
| F044 | KL(P, Q) < 0 | Property test |
| F045 | KL not finite when Q=0 and P>0 | Edge case test |
| F046 | KL symmetric (should NOT be) | Asymmetry verification |
| F047 | JS(P, Q) ≠ JS(Q, P) | Symmetry test |
| F048 | JS outside [0, 1] (for log base 2) | Property test |
| F049 | Temperature scaling changes KL incorrectly | Temperature sweep test |
| F050 | KL gradient incorrect | Gradient checking |
| F051 | KL overflow on extreme logits | Numerical stability test |
| F052 | KL differs from torch.nn.KLDivLoss >1e-5 | Cross-validation test |

### 13.6 Structural Validation Correctness (F053-F062)

| ID | Falsification Criterion | Test Method |
|----|------------------------|-------------|
| F053 | Valid CLI help scores < 0.8 structural | Known good samples |
| F054 | Invalid CLI help scores > 0.5 structural | Known bad samples |
| F055 | Missing "Usage:" not detected | Negative test |
| F056 | Missing "Options:" not detected | Negative test |
| F057 | Invalid flag format "-x--long" accepted | Regex test |
| F058 | Valid flag format "-x, --long" rejected | Regex test |
| F059 | Structural score outside [0, 1] | Property test |
| F060 | Content accuracy penalizes real flags | False negative test |
| F061 | Content accuracy misses hallucinated flags | False positive test |
| F062 | Hallucination rate > 1.0 | Property test |

### 13.7 Agreement Metrics Correctness (F063-F070)

| ID | Falsification Criterion | Test Method |
|----|------------------------|-------------|
| F063 | Top-1 agreement(identical) ≠ 1.0 | Property test |
| F064 | Top-k agreement > 1.0 | Property test |
| F065 | Top-k agreement < 0.0 | Property test |
| F066 | Rank correlation outside [-1, 1] | Property test |
| F067 | Rank correlation(identical) ≠ 1.0 | Property test |
| F068 | Agreement increases when predictions diverge | Monotonicity test |
| F069 | k > vocab_size not handled | Edge case test |
| F070 | Agreement fails on tied scores | Tie-breaking test |

### 13.8 Layer Similarity Correctness (F071-F078)

| ID | Falsification Criterion | Test Method |
|----|------------------------|-------------|
| F071 | CKA(X, X) ≠ 1.0 | Property test |
| F072 | CKA outside [0, 1] | Property test |
| F073 | CKA(X, Y) ≠ CKA(Y, X) | Symmetry test |
| F074 | CKA not invariant to orthogonal transform | Invariance test |
| F075 | CKA not invariant to isotropic scaling | Invariance test |
| F076 | CCA dimensions mismatch not handled | Error handling test |
| F077 | PWCCA weights negative | Property test |
| F078 | Layer similarity fails on batch size 1 | Edge case test |

### 13.9 Transfer Efficiency Correctness (F079-F085)

| ID | Falsification Criterion | Test Method |
|----|------------------------|-------------|
| F079 | Efficiency > 1.0 when student > teacher | Expected behavior |
| F080 | Efficiency < 0 when student < baseline | Property test |
| F081 | Compression ratio ≤ 0 | Property test |
| F082 | Quality retention outside [0, ∞) | Property test |
| F083 | Efficiency undefined when teacher = baseline | Edge case test |
| F084 | Parameter count = 0 not handled | Error handling test |
| F085 | Efficiency not monotonic in student quality | Monotonicity test |

### 13.10 API Contract Violations (F086-F092)

| ID | Falsification Criterion | Test Method |
|----|------------------------|-------------|
| F086 | Metric enum variants missing from compute() | Exhaustiveness test |
| F087 | EvalConfig fields not validated | Invalid config test |
| F088 | Result structs not serializable | Serde roundtrip test |
| F089 | Async evaluation deadlocks | Timeout test |
| F090 | Parallel evaluation gives different results | Determinism test |
| F091 | Quality gate thresholds not enforced | Threshold boundary test |
| F092 | Callback errors not propagated | Error handling test |

### 13.11 Performance Violations (F093-F097)

| ID | Falsification Criterion | Test Method |
|----|------------------------|-------------|
| F093 | BLEU > 100ms for 1000 tokens | Benchmark test |
| F094 | Perplexity > 10ms per sample | Benchmark test |
| F095 | Memory leak in repeated evaluation | Valgrind/miri test |
| F096 | Evaluation not parallelizable | Rayon compatibility test |
| F097 | GPU evaluation slower than CPU for small batches | Crossover point test |

### 13.12 Security/Safety Violations (F098-F100)

| ID | Falsification Criterion | Test Method |
|----|------------------------|-------------|
| F098 | Metric accepts arbitrary code in reference | Injection test |
| F099 | Large input causes OOM without error | Memory limit test |
| F100 | Evaluation leaks training data in output | Information flow test |

---

## 14. Falsification Test Implementation

```rust
#[cfg(test)]
mod falsification_tests {
    use super::*;
    use proptest::prelude::*;

    // F001: BLEU(identical, identical) = 1.0
    proptest! {
        #[test]
        fn f001_bleu_identical(s in "\\PC{1,100}") {
            let bleu = BleuScore::default();
            let score = bleu.compute(&s, &[&s]);
            prop_assert!((score - 1.0).abs() < 1e-6,
                "F001 FALSIFIED: BLEU(identical) = {}", score);
        }
    }

    // F002: BLEU(empty, reference) = 0.0
    #[test]
    fn f002_bleu_empty() {
        let bleu = BleuScore::default();
        let score = bleu.compute("", &["reference text"]);
        assert!((score - 0.0).abs() < 1e-6,
            "F002 FALSIFIED: BLEU(empty) = {}", score);
    }

    // F003: BLEU in [0, 1]
    proptest! {
        #[test]
        fn f003_bleu_bounded(
            hyp in "\\PC{0,100}",
            refs in prop::collection::vec("\\PC{1,100}", 1..5)
        ) {
            let bleu = BleuScore::default();
            let ref_strs: Vec<&str> = refs.iter().map(|s| s.as_str()).collect();
            let score = bleu.compute(&hyp, &ref_strs);
            prop_assert!(score >= 0.0 && score <= 1.0,
                "F003 FALSIFIED: BLEU = {}", score);
        }
    }

    // F043: KL(P, P) = 0
    proptest! {
        #[test]
        fn f043_kl_self_zero(logits in prop::collection::vec(-10.0f64..10.0, 10..100)) {
            let tensor = Tensor::from_slice(&logits);
            let kl = kl_divergence(&tensor, &tensor, 1.0);
            prop_assert!(kl.abs() < 1e-6,
                "F043 FALSIFIED: KL(P,P) = {}", kl);
        }
    }

    // ... 96 more falsification tests ...
}
```

---

## 15. Example Usage

### 15.1 Fine-Tuning Evaluation

```rust
use entrenar::eval::{FineTuneEvaluator, FineTuneEvalConfig, GenerationMetric};
use entrenar::generation::{BleuScore, RougeScore, RougeVariant};

fn evaluate_cli_help_finetune() -> Result<()> {
    let evaluator = FineTuneEvaluator::new(
        "models/qwen-1.5b-base.apr",
        "models/cli-help-adapter.lora",
        FineTuneEvalConfig {
            generation_metrics: vec![
                GenerationMetric::Bleu { max_n: 4 },
                GenerationMetric::RougeL,
                GenerationMetric::ExactMatch,
                GenerationMetric::Structural,
                GenerationMetric::ContentAccuracy,
            ],
            test_data: "data/cli-help-test.jsonl".into(),
            compare_base: true,
            num_samples: 100,
        },
    );

    let result = evaluator.evaluate()?;

    // Kaizen: Log for continuous improvement
    println!("=== Fine-Tune Evaluation Results ===");
    println!("BLEU-4:           {:.2}%", result.scores[&GenerationMetric::Bleu { max_n: 4 }] * 100.0);
    println!("ROUGE-L F1:       {:.2}%", result.scores[&GenerationMetric::RougeL] * 100.0);
    println!("Exact Match:      {:.2}%", result.scores[&GenerationMetric::ExactMatch] * 100.0);
    println!("Structural:       {:.2}%", result.scores[&GenerationMetric::Structural] * 100.0);
    println!("Content Accuracy: {:.2}%", result.scores[&GenerationMetric::ContentAccuracy] * 100.0);

    // Genchi Genbutsu: Inspect worst samples
    if let Some(comparison) = &result.base_comparison {
        println!("\n=== Improvement over Base Model ===");
        for (metric, improvement) in &comparison.improvement {
            println!("{:?}: +{:.2}%", metric, improvement * 100.0);
        }
    }

    // Jidoka: Quality gate check
    if result.scores[&GenerationMetric::ContentAccuracy] < 0.70 {
        eprintln!("WARNING: Content accuracy below 70% threshold");
        eprintln!("Human review required before deployment");
        return Err(Error::QualityGateFailed("content_accuracy"));
    }

    Ok(())
}
```

### 15.2 Distillation Evaluation

```rust
use entrenar::distillation::{
    kl_divergence, AgreementMetrics, LayerSimilarity, SimilarityMetric,
    TemperatureAnalysis, TransferEfficiency,
};

fn evaluate_distillation(
    teacher: &dyn Model,
    student: &dyn Model,
    test_data: &Dataset,
) -> Result<DistillationReport> {
    // KL divergence at different temperatures
    let temp_analysis = TemperatureAnalysis {
        temperatures: vec![1.0, 2.0, 4.0, 8.0, 16.0],
    };
    let temp_results = temp_analysis.analyze(teacher, student, test_data);
    let optimal_temp = temp_analysis.optimal_temperature(&temp_results);

    // Agreement metrics
    let (teacher_logits, student_logits) = get_logits(teacher, student, test_data);
    let top1 = AgreementMetrics::top1_agreement(&teacher_preds, &student_preds);
    let top5 = AgreementMetrics::topk_agreement(&teacher_logits, &student_logits, 5);
    let rank_corr = AgreementMetrics::rank_correlation(&teacher_logits, &student_logits);

    // Layer similarity (CKA)
    let layer_sim = LayerSimilarity { metric: SimilarityMetric::CKA };
    let mapping = layer_sim.layer_mapping(teacher, student, test_data);

    // Transfer efficiency
    let efficiency = TransferEfficiency::compute(
        teacher_accuracy,
        student_accuracy,
        random_baseline,
    );

    let compression = TransferEfficiency::compression_quality(
        teacher.param_count(),
        student.param_count(),
        teacher_accuracy,
        student_accuracy,
    );

    Ok(DistillationReport {
        optimal_temperature: optimal_temp,
        top1_agreement: top1,
        top5_agreement: top5,
        rank_correlation: rank_corr,
        layer_mapping: mapping,
        transfer_efficiency: efficiency,
        compression_ratio: compression.compression_ratio,
        quality_retention: compression.quality_retention,
    })
}
```

---

## 16. References

### Text Generation Metrics

1. Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: a method for automatic evaluation of machine translation. *ACL*, 311-318. https://doi.org/10.3115/1073083.1073135

2. Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries. *Text Summarization Branches Out*, 74-81. https://aclanthology.org/W04-1013/

3. Banerjee, S., & Lavie, A. (2005). METEOR: An automatic metric for MT evaluation. *ACL Workshop*, 65-72. https://aclanthology.org/W05-0909/

4. Zhang, T., et al. (2020). BERTScore: Evaluating text generation with BERT. *ICLR*. https://openreview.net/forum?id=SkeHuCVFDr

5. Post, M. (2018). A call for clarity in reporting BLEU scores. *WMT*, 186-191. https://doi.org/10.18653/v1/W18-6319

### Knowledge Distillation

6. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv:1503.02531*. https://arxiv.org/abs/1503.02531

7. Gou, J., et al. (2021). Knowledge distillation: A survey. *IJCV*, 129, 1789-1819. https://doi.org/10.1007/s11263-021-01453-z

8. Stanton, S., et al. (2021). Does knowledge distillation really work? *NeurIPS*, 34. https://arxiv.org/abs/2106.05945

9. Kornblith, S., et al. (2019). Similarity of neural network representations revisited. *ICML*, 3519-3529. https://proceedings.mlr.press/v97/kornblith19a.html

10. Jiao, X., et al. (2020). TinyBERT: Distilling BERT for natural language understanding. *EMNLP Findings*, 4163-4174. https://doi.org/10.18653/v1/2020.findings-emnlp.372

### Fine-Tuning Evaluation

11. Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR*. https://openreview.net/forum?id=nZeVKeeFYf9

12. Dettmers, T., et al. (2023). QLoRA: Efficient finetuning of quantized LLMs. *NeurIPS*. https://arxiv.org/abs/2305.14314

13. Lester, B., Al-Rfou, R., & Constant, N. (2021). The power of scale for parameter-efficient prompt tuning. *EMNLP*, 3045-3059. https://doi.org/10.18653/v1/2021.emnlp-main.243

### Methodology

14. Popper, K. (1959). *The Logic of Scientific Discovery*. Routledge. ISBN: 978-0415278447

15. Liker, J. K. (2004). *The Toyota Way: 14 Management Principles*. McGraw-Hill. ISBN: 978-0071392310

16. Ohno, T. (1988). *Toyota Production System: Beyond Large-Scale Production*. Productivity Press. ISBN: 978-0915299140

### Statistical Tests

17. Massey, F. J. (1951). The Kolmogorov-Smirnov test for goodness of fit. *JASA*, 46(253), 68-78. https://doi.org/10.1080/01621459.1951.10500769

18. Pearson, K. (1900). On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling. *Philosophical Magazine*, 50(302), 157-175.

---

## 17. Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-03 | Initial spec: classification, drift |
| 2.0.0 | 2026-01 | Added: text generation, distillation, Toyota Way, 100-point falsification |

---

*Specification follows ISO/IEC/IEEE 29148:2018 format*
*Quality gates enforced by certeza*
*Generated via pmat workflow*
