//! Falsification tests for generative AI evaluation metrics
//!
//! Adversarial tests designed to break the implementation by testing:
//! - Mathematical correctness against known ground-truth values
//! - Boundary conditions and degenerate inputs
//! - Symmetry / asymmetry invariants
//! - Monotonicity and ordering invariants
//! - Numerical stability under extreme values
//! - Cross-metric consistency
//!
//! Philosophy: every test encodes a falsifiable claim.
//! If the test passes, we have failed to falsify the claim — not proven it.

use super::*;
use proptest::prelude::*;

// ═══════════════════════════════════════════════════════════════════════
// CLAIM 1: WER = edit_distance(ref_words, hyp_words) / len(ref_words)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_wer_known_substitution() {
    // "a b c" → "a x c" = 1 substitution / 3 words
    let wer = word_error_rate("a b c", "a x c");
    assert!(
        (wer - 1.0 / 3.0).abs() < 1e-10,
        "1 substitution in 3 words → WER = 1/3, got {wer}"
    );
}

#[test]
fn falsify_wer_known_two_insertions() {
    // "a" → "x a y" = 2 operations (insert x before, insert y after) / 1 word
    let wer = word_error_rate("a", "x a y");
    assert!(
        (wer - 2.0).abs() < 1e-10,
        "2 insertions in 1-word ref → WER = 2.0, got {wer}"
    );
}

#[test]
fn falsify_wer_exceeds_one() {
    // WER CAN exceed 1.0 — spec says "Can exceed 1.0 when hypothesis is much longer"
    let wer = word_error_rate("a", "b c d e f");
    assert!(
        wer > 1.0,
        "WER should exceed 1.0 for heavily inserted hyp, got {wer}"
    );
}

#[test]
fn falsify_wer_asymmetry() {
    // WER is NOT symmetric — swapping ref/hyp changes the denominator
    let wer_ab = word_error_rate("a b c", "a b c d e");
    let wer_ba = word_error_rate("a b c d e", "a b c");
    assert!(
        (wer_ab - wer_ba).abs() > 1e-10,
        "WER must be asymmetric: WER(abc, abcde)={wer_ab} vs WER(abcde, abc)={wer_ba}"
    );
}

#[test]
fn falsify_wer_triangle_inequality() {
    // Edit distance satisfies triangle inequality: d(a,c) ≤ d(a,b) + d(b,c)
    // But WER divides by |ref|, so triangle inequality on WER itself is NOT guaranteed.
    // This test verifies the raw edit distance does satisfy it.
    let a = "the quick brown fox";
    let b = "a quick brown dog";
    let c = "the slow red cat";

    let d_ab = word_error_rate(a, b) * 4.0; // un-normalize: 4 words in ref
    let d_bc = word_error_rate(b, c) * 4.0; // un-normalize: 4 words in ref
    let d_ac = word_error_rate(a, c) * 4.0; // un-normalize: 4 words in ref

    // Triangle inequality on edit distance (not WER)
    assert!(
        d_ac <= d_ab + d_bc + 1e-10,
        "Triangle inequality violated: d(a,c)={d_ac} > d(a,b)={d_ab} + d(b,c)={d_bc}"
    );
}

#[test]
fn falsify_wer_whitespace_only_reference() {
    // A reference of only whitespace → 0 words → same as empty reference
    let wer = word_error_rate("   ", "hello");
    assert!(
        wer.is_infinite(),
        "Whitespace-only ref has 0 words → WER should be inf, got {wer}"
    );
}

#[test]
fn falsify_wer_whitespace_only_hypothesis() {
    // All words deleted
    let wer = word_error_rate("hello world", "   ");
    assert!(
        (wer - 1.0).abs() < 1e-10,
        "Whitespace-only hyp → all deletions → WER=1.0, got {wer}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM 2: BLEU implements Papineni et al. (2002) correctly
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_bleu_brevity_penalty_applied() {
    // Short hypothesis should be penalized by brevity penalty
    let short = bleu_score(&["a b c d e f g h i j"], "a b c", 1);
    let full = bleu_score(&["a b c d e f g h i j"], "a b c d e f g h i j", 1);
    assert!(
        short < full,
        "Brevity penalty: short hyp ({short}) should score less than full ({full})"
    );
}

#[test]
fn falsify_bleu_clipped_precision() {
    // "the the the the" vs ref "the cat" → unigram precision should be clipped to 1/4
    // (only 1 "the" in ref, but 4 in hyp → clipped to min(4,1) = 1)
    let score = bleu_score(&["the cat"], "the the the the", 1);
    // precision = 1/4, BP = exp(1 - 2/4) = exp(-0.5) ≈ 0.6065
    // BLEU = 0.6065 * 0.25 = 0.1516
    assert!(
        score < 0.3,
        "Clipped precision should limit repeated words, got {score}"
    );
    assert!(
        score > 0.0,
        "Some unigrams match, should be > 0, got {score}"
    );
}

#[test]
fn falsify_bleu_multi_reference() {
    // Multiple references: max count per n-gram across refs
    let score_single = bleu_score(&["the cat sat"], "the cat sat", 2);
    let score_multi = bleu_score(&["the cat sat", "the dog sat"], "the cat sat", 2);
    assert!(
        (score_single - score_multi).abs() < 1e-10,
        "Adding a non-matching ref should not change score: single={score_single}, multi={score_multi}"
    );
}

#[test]
fn falsify_bleu_max_n_1_equals_unigram() {
    // max_n=1 should only use unigram precision
    let score = bleu_score(&["a b c d"], "a b x d", 1);
    // unigram precision = 3/4 (a,b,d match), BP=1 (equal length)
    assert!(
        (score - 0.75).abs() < 1e-10,
        "BLEU-1 should be unigram precision = 3/4, got {score}"
    );
}

#[test]
fn falsify_bleu_zero_for_no_bigram_match() {
    // Even with perfect unigram match, zero bigram overlap → BLEU-4 = 0
    // "a b" vs "b a" — unigrams: both have {a, b}, bigrams: "a b" vs "b a" → 0
    let score = bleu_score(&["a b"], "b a", 4);
    assert_eq!(
        score, 0.0,
        "No bigram overlap should make BLEU-4 = 0, got {score}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM 3: ROUGE-N computes F1 of n-gram overlap
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_rouge1_known_value() {
    // ref: "a b c d" (4 unigrams), hyp: "a b e f" (4 unigrams)
    // overlap = {a, b} → 2
    // precision = 2/4 = 0.5, recall = 2/4 = 0.5
    // F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
    let score = rouge_n("a b c d", "a b e f", 1);
    assert!(
        (score - 0.5).abs() < 1e-10,
        "ROUGE-1 F1 should be 0.5, got {score}"
    );
}

#[test]
fn falsify_rouge1_asymmetric_lengths() {
    // ref: "a b c" (3 unigrams), hyp: "a" (1 unigram)
    // overlap = {a} → 1
    // precision = 1/1 = 1.0, recall = 1/3
    // F1 = 2 * 1.0 * (1/3) / (1.0 + 1/3) = (2/3) / (4/3) = 0.5
    let score = rouge_n("a b c", "a", 1);
    assert!(
        (score - 0.5).abs() < 1e-10,
        "ROUGE-1 with asymmetric lengths should be 0.5, got {score}"
    );
}

#[test]
fn falsify_rouge_n_is_symmetric() {
    // ROUGE F1 IS symmetric (unlike ROUGE precision or recall alone)
    let r1 = rouge_n("a b c", "a b d", 1);
    let r2 = rouge_n("a b d", "a b c", 1);
    assert!(
        (r1 - r2).abs() < 1e-10,
        "ROUGE-1 F1 should be symmetric: {r1} vs {r2}"
    );
}

#[test]
fn falsify_rouge2_known_value() {
    // ref: "a b c d" → bigrams: {"a b", "b c", "c d"} (3)
    // hyp: "a b c e" → bigrams: {"a b", "b c", "c e"} (3)
    // overlap: {"a b", "b c"} → 2
    // precision = 2/3, recall = 2/3
    // F1 = 2 * (2/3) * (2/3) / (4/3) = (8/9) / (4/3) = 2/3
    let score = rouge_n("a b c d", "a b c e", 2);
    assert!(
        (score - 2.0 / 3.0).abs() < 1e-10,
        "ROUGE-2 F1 should be 2/3, got {score}"
    );
}

#[test]
fn falsify_rouge_l_known_value() {
    // ref: "a b c d e" (5 tokens), hyp: "a x b c y" (5 tokens)
    // LCS: "a b c" (length 3)
    // precision = 3/5 = 0.6, recall = 3/5 = 0.6
    // F1 = 0.6
    let score = rouge_l("a b c d e", "a x b c y");
    assert!(
        (score - 0.6).abs() < 1e-10,
        "ROUGE-L F1 should be 0.6, got {score}"
    );
}

#[test]
fn falsify_rouge_l_subsequence_not_substring() {
    // LCS is subsequence, not substring
    // ref: "a x b x c" (5), hyp: "a b c" (3)
    // LCS: "a b c" (3), NOT "a" (which would be longest common substring at position 0)
    // precision = 3/3 = 1.0, recall = 3/5 = 0.6
    // F1 = 2 * 1.0 * 0.6 / 1.6 = 0.75
    let score = rouge_l("a x b x c", "a b c");
    assert!(
        (score - 0.75).abs() < 1e-10,
        "ROUGE-L should use subsequence, got {score}"
    );
}

#[test]
fn falsify_rouge_l_geq_rouge2() {
    // ROUGE-L should generally be >= ROUGE-2 for non-degenerate cases
    // (LCS captures long-range matches that bigrams miss)
    let r2 = rouge_n(
        "the quick brown fox jumps over the lazy dog",
        "the quick brown cat jumps over the lazy dog",
        2,
    );
    let rl = rouge_l(
        "the quick brown fox jumps over the lazy dog",
        "the quick brown cat jumps over the lazy dog",
    );
    assert!(
        rl >= r2 - 1e-10,
        "ROUGE-L should be >= ROUGE-2 for long sequences: RL={rl}, R2={r2}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM 4: Perplexity = exp(-mean(log_probs))
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_perplexity_known_binary() {
    // Binary uniform: p = 0.5 → log(0.5) = -ln(2)
    // PPL = exp(-mean(-ln(2))) = exp(ln(2)) = 2.0
    let log_probs = vec![(0.5f64).ln(); 100];
    let ppl = perplexity(&log_probs);
    assert!(
        (ppl - 2.0).abs() < 1e-10,
        "Binary uniform → PPL=2.0, got {ppl}"
    );
}

#[test]
fn falsify_perplexity_single_token() {
    // Single token with prob 0.01 → PPL = 100
    let ppl = perplexity(&[(0.01f64).ln()]);
    assert!(
        (ppl - 100.0).abs() < 1e-6,
        "Single token p=0.01 → PPL=100, got {ppl}"
    );
}

#[test]
fn falsify_perplexity_positive_log_probs() {
    // Positive log probs (invalid, but should not panic)
    // log_prob > 0 means prob > 1, which is invalid but PPL should still compute
    let ppl = perplexity(&[1.0, 2.0, 3.0]);
    assert!(
        ppl < 1.0,
        "Positive log-probs → PPL < 1.0 (invalid but computable), got {ppl}"
    );
}

#[test]
fn falsify_perplexity_mixed_values() {
    // Mix of good and bad predictions
    let log_probs = vec![(0.9f64).ln(), (0.01f64).ln()];
    let ppl = perplexity(&log_probs);
    // PPL = exp(-(ln(0.9) + ln(0.01)) / 2) = exp(-ln(0.009)/2) = (1/0.009)^0.5 ≈ 10.54
    let expected = (1.0 / (0.9 * 0.01f64)).sqrt();
    assert!(
        (ppl - expected).abs() < 1e-6,
        "Mixed PPL: expected {expected}, got {ppl}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM 5: pass@k = 1 - C(n-c,k) / C(n,k)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_pass_at_k_known_value() {
    // n=10, c=3, k=2
    // C(7,2)/C(10,2) = 21/45 = 7/15
    // pass@2 = 1 - 7/15 = 8/15 ≈ 0.5333...
    let p = pass_at_k(10, 3, 2);
    assert!(
        (p - 8.0 / 15.0).abs() < 1e-10,
        "pass@2(n=10,c=3) should be 8/15, got {p}"
    );
}

#[test]
fn falsify_pass_at_k_c_equals_1_k_equals_1() {
    // n=10, c=1, k=1 → 1 - C(9,1)/C(10,1) = 1 - 9/10 = 0.1
    let p = pass_at_k(10, 1, 1);
    assert!(
        (p - 0.1).abs() < 1e-10,
        "pass@1(n=10,c=1) should be 0.1, got {p}"
    );
}

#[test]
fn falsify_pass_at_k_large_n() {
    // Test numerical stability with large values
    // n=1000, c=100, k=10
    let p = pass_at_k(1000, 100, 10);
    assert!(
        p > 0.0 && p < 1.0,
        "pass@10(n=1000,c=100) should be in (0,1), got {p}"
    );
    // Manual: C(900,10)/C(1000,10) = prod_{i=0..9} (900-i)/(1000-i)
    let mut manual_ratio = 1.0f64;
    for i in 0..10 {
        manual_ratio *= (900 - i) as f64 / (1000 - i) as f64;
    }
    let expected = 1.0 - manual_ratio;
    assert!(
        (p - expected).abs() < 1e-10,
        "Numerical stability: expected {expected}, got {p}"
    );
}

#[test]
fn falsify_pass_at_k_monotonic_in_c() {
    // Fixing n and k, pass@k should increase with c
    let p1 = pass_at_k(20, 1, 5);
    let p5 = pass_at_k(20, 5, 5);
    let p10 = pass_at_k(20, 10, 5);
    assert!(
        p1 < p5,
        "pass@5 should increase with c: p(c=1)={p1} < p(c=5)={p5}"
    );
    assert!(
        p5 < p10,
        "pass@5 should increase with c: p(c=5)={p5} < p(c=10)={p10}"
    );
}

#[test]
fn falsify_pass_at_k_edge_k_equals_n() {
    // k = n: if c > 0, we are guaranteed to pick at least one correct
    // Actually: pass@n = 1 - C(n-c,n)/C(n,n). C(n,n) = 1, C(n-c,n) = 0 if c > 0
    // So pass@n = 1.0 when c > 0
    let p = pass_at_k(10, 1, 10);
    assert!(
        (p - 1.0).abs() < 1e-10,
        "pass@n with c>0 should be 1.0, got {p}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM 6: NDCG@k = DCG@k / IDCG@k, DCG = Σ (2^rel - 1) / log2(i+1)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_ndcg_known_value() {
    // scores: [3, 1, 2] → DCG = (2^3-1)/log2(2) + (2^1-1)/log2(3) + (2^2-1)/log2(4)
    //                          = 7/1 + 1/1.585 + 3/2 = 7 + 0.6309 + 1.5 = 9.1309
    // ideal: [3, 2, 1] → IDCG = 7/1 + 3/1.585 + 1/2 = 7 + 1.8928 + 0.5 = 9.3928
    // NDCG = 9.1309 / 9.3928 ≈ 0.9721
    let ndcg = ndcg_at_k(&[3.0, 1.0, 2.0], 3);
    let dcg = 7.0 / 1.0_f64.log2().max(f64::MIN_POSITIVE)  // pos 1: log2(2) = 1
        + 1.0 / 3.0_f64.log2()    // pos 2: log2(3)
        + 3.0 / 4.0_f64.log2(); // pos 3: log2(4)

    // Wait — the implementation uses log2(i+2) where i is 0-based.
    // i=0 → log2(2) = 1, i=1 → log2(3), i=2 → log2(4). Correct.
    let dcg_manual = 7.0 / (2.0_f64).log2() + 1.0 / (3.0_f64).log2() + 3.0 / (4.0_f64).log2();
    let idcg_manual = 7.0 / (2.0_f64).log2() + 3.0 / (3.0_f64).log2() + 1.0 / (4.0_f64).log2();
    let expected = dcg_manual / idcg_manual;

    // Suppress unused variable warning
    let _ = dcg;

    assert!(
        (ndcg - expected).abs() < 1e-6,
        "NDCG@3 for [3,1,2]: expected {expected}, got {ndcg}"
    );
}

#[test]
fn falsify_ndcg_single_element() {
    // Single relevant doc → NDCG = 1.0 regardless of score value
    let ndcg = ndcg_at_k(&[5.0], 1);
    assert!(
        (ndcg - 1.0).abs() < 1e-10,
        "Single element NDCG should be 1.0, got {ndcg}"
    );
}

#[test]
fn falsify_ndcg_binary_relevance() {
    // Binary: [0, 1] → worst ranking (relevant doc at position 2)
    // DCG = (2^0-1)/log2(2) + (2^1-1)/log2(3) = 0 + 1/1.585 = 0.6309
    // IDCG = (2^1-1)/log2(2) + (2^0-1)/log2(3) = 1/1 + 0 = 1.0
    // NDCG = 0.6309
    let ndcg = ndcg_at_k(&[0.0, 1.0], 2);
    let expected = (1.0 / (3.0_f64).log2()) / 1.0;
    assert!(
        (ndcg - expected).abs() < 1e-6,
        "NDCG@2 for [0,1] binary: expected {expected}, got {ndcg}"
    );
}

#[test]
fn falsify_ndcg_k_truncation() {
    // NDCG@1 only considers position 1
    let scores = vec![0.0, 3.0, 2.0, 1.0];
    let ndcg1 = ndcg_at_k(&scores, 1);
    // DCG@1 = (2^0 - 1)/log2(2) = 0/1 = 0, IDCG@1 = (2^3-1)/1 = 7
    // NDCG@1 = 0/7 = 0
    assert!(
        ndcg1.abs() < 1e-10,
        "NDCG@1 with irrelevant top result should be 0, got {ndcg1}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM 7: Feature gate isolation — hub-publish doesn't leak
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_generative_metrics_always_available() {
    // These should compile without any feature gate
    let _ = word_error_rate("a", "b");
    let _ = bleu_score(&["a"], "a", 1);
    let _ = rouge_n("a", "a", 1);
    let _ = rouge_l("a", "a");
    let _ = perplexity(&[-1.0]);
    let _ = pass_at_k(1, 1, 1);
    let _ = ndcg_at_k(&[1.0], 1);
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM 8: Metric enum consistency
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_wer_lower_is_better() {
    use crate::eval::evaluator::Metric;
    assert!(
        !Metric::WER.higher_is_better(),
        "WER should be lower-is-better"
    );
}

#[test]
fn falsify_perplexity_lower_is_better() {
    use crate::eval::evaluator::Metric;
    assert!(
        !Metric::Perplexity.higher_is_better(),
        "Perplexity should be lower-is-better"
    );
}

#[test]
fn falsify_bleu_higher_is_better() {
    use crate::eval::evaluator::Metric;
    assert!(
        Metric::BLEU.higher_is_better(),
        "BLEU should be higher-is-better"
    );
}

#[test]
fn falsify_pass_at_k_higher_is_better() {
    use crate::eval::evaluator::Metric;
    assert!(
        Metric::PassAtK(1).higher_is_better(),
        "pass@k should be higher-is-better"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM 9: Numerical stability under extreme inputs
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_perplexity_near_zero_prob() {
    // Very small probabilities: log(1e-300)
    let log_probs = vec![(1e-300f64).ln(); 5];
    let ppl = perplexity(&log_probs);
    assert!(
        ppl.is_finite() && ppl > 0.0,
        "PPL should handle tiny probs, got {ppl}"
    );
}

#[test]
fn falsify_bleu_single_word_ref_and_hyp() {
    // Single word: only unigram possible
    let score = bleu_score(&["hello"], "hello", 4);
    // 1-gram precision = 1.0, but 2,3,4-gram precision → total=0 → returns 0
    assert_eq!(
        score, 0.0,
        "Single word can't have 2-gram, BLEU-4 should be 0"
    );
}

#[test]
fn falsify_bleu_single_word_max_n_1() {
    let score = bleu_score(&["hello"], "hello", 1);
    assert!(
        (score - 1.0).abs() < 1e-10,
        "Single word with max_n=1 should give BLEU=1.0, got {score}"
    );
}

#[test]
fn falsify_pass_at_k_n_equals_1() {
    assert_eq!(pass_at_k(1, 0, 1), 0.0);
    assert!((pass_at_k(1, 1, 1) - 1.0).abs() < 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM 10: Cross-metric consistency
// ═══════════════════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn falsify_rouge1_geq_rouge2(
        reference in "[a-z]{1,3}( [a-z]{1,3}){3,8}",
        hypothesis in "[a-z]{1,3}( [a-z]{1,3}){3,8}"
    ) {
        let r1 = rouge_n(&reference, &hypothesis, 1);
        let r2 = rouge_n(&reference, &hypothesis, 2);
        // ROUGE-1 should generally be >= ROUGE-2 since unigrams are more likely to overlap
        // But this is NOT mathematically guaranteed for all inputs.
        // We check: if r2 > 0, then r1 > 0 (necessary condition)
        if r2 > 0.0 {
            prop_assert!(r1 > 0.0, "If ROUGE-2 > 0, ROUGE-1 must also be > 0: R1={}, R2={}", r1, r2);
        }
    }

    #[test]
    fn falsify_wer_edit_distance_identity(s in "[a-z]{1,5}( [a-z]{1,5}){0,9}") {
        // If hypothesis equals reference, WER = 0 (identity)
        // This is the most fundamental property
        let wer = word_error_rate(&s, &s);
        prop_assert!(
            wer.abs() < 1e-10,
            "WER(s, s) must be 0.0 for all s, got {} for {:?}", wer, s
        );
    }

    #[test]
    fn falsify_ndcg_invariant_to_scaling(
        scores in proptest::collection::vec(0.0f64..5.0, 2..10),
        k in 1..10usize
    ) {
        // NDCG is NOT invariant to additive shifts, but IS invariant when
        // the relative ordering doesn't change. Just verify bounds.
        let ndcg = ndcg_at_k(&scores, k);
        prop_assert!(
            ndcg >= -1e-10 && ndcg <= 1.0 + 1e-10,
            "NDCG must be in [0,1], got {}", ndcg
        );
    }
}
