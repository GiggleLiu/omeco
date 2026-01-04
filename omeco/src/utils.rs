//! Utility functions for numerical computations.
//!
//! Provides numerically stable operations for complexity calculations.

/// Numerically stable computation of log2(2^a + 2^b).
///
/// Uses the identity: log2(2^a + 2^b) = max(a,b) + log2(1 + 2^(min-max))
///
/// # Example
/// ```
/// use omeco::utils::fast_log2sumexp2;
///
/// let result = fast_log2sumexp2(10.0, 10.0);
/// assert!((result - 11.0).abs() < 1e-10); // log2(2^10 + 2^10) = log2(2*2^10) = 11
/// ```
#[inline(always)]
pub fn fast_log2sumexp2(a: f64, b: f64) -> f64 {
    let (min, max) = if a < b { (a, b) } else { (b, a) };
    if min == f64::NEG_INFINITY {
        return max;
    }
    // Use log1p_exp for numerical stability: log2(1 + 2^x) = (ln(1 + e^(x*ln2))) / ln2
    // For small |x|, this is more accurate
    let diff = min - max;
    if diff < -50.0 {
        // 2^diff is negligible
        max
    } else {
        // log2(1 + 2^diff) = log2(e) * ln(1 + e^(diff * ln(2)))
        let ln2 = std::f64::consts::LN_2;
        let log2e = std::f64::consts::LOG2_E;
        max + log2e * (diff * ln2).exp().ln_1p()
    }
}

/// Numerically stable computation of log2(2^a + 2^b + 2^c).
///
/// # Example
/// ```
/// use omeco::utils::fast_log2sumexp2_3;
///
/// let result = fast_log2sumexp2_3(10.0, 10.0, 10.0);
/// // log2(3 * 2^10) ≈ 10 + log2(3) ≈ 11.585
/// ```
#[inline(always)]
pub fn fast_log2sumexp2_3(a: f64, b: f64, c: f64) -> f64 {
    let max = a.max(b).max(c);
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    // Compute scaled sum avoiding overflow
    let da = a - max;
    let db = b - max;
    let dc = c - max;

    // Skip negligible terms (2^-50 ≈ 1e-15)
    let mut sum = 0.0_f64;
    if da > -50.0 {
        sum += fast_exp2(da);
    }
    if db > -50.0 {
        sum += fast_exp2(db);
    }
    if dc > -50.0 {
        sum += fast_exp2(dc);
    }

    if sum == 0.0 {
        max
    } else {
        sum.log2() + max
    }
}

/// Fast 2^x approximation for small x (used in log2sumexp)
#[inline(always)]
fn fast_exp2(x: f64) -> f64 {
    // For x close to 0, use exp2 directly; it's well-optimized
    // For very negative x, return 0
    if x < -50.0 {
        0.0
    } else {
        2_f64.powf(x)
    }
}

/// Numerically stable computation of log2(sum(2^x for x in values)).
///
/// # Example
/// ```
/// use omeco::utils::log2sumexp2;
///
/// let result = log2sumexp2(&[10.0, 10.0, 10.0, 10.0]);
/// // log2(4 * 2^10) = 12
/// assert!((result - 12.0).abs() < 1e-10);
/// ```
pub fn log2sumexp2(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = values.iter().map(|&x| 2_f64.powf(x - max)).sum();
    sum.log2() + max
}

/// Compute log2 of the product of values from a size dictionary.
///
/// Returns the sum of log2(size) for each label.
#[inline]
pub fn log2_prod<L, I>(labels: I, log2_sizes: &std::collections::HashMap<L, f64>) -> f64
where
    L: std::hash::Hash + Eq,
    I: IntoIterator<Item = L>,
{
    labels
        .into_iter()
        .map(|l| log2_sizes.get(&l).copied().unwrap_or(0.0))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_fast_log2sumexp2_equal() {
        let result = fast_log2sumexp2(10.0, 10.0);
        assert!((result - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_fast_log2sumexp2_different() {
        // log2(2^3 + 2^5) = log2(8 + 32) = log2(40) ≈ 5.32
        let result = fast_log2sumexp2(3.0, 5.0);
        let expected = 40_f64.log2();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_fast_log2sumexp2_neg_inf() {
        let result = fast_log2sumexp2(f64::NEG_INFINITY, 5.0);
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_fast_log2sumexp2_large_difference() {
        // When difference is very large, result should be close to max
        let result = fast_log2sumexp2(-100.0, 10.0);
        assert!((result - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_fast_log2sumexp2_reversed_order() {
        // Test with a < b and b < a
        let result1 = fast_log2sumexp2(3.0, 5.0);
        let result2 = fast_log2sumexp2(5.0, 3.0);
        assert!((result1 - result2).abs() < 1e-10);
    }

    #[test]
    fn test_fast_log2sumexp2_3() {
        let result = fast_log2sumexp2_3(10.0, 10.0, 10.0);
        let expected = (3.0 * 2_f64.powi(10)).log2();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_fast_log2sumexp2_3_all_neg_inf() {
        let result = fast_log2sumexp2_3(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        assert!(result == f64::NEG_INFINITY);
    }

    #[test]
    fn test_fast_log2sumexp2_3_one_neg_inf() {
        let result = fast_log2sumexp2_3(f64::NEG_INFINITY, 10.0, 10.0);
        let expected = (2.0 * 2_f64.powi(10)).log2();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_fast_log2sumexp2_3_large_difference() {
        // When all values are much smaller than max
        let result = fast_log2sumexp2_3(-100.0, -100.0, 10.0);
        assert!((result - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_log2sumexp2_vec() {
        let result = log2sumexp2(&[10.0, 10.0, 10.0, 10.0]);
        assert!((result - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_log2sumexp2_empty() {
        let result = log2sumexp2(&[]);
        assert!(result == f64::NEG_INFINITY);
    }

    #[test]
    fn test_log2sumexp2_all_neg_inf() {
        let result = log2sumexp2(&[f64::NEG_INFINITY, f64::NEG_INFINITY]);
        assert!(result == f64::NEG_INFINITY);
    }

    #[test]
    fn test_log2sumexp2_single() {
        let result = log2sumexp2(&[10.0]);
        assert!((result - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_log2_prod() {
        let mut log2_sizes: HashMap<char, f64> = HashMap::new();
        log2_sizes.insert('i', 2.0);
        log2_sizes.insert('j', 3.0);
        log2_sizes.insert('k', 4.0);

        let result = log2_prod(['i', 'j', 'k'].iter().cloned(), &log2_sizes);
        assert!((result - 9.0).abs() < 1e-10); // 2 + 3 + 4
    }

    #[test]
    fn test_log2_prod_missing_label() {
        let mut log2_sizes: HashMap<char, f64> = HashMap::new();
        log2_sizes.insert('i', 2.0);

        // 'j' is missing, should default to 0.0
        let result = log2_prod(['i', 'j'].iter().cloned(), &log2_sizes);
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_log2_prod_empty() {
        let log2_sizes: HashMap<char, f64> = HashMap::new();
        let result = log2_prod(std::iter::empty::<char>(), &log2_sizes);
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_fast_exp2_very_negative() {
        // Test that fast_exp2 returns 0 for very negative values
        // This is implicitly tested via fast_log2sumexp2_3 with large differences
        let result = fast_log2sumexp2_3(-100.0, -100.0, 10.0);
        // Should effectively be just 10.0 since -100 terms are negligible
        assert!((result - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_fast_log2sumexp2_3_all_same() {
        // All values are the same
        let result = fast_log2sumexp2_3(-100.0, -100.0, -100.0);
        // log2(3 * 2^-100) = -100 + log2(3) ≈ -98.415
        let expected = -100.0 + 3_f64.log2();
        assert!((result - expected).abs() < 1e-10);
    }
}
