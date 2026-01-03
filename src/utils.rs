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
#[inline]
pub fn fast_log2sumexp2(a: f64, b: f64) -> f64 {
    let (min, max) = if a < b { (a, b) } else { (b, a) };
    if min == f64::NEG_INFINITY {
        return max;
    }
    (2_f64.powf(min - max) + 1.0).log2() + max
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
#[inline]
pub fn fast_log2sumexp2_3(a: f64, b: f64, c: f64) -> f64 {
    let max = a.max(b).max(c);
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let sum = 2_f64.powf(a - max) + 2_f64.powf(b - max) + 2_f64.powf(c - max);
    sum.log2() + max
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
    fn test_fast_log2sumexp2_3() {
        let result = fast_log2sumexp2_3(10.0, 10.0, 10.0);
        let expected = (3.0 * 2_f64.powi(10)).log2();
        assert!((result - expected).abs() < 1e-10);
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
}
