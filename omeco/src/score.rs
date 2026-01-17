//! Scoring function for contraction order optimization.
//!
//! The score function balances time complexity, space complexity,
//! and read-write complexity when evaluating contraction orders.

use serde::{Deserialize, Serialize};

/// Weighted scoring function for contraction quality evaluation.
///
/// The score is computed as:
/// ```text
/// score = tc_weight * 2^tc + rw_weight * 2^rw + sc_weight * max(0, 2^sc - 2^sc_target)
/// ```
///
/// Where:
/// - `tc` is the time complexity (log2 of FLOP count)
/// - `sc` is the space complexity (log2 of max intermediate tensor size)
/// - `rw` is the read-write complexity (log2 of total I/O operations)
///
/// # Example
/// ```
/// use omeco::ScoreFunction;
///
/// let score = ScoreFunction::default();
/// let result = score.evaluate(10.0, 5.0, 8.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreFunction {
    /// Weight for time complexity (default: 1.0)
    pub tc_weight: f64,
    /// Weight for space complexity penalty (default: 1.0)
    pub sc_weight: f64,
    /// Weight for read-write complexity (default: 0.0)
    pub rw_weight: f64,
    /// Target space complexity threshold (default: 20.0)
    /// Space complexity is only penalized if it exceeds this target.
    pub sc_target: f64,
}

impl Default for ScoreFunction {
    fn default() -> Self {
        Self {
            tc_weight: 1.0,
            sc_weight: 1.0,
            rw_weight: 0.0,
            sc_target: 20.0,
        }
    }
}

impl ScoreFunction {
    /// Create a new ScoreFunction with custom weights.
    pub fn new(tc_weight: f64, sc_weight: f64, rw_weight: f64, sc_target: f64) -> Self {
        Self {
            tc_weight,
            sc_weight,
            rw_weight,
            sc_target,
        }
    }

    /// Create a score function optimizing primarily for time complexity.
    pub fn time_optimized() -> Self {
        Self {
            tc_weight: 1.0,
            sc_weight: 0.0,
            rw_weight: 0.0,
            sc_target: f64::INFINITY,
        }
    }

    /// Create a score function optimizing primarily for space complexity.
    pub fn space_optimized(sc_target: f64) -> Self {
        Self {
            tc_weight: 0.0,
            sc_weight: 1.0,
            rw_weight: 0.0,
            sc_target,
        }
    }

    /// Evaluate the score for given complexity values.
    ///
    /// # Arguments
    /// * `tc` - Time complexity (log2)
    /// * `sc` - Space complexity (log2)
    /// * `rw` - Read-write complexity (log2)
    ///
    /// # Returns
    /// The weighted score value (lower is better).
    pub fn evaluate(&self, tc: f64, sc: f64, rw: f64) -> f64 {
        let tc_term = self.tc_weight * 2_f64.powf(tc);
        let rw_term = self.rw_weight * 2_f64.powf(rw);
        let sc_penalty = (2_f64.powf(sc) - 2_f64.powf(self.sc_target)).max(0.0);
        let sc_term = self.sc_weight * sc_penalty;

        tc_term + rw_term + sc_term
    }

    /// Check if space complexity exceeds the target.
    #[inline]
    pub fn exceeds_target(&self, sc: f64) -> bool {
        sc > self.sc_target
    }

    /// Builder method to set the space complexity target.
    pub fn with_sc_target(mut self, sc_target: f64) -> Self {
        self.sc_target = sc_target;
        self
    }

    /// Builder method to set the time complexity weight.
    pub fn with_tc_weight(mut self, tc_weight: f64) -> Self {
        self.tc_weight = tc_weight;
        self
    }

    /// Builder method to set the space complexity weight.
    pub fn with_sc_weight(mut self, sc_weight: f64) -> Self {
        self.sc_weight = sc_weight;
        self
    }

    /// Builder method to set the read-write complexity weight.
    pub fn with_rw_weight(mut self, rw_weight: f64) -> Self {
        self.rw_weight = rw_weight;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_score() {
        let score = ScoreFunction::default();
        assert_eq!(score.tc_weight, 1.0);
        assert_eq!(score.sc_weight, 1.0);
        assert_eq!(score.rw_weight, 0.0);
        assert_eq!(score.sc_target, 20.0);
    }

    #[test]
    fn test_new_score() {
        let score = ScoreFunction::new(2.0, 3.0, 0.5, 15.0);
        assert_eq!(score.tc_weight, 2.0);
        assert_eq!(score.sc_weight, 3.0);
        assert_eq!(score.rw_weight, 0.5);
        assert_eq!(score.sc_target, 15.0);
    }

    #[test]
    fn test_evaluate_below_target() {
        let score = ScoreFunction::default();
        // tc=10, sc=5 (below target of 20), rw=8
        let result = score.evaluate(10.0, 5.0, 8.0);
        // Only tc contributes: 2^10 = 1024
        assert!((result - 1024.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_above_target() {
        let score = ScoreFunction::new(1.0, 1.0, 0.0, 10.0);
        // tc=10, sc=12 (above target of 10)
        let result = score.evaluate(10.0, 12.0, 0.0);
        // tc: 2^10 = 1024
        // sc penalty: 2^12 - 2^10 = 4096 - 1024 = 3072
        let expected = 1024.0 + 3072.0;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_with_rw_weight() {
        let score = ScoreFunction::new(1.0, 0.0, 1.0, 100.0);
        // tc=10, sc=5, rw=8
        let result = score.evaluate(10.0, 5.0, 8.0);
        // tc: 2^10 = 1024, rw: 2^8 = 256
        let expected = 1024.0 + 256.0;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_time_optimized() {
        let score = ScoreFunction::time_optimized();
        assert_eq!(score.tc_weight, 1.0);
        assert_eq!(score.sc_weight, 0.0);
        assert_eq!(score.rw_weight, 0.0);
        assert!(score.sc_target.is_infinite());
        // Space penalty should be zero even with high sc
        let result = score.evaluate(10.0, 100.0, 0.0);
        assert!((result - 1024.0).abs() < 1e-10);
    }

    #[test]
    fn test_space_optimized() {
        let score = ScoreFunction::space_optimized(10.0);
        assert_eq!(score.tc_weight, 0.0);
        assert_eq!(score.sc_weight, 1.0);
        assert_eq!(score.rw_weight, 0.0);
        assert_eq!(score.sc_target, 10.0);

        // Should only penalize space above target
        let result_below = score.evaluate(10.0, 5.0, 8.0);
        assert!((result_below - 0.0).abs() < 1e-10);

        let result_above = score.evaluate(10.0, 12.0, 8.0);
        // sc penalty: 2^12 - 2^10 = 4096 - 1024 = 3072
        assert!((result_above - 3072.0).abs() < 1e-10);
    }

    #[test]
    fn test_exceeds_target() {
        let score = ScoreFunction::default();
        assert!(!score.exceeds_target(10.0));
        assert!(!score.exceeds_target(20.0));
        assert!(score.exceeds_target(21.0));
    }

    #[test]
    fn test_score_serialization() {
        let score = ScoreFunction::new(1.0, 2.0, 0.5, 15.0);
        let json = serde_json::to_string(&score).unwrap();
        let decoded: ScoreFunction = serde_json::from_str(&json).unwrap();

        assert!((score.tc_weight - decoded.tc_weight).abs() < 1e-10);
        assert!((score.sc_weight - decoded.sc_weight).abs() < 1e-10);
        assert!((score.rw_weight - decoded.rw_weight).abs() < 1e-10);
        assert!((score.sc_target - decoded.sc_target).abs() < 1e-10);
    }

    #[test]
    fn test_with_sc_target_builder() {
        let score = ScoreFunction::default().with_sc_target(25.0);
        assert_eq!(score.sc_target, 25.0);
        assert_eq!(score.tc_weight, 1.0); // Other values unchanged
    }

    #[test]
    fn test_with_tc_weight_builder() {
        let score = ScoreFunction::default().with_tc_weight(2.5);
        assert_eq!(score.tc_weight, 2.5);
        assert_eq!(score.sc_weight, 1.0); // Other values unchanged
    }

    #[test]
    fn test_with_sc_weight_builder() {
        let score = ScoreFunction::default().with_sc_weight(3.0);
        assert_eq!(score.sc_weight, 3.0);
        assert_eq!(score.tc_weight, 1.0); // Other values unchanged
    }

    #[test]
    fn test_with_rw_weight_builder() {
        let score = ScoreFunction::default().with_rw_weight(0.5);
        assert_eq!(score.rw_weight, 0.5);
        assert_eq!(score.tc_weight, 1.0); // Other values unchanged
    }

    #[test]
    fn test_builder_chaining() {
        let score = ScoreFunction::default()
            .with_tc_weight(2.0)
            .with_sc_weight(3.0)
            .with_rw_weight(0.5)
            .with_sc_target(15.0);

        assert_eq!(score.tc_weight, 2.0);
        assert_eq!(score.sc_weight, 3.0);
        assert_eq!(score.rw_weight, 0.5);
        assert_eq!(score.sc_target, 15.0);
    }
}
