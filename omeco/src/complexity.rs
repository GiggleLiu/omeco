//! Complexity calculations for einsum contractions.
//!
//! This module provides functions to compute:
//! - Time complexity (FLOP count)
//! - Space complexity (peak memory usage)
//! - Read-write complexity (total I/O operations)

use crate::eincode::{log2_size_dict, EinCode, NestedEinsum, SlicedEinsum};
use crate::utils::log2sumexp2;
use crate::Label;
use std::collections::HashMap;

/// Contraction complexity metrics.
#[derive(Debug, Clone, Copy)]
pub struct ContractionComplexity {
    /// Log2 of time complexity (FLOP count)
    pub tc: f64,
    /// Log2 of space complexity (max intermediate tensor size)
    pub sc: f64,
    /// Log2 of read-write complexity (total I/O)
    pub rwc: f64,
}

impl ContractionComplexity {
    /// Create a new complexity result.
    pub fn new(tc: f64, sc: f64, rwc: f64) -> Self {
        Self { tc, sc, rwc }
    }

    /// Get the actual FLOP count (2^tc).
    pub fn flops(&self) -> f64 {
        2_f64.powf(self.tc)
    }

    /// Get the actual peak memory in elements (2^sc).
    pub fn peak_memory(&self) -> f64 {
        2_f64.powf(self.sc)
    }

    /// Get the actual read-write count (2^rwc).
    pub fn readwrites(&self) -> f64 {
        2_f64.powf(self.rwc)
    }
}

/// Compute the contraction complexity of an EinCode.
pub fn eincode_complexity<L: Label>(
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
) -> ContractionComplexity {
    let log2_sizes = log2_size_dict(size_dict);

    // Time complexity: product of all unique labels
    let tc: f64 = code
        .unique_labels()
        .iter()
        .map(|l| log2_sizes.get(l).copied().unwrap_or(0.0))
        .sum();

    // Space complexity: size of output tensor
    let sc: f64 = code
        .iy
        .iter()
        .map(|l| log2_sizes.get(l).copied().unwrap_or(0.0))
        .sum();

    // Read-write complexity: sum of all tensor sizes
    let input_sizes: Vec<f64> = code
        .ixs
        .iter()
        .map(|ix| {
            ix.iter()
                .map(|l| log2_sizes.get(l).copied().unwrap_or(0.0))
                .sum()
        })
        .collect();
    let all_sizes: Vec<f64> = input_sizes
        .iter()
        .cloned()
        .chain(std::iter::once(sc))
        .collect();
    let rwc = log2sumexp2(&all_sizes);

    ContractionComplexity { tc, sc, rwc }
}

/// Compute the contraction complexity of a NestedEinsum.
pub fn nested_complexity<L: Label>(
    code: &NestedEinsum<L>,
    size_dict: &HashMap<L, usize>,
    original_ixs: &[Vec<L>],
) -> ContractionComplexity {
    let log2_sizes = log2_size_dict(size_dict);
    let (tc, sc, rwc) = nested_complexity_inner(code, &log2_sizes, original_ixs);
    ContractionComplexity { tc, sc, rwc }
}

fn nested_complexity_inner<L: Label>(
    code: &NestedEinsum<L>,
    log2_sizes: &HashMap<L, f64>,
    original_ixs: &[Vec<L>],
) -> (f64, f64, f64) {
    match code {
        NestedEinsum::Leaf { tensor_index } => {
            let sc: f64 = original_ixs
                .get(*tensor_index)
                .map(|ix| {
                    ix.iter()
                        .map(|l| log2_sizes.get(l).copied().unwrap_or(0.0))
                        .sum()
                })
                .unwrap_or(0.0);
            (f64::NEG_INFINITY, sc, f64::NEG_INFINITY)
        }
        NestedEinsum::Node { args, eins } => {
            // Compute complexity of children
            let mut child_tcs = Vec::new();
            let mut max_sc = f64::NEG_INFINITY;
            let mut child_rwcs = Vec::new();

            for arg in args {
                let (tc, sc, rwc) = nested_complexity_inner(arg, log2_sizes, original_ixs);
                child_tcs.push(tc);
                max_sc = max_sc.max(sc);
                child_rwcs.push(rwc);
            }

            // Compute complexity of this contraction
            let this_tc: f64 = eins
                .unique_labels()
                .iter()
                .map(|l| log2_sizes.get(l).copied().unwrap_or(0.0))
                .sum();

            let this_sc: f64 = eins
                .iy
                .iter()
                .map(|l| log2_sizes.get(l).copied().unwrap_or(0.0))
                .sum();

            // Read-write for this operation
            let input_sizes: Vec<f64> = args
                .iter()
                .map(|arg| get_output_size(arg, log2_sizes, original_ixs))
                .collect();
            let all_sizes: Vec<f64> = input_sizes
                .iter()
                .cloned()
                .chain(std::iter::once(this_sc))
                .collect();
            let this_rwc = log2sumexp2(&all_sizes);

            // Combine complexities
            let all_tcs: Vec<f64> = child_tcs
                .iter()
                .cloned()
                .chain(std::iter::once(this_tc))
                .collect();
            let total_tc = log2sumexp2(&all_tcs);
            let total_sc = max_sc.max(this_sc);
            let all_rwcs: Vec<f64> = child_rwcs
                .iter()
                .cloned()
                .chain(std::iter::once(this_rwc))
                .collect();
            let total_rwc = log2sumexp2(&all_rwcs);

            (total_tc, total_sc, total_rwc)
        }
    }
}

fn get_output_size<L: Label>(
    code: &NestedEinsum<L>,
    log2_sizes: &HashMap<L, f64>,
    original_ixs: &[Vec<L>],
) -> f64 {
    match code {
        NestedEinsum::Leaf { tensor_index } => original_ixs
            .get(*tensor_index)
            .map(|ix| {
                ix.iter()
                    .map(|l| log2_sizes.get(l).copied().unwrap_or(0.0))
                    .sum()
            })
            .unwrap_or(0.0),
        NestedEinsum::Node { eins, .. } => eins
            .iy
            .iter()
            .map(|l| log2_sizes.get(l).copied().unwrap_or(0.0))
            .sum(),
    }
}

/// Compute the contraction complexity of a SlicedEinsum.
pub fn sliced_complexity<L: Label>(
    code: &SlicedEinsum<L>,
    size_dict: &HashMap<L, usize>,
    original_ixs: &[Vec<L>],
) -> ContractionComplexity {
    // Create a modified size dict with sliced indices set to 1
    let mut sliced_size_dict = size_dict.clone();
    for l in &code.slicing {
        sliced_size_dict.insert(l.clone(), 1);
    }

    // Compute base complexity with sliced sizes
    let base = nested_complexity(&code.eins, &sliced_size_dict, original_ixs);

    // Add slicing overhead to tc and rwc
    let log2_sizes = log2_size_dict(size_dict);
    let slice_overhead: f64 = code
        .slicing
        .iter()
        .map(|l| log2_sizes.get(l).copied().unwrap_or(0.0))
        .sum();

    ContractionComplexity {
        tc: base.tc + slice_overhead,
        sc: base.sc, // Space complexity is reduced by slicing
        rwc: base.rwc + slice_overhead,
    }
}

/// Compute the total FLOP count for an EinCode.
pub fn flop<L: Label>(code: &EinCode<L>, size_dict: &HashMap<L, usize>) -> usize {
    let unique_labels = code.unique_labels();
    unique_labels
        .iter()
        .map(|l| size_dict.get(l).copied().unwrap_or(1))
        .product()
}

/// Compute the total FLOP count for a NestedEinsum.
pub fn nested_flop<L: Label>(code: &NestedEinsum<L>, size_dict: &HashMap<L, usize>) -> usize {
    match code {
        NestedEinsum::Leaf { .. } => 0,
        NestedEinsum::Node { args, eins } => {
            let child_flops: usize = args.iter().map(|arg| nested_flop(arg, size_dict)).sum();
            child_flops + flop(eins, size_dict)
        }
    }
}

/// Compute peak memory usage for a NestedEinsum.
///
/// Returns the maximum intermediate tensor size in number of elements.
pub fn peak_memory<L: Label>(
    code: &NestedEinsum<L>,
    size_dict: &HashMap<L, usize>,
    original_ixs: &[Vec<L>],
) -> usize {
    peak_memory_inner(code, size_dict, original_ixs, 0).0
}

fn peak_memory_inner<L: Label>(
    code: &NestedEinsum<L>,
    size_dict: &HashMap<L, usize>,
    original_ixs: &[Vec<L>],
    temp_size: usize,
) -> (usize, usize) {
    match code {
        NestedEinsum::Leaf { tensor_index } => {
            let size = tensor_size(
                original_ixs
                    .get(*tensor_index)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]),
                size_dict,
            );
            (size + temp_size, size)
        }
        NestedEinsum::Node { args, eins } => {
            let mut max_peak = 0;
            let mut current_temp = temp_size;

            for arg in args {
                let (peak, arg_size) =
                    peak_memory_inner(arg, size_dict, original_ixs, current_temp);
                max_peak = max_peak.max(peak);
                current_temp += arg_size;
            }

            // Add output size
            let output_size = tensor_size(&eins.iy, size_dict);
            max_peak = max_peak.max(current_temp + output_size);

            (max_peak, output_size)
        }
    }
}

fn tensor_size<L: Label>(labels: &[L], size_dict: &HashMap<L, usize>) -> usize {
    if labels.is_empty() {
        1
    } else {
        labels
            .iter()
            .map(|l| size_dict.get(l).copied().unwrap_or(1))
            .product()
    }
}

/// Get the loop indices (indices that appear more than once in inputs).
pub fn get_loop_indices<L: Label>(code: &EinCode<L>) -> Vec<L> {
    let mut counts: HashMap<L, usize> = HashMap::new();

    for ix in &code.ixs {
        for l in ix {
            *counts.entry(l.clone()).or_insert(0) += 1;
        }
    }

    counts
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .map(|(l, _)| l)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eincode_complexity() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 4);

        let complexity = eincode_complexity(&code, &size_dict);

        // tc = log2(4 * 8 * 4) = log2(128) = 7
        assert!((complexity.tc - 7.0).abs() < 1e-10);
        // sc = log2(4 * 4) = log2(16) = 4
        assert!((complexity.sc - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_contraction_complexity_methods() {
        let complexity = ContractionComplexity::new(10.0, 5.0, 8.0);

        // Test flops
        let flops = complexity.flops();
        assert!((flops - 1024.0).abs() < 1e-10); // 2^10 = 1024

        // Test peak_memory
        let peak = complexity.peak_memory();
        assert!((peak - 32.0).abs() < 1e-10); // 2^5 = 32

        // Test readwrites
        let rw = complexity.readwrites();
        assert!((rw - 256.0).abs() < 1e-10); // 2^8 = 256
    }

    #[test]
    fn test_flop() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 4);

        let flops = flop(&code, &size_dict);
        assert_eq!(flops, 4 * 8 * 4); // 128
    }

    #[test]
    fn test_flop_missing_label() {
        let code = EinCode::new(vec![vec!['i', 'j']], vec!['i']);
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        // 'j' is missing - should default to 1

        let flops = flop(&code, &size_dict);
        assert_eq!(flops, 4); // 4 * 1
    }

    #[test]
    fn test_nested_complexity() {
        let leaf0 = NestedEinsum::leaf(0);
        let leaf1 = NestedEinsum::leaf(1);
        let eins = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let nested = NestedEinsum::node(vec![leaf0, leaf1], eins);

        let original_ixs = vec![vec!['i', 'j'], vec!['j', 'k']];
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 4);

        let complexity = nested_complexity(&nested, &size_dict, &original_ixs);

        // Single contraction: tc = log2(4*8*4) = 7
        assert!((complexity.tc - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_nested_complexity_leaf_only() {
        let leaf = NestedEinsum::leaf(0);
        let original_ixs = vec![vec!['i', 'j']];
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);

        let complexity = nested_complexity(&leaf, &size_dict, &original_ixs);

        // Leaf has no contraction, tc should be NEG_INFINITY
        assert!(complexity.tc == f64::NEG_INFINITY || complexity.tc < 0.0);
        // sc should be log2(4*8) = 5
        assert!((complexity.sc - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_nested_complexity_invalid_tensor_index() {
        let leaf = NestedEinsum::leaf(99); // Invalid index
        let original_ixs = vec![vec!['i', 'j']];
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);

        let complexity = nested_complexity(&leaf, &size_dict, &original_ixs);
        // Should handle gracefully
        assert!((complexity.sc - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_nested_flop() {
        let leaf0 = NestedEinsum::leaf(0);
        let leaf1 = NestedEinsum::leaf(1);
        let eins = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let nested = NestedEinsum::node(vec![leaf0, leaf1], eins);

        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 4);

        let flops = nested_flop(&nested, &size_dict);
        assert_eq!(flops, 128);
    }

    #[test]
    fn test_nested_flop_leaf() {
        let leaf = NestedEinsum::leaf(0);
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);

        let flops = nested_flop(&leaf, &size_dict);
        assert_eq!(flops, 0); // Leaf has no contractions
    }

    #[test]
    fn test_peak_memory() {
        let leaf0 = NestedEinsum::leaf(0);
        let leaf1 = NestedEinsum::leaf(1);
        let eins = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let nested = NestedEinsum::node(vec![leaf0, leaf1], eins);

        let original_ixs = vec![vec!['i', 'j'], vec!['j', 'k']];
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 4);

        let peak = peak_memory(&nested, &size_dict, &original_ixs);
        // Peak = max(input1 + temp, input2 + input1 + temp, output + input1 + input2)
        // = max(32, 32+32, 16+32+32) = 80
        assert!(peak > 0);
    }

    #[test]
    fn test_peak_memory_leaf() {
        let leaf = NestedEinsum::leaf(0);
        let original_ixs = vec![vec!['i', 'j']];
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);

        let peak = peak_memory(&leaf, &size_dict, &original_ixs);
        assert_eq!(peak, 32); // 4 * 8
    }

    #[test]
    fn test_peak_memory_empty_output() {
        // Test trace operation where output is empty
        let leaf0 = NestedEinsum::leaf(0);
        let leaf1 = NestedEinsum::leaf(1);
        let eins = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'i']], vec![]); // Trace
        let nested = NestedEinsum::node(vec![leaf0, leaf1], eins);

        let original_ixs = vec![vec!['i', 'j'], vec!['j', 'i']];
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);

        let peak = peak_memory(&nested, &size_dict, &original_ixs);
        assert!(peak > 0);
    }

    #[test]
    fn test_get_loop_indices() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k'], vec!['i', 'k']], vec![]);
        let loop_indices = get_loop_indices(&code);

        // i, j, k all appear more than once
        assert!(loop_indices.contains(&'i'));
        assert!(loop_indices.contains(&'j'));
        assert!(loop_indices.contains(&'k'));
    }

    #[test]
    fn test_get_loop_indices_no_loops() {
        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['k', 'l']],
            vec!['i', 'j', 'k', 'l'],
        );
        let loop_indices = get_loop_indices(&code);

        // No indices appear more than once
        assert!(loop_indices.is_empty());
    }

    #[test]
    fn test_sliced_complexity() {
        let leaf0 = NestedEinsum::leaf(0);
        let leaf1 = NestedEinsum::leaf(1);
        let eins = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let nested = NestedEinsum::node(vec![leaf0, leaf1], eins);
        let sliced = SlicedEinsum::new(vec!['j'], nested);

        let original_ixs = vec![vec!['i', 'j'], vec!['j', 'k']];
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 4);

        let complexity = sliced_complexity(&sliced, &size_dict, &original_ixs);

        // Slicing reduces space complexity but adds overhead to tc
        // sc should be lower than unsliced version
        let unsliced_complexity = nested_complexity(&sliced.eins, &size_dict, &original_ixs);
        assert!(complexity.sc <= unsliced_complexity.sc);
    }

    #[test]
    fn test_sliced_complexity_no_slicing() {
        let leaf0 = NestedEinsum::leaf(0);
        let leaf1 = NestedEinsum::leaf(1);
        let eins = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let nested = NestedEinsum::node(vec![leaf0, leaf1], eins);
        let sliced = SlicedEinsum::new(vec![], nested.clone()); // No slicing

        let original_ixs = vec![vec!['i', 'j'], vec!['j', 'k']];
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 4);

        let sliced_c = sliced_complexity(&sliced, &size_dict, &original_ixs);
        let unsliced_c = nested_complexity(&nested, &size_dict, &original_ixs);

        // Without slicing, complexities should be equal
        assert!((sliced_c.tc - unsliced_c.tc).abs() < 1e-10);
        assert!((sliced_c.sc - unsliced_c.sc).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_size_empty_labels() {
        // Test tensor_size with empty labels returns 1
        let size_dict: HashMap<char, usize> = HashMap::new();
        let size = tensor_size(&[], &size_dict);
        assert_eq!(size, 1);
    }

    #[test]
    fn test_eincode_complexity_trace() {
        // Trace operation: all indices are contracted
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'i']], vec![]); // Output is empty
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);

        let complexity = eincode_complexity(&code, &size_dict);

        // tc = log2(4 * 8) = log2(32) = 5
        assert!((complexity.tc - 5.0).abs() < 1e-10);
        // sc = 0 (empty output)
        assert!((complexity.sc - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_nested_complexity_deep_tree() {
        // Three-level deep tree
        let leaf0 = NestedEinsum::leaf(0);
        let leaf1 = NestedEinsum::leaf(1);
        let leaf2 = NestedEinsum::leaf(2);
        let leaf3 = NestedEinsum::leaf(3);

        let eins1 = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let tree1 = NestedEinsum::node(vec![leaf0, leaf1], eins1);

        let eins2 = EinCode::new(vec![vec!['k', 'l'], vec!['l', 'm']], vec!['k', 'm']);
        let tree2 = NestedEinsum::node(vec![leaf2, leaf3], eins2);

        let eins3 = EinCode::new(vec![vec!['i', 'k'], vec!['k', 'm']], vec!['i', 'm']);
        let tree3 = NestedEinsum::node(vec![tree1, tree2], eins3);

        let original_ixs = vec![
            vec!['i', 'j'],
            vec!['j', 'k'],
            vec!['k', 'l'],
            vec!['l', 'm'],
        ];
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 4);
        size_dict.insert('l', 8);
        size_dict.insert('m', 4);

        let complexity = nested_complexity(&tree3, &size_dict, &original_ixs);
        assert!(complexity.tc > 0.0);
        assert!(complexity.sc > 0.0);
        assert!(complexity.rwc > 0.0);
    }

    #[test]
    fn test_peak_memory_deep_tree() {
        let leaf0 = NestedEinsum::leaf(0);
        let leaf1 = NestedEinsum::leaf(1);
        let leaf2 = NestedEinsum::leaf(2);

        let eins1 = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let tree1 = NestedEinsum::node(vec![leaf0, leaf1], eins1);

        let eins2 = EinCode::new(vec![vec!['i', 'k'], vec!['k', 'l']], vec!['i', 'l']);
        let tree2 = NestedEinsum::node(vec![tree1, leaf2], eins2);

        let original_ixs = vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']];
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 4);
        size_dict.insert('l', 8);

        let peak = peak_memory(&tree2, &size_dict, &original_ixs);
        assert!(peak > 0);
    }

    #[test]
    fn test_nested_flop_single_tensor_with_output() {
        // Julia test: flop(EinCode([['i']], []), Dict('i'=>4)) == 4
        // Single tensor with output indices should have FLOP = 0 (no contraction)
        let leaf = NestedEinsum::leaf(0);
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);

        let flops = nested_flop(&leaf, &size_dict);
        assert_eq!(flops, 0); // Leaf has no contractions
    }

    #[test]
    fn test_nested_flop_trace_operation() {
        // Test FLOP for trace operation: tensor with duplicate indices
        let leaf = NestedEinsum::leaf(0);
        let eins = EinCode::new(vec![vec!['i', 'i']], vec![]); // Trace on single tensor
        let nested = NestedEinsum::node(vec![leaf], eins);

        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);

        let flops = nested_flop(&nested, &size_dict);
        assert_eq!(flops, 4); // i*i contracted = 4*4 operations but output is scalar, so 4
    }

    #[test]
    fn test_nested_flop_vs_time_complexity() {
        // Julia test: cc.tc ≈ log2(flop(optcode, size_dict))
        // Verify relationship between time complexity and FLOPs
        let leaf0 = NestedEinsum::leaf(0);
        let leaf1 = NestedEinsum::leaf(1);
        let eins = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let nested = NestedEinsum::node(vec![leaf0, leaf1], eins);

        let original_ixs = vec![vec!['i', 'j'], vec!['j', 'k']];
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 2);
        size_dict.insert('j', 4);
        size_dict.insert('k', 8);

        let flops = nested_flop(&nested, &size_dict);
        let complexity = nested_complexity(&nested, &size_dict, &original_ixs);

        // Time complexity should approximate log2(flops)
        let log2_flops = (flops as f64).log2();
        assert!((complexity.tc - log2_flops).abs() < 0.01);
    }

    #[test]
    fn test_nested_flop_complex_tree() {
        // Test FLOP computation for a deeper tree
        let leaf0 = NestedEinsum::leaf(0);
        let leaf1 = NestedEinsum::leaf(1);
        let leaf2 = NestedEinsum::leaf(2);

        // First contraction: tensor0 (ij) x tensor1 (jk) -> ik
        let eins1 = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let tree1 = NestedEinsum::node(vec![leaf0, leaf1], eins1);

        // Second contraction: result (ik) x tensor2 (kl) -> il
        let eins2 = EinCode::new(vec![vec!['i', 'k'], vec!['k', 'l']], vec!['i', 'l']);
        let tree2 = NestedEinsum::node(vec![tree1, leaf2], eins2);

        let mut size_dict = HashMap::new();
        size_dict.insert('i', 2);
        size_dict.insert('j', 4);
        size_dict.insert('k', 8);
        size_dict.insert('l', 2);

        let flops = nested_flop(&tree2, &size_dict);
        // First: 2*4*8 = 64, Second: 2*8*2 = 32, Total = 96
        assert_eq!(flops, 96);
    }

    #[test]
    fn test_peak_memory_specific_value() {
        // Test peak memory with exact expected value
        // Simple chain: A(i,j) x B(j,k) -> C(i,k)
        let leaf0 = NestedEinsum::leaf(0);
        let leaf1 = NestedEinsum::leaf(1);
        let eins = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let nested = NestedEinsum::node(vec![leaf0, leaf1], eins);

        let original_ixs = vec![vec!['i', 'j'], vec!['j', 'k']];
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 2);
        size_dict.insert('j', 4);
        size_dict.insert('k', 2);

        let peak = peak_memory(&nested, &size_dict, &original_ixs);
        // Tensor A: 2*4 = 8
        // Tensor B: 4*2 = 8
        // Output C: 2*2 = 4
        // Peak = max(A, A+B, A+B+C) = max(8, 16, 20) = 20
        assert_eq!(peak, 20);
    }

    #[test]
    fn test_peak_memory_vs_space_complexity() {
        // Julia test: 10 * 2^sc1 > pm1 > 2^sc1
        // Verify relationship between peak memory and space complexity
        let leaf0 = NestedEinsum::leaf(0);
        let leaf1 = NestedEinsum::leaf(1);
        let eins = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let nested = NestedEinsum::node(vec![leaf0, leaf1], eins);

        let original_ixs = vec![vec!['i', 'j'], vec!['j', 'k']];
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 4);

        let peak = peak_memory(&nested, &size_dict, &original_ixs);
        let complexity = nested_complexity(&nested, &size_dict, &original_ixs);

        // Peak memory should be related to space complexity
        // pm should be roughly between 2^sc and 10*2^sc
        let lower_bound = 2_f64.powf(complexity.sc);
        let upper_bound = 10.0 * 2_f64.powf(complexity.sc);

        assert!(
            peak as f64 > lower_bound && (peak as f64) < upper_bound,
            "Peak memory {} should be between {} and {}",
            peak,
            lower_bound,
            upper_bound
        );
    }

    #[test]
    fn test_peak_memory_with_broadcast() {
        // Test peak memory with broadcasting (output has index not in inputs)
        let leaf0 = NestedEinsum::leaf(0);
        let leaf1 = NestedEinsum::leaf(1);
        // This creates an outer product in dimension 'k'
        let eins = EinCode::new(vec![vec!['i'], vec!['j']], vec!['i', 'j', 'k']);
        let nested = NestedEinsum::node(vec![leaf0, leaf1], eins);

        let original_ixs = vec![vec!['i'], vec!['j']];
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 2);

        let peak = peak_memory(&nested, &size_dict, &original_ixs);
        // Tensor A: 4
        // Tensor B: 8
        // Output C: 4*8*2 = 64
        // Peak should account for broadcast
        assert!(peak > 0);
        assert!(peak >= 64); // At least the output size
    }

    #[test]
    fn test_nested_complexity_missing_size() {
        // Test behavior when some labels are missing from size_dict
        let leaf0 = NestedEinsum::leaf(0);
        let leaf1 = NestedEinsum::leaf(1);
        let eins = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let nested = NestedEinsum::node(vec![leaf0, leaf1], eins);

        let original_ixs = vec![vec!['i', 'j'], vec!['j', 'k']];
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        // Missing 'j' and 'k' - should default to 1

        let complexity = nested_complexity(&nested, &size_dict, &original_ixs);

        // Should not panic, uses default size of 1 for missing labels
        assert!(complexity.tc >= 0.0);
        assert!(complexity.sc >= 0.0);
    }

    #[test]
    fn test_eincode_complexity_missing_size() {
        // Test eincode_complexity with missing labels
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        // Missing 'j' and 'k'

        let complexity = eincode_complexity(&code, &size_dict);

        // Should handle missing labels gracefully
        assert!(complexity.tc >= 0.0);
        assert!(complexity.sc >= 0.0);
    }

    #[test]
    fn test_flop_missing_size() {
        // Test flop computation with missing labels
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        // Missing 'j' and 'k' - should default to 1

        let flops = flop(&code, &size_dict);

        // FLOP = 4 * 1 * 1 = 4
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_nested_flop_missing_size() {
        // Test nested_flop with missing labels in size_dict
        let leaf0 = NestedEinsum::leaf(0);
        let leaf1 = NestedEinsum::leaf(1);
        let eins = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let nested = NestedEinsum::node(vec![leaf0, leaf1], eins);

        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        // Missing 'j' and 'k'

        let flops = nested_flop(&nested, &size_dict);

        // Should handle missing labels (default to 1)
        assert!(flops > 0);
    }

    #[test]
    fn test_peak_memory_missing_size() {
        // Test peak_memory with missing labels
        let leaf0 = NestedEinsum::leaf(0);
        let leaf1 = NestedEinsum::leaf(1);
        let eins = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let nested = NestedEinsum::node(vec![leaf0, leaf1], eins);

        let original_ixs = vec![vec!['i', 'j'], vec!['j', 'k']];
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        // Missing 'j' and 'k'

        let peak = peak_memory(&nested, &size_dict, &original_ixs);

        // Should handle missing labels gracefully
        assert!(peak > 0);
    }

    #[test]
    fn test_get_loop_indices_with_duplicates() {
        // Test get_loop_indices with duplicate indices in same tensor
        let code = EinCode::new(vec![vec!['i', 'i'], vec!['j', 'j']], vec![]);

        let loops = get_loop_indices(&code);

        // Both 'i' and 'j' appear twice in same tensor, so they're loop indices
        assert_eq!(loops.len(), 2);
        assert!(loops.contains(&'i'));
        assert!(loops.contains(&'j'));
    }

    #[test]
    fn test_sliced_complexity_with_empty_slicing() {
        // Test sliced_complexity with no slicing
        let leaf0 = NestedEinsum::leaf(0);
        let leaf1 = NestedEinsum::leaf(1);
        let eins = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let nested = NestedEinsum::node(vec![leaf0, leaf1], eins);

        let sliced = SlicedEinsum {
            eins: nested,
            slicing: vec![], // No slicing
        };

        let original_ixs = vec![vec!['i', 'j'], vec!['j', 'k']];
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 4);

        let complexity = sliced_complexity(&sliced, &size_dict, &original_ixs);

        // Should be same as non-sliced
        assert!(complexity.tc > 0.0);
        assert!(complexity.sc > 0.0);
    }
}
