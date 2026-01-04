//! Core data structures for einsum expressions and contraction trees.
//!
//! This module provides:
//! - [`EinCode`]: Representation of an einsum expression
//! - [`NestedEinsum`]: A binary tree encoding the contraction order
//! - [`SlicedEinsum`]: An einsum with sliced indices for reduced space complexity

use crate::Label;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// An einsum expression specifying how tensors are contracted.
///
/// The notation follows Einstein summation convention:
/// - `ixs` contains the index labels for each input tensor
/// - `iy` contains the index labels for the output tensor
/// - Repeated indices (appearing in multiple inputs but not output) are summed over
///
/// # Example
/// ```
/// use omeco::EinCode;
///
/// // Matrix multiplication: C[i,k] = A[i,j] * B[j,k]
/// // In einsum notation: "ij,jk->ik"
/// let code = EinCode::new(
///     vec![vec!['i', 'j'], vec!['j', 'k']],
///     vec!['i', 'k']
/// );
///
/// assert_eq!(code.num_tensors(), 2);
/// assert_eq!(code.contracted_indices(), vec!['j']);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EinCode<L: Label> {
    /// Input tensor indices (vector of index vectors for each tensor)
    pub ixs: Vec<Vec<L>>,
    /// Output indices
    pub iy: Vec<L>,
}

impl<L: Label> EinCode<L> {
    /// Create a new EinCode from input indices and output indices.
    pub fn new(ixs: Vec<Vec<L>>, iy: Vec<L>) -> Self {
        Self { ixs, iy }
    }

    /// Create an EinCode for a trace operation (no output indices).
    pub fn trace(ixs: Vec<Vec<L>>) -> Self {
        Self {
            ixs,
            iy: Vec::new(),
        }
    }

    /// Get the number of input tensors.
    #[inline]
    pub fn num_tensors(&self) -> usize {
        self.ixs.len()
    }

    /// Get all unique labels in this einsum.
    pub fn unique_labels(&self) -> Vec<L> {
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for ix in &self.ixs {
            for l in ix {
                if seen.insert(l.clone()) {
                    result.push(l.clone());
                }
            }
        }
        for l in &self.iy {
            if seen.insert(l.clone()) {
                result.push(l.clone());
            }
        }
        result
    }

    /// Get indices that are contracted (summed over).
    ///
    /// These are indices that appear in inputs but not in the output.
    pub fn contracted_indices(&self) -> Vec<L> {
        let output_set: HashSet<_> = self.iy.iter().cloned().collect();
        let mut contracted = Vec::new();
        let mut seen = HashSet::new();

        for ix in &self.ixs {
            for l in ix {
                if !output_set.contains(l) && seen.insert(l.clone()) {
                    contracted.push(l.clone());
                }
            }
        }
        contracted
    }

    /// Get indices that appear in the output (free indices).
    pub fn output_indices(&self) -> &[L] {
        &self.iy
    }

    /// Get the indices for a specific input tensor.
    pub fn input_indices(&self, tensor_idx: usize) -> Option<&[L]> {
        self.ixs.get(tensor_idx).map(|v| v.as_slice())
    }

    /// Check if this is a valid einsum (all output indices appear in inputs).
    pub fn is_valid(&self) -> bool {
        let input_set: HashSet<_> = self.ixs.iter().flatten().cloned().collect();
        self.iy.iter().all(|l| input_set.contains(l))
    }
}

impl<L: Label + fmt::Display> fmt::Display for EinCode<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inputs: Vec<String> = self
            .ixs
            .iter()
            .map(|ix| ix.iter().map(|l| l.to_string()).collect::<String>())
            .collect();
        let output: String = self.iy.iter().map(|l| l.to_string()).collect();
        write!(f, "{}->{}", inputs.join(","), output)
    }
}

/// A binary tree representing the contraction order for an einsum.
///
/// Each internal node represents a contraction of its two children,
/// and each leaf represents an input tensor.
///
/// # Example
/// ```
/// use omeco::{EinCode, NestedEinsum};
///
/// // Create leaves for input tensors
/// let leaf0 = NestedEinsum::<char>::leaf(0);
/// let leaf1 = NestedEinsum::<char>::leaf(1);
///
/// // Contract tensors 0 and 1
/// let contraction = EinCode::new(
///     vec![vec!['i', 'j'], vec!['j', 'k']],
///     vec!['i', 'k']
/// );
/// let tree = NestedEinsum::node(vec![leaf0, leaf1], contraction);
///
/// assert!(tree.is_binary());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NestedEinsum<L: Label> {
    /// A leaf node referencing an input tensor by index.
    Leaf {
        /// Index of the input tensor (0-indexed)
        tensor_index: usize,
    },
    /// An internal node representing a contraction.
    Node {
        /// Child nodes to contract
        args: Vec<NestedEinsum<L>>,
        /// The einsum operation for this contraction
        eins: EinCode<L>,
    },
}

impl<L: Label> NestedEinsum<L> {
    /// Create a leaf node for an input tensor.
    pub fn leaf(tensor_index: usize) -> Self {
        Self::Leaf { tensor_index }
    }

    /// Create an internal node for a contraction.
    pub fn node(args: Vec<NestedEinsum<L>>, eins: EinCode<L>) -> Self {
        Self::Node { args, eins }
    }

    /// Check if this node is a leaf.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf { .. })
    }

    /// Get the tensor index if this is a leaf node.
    pub fn tensor_index(&self) -> Option<usize> {
        match self {
            Self::Leaf { tensor_index } => Some(*tensor_index),
            Self::Node { .. } => None,
        }
    }

    /// Check if the tree is strictly binary (each internal node has exactly 2 children).
    pub fn is_binary(&self) -> bool {
        match self {
            Self::Leaf { .. } => true,
            Self::Node { args, .. } => args.len() == 2 && args.iter().all(|a| a.is_binary()),
        }
    }

    /// Count the total number of nodes in the tree.
    pub fn node_count(&self) -> usize {
        match self {
            Self::Leaf { .. } => 1,
            Self::Node { args, .. } => 1 + args.iter().map(|a| a.node_count()).sum::<usize>(),
        }
    }

    /// Count the number of leaf nodes (input tensors).
    pub fn leaf_count(&self) -> usize {
        match self {
            Self::Leaf { .. } => 1,
            Self::Node { args, .. } => args.iter().map(|a| a.leaf_count()).sum(),
        }
    }

    /// Get all leaf tensor indices in depth-first order.
    pub fn leaf_indices(&self) -> Vec<usize> {
        let mut result = Vec::new();
        self.collect_leaves(&mut result);
        result
    }

    fn collect_leaves(&self, result: &mut Vec<usize>) {
        match self {
            Self::Leaf { tensor_index } => result.push(*tensor_index),
            Self::Node { args, .. } => {
                for arg in args {
                    arg.collect_leaves(result);
                }
            }
        }
    }

    /// Get the output labels for this subtree.
    ///
    /// Requires the original input tensor labels to compute leaf outputs.
    pub fn output_labels(&self, input_labels: &[Vec<L>]) -> Vec<L> {
        match self {
            Self::Leaf { tensor_index } => {
                input_labels.get(*tensor_index).cloned().unwrap_or_default()
            }
            Self::Node { eins, .. } => eins.iy.clone(),
        }
    }

    /// Get the depth of the tree (longest path from root to leaf).
    pub fn depth(&self) -> usize {
        match self {
            Self::Leaf { .. } => 0,
            Self::Node { args, .. } => 1 + args.iter().map(|a| a.depth()).max().unwrap_or(0),
        }
    }
}

/// An einsum with sliced indices for reduced space complexity.
///
/// Slicing trades space for time: instead of computing the full contraction,
/// we iterate over the sliced indices and accumulate partial results.
///
/// # Example
/// ```
/// use omeco::{EinCode, NestedEinsum, SlicedEinsum};
///
/// let leaf0 = NestedEinsum::<char>::leaf(0);
/// let leaf1 = NestedEinsum::<char>::leaf(1);
/// let eins = EinCode::new(
///     vec![vec!['i', 'j'], vec!['j', 'k']],
///     vec!['i', 'k']
/// );
/// let tree = NestedEinsum::node(vec![leaf0, leaf1], eins);
///
/// // Slice over index 'j' to reduce intermediate tensor size
/// let sliced = SlicedEinsum::new(vec!['j'], tree);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlicedEinsum<L: Label> {
    /// Indices to slice over (iterate in outer loop)
    pub slicing: Vec<L>,
    /// The nested einsum to execute for each slice
    pub eins: NestedEinsum<L>,
}

impl<L: Label> SlicedEinsum<L> {
    /// Create a new sliced einsum.
    pub fn new(slicing: Vec<L>, eins: NestedEinsum<L>) -> Self {
        Self { slicing, eins }
    }

    /// Create a sliced einsum with no slicing (equivalent to the original).
    pub fn unsliced(eins: NestedEinsum<L>) -> Self {
        Self {
            slicing: Vec::new(),
            eins,
        }
    }

    /// Check if any slicing is applied.
    #[inline]
    pub fn is_sliced(&self) -> bool {
        !self.slicing.is_empty()
    }

    /// Get the number of sliced indices.
    #[inline]
    pub fn num_slices(&self) -> usize {
        self.slicing.len()
    }
}

/// Helper to create a size dictionary with uniform sizes.
///
/// # Example
/// ```
/// use omeco::{EinCode, uniform_size_dict};
///
/// let code = EinCode::new(
///     vec![vec!['i', 'j'], vec!['j', 'k']],
///     vec!['i', 'k']
/// );
/// let sizes = uniform_size_dict(&code, 10);
/// assert_eq!(sizes.get(&'i'), Some(&10));
/// ```
pub fn uniform_size_dict<L: Label>(code: &EinCode<L>, size: usize) -> HashMap<L, usize> {
    code.unique_labels()
        .into_iter()
        .map(|l| (l, size))
        .collect()
}

/// Convert a size dictionary to log2 sizes.
pub fn log2_size_dict<L: Label>(size_dict: &HashMap<L, usize>) -> HashMap<L, f64> {
    size_dict
        .iter()
        .map(|(k, &v)| (k.clone(), (v as f64).log2()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eincode_new() {
        let code: EinCode<char> =
            EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        assert_eq!(code.num_tensors(), 2);
        assert!(code.is_valid());
    }

    #[test]
    fn test_eincode_trace() {
        let code: EinCode<char> = EinCode::trace(vec![vec!['i', 'j'], vec!['j', 'i']]);
        assert_eq!(code.num_tensors(), 2);
        assert!(code.iy.is_empty());
    }

    #[test]
    fn test_eincode_unique_labels() {
        let code: EinCode<char> =
            EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let labels = code.unique_labels();
        assert_eq!(labels.len(), 3);
        assert!(labels.contains(&'i'));
        assert!(labels.contains(&'j'));
        assert!(labels.contains(&'k'));
    }

    #[test]
    fn test_eincode_unique_labels_with_output_only() {
        // Output label not in inputs (edge case)
        let code: EinCode<char> = EinCode::new(vec![vec!['i', 'j']], vec!['i', 'j', 'k']);
        let labels = code.unique_labels();
        assert_eq!(labels.len(), 3);
        assert!(labels.contains(&'k'));
    }

    #[test]
    fn test_eincode_contracted_indices() {
        let code: EinCode<char> =
            EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let contracted = code.contracted_indices();
        assert_eq!(contracted, vec!['j']);
    }

    #[test]
    fn test_eincode_contracted_indices_none() {
        // All indices are kept
        let code: EinCode<char> = EinCode::new(
            vec![vec!['i', 'j'], vec!['k', 'l']],
            vec!['i', 'j', 'k', 'l'],
        );
        let contracted = code.contracted_indices();
        assert!(contracted.is_empty());
    }

    #[test]
    fn test_eincode_output_indices() {
        let code: EinCode<char> =
            EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        assert_eq!(code.output_indices(), &['i', 'k']);
    }

    #[test]
    fn test_eincode_input_indices() {
        let code: EinCode<char> =
            EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        assert_eq!(code.input_indices(0), Some(&['i', 'j'][..]));
        assert_eq!(code.input_indices(1), Some(&['j', 'k'][..]));
        assert_eq!(code.input_indices(99), None); // Invalid index
    }

    #[test]
    fn test_eincode_is_valid() {
        let valid_code: EinCode<char> =
            EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        assert!(valid_code.is_valid());

        // Invalid: output contains index not in inputs
        let invalid_code: EinCode<char> = EinCode::new(vec![vec!['i', 'j']], vec!['i', 'k']);
        assert!(!invalid_code.is_valid());
    }

    #[test]
    fn test_eincode_display() {
        let code: EinCode<char> =
            EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        assert_eq!(format!("{}", code), "ij,jk->ik");
    }

    #[test]
    fn test_eincode_display_empty_output() {
        let code: EinCode<char> = EinCode::trace(vec![vec!['i', 'j'], vec!['j', 'i']]);
        assert_eq!(format!("{}", code), "ij,ji->");
    }

    #[test]
    fn test_nested_einsum_leaf() {
        let leaf: NestedEinsum<char> = NestedEinsum::leaf(0);
        assert!(leaf.is_leaf());
        assert_eq!(leaf.tensor_index(), Some(0));
        assert!(leaf.is_binary());
    }

    #[test]
    fn test_nested_einsum_node() {
        let leaf0: NestedEinsum<char> = NestedEinsum::leaf(0);
        let leaf1: NestedEinsum<char> = NestedEinsum::leaf(1);
        let eins = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let tree = NestedEinsum::node(vec![leaf0, leaf1], eins);

        assert!(!tree.is_leaf());
        assert!(tree.is_binary());
        assert_eq!(tree.leaf_count(), 2);
        assert_eq!(tree.leaf_indices(), vec![0, 1]);
    }

    #[test]
    fn test_nested_einsum_tensor_index_on_node() {
        let leaf0: NestedEinsum<char> = NestedEinsum::leaf(0);
        let leaf1: NestedEinsum<char> = NestedEinsum::leaf(1);
        let eins = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let tree = NestedEinsum::node(vec![leaf0, leaf1], eins);

        // Node should return None for tensor_index
        assert_eq!(tree.tensor_index(), None);
    }

    #[test]
    fn test_nested_einsum_node_count() {
        let leaf0: NestedEinsum<char> = NestedEinsum::leaf(0);
        let leaf1: NestedEinsum<char> = NestedEinsum::leaf(1);
        let leaf2: NestedEinsum<char> = NestedEinsum::leaf(2);

        let eins1 = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let tree1 = NestedEinsum::node(vec![leaf0, leaf1], eins1);

        let eins2 = EinCode::new(vec![vec!['i', 'k'], vec!['k', 'l']], vec!['i', 'l']);
        let tree2 = NestedEinsum::node(vec![tree1, leaf2], eins2);

        // node_count: 2 internal nodes + 3 leaves = 5
        assert_eq!(tree2.node_count(), 5);
    }

    #[test]
    fn test_nested_einsum_output_labels_leaf() {
        let leaf: NestedEinsum<char> = NestedEinsum::leaf(0);
        let input_labels = vec![vec!['i', 'j'], vec!['j', 'k']];

        let output = leaf.output_labels(&input_labels);
        assert_eq!(output, vec!['i', 'j']);
    }

    #[test]
    fn test_nested_einsum_output_labels_node() {
        let leaf0: NestedEinsum<char> = NestedEinsum::leaf(0);
        let leaf1: NestedEinsum<char> = NestedEinsum::leaf(1);
        let eins = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let tree = NestedEinsum::node(vec![leaf0, leaf1], eins);

        let input_labels = vec![vec!['i', 'j'], vec!['j', 'k']];
        let output = tree.output_labels(&input_labels);
        assert_eq!(output, vec!['i', 'k']);
    }

    #[test]
    fn test_nested_einsum_output_labels_invalid_index() {
        let leaf: NestedEinsum<char> = NestedEinsum::leaf(99); // Invalid index
        let input_labels = vec![vec!['i', 'j']];

        let output = leaf.output_labels(&input_labels);
        assert!(output.is_empty()); // Should return empty for invalid index
    }

    #[test]
    fn test_nested_einsum_is_binary_non_binary() {
        let leaf0: NestedEinsum<char> = NestedEinsum::leaf(0);
        let leaf1: NestedEinsum<char> = NestedEinsum::leaf(1);
        let leaf2: NestedEinsum<char> = NestedEinsum::leaf(2);

        // Non-binary: 3 children
        let eins = EinCode::new(
            vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
            vec!['i', 'l'],
        );
        let tree = NestedEinsum::node(vec![leaf0, leaf1, leaf2], eins);

        assert!(!tree.is_binary());
    }

    #[test]
    fn test_nested_einsum_depth() {
        let leaf0: NestedEinsum<char> = NestedEinsum::leaf(0);
        let leaf1: NestedEinsum<char> = NestedEinsum::leaf(1);
        let leaf2: NestedEinsum<char> = NestedEinsum::leaf(2);

        let eins1 = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let tree1 = NestedEinsum::node(vec![leaf0, leaf1], eins1);

        let eins2 = EinCode::new(vec![vec!['i', 'k'], vec!['k', 'l']], vec!['i', 'l']);
        let tree2 = NestedEinsum::node(vec![tree1, leaf2], eins2);

        assert_eq!(tree2.depth(), 2);
        assert_eq!(tree2.leaf_count(), 3);
    }

    #[test]
    fn test_nested_einsum_depth_leaf() {
        let leaf: NestedEinsum<char> = NestedEinsum::leaf(0);
        assert_eq!(leaf.depth(), 0);
    }

    #[test]
    fn test_sliced_einsum() {
        let leaf0: NestedEinsum<char> = NestedEinsum::leaf(0);
        let sliced = SlicedEinsum::new(vec!['j'], leaf0);

        assert!(sliced.is_sliced());
        assert_eq!(sliced.num_slices(), 1);
    }

    #[test]
    fn test_sliced_einsum_unsliced() {
        let leaf: NestedEinsum<char> = NestedEinsum::leaf(0);
        let sliced = SlicedEinsum::unsliced(leaf);

        assert!(!sliced.is_sliced());
        assert_eq!(sliced.num_slices(), 0);
    }

    #[test]
    fn test_uniform_size_dict() {
        let code: EinCode<char> =
            EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let sizes = uniform_size_dict(&code, 10);
        assert_eq!(sizes.len(), 3);
        assert_eq!(sizes.get(&'i'), Some(&10));
        assert_eq!(sizes.get(&'j'), Some(&10));
        assert_eq!(sizes.get(&'k'), Some(&10));
    }

    #[test]
    fn test_log2_size_dict() {
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 16);

        let log2_sizes = log2_size_dict(&size_dict);

        assert!((log2_sizes.get(&'i').unwrap() - 2.0).abs() < 1e-10);
        assert!((log2_sizes.get(&'j').unwrap() - 3.0).abs() < 1e-10);
        assert!((log2_sizes.get(&'k').unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_eincode_serialization() {
        let code: EinCode<char> =
            EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);

        // Test serialization/deserialization
        let json = serde_json::to_string(&code).unwrap();
        let decoded: EinCode<char> = serde_json::from_str(&json).unwrap();

        assert_eq!(code, decoded);
    }

    #[test]
    fn test_nested_einsum_serialization() {
        let leaf0: NestedEinsum<char> = NestedEinsum::leaf(0);
        let leaf1: NestedEinsum<char> = NestedEinsum::leaf(1);
        let eins = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let tree = NestedEinsum::node(vec![leaf0, leaf1], eins);

        let json = serde_json::to_string(&tree).unwrap();
        let decoded: NestedEinsum<char> = serde_json::from_str(&json).unwrap();

        assert_eq!(tree, decoded);
    }

    #[test]
    fn test_sliced_einsum_serialization() {
        let leaf: NestedEinsum<char> = NestedEinsum::leaf(0);
        let sliced = SlicedEinsum::new(vec!['j'], leaf);

        let json = serde_json::to_string(&sliced).unwrap();
        let decoded: SlicedEinsum<char> = serde_json::from_str(&json).unwrap();

        assert_eq!(sliced, decoded);
    }

    #[test]
    fn test_nested_einsum_depth_no_children() {
        // Test depth when a node has no args (edge case)
        let leaf0: NestedEinsum<char> = NestedEinsum::leaf(0);
        let leaf1: NestedEinsum<char> = NestedEinsum::leaf(1);
        let eins = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let tree = NestedEinsum::node(vec![leaf0, leaf1], eins);

        // depth of (leaf, leaf) should be 1
        assert_eq!(tree.depth(), 1);
    }
}
