//! Tensor network simplification utilities.
//!
//! This module provides simplification passes that reduce tensor network
//! complexity before optimization, improving both speed and solution quality.
//!
//! # Overview
//!
//! The main simplification strategy is **merge_vectors**, which merges
//! 1-index tensors (vectors) into compatible higher-rank tensors. This
//! reduces the number of tensors in the network, making optimization faster.
//!
//! # Example
//!
//! ```rust
//! use omeco::{EinCode, GreedyMethod, optimize_code};
//! use omeco::simplifier::{merge_vectors, NetworkSimplifier};
//! use std::collections::HashMap;
//!
//! // A[i,j], v[j], B[j,k] -> [i,k] where v is a vector
//! let code = EinCode::new(
//!     vec![vec!['i', 'j'], vec!['j'], vec!['j', 'k']],
//!     vec!['i', 'k'],
//! );
//!
//! // Merge vectors before optimization
//! let simplifier = merge_vectors(&code);
//! assert!(simplifier.is_simplified()); // v was merged
//!
//! // Optimize the simplified network (fewer tensors)
//! let sizes: HashMap<char, usize> = [('i', 2), ('j', 3), ('k', 2)].into();
//! let tree = optimize_code(simplifier.simplified_code(), &sizes, &GreedyMethod::default())
//!     .expect("should optimize");
//!
//! // Embed the simplified result back into the original structure
//! let full_tree = simplifier.embed(tree);
//! assert_eq!(full_tree.leaf_count(), 3); // All original tensors accounted for
//! ```

use crate::eincode::{EinCode, NestedEinsum};
use crate::Label;
use std::collections::HashSet;

/// Describes a single merge operation where a vector is merged into another tensor.
#[derive(Debug, Clone)]
pub struct MergeOperation<L: Label> {
    /// Index of the vector tensor that was merged
    pub vector_index: usize,
    /// Index of the target tensor it was merged into
    pub target_index: usize,
    /// The index (label) that connects them
    pub shared_index: L,
}

/// A record of simplification operations applied to a tensor network.
///
/// This struct stores the mapping from simplified tensors back to original
/// tensors, enabling reconstruction of the full contraction tree after
/// optimization.
#[derive(Debug, Clone)]
pub struct NetworkSimplifier<L: Label> {
    /// List of merge operations applied
    merges: Vec<MergeOperation<L>>,
    /// The simplified EinCode after merging
    simplified_code: EinCode<L>,
    /// The original EinCode for reference
    original_code: EinCode<L>,
    /// Mapping from simplified tensor index -> original tensor index
    /// For tensors that received merges, maps to the target tensor
    index_map: Vec<usize>,
}

impl<L: Label> NetworkSimplifier<L> {
    /// Get the simplified EinCode.
    pub fn simplified_code(&self) -> &EinCode<L> {
        &self.simplified_code
    }

    /// Get the original EinCode.
    pub fn original_code(&self) -> &EinCode<L> {
        &self.original_code
    }

    /// Check if any simplifications were applied.
    pub fn is_simplified(&self) -> bool {
        !self.merges.is_empty()
    }

    /// Get the number of merge operations.
    pub fn num_merges(&self) -> usize {
        self.merges.len()
    }

    /// Get the merge operations.
    pub fn merges(&self) -> &[MergeOperation<L>] {
        &self.merges
    }

    /// Embed a simplified NestedEinsum back into the full tree.
    ///
    /// This reconstructs the original tensor structure by inserting
    /// binary contraction nodes for each merged vector.
    ///
    /// # Arguments
    /// * `simplified_tree` - The optimized tree for the simplified code
    ///
    /// # Returns
    /// A NestedEinsum that references all original tensors.
    pub fn embed(&self, simplified_tree: NestedEinsum<L>) -> NestedEinsum<L> {
        if !self.is_simplified() {
            return simplified_tree;
        }
        self.embed_recursive(simplified_tree)
    }

    /// Recursively embed merge operations into the tree.
    fn embed_recursive(&self, tree: NestedEinsum<L>) -> NestedEinsum<L> {
        match tree {
            NestedEinsum::Leaf { tensor_index } => {
                // Map simplified index back to original
                let original_idx = self.index_map[tensor_index];

                // Check if any vectors were merged into this tensor
                let merged_vectors: Vec<&MergeOperation<L>> = self
                    .merges
                    .iter()
                    .filter(|m| m.target_index == original_idx)
                    .collect();

                if merged_vectors.is_empty() {
                    // No vectors merged, return simple leaf
                    NestedEinsum::leaf(original_idx)
                } else {
                    // Build a chain of contractions for merged vectors
                    let target_indices = &self.original_code.ixs[original_idx];
                    let mut current = NestedEinsum::leaf(original_idx);
                    let current_indices = target_indices.clone();

                    for merge in merged_vectors {
                        let vector_indices = &self.original_code.ixs[merge.vector_index];
                        let vector_leaf = NestedEinsum::leaf(merge.vector_index);

                        // Output is same as current (vector is absorbed)
                        let eins = EinCode::new(
                            vec![current_indices.clone(), vector_indices.clone()],
                            current_indices.clone(),
                        );

                        current = NestedEinsum::node(vec![current, vector_leaf], eins);
                    }

                    current
                }
            }
            NestedEinsum::Node { args, eins } => {
                // Recursively embed children
                let embedded_args: Vec<NestedEinsum<L>> =
                    args.into_iter().map(|a| self.embed_recursive(a)).collect();

                // Remap the eins.ixs to match the embedded structure
                // Since embedding adds extra nodes, the ixs from the simplified tree
                // still apply to the result of each embedded subtree
                NestedEinsum::node(embedded_args, eins)
            }
        }
    }
}

/// Merge all 1-index tensors (vectors) into compatible tensors.
///
/// For each vector tensor, finds the first tensor that shares its index
/// and creates a binary contraction merging them. This reduces the number
/// of tensors in the network.
///
/// # Arguments
/// * `code` - The EinCode to simplify
///
/// # Returns
/// A NetworkSimplifier containing the simplified code and merge operations.
///
/// # Example
/// ```rust
/// use omeco::EinCode;
/// use omeco::simplifier::merge_vectors;
///
/// // A[i,j], v[j], B[j,k] -> merge v into A or B
/// let code = EinCode::new(
///     vec![vec!['i', 'j'], vec!['j'], vec!['j', 'k']],
///     vec!['i', 'k']
/// );
/// let simplifier = merge_vectors(&code);
/// assert_eq!(simplifier.num_merges(), 1); // v was merged
/// assert_eq!(simplifier.simplified_code().num_tensors(), 2); // Now 2 tensors
/// ```
pub fn merge_vectors<L: Label>(code: &EinCode<L>) -> NetworkSimplifier<L> {
    let num_tensors = code.ixs.len();

    // Track which tensors have been removed (merged into others)
    let mut removed: HashSet<usize> = HashSet::new();
    let mut merges: Vec<MergeOperation<L>> = Vec::new();

    // Find all vectors and try to merge them
    for vec_idx in 0..num_tensors {
        if removed.contains(&vec_idx) {
            continue;
        }

        let tensor_indices = &code.ixs[vec_idx];
        if tensor_indices.len() != 1 {
            continue; // Not a vector
        }

        let vec_label = &tensor_indices[0];

        // Find first tensor containing this label (that isn't the vector itself)
        let target = code
            .ixs
            .iter()
            .enumerate()
            .find(|(i, ix)| *i != vec_idx && !removed.contains(i) && ix.contains(vec_label));

        if let Some((target_idx, _)) = target {
            // Record the merge
            merges.push(MergeOperation {
                vector_index: vec_idx,
                target_index: target_idx,
                shared_index: vec_label.clone(),
            });

            // Mark vector as removed
            removed.insert(vec_idx);
        }
    }

    // Build the simplified code by keeping only non-removed tensors
    let kept_indices: Vec<usize> = (0..num_tensors).filter(|i| !removed.contains(i)).collect();

    let simplified_ixs: Vec<Vec<L>> = kept_indices.iter().map(|&i| code.ixs[i].clone()).collect();

    let simplified_code = EinCode::new(simplified_ixs, code.iy.clone());

    // Build the index map: simplified index -> original index
    let index_map = kept_indices;

    NetworkSimplifier {
        merges,
        simplified_code,
        original_code: code.clone(),
        index_map,
    }
}

/// Optimize with automatic simplification.
///
/// Applies merge_vectors simplification before optimization, then
/// embeds the result back into the original tensor structure.
///
/// This is a convenience function that combines simplification and optimization.
///
/// # Arguments
/// * `code` - The tensor network to optimize
/// * `size_dict` - Dimension sizes for each index
/// * `optimizer` - The optimizer to use
///
/// # Returns
/// An optimized NestedEinsum tree referencing all original tensors.
///
/// # Example
/// ```rust
/// use omeco::{EinCode, GreedyMethod};
/// use omeco::simplifier::optimize_simplified;
/// use std::collections::HashMap;
///
/// let code = EinCode::new(
///     vec![vec!['i', 'j'], vec!['j'], vec!['j', 'k']],
///     vec!['i', 'k'],
/// );
/// let sizes: HashMap<char, usize> = [('i', 2), ('j', 3), ('k', 2)].into();
///
/// let tree = optimize_simplified(&code, &sizes, &GreedyMethod::default()).unwrap();
/// assert_eq!(tree.leaf_count(), 3); // All 3 original tensors
/// ```
pub fn optimize_simplified<L: Label, O: crate::CodeOptimizer>(
    code: &EinCode<L>,
    size_dict: &std::collections::HashMap<L, usize>,
    optimizer: &O,
) -> Option<NestedEinsum<L>> {
    let simplifier = merge_vectors(code);

    if !simplifier.is_simplified() {
        // No simplification possible, optimize directly
        return optimizer.optimize(code, size_dict);
    }

    // Optimize simplified code
    let simplified_tree = optimizer.optimize(simplifier.simplified_code(), size_dict)?;

    // Embed back into original structure
    Some(simplifier.embed(simplified_tree))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eincode::uniform_size_dict;
    use crate::greedy::{optimize_greedy, GreedyMethod};

    #[test]
    fn test_merge_vectors_simple() {
        // A[i,j], v[j], B[j,k] -> [i,k]
        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['j'], vec!['j', 'k']],
            vec!['i', 'k'],
        );

        let simplifier = merge_vectors(&code);

        assert!(simplifier.is_simplified());
        assert_eq!(simplifier.num_merges(), 1);
        assert_eq!(simplifier.simplified_code().num_tensors(), 2);

        // Check the merge operation
        let merge = &simplifier.merges()[0];
        assert_eq!(merge.vector_index, 1);
        assert!(merge.target_index == 0 || merge.target_index == 2);
        assert_eq!(merge.shared_index, 'j');
    }

    #[test]
    fn test_merge_vectors_multiple_vectors() {
        // A[i,j], v1[j], v2[k], B[j,k] -> merge both vectors
        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['j'], vec!['k'], vec!['j', 'k']],
            vec!['i'],
        );

        let simplifier = merge_vectors(&code);
        assert_eq!(simplifier.num_merges(), 2);
        assert_eq!(simplifier.simplified_code().num_tensors(), 2);
    }

    #[test]
    fn test_merge_vectors_no_vectors() {
        // All tensors have rank >= 2
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);

        let simplifier = merge_vectors(&code);
        assert!(!simplifier.is_simplified());
        assert_eq!(simplifier.num_merges(), 0);
        assert_eq!(simplifier.simplified_code().num_tensors(), 2);
    }

    #[test]
    fn test_merge_vectors_disconnected_vector() {
        // Vector that doesn't share index with any other tensor
        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['k'], vec!['j', 'l']],
            vec!['i', 'l', 'k'],
        );

        let simplifier = merge_vectors(&code);
        // Vector [k] is disconnected, cannot be merged
        assert!(!simplifier.is_simplified());
    }

    #[test]
    fn test_embed_simplifier_simple() {
        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['j'], vec!['j', 'k']],
            vec!['i', 'k'],
        );
        let sizes = uniform_size_dict(&code, 4);

        let simplifier = merge_vectors(&code);
        let simplified_tree = optimize_greedy(
            simplifier.simplified_code(),
            &sizes,
            &GreedyMethod::default(),
        )
        .unwrap();

        let embedded = simplifier.embed(simplified_tree);

        // Embedded tree should reference all 3 original tensors
        assert_eq!(embedded.leaf_count(), 3);
        assert!(embedded.is_binary());
    }

    #[test]
    fn test_embed_preserves_correctness() {
        // Verify that embedding produces a tree with the correct structure
        let code = EinCode::new(
            vec![vec!['a', 'b'], vec!['b'], vec!['b', 'c'], vec!['c']],
            vec!['a'],
        );
        let sizes = uniform_size_dict(&code, 2);

        let simplifier = merge_vectors(&code);
        assert_eq!(simplifier.num_merges(), 2); // Both vectors merged

        let simplified_tree = optimize_greedy(
            simplifier.simplified_code(),
            &sizes,
            &GreedyMethod::default(),
        )
        .unwrap();

        let embedded = simplifier.embed(simplified_tree);
        assert_eq!(embedded.leaf_count(), 4); // All 4 original tensors
    }

    #[test]
    fn test_optimize_simplified() {
        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['j'], vec!['j', 'k'], vec!['k']],
            vec!['i'],
        );
        let sizes = uniform_size_dict(&code, 4);

        let result = optimize_simplified(&code, &sizes, &GreedyMethod::default());

        assert!(result.is_some());
        let tree = result.unwrap();
        assert_eq!(tree.leaf_count(), 4);
    }

    #[test]
    fn test_optimize_simplified_no_vectors() {
        // When there are no vectors, optimize_simplified should work normally
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let sizes = uniform_size_dict(&code, 4);

        let result = optimize_simplified(&code, &sizes, &GreedyMethod::default());

        assert!(result.is_some());
        let tree = result.unwrap();
        assert_eq!(tree.leaf_count(), 2);
    }

    #[test]
    fn test_simplification_with_treesa() {
        use crate::treesa::TreeSA;

        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['j'], vec!['j', 'k']],
            vec!['i', 'k'],
        );
        let sizes = uniform_size_dict(&code, 4);

        let result = optimize_simplified(&code, &sizes, &TreeSA::fast());

        assert!(result.is_some());
        let tree = result.unwrap();
        assert_eq!(tree.leaf_count(), 3);
    }

    #[test]
    fn test_network_simplifier_accessors() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j']], vec!['i']);
        let simplifier = merge_vectors(&code);

        assert_eq!(simplifier.original_code(), &code);
        assert_eq!(simplifier.simplified_code().num_tensors(), 1);
        assert!(simplifier.is_simplified());
        assert_eq!(simplifier.num_merges(), 1);
    }

    #[test]
    fn test_merge_operation_fields() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j']], vec!['i']);
        let simplifier = merge_vectors(&code);

        let merge = &simplifier.merges()[0];
        assert_eq!(merge.vector_index, 1);
        assert_eq!(merge.target_index, 0);
        assert_eq!(merge.shared_index, 'j');
    }

    #[test]
    fn test_embed_no_simplification() {
        // When there's no simplification, embed should return the same tree
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let sizes = uniform_size_dict(&code, 4);

        let simplifier = merge_vectors(&code);
        let tree = optimize_greedy(&code, &sizes, &GreedyMethod::default()).unwrap();

        let embedded = simplifier.embed(tree.clone());
        assert_eq!(embedded.leaf_count(), tree.leaf_count());
    }

    #[test]
    fn test_simplification_complex_network() {
        // More complex network with multiple vectors
        // A[a,b], v1[b], B[b,c,d], v2[c], C[d,e], v3[e] -> [a]
        let code = EinCode::new(
            vec![
                vec!['a', 'b'],      // A
                vec!['b'],           // v1
                vec!['b', 'c', 'd'], // B
                vec!['c'],           // v2
                vec!['d', 'e'],      // C
                vec!['e'],           // v3
            ],
            vec!['a'],
        );
        let sizes = uniform_size_dict(&code, 2);

        let simplifier = merge_vectors(&code);
        assert_eq!(simplifier.num_merges(), 3); // All 3 vectors merged

        let result = optimize_simplified(&code, &sizes, &GreedyMethod::default());
        assert!(result.is_some());

        let tree = result.unwrap();
        assert_eq!(tree.leaf_count(), 6); // All 6 original tensors
    }
}
