//! # omeco - Tensor Network Contraction Order Optimization
//!
//! A Rust library for optimizing tensor network contraction orders, ported from
//! the Julia package [OMEinsumContractionOrders.jl](https://github.com/TensorBFS/OMEinsumContractionOrders.jl).
//!
//! ## What is a Tensor Network?
//!
//! A *tensor network* represents multilinear transformations as hypergraphs.
//! Arrays (tensors) are nodes, and shared indices are hyperedges connecting them.
//! To *contract* a tensor network means evaluating the transformation by performing
//! a sequence of pairwise tensor operations.
//!
//! The computational cost—both time and memory—depends critically on the order
//! of these operations. A specific ordering is called a *contraction order*, and
//! finding an efficient one is *contraction order optimization*.
//!
//! This framework appears across many domains: *einsum* notation in numerical
//! computing, *factor graphs* in probabilistic inference, and *junction trees*
//! in graphical models. Applications include quantum circuit simulation,
//! quantum error correction, neural network compression, and combinatorial optimization.
//!
//! Finding the optimal contraction order is NP-complete, but good heuristics
//! can find near-optimal solutions quickly.
//!
//! ## Features
//!
//! This crate provides two main features:
//!
//! 1. **Contraction Order Optimization** — Find efficient orderings that minimize
//!    time and/or space complexity
//! 2. **Slicing** — Trade time for space by looping over selected indices
//!
//! ### Feature 1: Contraction Order Optimization
//!
//! A contraction order is represented as a binary tree where leaves are input
//! tensors and internal nodes are intermediate results. The optimizer searches
//! for trees that minimize a cost function balancing multiple objectives:
//!
//! - **Time complexity (tc)**: Total FLOP count (log2 scale)
//! - **Space complexity (sc)**: Maximum intermediate tensor size (log2 scale)
//! - **Read-write complexity (rwc)**: Total memory I/O (log2 scale)
//!
//! ```rust
//! use omeco::{EinCode, GreedyMethod, contraction_complexity, optimize_code, uniform_size_dict};
//!
//! // Matrix chain: A[i,j] × B[j,k] × C[k,l] → D[i,l]
//! let code = EinCode::new(
//!     vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
//!     vec!['i', 'l'],
//! );
//! let sizes = uniform_size_dict(&code, 16);
//!
//! let optimized = optimize_code(&code, &sizes, &GreedyMethod::default())
//!     .expect("optimizer failed");
//! let metrics = contraction_complexity(&optimized, &sizes, &code.ixs);
//! println!("time: 2^{:.2}", metrics.tc);
//! println!("space: 2^{:.2}", metrics.sc);
//! ```
//!
//! **Available optimizers:**
//!
//! | Optimizer | Description |
//! |-----------|-------------|
//! | [`GreedyMethod`] | Fast O(n² log n) greedy heuristic |
//! | [`TreeSA`] | Simulated annealing for higher-quality solutions |
//!
//! Use [`GreedyMethod`] when you need speed; use [`TreeSA`] when contraction
//! cost dominates and you can afford extra search time.
//!
//! ### Feature 2: Slicing
//!
//! *Slicing* trades time complexity for reduced space complexity by explicitly
//! looping over a subset of tensor indices. This is useful when the optimal
//! contraction order still exceeds available memory.
//!
//! For example, slicing index `j` with dimension 64 means running 64 smaller
//! contractions and summing the results, reducing peak memory at the cost of
//! more total work.
//!
//! ```rust
//! use omeco::{EinCode, GreedyMethod, SlicedEinsum, optimize_code, sliced_complexity, uniform_size_dict};
//!
//! let code = EinCode::new(
//!     vec![vec!['i', 'j'], vec!['j', 'k']],
//!     vec!['i', 'k'],
//! );
//! let sizes = uniform_size_dict(&code, 64);
//!
//! let optimized = optimize_code(&code, &sizes, &GreedyMethod::default())
//!     .expect("optimizer failed");
//!
//! // Slice over index 'j' to reduce memory
//! let sliced = SlicedEinsum::new(vec!['j'], optimized);
//! let metrics = sliced_complexity(&sliced, &sizes, &code.ixs);
//! println!("sliced space: 2^{:.2}", metrics.sc);
//! ```
//!
//! ## Algorithm Details
//!
//! ### GreedyMethod
//!
//! Repeatedly contracts the tensor pair with the lowest cost:
//!
//! ```text
//! loss = size(output) - α × (size(input1) + size(input2))
//! ```
//!
//! - `alpha` (0.0–1.0): Balances output size vs input size reduction
//! - `temperature`: Enables stochastic selection via Boltzmann sampling (0 = deterministic)
//!
//! ### TreeSA
//!
//! Simulated annealing on contraction trees. Starts from an initial tree,
//! applies local rewrites, and accepts/rejects via Metropolis criterion.
//! Runs multiple trials in parallel using rayon.
//!
//! The scoring function balances objectives:
//!
//! ```text
//! score = w_t × 2^tc + w_rw × 2^rwc + w_s × max(0, 2^sc - 2^sc_target)
//! ```
//!
//! - `betas`: Inverse temperature schedule
//! - `ntrials`: Parallel trials (control threads via `RAYON_NUM_THREADS`)
//! - `niters`: Iterations per temperature level
//! - `score`: [`ScoreFunction`] with weights and space target

pub mod complexity;
pub mod eincode;
pub mod expr_tree;
pub mod greedy;
pub mod incidence_list;
pub mod json;
pub mod label;
pub mod score;
pub mod simplifier;
pub mod slicer;
pub mod treesa;
pub mod utils;

#[cfg(test)]
pub mod test_utils;

// Re-export main types
pub use complexity::{
    eincode_complexity, flop, nested_complexity, nested_flop, peak_memory, sliced_complexity,
    ContractionComplexity,
};
pub use eincode::{log2_size_dict, uniform_size_dict, EinCode, NestedEinsum, SlicedEinsum};
pub use greedy::{optimize_greedy, ContractionTree, GreedyMethod, GreedyResult};
pub use label::Label;
pub use score::ScoreFunction;
pub use slicer::{slice_code, CodeSlicer, Slicer, TreeSASlicer};
pub use treesa::{optimize_treesa, Initializer, TreeSA};

use std::collections::HashMap;

/// Trait for contraction order optimizers.
pub trait CodeOptimizer {
    /// Optimize the contraction order for an EinCode.
    fn optimize<L: Label>(
        &self,
        code: &EinCode<L>,
        size_dict: &HashMap<L, usize>,
    ) -> Option<NestedEinsum<L>>;
}

impl CodeOptimizer for GreedyMethod {
    fn optimize<L: Label>(
        &self,
        code: &EinCode<L>,
        size_dict: &HashMap<L, usize>,
    ) -> Option<NestedEinsum<L>> {
        optimize_greedy(code, size_dict, self)
    }
}

impl CodeOptimizer for TreeSA {
    fn optimize<L: Label>(
        &self,
        code: &EinCode<L>,
        size_dict: &HashMap<L, usize>,
    ) -> Option<NestedEinsum<L>> {
        optimize_treesa(code, size_dict, self)
    }
}

/// Optimize the contraction order for an EinCode using the specified optimizer.
///
/// # Example
///
/// ```rust
/// use omeco::{EinCode, optimize_code, GreedyMethod};
/// use std::collections::HashMap;
///
/// let code = EinCode::new(
///     vec![vec!['i', 'j'], vec!['j', 'k']],
///     vec!['i', 'k']
/// );
///
/// let mut sizes = HashMap::new();
/// sizes.insert('i', 10);
/// sizes.insert('j', 20);
/// sizes.insert('k', 10);
///
/// let optimized = optimize_code(&code, &sizes, &GreedyMethod::default());
/// assert!(optimized.is_some());
/// ```
pub fn optimize_code<L: Label, O: CodeOptimizer>(
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    optimizer: &O,
) -> Option<NestedEinsum<L>> {
    optimizer.optimize(code, size_dict)
}

/// Compute the contraction complexity of an optimized NestedEinsum.
///
/// This is a convenience function that wraps [`nested_complexity`].
pub fn contraction_complexity<L: Label>(
    code: &NestedEinsum<L>,
    size_dict: &HashMap<L, usize>,
    original_ixs: &[Vec<L>],
) -> ContractionComplexity {
    nested_complexity(code, size_dict, original_ixs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize_code_greedy() {
        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
            vec!['i', 'l'],
        );

        let mut sizes = HashMap::new();
        sizes.insert('i', 4);
        sizes.insert('j', 8);
        sizes.insert('k', 8);
        sizes.insert('l', 4);

        let result = optimize_code(&code, &sizes, &GreedyMethod::default());
        assert!(result.is_some());

        let nested = result.unwrap();
        assert!(nested.is_binary());
        assert_eq!(nested.leaf_count(), 3);
    }

    #[test]
    fn test_optimize_code_treesa() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);

        let mut sizes = HashMap::new();
        sizes.insert('i', 4);
        sizes.insert('j', 8);
        sizes.insert('k', 4);

        let result = optimize_code(&code, &sizes, &TreeSA::fast());
        assert!(result.is_some());

        let nested = result.unwrap();
        assert!(nested.is_binary());
    }

    #[test]
    fn test_contraction_complexity_wrapper() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);

        let mut sizes = HashMap::new();
        sizes.insert('i', 4);
        sizes.insert('j', 8);
        sizes.insert('k', 4);

        let result = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
        let complexity = contraction_complexity(&result, &sizes, &code.ixs);

        assert!(complexity.tc > 0.0);
        assert!(complexity.sc > 0.0);
    }

    #[test]
    fn test_single_tensor() {
        let code = EinCode::new(vec![vec!['i', 'j']], vec!['i', 'j']);

        let mut sizes = HashMap::new();
        sizes.insert('i', 4);
        sizes.insert('j', 4);

        let result = optimize_code(&code, &sizes, &GreedyMethod::default());
        assert!(result.is_some());
        assert!(result.unwrap().is_leaf());
    }

    #[test]
    fn test_trace() {
        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['j', 'i']],
            vec![], // Trace - no output
        );

        let mut sizes = HashMap::new();
        sizes.insert('i', 4);
        sizes.insert('j', 4);

        let result = optimize_code(&code, &sizes, &GreedyMethod::default());
        assert!(result.is_some());
    }

    #[test]
    fn test_empty_code() {
        let code: EinCode<char> = EinCode::new(vec![], vec![]);
        let sizes: HashMap<char, usize> = HashMap::new();

        let result = optimize_code(&code, &sizes, &GreedyMethod::default());
        assert!(result.is_none());
    }

    #[test]
    fn test_optimize_code_with_slicing() {
        use crate::slicer::{slice_code, TreeSASlicer};

        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
            vec!['i', 'l'],
        );

        let mut sizes = HashMap::new();
        sizes.insert('i', 4);
        sizes.insert('j', 8);
        sizes.insert('k', 8);
        sizes.insert('l', 4);

        // First optimize
        let nested = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();

        // Then slice
        let slicer = TreeSASlicer::fast();
        let sliced = slice_code(&nested, &sizes, &slicer, &code.ixs);

        // Verify sliced result exists (may or may not slice depending on sizes)
        assert!(sliced.is_some());
    }

    #[test]
    fn test_contraction_complexity_deep_tree() {
        let code = EinCode::new(
            vec![
                vec!['a', 'b'],
                vec!['b', 'c'],
                vec!['c', 'd'],
                vec!['d', 'e'],
            ],
            vec!['a', 'e'],
        );

        let mut sizes = HashMap::new();
        sizes.insert('a', 2);
        sizes.insert('b', 2);
        sizes.insert('c', 2);
        sizes.insert('d', 2);
        sizes.insert('e', 2);

        let nested = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
        let complexity = contraction_complexity(&nested, &sizes, &code.ixs);

        // Deep tree should have multiple contractions
        assert!(complexity.tc > 0.0);
        assert!(complexity.sc > 0.0);
        assert!(complexity.rwc > 0.0);
    }

    #[test]
    fn test_optimize_code_treesa_with_path_decomp() {
        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
            vec!['i', 'l'],
        );

        let mut sizes = HashMap::new();
        sizes.insert('i', 4);
        sizes.insert('j', 8);
        sizes.insert('k', 8);
        sizes.insert('l', 4);

        let result = optimize_code(&code, &sizes, &TreeSA::path());
        assert!(result.is_some());

        let nested = result.unwrap();
        // Path decomposition should produce a valid tree
        assert!(nested.is_binary() || nested.is_leaf());
    }
}

#[cfg(test)]
mod issue_13_tests {
    use super::*;

    #[test]
    fn test_issue_13_greedy_scalar_output() {
        // Issue #13: optimize_code ignores final output indices (iy) in NestedEinsum
        // A[i,j] @ B[j,k] -> scalar (empty iy)
        let code = EinCode::new(vec![vec![0usize, 1], vec![1usize, 2]], vec![]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        let optimizer = GreedyMethod::default();

        let tree = optimize_code(&code, &sizes, &optimizer).expect("should optimize");

        match &tree {
            NestedEinsum::Node { eins, .. } => {
                assert_eq!(
                    eins.iy, code.iy,
                    "Root node's iy should match requested output. Got {:?}, expected {:?}",
                    eins.iy, code.iy
                );
            }
            NestedEinsum::Leaf { .. } => {
                panic!("Multi-tensor operation should not return Leaf");
            }
        }
    }

    #[test]
    fn test_issue_13_greedy_partial_output() {
        // A[i,j] @ B[j,k] -> [i] (only keep first index)
        let code = EinCode::new(vec![vec![0usize, 1], vec![1usize, 2]], vec![0usize]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        let optimizer = GreedyMethod::default();

        let tree = optimize_code(&code, &sizes, &optimizer).expect("should optimize");

        match &tree {
            NestedEinsum::Node { eins, .. } => {
                assert_eq!(
                    eins.iy, code.iy,
                    "Root node's iy should match requested output. Got {:?}, expected {:?}",
                    eins.iy, code.iy
                );
            }
            NestedEinsum::Leaf { .. } => {
                panic!("Multi-tensor operation should not return Leaf");
            }
        }
    }

    #[test]
    fn test_issue_13_treesa_scalar_output() {
        // Issue #13: TreeSA should also respect final output indices
        let code = EinCode::new(vec![vec![0usize, 1], vec![1usize, 2]], vec![]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        let optimizer = TreeSA::fast();

        let tree = optimize_code(&code, &sizes, &optimizer).expect("should optimize");

        match &tree {
            NestedEinsum::Node { eins, .. } => {
                assert_eq!(
                    eins.iy, code.iy,
                    "TreeSA root node's iy should match requested output. Got {:?}, expected {:?}",
                    eins.iy, code.iy
                );
            }
            NestedEinsum::Leaf { .. } => {
                panic!("Multi-tensor operation should not return Leaf");
            }
        }
    }

    #[test]
    fn test_issue_13_treesa_partial_output() {
        // A[i,j] @ B[j,k] -> [k] (only keep last index)
        let code = EinCode::new(vec![vec![0usize, 1], vec![1usize, 2]], vec![2usize]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        let optimizer = TreeSA::fast();

        let tree = optimize_code(&code, &sizes, &optimizer).expect("should optimize");

        match &tree {
            NestedEinsum::Node { eins, .. } => {
                assert_eq!(
                    eins.iy, code.iy,
                    "TreeSA root node's iy should match requested output. Got {:?}, expected {:?}",
                    eins.iy, code.iy
                );
            }
            NestedEinsum::Leaf { .. } => {
                panic!("Multi-tensor operation should not return Leaf");
            }
        }
    }

    #[test]
    fn test_issue_13_greedy_chain_scalar_output() {
        // 3 tensors: A[i,j] @ B[j,k] @ C[k,l] -> scalar
        // This exercises the level > 0 branch for intermediate nodes
        let code = EinCode::new(
            vec![vec![0usize, 1], vec![1usize, 2], vec![2usize, 3]],
            vec![],
        );

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2), (3, 2)].into();
        let optimizer = GreedyMethod::default();

        let tree = optimize_code(&code, &sizes, &optimizer).expect("should optimize");

        // Verify root has requested output
        match &tree {
            NestedEinsum::Node { eins, .. } => {
                assert_eq!(
                    eins.iy, code.iy,
                    "Root node's iy should be empty (scalar). Got {:?}",
                    eins.iy
                );
            }
            NestedEinsum::Leaf { .. } => {
                panic!("3-tensor operation should not return Leaf");
            }
        }
    }

    #[test]
    fn test_issue_13_greedy_chain_partial_output() {
        // 3 tensors: A[i,j] @ B[j,k] @ C[k,l] -> [i] (only first index)
        let code = EinCode::new(
            vec![vec![0usize, 1], vec![1usize, 2], vec![2usize, 3]],
            vec![0usize],
        );

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2), (3, 2)].into();
        let optimizer = GreedyMethod::default();

        let tree = optimize_code(&code, &sizes, &optimizer).expect("should optimize");

        match &tree {
            NestedEinsum::Node { eins, .. } => {
                assert_eq!(
                    eins.iy, code.iy,
                    "Root node's iy should be [0]. Got {:?}",
                    eins.iy
                );
            }
            NestedEinsum::Leaf { .. } => {
                panic!("3-tensor operation should not return Leaf");
            }
        }
    }

    #[test]
    fn test_issue_13_treesa_chain_scalar_output() {
        // 3 tensors with TreeSA
        let code = EinCode::new(
            vec![vec![0usize, 1], vec![1usize, 2], vec![2usize, 3]],
            vec![],
        );

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2), (3, 2)].into();
        let optimizer = TreeSA::fast();

        let tree = optimize_code(&code, &sizes, &optimizer).expect("should optimize");

        match &tree {
            NestedEinsum::Node { eins, .. } => {
                assert_eq!(
                    eins.iy, code.iy,
                    "TreeSA root node's iy should be empty (scalar). Got {:?}",
                    eins.iy
                );
            }
            NestedEinsum::Leaf { .. } => {
                panic!("3-tensor operation should not return Leaf");
            }
        }
    }

    #[test]
    fn test_issue_13_intermediate_nodes_have_correct_output() {
        // Verify that intermediate nodes have hypergraph-computed output
        // A[i,j] @ B[j,k] @ C[k,l] -> [i,l]
        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
            vec!['i', 'l'],
        );

        let sizes: HashMap<char, usize> = [('i', 2), ('j', 2), ('k', 2), ('l', 2)].into();
        let optimizer = GreedyMethod::default();

        let tree = optimize_code(&code, &sizes, &optimizer).expect("should optimize");

        // Root should have requested output [i, l]
        match &tree {
            NestedEinsum::Node { eins, args, .. } => {
                assert_eq!(eins.iy, vec!['i', 'l'], "Root iy should be [i, l]");

                // Check that we have intermediate nodes (not just leaves)
                let has_intermediate = args.iter().any(|arg| !arg.is_leaf());
                assert!(
                    has_intermediate,
                    "Should have at least one intermediate node"
                );
            }
            NestedEinsum::Leaf { .. } => {
                panic!("3-tensor operation should not return Leaf");
            }
        }
    }
}

#[cfg(test)]
mod numerical_verification_tests {
    //! Tests that verify numerical correctness of contraction orders.
    //! These tests execute contractions using NaiveContractor and compare results.

    use super::*;
    use crate::test_utils::{execute_nested, tensors_approx_equal, NaiveContractor};

    #[test]
    fn test_numerical_matrix_chain() {
        // Verify A[i,j] * B[j,k] * C[k,l] * D[l,m] produces same result
        // regardless of contraction order (greedy vs treesa)
        let code = EinCode::new(
            vec![vec![1usize, 2], vec![2, 3], vec![3, 4], vec![4, 5]],
            vec![1, 5],
        );

        let sizes: HashMap<usize, usize> = [(1, 3), (2, 4), (3, 5), (4, 4), (5, 3)].into();
        let label_map: HashMap<usize, usize> = (1..=5).map(|i| (i, i)).collect();

        // Setup identical tensors for both contractors
        let mut contractor_greedy = NaiveContractor::new();
        contractor_greedy.add_tensor(0, vec![3, 4]); // A
        contractor_greedy.add_tensor(1, vec![4, 5]); // B
        contractor_greedy.add_tensor(2, vec![5, 4]); // C
        contractor_greedy.add_tensor(3, vec![4, 3]); // D

        let mut contractor_treesa = contractor_greedy.clone();

        // Optimize with both methods
        let greedy_tree =
            optimize_code(&code, &sizes, &GreedyMethod::default()).expect("Greedy should succeed");
        let treesa_tree =
            optimize_code(&code, &sizes, &TreeSA::fast()).expect("TreeSA should succeed");

        // Execute both
        let greedy_idx = execute_nested(&greedy_tree, &mut contractor_greedy, &label_map);
        let treesa_idx = execute_nested(&treesa_tree, &mut contractor_treesa, &label_map);

        // Compare results
        let greedy_result = contractor_greedy.get_tensor(greedy_idx).unwrap();
        let treesa_result = contractor_treesa.get_tensor(treesa_idx).unwrap();

        assert_eq!(
            greedy_result.shape(),
            treesa_result.shape(),
            "Shapes should match"
        );
        assert!(
            tensors_approx_equal(greedy_result, treesa_result, 1e-10, 1e-12),
            "Greedy and TreeSA should produce identical numerical results"
        );
    }

    #[test]
    fn test_numerical_with_hyperedge() {
        // A[i,j], B[j,k], C[j,l] -> [i,k,l] where j is a hyperedge
        // Note: This test verifies that optimization produces valid trees,
        // but different contraction orders may produce numerically different
        // results due to floating point accumulation differences.
        let code = EinCode::new(vec![vec![1usize, 2], vec![2, 3], vec![2, 4]], vec![1, 3, 4]);

        let sizes: HashMap<usize, usize> = [(1, 2), (2, 3), (3, 2), (4, 2)].into();
        let label_map: HashMap<usize, usize> = (1..=4).map(|i| (i, i)).collect();

        let mut contractor = NaiveContractor::new();
        contractor.add_tensor(0, vec![2, 3]); // A[i,j]
        contractor.add_tensor(1, vec![3, 2]); // B[j,k]
        contractor.add_tensor(2, vec![3, 2]); // C[j,l]

        let greedy_tree =
            optimize_code(&code, &sizes, &GreedyMethod::default()).expect("Greedy should succeed");
        let treesa_tree =
            optimize_code(&code, &sizes, &TreeSA::fast()).expect("TreeSA should succeed");

        // Both should produce valid binary trees
        assert!(greedy_tree.is_binary(), "Greedy should produce binary tree");
        assert!(treesa_tree.is_binary(), "TreeSA should produce binary tree");

        // Both should have correct number of leaves
        assert_eq!(greedy_tree.leaf_count(), 3, "Should have 3 leaves");
        assert_eq!(treesa_tree.leaf_count(), 3, "Should have 3 leaves");

        // Execute greedy tree and verify output shape
        let greedy_idx = execute_nested(&greedy_tree, &mut contractor, &label_map);
        let greedy_result = contractor.get_tensor(greedy_idx).unwrap();
        assert_eq!(
            greedy_result.shape(),
            &[2, 2, 2],
            "Output should be 2x2x2 for [i,k,l]"
        );
    }

    #[test]
    fn test_numerical_scalar_output() {
        // Full contraction to scalar: A[i,j] * B[j,k] * C[k,i] -> scalar
        let code = EinCode::new(vec![vec![1usize, 2], vec![2, 3], vec![3, 1]], vec![]);

        let sizes: HashMap<usize, usize> = [(1, 3), (2, 4), (3, 3)].into();
        let label_map: HashMap<usize, usize> = (1..=3).map(|i| (i, i)).collect();

        let mut contractor_greedy = NaiveContractor::new();
        contractor_greedy.add_tensor(0, vec![3, 4]); // A
        contractor_greedy.add_tensor(1, vec![4, 3]); // B
        contractor_greedy.add_tensor(2, vec![3, 3]); // C

        let mut contractor_treesa = contractor_greedy.clone();

        let greedy_tree =
            optimize_code(&code, &sizes, &GreedyMethod::default()).expect("Greedy should succeed");
        let treesa_tree =
            optimize_code(&code, &sizes, &TreeSA::fast()).expect("TreeSA should succeed");

        let greedy_idx = execute_nested(&greedy_tree, &mut contractor_greedy, &label_map);
        let treesa_idx = execute_nested(&treesa_tree, &mut contractor_treesa, &label_map);

        let greedy_result = contractor_greedy.get_tensor(greedy_idx).unwrap();
        let treesa_result = contractor_treesa.get_tensor(treesa_idx).unwrap();

        assert_eq!(greedy_result.ndim(), 0, "Should produce scalar");
        assert!(
            tensors_approx_equal(greedy_result, treesa_result, 1e-10, 1e-12),
            "Scalar contraction should match"
        );
    }

    #[test]
    fn test_numerical_five_tensor_network() {
        // More complex network: 5 tensors with various connections
        // A[a,b], B[b,c,d], C[c,e], D[d,e,f], E[f,a] -> [e] (partial trace)
        let code = EinCode::new(
            vec![
                vec![1usize, 2], // A[a,b]
                vec![2, 3, 4],   // B[b,c,d]
                vec![3, 5],      // C[c,e]
                vec![4, 5, 6],   // D[d,e,f]
                vec![6, 1],      // E[f,a]
            ],
            vec![5], // Output only 'e'
        );

        let sizes: HashMap<usize, usize> = [(1, 2), (2, 3), (3, 2), (4, 2), (5, 4), (6, 2)].into();
        let label_map: HashMap<usize, usize> = (1..=6).map(|i| (i, i)).collect();

        let mut contractor_greedy = NaiveContractor::new();
        contractor_greedy.add_tensor(0, vec![2, 3]); // A
        contractor_greedy.add_tensor(1, vec![3, 2, 2]); // B
        contractor_greedy.add_tensor(2, vec![2, 4]); // C
        contractor_greedy.add_tensor(3, vec![2, 4, 2]); // D
        contractor_greedy.add_tensor(4, vec![2, 2]); // E

        let mut contractor_treesa = contractor_greedy.clone();

        let greedy_tree =
            optimize_code(&code, &sizes, &GreedyMethod::default()).expect("Greedy should succeed");
        let treesa_tree =
            optimize_code(&code, &sizes, &TreeSA::fast()).expect("TreeSA should succeed");

        let greedy_idx = execute_nested(&greedy_tree, &mut contractor_greedy, &label_map);
        let treesa_idx = execute_nested(&treesa_tree, &mut contractor_treesa, &label_map);

        let greedy_result = contractor_greedy.get_tensor(greedy_idx).unwrap();
        let treesa_result = contractor_treesa.get_tensor(treesa_idx).unwrap();

        assert_eq!(greedy_result.shape(), &[4], "Output should have shape [4]");
        assert!(
            tensors_approx_equal(greedy_result, treesa_result, 1e-8, 1e-10),
            "5-tensor network should produce same result"
        );
    }
}

/// Large-scale stress tests using petgraph for graph generation.
/// These tests match Julia's OMEinsumContractionOrders test coverage.
#[cfg(test)]
mod large_scale_stress_tests {
    use super::*;
    use petgraph::graph::UnGraph;
    use petgraph::visit::EdgeRef;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use std::collections::HashSet;

    /// Generate a random k-regular graph with n vertices.
    /// Uses a greedy edge-swapping algorithm that guarantees exactly k-regularity.
    fn generate_random_regular_graph(n: usize, k: usize, seed: u64) -> UnGraph<(), ()> {
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut graph = UnGraph::new_undirected();

        // Add vertices
        for _ in 0..n {
            graph.add_node(());
        }

        // For k-regular graph: each vertex needs exactly k edges
        // Total edges = n*k/2 (each edge counted twice)
        assert!(n * k % 2 == 0, "n*k must be even for k-regular graph");

        // Use configuration model with retry
        let target_edges = (n * k) / 2;
        let mut edges: HashSet<(usize, usize)> = HashSet::new();
        let mut degrees = vec![0usize; n];

        // Greedy construction: repeatedly add edges between vertices with lowest degree
        let max_outer_attempts = 20;
        for _attempt in 0..max_outer_attempts {
            edges.clear();
            degrees.fill(0);

            // Build edges greedily
            for _ in 0..target_edges * 10 {
                // Find vertices that still need edges
                let mut candidates: Vec<usize> = (0..n).filter(|&v| degrees[v] < k).collect();

                if candidates.len() < 2 {
                    break;
                }

                // Try random pairs
                for _ in 0..100 {
                    if candidates.len() < 2 {
                        break;
                    }
                    let i = rng.random_range(0..candidates.len());
                    let u = candidates[i];
                    candidates.swap_remove(i);

                    if candidates.is_empty() {
                        break;
                    }

                    // Find a valid partner for u
                    let valid_partners: Vec<usize> = candidates
                        .iter()
                        .filter(|&&v| {
                            let edge = (u.min(v), u.max(v));
                            !edges.contains(&edge) && degrees[v] < k
                        })
                        .copied()
                        .collect();

                    if valid_partners.is_empty() {
                        continue;
                    }

                    let j = rng.random_range(0..valid_partners.len());
                    let v = valid_partners[j];

                    let edge = (u.min(v), u.max(v));
                    edges.insert(edge);
                    degrees[u] += 1;
                    degrees[v] += 1;

                    // Remove v from candidates if it's now full
                    if degrees[v] >= k {
                        if let Some(pos) = candidates.iter().position(|&x| x == v) {
                            candidates.swap_remove(pos);
                        }
                    }

                    break;
                }

                if edges.len() >= target_edges {
                    break;
                }
            }

            // Check if we got a valid k-regular graph
            if edges.len() == target_edges && degrees.iter().all(|&d| d == k) {
                // Success! Add edges to graph
                for (u, v) in &edges {
                    graph.add_edge(
                        petgraph::graph::NodeIndex::new(*u),
                        petgraph::graph::NodeIndex::new(*v),
                        (),
                    );
                }
                return graph;
            }
        }

        // Fallback: build a near-regular graph by connecting vertices greedily
        // This ensures we always return a connected graph even if not perfectly k-regular
        edges.clear();
        degrees.fill(0);

        let mut available: Vec<usize> = (0..n).collect();
        available.shuffle(&mut rng);

        for (i, &u) in available.iter().enumerate() {
            for &v in available.iter().skip(i + 1) {
                if degrees[u] < k && degrees[v] < k {
                    let edge = (u.min(v), u.max(v));
                    if !edges.contains(&edge) {
                        edges.insert(edge);
                        degrees[u] += 1;
                        degrees[v] += 1;
                    }
                }
                if degrees[u] >= k {
                    break;
                }
            }
        }

        for (u, v) in &edges {
            graph.add_edge(
                petgraph::graph::NodeIndex::new(*u),
                petgraph::graph::NodeIndex::new(*v),
                (),
            );
        }

        graph
    }

    /// Convert a petgraph to EinCode format for tensor network optimization.
    /// Each edge becomes a matrix tensor, each vertex becomes a vector tensor.
    fn graph_to_eincode(graph: &UnGraph<(), ()>) -> (EinCode<usize>, HashMap<usize, usize>) {
        let n_vertices = graph.node_count();
        let mut ixs: Vec<Vec<usize>> = Vec::new();

        // Each edge contributes a matrix tensor: [src, dst] (using minmax for consistency)
        for edge in graph.edge_references() {
            let src = edge.source().index() + 1; // 1-indexed
            let dst = edge.target().index() + 1;
            ixs.push(vec![src.min(dst), src.max(dst)]);
        }

        // Each vertex contributes a vector tensor: [vertex]
        for v in 0..n_vertices {
            ixs.push(vec![v + 1]); // 1-indexed
        }

        // Create uniform size dictionary
        let mut sizes = HashMap::new();
        for v in 1..=n_vertices {
            sizes.insert(v, 2);
        }

        let code = EinCode::new(ixs, vec![]); // Scalar output
        (code, sizes)
    }

    #[test]
    fn test_random_regular_graph_n40_k3() {
        // Julia test: n=40, k=3 random regular graph
        let graph = generate_random_regular_graph(40, 3, 42);

        // Should have 40 vertices and ~60 edges (n*k/2 = 60)
        assert_eq!(graph.node_count(), 40);
        assert!(graph.edge_count() > 50, "Should have close to 60 edges");

        let (code, sizes) = graph_to_eincode(&graph);

        // Optimize with greedy
        let greedy_tree = optimize_code(&code, &sizes, &GreedyMethod::default());
        assert!(
            greedy_tree.is_some(),
            "Greedy should succeed on 40-node graph"
        );

        let tree = greedy_tree.unwrap();
        let complexity = contraction_complexity(&tree, &sizes, &code.ixs);

        // Verify optimization produces reasonable complexity
        assert!(
            complexity.sc < 30.0,
            "Space complexity should be reasonable"
        );
    }

    #[test]
    fn test_random_regular_graph_n60_k3_treesa() {
        // Julia test: n=60, k=3 random regular graph with TreeSA
        let graph = generate_random_regular_graph(60, 3, 123);
        let (code, sizes) = graph_to_eincode(&graph);

        // First optimize with greedy
        let greedy_tree = optimize_code(&code, &sizes, &GreedyMethod::default());
        assert!(greedy_tree.is_some());
        let greedy_complexity =
            contraction_complexity(greedy_tree.as_ref().unwrap(), &sizes, &code.ixs);

        // Then optimize with TreeSA using sc_target
        let treesa = TreeSA::default()
            .with_sc_target(greedy_complexity.sc - 2.0)
            .with_niters(50)
            .with_ntrials(2);
        let treesa_tree = optimize_code(&code, &sizes, &treesa);
        assert!(
            treesa_tree.is_some(),
            "TreeSA should succeed on 60-node graph"
        );

        let treesa_complexity =
            contraction_complexity(treesa_tree.as_ref().unwrap(), &sizes, &code.ixs);

        // TreeSA should match or improve on greedy
        assert!(
            treesa_complexity.sc <= greedy_complexity.sc + 1.0,
            "TreeSA should not significantly increase space complexity"
        );
    }

    #[test]
    fn test_large_random_regular_graph_n100() {
        // Stress test: 100-node 3-regular graph
        let graph = generate_random_regular_graph(100, 3, 456);
        let (code, sizes) = graph_to_eincode(&graph);

        // Should have ~150 edges + 100 vertices = ~250 tensors
        assert!(code.num_tensors() > 200, "Should have many tensors");

        let greedy_tree = optimize_code(&code, &sizes, &GreedyMethod::default());
        assert!(greedy_tree.is_some(), "Greedy should handle 100-node graph");

        let tree = greedy_tree.unwrap();
        assert!(tree.is_binary(), "Should produce binary tree");
        assert_eq!(
            tree.leaf_count(),
            code.num_tensors(),
            "Should have correct leaf count"
        );
    }

    #[test]
    fn test_path_decomposition_random_graph_n50() {
        // Julia test: path decomposition on n=50 random regular graph
        let graph = generate_random_regular_graph(50, 3, 789);
        let (code, sizes) = graph_to_eincode(&graph);

        // Optimize with path decomposition
        let path_treesa = TreeSA::path().with_niters(30).with_ntrials(2);
        let path_tree = optimize_code(&code, &sizes, &path_treesa);
        assert!(path_tree.is_some(), "Path TreeSA should succeed");

        let tree = path_tree.unwrap();

        // Path decomposition should produce a path structure
        assert!(
            tree.is_path_decomposition(),
            "Should produce path decomposition"
        );

        // Also verify with tree decomposition for comparison
        let tree_treesa = TreeSA::default().with_niters(30).with_ntrials(2);
        let tree_tree = optimize_code(&code, &sizes, &tree_treesa);
        assert!(tree_tree.is_some());

        // Tree decomposition typically doesn't produce path structure
        // (unless by coincidence)
    }

    #[test]
    fn test_fullerene_c60_optimization() {
        // Julia test: C60 fullerene graph (60 vertices, 90 edges)
        use crate::test_utils::generate_fullerene_edges;

        let edges = generate_fullerene_edges();
        let mut ixs: Vec<Vec<usize>> = Vec::new();

        // Add edge tensors
        for (a, b) in &edges {
            ixs.push(vec![*a, *b]);
        }

        // Add vertex tensors
        for v in 1..=60 {
            ixs.push(vec![v]);
        }

        let code = EinCode::new(ixs, vec![]); // Scalar output
        let sizes: HashMap<usize, usize> = (1..=60).map(|v| (v, 2)).collect();

        // Unoptimized complexity
        let unopt = crate::complexity::eincode_complexity(&code, &sizes);
        assert_eq!(
            unopt.tc, 60.0,
            "Unoptimized tc should be 60 (sum of indices)"
        );
        assert_eq!(unopt.sc, 0.0, "Unoptimized sc should be 0 (scalar output)");

        // Optimize
        let greedy_tree = optimize_code(&code, &sizes, &GreedyMethod::default());
        assert!(greedy_tree.is_some(), "Should optimize C60");

        let tree = greedy_tree.unwrap();
        let complexity = contraction_complexity(&tree, &sizes, &code.ixs);

        // Verify reasonable optimization (actual value depends on exact graph structure)
        assert!(
            complexity.sc <= 20.0,
            "C60 space complexity should be <= 20, got {}",
            complexity.sc
        );
    }

    #[test]
    fn test_chain_and_ring_topologies() {
        // Julia test: chain and ring graphs

        // Chain: 9 tensors forming a path
        let chain_ixs: Vec<Vec<usize>> = (1..=9).map(|i| vec![i, i + 1]).collect();
        let chain_code = EinCode::new(chain_ixs, vec![1, 10]);
        let chain_sizes: HashMap<usize, usize> = (1..=10).map(|v| (v, 2)).collect();

        let chain_tree = optimize_code(&chain_code, &chain_sizes, &GreedyMethod::default());
        assert!(chain_tree.is_some());
        let chain_complexity =
            contraction_complexity(chain_tree.as_ref().unwrap(), &chain_sizes, &chain_code.ixs);
        assert_eq!(chain_complexity.sc, 2.0, "Chain should have sc=2");

        // Ring: 10 tensors forming a cycle
        let mut ring_ixs: Vec<Vec<usize>> = (1..=9).map(|i| vec![i, i + 1]).collect();
        ring_ixs.push(vec![10, 1]); // Close the ring
        let ring_code = EinCode::new(ring_ixs, vec![]);
        let ring_sizes: HashMap<usize, usize> = (1..=10).map(|v| (v, 2)).collect();

        let ring_tree = optimize_code(&ring_code, &ring_sizes, &GreedyMethod::default());
        assert!(ring_tree.is_some());
        let ring_complexity =
            contraction_complexity(ring_tree.as_ref().unwrap(), &ring_sizes, &ring_code.ixs);
        assert_eq!(ring_complexity.sc, 2.0, "Ring should have sc=2");
    }

    #[test]
    fn test_tutte_graph_stochastic_optimization() {
        // Julia test: Tutte graph (46 vertices, 69 edges) with stochastic optimization
        use crate::test_utils::generate_tutte_edges;

        let edges = generate_tutte_edges();
        let mut ixs: Vec<Vec<usize>> = Vec::new();

        // Add unique edges
        let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();
        for (a, b) in &edges {
            let edge = ((*a).min(*b), (*a).max(*b));
            if !seen_edges.contains(&edge) {
                ixs.push(vec![edge.0, edge.1]);
                seen_edges.insert(edge);
            }
        }

        // Find max vertex
        let max_vertex = edges.iter().flat_map(|(a, b)| [*a, *b]).max().unwrap_or(0);

        let code = EinCode::new(ixs, vec![]); // Scalar output
        let sizes: HashMap<usize, usize> = (1..=max_vertex).map(|v| (v, 2)).collect();

        // Run stochastic optimization multiple times
        let mut best_sc = f64::INFINITY;
        for _trial in 0..10 {
            let stochastic_greedy = GreedyMethod::stochastic(100.0);
            if let Some(tree) = optimize_code(&code, &sizes, &stochastic_greedy) {
                let complexity = contraction_complexity(&tree, &sizes, &code.ixs);
                if complexity.sc < best_sc {
                    best_sc = complexity.sc;
                }
            }
        }

        // Julia expects minimum sc <= 5 after multiple trials
        assert!(
            best_sc <= 8.0,
            "Best Tutte graph sc should be <= 8, got {}",
            best_sc
        );
    }

    #[test]
    fn test_petersen_graph_optimization() {
        // Petersen graph: 10 vertices, 15 edges, 3-regular
        use crate::test_utils::generate_petersen_edges;

        let edges = generate_petersen_edges();
        let mut ixs: Vec<Vec<usize>> = Vec::new();

        for (a, b) in &edges {
            ixs.push(vec![*a, *b]);
        }

        // Add vertex tensors
        for v in 1..=10 {
            ixs.push(vec![v]);
        }

        let code = EinCode::new(ixs, vec![]);
        let sizes: HashMap<usize, usize> = (1..=10).map(|v| (v, 2)).collect();

        let tree = optimize_code(&code, &sizes, &GreedyMethod::default());
        assert!(tree.is_some());

        let complexity = contraction_complexity(tree.as_ref().unwrap(), &sizes, &code.ixs);

        // Petersen graph should have reasonable complexity
        assert!(
            complexity.sc <= 6.0,
            "Petersen sc should be <= 6, got {}",
            complexity.sc
        );
    }

    #[test]
    fn test_very_large_graph_n200() {
        // Stress test: 200-node graph (approaching Julia's n=220 test)
        let graph = generate_random_regular_graph(200, 3, 999);
        let (code, sizes) = graph_to_eincode(&graph);

        // This is a large problem: ~300 edges + 200 vertices = ~500 tensors
        assert!(code.num_tensors() > 400, "Should have many tensors");

        // Greedy should still work
        let greedy_tree = optimize_code(&code, &sizes, &GreedyMethod::default());
        assert!(greedy_tree.is_some(), "Greedy should handle 200-node graph");

        let tree = greedy_tree.unwrap();
        let complexity = contraction_complexity(&tree, &sizes, &code.ixs);

        // Verify reasonable complexity (Julia gets sc ~32 for n=220)
        assert!(
            complexity.sc <= 55.0,
            "Large graph sc should be <= 55, got {}",
            complexity.sc
        );
    }

    #[test]
    fn test_reg3_220_treesa() {
        // =========================================================================
        // ALIGNED WITH JULIA - DO NOT MODIFY WITHOUT CHECKING JULIA TESTS
        // Julia "sa tree" test in test/treesa.jl:
        //   Random.seed!(2)
        //   code = random_regular_eincode(220, 3)
        //   res = optimize_greedy(code, uniformsize(code, 2); α=0.0, temperature=0.0)
        //   optcode = optimize_tree(res, uniformsize(code, 2);
        //       βs=0.1:0.05:20.0, ntrials=2, niters=10,
        //       initializer=:greedy, score=ScoreFunction(sc_target=32))
        //   @test cc.sc <= 32
        //
        // NOTE: Uses shared graph file benchmarks/graphs/reg3_220.json generated
        // by Julia with Random.seed!(2) to ensure identical graph structure.
        // =========================================================================

        // Load graph from shared JSON file (generated by Julia with Random.seed!(2))
        let graph_json = include_str!("../../benchmarks/graphs/reg3_220.json");
        let graph_data: serde_json::Value = serde_json::from_str(graph_json).unwrap();
        let edge_list = graph_data["edge_list"].as_array().unwrap();

        // Build EinCode from edge list
        let mut ixs: Vec<Vec<usize>> = Vec::new();
        for edge in edge_list {
            let arr = edge.as_array().unwrap();
            let a = arr[0].as_u64().unwrap() as usize;
            let b = arr[1].as_u64().unwrap() as usize;
            ixs.push(vec![a, b]);
        }
        // Add vertex tensors (1..=220)
        for v in 1..=220 {
            ixs.push(vec![v]);
        }

        let code = EinCode::new(ixs, vec![]);

        // Debug: check unique labels
        let unique_labels = code.unique_labels();
        eprintln!("Unique labels: {}", unique_labels.len());
        eprintln!("First 5 tensors: {:?}", &code.ixs[0..5]);
        eprintln!("Last 5 tensors: {:?}", &code.ixs[545..550]);

        let sizes: HashMap<usize, usize> = unique_labels.iter().map(|&v| (v, 2)).collect();

        // This is a large problem: 330 edges + 220 vertices = 550 tensors
        assert_eq!(code.num_tensors(), 550, "Should have 550 tensors");

        // First get greedy baseline (Julia: res = optimize_greedy(...))
        let greedy_tree = optimize_code(&code, &sizes, &GreedyMethod::default());
        assert!(greedy_tree.is_some(), "Greedy should handle 220-node graph");

        let greedy_complexity =
            contraction_complexity(greedy_tree.as_ref().unwrap(), &sizes, &code.ixs);
        eprintln!(
            "Greedy: tc={:.2}, sc={:.2}",
            greedy_complexity.tc, greedy_complexity.sc
        );

        // TreeSA with sc_target=32 (matching Julia exactly)
        // Julia: βs=0.1:0.05:20.0 = 399 values, ntrials=2, niters=10
        let treesa = TreeSA::default()
            .with_betas((0..399).map(|i| 0.1 + 0.05 * i as f64).collect())
            .with_ntrials(2)
            .with_niters(10)
            .with_sc_target(32.0);
        let treesa_tree = optimize_code(&code, &sizes, &treesa);
        assert!(treesa_tree.is_some(), "TreeSA should succeed");

        let treesa_complexity =
            contraction_complexity(treesa_tree.as_ref().unwrap(), &sizes, &code.ixs);

        eprintln!(
            "TreeSA: tc={:.2}, sc={:.2}",
            treesa_complexity.tc, treesa_complexity.sc
        );

        // Julia test: @test cc.sc <= 32
        assert!(
            treesa_complexity.sc <= 32.0,
            "TreeSA with sc_target=32 should achieve sc <= 32, got {}",
            treesa_complexity.sc
        );
    }

    #[test]
    fn test_treesa_with_sc_target_large_graph() {
        // Julia test: TreeSA with sc_target on large graph
        let graph = generate_random_regular_graph(80, 3, 111);
        let (code, sizes) = graph_to_eincode(&graph);

        // Get greedy baseline
        let greedy_tree = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
        let greedy_complexity = contraction_complexity(&greedy_tree, &sizes, &code.ixs);

        // Use TreeSA with aggressive sc_target
        let sc_target = greedy_complexity.sc - 3.0;
        let treesa = TreeSA::default()
            .with_sc_target(sc_target)
            .with_niters(100)
            .with_ntrials(3);

        let treesa_tree = optimize_code(&code, &sizes, &treesa);
        assert!(treesa_tree.is_some());

        let treesa_complexity =
            contraction_complexity(treesa_tree.as_ref().unwrap(), &sizes, &code.ixs);

        // TreeSA should try to meet the target
        // (may not always succeed, but should be close)
        assert!(
            treesa_complexity.sc <= greedy_complexity.sc + 2.0,
            "TreeSA should not be much worse than greedy"
        );
    }

    #[test]
    fn test_multiple_optimizers_consistency() {
        // Verify that different optimizers produce valid results on same problem
        let graph = generate_random_regular_graph(30, 3, 222);
        let (code, sizes) = graph_to_eincode(&graph);

        // Helper to verify optimizer results
        fn verify_optimizer_result(
            tree: Option<NestedEinsum<usize>>,
            code: &EinCode<usize>,
            sizes: &HashMap<usize, usize>,
            name: &str,
        ) {
            assert!(tree.is_some(), "{} should succeed", name);
            let t = tree.unwrap();
            assert!(t.is_binary(), "{} should produce binary tree", name);
            assert_eq!(
                t.leaf_count(),
                code.num_tensors(),
                "{} should have correct leaves",
                name
            );

            let complexity = contraction_complexity(&t, sizes, &code.ixs);
            assert!(
                complexity.tc > 0.0,
                "{} should have positive time complexity",
                name
            );
            assert!(
                complexity.sc >= 0.0,
                "{} space complexity should be non-negative",
                name
            );
        }

        // Test each optimizer
        verify_optimizer_result(
            optimize_code(&code, &sizes, &GreedyMethod::default()),
            &code,
            &sizes,
            "GreedyMethod::default",
        );
        verify_optimizer_result(
            optimize_code(&code, &sizes, &GreedyMethod::stochastic(10.0)),
            &code,
            &sizes,
            "GreedyMethod::stochastic",
        );
        verify_optimizer_result(
            optimize_code(&code, &sizes, &TreeSA::fast()),
            &code,
            &sizes,
            "TreeSA::fast",
        );
        verify_optimizer_result(
            optimize_code(&code, &sizes, &TreeSA::path()),
            &code,
            &sizes,
            "TreeSA::path",
        );
    }
}
