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
pub mod label;
pub mod score;
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
