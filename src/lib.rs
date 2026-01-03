//! # omeco - Tensor Network Contraction Order Optimization
//!
//! A Rust library for optimizing tensor network contraction orders, ported from
//! the Julia package [OMEinsumContractionOrders.jl](https://github.com/TensorBFS/OMEinsumContractionOrders.jl).
//!
//! ## Overview
//!
//! When contracting multiple tensors together, the order of contractions significantly
//! affects the computational cost. Finding the optimal contraction order is NP-complete,
//! but good heuristics can find near-optimal solutions quickly.
//!
//! This library provides two main optimization algorithms:
//!
//! - **GreedyMethod**: Fast O(n² log n) greedy algorithm that iteratively contracts
//!   the tensor pair with minimum cost.
//!
//! - **TreeSA**: Simulated annealing algorithm that searches for better contraction
//!   orders by applying local tree mutations.
//!
//! ## Quick Start
//!
//! ```rust
//! use omeco::{EinCode, optimize_code, GreedyMethod, TreeSA, contraction_complexity};
//! use std::collections::HashMap;
//!
//! // Define an einsum: matrix chain multiplication A[i,j] * B[j,k] * C[k,l] -> D[i,l]
//! let code = EinCode::new(
//!     vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
//!     vec!['i', 'l']
//! );
//!
//! // Define tensor dimensions
//! let mut sizes = HashMap::new();
//! sizes.insert('i', 100);
//! sizes.insert('j', 200);
//! sizes.insert('k', 50);
//! sizes.insert('l', 100);
//!
//! // Optimize with greedy method (fast)
//! let greedy_result = optimize_code(&code, &sizes, &GreedyMethod::default());
//!
//! // Optimize with TreeSA (higher quality)
//! let treesa_result = optimize_code(&code, &sizes, &TreeSA::fast());
//!
//! // Check complexity of the optimized contraction
//! if let Some(ref optimized) = greedy_result {
//!     let complexity = contraction_complexity(optimized, &sizes, &code.ixs);
//!     println!("Time complexity: 2^{:.2}", complexity.tc);
//!     println!("Space complexity: 2^{:.2}", complexity.sc);
//! }
//! ```
//!
//! ## Complexity Metrics
//!
//! The library computes three complexity metrics:
//!
//! - **Time Complexity (tc)**: Log2 of the total FLOP count
//! - **Space Complexity (sc)**: Log2 of the maximum intermediate tensor size
//! - **Read-Write Complexity (rwc)**: Log2 of total I/O operations
//!
//! ## Algorithms
//!
//! ### GreedyMethod
//!
//! The greedy algorithm works by:
//! 1. Building a hypergraph where vertices are tensors and edges are indices
//! 2. Iteratively selecting the tensor pair with minimum contraction cost
//! 3. Contracting the pair and updating the graph
//!
//! Parameters:
//! - `alpha`: Balances output size vs input size reduction (0.0 to 1.0)
//! - `temperature`: Enables stochastic selection for escaping local minima
//!
//! ### TreeSA
//!
//! The simulated annealing algorithm:
//! 1. Initializes a contraction tree (using greedy or random)
//! 2. Applies local mutations to the tree structure
//! 3. Accepts or rejects changes based on the Metropolis criterion
//! 4. Runs multiple trials in parallel and returns the best result
//!
//! Parameters:
//! - `betas`: Inverse temperature schedule
//! - `ntrials`: Number of parallel trials
//! - `niters`: Iterations per temperature level
//! - `score`: Scoring function for evaluating solutions

pub mod complexity;
pub mod eincode;
pub mod expr_tree;
pub mod greedy;
pub mod incidence_list;
pub mod label;
pub mod score;
pub mod treesa;
pub mod utils;

// Re-export main types
pub use complexity::{
    eincode_complexity, flop, nested_complexity, nested_flop, peak_memory, sliced_complexity,
    ContractionComplexity,
};
pub use eincode::{log2_size_dict, uniform_size_dict, EinCode, NestedEinsum, SlicedEinsum};
pub use greedy::{optimize_greedy, ContractionTree, GreedyMethod, GreedyResult};
pub use label::Label;
pub use score::ScoreFunction;
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
}
