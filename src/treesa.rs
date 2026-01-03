//! TreeSA: Simulated Annealing optimizer for contraction order.
//!
//! This optimizer uses simulated annealing to search for optimal contraction
//! orders by applying local tree mutations and accepting changes based on
//! the Metropolis criterion.

use crate::eincode::{EinCode, NestedEinsum};
use crate::expr_tree::{
    apply_rule, contraction_output, rule_diff, tree_complexity, DecompositionType, ExprTree, Rule,
};
use crate::greedy::{optimize_greedy, GreedyMethod};
use crate::score::ScoreFunction;
use crate::Label;
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

/// Configuration for the TreeSA optimizer.
#[derive(Debug, Clone)]
pub struct TreeSA {
    /// Inverse temperature schedule (β values)
    pub betas: Vec<f64>,
    /// Number of independent trials to run
    pub ntrials: usize,
    /// Iterations per temperature level
    pub niters: usize,
    /// Initialization method
    pub initializer: Initializer,
    /// Scoring function for evaluating solutions
    pub score: ScoreFunction,
    /// Decomposition type (Tree or Path)
    pub decomposition_type: DecompositionType,
}

/// Method for initializing the contraction tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Initializer {
    /// Use greedy algorithm to initialize
    Greedy,
    /// Random tree initialization
    Random,
}

impl Default for Initializer {
    fn default() -> Self {
        Self::Greedy
    }
}

impl Default for TreeSA {
    fn default() -> Self {
        // Default schedule: β from 0.01 to 15.0 in steps of 0.05
        let betas: Vec<f64> = (1..=300).map(|i| 0.01 + 0.05 * i as f64).collect();
        Self {
            betas,
            ntrials: 10,
            niters: 50,
            initializer: Initializer::Greedy,
            score: ScoreFunction::default(),
            decomposition_type: DecompositionType::Tree,
        }
    }
}

impl TreeSA {
    /// Create a new TreeSA with custom parameters.
    pub fn new(
        betas: Vec<f64>,
        ntrials: usize,
        niters: usize,
        initializer: Initializer,
        score: ScoreFunction,
    ) -> Self {
        Self {
            betas,
            ntrials,
            niters,
            initializer,
            score,
            decomposition_type: DecompositionType::Tree,
        }
    }

    /// Create a fast TreeSA configuration with fewer iterations.
    pub fn fast() -> Self {
        let betas: Vec<f64> = (1..=100).map(|i| 0.01 + 0.15 * i as f64).collect();
        Self {
            betas,
            ntrials: 1,
            niters: 20,
            ..Default::default()
        }
    }

    /// Create a path decomposition variant (linear contraction order).
    pub fn path() -> Self {
        Self {
            initializer: Initializer::Random,
            decomposition_type: DecompositionType::Path,
            ..Default::default()
        }
    }

    /// Set the space complexity target.
    pub fn with_sc_target(mut self, sc_target: f64) -> Self {
        self.score.sc_target = sc_target;
        self
    }

    /// Set the number of trials.
    pub fn with_ntrials(mut self, ntrials: usize) -> Self {
        self.ntrials = ntrials;
        self
    }
}

/// Build a label-to-integer mapping for an EinCode.
fn build_label_map<L: Label>(code: &EinCode<L>) -> (HashMap<L, usize>, Vec<L>) {
    let labels = code.unique_labels();
    let map: HashMap<L, usize> = labels
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, l)| (l, i))
        .collect();
    (map, labels)
}

/// Convert EinCode input indices to integer indices.
fn convert_to_int_indices<L: Label>(
    ixs: &[Vec<L>],
    label_map: &HashMap<L, usize>,
) -> Vec<Vec<usize>> {
    ixs.iter()
        .map(|ix| ix.iter().map(|l| label_map[l]).collect())
        .collect()
}

/// Initialize an ExprTree from an EinCode using greedy method.
fn init_greedy<L: Label>(
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    label_map: &HashMap<L, usize>,
    int_ixs: &[Vec<usize>],
    int_iy: &[usize],
) -> Option<ExprTree> {
    let nested = optimize_greedy(code, size_dict, &GreedyMethod::default())?;
    nested_to_expr_tree(&nested, int_ixs, int_iy, label_map)
}

/// Convert a NestedEinsum to an ExprTree.
fn nested_to_expr_tree<L: Label>(
    nested: &NestedEinsum<L>,
    int_ixs: &[Vec<usize>],
    _int_iy: &[usize],
    label_map: &HashMap<L, usize>,
) -> Option<ExprTree> {
    match nested {
        NestedEinsum::Leaf { tensor_index } => {
            let out_dims = int_ixs.get(*tensor_index)?.clone();
            Some(ExprTree::leaf(out_dims, *tensor_index))
        }
        NestedEinsum::Node { args, eins } => {
            if args.len() != 2 {
                return None;
            }
            let left = nested_to_expr_tree(&args[0], int_ixs, &[], label_map)?;
            let right = nested_to_expr_tree(&args[1], int_ixs, &[], label_map)?;
            let out_dims: Vec<usize> = eins.iy.iter().map(|l| label_map[l]).collect();
            Some(ExprTree::node(left, right, out_dims))
        }
    }
}

/// Initialize a random ExprTree.
fn init_random<R: Rng>(
    int_ixs: &[Vec<usize>],
    int_iy: &[usize],
    decomp: DecompositionType,
    rng: &mut R,
) -> ExprTree {
    let n = int_ixs.len();
    if n == 0 {
        panic!("Cannot create tree with no tensors");
    }
    if n == 1 {
        return ExprTree::leaf(int_ixs[0].clone(), 0);
    }

    // Create leaf nodes
    let mut tensors: Vec<ExprTree> = int_ixs
        .iter()
        .enumerate()
        .map(|(i, ix)| ExprTree::leaf(ix.clone(), i))
        .collect();

    // Shuffle for randomness
    tensors.shuffle(rng);

    // Build tree by pairing tensors
    while tensors.len() > 1 {
        match decomp {
            DecompositionType::Tree => {
                // Random pairing
                let mut new_tensors = Vec::new();
                while tensors.len() >= 2 {
                    let left = tensors.pop().unwrap();
                    let right = tensors.pop().unwrap();
                    let out_dims = contraction_output(left.labels(), right.labels(), int_iy);
                    new_tensors.push(ExprTree::node(left, right, out_dims));
                }
                if let Some(remaining) = tensors.pop() {
                    new_tensors.push(remaining);
                }
                tensors = new_tensors;
            }
            DecompositionType::Path => {
                // Linear chain: always contract first two
                let left = tensors.remove(0);
                let right = tensors.remove(0);
                let out_dims = contraction_output(left.labels(), right.labels(), int_iy);
                tensors.insert(0, ExprTree::node(left, right, out_dims));
            }
        }
    }

    tensors.pop().unwrap()
}

/// Run simulated annealing on a single tree.
fn optimize_tree_sa<R: Rng>(
    mut tree: ExprTree,
    log2_sizes: &[f64],
    betas: &[f64],
    niters: usize,
    score: &ScoreFunction,
    decomp: DecompositionType,
    rng: &mut R,
) -> ExprTree {
    for &beta in betas {
        for _ in 0..niters {
            tree = optimize_subtree(tree, beta, log2_sizes, score, decomp, rng);
        }
    }
    tree
}

/// Recursively optimize a subtree using simulated annealing.
fn optimize_subtree<R: Rng>(
    tree: ExprTree,
    beta: f64,
    log2_sizes: &[f64],
    score: &ScoreFunction,
    decomp: DecompositionType,
    rng: &mut R,
) -> ExprTree {
    let rules = Rule::applicable_rules(&tree, decomp);

    if rules.is_empty() {
        return tree;
    }

    // Select a random rule
    let rule = rules[rng.random_range(0..rules.len())];

    // Compute the complexity change
    if let Some(diff) = rule_diff(&tree, rule, log2_sizes, score.rw_weight > 0.0) {
        // Compute energy change
        let dtc = diff.tc1 - diff.tc0;

        // Get current space complexity
        let (_, sc, _) = tree_complexity(&tree, log2_sizes);

        // Energy change calculation
        let sc_penalty = if sc.max(sc + diff.dsc) > score.sc_target {
            score.sc_weight
        } else {
            0.0
        };
        let d_energy = sc_penalty * diff.dsc + dtc;

        // Metropolis acceptance
        let accept = if d_energy <= 0.0 {
            true
        } else {
            rng.random::<f64>() < (-beta * d_energy).exp()
        };

        if accept {
            let new_tree = apply_rule(tree, rule, diff.new_labels);
            // Recursively optimize children
            return optimize_children(new_tree, beta, log2_sizes, score, decomp, rng);
        }
    }

    // If not accepted, still try to optimize children
    optimize_children(tree, beta, log2_sizes, score, decomp, rng)
}

/// Optimize children of a tree node.
fn optimize_children<R: Rng>(
    tree: ExprTree,
    beta: f64,
    log2_sizes: &[f64],
    score: &ScoreFunction,
    decomp: DecompositionType,
    rng: &mut R,
) -> ExprTree {
    match tree {
        ExprTree::Leaf(_) => tree,
        ExprTree::Node { left, right, info } => {
            let new_left = optimize_subtree(*left, beta, log2_sizes, score, decomp, rng);
            let new_right = optimize_subtree(*right, beta, log2_sizes, score, decomp, rng);
            ExprTree::Node {
                left: Box::new(new_left),
                right: Box::new(new_right),
                info,
            }
        }
    }
}

/// Convert an ExprTree back to a NestedEinsum.
fn expr_tree_to_nested<L: Label>(
    tree: &ExprTree,
    original_ixs: &[Vec<L>],
    inverse_map: &[L],
) -> NestedEinsum<L> {
    match tree {
        ExprTree::Leaf(info) => NestedEinsum::leaf(info.tensor_id.unwrap_or(0)),
        ExprTree::Node { left, right, info } => {
            let left_nested = expr_tree_to_nested(left, original_ixs, inverse_map);
            let right_nested = expr_tree_to_nested(right, original_ixs, inverse_map);

            // Convert output dims back to labels
            let iy: Vec<L> = info
                .out_dims
                .iter()
                .map(|&i| inverse_map[i].clone())
                .collect();

            // Get input labels from children
            let left_labels = get_child_labels(&left_nested, original_ixs);
            let right_labels = get_child_labels(&right_nested, original_ixs);

            let eins = EinCode::new(vec![left_labels, right_labels], iy);
            NestedEinsum::node(vec![left_nested, right_nested], eins)
        }
    }
}

fn get_child_labels<L: Label>(nested: &NestedEinsum<L>, original_ixs: &[Vec<L>]) -> Vec<L> {
    match nested {
        NestedEinsum::Leaf { tensor_index } => {
            original_ixs.get(*tensor_index).cloned().unwrap_or_default()
        }
        NestedEinsum::Node { eins, .. } => eins.iy.clone(),
    }
}

/// Optimize an EinCode using TreeSA.
pub fn optimize_treesa<L: Label>(
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    config: &TreeSA,
) -> Option<NestedEinsum<L>> {
    if code.num_tensors() == 0 {
        return None;
    }

    if code.num_tensors() == 1 {
        return Some(NestedEinsum::leaf(0));
    }

    // Build label mapping
    let (label_map, labels) = build_label_map(code);
    let log2_sizes: Vec<f64> = labels
        .iter()
        .map(|l| (size_dict[l] as f64).log2())
        .collect();
    let int_ixs = convert_to_int_indices(&code.ixs, &label_map);
    let int_iy: Vec<usize> = code.iy.iter().map(|l| label_map[l]).collect();

    // Run parallel trials
    let results: Vec<_> = (0..config.ntrials)
        .into_par_iter()
        .map(|trial_idx| {
            let mut rng = rand::rng();
            // Add some randomness based on trial index
            for _ in 0..trial_idx {
                let _: u64 = rng.random();
            }

            // Initialize tree
            let tree = match config.initializer {
                Initializer::Greedy => init_greedy(code, size_dict, &label_map, &int_ixs, &int_iy)
                    .unwrap_or_else(|| {
                        init_random(&int_ixs, &int_iy, config.decomposition_type, &mut rng)
                    }),
                Initializer::Random => {
                    init_random(&int_ixs, &int_iy, config.decomposition_type, &mut rng)
                }
            };

            // Optimize
            let optimized = optimize_tree_sa(
                tree,
                &log2_sizes,
                &config.betas,
                config.niters,
                &config.score,
                config.decomposition_type,
                &mut rng,
            );

            // Compute final complexity
            let (tc, sc, rw) = tree_complexity(&optimized, &log2_sizes);
            let score = config.score.evaluate(tc, sc, rw);

            (optimized, score, tc, sc, rw)
        })
        .collect();

    // Find best result
    let (best_tree, _, _, _, _) = results
        .into_iter()
        .min_by(|(_, s1, _, _, _), (_, s2, _, _, _)| s1.partial_cmp(s2).unwrap())?;

    // Convert back to NestedEinsum
    Some(expr_tree_to_nested(&best_tree, &code.ixs, &labels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_treesa_default() {
        let config = TreeSA::default();
        assert_eq!(config.ntrials, 10);
        assert_eq!(config.niters, 50);
        assert!(!config.betas.is_empty());
    }

    #[test]
    fn test_treesa_fast() {
        let config = TreeSA::fast();
        assert_eq!(config.ntrials, 1);
        assert_eq!(config.niters, 20);
    }

    #[test]
    fn test_optimize_treesa_simple() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 4);

        let config = TreeSA::fast();
        let result = optimize_treesa(&code, &size_dict, &config);

        assert!(result.is_some());
        let nested = result.unwrap();
        assert!(nested.is_binary());
        assert_eq!(nested.leaf_count(), 2);
    }

    #[test]
    fn test_optimize_treesa_chain() {
        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
            vec!['i', 'l'],
        );
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 8);
        size_dict.insert('l', 4);

        let config = TreeSA::fast();
        let result = optimize_treesa(&code, &size_dict, &config);

        assert!(result.is_some());
        let nested = result.unwrap();
        assert!(nested.is_binary());
        assert_eq!(nested.leaf_count(), 3);
    }

    #[test]
    fn test_init_random() {
        let int_ixs = vec![vec![0, 1], vec![1, 2], vec![2, 3]];
        let int_iy = vec![0, 3];
        let mut rng = rand::rng();

        let tree = init_random(&int_ixs, &int_iy, DecompositionType::Tree, &mut rng);
        assert_eq!(tree.leaf_count(), 3);
    }

    #[test]
    fn test_build_label_map() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let (map, labels) = build_label_map(&code);

        assert_eq!(labels.len(), 3);
        assert!(map.contains_key(&'i'));
        assert!(map.contains_key(&'j'));
        assert!(map.contains_key(&'k'));
    }
}
