//! TreeSA: Simulated Annealing optimizer for contraction order.
//!
//! This optimizer uses simulated annealing to search for optimal contraction
//! orders by applying local tree mutations and accepting changes based on
//! the Metropolis criterion.

use crate::eincode::{EinCode, NestedEinsum};
use crate::expr_tree::{
    apply_rule_mut, tree_complexity, DecompositionType, ExprTree, Rule, ScratchSpace,
};
use crate::greedy::{optimize_greedy, GreedyMethod};
use crate::score::ScoreFunction;
use crate::utils::fast_log2sumexp2;
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Initializer {
    /// Use greedy algorithm to initialize
    #[default]
    Greedy,
    /// Random tree initialization
    Random,
}

impl Default for TreeSA {
    fn default() -> Self {
        // Default schedule: β from 0.01 to ~15.0 in steps of 0.05 (matching Julia's 0.01:0.05:15)
        let betas: Vec<f64> = (0..300).map(|i| 0.01 + 0.05 * i as f64).collect();
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

    /// Set the number of iterations per temperature level.
    pub fn with_niters(mut self, niters: usize) -> Self {
        self.niters = niters;
        self
    }

    /// Set the inverse temperature schedule.
    pub fn with_betas(mut self, betas: Vec<f64>) -> Self {
        self.betas = betas;
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
/// Matches Julia's `_exprtree` function exactly.
fn nested_to_expr_tree<L: Label>(
    nested: &NestedEinsum<L>,
    _int_ixs: &[Vec<usize>],
    _int_iy: &[usize],
    label_map: &HashMap<L, usize>,
) -> Option<ExprTree> {
    // Julia: _exprtree(code::NestedEinsum, labels)
    // For leaf nodes, Julia uses the parent's einsum input indices.
    // For non-leaf nodes, Julia recursively processes children.
    // We need to handle this differently - process at the Node level.
    nested_to_expr_tree_inner(nested, label_map)
}

/// Inner conversion function that matches Julia's _exprtree exactly.
/// Julia processes leaves using the parent's einsum.ixs[i], not original tensor indices.
fn nested_to_expr_tree_inner<L: Label>(
    nested: &NestedEinsum<L>,
    label_map: &HashMap<L, usize>,
) -> Option<ExprTree> {
    match nested {
        NestedEinsum::Leaf { .. } => {
            // This case shouldn't happen at top level for binary trees
            // Julia asserts length(code.args) == 2 at entry
            None
        }
        NestedEinsum::Node { args, eins } => {
            if args.len() != 2 {
                return None;
            }

            // Julia: map(enumerate(code.args)) do (i,arg)
            //   if isleaf(arg)
            //     ExprTree(ExprInfo(getindex.(Ref(labels), getixsv(code.eins)[i]), arg.tensorindex))
            //   else
            //     _exprtree(arg, labels)
            //   end
            // end

            // Process left child (index 0)
            let left = match &args[0] {
                NestedEinsum::Leaf { tensor_index } => {
                    // Julia: getixsv(code.eins)[1] = eins.ixs[0]
                    let out_dims: Vec<usize> = eins.ixs[0].iter().map(|l| label_map[l]).collect();
                    ExprTree::leaf(out_dims, *tensor_index)
                }
                NestedEinsum::Node { .. } => nested_to_expr_tree_inner(&args[0], label_map)?,
            };

            // Process right child (index 1)
            let right = match &args[1] {
                NestedEinsum::Leaf { tensor_index } => {
                    // Julia: getixsv(code.eins)[2] = eins.ixs[1]
                    let out_dims: Vec<usize> = eins.ixs[1].iter().map(|l| label_map[l]).collect();
                    ExprTree::leaf(out_dims, *tensor_index)
                }
                NestedEinsum::Node { .. } => nested_to_expr_tree_inner(&args[1], label_map)?,
            };

            // Julia: ExprInfo(Int[labels[i] for i=getiyv(code.eins)])
            let out_dims: Vec<usize> = eins.iy.iter().map(|l| label_map[l]).collect();
            Some(ExprTree::node(left, right, out_dims))
        }
    }
}

/// Initialize a random ExprTree using Julia's recursive partitioning algorithm.
///
/// This matches Julia's `random_exprtree` which uses outercount/allcount tracking
/// to correctly compute intermediate outputs.
fn init_random<R: Rng>(
    int_ixs: &[Vec<usize>],
    int_iy: &[usize],
    nedge: usize,
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

    // Initialize counts like Julia
    let mut outercount = vec![0usize; nedge];
    let mut allcount = vec![0usize; nedge];

    // Count output indices
    for &l in int_iy {
        outercount[l] += 1;
        allcount[l] += 1;
    }

    // Count all indices in inputs
    for ix in int_ixs {
        for &l in ix {
            allcount[l] += 1;
        }
    }

    let xindices: Vec<usize> = (0..n).collect();
    init_random_recursive(
        int_ixs, &xindices, outercount, &allcount, nedge, decomp, rng,
    )
}

/// Recursive helper for random tree initialization (matches Julia's _random_exprtree).
fn init_random_recursive<R: Rng>(
    ixs: &[Vec<usize>],
    xindices: &[usize],
    outercount: Vec<usize>,
    allcount: &[usize],
    nedge: usize,
    decomp: DecompositionType,
    rng: &mut R,
) -> ExprTree {
    let n = ixs.len();
    if n == 1 {
        return ExprTree::leaf(ixs[0].clone(), xindices[0]);
    }

    // Create partition mask
    let mask: Vec<bool> = match decomp {
        DecompositionType::Tree => {
            let mut mask: Vec<bool> = (0..n).map(|_| rng.random()).collect();
            // Prevent invalid partitions (all true or all false)
            if mask.iter().all(|&b| b) || mask.iter().all(|&b| !b) {
                let i = rng.random_range(0..n);
                mask[i] = !mask[i];
            }
            mask
        }
        DecompositionType::Path => {
            // For path decomposition, last tensor goes to right tree
            let mut mask = vec![true; n];
            mask[n - 1] = false;
            mask
        }
    };

    // Compute output dimensions: indices where outercount != allcount AND outercount != 0
    // This matches Julia's: Int[i for i=1:length(outercount) if outercount[i]!=allcount[i] && outercount[i]!=0]
    let out_dims: Vec<usize> = (0..nedge)
        .filter(|&i| outercount[i] != allcount[i] && outercount[i] != 0)
        .collect();

    // Split inputs and update counts for each subtree
    let mut outercount1 = outercount.clone();
    let mut outercount2 = outercount.clone();

    // Julia: for i=1:n; counter = mask[i] ? outercount2 : outercount1; for l in ixs[i]; counter[l] += 1; end; end
    for (i, ix) in ixs.iter().enumerate() {
        let counter = if mask[i] {
            &mut outercount2
        } else {
            &mut outercount1
        };
        for &l in ix {
            counter[l] += 1;
        }
    }

    // Partition ixs and xindices based on mask
    let (ixs_left, xindices_left): (Vec<_>, Vec<_>) = ixs
        .iter()
        .zip(xindices.iter())
        .zip(mask.iter())
        .filter(|((_, _), &m)| m)
        .map(|((ix, &xi), _)| (ix.clone(), xi))
        .unzip();

    let (ixs_right, xindices_right): (Vec<_>, Vec<_>) = ixs
        .iter()
        .zip(xindices.iter())
        .zip(mask.iter())
        .filter(|((_, _), &m)| !m)
        .map(|((ix, &xi), _)| (ix.clone(), xi))
        .unzip();

    let left = init_random_recursive(
        &ixs_left,
        &xindices_left,
        outercount1,
        allcount,
        nedge,
        decomp,
        rng,
    );
    let right = init_random_recursive(
        &ixs_right,
        &xindices_right,
        outercount2,
        allcount,
        nedge,
        decomp,
        rng,
    );

    ExprTree::node(left, right, out_dims)
}

/// Run simulated annealing on a single tree.
/// Each iteration sweeps through all nodes in the tree, attempting mutations.
/// Matches Julia's `optimize_tree_sa!` exactly, with in-place mutation for performance.
#[allow(clippy::too_many_arguments)]
fn optimize_tree_sa<R: Rng>(
    mut tree: ExprTree,
    log2_sizes: &[f64],
    betas: &[f64],
    niters: usize,
    score: &ScoreFunction,
    decomp: DecompositionType,
    rng: &mut R,
    nedge: usize,
) -> ExprTree {
    // Compute log2_rw_weight once (matches Julia: log2rw_weight = log2(score.rw_weight))
    let log2_rw_weight = if score.rw_weight > 0.0 {
        score.rw_weight.log2()
    } else {
        f64::NEG_INFINITY
    };

    // Create scratch space for large graphs (bitset-based O(1) lookups)
    let mut scratch = ScratchSpace::new(nedge);

    for &beta in betas {
        for _ in 0..niters {
            // Single sweep through all nodes (in-place mutation)
            optimize_subtree_mut(
                &mut tree,
                beta,
                log2_sizes,
                score.sc_target,
                score.sc_weight,
                log2_rw_weight,
                decomp,
                rng,
                &mut scratch,
            );
        }
    }
    tree
}

/// Optimize a subtree recursively using simulated annealing (in-place mutation).
/// Matches Julia's `optimize_subtree!` exactly:
/// 1. Try mutation at current node first
/// 2. Then recurse to children (post-order)
#[inline]
#[allow(clippy::too_many_arguments)]
fn optimize_subtree_mut<R: Rng>(
    tree: &mut ExprTree,
    beta: f64,
    log2_sizes: &[f64],
    sc_target: f64,
    sc_weight: f64,
    log2_rw_weight: f64,
    decomp: DecompositionType,
    rng: &mut R,
    scratch: &mut ScratchSpace,
) {
    let rules = Rule::applicable_rules(tree, decomp);

    if rules.is_empty() {
        return;
    }

    // Select a random rule (matches Julia: rule = rand(rst))
    let rule = rules[rng.random_range(0..rules.len())];

    // Check if we should optimize rw (matches Julia: optimize_rw = log2rw_weight != -Inf)
    let optimize_rw = log2_rw_weight > f64::NEG_INFINITY;

    // Compute the complexity change using bitset-optimized scratch space
    if let Some(diff) = scratch.rule_diff(tree, rule, log2_sizes, optimize_rw) {
        // Compute dtc (matches Julia exactly)
        let dtc = if optimize_rw {
            fast_log2sumexp2(diff.tc1, log2_rw_weight + diff.rw1)
                - fast_log2sumexp2(diff.tc0, log2_rw_weight + diff.rw0)
        } else {
            diff.tc1 - diff.tc0
        };

        // Compute local sc at this node
        let sc = local_sc(tree, rule, log2_sizes);

        // Energy change (matches Julia exactly)
        let sc_after = sc.max(sc + diff.dsc);
        let d_energy = if sc_after > sc_target {
            sc_weight * diff.dsc + dtc
        } else {
            dtc
        };

        // Metropolis acceptance (matches Julia: rand() < exp(-β*dE))
        let accept = rng.random::<f64>() < (-beta * d_energy).exp();

        if accept {
            apply_rule_mut(tree, rule, diff.new_labels);
        }
    }

    // Recurse to children AFTER trying mutation (matches Julia: for subtree in siblings(tree))
    if let ExprTree::Node { left, right, .. } = tree {
        optimize_subtree_mut(
            left,
            beta,
            log2_sizes,
            sc_target,
            sc_weight,
            log2_rw_weight,
            decomp,
            rng,
            scratch,
        );
        optimize_subtree_mut(
            right,
            beta,
            log2_sizes,
            sc_target,
            sc_weight,
            log2_rw_weight,
            decomp,
            rng,
            scratch,
        );
    }
}

/// Compute local space complexity at a node for the given rule.
/// Matches Julia's `_sc(tree, rule, log2_sizes)`:
/// - For Rule1/Rule2: max(sc(tree), sc(tree.left))
/// - For Rule3/Rule4/Rule5: max(sc(tree), sc(tree.right))
#[inline]
fn local_sc(tree: &ExprTree, rule: Rule, log2_sizes: &[f64]) -> f64 {
    match tree {
        ExprTree::Leaf(info) => node_sc(&info.out_dims, log2_sizes),
        ExprTree::Node { left, right, info } => {
            let tree_sc = node_sc(&info.out_dims, log2_sizes);
            let child_sc = match rule {
                Rule::Rule1 | Rule::Rule2 => node_sc(left.labels(), log2_sizes),
                Rule::Rule3 | Rule::Rule4 | Rule::Rule5 => node_sc(right.labels(), log2_sizes),
            };
            tree_sc.max(child_sc)
        }
    }
}

/// Compute space complexity for a single node's output dimensions.
/// Matches Julia's `__sc(tree, log2_sizes)`.
#[inline]
fn node_sc(out_dims: &[usize], log2_sizes: &[f64]) -> f64 {
    if out_dims.is_empty() {
        0.0
    } else {
        out_dims.iter().map(|&l| log2_sizes[l]).sum()
    }
}

/// Convert an ExprTree back to a NestedEinsum.
/// The `openedges` parameter specifies the final output indices for the root node.
fn expr_tree_to_nested<L: Label>(
    tree: &ExprTree,
    original_ixs: &[Vec<L>],
    inverse_map: &[L],
    openedges: &[L],
    level: usize,
) -> NestedEinsum<L> {
    match tree {
        ExprTree::Leaf(info) => NestedEinsum::leaf(info.tensor_id.unwrap_or(0)),
        ExprTree::Node { left, right, info } => {
            let left_nested =
                expr_tree_to_nested(left, original_ixs, inverse_map, openedges, level + 1);
            let right_nested =
                expr_tree_to_nested(right, original_ixs, inverse_map, openedges, level + 1);

            // At level 0 (root), use openedges; otherwise use computed output
            let iy: Vec<L> = if level == 0 {
                openedges.to_vec()
            } else {
                info.out_dims
                    .iter()
                    .map(|&i| inverse_map[i].clone())
                    .collect()
            };

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
    let nedge = labels.len(); // Number of unique edge labels
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
            // Use thread-local RNG seeded with trial index for reproducibility
            use rand::SeedableRng;
            let mut rng = rand::rngs::SmallRng::seed_from_u64(trial_idx as u64 + 42);

            // Initialize tree
            let tree = match config.initializer {
                Initializer::Greedy => init_greedy(code, size_dict, &label_map, &int_ixs, &int_iy)
                    .unwrap_or_else(|| {
                        init_random(
                            &int_ixs,
                            &int_iy,
                            nedge,
                            config.decomposition_type,
                            &mut rng,
                        )
                    }),
                Initializer::Random => init_random(
                    &int_ixs,
                    &int_iy,
                    nedge,
                    config.decomposition_type,
                    &mut rng,
                ),
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
                nedge,
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

    // Convert back to NestedEinsum with openedges for correct root output (issue #13)
    Some(expr_tree_to_nested(
        &best_tree, &code.ixs, &labels, &code.iy, 0,
    ))
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
        let nedge = 4; // Labels 0, 1, 2, 3
        let mut rng = rand::rng();

        let tree = init_random(&int_ixs, &int_iy, nedge, DecompositionType::Tree, &mut rng);
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

    #[test]
    fn test_treesa_with_random_init() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 4);

        let mut config = TreeSA::fast();
        config.initializer = Initializer::Random;

        let result = optimize_treesa(&code, &size_dict, &config);
        assert!(result.is_some());
    }

    #[test]
    fn test_treesa_path_decomposition() {
        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
            vec!['i', 'l'],
        );
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 8);
        size_dict.insert('l', 4);

        let mut config = TreeSA::fast();
        config.decomposition_type = DecompositionType::Path;

        let result = optimize_treesa(&code, &size_dict, &config);
        assert!(result.is_some());
    }

    #[test]
    fn test_treesa_with_sc_target() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 4);

        let mut config = TreeSA::fast();
        config.score.sc_target = 10.0;
        config.score.sc_weight = 1.0;

        let result = optimize_treesa(&code, &size_dict, &config);
        assert!(result.is_some());
    }

    #[test]
    fn test_treesa_with_rw_weight() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 4);

        let mut config = TreeSA::fast();
        config.score.rw_weight = 0.5;

        let result = optimize_treesa(&code, &size_dict, &config);
        assert!(result.is_some());
    }

    #[test]
    fn test_treesa_single_tensor() {
        let code = EinCode::new(vec![vec!['i', 'j']], vec!['i', 'j']);
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);

        let config = TreeSA::fast();
        let result = optimize_treesa(&code, &size_dict, &config);
        assert!(result.is_some());
        assert_eq!(result.unwrap().leaf_count(), 1);
    }

    #[test]
    fn test_score_function() {
        let score = ScoreFunction {
            tc_weight: 1.0,
            sc_target: 10.0,
            sc_weight: 2.0,
            rw_weight: 0.5,
        };

        assert_eq!(score.sc_target, 10.0);
        assert_eq!(score.sc_weight, 2.0);
        assert_eq!(score.rw_weight, 0.5);
    }

    #[test]
    fn test_init_random_path_decomp() {
        let int_ixs = vec![vec![0, 1], vec![1, 2], vec![2, 3]];
        let int_iy = vec![0, 3];
        let nedge = 4; // Labels 0, 1, 2, 3
        let mut rng = rand::rng();

        let tree = init_random(&int_ixs, &int_iy, nedge, DecompositionType::Path, &mut rng);
        assert_eq!(tree.leaf_count(), 3);
    }

    #[test]
    fn test_treesa_with_betas() {
        let config = TreeSA::default().with_betas(vec![0.1, 0.5, 1.0]);
        assert_eq!(config.betas, vec![0.1, 0.5, 1.0]);
    }

    #[test]
    fn test_treesa_with_ntrials() {
        let config = TreeSA::default().with_ntrials(5);
        assert_eq!(config.ntrials, 5);
    }

    #[test]
    fn test_treesa_with_niters() {
        let config = TreeSA::default().with_niters(100);
        assert_eq!(config.niters, 100);
    }

    #[test]
    fn test_treesa_with_sc_target_builder() {
        let config = TreeSA::default().with_sc_target(15.0);
        assert_eq!(config.score.sc_target, 15.0);
    }

    #[test]
    fn test_treesa_path() {
        let config = TreeSA::path();
        assert_eq!(config.decomposition_type, DecompositionType::Path);
        assert_eq!(config.initializer, Initializer::Random);
    }

    #[test]
    fn test_treesa_new() {
        let score = ScoreFunction::new(1.0, 2.0, 0.5, 10.0);
        let config = TreeSA::new(vec![0.1, 0.2, 0.3], 5, 10, Initializer::Random, score);
        assert_eq!(config.betas, vec![0.1, 0.2, 0.3]);
        assert_eq!(config.ntrials, 5);
        assert_eq!(config.niters, 10);
        assert_eq!(config.initializer, Initializer::Random);
    }

    #[test]
    fn test_convert_to_int_indices() {
        let ixs = vec![vec!['i', 'j'], vec!['j', 'k']];
        let mut label_map = HashMap::new();
        label_map.insert('i', 0);
        label_map.insert('j', 1);
        label_map.insert('k', 2);

        let int_ixs = convert_to_int_indices(&ixs, &label_map);
        assert_eq!(int_ixs, vec![vec![0, 1], vec![1, 2]]);
    }

    #[test]
    fn test_init_random_single_tensor() {
        let int_ixs = vec![vec![0, 1]];
        let int_iy = vec![0, 1];
        let nedge = 2; // Labels 0, 1
        let mut rng = rand::rng();

        let tree = init_random(&int_ixs, &int_iy, nedge, DecompositionType::Tree, &mut rng);
        assert!(tree.is_leaf());
        assert_eq!(tree.leaf_count(), 1);
    }

    #[test]
    fn test_init_random_odd_number() {
        // Test with odd number of tensors for tree decomposition
        let int_ixs = vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 4], vec![4, 0]];
        let int_iy = vec![];
        let nedge = 5; // Labels 0, 1, 2, 3, 4
        let mut rng = rand::rng();

        let tree = init_random(&int_ixs, &int_iy, nedge, DecompositionType::Tree, &mut rng);
        assert_eq!(tree.leaf_count(), 5);
    }

    #[test]
    fn test_optimize_treesa_empty() {
        let code: EinCode<char> = EinCode::new(vec![], vec![]);
        let size_dict: HashMap<char, usize> = HashMap::new();

        let config = TreeSA::fast();
        let result = optimize_treesa(&code, &size_dict, &config);
        assert!(result.is_none());
    }

    #[test]
    fn test_optimize_treesa_many_tensors() {
        // Test with more tensors
        let code = EinCode::new(
            vec![
                vec!['a', 'b'],
                vec!['b', 'c'],
                vec!['c', 'd'],
                vec!['d', 'e'],
            ],
            vec!['a', 'e'],
        );
        let mut size_dict = HashMap::new();
        size_dict.insert('a', 4);
        size_dict.insert('b', 8);
        size_dict.insert('c', 8);
        size_dict.insert('d', 8);
        size_dict.insert('e', 4);

        let config = TreeSA::fast();
        let result = optimize_treesa(&code, &size_dict, &config);

        assert!(result.is_some());
        let nested = result.unwrap();
        assert_eq!(nested.leaf_count(), 4);
    }

    #[test]
    fn test_optimize_treesa_path_multiple_tensors() {
        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
            vec!['i', 'l'],
        );
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 8);
        size_dict.insert('l', 4);

        let config = TreeSA::path()
            .with_ntrials(1)
            .with_niters(5)
            .with_betas(vec![0.1, 0.5]);
        let result = optimize_treesa(&code, &size_dict, &config);

        assert!(result.is_some());
    }

    #[test]
    fn test_initializer_default() {
        let init = Initializer::default();
        assert_eq!(init, Initializer::Greedy);
    }

    #[test]
    fn test_decomposition_type_default() {
        let decomp = DecompositionType::default();
        assert_eq!(decomp, DecompositionType::Tree);
    }
}
