//! TreeSA: Simulated Annealing optimizer for contraction order.
//!
//! This optimizer uses simulated annealing to search for optimal contraction
//! orders by applying local tree mutations and accepting changes based on
//! the Metropolis criterion.

use crate::eincode::{EinCode, NestedEinsum};
use crate::expr_tree::{apply_rule_mut, DecompositionType, ExprTree, Rule, ScratchSpace};
use crate::greedy::{optimize_greedy, GreedyMethod};
use crate::preprocess::{simplify, splice};
use crate::score::ScoreFunction;
use crate::utils::fast_log2sumexp2;
use crate::waist_surgery::refine_capped;
use crate::Label;
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

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
    /// Run the structural simplification front-end before annealing
    /// (simplify → optimize the reduced network → splice back). Deterministic
    /// and exactness-preserving; see [`crate::preprocess`].
    ///
    /// [`optimize_treesa`] auto-skips this step (treating it as `false`)
    /// whenever `decomposition_type` is [`DecompositionType::Path`], even if
    /// this field is set to `true`: splice does not preserve the
    /// path-decomposition guarantee. See [`TreeSA::path`].
    pub preprocess: bool,
    /// Deterministic cap on the number of waist-surgery iterations run as a
    /// post-pass on the selected tree; `0` disables surgery entirely (the
    /// default). A positive value runs at most that many surgery iterations
    /// after the pipeline: the result is never worse than with surgery off,
    /// and more iterations can only be equal or better. See
    /// [`crate::waist_surgery`].
    ///
    /// # Determinism
    ///
    /// Unlike the low-level [`crate::waist_surgery::refine`]/[`refine_capped`]
    /// wall-clock APIs, `optimize_treesa` binds surgery to no deadline —
    /// internal RNG seeds are fixed throughout. The whole `TreeSA` API is
    /// therefore fully reproducible across machines for any fixed config,
    /// including configs with `surgery_iters > 0`.
    ///
    /// [`refine_capped`]: crate::waist_surgery::refine_capped
    pub surgery_iters: u64,
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
            preprocess: true,
            surgery_iters: 0,
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
            preprocess: true,
            surgery_iters: 0,
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
    ///
    /// Sets `preprocess: false`: [`crate::preprocess::splice`] is
    /// decomposition-agnostic — it substitutes each reduced-network leaf with
    /// whatever binary subtree `simplify` merged for it, which is not
    /// path-shaped in general. Running the front-end here can give the
    /// spliced tree a node with two non-leaf children, breaking this preset's
    /// documented "linear contraction order" guarantee
    /// (see [`NestedEinsum::is_path_decomposition`]).
    pub fn path() -> Self {
        Self {
            initializer: Initializer::Random,
            decomposition_type: DecompositionType::Path,
            preprocess: false,
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

    /// Enable or disable the structural simplification front-end.
    ///
    /// # Example
    ///
    /// ```
    /// use omeco::{optimize_treesa, EinCode, TreeSA};
    /// use std::collections::HashMap;
    ///
    /// let code = EinCode::new(
    ///     vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
    ///     vec!['i', 'l'],
    /// );
    /// let sizes: HashMap<char, usize> = [('i', 2), ('j', 2), ('k', 2), ('l', 2)].into();
    /// let config = TreeSA::fast().with_preprocess(false);
    /// let tree = optimize_treesa(&code, &sizes, &config).unwrap();
    /// assert_eq!(tree.leaf_count(), 3);
    /// ```
    pub fn with_preprocess(mut self, preprocess: bool) -> Self {
        self.preprocess = preprocess;
        self
    }

    /// Set the deterministic waist-surgery iteration cap (0 disables).
    ///
    /// # Example
    ///
    /// ```
    /// use omeco::{optimize_treesa, EinCode, TreeSA};
    /// use std::collections::HashMap;
    ///
    /// let code = EinCode::new(
    ///     vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
    ///     vec!['i', 'l'],
    /// );
    /// let sizes: HashMap<char, usize> = [('i', 2), ('j', 2), ('k', 2), ('l', 2)].into();
    /// let config = TreeSA::fast().with_surgery_iters(5);
    /// let tree = optimize_treesa(&code, &sizes, &config).unwrap();
    /// assert_eq!(tree.leaf_count(), 3);
    /// ```
    pub fn with_surgery_iters(mut self, max_iters: u64) -> Self {
        self.surgery_iters = max_iters;
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
        NestedEinsum::Node { args, eins } => match args.as_slice() {
            [child] => {
                // ExprTree is binary-only. Fuse a unary trace/reduction into
                // the child's materialized output interface, retaining the
                // original leaf interface so complexity remains exact.
                let input = eins.ixs.first()?;
                let input_dims: Vec<usize> = input.iter().map(|l| label_map[l]).collect();
                let out_dims: Vec<usize> = eins.iy.iter().map(|l| label_map[l]).collect();
                let mut tree = match child {
                    NestedEinsum::Leaf { tensor_index } => {
                        ExprTree::leaf(input_dims.clone(), *tensor_index)
                    }
                    NestedEinsum::Node { .. } => nested_to_expr_tree_inner(child, label_map)?,
                };
                if let ExprTree::Leaf(info) = &mut tree {
                    if info.leaf_input_dims.is_none() && info.out_dims != out_dims {
                        info.leaf_input_dims = Some(info.out_dims.clone());
                    }
                }
                tree.info_mut().out_dims = out_dims;
                tree.info_mut().cached = None;
                Some(tree)
            }
            [left_arg, right_arg] => {
                let child = |arg: &NestedEinsum<L>, side: usize| -> Option<ExprTree> {
                    match arg {
                        NestedEinsum::Leaf { tensor_index } => {
                            let input = &eins.ixs[side];
                            let out_dims: Vec<usize> = input.iter().map(|l| label_map[l]).collect();
                            Some(ExprTree::leaf(out_dims, *tensor_index))
                        }
                        NestedEinsum::Node { .. } => nested_to_expr_tree_inner(arg, label_map),
                    }
                };
                let left = child(left_arg, 0)?;
                let right = child(right_arg, 1)?;
                let out_dims: Vec<usize> = eins.iy.iter().map(|l| label_map[l]).collect();
                Some(ExprTree::node(left, right, out_dims))
            }
            _ => None,
        },
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
///
/// Every internal node's output index set is derived from the tree topology by
/// **outside-occurrence counting** — a label is an output of a node iff it occurs
/// in a tensor outside the node's subtree or is a final-output (open) label —
/// rather than trusting the `out_dims` cached on the tree. The cached `out_dims`
/// can disagree with the topology for labels that appear in more than two tensors
/// (hypergraph edges), which would make [`crate::contraction_complexity`] and any
/// topology-derived scorer report different costs for the same tree. Deriving the
/// outputs from occurrence counts keeps the emitted `eins` bodies consistent with
/// the topology in all cases. `inverse_map` restores fused unary leaf interfaces
/// to their original label type. The `openedges` parameter fixes the root output
/// (issue #13).
fn expr_tree_to_nested<L: Label>(
    tree: &ExprTree,
    original_ixs: &[Vec<L>],
    inverse_map: &[L],
    openedges: &[L],
) -> NestedEinsum<L> {
    // Global occurrence count of every label across all original tensors.
    let mut global_count: HashMap<L, usize> = HashMap::new();
    for ix in original_ixs {
        for l in ix {
            *global_count.entry(l.clone()).or_insert(0) += 1;
        }
    }
    let open_set: HashSet<L> = openedges.iter().cloned().collect();
    let ixs = original_ixs;
    let inverse = inverse_map;
    let globals = &global_count;
    expr_tree_to_nested_counted(tree, ixs, inverse, &open_set, globals, openedges, 0).0
}

/// Recursive worker for [`expr_tree_to_nested`]. Returns the converted subtree,
/// the multiset of label occurrence counts within the subtree, and the subtree's
/// output labels (the labels it exposes to its parent).
fn expr_tree_to_nested_counted<L: Label>(
    tree: &ExprTree,
    original_ixs: &[Vec<L>],
    inverse_map: &[L],
    open_set: &HashSet<L>,
    global_count: &HashMap<L, usize>,
    openedges: &[L],
    level: usize,
) -> (NestedEinsum<L>, HashMap<L, usize>, Vec<L>) {
    match tree {
        ExprTree::Leaf(info) => {
            let tid = info.tensor_id.unwrap_or(0);
            let input_labels = original_ixs.get(tid).cloned().unwrap_or_default();
            let output_ids = &info.out_dims;
            let output_labels = output_ids
                .iter()
                .map(|&id| inverse_map[id].clone())
                .collect::<Vec<L>>();
            let mut within: HashMap<L, usize> = HashMap::new();
            for l in &input_labels {
                *within.entry(l.clone()).or_insert(0) += 1;
            }
            let leaf = NestedEinsum::leaf(tid);
            let nested = if input_labels == output_labels {
                leaf
            } else {
                NestedEinsum::node(
                    vec![leaf],
                    EinCode::new(vec![input_labels], output_labels.clone()),
                )
            };
            (nested, within, output_labels)
        }
        ExprTree::Node { left, right, .. } => {
            let ixs = original_ixs;
            let inverse = inverse_map;
            let globals = global_count;
            let next_level = level + 1;
            let (left_nested, left_within, left_out) = expr_tree_to_nested_counted(
                left, ixs, inverse, open_set, globals, openedges, next_level,
            );
            let (right_nested, right_within, right_out) = expr_tree_to_nested_counted(
                right, ixs, inverse, open_set, globals, openedges, next_level,
            );

            // Merge the two subtrees' occurrence counts.
            let mut within = left_within;
            for (l, c) in right_within {
                *within.entry(l).or_insert(0) += c;
            }

            // At the root use openedges verbatim (issue #13). Otherwise a label is
            // an output iff it still occurs outside this subtree (within < global)
            // or is an open/output label. Iterate children outputs for a stable,
            // Ord-free ordering.
            let iy: Vec<L> = if level == 0 {
                openedges.to_vec()
            } else {
                let mut out: Vec<L> = Vec::new();
                for l in left_out.iter().chain(right_out.iter()) {
                    if !out.contains(l) {
                        let w = within.get(l).copied().unwrap_or(0);
                        let g = global_count.get(l).copied().unwrap_or(0);
                        if open_set.contains(l) || w < g {
                            out.push(l.clone());
                        }
                    }
                }
                out
            };

            let eins = EinCode::new(vec![left_out, right_out], iy.clone());
            (
                NestedEinsum::node(vec![left_nested, right_nested], eins),
                within,
                iy,
            )
        }
    }
}

/// Return the labels a converted child exposes to its parent: a leaf exposes its
/// original tensor indices, a node exposes its `eins` output. No longer used by
/// the conversion itself (which now returns child outputs directly), but retained
/// for its existing unit tests.
#[cfg(test)]
fn get_child_labels<L: Label>(nested: &NestedEinsum<L>, original_ixs: &[Vec<L>]) -> Vec<L> {
    match nested {
        NestedEinsum::Leaf { tensor_index } => {
            original_ixs.get(*tensor_index).cloned().unwrap_or_default()
        }
        NestedEinsum::Node { eins, .. } => eins.iy.clone(),
    }
}

/// Optimize an EinCode using TreeSA.
///
/// By default this runs the full pipeline: structural simplification
/// ([`crate::preprocess::simplify`]), the annealing trial loop on the reduced
/// network, and splice-back — controlled by [`TreeSA::preprocess`]. A
/// positive [`TreeSA::surgery_iters`] additionally refines the result with
/// [`crate::waist_surgery::refine_capped`] (never worse than surgery off;
/// more iterations are equal or better; fully deterministic, since the cap
/// binds iteration count rather than wall-clock time).
///
/// [`TreeSA::preprocess`] is automatically treated as disabled whenever
/// [`TreeSA::decomposition_type`] is [`DecompositionType::Path`], even if the
/// field was manually set to `true`: [`crate::preprocess::splice`] is
/// decomposition-agnostic and can turn a path decomposition into a
/// non-path tree (see the doc comment on [`TreeSA::path`]).
pub fn optimize_treesa<L: Label>(
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    config: &TreeSA,
) -> Option<NestedEinsum<L>> {
    let preprocess = config.preprocess && config.decomposition_type != DecompositionType::Path;
    let tree = if preprocess {
        let simplified = simplify(code, size_dict);
        let reduced = optimize_treesa_core(&simplified.code, size_dict, config)?;
        splice(&reduced, &simplified.subtrees)
    } else {
        optimize_treesa_core(code, size_dict, config)?
    };

    if config.surgery_iters > 0 {
        let budget = std::time::Duration::MAX;
        let (refined, _report) =
            refine_capped(&tree, code, size_dict, budget, config.surgery_iters);
        return Some(refined);
    }

    Some(tree)
}

/// Bare TreeSA trial loop, without the structural-simplification front-end.
///
/// Used directly by [`optimize_treesa`] when [`TreeSA::preprocess`] is `false`,
/// and by the preprocessed path to optimize the already-reduced network.
fn optimize_treesa_core<L: Label>(
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

            // Convert with openedges for correct root output (issue #13) and
            // score the trial by the tree as it will be emitted: the
            // SA-internal `tree_complexity` does not count dangling-label
            // reductions (matching Julia's `tcscrw`), so ranking trials by it
            // can select a tree that is worse than another trial's.
            let nested = expr_tree_to_nested(&optimized, &code.ixs, &labels, &code.iy);
            let cc = crate::contraction_complexity(&nested, size_dict, &code.ixs);
            let score = config.score.evaluate(cc.tc, cc.sc, cc.rwc);

            (nested, score)
        })
        .collect();

    // Find best result
    let (best_tree, _) = results
        .into_iter()
        .min_by(|(_, s1), (_, s2)| s1.partial_cmp(s2).unwrap())?;

    Some(best_tree)
}

/// Warm-start context for driving a simulated-annealing loop from a seed tree.
///
/// Produced by [`prepare_warm_anneal`]. Holds the mutable [`ExprTree`] together
/// with the label inverse map (`labels`, id → label), per-id `log2_sizes`, and
/// the edge count `nedge` needed to build a [`ScratchSpace`]. This lets a caller
/// (e.g. an example binary) run its own wall-clock-indexed anneal loop over the
/// public `expr_tree` utilities and convert back with [`warm_exprtree_to_nested`].
pub struct WarmAnnealCtx<L: Label> {
    /// The seed tree as a mutable expression tree.
    pub tree: ExprTree,
    /// Inverse label map: integer id → original label.
    pub labels: Vec<L>,
    /// `log2` of each label's dimension, indexed by integer id.
    pub log2_sizes: Vec<f64>,
    /// Number of unique edge labels (bitset capacity for [`ScratchSpace`]).
    pub nedge: usize,
}

/// Convert a seed [`NestedEinsum`] into a [`WarmAnnealCtx`] for warm-start SA.
///
/// Returns `None` if the seed is a bare leaf (nothing to anneal) or conversion
/// fails. The resulting [`ExprTree`] carries no cached complexity, so the public
/// `tree_complexity` recomputes exactly after each in-place mutation.
///
/// # Panics
///
/// Panics if `size_dict` is missing a label that appears in `code`, or if
/// `seed` references a label absent from `code` — the same completeness
/// requirement as the TreeSA optimizer itself.
///
/// # Example
///
/// ```
/// use omeco::{EinCode, GreedyMethod, optimize_code};
/// use omeco::treesa::prepare_warm_anneal;
/// use std::collections::HashMap;
///
/// let code = EinCode::new(
///     vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
///     vec!['i', 'l'],
/// );
/// let sizes: HashMap<char, usize> = [('i', 4), ('j', 8), ('k', 8), ('l', 4)].into();
/// let seed = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
/// let ctx = prepare_warm_anneal(&code, &sizes, &seed).unwrap();
/// assert_eq!(ctx.tree.leaf_count(), 3);
/// ```
pub fn prepare_warm_anneal<L: Label>(
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    seed: &NestedEinsum<L>,
) -> Option<WarmAnnealCtx<L>> {
    let (label_map, labels) = build_label_map(code);
    let nedge = labels.len();
    let log2_sizes: Vec<f64> = labels
        .iter()
        .map(|l| (size_dict[l] as f64).log2())
        .collect();
    let tree = nested_to_expr_tree_inner(seed, &label_map)?;
    Some(WarmAnnealCtx {
        tree,
        labels,
        log2_sizes,
        nedge,
    })
}

/// Convert an annealed [`ExprTree`] back into a [`NestedEinsum`].
///
/// `labels` is the inverse map from [`WarmAnnealCtx`]; the root output is set to
/// `code.iy` verbatim (issue #13). Inverse of [`prepare_warm_anneal`].
///
/// # Example
///
/// ```
/// use omeco::{EinCode, GreedyMethod, optimize_code};
/// use omeco::treesa::{prepare_warm_anneal, warm_exprtree_to_nested};
/// use std::collections::HashMap;
///
/// let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
/// let sizes: HashMap<char, usize> = [('i', 4), ('j', 8), ('k', 4)].into();
/// let seed = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
/// let ctx = prepare_warm_anneal(&code, &sizes, &seed).unwrap();
/// let back = warm_exprtree_to_nested(&ctx.tree, &code, &ctx.labels);
/// assert_eq!(back.leaf_count(), 2);
/// ```
pub fn warm_exprtree_to_nested<L: Label>(
    tree: &ExprTree,
    code: &EinCode<L>,
    labels: &[L],
) -> NestedEinsum<L> {
    expr_tree_to_nested(tree, &code.ixs, labels, &code.iy)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trial selection must rank trials by the cost of the tree the optimizer
    /// actually emits. The SA-internal `tree_complexity` shares Julia's
    /// `tcscrw` blind spot: a dangling label (single occurrence, not in the
    /// output) is summed away at a node where it appears in only one input,
    /// and that summation cost is invisible to the internal metric while
    /// `contraction_complexity` on the emitted tree counts it. Ranking by the
    /// internal metric can therefore return a worse tree than another trial's.
    #[test]
    fn test_trial_selection_ranks_by_emitted_tree_cost() {
        use crate::contraction_complexity;
        use rand::SeedableRng;

        for net_seed in 0..60u64 {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(net_seed);
            let n_tensors = rng.random_range(4..=8);
            let n_labels = rng.random_range(3..=6);
            // Small label pool relative to tensor count yields hyperedges and
            // occasional dangling (single-occurrence) labels.
            let ixs: Vec<Vec<usize>> = (0..n_tensors)
                .map(|_| {
                    let k = rng.random_range(1..=3.min(n_labels));
                    let mut pool: Vec<usize> = (0..n_labels).collect();
                    let mut chosen = Vec::with_capacity(k);
                    for _ in 0..k {
                        let idx = rng.random_range(0..pool.len());
                        chosen.push(pool.swap_remove(idx));
                    }
                    chosen.sort_unstable();
                    chosen
                })
                .collect();
            let iy: Vec<usize> = (0..n_labels)
                .filter(|_| rng.random_range(0..3) == 0)
                .collect();
            let code = EinCode::new(ixs, iy);
            let size_dict: HashMap<usize, usize> = (0..n_labels)
                .map(|l| (l, rng.random_range(2..=4)))
                .collect();

            let config = TreeSA {
                betas: (1..=10).map(|i| i as f64).collect(),
                ntrials: 4,
                niters: 20,
                score: ScoreFunction::default(),
                decomposition_type: DecompositionType::Tree,
                initializer: Initializer::Random,
                preprocess: false,
                surgery_iters: 0,
            };
            let returned = optimize_treesa(&code, &size_dict, &config).unwrap();
            let cc = contraction_complexity(&returned, &size_dict, &code.ixs);
            let returned_score = config.score.evaluate(cc.tc, cc.sc, cc.rwc);

            // Mirror the trial loop and score every trial's *emitted* tree.
            let (label_map, labels) = build_label_map(&code);
            let nedge = labels.len();
            let log2_sizes: Vec<f64> = labels
                .iter()
                .map(|l| (size_dict[l] as f64).log2())
                .collect();
            let int_ixs = convert_to_int_indices(&code.ixs, &label_map);
            let int_iy: Vec<usize> = code.iy.iter().map(|l| label_map[l]).collect();
            let best_emitted_score = (0..config.ntrials)
                .map(|trial_idx| {
                    let mut trng = rand::rngs::SmallRng::seed_from_u64(trial_idx as u64 + 42);
                    let tree = init_random(
                        &int_ixs,
                        &int_iy,
                        nedge,
                        config.decomposition_type,
                        &mut trng,
                    );
                    let optimized = optimize_tree_sa(
                        tree,
                        &log2_sizes,
                        &config.betas,
                        config.niters,
                        &config.score,
                        config.decomposition_type,
                        &mut trng,
                        nedge,
                    );
                    let nested = expr_tree_to_nested(&optimized, &code.ixs, &labels, &code.iy);
                    let tcc = contraction_complexity(&nested, &size_dict, &code.ixs);
                    config.score.evaluate(tcc.tc, tcc.sc, tcc.rwc)
                })
                .fold(f64::INFINITY, f64::min);

            assert!(
                returned_score <= best_emitted_score + 1e-9,
                "net_seed {net_seed}: returned tree scores {returned_score} but a \
                 trial emitted a tree scoring {best_emitted_score} \
                 (ixs {:?} iy {:?})",
                code.ixs,
                code.iy,
            );
        }
    }

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

    #[test]
    fn test_node_sc() {
        let log2_sizes = vec![2.0, 3.0, 4.0];

        // Empty output dims
        assert_eq!(node_sc(&[], &log2_sizes), 0.0);

        // Single label
        assert!((node_sc(&[0], &log2_sizes) - 2.0).abs() < 1e-10);

        // Multiple labels
        assert!((node_sc(&[0, 1, 2], &log2_sizes) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_local_sc_leaf() {
        use crate::expr_tree::{ExprTree, Rule};

        let leaf = ExprTree::leaf(vec![0, 1], 0);
        let log2_sizes = vec![2.0, 3.0];

        // local_sc on a leaf should return the node's sc
        let sc = local_sc(&leaf, Rule::Rule1, &log2_sizes);
        assert!((sc - 5.0).abs() < 1e-10); // 2 + 3 = 5
    }

    #[test]
    fn test_local_sc_node_rules() {
        use crate::expr_tree::{ExprTree, Rule};

        let leaf0 = ExprTree::leaf(vec![0, 1], 0); // sc = 2+3 = 5
        let leaf1 = ExprTree::leaf(vec![1, 2], 1); // sc = 3+4 = 7
        let leaf2 = ExprTree::leaf(vec![2, 3], 2); // sc = 4+2 = 6

        let log2_sizes = vec![2.0, 3.0, 4.0, 2.0];

        // Tree for Rules 1 and 2: ((0,1),2)
        let inner = ExprTree::node(leaf0.clone(), leaf1.clone(), vec![0, 2]); // sc = 2+4 = 6
        let tree12 = ExprTree::node(inner, leaf2.clone(), vec![0, 3]); // sc = 2+2 = 4

        // Rule1/Rule2 uses left child: max(tree_sc, left_sc) = max(4, 6) = 6
        let sc1 = local_sc(&tree12, Rule::Rule1, &log2_sizes);
        assert!((sc1 - 6.0).abs() < 1e-10);

        let sc2 = local_sc(&tree12, Rule::Rule2, &log2_sizes);
        assert!((sc2 - 6.0).abs() < 1e-10);

        // Tree for Rules 3 and 4: (0,(1,2))
        let inner2 = ExprTree::node(leaf1, leaf2, vec![1, 3]); // sc = 3+2 = 5
        let tree34 = ExprTree::node(leaf0, inner2, vec![0, 3]); // sc = 2+2 = 4

        // Rule3/Rule4 uses right child: max(tree_sc, right_sc) = max(4, 5) = 5
        let sc3 = local_sc(&tree34, Rule::Rule3, &log2_sizes);
        assert!((sc3 - 5.0).abs() < 1e-10);

        let sc4 = local_sc(&tree34, Rule::Rule4, &log2_sizes);
        assert!((sc4 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_nested_to_expr_tree_conversion() {
        use crate::greedy::optimize_greedy;
        use crate::GreedyMethod;

        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
            vec!['i', 'l'],
        );
        let sizes: HashMap<char, usize> = [('i', 4), ('j', 8), ('k', 8), ('l', 4)].into();
        let original = optimize_greedy(&code, &sizes, &GreedyMethod::default()).unwrap();

        // Convert to ExprTree using the full conversion path
        let (label_map, labels) = build_label_map(&code);
        let int_ixs = convert_to_int_indices(&code.ixs, &label_map);
        let int_iy: Vec<usize> = code.iy.iter().map(|l| label_map[l]).collect();
        let expr_tree = nested_to_expr_tree(&original, &int_ixs, &int_iy, &label_map);

        assert!(expr_tree.is_some());
        let tree = expr_tree.unwrap();
        assert_eq!(tree.leaf_count(), 3);
        assert!(!tree.is_leaf());

        // Test labels vector is correct
        assert_eq!(labels.len(), 4); // i, j, k, l
    }

    #[test]
    fn test_optimize_treesa_with_rw_optimization() {
        // Test with rw_weight > 0 to exercise that code path
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let sizes: HashMap<char, usize> = [('i', 4), ('j', 8), ('k', 4)].into();

        let mut config = TreeSA::fast();
        config.score.rw_weight = 0.5;
        let result = optimize_treesa(&code, &sizes, &config);

        assert!(result.is_some());
    }

    #[test]
    fn test_optimize_treesa_with_high_sc_target() {
        // Test with very high sc_target (should not penalize space)
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let sizes: HashMap<char, usize> = [('i', 4), ('j', 8), ('k', 4)].into();

        let mut config = TreeSA::fast();
        config.score.sc_target = 1000.0;
        let result = optimize_treesa(&code, &sizes, &config);

        assert!(result.is_some());
    }

    #[test]
    fn test_expr_tree_to_nested() {
        use crate::expr_tree::ExprTree;

        // Create a simple binary tree
        let leaf0 = ExprTree::leaf(vec![0, 1], 0);
        let leaf1 = ExprTree::leaf(vec![1, 2], 1);
        let tree = ExprTree::node(leaf0, leaf1, vec![0, 2]);

        let original_ixs = vec![vec!['i', 'j'], vec!['j', 'k']];
        let inverse_map = vec!['i', 'j', 'k'];
        let openedges = vec!['i', 'k'];

        let nested = expr_tree_to_nested(&tree, &original_ixs, &inverse_map, &openedges);

        assert!(nested.is_binary());
        assert_eq!(nested.leaf_count(), 2);
    }

    #[test]
    fn test_expr_tree_to_nested_deep() {
        use crate::expr_tree::ExprTree;

        // Create a deeper tree: ((0,1),2)
        let leaf0 = ExprTree::leaf(vec![0, 1], 0);
        let leaf1 = ExprTree::leaf(vec![1, 2], 1);
        let leaf2 = ExprTree::leaf(vec![2, 3], 2);
        let inner = ExprTree::node(leaf0, leaf1, vec![0, 2]);
        let tree = ExprTree::node(inner, leaf2, vec![0, 3]);

        let original_ixs = vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']];
        let inverse_map = vec!['i', 'j', 'k', 'l'];
        let openedges = vec!['i', 'l'];

        let nested = expr_tree_to_nested(&tree, &original_ixs, &inverse_map, &openedges);

        assert!(nested.is_binary());
        assert_eq!(nested.leaf_count(), 3);
    }

    #[test]
    fn test_get_child_labels_leaf() {
        let nested: NestedEinsum<char> = NestedEinsum::leaf(0);
        let original_ixs = vec![vec!['i', 'j'], vec!['j', 'k']];

        let labels = get_child_labels(&nested, &original_ixs);
        assert_eq!(labels, vec!['i', 'j']);
    }

    #[test]
    fn test_get_child_labels_node() {
        let leaf0 = NestedEinsum::leaf(0);
        let leaf1 = NestedEinsum::leaf(1);
        let eins = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let nested = NestedEinsum::node(vec![leaf0, leaf1], eins);

        let original_ixs = vec![vec!['i', 'j'], vec!['j', 'k']];

        let labels = get_child_labels(&nested, &original_ixs);
        assert_eq!(labels, vec!['i', 'k']); // Output labels of the node
    }

    #[test]
    fn test_get_child_labels_out_of_bounds() {
        // Test when tensor_index is out of bounds
        let nested: NestedEinsum<char> = NestedEinsum::leaf(99);
        let original_ixs = vec![vec!['i', 'j']];

        let labels = get_child_labels(&nested, &original_ixs);
        assert!(labels.is_empty()); // Should return default empty vec
    }

    /// Regression test for the eins-metadata inconsistency bug.
    ///
    /// `expr_tree_to_nested` used to trust each node's cached `out_dims`. For a
    /// label appearing in more than two tensors, a stale `out_dims` on an
    /// intermediate node produces `eins` bodies that under-report the cost: the
    /// label is dropped from a subtree's output, so the parent contraction never
    /// "sees" it and [`crate::contraction_complexity`] disagrees with the
    /// topology-derived scorer. Here we build a network where label `0` appears in
    /// all four tensors (a hypergraph edge) and hand-build an `ExprTree` whose two
    /// intermediate nodes carry *wrong* `out_dims` that omit `0`. The conversion
    /// must still emit a topology-consistent tree: label `0` must be contracted at
    /// the root, so its cost is counted exactly once.
    #[test]
    fn test_expr_tree_to_nested_ignores_stale_out_dims_on_hyperedge() {
        use crate::contraction_complexity;
        // Labels 0..=4; 0 occurs in every tensor (4-way hyperedge), 1..=4 are open.
        let original_ixs: Vec<Vec<usize>> = vec![vec![0, 1], vec![0, 2], vec![0, 3], vec![0, 4]];
        let openedges = vec![1, 2, 3, 4];
        // Label id == bit index here, so inverse_map is the identity 0..=4.
        let inverse_map: Vec<usize> = vec![0, 1, 2, 3, 4];

        // Subtree A = (t0, t1) with DELIBERATELY WRONG out_dims [1,2] (drops 0).
        let a = ExprTree::node(
            ExprTree::leaf(vec![0, 1], 0),
            ExprTree::leaf(vec![0, 2], 1),
            vec![1, 2],
        );
        // Subtree B = (t2, t3) with wrong out_dims [3,4] (drops 0).
        let b = ExprTree::node(
            ExprTree::leaf(vec![0, 3], 2),
            ExprTree::leaf(vec![0, 4], 3),
            vec![3, 4],
        );
        let root = ExprTree::node(a, b, vec![1, 2, 3, 4]);

        let nested = expr_tree_to_nested(&root, &original_ixs, &inverse_map, &openedges);

        // Each intermediate node must keep label 0 (it occurs outside its subtree).
        if let NestedEinsum::Node { args, .. } = &nested {
            for child in args {
                if let NestedEinsum::Node { eins, .. } = child {
                    assert!(
                        eins.iy.contains(&0),
                        "intermediate node dropped the shared hyperedge label 0: {:?}",
                        eins.iy
                    );
                }
            }
        } else {
            panic!("expected a root node");
        }

        // With all dims 2, the three nodes cost: A over {0,1,2} = 2^3, B over
        // {0,3,4} = 2^3, root over {0,1,2,3,4} = 2^5 (label 0 is contracted here).
        // Total tc = log2(2^3 + 2^3 + 2^5) = log2(48) ~= 5.585. Trusting the stale
        // out_dims would drop 0 from the root union, giving the wrong log2(32) = 5.0.
        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)].into();
        let cc = contraction_complexity(&nested, &sizes, &original_ixs);
        let expected = (48.0_f64).log2();
        assert!(
            (cc.tc - expected).abs() < 1e-9,
            "topology tc should be {expected}, got {} (stale out_dims would under-report as 5.0)",
            cc.tc
        );
    }

    #[test]
    fn test_optimize_treesa_multiple_trials() {
        // Test with multiple trials to ensure parallel execution works
        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
            vec!['i', 'l'],
        );
        let sizes: HashMap<char, usize> = [('i', 4), ('j', 8), ('k', 8), ('l', 4)].into();

        let mut config = TreeSA::fast();
        config.ntrials = 3; // Multiple trials

        let result = optimize_treesa(&code, &sizes, &config);
        assert!(result.is_some());
        assert_eq!(result.unwrap().leaf_count(), 3);
    }

    #[test]
    fn test_init_random_two_tensors() {
        // Test with exactly 2 tensors
        let int_ixs = vec![vec![0, 1], vec![1, 2]];
        let int_iy = vec![0, 2];
        let nedge = 3;
        let mut rng = rand::rng();

        let tree = init_random(&int_ixs, &int_iy, nedge, DecompositionType::Tree, &mut rng);
        assert_eq!(tree.leaf_count(), 2);
    }

    #[test]
    fn test_init_random_many_tensors() {
        // Test with many tensors to exercise recursive partitioning
        let int_ixs = vec![
            vec![0, 1],
            vec![1, 2],
            vec![2, 3],
            vec![3, 4],
            vec![4, 5],
            vec![5, 6],
        ];
        let int_iy = vec![0, 6];
        let nedge = 7;
        let mut rng = rand::rng();

        let tree = init_random(&int_ixs, &int_iy, nedge, DecompositionType::Tree, &mut rng);
        assert_eq!(tree.leaf_count(), 6);
    }

    #[test]
    fn test_init_random_path_two_tensors() {
        let int_ixs = vec![vec![0, 1], vec![1, 2]];
        let int_iy = vec![0, 2];
        let nedge = 3;
        let mut rng = rand::rng();

        let tree = init_random(&int_ixs, &int_iy, nedge, DecompositionType::Path, &mut rng);
        assert_eq!(tree.leaf_count(), 2);
    }

    #[test]
    fn test_init_greedy_success() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let sizes: HashMap<char, usize> = [('i', 4), ('j', 8), ('k', 4)].into();

        let (label_map, _labels) = build_label_map(&code);
        let int_ixs = convert_to_int_indices(&code.ixs, &label_map);
        let int_iy: Vec<usize> = code.iy.iter().map(|l| label_map[l]).collect();

        let tree = init_greedy(&code, &sizes, &label_map, &int_ixs, &int_iy);
        assert!(tree.is_some());
        assert_eq!(tree.unwrap().leaf_count(), 2);
    }

    #[test]
    fn test_optimize_treesa_scalar_output() {
        // Test with scalar output (empty iy)
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'i']], vec![]);
        let sizes: HashMap<char, usize> = [('i', 4), ('j', 8)].into();

        let config = TreeSA::fast();
        let result = optimize_treesa(&code, &sizes, &config);

        assert!(result.is_some());
        let nested = result.unwrap();
        assert_eq!(nested.leaf_count(), 2);
    }

    #[test]
    fn test_optimize_treesa_with_different_decomp() {
        // Test tree vs path decomposition
        let code = EinCode::new(
            vec![vec!['a', 'b'], vec!['b', 'c'], vec!['c', 'd']],
            vec!['a', 'd'],
        );
        let sizes: HashMap<char, usize> = [('a', 2), ('b', 4), ('c', 4), ('d', 2)].into();

        // Tree decomposition
        let mut config_tree = TreeSA::fast();
        config_tree.decomposition_type = DecompositionType::Tree;
        let result_tree = optimize_treesa(&code, &sizes, &config_tree);
        assert!(result_tree.is_some());

        // Path decomposition
        let mut config_path = TreeSA::fast();
        config_path.decomposition_type = DecompositionType::Path;
        config_path.initializer = Initializer::Random;
        let result_path = optimize_treesa(&code, &sizes, &config_path);
        assert!(result_path.is_some());
    }

    #[test]
    fn test_nested_to_expr_tree_inner_leaf() {
        // Test the inner function with a leaf (edge case - should return None)
        let nested: NestedEinsum<char> = NestedEinsum::leaf(0);
        let label_map: HashMap<char, usize> = [('i', 0), ('j', 1)].into();

        let result = nested_to_expr_tree_inner(&nested, &label_map);
        assert!(result.is_none());
    }

    #[test]
    fn test_prepare_warm_anneal_roundtrip() {
        use crate::{contraction_complexity, optimize_code, GreedyMethod};
        let code = EinCode::new(
            vec![
                vec!['i', 'j'],
                vec!['j', 'k'],
                vec!['k', 'l'],
                vec!['l', 'm'],
            ],
            vec!['i', 'm'],
        );
        let sizes: HashMap<char, usize> = [('i', 4), ('j', 8), ('k', 8), ('l', 8), ('m', 4)].into();
        let seed = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
        let ctx = prepare_warm_anneal(&code, &sizes, &seed).unwrap();
        assert_eq!(ctx.tree.leaf_count(), 4);
        assert_eq!(ctx.labels.len(), ctx.log2_sizes.len());
        assert_eq!(ctx.nedge, ctx.labels.len());

        // Round-tripping an unmodified context reproduces the seed's leaves and a
        // finite, matching time complexity.
        let back = warm_exprtree_to_nested(&ctx.tree, &code, &ctx.labels);
        assert_eq!(back.leaf_count(), 4);
        let cc_seed = contraction_complexity(&seed, &sizes, &code.ixs);
        let cc_back = contraction_complexity(&back, &sizes, &code.ixs);
        assert!((cc_seed.tc - cc_back.tc).abs() < 1e-9);
    }

    #[test]
    fn test_prepare_warm_anneal_accepts_treewidth_unary_nodes() {
        use crate::{contraction_complexity, optimize_code, Treewidth};

        // Treewidth reduces each private leg locally, producing unary nodes
        // beneath a binary contraction tree.
        let code = EinCode::new(
            vec![
                vec!['x', 'a'],
                vec!['x', 'b'],
                vec!['x', 'c'],
                vec!['x', 'd'],
            ],
            vec![],
        );
        let sizes: HashMap<char, usize> = [('x', 2), ('a', 8), ('b', 8), ('c', 8), ('d', 8)]
            .into_iter()
            .collect();
        let seed = optimize_code(&code, &sizes, &Treewidth::default()).unwrap();
        let ctx = prepare_warm_anneal(&code, &sizes, &seed)
            .expect("warm-start should accept at-most-binary Treewidth trees");
        let back = warm_exprtree_to_nested(&ctx.tree, &code, &ctx.labels);

        assert_eq!(back.leaf_count(), code.num_tensors());
        assert_eq!(back.output_labels(&code.ixs), code.iy);
        let seed_cc = contraction_complexity(&seed, &sizes, &code.ixs);
        let back_cc = contraction_complexity(&back, &sizes, &code.ixs);
        assert!((seed_cc.tc - back_cc.tc).abs() < 1e-9);
        assert!((seed_cc.sc - back_cc.sc).abs() < 1e-9);
    }

    #[test]
    fn test_warm_conversion_handles_unary_internal_and_rejects_nary() {
        let label_map: HashMap<char, usize> = [('a', 0), ('b', 1), ('c', 2)].into();
        let binary = NestedEinsum::node(
            vec![NestedEinsum::leaf(0), NestedEinsum::leaf(1)],
            EinCode::new(vec![vec!['a', 'b'], vec!['b', 'c']], vec!['a', 'c']),
        );
        let unary = NestedEinsum::node(vec![binary], EinCode::new(vec![vec!['a', 'c']], vec!['a']));
        let converted = nested_to_expr_tree_inner(&unary, &label_map)
            .expect("unary node around a binary child should be fused");
        assert_eq!(converted.labels(), &[0]);
        assert_eq!(converted.leaf_count(), 2);

        let nary = NestedEinsum::node(
            vec![
                NestedEinsum::leaf(0),
                NestedEinsum::leaf(1),
                NestedEinsum::leaf(2),
            ],
            EinCode::new(vec![vec!['a'], vec!['b'], vec!['c']], Vec::<char>::new()),
        );
        assert!(nested_to_expr_tree_inner(&nary, &label_map).is_none());
    }

    #[test]
    #[should_panic(expected = "Cannot create tree with no tensors")]
    fn test_init_random_rejects_empty_input() {
        let mut rng = SmallRng::seed_from_u64(1);
        let _ = init_random(&[], &[], 0, DecompositionType::Tree, &mut rng);
    }

    #[test]
    fn test_prepare_warm_anneal_leaf_seed_is_none() {
        // A bare leaf seed has nothing to anneal.
        let code = EinCode::new(vec![vec!['i', 'j']], vec!['i', 'j']);
        let sizes: HashMap<char, usize> = [('i', 4), ('j', 4)].into();
        let seed: NestedEinsum<char> = NestedEinsum::leaf(0);
        assert!(prepare_warm_anneal(&code, &sizes, &seed).is_none());
    }

    #[test]
    fn test_treesa_pipeline_defaults() {
        let config = TreeSA::default();
        assert!(config.preprocess);
        assert_eq!(config.surgery_iters, 0);
        let fast = TreeSA::fast();
        assert!(fast.preprocess);
        assert_eq!(fast.surgery_iters, 0);
        let tuned = TreeSA::default()
            .with_preprocess(false)
            .with_surgery_iters(30);
        assert!(!tuned.preprocess);
        assert_eq!(tuned.surgery_iters, 30);
    }

    /// Pins the preset-level preprocess contract: `TreeSA::default()` (and
    /// hence `TreeSA::fast()`, which builds on it) opts into the
    /// simplify/splice front-end, while `TreeSA::path()` opts out because
    /// splice does not preserve the path-decomposition guarantee (see the
    /// doc comment on `TreeSA::path`, and
    /// `test_path_decomposition_random_graph_n50` in `lib.rs` for the
    /// end-to-end regression coverage).
    #[test]
    fn test_preprocess_default_wiring_per_preset() {
        assert!(TreeSA::default().preprocess);
        assert!(!TreeSA::path().preprocess);
    }

    #[test]
    fn test_default_pipeline_preprocess_preserves_interfaces() {
        // Matrix chain: simplify collapses it; the spliced tree must keep all leaves.
        let code = EinCode::new(
            vec![
                vec!['a', 'b'],
                vec!['b', 'c'],
                vec!['c', 'd'],
                vec!['d', 'e'],
            ],
            vec!['a', 'e'],
        );
        let sizes: HashMap<char, usize> = [('a', 2), ('b', 2), ('c', 2), ('d', 2), ('e', 2)].into();
        let tree = optimize_treesa(&code, &sizes, &TreeSA::fast()).unwrap();
        assert_eq!(tree.leaf_count(), 4);
        let cc = crate::contraction_complexity(&tree, &sizes, &code.ixs);
        assert!(cc.tc.is_finite());
    }

    /// Loader for the shared benchmark graph JSON (`{ "ixs", "iy", "sizes" }`,
    /// as read by `omeco/examples/benchmark.rs`), not the `edge_list` schema
    /// used only by `reg3_220.json` (see `test_reg3_220_treesa` in `lib.rs`).
    fn load_benchmark_graph(name: &str) -> (EinCode<usize>, HashMap<usize, usize>) {
        let graph_json =
            std::fs::read_to_string(format!("../benchmarks/graphs/{name}.json")).unwrap();
        let graph: serde_json::Value = serde_json::from_str(&graph_json).unwrap();
        let ixs: Vec<Vec<usize>> = graph["ixs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|ix| {
                ix.as_array()
                    .unwrap()
                    .iter()
                    .map(|l| l.as_u64().unwrap() as usize)
                    .collect()
            })
            .collect();
        let iy: Vec<usize> = graph["iy"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l.as_u64().unwrap() as usize)
            .collect();
        let sizes: HashMap<usize, usize> = graph["sizes"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.parse::<usize>().unwrap(), v.as_u64().unwrap() as usize))
            .collect();
        (EinCode::new(ixs, iy), sizes)
    }

    #[test]
    fn test_default_pipeline_quality_on_benchmark_graphs() {
        // Spec §4: default (preprocess on) is within 0.5 bits of the bare loop on
        // the benchmark graphs; exact equality where simplify is a no-op.
        //
        // Deviation from task-3-brief.md Step 1: the brief paired "reg3_50" with
        // `no_op: true`, assuming 3-regular graphs are never simplifiable. That
        // holds for a graph with an even total degree sum, but
        // `benchmarks/graphs/reg3_50.json` (50 vertices, odd) carries one
        // rank-1 "defect" tensor to balance parity; simplify's rank-non-increasing
        // rule fuses it into a neighbour (50 -> 46 tensors), so tc_pre and tc_raw
        // are close but not bit-identical (verified: diff ~0.045 bits, well inside
        // the 0.5-bit tolerance). Swapped in "petersen" (10 vertices, uniformly
        // rank-3, empirically confirmed n_reduced == n_original) as the true
        // no-op case; reg3_50 would still pass under the tolerance branch.
        for (name, no_op) in [("grid_4x4", false), ("petersen", true)] {
            let (code, sizes) = load_benchmark_graph(name);
            let cfg = TreeSA::fast();
            let with_pre = optimize_treesa(&code, &sizes, &cfg).unwrap();
            let without =
                optimize_treesa(&code, &sizes, &cfg.clone().with_preprocess(false)).unwrap();
            let tc_pre = crate::contraction_complexity(&with_pre, &sizes, &code.ixs).tc;
            let tc_raw = crate::contraction_complexity(&without, &sizes, &code.ixs).tc;
            if no_op {
                assert!(
                    (tc_pre - tc_raw).abs() < 1e-9,
                    "{name}: {tc_pre} vs {tc_raw}"
                );
            } else {
                assert!(tc_pre <= tc_raw + 0.5, "{name}: {tc_pre} > {tc_raw} + 0.5");
            }
        }
    }

    #[test]
    fn test_preprocess_off_matches_core_loop() {
        let code = EinCode::new(
            vec![vec!['a', 'b'], vec!['b', 'c'], vec!['c', 'a']],
            Vec::<char>::new(),
        );
        let sizes: HashMap<char, usize> = [('a', 4), ('b', 4), ('c', 4)].into();
        let config = TreeSA::fast().with_preprocess(false);
        let via_public = optimize_treesa(&code, &sizes, &config).unwrap();
        let via_core = optimize_treesa_core(&code, &sizes, &config).unwrap();
        assert_eq!(
            format!("{via_public:?}"),
            format!("{via_core:?}"),
            "preprocess=false must be byte-identical to the bare trial loop"
        );
    }

    /// Build a 2D periodic grid tensor network (each bond a distinct label),
    /// mirroring `waist_surgery`'s own `grid` test helper.
    fn grid_code(rows: usize, cols: usize) -> EinCode<usize> {
        let mut next = 0usize;
        let mut edge = |_a: (usize, usize), _b: (usize, usize)| {
            let e = next;
            next += 1;
            e
        };
        // Assign an id per undirected grid edge.
        let mut hbond = vec![vec![0usize; cols]; rows];
        let mut vbond = vec![vec![0usize; cols]; rows];
        for r in 0..rows {
            for c in 0..cols {
                hbond[r][c] = edge((r, c), (r, (c + 1) % cols));
                vbond[r][c] = edge((r, c), ((r + 1) % rows, c));
            }
        }
        let mut ixs = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let left = hbond[r][(c + cols - 1) % cols];
                let right = hbond[r][c];
                let up = vbond[(r + rows - 1) % rows][c];
                let down = vbond[r][c];
                ixs.push(vec![left, right, up, down]);
            }
        }
        EinCode::new(ixs, vec![])
    }

    /// `surgery_iters > 0` is fully deterministic (internal RNG seeds are
    /// fixed and no wall-clock deadline binds `optimize_treesa`'s cap): two
    /// runs of the same config produce byte-identical trees. It is also
    /// never worse than the no-surgery baseline.
    #[test]
    fn test_surgery_iters_deterministic_and_never_worse() {
        use crate::contraction_complexity;
        // 4x4 periodic grid — a frozen-waist-style instance where surgery acts.
        let code = grid_code(4, 4);
        let sizes: HashMap<usize, usize> =
            code.unique_labels().into_iter().map(|l| (l, 2)).collect();
        let base_cfg = TreeSA::fast();
        let base = optimize_treesa(&code, &sizes, &base_cfg).unwrap();
        let base_tc = contraction_complexity(&base, &sizes, &code.ixs).tc;

        let cfg = TreeSA::fast().with_surgery_iters(5);
        let refined_a = optimize_treesa(&code, &sizes, &cfg).unwrap();
        let refined_b = optimize_treesa(&code, &sizes, &cfg).unwrap();
        assert_eq!(
            format!("{refined_a:?}"),
            format!("{refined_b:?}"),
            "surgery_iters > 0 must be deterministic across runs"
        );

        let refined_tc = contraction_complexity(&refined_a, &sizes, &code.ixs).tc;
        assert!(refined_tc <= base_tc + 1e-9, "{refined_tc} > {base_tc}");
        assert_eq!(refined_a.leaf_count(), code.num_tensors());
    }

    #[test]
    fn test_surgery_off_is_reproducible() {
        let code = EinCode::new(
            vec![vec![0usize, 1], vec![1, 2], vec![2, 3], vec![3, 0]],
            vec![],
        );
        let sizes: HashMap<usize, usize> = (0..4).map(|l| (l, 8)).collect();
        let cfg = TreeSA::fast(); // surgery_iters == 0
        let a = optimize_treesa(&code, &sizes, &cfg).unwrap();
        let b = optimize_treesa(&code, &sizes, &cfg).unwrap();
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }

    /// Regression: manually setting `decomposition_type: Path` while leaving
    /// `preprocess: true` (the default) must not silently void the
    /// path-decomposition guarantee — `optimize_treesa` auto-skips
    /// preprocessing in this case, same as the `TreeSA::path()` preset.
    #[test]
    fn test_path_decomposition_holds_with_preprocess_true() {
        let code = EinCode::new(
            vec![
                vec!['a', 'b'],
                vec!['b', 'c'],
                vec!['c', 'd'],
                vec!['d', 'e'],
            ],
            vec!['a', 'e'],
        );
        let sizes: HashMap<char, usize> = [('a', 2), ('b', 2), ('c', 2), ('d', 2), ('e', 2)].into();
        let mut config = TreeSA::fast();
        config.initializer = Initializer::Random;
        config.decomposition_type = DecompositionType::Path;
        config.preprocess = true; // manually left on, unlike TreeSA::path()
        let tree = optimize_treesa(&code, &sizes, &config).unwrap();
        assert!(tree.is_path_decomposition());
        assert_eq!(tree.leaf_count(), 4);
    }

    #[test]
    fn test_random_initializer_produces_valid_tree() {
        let code = EinCode::new(
            vec![vec!['a', 'b'], vec!['b', 'c'], vec!['c', 'd']],
            vec!['a', 'd'],
        );
        let sizes: HashMap<char, usize> = [('a', 2), ('b', 2), ('c', 2), ('d', 2)].into();
        let config = TreeSA {
            initializer: Initializer::Random,
            ntrials: 1,
            niters: 5,
            ..TreeSA::fast()
        };
        let tree = optimize_treesa(&code, &sizes, &config).unwrap();
        assert_eq!(tree.leaf_count(), 3);
        let cc = crate::contraction_complexity(&tree, &sizes, &code.ixs);
        assert!(cc.tc.is_finite());
    }
}
