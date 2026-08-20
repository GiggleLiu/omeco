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
use crate::preprocess::{simplify, splice};
use crate::score::ScoreFunction;
use crate::utils::fast_log2sumexp2;
#[cfg(test)]
use crate::waist_surgery::refine_capped;
use crate::waist_surgery::{
    gated_sweep, refine_capped_seeded_with_trace, refine_capped_seeded_with_trace_opts,
    RebuildMode, SurgeryOptions, SurgeryScope, WaistUpdate,
};
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
    /// Number of interleaved anneal–surgery rounds ([`anneal_surgery_rounds`],
    /// the paper's Algorithm 1) run on the tree the pipeline selected; `0`
    /// disables the loop entirely (the default). Each round applies one
    /// waist-surgery iteration and then a cold fine-tuning pass: at most 15
    /// stratified configured levels with `β >= 1`, at most 30 span-gated sweeps
    /// per coarse-to-fine span, and at most three deterministic serial trials.
    /// This follows TensorBranching's specified-tree `TreeSARefiner` policy and
    /// avoids restarting an already optimized tree at the default schedule's
    /// hot end.
    ///
    /// The best reduced-network tree seen in the loop is returned. After
    /// splice-back, [`optimize_treesa`] compares it with the rounds-off result
    /// on the original network, so enabling any positive round count is never
    /// worse than leaving the loop off under the configured [`TreeSA::score`]
    /// (`tc` alone carries no such guarantee). The standalone
    /// [`anneal_surgery_rounds`] loop is additionally monotone in its round
    /// count, under that same score, on the network it is given. See
    /// [`crate::waist_surgery`] for the surgery step itself.
    ///
    /// # Path decomposition
    ///
    /// [`optimize_treesa`] auto-skips the rounds loop (treating this field as
    /// `0`) whenever `decomposition_type` is [`DecompositionType::Path`]:
    /// neither surgery nor the specified-tree fine tuning preserves the
    /// path-decomposition guarantee. See [`TreeSA::path`].
    ///
    /// # Determinism
    ///
    /// Unlike the low-level [`crate::waist_surgery::refine`]/[`refine_capped`]
    /// wall-clock APIs, `optimize_treesa` binds the loop to no deadline —
    /// internal RNG seeds are fixed throughout. The whole `TreeSA` API is
    /// therefore fully reproducible across machines for any fixed config,
    /// including configs with `surgery_iters > 0`.
    ///
    /// [`refine_capped`]: crate::waist_surgery::refine_capped
    pub surgery_iters: u64,
    /// Probability that a TreeSA sweep is replaced by one global waist-surgery
    /// proposal. The proposal is accepted by the Metropolis rule at the current
    /// inverse temperature; cooling then continues without a restart or a
    /// post-surgery anneal. `0.0` disables the rule (the default).
    ///
    /// The rule is skipped for [`DecompositionType::Path`] because the global
    /// repartition is not path-preserving.
    pub surgery_probability: f64,
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
            surgery_probability: 0.0,
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
            surgery_probability: 0.0,
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
    ///
    /// For the same reason, [`optimize_treesa`] also skips the
    /// [`TreeSA::surgery_iters`] rounds loop and the
    /// [`TreeSA::surgery_probability`] update rule for path configs: waist
    /// surgery rebuilds subtrees around a re-optimized cut and the
    /// specified-tree fine tuner applies general binary-tree moves, neither of which
    /// is path-preserving.
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

    /// Set the number of interleaved anneal–surgery rounds (0 disables).
    ///
    /// Each round uses the cold specified-tree fine-tuning policy documented on
    /// [`TreeSA::surgery_iters`]; the end-to-end result is guarded against the
    /// rounds-off baseline. The loop is skipped for
    /// [`DecompositionType::Path`] configs (it is not path-preserving).
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
    pub fn with_surgery_iters(mut self, rounds: u64) -> Self {
        self.surgery_iters = rounds;
        self
    }

    /// Set the probability that one local sweep is replaced by a global
    /// surgery proposal at the same inverse temperature.
    ///
    /// # Panics
    ///
    /// Panics unless `probability` is finite and in `[0, 1]`.
    pub fn with_surgery_probability(mut self, probability: f64) -> Self {
        assert!(
            probability.is_finite() && (0.0..=1.0).contains(&probability),
            "surgery probability must be finite and in [0, 1]"
        );
        self.surgery_probability = probability;
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

/// Cold, sparse schedule for refining an already-optimized tree.
///
/// TensorBranching's `TreeSARefiner` uses `β = 1:1:15`: a specified-tree
/// refinement does not need the initializer-forgetting, high-temperature end
/// of the default TreeSA schedule. Preserve custom schedules that contain no
/// `β >= 1`, and otherwise retain at most 15 stratified cold levels.
fn fine_tune_beta_schedule(betas: &[f64]) -> Vec<f64> {
    const MAX_LEVELS: usize = 15;
    let cold: Vec<f64> = betas.iter().copied().filter(|beta| *beta >= 1.0).collect();
    let source = if cold.is_empty() { betas } else { &cold };
    if source.len() <= MAX_LEVELS {
        return source.to_vec();
    }
    let last = source.len() - 1;
    (0..MAX_LEVELS)
        .map(|i| source[i * last / (MAX_LEVELS - 1)])
        .collect()
}

/// Test-only wrapper around [`fine_tune_tree_sa_counted`] that drops the
/// sweep count and returns `(best_seen, final_endpoint)`.
#[cfg(test)]
fn fine_tune_tree_sa<F>(
    tree: ExprTree,
    log2_sizes: &[f64],
    betas: &[f64],
    sweeps_per_span: usize,
    score_tree: &F,
    rng: &mut rand::rngs::SmallRng,
    nedge: usize,
) -> (ExprTree, ExprTree)
where
    F: Fn(&ExprTree) -> f64,
{
    let (best, endpoint, _) = fine_tune_tree_sa_counted(
        tree,
        log2_sizes,
        betas,
        sweeps_per_span,
        score_tree,
        rng,
        nedge,
    );
    (best, endpoint)
}

/// Fine-tune a specified tree and return `(best_seen, final_endpoint, sweeps)`.
///
/// The endpoint remains useful diagnostics for fine-tuning damage, but the
/// paper's `Anneal` operator returns its best tree. The relaxation uses the
/// span-gated kernel from attempt-054: each level of the coarse-to-fine span
/// ladder receives `sweeps_per_span` sweeps across the cold beta schedule. The
/// tree is rescored after every sweep with the caller's emitted-tree scorer, so
/// a later uphill sweep cannot erase an earlier, better checkpoint. `sweeps`
/// counts every span-gated sweep executed, which
/// [`RoundsReport::fine_tune_sweeps_total`] accumulates as a deterministic
/// work measure.
fn fine_tune_tree_sa_counted<F>(
    mut tree: ExprTree,
    log2_sizes: &[f64],
    betas: &[f64],
    sweeps_per_span: usize,
    score_tree: &F,
    rng: &mut rand::rngs::SmallRng,
    nedge: usize,
) -> (ExprTree, ExprTree, u64)
where
    F: Fn(&ExprTree) -> f64,
{
    let mut best = tree.clone();
    let mut best_score = score_tree(&best);
    let mut scratch = ScratchSpace::new(nedge);
    let mut s_lin = f64::exp2(tree_complexity(&tree, log2_sizes).0);
    let mut sweeps_total = 0_u64;

    let n = tree.leaf_count();
    let mut span = ((n + 29) / 30).max(2);
    let mut spans = Vec::new();
    while span > 2 {
        spans.push(span);
        span /= 2;
    }
    spans.push(2);

    if !betas.is_empty() {
        let denom = sweeps_per_span.saturating_sub(1).max(1);
        for span in spans {
            for sweep in 0..sweeps_per_span {
                let beta_index = sweep * (betas.len() - 1) / denom;
                gated_sweep(
                    &mut tree,
                    betas[beta_index],
                    span,
                    log2_sizes,
                    rng,
                    &mut scratch,
                    &mut s_lin,
                );
                sweeps_total += 1;
                let candidate_score = score_tree(&tree);
                if candidate_score < best_score {
                    best_score = candidate_score;
                    best = tree.clone();
                }
            }
            s_lin = f64::exp2(tree_complexity(&tree, log2_sizes).0);
        }
    }
    (best, tree, sweeps_total)
}

/// Run TreeSA with a root-level mixture of local sweeps and global surgery
/// proposals. Surgery replaces a sweep, uses the current beta, and never
/// changes or restarts the cooling schedule.
#[allow(clippy::too_many_arguments)]
fn optimize_tree_sa_mixed<R: Rng>(
    mut tree: ExprTree,
    log2_sizes: &[f64],
    betas: &[f64],
    niters: usize,
    score: &ScoreFunction,
    decomp: DecompositionType,
    rng: &mut R,
    nedge: usize,
    surgery: &WaistUpdate,
    surgery_probability: f64,
) -> ExprTree {
    let log2_rw_weight = if score.rw_weight > 0.0 {
        score.rw_weight.log2()
    } else {
        f64::NEG_INFINITY
    };
    let mut scratch = ScratchSpace::new(nedge);

    for &beta in betas {
        for _ in 0..niters {
            if rng.random::<f64>() < surgery_probability {
                if let Some(candidate) = surgery.propose(&tree, rng) {
                    let before = tree_complexity(&tree, log2_sizes);
                    let after = tree_complexity(&candidate, log2_sizes);
                    let d_energy = surgery_energy_difference(
                        before,
                        after,
                        score.sc_target,
                        score.sc_weight,
                        log2_rw_weight,
                    );
                    if rng.random::<f64>() < (-beta * d_energy).exp() {
                        tree = candidate;
                        scratch = ScratchSpace::new(nedge);
                    }
                }
            } else {
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
    }
    tree
}

/// Whole-tree analogue of TreeSA's local rewrite energy difference.
///
/// Complexity values stay in log2 units. In particular, this must not call
/// `ScoreFunction::evaluate`, whose linear `2^tc` scale is not the scale to
/// which TreeSA's beta schedule is calibrated.
#[inline]
fn surgery_energy_difference(
    before: (f64, f64, f64),
    after: (f64, f64, f64),
    sc_target: f64,
    sc_weight: f64,
    log2_rw_weight: f64,
) -> f64 {
    let primary = |(tc, _, rw): (f64, f64, f64)| {
        if log2_rw_weight > f64::NEG_INFINITY {
            fast_log2sumexp2(tc, log2_rw_weight + rw)
        } else {
            tc
        }
    };
    let mut d_energy = primary(after) - primary(before);
    if before.1.max(after.1) > sc_target {
        d_energy += sc_weight * (after.1 - before.1);
    }
    d_energy
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

            // Merge the two subtrees' occurrence counts, folding the smaller
            // map into the larger. Always folding into the left map costs
            // O(|right|) per node, which is quadratic on trees whose heavy
            // side is the right child; small-to-large merging keeps the total
            // near-linear. Counts are added, so the merged contents — and
            // therefore every lookup below and in the caller — are unchanged.
            let mut within = left_within;
            let mut folded = right_within;
            if folded.len() > within.len() {
                std::mem::swap(&mut within, &mut folded);
            }
            for (l, c) in folded {
                *within.entry(l).or_insert(0) += c;
            }

            // At the root use openedges verbatim (issue #13). Otherwise a label is
            // an output iff it still occurs outside this subtree (within < global)
            // or is an open/output label. Iterate children outputs for a stable,
            // Ord-free ordering.
            let iy: Vec<L> = if level == 0 {
                openedges.to_vec()
            } else {
                // `seen` replaces a `Vec::contains` scan per label, which was
                // quadratic in the width of the intermediate — the dominant
                // cost on circuit-like networks, whose intermediates expose
                // hundreds of labels. Insertion order is preserved, and a
                // label that fails the output test is deterministic, so
                // skipping its duplicates yields the same `out` as retesting.
                let mut out: Vec<L> = Vec::new();
                let mut seen: HashSet<L> = HashSet::with_capacity(left_out.len() + right_out.len());
                for l in left_out.iter().chain(right_out.iter()) {
                    let first_occurrence = seen.insert(l.clone());
                    let w = within.get(l).copied().unwrap_or(0);
                    let g = global_count.get(l).copied().unwrap_or(0);
                    if first_occurrence && (open_set.contains(l) || w < g) {
                        out.push(l.clone());
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
/// network, optional interleaved anneal–surgery rounds, and splice-back —
/// controlled by [`TreeSA::preprocess`]. A positive
/// [`TreeSA::surgery_iters`] runs those rounds
/// ([`anneal_surgery_rounds`], the paper's Algorithm 1) on the reduced network
/// before splice-back: each round is one waist-surgery iteration followed by a
/// cold specified-tree fine-tuning pass. Running surgery before splice-back is
/// essential: the paper's FM cut
/// search and side rebuilds operate on the simplified tensor hypergraph, not on
/// the restored full graph. After splice-back, the candidate is compared with
/// the rounds-off tree on the original network, so the result is never worse
/// than with `surgery_iters = 0` under the configured [`TreeSA::score`]. The
/// whole pipeline is fully deterministic (rounds are counted, never timed).
///
/// [`TreeSA::preprocess`] is automatically treated as disabled whenever
/// [`TreeSA::decomposition_type`] is [`DecompositionType::Path`], even if the
/// field was manually set to `true`: [`crate::preprocess::splice`] is
/// decomposition-agnostic and can turn a path decomposition into a
/// non-path tree (see the doc comment on [`TreeSA::path`]).
/// [`TreeSA::surgery_iters`] is skipped in exactly the same case and for the
/// same reason: neither the surgery step nor the specified-tree fine tuning is
/// path-preserving, so a `Path` config always returns the plain annealed
/// path regardless of how many rounds were requested.
pub fn optimize_treesa<L: Label>(
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    config: &TreeSA,
) -> Option<NestedEinsum<L>> {
    optimize_treesa_with_seed(code, size_dict, config, 42)
}

/// Optimize an [`EinCode`] with TreeSA using a caller-selected trial RNG seed.
///
/// This is identical to [`optimize_treesa`] except that trial `i` starts from
/// `seed + i` instead of the historical `42 + i`. It is intended for matched,
/// reproducible experimental repetitions; [`optimize_treesa`] and all
/// [`TreeSA`] defaults remain unchanged.
///
/// # Example
///
/// ```
/// use omeco::treesa::optimize_treesa_seeded;
/// use omeco::{EinCode, TreeSA};
/// use std::collections::HashMap;
///
/// let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
/// let sizes: HashMap<char, usize> = [('i', 2), ('j', 4), ('k', 2)].into();
/// let first = optimize_treesa_seeded(&code, &sizes, &TreeSA::fast(), 7000).unwrap();
/// let again = optimize_treesa_seeded(&code, &sizes, &TreeSA::fast(), 7000).unwrap();
/// assert_eq!(first.leaf_indices(), again.leaf_indices());
/// ```
pub fn optimize_treesa_seeded<L: Label>(
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    config: &TreeSA,
    seed: u64,
) -> Option<NestedEinsum<L>> {
    optimize_treesa_with_seed(code, size_dict, config, seed)
}

fn optimize_treesa_with_seed<L: Label>(
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    config: &TreeSA,
    seed: u64,
) -> Option<NestedEinsum<L>> {
    let preprocess = config.preprocess && config.decomposition_type != DecompositionType::Path;
    if preprocess {
        let simplified = simplify(code, size_dict);
        let reduced = optimize_treesa_core_seeded(&simplified.code, size_dict, config, seed)?;
        if config.surgery_iters > 0 {
            let candidate = anneal_surgery_rounds(
                &reduced,
                &simplified.code,
                size_dict,
                config,
                config.surgery_iters,
            )
            .0;
            let baseline = splice(&reduced, &simplified.subtrees);
            let candidate = splice(&candidate, &simplified.subtrees);
            let score_original = |tree: &NestedEinsum<L>| {
                let cc = crate::contraction_complexity(tree, size_dict, &code.ixs);
                config.score.evaluate(cc.tc, cc.sc, cc.rwc)
            };
            return Some(if score_original(&candidate) < score_original(&baseline) {
                candidate
            } else {
                baseline
            });
        }
        return Some(splice(&reduced, &simplified.subtrees));
    }

    let tree = optimize_treesa_core_seeded(code, size_dict, config, seed)?;
    if config.surgery_iters > 0 && config.decomposition_type != DecompositionType::Path {
        return Some(anneal_surgery_rounds(&tree, code, size_dict, config, config.surgery_iters).0);
    }

    Some(tree)
}

/// Run `f` on `pool`, or on the current (global) pool if it could not be built.
///
/// Splitting this out from the call site keeps the fallback testable: a
/// thread-spawn failure is otherwise impossible to provoke in a unit test, and
/// untested error handling is how a "cannot allocate threads" turns into a
/// silent wrong answer.
fn install_or_run<T, F>(pool: Result<rayon::ThreadPool, rayon::ThreadPoolBuildError>, f: F) -> T
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    match pool {
        Ok(pool) => pool.install(f),
        Err(_) => f(),
    }
}

/// Order trial scores so that any NaN loses, whatever its sign bit.
///
/// A trial whose tree has an intermediate too large to represent scores NaN:
/// with the default `rw_weight` of zero, `rw_weight * 2f64.powf(rwc)` is
/// `0.0 * inf`. Ranking such trials needs care on two counts.
///
/// `partial_cmp(..).unwrap()` panics on them, aborting the optimizer on valid
/// input. `f64::total_cmp` does not panic but is not a fix either: it orders
/// by sign bit, and the sign of a hardware-produced NaN is platform-dependent
/// — x86_64 yields a negative NaN for `0.0 * inf` where aarch64 yields a
/// positive one. Under `total_cmp` the overflowed trial would therefore *win*
/// on x86_64 and lose on aarch64, which both returns a nonsense tree and
/// breaks the cross-platform determinism the committed benchmark artifact
/// depends on. Testing `is_nan()` explicitly is sign-agnostic.
fn nan_last(a: f64, b: f64) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a.is_nan(), b.is_nan()) {
        (false, false) => a.total_cmp(&b),
        (false, true) => Ordering::Less,
        (true, false) => Ordering::Greater,
        (true, true) => Ordering::Equal,
    }
}

/// Worker-thread stack reservation for a network of `num_tensors` tensors.
///
/// The recursive tree walks use a few hundred bytes per level and can recurse
/// once per leaf on a fully unbalanced tree — a path decomposition reaches
/// exactly that — so the requirement grows linearly with the network. 4 KiB
/// per tensor leaves roughly an order of magnitude of headroom over the
/// measured frame sizes, and the floor keeps small networks on a conventional
/// stack.
///
/// The 1 GiB ceiling means this is a mitigation sized for realistic networks,
/// not a guarantee for arbitrary ones: beyond about 262 000 tensors the budget
/// per level starts shrinking again, and a deep enough tree at that scale
/// could still overflow. Making the walks iterative is the only complete fix.
/// Stacks are virtual, so unused pages are not resident, but they do consume
/// address space and per-thread resources.
fn trial_stack_size(num_tensors: usize) -> usize {
    const PER_TENSOR: usize = 4 * 1024;
    const MIN: usize = 32 * 1024 * 1024;
    const MAX: usize = 1024 * 1024 * 1024;
    num_tensors.saturating_mul(PER_TENSOR).clamp(MIN, MAX)
}

/// Test-only wrapper around [`optimize_treesa_core_seeded`] with the historical
/// trial seed base `42`.
#[cfg(test)]
fn optimize_treesa_core<L: Label>(
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    config: &TreeSA,
) -> Option<NestedEinsum<L>> {
    optimize_treesa_core_seeded(code, size_dict, config, 42)
}

/// Bare TreeSA trial loop, without the structural-simplification front-end.
///
/// Used by [`optimize_treesa`] / [`optimize_treesa_seeded`] when
/// [`TreeSA::preprocess`] is `false`, and by the preprocessed path to optimize
/// the already-reduced network. Trial `i` seeds its RNG from `seed + i`.
fn optimize_treesa_core_seeded<L: Label>(
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    config: &TreeSA,
    seed: u64,
) -> Option<NestedEinsum<L>> {
    assert!(
        config.surgery_probability.is_finite() && (0.0..=1.0).contains(&config.surgery_probability),
        "surgery probability must be finite and in [0, 1]"
    );
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
    let surgery = (config.surgery_probability > 0.0
        && config.decomposition_type == DecompositionType::Tree)
        .then(|| WaistUpdate::new(&int_ixs, &int_iy, &log2_sizes));

    // Run parallel trials on a pool whose worker stacks are sized for this
    // network (issue #29). The tree walks below — conversion, complexity,
    // and the SA cost model — recurse once per tree level, and a contraction
    // tree can be as deep as it has leaves. Rayon's default worker stack is
    // far smaller than the main thread's, so a deep tree overflows it and the
    // process dies on a signal (SIGBUS/SIGSEGV) with no Rust-level error —
    // observed on circuit networks, where trees are deep. `ntrials == 1` hid
    // the fault because that work runs on the calling thread.
    //
    // The pool is capped at the number of trials — a trial is sequential, so
    // extra workers would idle while still reserving a stack each — and
    // otherwise takes the global pool's width, which is what honors
    // `RAYON_NUM_THREADS`. Note this does not inherit a caller's own custom
    // pool: work that used to run on it now runs here instead.
    //
    // If the pool cannot be built (thread or address-space exhaustion), fall
    // back to the global pool rather than reporting failure: returning `None`
    // would be indistinguishable from "this network has no contraction", and
    // a caller that is merely out of threads should still get an answer.
    let stack_size = trial_stack_size(code.num_tensors());
    let num_threads = config.ntrials.min(rayon::current_num_threads()).max(1);
    let run_trials = || {
        (0..config.ntrials)
            .into_par_iter()
            .map(|trial_idx| {
                // Use thread-local RNG seeded with trial index for reproducibility
                use rand::SeedableRng;
                let mut rng =
                    rand::rngs::SmallRng::seed_from_u64(seed.wrapping_add(trial_idx as u64));

                // Initialize tree
                let tree = match config.initializer {
                    Initializer::Greedy => init_greedy(
                        code, size_dict, &label_map, &int_ixs, &int_iy,
                    )
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

                // Optimize. A configured surgery rule replaces local sweeps at
                // the current beta; it never starts a separate anneal.
                let optimized = if let Some(surgery) = &surgery {
                    optimize_tree_sa_mixed(
                        tree,
                        &log2_sizes,
                        &config.betas,
                        config.niters,
                        &config.score,
                        config.decomposition_type,
                        &mut rng,
                        nedge,
                        surgery,
                        config.surgery_probability,
                    )
                } else {
                    optimize_tree_sa(
                        tree,
                        &log2_sizes,
                        &config.betas,
                        config.niters,
                        &config.score,
                        config.decomposition_type,
                        &mut rng,
                        nedge,
                    )
                };

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
            .collect::<Vec<_>>()
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .stack_size(stack_size)
        .build();
    let results = install_or_run(pool, run_trials);

    // Find best result
    let (best_tree, _) = results
        .into_iter()
        .min_by(|(_, s1), (_, s2)| nan_last(*s1, *s2))?;

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

/// Diagnostics from an [`anneal_surgery_rounds`] run.
#[derive(Debug, Clone, PartialEq)]
pub struct RoundsReport {
    /// Number of interleaved rounds actually executed. Smaller than the
    /// requested `rounds` when the loop stops early (the incumbent tree
    /// degenerated to something that cannot be fine-tuned, e.g. a bare leaf).
    pub rounds_run: u64,
    /// Index of the round during which the returned best tree was produced —
    /// by either that round's surgery step or its fine tuner, whichever won — or
    /// [`u64::MAX`] if no round improved on the seed. It does not imply the
    /// winner was produced by fine tuning; the surgery tree itself can be the
    /// best of the run.
    pub best_round: u64,
    /// Score of the raw fine-tuning endpoint in each executed round. This
    /// diagnostic may increase; the endpoint is retained only if it improves
    /// the round incumbent.
    pub round_scores: Vec<f64>,
    /// Per-round time-complexity trace, including the raw fine-tuning endpoint
    /// and the retained incumbent used by the next round.
    pub round_trace: Vec<RoundTrace>,
    /// Total number of waist-surgery iterations attempted across all rounds
    /// (sum of [`crate::waist_surgery::WaistReport::surgery_calls`]).
    pub surgery_calls_total: u64,
    /// Total span-gated sweeps executed by all cold fine-tuning trials.
    pub fine_tune_sweeps_total: u64,
}

/// Options for [`anneal_refine_rounds`].
///
/// The default exactly reproduces [`anneal_surgery_rounds`]: one historical
/// greedy, root-scoped waist-surgery call precedes each cold fine-tuning pass.
/// Set [`RoundsOptions::surgery`] to `false` for the matched cold-only control.
///
/// # Example
///
/// ```
/// use omeco::treesa::RoundsOptions;
/// use omeco::waist_surgery::{RebuildMode, SurgeryScope};
///
/// assert_eq!(RoundsOptions::default().rebuild, RebuildMode::Greedy);
/// let cold_only = RoundsOptions {
///     surgery: false,
///     scope: SurgeryScope::Local,
///     ..RoundsOptions::default()
/// };
/// assert!(!cold_only.surgery);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct RoundsOptions {
    /// Run the global waist-surgery call at the start of each round.
    pub surgery: bool,
    /// How rebuilt sides are initialized.
    pub rebuild: RebuildMode,
    /// Where surgery operates.
    pub scope: SurgeryScope,
}

impl Default for RoundsOptions {
    fn default() -> Self {
        Self {
            surgery: true,
            rebuild: RebuildMode::default(),
            scope: SurgeryScope::default(),
        }
    }
}

/// One interleaved surgery/fine-tuning round in log2 time-complexity units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoundTrace {
    /// Zero-based round index.
    pub round: u64,
    /// Retained incumbent before surgery.
    pub tc_before: f64,
    /// Configured TreeSA score of the retained incumbent before surgery.
    pub score_before: f64,
    /// Candidate after the surgery step (equal to `tc_before` if rejected).
    pub tc_after_surgery: f64,
    /// Raw endpoint of the cold fine-tuning pass, whether retained or rejected.
    /// The field keeps its historical `anneal` name for artifact compatibility.
    pub tc_after_anneal: f64,
    /// Best of the round incumbent, surgery candidate, and all sweep
    /// checkpoints observed during fine tuning, reported in time-complexity
    /// units. This can rise when the configured multi-objective score improves
    /// by reducing its space or read-write term.
    pub tc_retained: f64,
    /// Configured TreeSA score of the incumbent retained for the next round.
    /// Unlike `tc_retained`, this is guaranteed not to exceed `score_before`.
    pub score_retained: f64,
    /// Whether waist surgery accepted a rebuilt tree in this round.
    pub surgery_accepted: bool,
    /// Exact cut-space comparison made by this round's surgery call.
    pub waist: Option<crate::waist_surgery::WaistCallTrace>,
}

/// Deterministic interleaved anneal–surgery loop (the paper's Algorithm 1).
///
/// Each round applies one waist-surgery iteration
/// ([`crate::waist_surgery::refine_capped`]) to the retained incumbent and
/// then runs a cold fine-tuning pass over the surgical result. Fine tuning uses
/// at most 15 stratified levels from the configured schedule with `β >= 1`, at
/// most 30 span-gated sweeps per coarse-to-fine span, and at most three
/// deterministic serial trials. Custom schedules containing no `β >= 1` are
/// retained verbatim.
///
/// # Incumbent ratchet
///
/// Even cold fine tuning may finish at a worse tree. That endpoint is recorded in
/// [`RoundsReport::round_scores`] and [`RoundsReport::round_trace`], but it does
/// not replace the incumbent unless it improves [`TreeSA::score`]. Round `r + 1`
/// starts from the best configured-score candidate among round `r`'s input,
/// surgery result, and best sweep checkpoint observed during fine tuning. This
/// matches the paper's `Anneal(T, C_outer)` contract: annealing returns its best
/// retained tree. With a multi-objective score, the retained tree's `tc` can
/// rise when another weighted term improves; [`RoundTrace::score_retained`] is
/// the monotone ratchet diagnostic.
///
/// # Never worse, monotone in `rounds`
///
/// The *returned* tree is the best configured-score tree seen anywhere in the
/// run, including the `seed` itself and each round's post-surgery,
/// pre-fine-tuning tree. Consequently the result is never worse than `seed`
/// under [`TreeSA::score`], and — because rounds are chained deterministically —
/// running `n + 1` rounds is always equal or better than running `n` under that
/// same score.
///
/// # Configuration used
///
/// Four fields of `config` are read: [`TreeSA::betas`], [`TreeSA::niters`],
/// [`TreeSA::ntrials`] and [`TreeSA::score`]. Fine tuning caps `niters` at 30
/// and `ntrials` at three. [`TreeSA::preprocess`], [`TreeSA::surgery_iters`]
/// and [`TreeSA::decomposition_type`] are **ignored**: the function takes its
/// seed and network as given and fixes surgery at one iteration per round.
///
/// # Not path-preserving
///
/// The fine-tuning step applies binary-tree moves, and waist surgery
/// rebuilds subtrees in a way that is not path-preserving, so the returned tree
/// is a general (tree) decomposition **regardless of
/// `config.decomposition_type`** — passing a [`TreeSA::path`] config does not
/// yield a path decomposition, and a path-decomposed `seed` will generally come
/// back as a non-path tree. Callers who need the path guarantee must not use
/// this function; [`optimize_treesa`] auto-skips the rounds loop for
/// [`DecompositionType::Path`] configs.
///
/// # Determinism
///
/// Fully reproducible: each fine-tuning trial has a fixed seed derived from its
/// round and trial indices, and surgery is capped by iteration count with a
/// [`std::time::Duration::MAX`] budget, so no wall-clock deadline can ever
/// trigger and results never depend on machine speed. Two runs with the same
/// inputs return byte-identical trees and reports.
///
/// # Cost
///
/// A default fine-tuning trial uses 30 sweeps at each level of a halving span
/// ladder from `ceil(n / 30)` to 2; three serial trials therefore use 450
/// sweeps at `n = 761`, versus 15,000 sweeps for one default cold-start trial.
/// Total time scales linearly in `rounds`.
///
/// # Panics
///
/// Panics if `size_dict` is missing a label appearing in `code` — the same
/// completeness requirement as [`optimize_treesa`].
///
/// # Example
///
/// ```
/// use omeco::treesa::anneal_surgery_rounds;
/// use omeco::{optimize_code, EinCode, GreedyMethod, TreeSA};
/// use std::collections::HashMap;
///
/// let code = EinCode::new(
///     vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
///     vec!['i', 'l'],
/// );
/// let sizes: HashMap<char, usize> = [('i', 4), ('j', 8), ('k', 8), ('l', 4)].into();
/// let seed = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
/// let (tree, report) = anneal_surgery_rounds(&seed, &code, &sizes, &TreeSA::fast(), 2);
/// assert_eq!(tree.leaf_count(), 3);
/// assert_eq!(report.rounds_run, 2);
/// ```
pub fn anneal_surgery_rounds<L: Label>(
    seed: &NestedEinsum<L>,
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    config: &TreeSA,
    rounds: u64,
) -> (NestedEinsum<L>, RoundsReport) {
    anneal_refine_rounds(
        seed,
        code,
        size_dict,
        config,
        rounds,
        &RoundsOptions::default(),
    )
}

/// Deterministic surgery/fine-tuning rounds with opt-in rebuild controls.
///
/// This is the configurable form of [`anneal_surgery_rounds`]. With
/// [`RoundsOptions::default`] the returned tree and report are byte-identical
/// to that historical API. With `opts.surgery == false`, every round runs the
/// same cold fine-tuning trials and ratchet but skips surgery; surgery trace
/// fields remain empty and [`RoundsReport::surgery_calls_total`] remains zero.
///
/// # Example
///
/// ```
/// use omeco::treesa::{anneal_refine_rounds, RoundsOptions};
/// use omeco::{optimize_code, EinCode, GreedyMethod, TreeSA};
/// use std::collections::HashMap;
///
/// let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
/// let sizes: HashMap<char, usize> = [('i', 2), ('j', 4), ('k', 2)].into();
/// let seed = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
/// let opts = RoundsOptions { surgery: false, ..RoundsOptions::default() };
/// let (tree, report) = anneal_refine_rounds(&seed, &code, &sizes, &TreeSA::fast(), 1, &opts);
/// assert_eq!(tree.leaf_count(), 2);
/// assert_eq!(report.surgery_calls_total, 0);
/// ```
pub fn anneal_refine_rounds<L: Label>(
    seed: &NestedEinsum<L>,
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    config: &TreeSA,
    rounds: u64,
    opts: &RoundsOptions,
) -> (NestedEinsum<L>, RoundsReport) {
    use rand::SeedableRng;
    let score_of = |t: &NestedEinsum<L>| {
        let cc = crate::contraction_complexity(t, size_dict, &code.ixs);
        config.score.evaluate(cc.tc, cc.sc, cc.rwc)
    };
    let mut best = seed.clone();
    let mut best_score = score_of(&best);
    let mut trajectory = seed.clone();
    let mut report = RoundsReport {
        rounds_run: 0,
        best_round: u64::MAX,
        round_scores: Vec::new(),
        round_trace: Vec::new(),
        surgery_calls_total: 0,
        fine_tune_sweeps_total: 0,
    };
    let fine_betas = fine_tune_beta_schedule(&config.betas);
    let fine_niters = config.niters.min(30);
    let fine_trials = config.ntrials.clamp(1, 3);
    for r in 0..rounds {
        let tc_before = crate::contraction_complexity(&trajectory, size_dict, &code.ixs).tc;
        let incumbent_score = score_of(&trajectory);
        let (t_surg, wr, waist_trace) = if opts.surgery {
            let surgery_seed = 0x0000_0054_c0ff_ee00_u64.wrapping_add(r);
            if opts.rebuild == RebuildMode::default() && opts.scope == SurgeryScope::default() {
                refine_capped_seeded_with_trace(
                    &trajectory,
                    code,
                    size_dict,
                    std::time::Duration::MAX,
                    1,
                    surgery_seed,
                )
            } else {
                refine_capped_seeded_with_trace_opts(
                    &trajectory,
                    code,
                    size_dict,
                    std::time::Duration::MAX,
                    1,
                    surgery_seed,
                    SurgeryOptions {
                        rebuild: opts.rebuild,
                        scope: opts.scope,
                    },
                )
            }
        } else {
            (
                trajectory.clone(),
                crate::waist_surgery::WaistReport {
                    n_original: code.num_tensors(),
                    surgery_calls: 0,
                    cheaper_cuts: 0,
                    rebuild_attempts: 0,
                    rebuild_accepts: 0,
                    waist_min_hits: 0,
                },
                None,
            )
        };
        report.surgery_calls_total += wr.surgery_calls;
        // The fine-tuning arguments are hoisted into single-line bindings so
        // that the call itself is on one line: coverage instrumentation
        // attributes a multi-line statement to its continuation lines, which
        // then report as unreached even though the statement runs every round.
        // Returning early (rather than breaking) out of the same-shaped match
        // is part of that: the returned tuple is real work on a line a bare
        // `break` would leave without any instructions of its own.
        let ctx = match prepare_warm_anneal(code, size_dict, &t_surg) {
            Some(ctx) => ctx,
            None => return (best, report),
        };
        let (start, log2, nedge) = (ctx.tree, ctx.log2_sizes, ctx.nedge);
        let (betas, niters) = (&fine_betas, fine_niters);
        let emitted_score = |candidate: &ExprTree| {
            let nested = warm_exprtree_to_nested(candidate, code, &ctx.labels);
            score_of(&nested)
        };
        let mut best_fine_tuned: Option<(NestedEinsum<L>, f64, f64, f64)> = None;
        for trial in 0..fine_trials {
            let seed = 0xA55E_u64 + r * fine_trials as u64 + trial as u64;
            let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
            let (fine_tuned, endpoint, sweeps) = fine_tune_tree_sa_counted(
                start.clone(),
                &log2,
                betas,
                niters,
                &emitted_score,
                &mut rng,
                nedge,
            );
            report.fine_tune_sweeps_total += sweeps;
            let candidate = warm_exprtree_to_nested(&fine_tuned, code, &ctx.labels);
            let candidate_score = score_of(&candidate);
            let endpoint = warm_exprtree_to_nested(&endpoint, code, &ctx.labels);
            let endpoint_score = score_of(&endpoint);
            let endpoint_tc = crate::contraction_complexity(&endpoint, size_dict, &code.ixs).tc;
            if best_fine_tuned
                .as_ref()
                .map_or(true, |(_, incumbent, _, _)| candidate_score < *incumbent)
            {
                best_fine_tuned = Some((candidate, candidate_score, endpoint_score, endpoint_tc));
            }
        }
        let Some((cand, cand_score, endpoint_score, tc_after_anneal)) = best_fine_tuned else {
            return (best, report);
        };
        let surg_score = score_of(&t_surg);
        let tc_after_surgery = crate::contraction_complexity(&t_surg, size_dict, &code.ixs).tc;

        let mut retained = trajectory;
        let mut retained_score = incumbent_score;
        if surg_score < retained_score {
            retained_score = surg_score;
            retained = t_surg;
        }
        if cand_score < retained_score {
            retained_score = cand_score;
            retained = cand;
        }
        if retained_score < best_score {
            best_score = retained_score;
            best = retained.clone();
            report.best_round = r;
        }
        report.round_scores.push(endpoint_score);
        report.round_trace.push(RoundTrace {
            round: r,
            tc_before,
            score_before: incumbent_score,
            tc_after_surgery,
            tc_after_anneal,
            tc_retained: crate::contraction_complexity(&retained, size_dict, &code.ixs).tc,
            score_retained: retained_score,
            surgery_accepted: wr.rebuild_accepts > 0,
            waist: waist_trace,
        });
        report.rounds_run = r + 1;
        trajectory = retained;
    }
    (best, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frozen pre-optimization body of [`expr_tree_to_nested_counted`]
    /// (issue #29), kept verbatim as the differential-test oracle: the
    /// left-biased count merge and the `Vec::contains` dedup that the
    /// optimized version replaces. Any output difference is a regression.
    fn expr_tree_to_nested_counted_reference<L: Label>(
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
                let output_labels = info
                    .out_dims
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
                let (left_nested, left_within, left_out) = expr_tree_to_nested_counted_reference(
                    left,
                    original_ixs,
                    inverse_map,
                    open_set,
                    global_count,
                    openedges,
                    level + 1,
                );
                let (right_nested, right_within, right_out) = expr_tree_to_nested_counted_reference(
                    right,
                    original_ixs,
                    inverse_map,
                    open_set,
                    global_count,
                    openedges,
                    level + 1,
                );
                let mut within = left_within;
                for (l, c) in right_within {
                    *within.entry(l).or_insert(0) += c;
                }
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

    /// Reference entry point mirroring [`expr_tree_to_nested`].
    fn expr_tree_to_nested_ref<L: Label>(
        tree: &ExprTree,
        original_ixs: &[Vec<L>],
        inverse_map: &[L],
        openedges: &[L],
    ) -> NestedEinsum<L> {
        let mut global_count: HashMap<L, usize> = HashMap::new();
        for ix in original_ixs {
            for l in ix {
                *global_count.entry(l.clone()).or_insert(0) += 1;
            }
        }
        let open_set: HashSet<L> = openedges.iter().cloned().collect();
        expr_tree_to_nested_counted_reference(
            tree,
            original_ixs,
            inverse_map,
            &open_set,
            &global_count,
            openedges,
            0,
        )
        .0
    }

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
                surgery_probability: 0.0,
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
        assert_eq!(config.surgery_probability, 0.0);
        let fast = TreeSA::fast();
        assert!(fast.preprocess);
        assert_eq!(fast.surgery_iters, 0);
        assert_eq!(fast.surgery_probability, 0.0);
        let tuned = TreeSA::default()
            .with_preprocess(false)
            .with_surgery_iters(30)
            .with_surgery_probability(0.05);
        assert!(!tuned.preprocess);
        assert_eq!(tuned.surgery_iters, 30);
        assert_eq!(tuned.surgery_probability, 0.05);
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

    /// A brick-wall circuit amplitude network: `|0>` boundary tensors, rank-4
    /// two-qubit gates in alternating even/odd layers, `<0|` boundary. All
    /// bonds dimension 2, no open index. Its contraction trees are deep and
    /// expose wide intermediates — the shape absent from every
    /// `benchmarks/graphs` fixture, and the one that exposed issue #29.
    fn brickwall_circuit(nqubits: usize, ngates: usize) -> (EinCode<usize>, HashMap<usize, usize>) {
        let mut ixs: Vec<Vec<usize>> = Vec::new();
        let mut wire: Vec<usize> = (0..nqubits).collect();
        let mut next = nqubits;
        for &w in wire.iter() {
            ixs.push(vec![w]);
        }
        let mut placed = 0;
        let mut layer = 0;
        while placed < ngates {
            let mut i = layer % 2;
            while i + 1 < nqubits && placed < ngates {
                let (a, b) = (next, next + 1);
                next += 2;
                ixs.push(vec![wire[i], wire[i + 1], a, b]);
                wire[i] = a;
                wire[i + 1] = b;
                placed += 1;
                i += 2;
            }
            layer += 1;
        }
        for &w in wire.iter() {
            ixs.push(vec![w]);
        }
        let sizes: HashMap<usize, usize> = (0..next).map(|l| (l, 2)).collect();
        (EinCode::new(ixs, Vec::new()), sizes)
    }

    /// Issue #29: the optimized conversion must be output-identical to the
    /// frozen pre-optimization body on every tree shape, including the deep,
    /// wide-intermediate trees where the two differ in cost.
    #[test]
    fn test_conversion_matches_frozen_reference() {
        use rand::SeedableRng;

        for seed in 0..40u64 {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
            let n_tensors = rng.random_range(4..=14);
            let n_labels = rng.random_range(3..=9);
            let ixs: Vec<Vec<usize>> = (0..n_tensors)
                .map(|_| {
                    let rank = rng.random_range(1..=3);
                    (0..rank).map(|_| rng.random_range(0..n_labels)).collect()
                })
                .collect();
            let n_open = rng.random_range(0..=2);
            let iy: Vec<usize> = (0..n_open).map(|_| rng.random_range(0..n_labels)).collect();
            let code = EinCode::new(ixs, iy);
            let sizes: HashMap<usize, usize> = (0..n_labels).map(|l| (l, 2)).collect();

            let (label_map, labels) = build_label_map(&code);
            let int_ixs = convert_to_int_indices(&code.ixs, &label_map);
            let int_iy: Vec<usize> = code.iy.iter().map(|l| label_map[l]).collect();
            let tree = init_random(
                &int_ixs,
                &int_iy,
                labels.len(),
                DecompositionType::Tree,
                &mut rng,
            );

            let got = expr_tree_to_nested(&tree, &code.ixs, &labels, &code.iy);
            let want = expr_tree_to_nested_ref(&tree, &code.ixs, &labels, &code.iy);
            assert_eq!(
                crate::json::to_json_string(&got).unwrap(),
                crate::json::to_json_string(&want).unwrap(),
                "seed {seed}: optimized conversion diverged from the frozen reference"
            );
            let _ = sizes;
        }
    }

    /// Issue #29: multi-trial optimization of a circuit network completes.
    ///
    /// The reported fault was a worker-thread stack overflow that killed the
    /// process on a signal (exit 138, SIGBUS on macOS) for `ntrials > 1`
    /// while `ntrials == 1` survived, because only the multi-trial path moves
    /// the recursive tree walks onto Rayon workers, whose default stack is
    /// far smaller than the main thread's. Reproducing the overflow itself
    /// needs a ~6500-tensor network and minutes of annealing (recorded in the
    /// issue), and an overflow aborts the whole test binary rather than
    /// failing one case, so this is a fast smoke test over the same code path
    /// rather than a reproduction: it would not have failed before the fix.
    #[test]
    fn test_multi_trial_survives_deep_circuit_trees() {
        let (code, sizes) = brickwall_circuit(201, 2000);
        let config = TreeSA {
            ntrials: 4,
            niters: 1,
            betas: vec![1.0],
            preprocess: false,
            ..TreeSA::default()
        };
        let tree = optimize_treesa(&code, &sizes, &config).expect("must not crash");
        assert_eq!(tree.leaf_count(), code.num_tensors());
    }

    /// The optimizer must still produce a result when a worker pool cannot be
    /// created: "out of threads" is not "this network has no contraction".
    #[test]
    fn test_install_or_run_falls_back_when_the_pool_cannot_be_built() {
        let good = rayon::ThreadPoolBuilder::new().num_threads(2).build();
        assert!(good.is_ok());
        assert_eq!(install_or_run(good, || 7u32), 7);

        let refused = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .spawn_handler(|_| {
                Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "spawn refused",
                ))
            })
            .build();
        assert!(refused.is_err(), "the spawn handler must fail the build");
        assert_eq!(
            install_or_run(refused, || 7u32),
            7,
            "work must still run when the pool cannot be built"
        );
    }

    /// A NaN score must lose regardless of its sign bit, and regardless of
    /// which side of the comparison it is on.
    ///
    /// `f64::total_cmp` alone is not enough: it orders by sign bit, and the
    /// sign of a hardware NaN from `0.0 * inf` differs between x86_64
    /// (negative) and aarch64 (positive), so an overflowed trial would win on
    /// one architecture and lose on the other.
    #[test]
    fn test_nan_scores_always_lose() {
        use std::cmp::Ordering;

        let pos_nan = f64::NAN;
        let neg_nan = -f64::NAN;
        assert!(pos_nan.is_nan() && neg_nan.is_nan());
        assert!(
            neg_nan.is_sign_negative(),
            "need a negative NaN for this test"
        );

        for nan in [pos_nan, neg_nan] {
            for finite in [0.0_f64, -1e30, 1e300, f64::INFINITY] {
                assert_eq!(nan_last(finite, nan), Ordering::Less);
                assert_eq!(nan_last(nan, finite), Ordering::Greater);
            }
            assert_eq!(nan_last(nan, nan), Ordering::Equal);
        }

        // Ordinary scores keep their usual order, so selection is unchanged.
        assert_eq!(nan_last(1.0, 2.0), Ordering::Less);
        assert_eq!(nan_last(2.0, 1.0), Ordering::Greater);
        assert_eq!(nan_last(1.0, 1.0), Ordering::Equal);

        // `min_by` therefore never returns the NaN element.
        let scores = [neg_nan, 5.0_f64, pos_nan, 3.0_f64];
        let best = scores
            .iter()
            .copied()
            .min_by(|a, b| nan_last(*a, *b))
            .unwrap();
        assert_eq!(best, 3.0, "min_by must skip NaN scores of either sign");
    }

    /// The worker stack reservation grows with the network and stays within
    /// its documented bounds.
    #[test]
    fn test_trial_stack_size_bounds() {
        // Small networks sit on the floor; large ones scale; huge ones cap.
        assert_eq!(trial_stack_size(1), 32 * 1024 * 1024);
        assert_eq!(trial_stack_size(6492), 32 * 1024 * 1024);
        assert_eq!(trial_stack_size(100_000), 100_000 * 4 * 1024);
        assert_eq!(trial_stack_size(usize::MAX), 1024 * 1024 * 1024);
        assert!(trial_stack_size(100_000) > trial_stack_size(6492));
    }

    /// The conversion's output dedup must keep a label shared by both
    /// children exactly once. Every contracted bond between two subtrees
    /// appears in both children's outputs, so this exercises the duplicate
    /// path directly, including a shared label that is still needed above and
    /// therefore must survive as an output.
    #[test]
    fn test_conversion_dedups_labels_shared_by_both_children() {
        // Label 0 joins t0 and t1 and also appears in t2 outside the pair, so
        // it is exposed by both children of the (t0, t1) node and must remain
        // an output of that node exactly once.
        let code: EinCode<usize> = EinCode::new(
            vec![vec![0, 1], vec![0, 2], vec![0, 3], vec![1, 2, 3]],
            vec![],
        );
        let sizes: HashMap<usize, usize> = (0..4).map(|l| (l, 2)).collect();
        let (label_map, labels) = build_label_map(&code);
        let int_ixs = convert_to_int_indices(&code.ixs, &label_map);
        let int_iy: Vec<usize> = code.iy.iter().map(|l| label_map[l]).collect();
        use rand::SeedableRng;

        for seed in 0..12u64 {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
            let tree = init_random(
                &int_ixs,
                &int_iy,
                labels.len(),
                DecompositionType::Tree,
                &mut rng,
            );
            let got = expr_tree_to_nested(&tree, &code.ixs, &labels, &code.iy);
            let want = expr_tree_to_nested_ref(&tree, &code.ixs, &labels, &code.iy);
            assert_eq!(
                crate::json::to_json_string(&got).unwrap(),
                crate::json::to_json_string(&want).unwrap()
            );
            // No node may list a label twice in its output.
            fn check_unique(n: &NestedEinsum<usize>) {
                if let NestedEinsum::Node { eins, args } = n {
                    let mut seen = HashSet::new();
                    for l in &eins.iy {
                        assert!(seen.insert(*l), "duplicate label {l} in node output");
                    }
                    for a in args {
                        check_unique(a);
                    }
                }
            }
            check_unique(&got);
        }
        let _ = sizes;
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
    /// fixed and no wall-clock deadline binds `optimize_treesa`'s rounds
    /// loop): two runs of the same config produce byte-identical trees. It is
    /// also never worse than the rounds-off baseline — the loop returns the
    /// best tree it saw, and the baseline tree is the seed it starts from.
    ///
    /// "Never worse" is asserted on the config's own score function, which is
    /// the quantity the loop ranks candidates by; `tc` alone carries no such
    /// guarantee, since a lower-scoring tree may trade `tc` against `sc`.
    #[test]
    fn test_surgery_iters_rounds_deterministic_and_never_worse() {
        // 4x4 periodic grid — a frozen-waist-style instance where surgery acts.
        let code = grid_code(4, 4);
        let sizes: HashMap<usize, usize> =
            code.unique_labels().into_iter().map(|l| (l, 2)).collect();
        let cfg = TreeSA::fast().with_surgery_iters(5);
        let score_of = |t: &NestedEinsum<usize>| {
            let cc = crate::contraction_complexity(t, &sizes, &code.ixs);
            cfg.score.evaluate(cc.tc, cc.sc, cc.rwc)
        };
        let base = optimize_treesa(&code, &sizes, &cfg.clone().with_surgery_iters(0)).unwrap();

        let refined_a = optimize_treesa(&code, &sizes, &cfg).unwrap();
        let refined_b = optimize_treesa(&code, &sizes, &cfg).unwrap();
        assert_eq!(
            format!("{refined_a:?}"),
            format!("{refined_b:?}"),
            "surgery_iters > 0 must be deterministic across runs"
        );

        let (base_score, refined_score) = (score_of(&base), score_of(&refined_a));
        assert!(
            refined_score <= base_score + 1e-9,
            "{refined_score} > {base_score}"
        );
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

    #[test]
    fn test_surgery_update_rule_is_deterministic_and_valid() {
        let code = grid_code(4, 4);
        let sizes: HashMap<usize, usize> =
            code.unique_labels().into_iter().map(|l| (l, 2)).collect();
        let cfg = TreeSA::fast()
            .with_preprocess(false)
            .with_surgery_probability(0.05);
        let a = optimize_treesa(&code, &sizes, &cfg).unwrap();
        let b = optimize_treesa(&code, &sizes, &cfg).unwrap();
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
        assert_eq!(a.leaf_count(), code.num_tensors());
    }

    #[test]
    fn test_zero_probability_is_byte_identical_to_default() {
        let (code, sizes) = load_benchmark_graph("petersen");
        let base = TreeSA::fast();
        let explicit = base.clone().with_surgery_probability(0.0);
        let a = optimize_treesa(&code, &sizes, &base).unwrap();
        let b = optimize_treesa(&code, &sizes, &explicit).unwrap();
        assert_eq!(
            crate::json::to_json_string(&a).unwrap(),
            crate::json::to_json_string(&b).unwrap()
        );
    }

    #[test]
    fn test_surgery_probability_validation() {
        for probability in [-0.01, 1.01, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let result = std::panic::catch_unwind(|| {
                TreeSA::default().with_surgery_probability(probability)
            });
            assert!(
                result.is_err(),
                "accepted invalid probability {probability}"
            );
        }
        assert_eq!(
            TreeSA::default()
                .with_surgery_probability(0.0)
                .surgery_probability,
            0.0
        );
        assert_eq!(
            TreeSA::default()
                .with_surgery_probability(1.0)
                .surgery_probability,
            1.0
        );

        // Public fields allow bypassing the builder, so the optimizer must
        // enforce the same invariant at its boundary.
        let code = EinCode::new(vec![vec![0usize], vec![0]], vec![]);
        let sizes: HashMap<usize, usize> = [(0, 2)].into();
        let mut config = TreeSA::fast();
        config.surgery_probability = -0.5;
        assert!(std::panic::catch_unwind(|| optimize_treesa(&code, &sizes, &config)).is_err());
    }

    #[test]
    fn test_surgery_energy_uses_treesa_log2_scale() {
        let d = surgery_energy_difference(
            (40.0, 10.0, 30.0),
            (41.0, 10.0, 30.0),
            20.0,
            1.0,
            f64::NEG_INFINITY,
        );
        assert_eq!(d, 1.0);
        assert_eq!(
            surgery_energy_difference(
                (41.0, 10.0, 30.0),
                (40.0, 10.0, 30.0),
                20.0,
                1.0,
                f64::NEG_INFINITY,
            ),
            -1.0
        );

        let with_space = surgery_energy_difference(
            (40.0, 21.0, 30.0),
            (40.0, 23.0, 30.0),
            20.0,
            1.5,
            f64::NEG_INFINITY,
        );
        assert_eq!(with_space, 3.0);

        let log2_rw_weight = 0.25_f64.log2();
        let expected = fast_log2sumexp2(41.0, log2_rw_weight + 32.0)
            - fast_log2sumexp2(40.0, log2_rw_weight + 30.0);
        let with_rw = surgery_energy_difference(
            (40.0, 10.0, 30.0),
            (41.0, 10.0, 32.0),
            20.0,
            0.0,
            log2_rw_weight,
        );
        assert!((with_rw - expected).abs() < 1e-12);
    }

    #[test]
    fn test_probability_one_replaces_local_sweep_even_when_no_move_exists() {
        // With every label open, the root is the waist and contains every
        // tensor, so WaistUpdate has no nontrivial bipartition to propose.
        // At p=1 these failed proposal attempts still replace local sweeps;
        // they must not silently fall back to local rewrites.
        let tree = ExprTree::node(
            ExprTree::node(
                ExprTree::leaf(vec![0], 0),
                ExprTree::leaf(vec![1], 1),
                vec![0, 1],
            ),
            ExprTree::node(
                ExprTree::leaf(vec![2], 2),
                ExprTree::leaf(vec![3], 3),
                vec![2, 3],
            ),
            vec![0, 1, 2, 3],
        );
        let before = format!("{tree:?}");
        let surgery = WaistUpdate::new(
            &[vec![0], vec![1], vec![2], vec![3]],
            &[0, 1, 2, 3],
            &[1.0; 4],
        );
        let mut rng = SmallRng::seed_from_u64(11);
        let score = ScoreFunction::default().with_rw_weight(0.25);
        let after = optimize_tree_sa_mixed(
            tree,
            &[1.0; 4],
            &[0.1, 1.0],
            10,
            &score,
            DecompositionType::Tree,
            &mut rng,
            4,
            &surgery,
            1.0,
        );
        assert_eq!(format!("{after:?}"), before);
    }

    #[test]
    fn test_fine_tune_builds_a_coarse_to_fine_span_ladder() {
        let mut tree = ExprTree::leaf(Vec::new(), 0);
        for tensor in 1..61 {
            tree = ExprTree::node(tree, ExprTree::leaf(Vec::new(), tensor), Vec::new());
        }
        let mut rng = SmallRng::seed_from_u64(17);
        let score = |candidate: &ExprTree| candidate.leaf_count() as f64;

        let (best, endpoint) = fine_tune_tree_sa(tree, &[], &[], 0, &score, &mut rng, 0);

        assert_eq!(best.leaf_count(), 61);
        assert_eq!(endpoint.leaf_count(), 61);
    }

    #[test]
    fn test_probability_one_commits_a_successful_surgery_proposal() {
        let code = grid_code(4, 4);
        let sizes: HashMap<usize, usize> =
            code.unique_labels().into_iter().map(|l| (l, 2)).collect();
        let seed_config = TreeSA {
            betas: vec![1.0],
            ntrials: 1,
            niters: 0,
            initializer: Initializer::Random,
            preprocess: false,
            ..TreeSA::fast()
        };
        let seed = optimize_treesa(&code, &sizes, &seed_config).unwrap();
        let ctx = prepare_warm_anneal(&code, &sizes, &seed).unwrap();
        let surgery = WaistUpdate::new(&code.ixs, &code.iy, &ctx.log2_sizes);

        // Find a deterministic RNG stream that yields a proposal after the
        // mixture's Bernoulli draw. At beta=0 every finite-energy proposal is
        // accepted, so the public mixed kernel must return that exact tree.
        for seed in 0..1024u64 {
            let mut expected_rng = SmallRng::seed_from_u64(seed);
            let _mixture_draw = expected_rng.random::<f64>();
            let Some(expected) = surgery.propose(&ctx.tree, &mut expected_rng) else {
                continue;
            };
            let mut actual_rng = SmallRng::seed_from_u64(seed);
            let actual = optimize_tree_sa_mixed(
                ctx.tree.clone(),
                &ctx.log2_sizes,
                &[0.0],
                1,
                &ScoreFunction::default(),
                DecompositionType::Tree,
                &mut actual_rng,
                ctx.nedge,
                &surgery,
                1.0,
            );
            assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
            return;
        }
        panic!("fixture produced no waist-surgery proposal");
    }

    /// With preprocessing enabled, `surgery_iters = k` must run `k`
    /// interleaved anneal–surgery rounds on the reduced graph and splice back
    /// only afterward. This is the paper pipeline; running rounds on the
    /// already-restored graph changes both the waist and FM search space.
    #[test]
    fn test_surgery_iters_runs_interleaved_rounds() {
        // The wrapper must equal simplify -> seed -> rounds -> splice composed
        // explicitly through the public APIs.
        let (code, sizes) = load_benchmark_graph("grid_4x4");
        let base = TreeSA::fast();
        let simplified = simplify(&code, &sizes);
        let reduced_config = TreeSA {
            preprocess: false,
            surgery_iters: 0,
            ..base.clone()
        };
        let seed = optimize_treesa(&simplified.code, &sizes, &reduced_config).unwrap();
        let (reduced, _) = anneal_surgery_rounds(&seed, &simplified.code, &sizes, &base, 2);
        let seed_spliced = splice(&seed, &simplified.subtrees);
        let reduced_spliced = splice(&reduced, &simplified.subtrees);
        let score_of = |tree: &NestedEinsum<usize>| {
            let cc = crate::contraction_complexity(tree, &sizes, &code.ixs);
            base.score.evaluate(cc.tc, cc.sc, cc.rwc)
        };
        let by_hand = if score_of(&reduced_spliced) < score_of(&seed_spliced) {
            reduced_spliced
        } else {
            seed_spliced
        };
        let wrapped = optimize_treesa(
            &code,
            &sizes,
            &TreeSA {
                surgery_iters: 2,
                ..base
            },
        )
        .unwrap();
        assert_eq!(
            crate::json::to_json_string(&wrapped).unwrap(),
            crate::json::to_json_string(&by_hand).unwrap()
        );
    }

    #[test]
    fn test_preprocessed_rounds_never_worse_after_splice_for_custom_score() {
        let (code, sizes) = load_benchmark_graph("grid_4x4");
        let score = ScoreFunction::new(1.0, 4.0, 0.5, 6.0);
        let base_config = TreeSA {
            score,
            ..TreeSA::fast()
        };
        let baseline = optimize_treesa(&code, &sizes, &base_config).unwrap();
        let refined = optimize_treesa(
            &code,
            &sizes,
            &TreeSA {
                surgery_iters: 3,
                ..base_config.clone()
            },
        )
        .unwrap();
        let score_of = |tree: &NestedEinsum<usize>| {
            let cc = crate::contraction_complexity(tree, &sizes, &code.ixs);
            base_config.score.evaluate(cc.tc, cc.sc, cc.rwc)
        };
        assert!(score_of(&refined) <= score_of(&baseline));
    }

    /// The rounds loop is not path-preserving, so `optimize_treesa` must skip
    /// it entirely for `DecompositionType::Path` configs: the result is
    /// byte-identical to the same config with surgery off, and still a path.
    ///
    /// The fixture must be a network the rounds loop would actually change.
    /// The task brief specified `chain_10`, but that makes the test vacuous:
    /// an annealed chain is already optimal, so the loop's best-seen tree is
    /// the seed and the assertion holds even with the guard deleted (verified
    /// by mutation). `grid_4x4` has a waist for surgery to act on — deleting
    /// the `!= Path` guard makes this test fail.
    #[test]
    fn test_path_decomposition_skips_surgery() {
        let (code, sizes) = load_benchmark_graph("grid_4x4");
        let cfg = TreeSA::path()
            .with_surgery_iters(2)
            .with_surgery_probability(0.05);
        let with_surgery = optimize_treesa(&code, &sizes, &cfg).unwrap();
        let without = optimize_treesa(&code, &sizes, &TreeSA::path()).unwrap();
        assert_eq!(
            crate::json::to_json_string(&with_surgery).unwrap(),
            crate::json::to_json_string(&without).unwrap()
        );
        assert!(with_surgery.is_path_decomposition());
    }

    #[test]
    fn test_fine_tune_schedule_skips_hot_restart_and_is_bounded() {
        let dense: Vec<f64> = (0..300).map(|i| 0.01 + 0.05 * i as f64).collect();
        let fine = fine_tune_beta_schedule(&dense);
        assert_eq!(fine.len(), 15);
        assert!(fine.first().is_some_and(|beta| *beta >= 1.0));
        assert_eq!(fine.last(), dense.last());
        assert!(fine.windows(2).all(|pair| pair[0] < pair[1]));

        let custom_hot = vec![0.0, 0.1, 0.9];
        assert_eq!(fine_tune_beta_schedule(&custom_hot), custom_hot);
        assert!(fine_tune_beta_schedule(&[]).is_empty());
    }

    #[test]
    fn test_rounds_options_default_is_byte_identical_to_historical_wrapper() {
        let (code, sizes) = load_benchmark_graph("petersen");
        let config = TreeSA::fast();
        let seed = optimize_treesa(&code, &sizes, &config).unwrap();

        let (historical_tree, historical_report) =
            anneal_surgery_rounds(&seed, &code, &sizes, &config, 2);
        let (default_tree, default_report) =
            anneal_refine_rounds(&seed, &code, &sizes, &config, 2, &RoundsOptions::default());

        assert_eq!(
            crate::json::to_json_string(&default_tree).unwrap(),
            crate::json::to_json_string(&historical_tree).unwrap()
        );
        assert_eq!(default_report, historical_report);
    }

    #[test]
    fn test_seeded_optimizer_preserves_historical_seed_42_bytes() {
        let (code, sizes) = load_benchmark_graph("petersen");
        let config = TreeSA::fast();
        let historical = optimize_treesa(&code, &sizes, &config).unwrap();
        let seeded = optimize_treesa_seeded(&code, &sizes, &config, 42).unwrap();
        assert_eq!(
            crate::json::to_json_string(&seeded).unwrap(),
            crate::json::to_json_string(&historical).unwrap()
        );
    }

    #[test]
    fn test_rounds_report_counts_fine_tune_sweeps() {
        let (code, sizes) = load_benchmark_graph("petersen");
        let config = TreeSA::fast();
        let seed = optimize_treesa(&code, &sizes, &config).unwrap();
        let (_, report) = anneal_refine_rounds(
            &seed,
            &code,
            &sizes,
            &config,
            2,
            &RoundsOptions {
                surgery: false,
                ..RoundsOptions::default()
            },
        );
        assert_eq!(report.fine_tune_sweeps_total, 40);
    }

    #[test]
    fn test_cold_only_round_matches_fine_tuning_path_alone() {
        let (code, sizes) = load_benchmark_graph("petersen");
        let config = TreeSA::fast();
        let seed = optimize_treesa(&code, &sizes, &config).unwrap();
        let score_of = |tree: &NestedEinsum<usize>| {
            let cc = crate::contraction_complexity(tree, &sizes, &code.ixs);
            config.score.evaluate(cc.tc, cc.sc, cc.rwc)
        };

        let ctx = prepare_warm_anneal(&code, &sizes, &seed).unwrap();
        let emitted_score = |candidate: &ExprTree| {
            let nested = warm_exprtree_to_nested(candidate, &code, &ctx.labels);
            score_of(&nested)
        };
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0xA55E);
        let (fine_tuned, endpoint) = fine_tune_tree_sa(
            ctx.tree.clone(),
            &ctx.log2_sizes,
            &fine_tune_beta_schedule(&config.betas),
            config.niters.min(30),
            &emitted_score,
            &mut rng,
            ctx.nedge,
        );
        let candidate = warm_exprtree_to_nested(&fine_tuned, &code, &ctx.labels);
        let endpoint = warm_exprtree_to_nested(&endpoint, &code, &ctx.labels);
        let expected = if score_of(&candidate) < score_of(&seed) {
            candidate
        } else {
            seed.clone()
        };

        let opts = RoundsOptions {
            surgery: false,
            ..RoundsOptions::default()
        };
        let (actual, report) = anneal_refine_rounds(&seed, &code, &sizes, &config, 1, &opts);

        assert_eq!(
            crate::json::to_json_string(&actual).unwrap(),
            crate::json::to_json_string(&expected).unwrap()
        );
        assert_eq!(report.round_scores, vec![score_of(&endpoint)]);
        assert_eq!(report.surgery_calls_total, 0);
        assert_eq!(report.round_trace.len(), 1);
        assert!(!report.round_trace[0].surgery_accepted);
        assert!(report.round_trace[0].waist.is_none());
    }

    #[test]
    fn test_rounds_report_shape_and_determinism() {
        let (code, sizes) = load_benchmark_graph("petersen");
        let config = TreeSA::fast();
        let seed = optimize_treesa(
            &code,
            &sizes,
            &TreeSA {
                surgery_iters: 0,
                ..config.clone()
            },
        )
        .unwrap();
        let (t1, r1) = anneal_surgery_rounds(&seed, &code, &sizes, &config, 3);
        let (t2, r2) = anneal_surgery_rounds(&seed, &code, &sizes, &config, 3);
        assert_eq!(r1.rounds_run, 3);
        assert_eq!(r1.round_scores.len(), 3);
        assert_eq!(r1.round_trace.len(), 3);
        assert!(r1.best_round == u64::MAX || r1.best_round < 3);
        // Determinism: identical trees and traces
        assert_eq!(
            crate::json::to_json_string(&t1).unwrap(),
            crate::json::to_json_string(&t2).unwrap()
        );
        assert_eq!(r1.round_scores, r2.round_scores);
        assert_eq!(r1.round_trace, r2.round_trace);
        assert_eq!(r1.surgery_calls_total, r2.surgery_calls_total);
        assert!(r1.round_trace.iter().all(|trace| trace.waist.is_some()));
        for pair in r1.round_trace.windows(2) {
            assert_eq!(pair[1].tc_before, pair[0].tc_retained);
            assert!(pair[1].score_retained <= pair[1].score_before);
        }
    }

    #[test]
    fn test_rounds_never_worse_and_monotone() {
        let (code, sizes) = load_benchmark_graph("grid_6x6");
        let config = TreeSA::fast();
        let seed = optimize_treesa(
            &code,
            &sizes,
            &TreeSA {
                surgery_iters: 0,
                ..config.clone()
            },
        )
        .unwrap();
        let score_of = |t: &NestedEinsum<usize>| {
            let cc = crate::contraction_complexity(t, &sizes, &code.ixs);
            config.score.evaluate(cc.tc, cc.sc, cc.rwc)
        };
        let s0 = score_of(&seed);
        let (t1, _) = anneal_surgery_rounds(&seed, &code, &sizes, &config, 1);
        let (t3, _) = anneal_surgery_rounds(&seed, &code, &sizes, &config, 3);
        assert!(score_of(&t1) <= s0);
        assert!(score_of(&t3) <= score_of(&t1));
    }

    /// A round's *pre-fine-tuning* surgery tree can be the run's final winner,
    /// not just a waypoint the following fine tuner improves on. The loop must
    /// therefore rank the surgery tree against the incumbent best in its own
    /// right.
    ///
    /// The instance is rigged so that only the surgery tree can win: the seed
    /// is a deliberately bad unannealed random tree, and the round's fine tuner
    /// runs at β = 0, where the Metropolis rule accepts every move — a pure
    /// random walk that leaves the surgery tree worse than it found it. So the
    /// improvement over the seed can only have come from the surgery step, and
    /// the returned tree is byte-identical to one `refine_capped` iteration on
    /// the seed. Everything is seeded, so this is deterministic.
    #[test]
    fn test_rounds_surgery_tree_can_win_the_round() {
        let (code, sizes) = load_benchmark_graph("grid_4x4");
        // Deliberately bad seed: a random tree with the annealing loop disabled.
        let seed_cfg = TreeSA {
            betas: vec![1.0],
            ntrials: 1,
            niters: 0,
            initializer: Initializer::Random,
            preprocess: false,
            ..TreeSA::fast()
        };
        let seed = optimize_treesa(&code, &sizes, &seed_cfg).unwrap();
        // β = 0: every proposed move is accepted, so fine tuning walks away
        // from whatever the surgery step handed it.
        let config = TreeSA {
            betas: vec![0.0],
            ..TreeSA::fast()
        };
        let score_of = |t: &NestedEinsum<usize>| {
            let cc = crate::contraction_complexity(t, &sizes, &code.ixs);
            config.score.evaluate(cc.tc, cc.sc, cc.rwc)
        };

        let (best, report) = anneal_surgery_rounds(&seed, &code, &sizes, &config, 1);
        let (surgical, _) = refine_capped(&seed, &code, &sizes, std::time::Duration::MAX, 1);

        assert_eq!(report.best_round, 0, "the single round must have won");
        assert!(
            score_of(&best) < score_of(&seed),
            "surgery must strictly improve the seed for this fixture to bite: \
             {} vs {}",
            score_of(&best),
            score_of(&seed)
        );
        assert!(
            report.round_scores[0] > score_of(&best),
            "the β = 0 fine tuner must end worse than the surgery tree, leaving it \
             the only candidate that can win: {} vs {}",
            report.round_scores[0],
            score_of(&best)
        );
        assert_eq!(
            crate::json::to_json_string(&best).unwrap(),
            crate::json::to_json_string(&surgical).unwrap(),
            "the returned tree must be the round's surgery tree itself"
        );
    }

    #[test]
    fn test_rounds_bare_leaf_seed() {
        let code: EinCode<usize> = EinCode::new(vec![vec![0, 1]], vec![0, 1]);
        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
        let seed = NestedEinsum::leaf(0);
        let (t, r) = anneal_surgery_rounds(&seed, &code, &sizes, &TreeSA::fast(), 2);
        assert_eq!(r.rounds_run, 0);
        assert!(r.round_scores.is_empty());
        assert!(r.round_trace.is_empty());
        assert_eq!(t.leaf_count(), 1);
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

    #[test]
    fn test_rw_weighted_score_optimizes() {
        let code = EinCode::new(
            vec![
                vec!['a', 'b'],
                vec!['b', 'c'],
                vec!['c', 'd'],
                vec!['d', 'a'],
            ],
            vec![],
        );
        let sizes: HashMap<char, usize> = [('a', 4), ('b', 4), ('c', 4), ('d', 4)].into();
        let score = ScoreFunction {
            rw_weight: 1.0,
            ..ScoreFunction::default()
        };
        let config = TreeSA {
            score,
            ntrials: 1,
            niters: 10,
            ..TreeSA::fast()
        };
        let tree = optimize_treesa(&code, &sizes, &config).unwrap();
        assert_eq!(tree.leaf_count(), 4);
        let cc = crate::contraction_complexity(&tree, &sizes, &code.ixs);
        assert!(cc.rwc.is_finite());
    }

    #[test]
    fn test_rounds_non_default_surgery_options_are_exercised() {
        let (code, sizes) = load_benchmark_graph("petersen");
        let config = TreeSA::fast();
        let seed = optimize_treesa(&code, &sizes, &config).unwrap();
        let seed_score = {
            let cc = crate::contraction_complexity(&seed, &sizes, &code.ixs);
            config.score.evaluate(cc.tc, cc.sc, cc.rwc)
        };
        let opts = RoundsOptions {
            surgery: true,
            rebuild: RebuildMode::WarmRestricted,
            scope: SurgeryScope::Local,
        };

        let (tree, report) = anneal_refine_rounds(&seed, &code, &sizes, &config, 1, &opts);
        let cc = crate::contraction_complexity(&tree, &sizes, &code.ixs);

        assert_eq!(report.rounds_run, 1);
        assert_eq!(tree.leaf_count(), code.num_tensors());
        assert!(config.score.evaluate(cc.tc, cc.sc, cc.rwc) <= seed_score);
    }
}
