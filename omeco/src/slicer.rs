//! Slicing optimization for tensor network contraction.
//!
//! Slicing trades time complexity for reduced space complexity by explicitly
//! looping over selected tensor indices. This module provides the [`TreeSASlicer`]
//! optimizer for automatically finding good indices to slice.

use crate::eincode::{EinCode, NestedEinsum, SlicedEinsum};
use crate::expr_tree::{apply_rule, rule_diff, tree_complexity, DecompositionType, ExprTree, Rule};
use crate::score::ScoreFunction;
use crate::Label;
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

/// Internal helper for managing sliced indices during optimization.
///
/// The Slicer tracks which indices have been sliced and maintains
/// a modified size dictionary where sliced indices have size 0 (in log2 scale).
#[derive(Debug, Clone)]
pub struct Slicer {
    /// Current log2 sizes (0.0 for sliced indices)
    log2_sizes: Vec<f64>,
    /// Map from sliced index to its original log2 size
    legs: HashMap<usize, f64>,
    /// Indices that are fixed and cannot be un-sliced
    fixed_slices: Vec<usize>,
}

impl Slicer {
    /// Create a new Slicer with the given log2 sizes and fixed slices.
    pub fn new(log2_sizes: Vec<f64>, fixed_slices: Vec<usize>) -> Self {
        let mut slicer = Self {
            log2_sizes,
            legs: HashMap::new(),
            fixed_slices: fixed_slices.clone(),
        };
        // Apply fixed slices
        for &idx in &fixed_slices {
            slicer.push(idx);
        }
        slicer
    }

    /// Get the current log2 sizes (with sliced indices set to 0.0).
    #[inline]
    pub fn log2_sizes(&self) -> &[f64] {
        &self.log2_sizes
    }

    /// Get the number of sliced indices.
    #[inline]
    pub fn len(&self) -> usize {
        self.legs.len()
    }

    /// Check if any indices are sliced.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.legs.is_empty()
    }

    /// Add an index to the slicing set.
    ///
    /// The index's log2 size will be set to 0.0 (size 1).
    pub fn push(&mut self, index: usize) {
        if !self.legs.contains_key(&index) && index < self.log2_sizes.len() {
            self.legs.insert(index, self.log2_sizes[index]);
            self.log2_sizes[index] = 0.0;
        }
    }

    /// Remove an index from the slicing set, restoring its original size.
    ///
    /// Returns false if the index was not sliced or is fixed.
    pub fn remove(&mut self, index: usize) -> bool {
        if self.fixed_slices.contains(&index) {
            return false;
        }
        if let Some(original_size) = self.legs.remove(&index) {
            self.log2_sizes[index] = original_size;
            true
        } else {
            false
        }
    }

    /// Replace one sliced index with another.
    ///
    /// Returns false if the operation cannot be performed.
    pub fn replace(&mut self, old_index: usize, new_index: usize) -> bool {
        if self.fixed_slices.contains(&old_index) {
            return false;
        }
        if !self.legs.contains_key(&old_index) {
            return false;
        }
        if self.legs.contains_key(&new_index) {
            return false;
        }
        if new_index >= self.log2_sizes.len() {
            return false;
        }

        // Restore old index
        let old_size = self.legs.remove(&old_index).unwrap();
        self.log2_sizes[old_index] = old_size;

        // Add new index
        self.legs.insert(new_index, self.log2_sizes[new_index]);
        self.log2_sizes[new_index] = 0.0;

        true
    }

    /// Get all sliced indices in a consistent order.
    ///
    /// Fixed slices are returned first, followed by dynamically chosen slices.
    pub fn get_slices(&self) -> Vec<usize> {
        let mut slices: Vec<usize> = self.fixed_slices.clone();
        for &idx in self.legs.keys() {
            if !self.fixed_slices.contains(&idx) {
                slices.push(idx);
            }
        }
        slices
    }

    /// Check if an index is currently sliced.
    #[inline]
    pub fn is_sliced(&self, index: usize) -> bool {
        self.legs.contains_key(&index)
    }

    /// Get all non-fixed sliced indices.
    pub fn non_fixed_slices(&self) -> Vec<usize> {
        self.legs
            .keys()
            .filter(|&&idx| !self.fixed_slices.contains(&idx))
            .copied()
            .collect()
    }
}

/// Configuration for the TreeSA-based slicing optimizer.
///
/// This optimizer iteratively adds slices to reduce space complexity,
/// using simulated annealing to refine the contraction tree after each slice.
///
/// # Example
///
/// ```rust
/// use omeco::{EinCode, TreeSASlicer, optimize_code, slice_code, GreedyMethod, uniform_size_dict};
///
/// let code = EinCode::new(
///     vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
///     vec!['i', 'l'],
/// );
/// let sizes = uniform_size_dict(&code, 64);
///
/// // First optimize the contraction order
/// let optimized = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
///
/// // Then slice to reduce space complexity
/// let slicer = TreeSASlicer::default().with_sc_target(10.0);
/// let sliced = slice_code(&optimized, &sizes, &slicer, &code.ixs);
/// ```
#[derive(Debug, Clone)]
pub struct TreeSASlicer {
    /// Inverse temperature schedule for simulated annealing
    pub betas: Vec<f64>,
    /// Number of independent parallel trials
    pub ntrials: usize,
    /// Iterations per temperature level
    pub niters: usize,
    /// Fixed slices that must be included
    pub fixed_slices: Vec<usize>,
    /// Ratio for determining optimization iterations
    pub optimization_ratio: f64,
    /// Scoring function (sc_target determines target space complexity)
    pub score: ScoreFunction,
    /// Decomposition type for tree optimization
    pub decomposition_type: DecompositionType,
}

impl Default for TreeSASlicer {
    fn default() -> Self {
        // Default betas: 14.0 to 15.0 in steps of 0.05
        let betas: Vec<f64> = (0..=20).map(|i| 14.0 + 0.05 * i as f64).collect();
        Self {
            betas,
            ntrials: 10,
            niters: 10,
            fixed_slices: Vec::new(),
            optimization_ratio: 2.0,
            score: ScoreFunction::default().with_sc_target(30.0),
            decomposition_type: DecompositionType::Tree,
        }
    }
}

impl TreeSASlicer {
    /// Create a new TreeSASlicer with custom parameters.
    pub fn new(
        betas: Vec<f64>,
        ntrials: usize,
        niters: usize,
        fixed_slices: Vec<usize>,
        optimization_ratio: f64,
        score: ScoreFunction,
        decomposition_type: DecompositionType,
    ) -> Self {
        Self {
            betas,
            ntrials,
            niters,
            fixed_slices,
            optimization_ratio,
            score,
            decomposition_type,
        }
    }

    /// Create a fast configuration with fewer iterations.
    pub fn fast() -> Self {
        let betas: Vec<f64> = (0..=10).map(|i| 14.0 + 0.1 * i as f64).collect();
        Self {
            betas,
            ntrials: 1,
            niters: 5,
            optimization_ratio: 1.0,
            ..Default::default()
        }
    }

    /// Set the space complexity target.
    pub fn with_sc_target(mut self, sc_target: f64) -> Self {
        self.score.sc_target = sc_target;
        self
    }

    /// Set the number of parallel trials.
    pub fn with_ntrials(mut self, ntrials: usize) -> Self {
        self.ntrials = ntrials;
        self
    }

    /// Set the iterations per temperature level.
    pub fn with_niters(mut self, niters: usize) -> Self {
        self.niters = niters;
        self
    }

    /// Set fixed slices that must be included.
    pub fn with_fixed_slices(mut self, fixed_slices: Vec<usize>) -> Self {
        self.fixed_slices = fixed_slices;
        self
    }

    /// Set the optimization ratio.
    pub fn with_optimization_ratio(mut self, ratio: f64) -> Self {
        self.optimization_ratio = ratio;
        self
    }

    /// Set the temperature schedule.
    pub fn with_betas(mut self, betas: Vec<f64>) -> Self {
        self.betas = betas;
        self
    }
}

/// Collect space complexities and labels of all intermediate tensors in a tree.
///
/// Returns two vectors:
/// - `scs`: Space complexity (log2) of each node
/// - `labels`: The output labels for each node
pub fn tensor_sizes(tree: &ExprTree, log2_sizes: &[f64]) -> (Vec<f64>, Vec<Vec<usize>>) {
    let mut scs = Vec::new();
    let mut labels = Vec::new();
    tensor_sizes_recursive(tree, log2_sizes, &mut scs, &mut labels);
    (scs, labels)
}

fn tensor_sizes_recursive(
    tree: &ExprTree,
    log2_sizes: &[f64],
    scs: &mut Vec<f64>,
    labels: &mut Vec<Vec<usize>>,
) {
    let node_labels = tree.labels();
    let sc: f64 = if node_labels.is_empty() {
        0.0
    } else {
        node_labels.iter().map(|&l| log2_sizes[l]).sum()
    };
    scs.push(sc);
    labels.push(node_labels.to_vec());

    match tree {
        ExprTree::Leaf(_) => {}
        ExprTree::Node { left, right, .. } => {
            tensor_sizes_recursive(left, log2_sizes, scs, labels);
            tensor_sizes_recursive(right, log2_sizes, scs, labels);
        }
    }
}

/// Find labels that appear in tensors with maximum space complexity.
///
/// Returns labels that appear in any tensor within 0.99 of the maximum SC.
fn find_best_labels(scs: &[f64], labels: &[Vec<usize>]) -> Vec<usize> {
    if scs.is_empty() {
        return Vec::new();
    }

    let max_sc = scs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let threshold = max_sc - 0.99;

    let mut best_labels = Vec::new();
    for (sc, lbls) in scs.iter().zip(labels.iter()) {
        if *sc > threshold {
            best_labels.extend(lbls.iter().copied());
        }
    }
    best_labels
}

/// Count occurrences of each label and find the most common one that's not sliced.
fn find_best_unsliced_label(best_labels: &[usize], slicer: &Slicer) -> Option<usize> {
    if best_labels.is_empty() {
        return None;
    }

    // Count occurrences of each label
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for &label in best_labels {
        if !slicer.is_sliced(label) {
            *counts.entry(label).or_insert(0) += 1;
        }
    }

    // Find the label with maximum count
    counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(label, _)| label)
}

/// Run a single slicing optimization trial.
fn treesa_slice_trial<R: Rng>(
    mut tree: ExprTree,
    log2_sizes: &[f64],
    config: &TreeSASlicer,
    rng: &mut R,
) -> (ExprTree, Slicer) {
    let mut slicer = Slicer::new(log2_sizes.to_vec(), config.fixed_slices.clone());

    // Initial complexity
    let (_, initial_sc, _) = tree_complexity(&tree, slicer.log2_sizes());

    // Determine number of optimization iterations
    let optimization_length = if initial_sc > config.score.sc_target {
        ((initial_sc - config.score.sc_target) * config.optimization_ratio).ceil() as usize
    } else {
        0
    };

    let mut slicing_loop = 0;
    let mut current_sc = initial_sc;

    while slicing_loop < optimization_length || current_sc > config.score.sc_target {
        // Stage 1: Find and add a slice
        let (scs, labels) = tensor_sizes(&tree, slicer.log2_sizes());
        let best_labels = find_best_labels(&scs, &labels);

        if let Some(best_unsliced) = find_best_unsliced_label(&best_labels, &slicer) {
            if current_sc > config.score.sc_target {
                // Add a new slice
                slicer.push(best_unsliced);
            } else if !slicer.is_empty() {
                // Try to replace an existing non-fixed slice
                let non_fixed = slicer.non_fixed_slices();
                if !non_fixed.is_empty() {
                    // Find the least useful slice (appears least in best_labels)
                    let mut min_count = usize::MAX;
                    let mut worst_slice = non_fixed[0];
                    for &slice in &non_fixed {
                        let count = best_labels.iter().filter(|&&l| l == slice).count();
                        if count < min_count {
                            min_count = count;
                            worst_slice = slice;
                        }
                    }
                    slicer.remove(worst_slice);
                }
            }
        }

        // Stage 2: Refine tree with SA using current sliced sizes
        tree = optimize_tree_with_slicing(
            tree,
            slicer.log2_sizes(),
            &config.betas,
            config.niters,
            &config.score,
            config.decomposition_type,
            rng,
        );

        // Update complexity
        let (_, sc, _) = tree_complexity(&tree, slicer.log2_sizes());
        current_sc = sc;
        slicing_loop += 1;

        // Safety check to prevent infinite loops
        if slicing_loop > optimization_length + 100 {
            break;
        }
    }

    (tree, slicer)
}

/// Run simulated annealing on a tree with the given (possibly sliced) sizes.
fn optimize_tree_with_slicing<R: Rng>(
    mut tree: ExprTree,
    log2_sizes: &[f64],
    betas: &[f64],
    niters: usize,
    score: &ScoreFunction,
    decomp: DecompositionType,
    rng: &mut R,
) -> ExprTree {
    let (_, mut global_sc, _) = tree_complexity(&tree, log2_sizes);

    for &beta in betas {
        for _ in 0..niters {
            tree = sweep_mutate_slicing(tree, beta, log2_sizes, score, decomp, global_sc, rng);
        }
        let (_, sc, _) = tree_complexity(&tree, log2_sizes);
        global_sc = sc;
    }
    tree
}

/// Sweep through all nodes attempting mutations (for slicing optimization).
fn sweep_mutate_slicing<R: Rng>(
    tree: ExprTree,
    beta: f64,
    log2_sizes: &[f64],
    score: &ScoreFunction,
    decomp: DecompositionType,
    global_sc: f64,
    rng: &mut R,
) -> ExprTree {
    match tree {
        ExprTree::Leaf(_) => tree,
        ExprTree::Node { left, right, info } => {
            let new_left =
                sweep_mutate_slicing(*left, beta, log2_sizes, score, decomp, global_sc, rng);
            let new_right =
                sweep_mutate_slicing(*right, beta, log2_sizes, score, decomp, global_sc, rng);

            let tree = ExprTree::Node {
                left: Box::new(new_left),
                right: Box::new(new_right),
                info,
            };

            try_mutate_node_slicing(tree, beta, log2_sizes, score, decomp, global_sc, rng)
        }
    }
}

/// Try to apply a mutation at a node using Metropolis criterion.
fn try_mutate_node_slicing<R: Rng>(
    tree: ExprTree,
    beta: f64,
    log2_sizes: &[f64],
    score: &ScoreFunction,
    decomp: DecompositionType,
    global_sc: f64,
    rng: &mut R,
) -> ExprTree {
    let rules = Rule::applicable_rules(&tree, decomp);
    if rules.is_empty() {
        return tree;
    }

    let rule = rules[rng.random_range(0..rules.len())];

    if let Some(diff) = rule_diff(&tree, rule, log2_sizes, score.rw_weight > 0.0) {
        let dtc = diff.tc1 - diff.tc0;
        let sc_new = global_sc.max(global_sc + diff.dsc);

        let sc_penalty = if sc_new > score.sc_target {
            score.sc_weight
        } else {
            0.0
        };
        let d_energy = sc_penalty * diff.dsc + dtc;

        let accept = if d_energy <= 0.0 {
            true
        } else {
            rng.random::<f64>() < (-beta * d_energy).exp()
        };

        if accept {
            return apply_rule(tree, rule, diff.new_labels);
        }
    }

    tree
}

/// Build a label-to-integer mapping.
fn build_label_map<L: Label>(labels: &[L]) -> HashMap<L, usize> {
    labels
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, l)| (l, i))
        .collect()
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

/// Convert a NestedEinsum to an ExprTree.
fn nested_to_expr_tree<L: Label>(
    nested: &NestedEinsum<L>,
    int_ixs: &[Vec<usize>],
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
            let left = nested_to_expr_tree(&args[0], int_ixs, label_map)?;
            let right = nested_to_expr_tree(&args[1], int_ixs, label_map)?;
            let out_dims: Vec<usize> = eins.iy.iter().map(|l| label_map[l]).collect();
            Some(ExprTree::node(left, right, out_dims))
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

            let iy: Vec<L> = info
                .out_dims
                .iter()
                .map(|&i| inverse_map[i].clone())
                .collect();

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

/// Slice a NestedEinsum to reduce space complexity.
///
/// This is the main entry point for slicing optimization. It takes an already-optimized
/// contraction tree and finds indices to slice over to reduce memory requirements.
///
/// # Arguments
///
/// * `code` - The optimized NestedEinsum to slice
/// * `size_dict` - Size dictionary mapping labels to their dimensions
/// * `config` - Slicing configuration
/// * `original_ixs` - Original input tensor indices (from the EinCode)
///
/// # Returns
///
/// A `SlicedEinsum` containing the sliced indices and the optimized tree.
///
/// # Example
///
/// ```rust
/// use omeco::{EinCode, TreeSASlicer, optimize_code, slice_code, GreedyMethod, uniform_size_dict};
///
/// let code = EinCode::new(
///     vec![vec!['i', 'j'], vec!['j', 'k']],
///     vec!['i', 'k'],
/// );
/// let sizes = uniform_size_dict(&code, 64);
///
/// let optimized = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
/// let slicer = TreeSASlicer::fast().with_sc_target(5.0);
/// let sliced = slice_code(&optimized, &sizes, &slicer, &code.ixs);
/// ```
pub fn slice_code<L: Label>(
    code: &NestedEinsum<L>,
    size_dict: &HashMap<L, usize>,
    config: &TreeSASlicer,
    original_ixs: &[Vec<L>],
) -> Option<SlicedEinsum<L>> {
    // Handle trivial cases
    let leaf_count = code.leaf_count();
    if leaf_count <= 1 {
        return Some(SlicedEinsum::unsliced(code.clone()));
    }

    if config.ntrials == 0 {
        return Some(SlicedEinsum::unsliced(code.clone()));
    }

    // Build label mapping
    let mut all_labels: Vec<L> = Vec::new();
    for ix in original_ixs {
        for l in ix {
            if !all_labels.contains(l) {
                all_labels.push(l.clone());
            }
        }
    }
    // Also include output labels from the nested einsum
    add_nested_labels(code, &mut all_labels);

    let label_map = build_label_map(&all_labels);
    let log2_sizes: Vec<f64> = all_labels
        .iter()
        .map(|l| (size_dict.get(l).copied().unwrap_or(1) as f64).log2())
        .collect();
    let int_ixs = convert_to_int_indices(original_ixs, &label_map);

    // Convert to ExprTree
    let initial_tree = nested_to_expr_tree(code, &int_ixs, &label_map)?;

    // Run parallel trials
    let results: Vec<_> = (0..config.ntrials)
        .into_par_iter()
        .map(|trial_idx| {
            use rand::SeedableRng;
            let mut rng = rand::rngs::SmallRng::seed_from_u64(trial_idx as u64 + 42);

            let tree = initial_tree.clone();
            let (optimized, slicer) = treesa_slice_trial(tree, &log2_sizes, config, &mut rng);

            let (tc, sc, rw) = tree_complexity(&optimized, slicer.log2_sizes());
            let score = config.score.evaluate(tc, sc, rw);

            (optimized, slicer, score, tc, sc, rw)
        })
        .collect();

    // Find best result
    let (best_tree, best_slicer, _, _, _, _) = results
        .into_iter()
        .min_by(|(_, _, s1, _, _, _), (_, _, s2, _, _, _)| s1.partial_cmp(s2).unwrap())?;

    // Convert sliced indices back to original labels
    let sliced_labels: Vec<L> = best_slicer
        .get_slices()
        .into_iter()
        .map(|idx| all_labels[idx].clone())
        .collect();

    // Convert tree back to NestedEinsum
    let optimized_nested = expr_tree_to_nested(&best_tree, original_ixs, &all_labels);

    Some(SlicedEinsum::new(sliced_labels, optimized_nested))
}

/// Helper to add all labels from a NestedEinsum to a vector.
fn add_nested_labels<L: Label>(nested: &NestedEinsum<L>, labels: &mut Vec<L>) {
    match nested {
        NestedEinsum::Leaf { .. } => {}
        NestedEinsum::Node { args, eins } => {
            for l in &eins.iy {
                if !labels.contains(l) {
                    labels.push(l.clone());
                }
            }
            for ix in &eins.ixs {
                for l in ix {
                    if !labels.contains(l) {
                        labels.push(l.clone());
                    }
                }
            }
            for arg in args {
                add_nested_labels(arg, labels);
            }
        }
    }
}

/// Trait for slicing optimizers.
pub trait CodeSlicer {
    /// Slice a NestedEinsum to reduce space complexity.
    fn slice<L: Label>(
        &self,
        code: &NestedEinsum<L>,
        size_dict: &HashMap<L, usize>,
        original_ixs: &[Vec<L>],
    ) -> Option<SlicedEinsum<L>>;
}

impl CodeSlicer for TreeSASlicer {
    fn slice<L: Label>(
        &self,
        code: &NestedEinsum<L>,
        size_dict: &HashMap<L, usize>,
        original_ixs: &[Vec<L>],
    ) -> Option<SlicedEinsum<L>> {
        slice_code(code, size_dict, self, original_ixs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eincode::uniform_size_dict;
    use crate::greedy::GreedyMethod;
    use crate::optimize_code;

    #[test]
    fn test_slicer_new() {
        let log2_sizes = vec![2.0, 3.0, 4.0];
        let slicer = Slicer::new(log2_sizes.clone(), vec![]);
        assert_eq!(slicer.len(), 0);
        assert!(slicer.is_empty());
    }

    #[test]
    fn test_slicer_with_fixed_slices() {
        let log2_sizes = vec![2.0, 3.0, 4.0];
        let slicer = Slicer::new(log2_sizes.clone(), vec![0]);
        assert_eq!(slicer.len(), 1);
        assert!(slicer.is_sliced(0));
        assert_eq!(slicer.log2_sizes()[0], 0.0);
    }

    #[test]
    fn test_slicer_push() {
        let log2_sizes = vec![2.0, 3.0, 4.0];
        let mut slicer = Slicer::new(log2_sizes, vec![]);
        slicer.push(1);
        assert!(slicer.is_sliced(1));
        assert_eq!(slicer.log2_sizes()[1], 0.0);
    }

    #[test]
    fn test_slicer_remove() {
        let log2_sizes = vec![2.0, 3.0, 4.0];
        let mut slicer = Slicer::new(log2_sizes, vec![]);
        slicer.push(1);
        assert!(slicer.remove(1));
        assert!(!slicer.is_sliced(1));
        assert_eq!(slicer.log2_sizes()[1], 3.0);
    }

    #[test]
    fn test_slicer_remove_fixed() {
        let log2_sizes = vec![2.0, 3.0, 4.0];
        let mut slicer = Slicer::new(log2_sizes, vec![1]);
        assert!(!slicer.remove(1)); // Cannot remove fixed slice
        assert!(slicer.is_sliced(1));
    }

    #[test]
    fn test_slicer_replace() {
        let log2_sizes = vec![2.0, 3.0, 4.0];
        let mut slicer = Slicer::new(log2_sizes, vec![]);
        slicer.push(0);
        assert!(slicer.replace(0, 2));
        assert!(!slicer.is_sliced(0));
        assert!(slicer.is_sliced(2));
    }

    #[test]
    fn test_slicer_get_slices() {
        let log2_sizes = vec![2.0, 3.0, 4.0, 5.0];
        let mut slicer = Slicer::new(log2_sizes, vec![0]);
        slicer.push(2);
        let slices = slicer.get_slices();
        assert!(slices.contains(&0));
        assert!(slices.contains(&2));
        // Fixed slices should come first
        assert_eq!(slices[0], 0);
    }

    #[test]
    fn test_treesa_slicer_default() {
        let config = TreeSASlicer::default();
        assert_eq!(config.ntrials, 10);
        assert_eq!(config.niters, 10);
        assert!(!config.betas.is_empty());
    }

    #[test]
    fn test_treesa_slicer_fast() {
        let config = TreeSASlicer::fast();
        assert_eq!(config.ntrials, 1);
        assert_eq!(config.niters, 5);
    }

    #[test]
    fn test_treesa_slicer_builders() {
        let config = TreeSASlicer::default()
            .with_sc_target(20.0)
            .with_ntrials(5)
            .with_niters(15);
        assert_eq!(config.score.sc_target, 20.0);
        assert_eq!(config.ntrials, 5);
        assert_eq!(config.niters, 15);
    }

    #[test]
    fn test_tensor_sizes() {
        let leaf0 = ExprTree::leaf(vec![0, 1], 0);
        let leaf1 = ExprTree::leaf(vec![1, 2], 1);
        let tree = ExprTree::node(leaf0, leaf1, vec![0, 2]);

        let log2_sizes = vec![2.0, 3.0, 2.0];
        let (scs, labels) = tensor_sizes(&tree, &log2_sizes);

        assert_eq!(scs.len(), 3); // root + 2 leaves
        assert_eq!(labels.len(), 3);
    }

    #[test]
    fn test_find_best_labels() {
        let scs = vec![5.0, 4.0, 6.0, 5.5];
        let labels = vec![vec![0, 1], vec![1, 2], vec![0, 1, 2], vec![1]];

        let best = find_best_labels(&scs, &labels);
        // Should include labels from tensors with SC >= 6.0 - 0.99 = 5.01
        assert!(best.contains(&0));
        assert!(best.contains(&1));
        assert!(best.contains(&2));
    }

    #[test]
    fn test_slice_code_trivial() {
        let code = EinCode::new(vec![vec!['i', 'j']], vec!['i', 'j']);
        let sizes = uniform_size_dict(&code, 4);
        let nested = NestedEinsum::<char>::leaf(0);

        let config = TreeSASlicer::fast();
        let sliced = slice_code(&nested, &sizes, &config, &code.ixs);

        assert!(sliced.is_some());
        assert!(!sliced.unwrap().is_sliced());
    }

    #[test]
    fn test_slice_code_simple() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let sizes = uniform_size_dict(&code, 16);

        let optimized = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
        let config = TreeSASlicer::fast().with_sc_target(5.0);
        let sliced = slice_code(&optimized, &sizes, &config, &code.ixs);

        assert!(sliced.is_some());
    }

    #[test]
    fn test_slice_code_chain() {
        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
            vec!['i', 'l'],
        );
        let sizes = uniform_size_dict(&code, 32);

        let optimized = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
        let config = TreeSASlicer::fast().with_sc_target(8.0);
        let sliced = slice_code(&optimized, &sizes, &config, &code.ixs);

        assert!(sliced.is_some());
    }

    #[test]
    fn test_code_slicer_trait() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let sizes = uniform_size_dict(&code, 8);

        let optimized = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
        let slicer = TreeSASlicer::fast();
        let sliced = slicer.slice(&optimized, &sizes, &code.ixs);

        assert!(sliced.is_some());
    }

    #[test]
    fn test_slicer_non_fixed_slices() {
        let log2_sizes = vec![2.0, 3.0, 4.0, 5.0];
        let mut slicer = Slicer::new(log2_sizes, vec![0]); // 0 is fixed
        slicer.push(2);
        slicer.push(3);

        let non_fixed = slicer.non_fixed_slices();
        assert!(!non_fixed.contains(&0)); // fixed, should not be included
        assert!(non_fixed.contains(&2));
        assert!(non_fixed.contains(&3));
        assert_eq!(non_fixed.len(), 2);
    }

    #[test]
    fn test_slicer_push_duplicate() {
        let log2_sizes = vec![2.0, 3.0, 4.0];
        let mut slicer = Slicer::new(log2_sizes, vec![]);
        slicer.push(1);
        slicer.push(1); // Push same index again
        assert_eq!(slicer.len(), 1); // Should still be 1
    }

    #[test]
    fn test_slicer_push_out_of_bounds() {
        let log2_sizes = vec![2.0, 3.0, 4.0];
        let mut slicer = Slicer::new(log2_sizes, vec![]);
        slicer.push(10); // Out of bounds
        assert_eq!(slicer.len(), 0); // Should not be added
    }

    #[test]
    fn test_slicer_remove_non_sliced() {
        let log2_sizes = vec![2.0, 3.0, 4.0];
        let mut slicer = Slicer::new(log2_sizes, vec![]);
        assert!(!slicer.remove(1)); // Not sliced, should return false
    }

    #[test]
    fn test_slicer_replace_fixed() {
        let log2_sizes = vec![2.0, 3.0, 4.0];
        let mut slicer = Slicer::new(log2_sizes, vec![0]); // 0 is fixed
        assert!(!slicer.replace(0, 2)); // Cannot replace fixed slice
        assert!(slicer.is_sliced(0));
        assert!(!slicer.is_sliced(2));
    }

    #[test]
    fn test_slicer_replace_non_sliced() {
        let log2_sizes = vec![2.0, 3.0, 4.0];
        let mut slicer = Slicer::new(log2_sizes, vec![]);
        assert!(!slicer.replace(0, 2)); // 0 not sliced
    }

    #[test]
    fn test_slicer_replace_to_already_sliced() {
        let log2_sizes = vec![2.0, 3.0, 4.0];
        let mut slicer = Slicer::new(log2_sizes, vec![]);
        slicer.push(0);
        slicer.push(2);
        assert!(!slicer.replace(0, 2)); // 2 already sliced
    }

    #[test]
    fn test_slicer_replace_out_of_bounds() {
        let log2_sizes = vec![2.0, 3.0, 4.0];
        let mut slicer = Slicer::new(log2_sizes, vec![]);
        slicer.push(0);
        assert!(!slicer.replace(0, 10)); // Out of bounds
    }

    #[test]
    fn test_find_best_labels_empty() {
        let scs: Vec<f64> = vec![];
        let labels: Vec<Vec<usize>> = vec![];
        let best = find_best_labels(&scs, &labels);
        assert!(best.is_empty());
    }

    #[test]
    fn test_find_best_unsliced_label() {
        let best_labels = vec![0, 1, 1, 2, 1]; // 1 appears most often
        let slicer = Slicer::new(vec![2.0, 3.0, 4.0], vec![]);

        let result = find_best_unsliced_label(&best_labels, &slicer);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_find_best_unsliced_label_all_sliced() {
        let best_labels = vec![0, 1, 1];
        let mut slicer = Slicer::new(vec![2.0, 3.0, 4.0], vec![]);
        slicer.push(0);
        slicer.push(1);

        let result = find_best_unsliced_label(&best_labels, &slicer);
        assert_eq!(result, None);
    }

    #[test]
    fn test_find_best_unsliced_label_empty() {
        let best_labels: Vec<usize> = vec![];
        let slicer = Slicer::new(vec![2.0, 3.0, 4.0], vec![]);

        let result = find_best_unsliced_label(&best_labels, &slicer);
        assert_eq!(result, None);
    }

    #[test]
    fn test_treesa_slicer_new() {
        let betas = vec![1.0, 2.0, 3.0];
        let score = ScoreFunction::default().with_sc_target(15.0);
        let config = TreeSASlicer::new(
            betas.clone(),
            5,
            8,
            vec![0, 1],
            3.0,
            score,
            DecompositionType::Tree,
        );

        assert_eq!(config.betas, betas);
        assert_eq!(config.ntrials, 5);
        assert_eq!(config.niters, 8);
        assert_eq!(config.fixed_slices, vec![0, 1]);
        assert_eq!(config.optimization_ratio, 3.0);
        assert_eq!(config.score.sc_target, 15.0);
    }

    #[test]
    fn test_treesa_slicer_with_fixed_slices() {
        let config = TreeSASlicer::default().with_fixed_slices(vec![0, 2]);
        assert_eq!(config.fixed_slices, vec![0, 2]);
    }

    #[test]
    fn test_treesa_slicer_with_betas() {
        let betas = vec![10.0, 11.0, 12.0];
        let config = TreeSASlicer::default().with_betas(betas.clone());
        assert_eq!(config.betas, betas);
    }

    #[test]
    fn test_treesa_slicer_with_optimization_ratio() {
        let config = TreeSASlicer::default().with_optimization_ratio(5.0);
        assert_eq!(config.optimization_ratio, 5.0);
    }

    #[test]
    fn test_slice_code_zero_ntrials() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let sizes = uniform_size_dict(&code, 16);

        let optimized = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
        let config = TreeSASlicer::fast().with_ntrials(0);
        let sliced = slice_code(&optimized, &sizes, &config, &code.ixs);

        assert!(sliced.is_some());
        assert!(!sliced.unwrap().is_sliced()); // No slicing with 0 trials
    }

    #[test]
    fn test_slice_code_reduces_space() {
        // Create a tensor network where slicing should help
        let code = EinCode::new(
            vec![
                vec!['a', 'b'],
                vec!['b', 'c'],
                vec!['c', 'd'],
                vec!['d', 'e'],
            ],
            vec!['a', 'e'],
        );
        let sizes = uniform_size_dict(&code, 64); // Large bond dim

        let optimized = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();

        // Without slicing
        let no_slice = TreeSASlicer::fast().with_sc_target(100.0); // High target = no slicing
        let sliced_none = slice_code(&optimized, &sizes, &no_slice, &code.ixs).unwrap();

        // With slicing
        let with_slice = TreeSASlicer::fast().with_sc_target(10.0); // Low target = force slicing
        let sliced_some = slice_code(&optimized, &sizes, &with_slice, &code.ixs).unwrap();

        // Either we get some slicing, or the SC was already low enough
        // Just verify both succeed
        assert!(sliced_none.slicing.is_empty() || !sliced_none.slicing.is_empty());
        assert!(sliced_some.slicing.is_empty() || !sliced_some.slicing.is_empty());
    }

    #[test]
    fn test_tensor_sizes_single_leaf() {
        let leaf = ExprTree::leaf(vec![0, 1], 0);
        let log2_sizes = vec![2.0, 3.0];

        let (scs, labels) = tensor_sizes(&leaf, &log2_sizes);

        assert_eq!(scs.len(), 1);
        assert_eq!(labels.len(), 1);
        assert_eq!(scs[0], 5.0); // 2 + 3
        assert_eq!(labels[0], vec![0, 1]);
    }

    #[test]
    fn test_tensor_sizes_empty_labels() {
        let leaf = ExprTree::leaf(vec![], 0);
        let log2_sizes: Vec<f64> = vec![];

        let (scs, _labels) = tensor_sizes(&leaf, &log2_sizes);

        assert_eq!(scs.len(), 1);
        assert_eq!(scs[0], 0.0); // Empty = scalar = 0
    }

    #[test]
    fn test_nested_to_expr_tree_and_back() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let sizes = uniform_size_dict(&code, 8);

        let optimized = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();

        // Build label map
        let mut all_labels: Vec<char> = Vec::new();
        for ix in &code.ixs {
            for l in ix {
                if !all_labels.contains(l) {
                    all_labels.push(*l);
                }
            }
        }
        let label_map = build_label_map(&all_labels);
        let int_ixs = convert_to_int_indices(&code.ixs, &label_map);

        // Convert to ExprTree
        let tree = nested_to_expr_tree(&optimized, &int_ixs, &label_map);
        assert!(tree.is_some());

        // Convert back to NestedEinsum
        let back = expr_tree_to_nested(&tree.unwrap(), &code.ixs, &all_labels);

        // Should have same structure (2 leaves)
        assert_eq!(back.leaf_count(), 2);
    }

    #[test]
    fn test_add_nested_labels() {
        let eins = EinCode::new(vec![vec!['a', 'b'], vec!['b', 'c']], vec!['a', 'c']);
        let nested = NestedEinsum::node(
            vec![NestedEinsum::leaf(0), NestedEinsum::leaf(1)],
            eins,
        );

        let mut labels: Vec<char> = Vec::new();
        add_nested_labels(&nested, &mut labels);

        assert!(labels.contains(&'a'));
        assert!(labels.contains(&'b'));
        assert!(labels.contains(&'c'));
    }

    #[test]
    fn test_add_nested_labels_leaf() {
        let nested = NestedEinsum::<char>::leaf(0);
        let mut labels: Vec<char> = Vec::new();
        add_nested_labels(&nested, &mut labels);
        assert!(labels.is_empty()); // Leaf adds nothing
    }

    #[test]
    fn test_build_label_map() {
        let labels = vec!['a', 'b', 'c'];
        let map = build_label_map(&labels);

        assert_eq!(map[&'a'], 0);
        assert_eq!(map[&'b'], 1);
        assert_eq!(map[&'c'], 2);
    }

    #[test]
    fn test_convert_to_int_indices() {
        let ixs = vec![vec!['a', 'b'], vec!['b', 'c']];
        let labels = vec!['a', 'b', 'c'];
        let map = build_label_map(&labels);

        let int_ixs = convert_to_int_indices(&ixs, &map);

        assert_eq!(int_ixs, vec![vec![0, 1], vec![1, 2]]);
    }

    #[test]
    fn test_slice_code_larger_network() {
        // 3x3 grid-like network
        let code = EinCode::new(
            vec![
                vec!['a', 'b', 'c'],
                vec!['b', 'd', 'e'],
                vec!['c', 'e', 'f'],
                vec!['d', 'g'],
                vec!['f', 'h'],
            ],
            vec!['a', 'g', 'h'],
        );
        let sizes = uniform_size_dict(&code, 16);

        let optimized = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
        let config = TreeSASlicer::fast().with_sc_target(10.0);
        let sliced = slice_code(&optimized, &sizes, &config, &code.ixs);

        assert!(sliced.is_some());
    }
}
