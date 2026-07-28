//! Structural simplification front-end for tensor-network contraction.
//!
//! Raw tensor networks from quantum circuits carry large amounts of *locally
//! reducible* structure: chains of single-qubit gates, rank-1 boundary tensors,
//! and other neighbours whose merge cannot grow an intermediate. Optimizing the
//! full network directly wastes the search budget on this structure. This module
//! collapses it up-front with a deterministic, size-safe pass, optimizes the
//! much smaller *reduced* network, then splices the collapsed structure back so
//! the returned tree's leaves are exactly the original tensor indices.
//!
//! # The pass
//!
//! [`simplify`] repeatedly contracts a neighbouring pair of (super-)tensors when
//! the surviving intermediate is both **rank-non-increasing** and
//! **size-non-increasing** relative to the larger input. A label survives a merge
//! iff it is an output label or still occurs in some *other* live tensor. The
//! dimension-aware size check matters for heterogeneous networks: equal rank does
//! not imply equal storage. The pass deliberately makes no claim that the local
//! contraction FLOP count is bounded by an input's storage size.
//!
//! On raw Sycamore circuits this shrinks the network by ~89% (e.g. 3369 → ~381
//! tensors); on graphs with no reducible local structure (e.g. 3-regular graphs)
//! it is a near no-op.
//!
//! # Splicing back
//!
//! Each surviving super-tensor carries its internal binary contraction subtree
//! over the *original* tensor indices. [`splice`] replaces each reduced-network
//! leaf by its super-tensor's subtree, yielding a tree whose leaves are a
//! permutation of `0..n_original`. The reduced-network time complexity excludes
//! the simplification's own internal cost. Use [`crate::contraction_complexity`]
//! on the spliced tree when the exact end-to-end cost is required.
//!
//! # Example
//!
//! ```
//! use omeco::preprocess::simplify_then_optimize;
//! use omeco::{EinCode, GreedyMethod};
//! use std::collections::HashMap;
//!
//! // A chain of matrices: b--c--d--e are all internal, degree-2 tensors that the
//! // rank-non-increasing pass fuses away before optimizing.
//! let code = EinCode::new(
//!     vec![
//!         vec!['a', 'b'],
//!         vec!['b', 'c'],
//!         vec!['c', 'd'],
//!         vec!['d', 'e'],
//!         vec!['e', 'f'],
//!     ],
//!     vec!['a', 'f'],
//! );
//! let sizes: HashMap<char, usize> =
//!     [('a', 2), ('b', 2), ('c', 2), ('d', 2), ('e', 2), ('f', 2)].into();
//! let (tree, report) = simplify_then_optimize(&code, &sizes, &GreedyMethod::default()).unwrap();
//! // Every original tensor is still a leaf of the spliced tree.
//! assert_eq!(tree.leaf_count(), 5);
//! assert!(report.n_reduced <= report.n_original);
//! ```

use crate::eincode::{EinCode, NestedEinsum};
use crate::{CodeOptimizer, Label};
use std::collections::{HashMap, HashSet, VecDeque};

/// Statistics describing how far the structural simplification pass shrank a
/// network. Returned alongside the optimized tree by [`simplify_then_optimize`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimplifyReport {
    /// Number of tensors in the original network.
    pub n_original: usize,
    /// Number of super-tensors surviving in the reduced network.
    pub n_reduced: usize,
    /// Fraction of tensors removed: `1 - n_reduced / n_original` (0 when empty).
    pub shrink: f64,
}

impl SimplifyReport {
    fn new(n_original: usize, n_reduced: usize) -> Self {
        let shrink = if n_original > 0 {
            1.0 - (n_reduced as f64) / (n_original as f64)
        } else {
            0.0
        };
        Self {
            n_original,
            n_reduced,
            shrink,
        }
    }
}

/// Result of [`simplify`]: a reduced einsum over super-tensors, the internal
/// subtree each super-tensor stands for, and shrink statistics.
///
/// `subtrees[k]` is the binary contraction subtree (over *original* tensor
/// indices) that reduced-network leaf `k` expands to under [`splice`].
#[derive(Debug, Clone)]
pub struct Simplified<L: Label> {
    /// The reduced network over super-tensors.
    pub code: EinCode<L>,
    /// Per super-tensor internal subtree over original tensor indices.
    pub subtrees: Vec<NestedEinsum<L>>,
    /// Shrink statistics for this pass.
    pub report: SimplifyReport,
}

/// Deterministic rank- and size-non-increasing structural simplification.
///
/// Repeatedly contracts a neighbouring pair of (super-)tensors whose merge leaves
/// a surviving tensor whose rank and dimension-aware storage are no larger than
/// the respective maximum of the inputs. Missing dimensions are treated as one,
/// consistently with the other optimizers. The pass is deterministic (labels,
/// neighbours, and requeued tensors are processed in sorted order) and requires
/// `L: Ord` for that determinism.
///
/// # Example
///
/// ```
/// use omeco::preprocess::simplify;
/// use omeco::EinCode;
/// use std::collections::HashMap;
///
/// // 'j' is internal to tensors 0 and 1 and appears nowhere else, so the two are
/// // fused into a single super-tensor.
/// let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
/// let sizes: HashMap<char, usize> = [('i', 2), ('j', 2), ('k', 2)].into();
/// let simplified = simplify(&code, &sizes);
/// assert_eq!(simplified.report.n_original, 2);
/// assert_eq!(simplified.report.n_reduced, 1);
/// ```
pub fn simplify<L: Label + Ord>(code: &EinCode<L>, size_dict: &HashMap<L, usize>) -> Simplified<L> {
    let n = code.ixs.len();
    let out_set: HashSet<L> = code.iy.iter().cloned().collect();

    // Per-super-tensor state, indexed by a stable id 0..n (ids of merged tensors
    // are retired but never reused).
    let mut labels: Vec<HashSet<L>> = code
        .ixs
        .iter()
        .map(|ix| ix.iter().cloned().collect())
        .collect();
    let mut label_vec: Vec<Vec<L>> = code.ixs.clone(); // deterministic order for eins
    let mut subtree: Vec<Option<NestedEinsum<L>>> =
        (0..n).map(|i| Some(NestedEinsum::leaf(i))).collect();
    let mut alive: Vec<bool> = vec![true; n];

    // label -> set of live super-tensor ids containing it.
    let mut lab2t: HashMap<L, HashSet<usize>> = HashMap::new();
    for (i, ls) in labels.iter().enumerate() {
        for l in ls {
            lab2t.entry(l.clone()).or_default().insert(i);
        }
    }

    // Survivors of merging tensors a and b (labels kept in the intermediate).
    let survivors =
        |labels: &[HashSet<L>], lab2t: &HashMap<L, HashSet<usize>>, a: usize, b: usize| -> Vec<L> {
            let mut kept: HashSet<L> = HashSet::new();
            for l in labels[a].iter().chain(labels[b].iter()) {
                let occ = lab2t.get(l).map(|s| s.len()).unwrap_or(0);
                // occurrences outside {a,b}: subtract self-memberships.
                let mut outside = occ;
                if labels[a].contains(l) {
                    outside -= 1;
                }
                if labels[b].contains(l) {
                    outside -= 1;
                }
                if outside > 0 || out_set.contains(l) {
                    kept.insert(l.clone());
                }
            }
            // Preserve the caller's requested output-axis order, then append
            // non-output labels canonically. The same vector becomes both this
            // node's output and its parent's input interface.
            let mut out: Vec<L> = code
                .iy
                .iter()
                .filter(|l| kept.remove(*l))
                .cloned()
                .collect();
            let mut internal: Vec<L> = kept.into_iter().collect();
            internal.sort_unstable();
            out.extend(internal);
            out
        };

    // Work queue of super-tensors to (re)consider; cascades push neighbours back.
    let mut queue: VecDeque<usize> = (0..n).collect();
    let mut in_queue: Vec<bool> = vec![true; n];

    while let Some(t) = queue.pop_front() {
        in_queue[t] = false;
        if !alive[t] {
            continue;
        }
        // Find a live neighbour u sharing a label such that merge is
        // rank-non-increasing. Deterministic: scan labels then neighbour ids in
        // sorted order.
        let mut lab_list: Vec<L> = labels[t].iter().cloned().collect();
        lab_list.sort_unstable();
        let mut chosen: Option<usize> = None;
        'find: for l in &lab_list {
            let mut neigh: Vec<usize> = lab2t
                .get(l)
                .map(|s| s.iter().copied().filter(|&x| x != t).collect())
                .unwrap_or_default();
            neigh.sort_unstable();
            for u in neigh {
                if !alive[u] {
                    continue;
                }
                let surv = survivors(&labels, &lab2t, t, u);
                let log2_size = |ls: &[L]| -> f64 {
                    ls.iter()
                        .map(|label| {
                            (size_dict.get(label).copied().unwrap_or(1).max(1) as f64).log2()
                        })
                        .sum()
                };
                let input_max = log2_size(&label_vec[t]).max(log2_size(&label_vec[u]));
                let output_size = log2_size(&surv);
                if surv.len() <= labels[t].len().max(labels[u].len())
                    && output_size <= input_max + f64::EPSILON
                {
                    chosen = Some(u);
                    break 'find;
                }
            }
        }
        let Some(u) = chosen else { continue };

        // Merge u into t. Build the internal binary subtree node.
        let surv = survivors(&labels, &lab2t, t, u);
        let eins = EinCode::new(
            vec![label_vec[t].clone(), label_vec[u].clone()],
            surv.clone(),
        );
        let child_t = subtree[t].take().expect("live tensor has subtree");
        let child_u = subtree[u].take().expect("live tensor has subtree");
        subtree[t] = Some(NestedEinsum::node(vec![child_t, child_u], eins));

        // Update label adjacency: remove u everywhere; retarget t's labels.
        for l in &labels[u] {
            if let Some(s) = lab2t.get_mut(l) {
                s.remove(&u);
            }
        }
        for l in &labels[t] {
            if let Some(s) = lab2t.get_mut(l) {
                s.remove(&t);
            }
        }
        // Collect neighbours (before retargeting) to re-enqueue.
        let mut to_requeue: HashSet<usize> = HashSet::new();
        let new_labels: HashSet<L> = surv.iter().cloned().collect();
        for l in &new_labels {
            if let Some(s) = lab2t.get(l) {
                for &x in s {
                    if x != t && x != u && alive[x] {
                        to_requeue.insert(x);
                    }
                }
            }
        }
        // Commit new label set for t.
        for l in &new_labels {
            lab2t.entry(l.clone()).or_default().insert(t);
        }
        labels[t] = new_labels;
        label_vec[t] = surv;
        alive[u] = false;

        // Re-enqueue t and its neighbours so cascades run to a fixpoint.
        if !in_queue[t] {
            in_queue[t] = true;
            queue.push_back(t);
        }
        let mut to_requeue: Vec<usize> = to_requeue.into_iter().collect();
        to_requeue.sort_unstable();
        for x in to_requeue {
            if !in_queue[x] {
                in_queue[x] = true;
                queue.push_back(x);
            }
        }
    }

    // Collect the surviving super-tensors (stable order by original id).
    let mut reduced_ixs: Vec<Vec<L>> = Vec::new();
    let mut subtrees: Vec<NestedEinsum<L>> = Vec::new();
    for i in 0..n {
        if alive[i] {
            reduced_ixs.push(label_vec[i].clone());
            subtrees.push(subtree[i].take().expect("live tensor has subtree"));
        }
    }
    let report = SimplifyReport::new(n, subtrees.len());
    Simplified {
        code: EinCode::new(reduced_ixs, code.iy.clone()),
        subtrees,
        report,
    }
}

/// Replace each leaf `k` of a reduced-network tree by super-tensor `k`'s internal
/// subtree, yielding a tree whose leaves are the original tensor indices.
///
/// `subtrees` is [`Simplified::subtrees`]; `tree` must be a contraction tree over
/// the reduced network's leaves (`0..subtrees.len()`).
///
/// # Example
///
/// ```
/// use omeco::preprocess::{simplify, splice};
/// use omeco::{EinCode, GreedyMethod, optimize_code};
/// use std::collections::HashMap;
///
/// let code = EinCode::new(
///     vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
///     vec!['i', 'l'],
/// );
/// let sizes: HashMap<char, usize> = [('i', 2), ('j', 2), ('k', 2), ('l', 2)].into();
/// let simplified = simplify(&code, &sizes);
/// let reduced = optimize_code(&simplified.code, &sizes, &GreedyMethod::default()).unwrap();
/// let full = splice(&reduced, &simplified.subtrees);
/// assert_eq!(full.leaf_count(), 3);
/// ```
pub fn splice<L: Label>(tree: &NestedEinsum<L>, subtrees: &[NestedEinsum<L>]) -> NestedEinsum<L> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => subtrees[*tensor_index].clone(),
        NestedEinsum::Node { args, eins } => {
            let new_args = args.iter().map(|a| splice(a, subtrees)).collect();
            NestedEinsum::node(new_args, eins.clone())
        }
    }
}

/// Simplify a network, optimize the reduced network with `optimizer`, then splice
/// the collapsed structure back.
///
/// Returns the optimized contraction tree over the **original** tensor indices
/// together with a [`SimplifyReport`]. Returns `None` if the optimizer returns no
/// tree (e.g. an empty network).
///
/// # Example
///
/// ```
/// use omeco::preprocess::simplify_then_optimize;
/// use omeco::{contraction_complexity, EinCode, GreedyMethod};
/// use std::collections::HashMap;
///
/// let code = EinCode::new(
///     vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
///     vec!['i', 'l'],
/// );
/// let sizes: HashMap<char, usize> = [('i', 2), ('j', 2), ('k', 2), ('l', 2)].into();
/// let (tree, report) =
///     simplify_then_optimize(&code, &sizes, &GreedyMethod::default()).unwrap();
/// assert_eq!(tree.leaf_count(), 3);
/// // The spliced tree is a valid contraction over the original ixs.
/// let cc = contraction_complexity(&tree, &sizes, &code.ixs);
/// assert!(cc.tc.is_finite());
/// assert_eq!(report.n_original, 3);
/// ```
pub fn simplify_then_optimize<L: Label + Ord, O: CodeOptimizer>(
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    optimizer: &O,
) -> Option<(NestedEinsum<L>, SimplifyReport)> {
    let simplified = simplify(code, size_dict);
    let reduced = optimizer.optimize(&simplified.code, size_dict)?;
    let full = splice(&reduced, &simplified.subtrees);
    Some((full, simplified.report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{contraction_complexity, optimize_code, GreedyMethod};

    fn uniform_sizes<L: Label + Ord>(code: &EinCode<L>, d: usize) -> HashMap<L, usize> {
        code.unique_labels().into_iter().map(|l| (l, d)).collect()
    }

    fn assert_recursive_interfaces<L: Label + std::fmt::Debug>(
        tree: &NestedEinsum<L>,
        original_ixs: &[Vec<L>],
    ) {
        if let NestedEinsum::Node { args, eins } = tree {
            assert_eq!(args.len(), eins.ixs.len());
            for (child, expected) in args.iter().zip(&eins.ixs) {
                assert_eq!(
                    child.output_labels(original_ixs),
                    *expected,
                    "child output must equal its parent input interface"
                );
                assert_recursive_interfaces(child, original_ixs);
            }
        }
    }

    #[test]
    fn test_simplify_chain_collapses_to_one() {
        // A pure chain with unique internal bonds fuses to a single super-tensor.
        let code = EinCode::new(
            vec![
                vec!['a', 'b'],
                vec!['b', 'c'],
                vec!['c', 'd'],
                vec!['d', 'e'],
            ],
            vec!['a', 'e'],
        );
        let sizes = uniform_sizes(&code, 2);
        let simplified = simplify(&code, &sizes);
        assert_eq!(simplified.report.n_original, 4);
        assert_eq!(simplified.report.n_reduced, 1);
        assert!((simplified.report.shrink - 0.75).abs() < 1e-12);
        // The single subtree still covers all four original tensors.
        assert_eq!(simplified.subtrees.len(), 1);
        assert_eq!(simplified.subtrees[0].leaf_count(), 4);
    }

    #[test]
    fn test_simplify_noop_on_hyperedge_star() {
        // A star whose centre label 'x' occurs in all four tensors: merging any
        // pair keeps 'x' plus both output legs, growing the rank, so no
        // rank-non-increasing merge exists and the pass is a no-op.
        let code = EinCode::new(
            vec![
                vec!['x', 'a'],
                vec!['x', 'b'],
                vec!['x', 'c'],
                vec!['x', 'd'],
            ],
            vec!['a', 'b', 'c', 'd'],
        );
        let sizes = uniform_sizes(&code, 2);
        let simplified = simplify(&code, &sizes);
        assert_eq!(simplified.report.n_reduced, 4);
        assert!((simplified.report.shrink).abs() < 1e-12);
        // Every subtree is a bare leaf (nothing fused).
        assert!(simplified.subtrees.iter().all(|s| s.leaf_count() == 1));
    }

    #[test]
    fn test_simplify_preserves_leaf_permutation() {
        // Every original tensor index must appear exactly once across all subtrees.
        let code = EinCode::new(
            vec![
                vec!['a', 'b'],
                vec!['b', 'c'],
                vec!['c', 'd'],
                vec!['d', 'e'],
                vec!['e', 'f'],
                vec!['f', 'g'],
            ],
            vec!['a', 'g'],
        );
        let sizes = uniform_sizes(&code, 2);
        let simplified = simplify(&code, &sizes);
        let mut leaves: Vec<usize> = simplified
            .subtrees
            .iter()
            .flat_map(|s| s.leaf_indices())
            .collect();
        leaves.sort_unstable();
        assert_eq!(leaves, (0..6).collect::<Vec<_>>());
    }

    #[test]
    fn test_simplify_then_optimize_spliced_tree_is_valid() {
        // The spliced tree must be a valid contraction over the ORIGINAL ixs: its
        // leaves are a permutation of 0..n and its tc is finite. Splicing only
        // adds the cheap internal fusion contractions, so the full tc is never
        // below the reduced tc.
        let code = EinCode::new(
            vec![
                vec!['a', 'b'],
                vec!['b', 'c'],
                vec!['c', 'd'],
                vec!['d', 'e'],
                vec!['e', 'f'],
            ],
            vec!['a', 'f'],
        );
        let sizes = uniform_sizes(&code, 2);
        let simplified = simplify(&code, &sizes);
        let reduced = optimize_code(&simplified.code, &sizes, &GreedyMethod::default()).unwrap();
        let reduced_tc = contraction_complexity(&reduced, &sizes, &simplified.code.ixs).tc;
        let full = splice(&reduced, &simplified.subtrees);
        let full_tc = contraction_complexity(&full, &sizes, &code.ixs).tc;
        assert!(full_tc.is_finite());
        assert!(
            full_tc >= reduced_tc - 1e-9,
            "full_tc={full_tc} reduced_tc={reduced_tc}"
        );

        let mut leaves = full.leaf_indices();
        leaves.sort_unstable();
        assert_eq!(leaves, (0..5).collect::<Vec<_>>());
        assert_recursive_interfaces(&full, &code.ixs);

        // The wrapper reproduces the same spliced tree structure.
        let (tree, report) =
            simplify_then_optimize(&code, &sizes, &GreedyMethod::default()).unwrap();
        assert_eq!(tree.leaf_count(), 5);
        assert_eq!(report.n_original, 5);
    }

    #[test]
    fn test_simplify_empty() {
        let code: EinCode<char> = EinCode::new(vec![], vec![]);
        let simplified = simplify(&code, &HashMap::new());
        assert_eq!(simplified.report.n_original, 0);
        assert_eq!(simplified.report.n_reduced, 0);
        assert_eq!(simplified.report.shrink, 0.0);
    }

    #[test]
    fn test_simplify_rejects_rank_equal_size_growth() {
        // Both inputs have rank two and size 2_000, while their rank-two output
        // would have size 1_000_000. A rank-only pass incorrectly fused these.
        let code = EinCode::new(vec![vec!['a', 'x'], vec!['b', 'x']], vec!['a', 'b']);
        let sizes: HashMap<char, usize> = [('a', 1_000), ('b', 1_000), ('x', 2)].into();

        let simplified = simplify(&code, &sizes);

        assert_eq!(simplified.report.n_reduced, 2);
        assert!(simplified
            .subtrees
            .iter()
            .all(|tree| tree.leaf_count() == 1));
    }

    #[test]
    fn test_simplify_is_deterministic_across_hash_seeds() {
        let code = EinCode::new(
            vec![
                vec!['a', 'b'],
                vec!['b', 'c'],
                vec!['c', 'd'],
                vec!['c', 'e'],
                vec!['e', 'f'],
            ],
            vec!['a', 'd', 'f'],
        );
        let sizes = uniform_sizes(&code, 2);
        let baseline = format!("{:?}", simplify(&code, &sizes).subtrees);
        for _ in 0..16 {
            assert_eq!(format!("{:?}", simplify(&code, &sizes).subtrees), baseline);
        }
    }

    #[test]
    fn test_simplify_preserves_output_axis_order_and_interfaces() {
        let code = EinCode::new(
            vec![vec!['b', 'x', 'c'], vec!['a', 'x']],
            vec!['b', 'a', 'c'],
        );
        let sizes = uniform_sizes(&code, 2);

        for _ in 0..16 {
            let simplified = simplify(&code, &sizes);
            assert_eq!(simplified.report.n_reduced, 1);
            let subtree = &simplified.subtrees[0];
            assert_eq!(subtree.output_labels(&code.ixs), code.iy);
            assert_recursive_interfaces(subtree, &code.ixs);
        }
    }
}
