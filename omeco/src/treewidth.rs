//! Treewidth-heuristic contraction order optimizer (elimination ordering).
//!
//! This module provides [`Treewidth`], a scalable contraction-order optimizer
//! built around a variable-elimination (tree-decomposition) view of the tensor
//! network. It mirrors the `Treewidth` optimizer of the Julia reference package
//! [OMEinsumContractionOrders.jl](https://github.com/TensorBFS/OMEinsumContractionOrders.jl):
//! an *elimination order* over the index labels is computed by a treewidth
//! heuristic, then replayed into a binary contraction tree.
//!
//! Where the Julia package delegates to CliqueTrees.jl (min-fill and friends),
//! omeco currently ships a single, self-contained heuristic:
//! [`EliminationAlgorithm::MinDegree`] — a **weighted minimum-degree** ordering
//! computed on a *quotient graph* with element absorption, in the manner of
//! sparse-matrix AMD codes. Fill edges are never materialized, so the ordering
//! scales to networks with tens of thousands of tensors and labels in a handful
//! of milliseconds while still finding the optimal treewidth on structured
//! (graphical-model / relational) instances where pairwise-greedy and annealing
//! optimizers stall.
//!
//! # How it works
//!
//! Each label becomes a vertex of the *primal graph*; each input tensor is an
//! initial clique (element) over the labels it carries. Output labels (`iy`) are
//! never eliminated. Eliminating a label forms a new clique over its current
//! neighborhood (the boundary of its incident elements) and absorbs the old
//! elements. A label's score is the weighted degree of that neighborhood — the
//! summed `log2` dimensions of the clique its elimination would create — which
//! reduces to a plain neighbor count for binary indices. A lazy min-heap pops
//! the lowest-scoring label, rescoring stale entries on pop; ties break by
//! interned label id, so the order is fully deterministic. The resulting
//! elimination order is then replayed: at each step the tensors sharing the
//! eliminated label are contracted into one intermediate (dropping labels that
//! become fully internal), yielding a binary [`NestedEinsum`] over all original
//! tensors.
//!
//! # Complexity metrics
//!
//! The returned tree is a standard [`NestedEinsum`]; its time (`tc`), space
//! (`sc`) and read-write (`rwc`) complexities are reported exactly like any
//! other optimizer via [`crate::contraction_complexity`]. The elimination
//! *width* (see [`EliminationOrder::width`]) equals, in `log2` scale, the size
//! of the largest clique created — an upper bound closely tracking the tree's
//! `sc`.
//!
//! # Example
//!
//! ```rust
//! use omeco::{EinCode, Treewidth, contraction_complexity, optimize_code};
//! use std::collections::HashMap;
//!
//! // A[i,j] B[j,k] C[k,l] -> [i,l] (a matrix chain).
//! let code = EinCode::new(
//!     vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
//!     vec!['i', 'l'],
//! );
//! let sizes: HashMap<char, usize> =
//!     [('i', 2), ('j', 4), ('k', 8), ('l', 2)].into_iter().collect();
//!
//! let tree = optimize_code(&code, &sizes, &Treewidth::default())
//!     .expect("optimization succeeds");
//! assert!(tree.is_binary());
//! let cc = contraction_complexity(&tree, &sizes, &code.ixs);
//! assert!(cc.tc > 0.0);
//! ```

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::eincode::{EinCode, NestedEinsum};
use crate::Label;

/// Elimination-ordering algorithm used by [`Treewidth`].
///
/// This mirrors the `alg` field of Julia's `Treewidth` optimizer. Only the
/// weighted minimum-degree heuristic is implemented today; the enum is
/// non-exhaustive so further heuristics (e.g. minimum-fill) can be added without
/// breaking callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum EliminationAlgorithm {
    /// Weighted minimum-degree ordering on a quotient graph with element
    /// absorption (AMD-style). Deterministic and highly scalable.
    #[default]
    MinDegree,
}

/// Treewidth-heuristic contraction-order optimizer.
///
/// Computes an elimination order over the index labels with the configured
/// [`EliminationAlgorithm`], then replays it into a binary contraction tree.
/// The optimizer is deterministic: the same input always yields the same tree.
///
/// # Example
///
/// ```rust
/// use omeco::{EinCode, Treewidth, optimize_code};
/// use std::collections::HashMap;
///
/// let code = EinCode::new(vec![vec!['a', 'b'], vec!['b', 'c']], vec!['a', 'c']);
/// let sizes: HashMap<char, usize> = [('a', 2), ('b', 2), ('c', 2)].into_iter().collect();
/// let tree = optimize_code(&code, &sizes, &Treewidth::min_degree()).unwrap();
/// assert_eq!(tree.leaf_count(), 2);
/// ```
#[derive(Debug, Clone, Default)]
pub struct Treewidth {
    /// The elimination algorithm used to compute the ordering.
    pub alg: EliminationAlgorithm,
}

impl Treewidth {
    /// Create a `Treewidth` optimizer using the given elimination algorithm.
    ///
    /// # Example
    ///
    /// ```rust
    /// use omeco::{EliminationAlgorithm, Treewidth};
    ///
    /// let opt = Treewidth::new(EliminationAlgorithm::MinDegree);
    /// assert_eq!(opt.alg, EliminationAlgorithm::MinDegree);
    /// ```
    pub fn new(alg: EliminationAlgorithm) -> Self {
        Self { alg }
    }

    /// Create a `Treewidth` optimizer using the weighted minimum-degree
    /// heuristic (the current default).
    ///
    /// # Example
    ///
    /// ```rust
    /// use omeco::Treewidth;
    ///
    /// let opt = Treewidth::min_degree();
    /// ```
    pub fn min_degree() -> Self {
        Self::new(EliminationAlgorithm::MinDegree)
    }

    /// Compute the elimination order and its weighted width for `code`.
    ///
    /// The order lists the labels in the sequence they are eliminated (output
    /// labels in `code.iy` are never eliminated and never appear). The width is
    /// the `log2` size of the largest clique formed during elimination.
    ///
    /// # Errors
    ///
    /// Returns [`TreewidthError::EmptyCode`] if `code` has no input tensors.
    ///
    /// # Example
    ///
    /// ```rust
    /// use omeco::{EinCode, Treewidth};
    /// use std::collections::HashMap;
    ///
    /// // A 4-tensor chain: eliminating the shared binary labels one at a time
    /// // never builds a clique wider than 2 labels, so the width is 2.
    /// let code = EinCode::new(
    ///     vec![vec!['a', 'b'], vec!['b', 'c'], vec!['c', 'd'], vec!['d', 'e']],
    ///     vec!['a', 'e'],
    /// );
    /// let sizes: HashMap<char, usize> =
    ///     ['a', 'b', 'c', 'd', 'e'].into_iter().map(|c| (c, 2)).collect();
    /// let elim = Treewidth::min_degree().elimination_order(&code, &sizes).unwrap();
    /// assert_eq!(elim.width, 2.0);
    /// ```
    pub fn elimination_order<L: Label>(
        &self,
        code: &EinCode<L>,
        size_dict: &HashMap<L, usize>,
    ) -> Result<EliminationOrder<L>, TreewidthError> {
        if code.ixs.is_empty() {
            return Err(TreewidthError::EmptyCode);
        }
        let hg = EliminationHyperGraph::build(code, size_dict);
        let (ids, width) = match self.alg {
            EliminationAlgorithm::MinDegree => hg.min_degree_order(),
        };
        let order = ids
            .into_iter()
            .map(|id| hg.id_label[id as usize].clone())
            .collect();
        Ok(EliminationOrder { order, width })
    }
}

/// An elimination order together with the weighted treewidth it realizes.
///
/// Returned by [`Treewidth::elimination_order`].
#[derive(Debug, Clone)]
pub struct EliminationOrder<L: Label> {
    /// Labels in the order they are eliminated (output labels excluded).
    pub order: Vec<L>,
    /// The `log2` size of the largest clique formed during elimination — an
    /// upper bound on the contraction tree's space complexity.
    pub width: f64,
}

/// Errors returned by the treewidth optimizer.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TreewidthError {
    /// The einsum code contains no input tensors, so there is nothing to
    /// optimize.
    #[error("cannot optimize an empty einsum code (no input tensors)")]
    EmptyCode,
}

/// Optimize the contraction order of `code` with the [`Treewidth`] heuristic.
///
/// Computes an elimination order (per `optimizer.alg`) and replays it into a
/// binary [`NestedEinsum`] over all input tensors. A single input tensor yields
/// a [`NestedEinsum::Leaf`].
///
/// # Errors
///
/// Returns [`TreewidthError::EmptyCode`] if `code` has no input tensors.
///
/// # Example
///
/// ```rust
/// use omeco::treewidth::{optimize_treewidth, Treewidth};
/// use omeco::EinCode;
/// use std::collections::HashMap;
///
/// let code = EinCode::new(
///     vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
///     vec!['i', 'l'],
/// );
/// let sizes: HashMap<char, usize> =
///     [('i', 2), ('j', 2), ('k', 2), ('l', 2)].into_iter().collect();
/// let tree = optimize_treewidth(&code, &sizes, &Treewidth::default()).unwrap();
/// assert_eq!(tree.leaf_count(), 3);
/// ```
pub fn optimize_treewidth<L: Label>(
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    optimizer: &Treewidth,
) -> Result<NestedEinsum<L>, TreewidthError> {
    if code.ixs.is_empty() {
        return Err(TreewidthError::EmptyCode);
    }
    let hg = EliminationHyperGraph::build(code, size_dict);
    let (order, _width) = match optimizer.alg {
        EliminationAlgorithm::MinDegree => hg.min_degree_order(),
    };
    let topo = hg.replay_order(&order);
    Ok(hg.build_nested(&topo))
}

// =============================================================================
// Internal elimination engine
// =============================================================================

/// A binary topology over leaf tensor indices, with no einsum metadata — the
/// per-node inputs/outputs are derived later by outside-occurrence counting.
enum TopoTree {
    Leaf(usize),
    Node(Box<TopoTree>, Box<TopoTree>),
}

/// Interned view of an einsum expression used by the elimination engine:
/// labels are mapped to a dense `u32` id space, with per-leaf id sets, the `iy`
/// id set, per-id `log2` sizes, and per-id holder counts (hyperedge degrees).
struct EliminationHyperGraph<L: Label> {
    /// Original label for each id (`id -> label`).
    id_label: Vec<L>,
    /// Per-leaf sorted-unique label ids.
    leaf_ids: Vec<Vec<u32>>,
    /// The set of output (`iy`) label ids; these are never eliminated.
    iy_ids: HashSet<u32>,
    /// `log2` of each id's dimension.
    log2: Vec<f64>,
    /// Number of leaves holding each id (its total hyperedge degree).
    total_count: Vec<u32>,
    /// Number of leaves.
    n: usize,
}

impl<L: Label> EliminationHyperGraph<L> {
    /// Intern an einsum code into the dense-id representation.
    fn build(code: &EinCode<L>, size_dict: &HashMap<L, usize>) -> Self {
        let mut id_of: HashMap<L, u32> = HashMap::new();
        let mut id_label: Vec<L> = Vec::new();
        let mut log2: Vec<f64> = Vec::new();

        let intern = |label: &L,
                      id_of: &mut HashMap<L, u32>,
                      id_label: &mut Vec<L>,
                      log2: &mut Vec<f64>|
         -> u32 {
            if let Some(&id) = id_of.get(label) {
                return id;
            }
            let id = id_label.len() as u32;
            id_of.insert(label.clone(), id);
            id_label.push(label.clone());
            let dim = size_dict.get(label).copied().unwrap_or(1).max(1);
            log2.push((dim as f64).log2());
            id
        };

        let leaf_ids: Vec<Vec<u32>> = code
            .ixs
            .iter()
            .map(|ix| {
                let mut v: Vec<u32> = ix
                    .iter()
                    .map(|l| intern(l, &mut id_of, &mut id_label, &mut log2))
                    .collect();
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();
        let iy_ids: HashSet<u32> = code
            .iy
            .iter()
            .map(|l| intern(l, &mut id_of, &mut id_label, &mut log2))
            .collect();

        let m = id_label.len();
        let mut total_count = vec![0u32; m];
        for v in &leaf_ids {
            for &id in v {
                total_count[id as usize] += 1;
            }
        }

        Self {
            n: leaf_ids.len(),
            id_label,
            leaf_ids,
            iy_ids,
            log2,
            total_count,
        }
    }

    /// `log2`-size sum over a sorted id set (the size, in `log2` scale, of a
    /// tensor carrying exactly that label set).
    fn set_cost(&self, ids: &[u32]) -> f64 {
        ids.iter().map(|&id| self.log2[id as usize]).sum()
    }

    /// Compute a weighted minimum-degree elimination order over the label
    /// quotient graph. Returns `(order, width)`: the elimination sequence of
    /// label ids and the `log2` size of the largest clique formed.
    fn min_degree_order(&self) -> (Vec<u32>, f64) {
        let m = self.id_label.len();
        let w = &self.log2;

        // Element store: original tensors first, then one element per
        // elimination. `le[e]` is element `e`'s boundary (its live members).
        let mut le: Vec<Vec<u32>> = Vec::with_capacity(self.n + m);
        let mut absorbed: Vec<bool> = Vec::with_capacity(self.n + m);
        // Per-variable list of incident elements.
        let mut aelem: Vec<Vec<u32>> = vec![Vec::new(); m];
        for ids in &self.leaf_ids {
            let e = le.len() as u32;
            for &v in ids {
                aelem[v as usize].push(e);
            }
            le.push(ids.clone());
            absorbed.push(false);
        }

        let mut alive = vec![true; m];
        for &id in &self.iy_ids {
            alive[id as usize] = false; // never eliminate output labels
        }

        // Timestamp scratch for O(size) neighbor dedup.
        let mut mark = vec![0u32; m];
        let mut tick = 0u32;

        // Lazy min-heap keyed on (weighted degree, id). `BinaryHeap` is a
        // max-heap, so `Reverse` the degree (min pops first) and `Reverse` the
        // id (smallest id wins ties -> deterministic).
        let mut heap: BinaryHeap<(std::cmp::Reverse<OrdF64>, std::cmp::Reverse<u32>)> =
            BinaryHeap::new();
        for (v, &live) in alive.iter().enumerate() {
            if live {
                let d = wdeg(
                    v, &mut aelem, &le, &absorbed, &alive, w, &mut mark, &mut tick,
                );
                heap.push((std::cmp::Reverse(OrdF64(d)), std::cmp::Reverse(v as u32)));
            }
        }

        let mut order: Vec<u32> = Vec::with_capacity(m);
        let mut width = 0.0f64;
        while let Some((std::cmp::Reverse(OrdF64(key)), std::cmp::Reverse(vid))) = heap.pop() {
            let v = vid as usize;
            if !alive[v] {
                continue;
            }
            // Rescore on pop: a stale entry (key no longer matches the fresh
            // degree) is re-pushed with the correct degree and skipped.
            let d = wdeg(
                v, &mut aelem, &le, &absorbed, &alive, w, &mut mark, &mut tick,
            );
            if (d - key).abs() > 1e-9 {
                heap.push((std::cmp::Reverse(OrdF64(d)), std::cmp::Reverse(vid)));
                continue;
            }

            // Eliminate `v`: its neighborhood becomes a new clique/element.
            let lp = boundary_of(v, &mut aelem, &le, &absorbed, &alive, &mut mark, &mut tick);
            let wclique = lp.iter().map(|&u| w[u as usize]).sum::<f64>() + w[v];
            if wclique > width {
                width = wclique;
            }
            order.push(v as u32);
            alive[v] = false;

            let q = le.len() as u32;
            // Absorb `v`'s old elements — they are subsumed by the new clique.
            for &e in &aelem[v] {
                absorbed[e as usize] = true;
                le[e as usize] = Vec::new();
            }
            aelem[v] = Vec::new();
            le.push(lp.clone());
            absorbed.push(false);

            // Each neighbor now sits on element `q`; drop its absorbed elements
            // and rescore it lazily (the old heap entry becomes stale).
            for &u in &lp {
                let uu = u as usize;
                aelem[uu].retain(|&e| !absorbed[e as usize]);
                aelem[uu].push(q);
                let d2 = wdeg(
                    uu, &mut aelem, &le, &absorbed, &alive, w, &mut mark, &mut tick,
                );
                heap.push((std::cmp::Reverse(OrdF64(d2)), std::cmp::Reverse(u)));
            }
        }

        (order, width)
    }

    /// Replay a precomputed elimination `order` into a contraction topology:
    /// for each label still carrying holders, contract its holder group into one
    /// super-tensor (dropping labels that become fully internal). Any tensors
    /// left over (disconnected components / scalars) are merged into the root
    /// smallest-first. Deterministic.
    fn replay_order(&self, order: &[u32]) -> TopoTree {
        let m = self.id_label.len();
        let n = self.n;

        let mut live: HashMap<usize, Vec<u32>> = HashMap::with_capacity(n * 2);
        let mut topo: HashMap<usize, TopoTree> = HashMap::with_capacity(n * 2);
        let mut holders: Vec<HashSet<usize>> = vec![HashSet::new(); m];
        for (i, ids) in self.leaf_ids.iter().enumerate() {
            live.insert(i, ids.clone());
            topo.insert(i, TopoTree::Leaf(i));
            for &id in ids {
                holders[id as usize].insert(i);
            }
        }
        let mut eliminated = vec![false; m];
        for &id in &self.iy_ids {
            eliminated[id as usize] = true; // never eliminate output labels
        }
        let mut next_tid = n;

        for &id in order {
            let id = id as usize;
            if eliminated[id] || holders[id].is_empty() {
                continue;
            }
            // Merge the holder group of `id` into one super-tensor. Sorting by
            // tensor id keeps the merge deterministic regardless of `HashSet`
            // iteration order.
            let mut group: Vec<usize> = holders[id].iter().copied().collect();
            group.sort_unstable();
            let group_set: HashSet<usize> = group.iter().copied().collect();

            let members: Vec<(Vec<u32>, TopoTree)> = group
                .iter()
                .filter_map(|&t| match (live.remove(&t), topo.remove(&t)) {
                    (Some(l), Some(tp)) => Some((l, tp)),
                    _ => None,
                })
                .collect();
            let Some((live_union, merged_topo)) = self.merge_group(members) else {
                continue;
            };

            // Drop `id`; keep a label iff it is an output label or still has a
            // holder outside this group.
            let mut new_live: Vec<u32> = Vec::with_capacity(live_union.len());
            let mut dropped: Vec<u32> = Vec::new();
            for &l in &live_union {
                if l as usize == id {
                    continue;
                }
                let outside = holders[l as usize].iter().any(|t| !group_set.contains(t));
                if self.iy_ids.contains(&l) || outside {
                    new_live.push(l);
                } else {
                    dropped.push(l);
                }
            }

            let tnew = next_tid;
            next_tid += 1;
            for &l in &live_union {
                let hs = &mut holders[l as usize];
                for t in &group {
                    hs.remove(t);
                }
            }
            for &l in &new_live {
                holders[l as usize].insert(tnew);
            }
            eliminated[id] = true;
            holders[id].clear();
            for &l in &dropped {
                eliminated[l as usize] = true;
                holders[l as usize].clear();
            }

            live.insert(tnew, new_live);
            topo.insert(tnew, merged_topo);
        }

        // Merge any remaining active tensors into a single root, smallest-first
        // (ties broken by tensor id for determinism).
        let mut remaining: Vec<(usize, Vec<u32>, TopoTree)> = topo
            .into_iter()
            .filter_map(|(t, tp)| live.remove(&t).map(|l| (t, l, tp)))
            .collect();
        remaining.sort_by(|a, b| {
            self.set_cost(&a.1)
                .total_cmp(&self.set_cost(&b.1))
                .then(a.0.cmp(&b.0))
        });
        let members: Vec<(Vec<u32>, TopoTree)> =
            remaining.into_iter().map(|(_, l, tp)| (l, tp)).collect();
        self.merge_group(members)
            .map(|(_, root)| root)
            .unwrap_or(TopoTree::Leaf(0))
    }

    /// Contract a group of tensors into one, choosing a local greedy
    /// min-union pairwise order. Returns the merged live set and the binary
    /// topology, or `None` if the group is empty. Groups larger than 12 fall
    /// back to a size-ordered chain to bound the `O(k^2)` pair search.
    fn merge_group(&self, mut members: Vec<(Vec<u32>, TopoTree)>) -> Option<(Vec<u32>, TopoTree)> {
        if members.len() <= 1 {
            return members.pop();
        }
        if members.len() > 12 {
            // Chain largest-first (cheap; avoids O(k^2) blow-up on big groups).
            members.sort_by(|a, b| self.set_cost(&b.0).total_cmp(&self.set_cost(&a.0)));
            let mut acc = members.pop()?;
            while let Some(next) = members.pop() {
                let u = sorted_union(&acc.0, &next.0);
                let node = TopoTree::Node(Box::new(acc.1), Box::new(next.1));
                acc = (u, node);
            }
            return Some(acc);
        }
        // Greedy: repeatedly merge the pair with the smallest union cost.
        while members.len() > 1 {
            let mut best = (0usize, 1usize, f64::INFINITY);
            for i in 0..members.len() {
                for j in (i + 1)..members.len() {
                    let u = sorted_union(&members[i].0, &members[j].0);
                    let c = self.set_cost(&u);
                    if c < best.2 {
                        best = (i, j, c);
                    }
                }
            }
            let (i, j, _) = best;
            // Remove the higher index first so the lower index stays valid.
            let (lj, tj) = members.remove(j);
            let (li, ti) = members.remove(i);
            let u = sorted_union(&li, &lj);
            let node = TopoTree::Node(Box::new(ti), Box::new(tj));
            members.push((u, node));
        }
        members.pop()
    }

    /// Convert a topology into a [`NestedEinsum`], deriving each node's output
    /// by exact outside-occurrence counting: a label is in a node's output iff
    /// it appears in `iy` or in some leaf outside the node's subtree.
    fn build_nested(&self, topo: &TopoTree) -> NestedEinsum<L> {
        self.build_inner(topo).0
    }

    /// Returns the subtree and its output label -> subtree-occurrence-count map.
    fn build_inner(&self, topo: &TopoTree) -> (NestedEinsum<L>, HashMap<u32, u32>) {
        match topo {
            TopoTree::Leaf(i) => {
                let mut counts = HashMap::with_capacity(self.leaf_ids[*i].len());
                for &id in &self.leaf_ids[*i] {
                    counts.insert(id, 1u32);
                }
                (NestedEinsum::leaf(*i), counts)
            }
            TopoTree::Node(l, r) => {
                let (ltree, lc) = self.build_inner(l);
                let (rtree, rc) = self.build_inner(r);

                // Children's inputs (their output label sets), ordered by id for
                // reproducibility.
                let left_out = self.ids_to_labels(lc.keys().copied());
                let right_out = self.ids_to_labels(rc.keys().copied());

                // Union of children output ids with subtree counts.
                let mut counts: HashMap<u32, u32> = lc;
                for (id, c) in rc {
                    *counts.entry(id).or_insert(0) += c;
                }
                // Node output: labels appearing outside the subtree or in `iy`.
                let mut out_ids: Vec<u32> = Vec::new();
                let mut out_counts: HashMap<u32, u32> = HashMap::with_capacity(counts.len());
                for (&id, &sub) in &counts {
                    if self.total_count[id as usize] > sub || self.iy_ids.contains(&id) {
                        out_ids.push(id);
                        out_counts.insert(id, sub);
                    }
                }
                out_ids.sort_unstable();
                let node_out = out_ids.iter().map(|&id| self.id_label[id as usize].clone());
                let eins = EinCode::new(vec![left_out, right_out], node_out.collect());
                (NestedEinsum::node(vec![ltree, rtree], eins), out_counts)
            }
        }
    }

    /// Map an iterator of ids to labels, sorted by id for a reproducible order.
    fn ids_to_labels(&self, ids: impl Iterator<Item = u32>) -> Vec<L> {
        let mut v: Vec<u32> = ids.collect();
        v.sort_unstable();
        v.into_iter()
            .map(|id| self.id_label[id as usize].clone())
            .collect()
    }
}

/// Weighted degree of variable `v` in the quotient graph: the summed `log2`
/// dims of its distinct live neighbors (the boundary of the clique its
/// elimination would create). Absorbed elements are pruned from `aelem[v]` in
/// passing; `mark`/`tick` dedups neighbors in `O(scanned size)`.
#[allow(clippy::too_many_arguments)]
fn wdeg(
    v: usize,
    aelem: &mut [Vec<u32>],
    le: &[Vec<u32>],
    absorbed: &[bool],
    alive: &[bool],
    w: &[f64],
    mark: &mut [u32],
    tick: &mut u32,
) -> f64 {
    *tick += 1;
    let t = *tick;
    aelem[v].retain(|&e| !absorbed[e as usize]);
    let mut deg = 0.0f64;
    for &e in &aelem[v] {
        for &u in &le[e as usize] {
            let u = u as usize;
            if u != v && alive[u] && mark[u] != t {
                mark[u] = t;
                deg += w[u];
            }
        }
    }
    deg
}

/// The current neighborhood (boundary) of variable `v` as a deduplicated list
/// of live variable ids, using the same element scan as [`wdeg`]. This is the
/// boundary of the element created by eliminating `v`.
fn boundary_of(
    v: usize,
    aelem: &mut [Vec<u32>],
    le: &[Vec<u32>],
    absorbed: &[bool],
    alive: &[bool],
    mark: &mut [u32],
    tick: &mut u32,
) -> Vec<u32> {
    *tick += 1;
    let t = *tick;
    aelem[v].retain(|&e| !absorbed[e as usize]);
    let mut out: Vec<u32> = Vec::new();
    for &e in &aelem[v] {
        for &u in &le[e as usize] {
            let uu = u as usize;
            if uu != v && alive[uu] && mark[uu] != t {
                mark[uu] = t;
                out.push(u);
            }
        }
    }
    out
}

/// Sorted-set union of two sorted-unique id slices.
fn sorted_union(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// Total-ordered `f64` wrapper for the priority queue (finite scores only).
#[derive(Clone, Copy, PartialEq)]
struct OrdF64(f64);
impl Eq for OrdF64 {}
impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{contraction_complexity, optimize_code};

    fn sizes_uniform<L: Label + Clone>(labels: &[L], dim: usize) -> HashMap<L, usize> {
        labels.iter().map(|l| (l.clone(), dim)).collect()
    }

    #[test]
    fn test_matrix_chain_binary_and_correct_output() {
        // A[i,j] B[j,k] C[k,l] -> [i,l]
        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
            vec!['i', 'l'],
        );
        let sizes = sizes_uniform(&['i', 'j', 'k', 'l'], 4);
        let tree = optimize_code(&code, &sizes, &Treewidth::default()).unwrap();
        assert!(tree.is_binary());
        assert_eq!(tree.leaf_count(), 3);
        // Root output must be exactly the requested iy.
        match &tree {
            NestedEinsum::Node { eins, .. } => {
                let mut got = eins.iy.clone();
                got.sort_unstable();
                assert_eq!(got, vec!['i', 'l']);
            }
            NestedEinsum::Leaf { .. } => panic!("3-tensor network must not be a leaf"),
        }
    }

    #[test]
    fn test_single_tensor_is_leaf() {
        let code = EinCode::new(vec![vec!['i', 'j']], vec!['i', 'j']);
        let sizes = sizes_uniform(&['i', 'j'], 4);
        let tree = optimize_code(&code, &sizes, &Treewidth::default()).unwrap();
        assert!(tree.is_leaf());
        assert_eq!(tree.tensor_index(), Some(0));
    }

    #[test]
    fn test_empty_code_errors() {
        let code: EinCode<char> = EinCode::new(vec![], vec![]);
        let sizes: HashMap<char, usize> = HashMap::new();
        assert_eq!(
            optimize_treewidth(&code, &sizes, &Treewidth::default()),
            Err(TreewidthError::EmptyCode)
        );
        // The generic optimize_code wrapper turns the error into None.
        assert!(optimize_code(&code, &sizes, &Treewidth::default()).is_none());
    }

    #[test]
    fn test_leaves_are_a_permutation() {
        // Star network: center label shared by 4 arms, plus per-arm labels.
        let code = EinCode::new(
            vec![
                vec!['x', 'a'],
                vec!['x', 'b'],
                vec!['x', 'c'],
                vec!['x', 'd'],
            ],
            vec!['a', 'b', 'c', 'd'],
        );
        let sizes = sizes_uniform(&['x', 'a', 'b', 'c', 'd'], 2);
        let tree = optimize_code(&code, &sizes, &Treewidth::default()).unwrap();
        let mut leaves = tree.leaf_indices();
        leaves.sort_unstable();
        assert_eq!(leaves, vec![0, 1, 2, 3]);
        assert!(tree.is_binary());
    }

    #[test]
    fn test_chain_width_equals_two() {
        // A path/chain of binary bonds has treewidth 1, so the largest clique
        // formed while eliminating a bond spans two labels -> width 2.0.
        let code = EinCode::new(
            vec![
                vec!['a', 'b'],
                vec!['b', 'c'],
                vec!['c', 'd'],
                vec!['d', 'e'],
            ],
            vec!['a', 'e'],
        );
        let sizes = sizes_uniform(&['a', 'b', 'c', 'd', 'e'], 2);
        let elim = Treewidth::min_degree()
            .elimination_order(&code, &sizes)
            .unwrap();
        assert_eq!(elim.width, 2.0);
        // Chain sc is 2 (one boundary bond of dim 2 kept at a time).
        let tree = optimize_code(&code, &sizes, &Treewidth::default()).unwrap();
        let cc = contraction_complexity(&tree, &sizes, &code.ixs);
        assert_eq!(cc.sc, 2.0);
    }

    #[test]
    fn test_grid_fragment_width() {
        // A 2x2 grid of bond tensors (4 plaquette labels around a center) has a
        // small, checkable treewidth. Build a ring of 4 tensors: labels a,b,c,d.
        //   T0[a,b] T1[b,c] T2[c,d] T3[d,a]  -> scalar
        // The optimal elimination width for a 4-cycle is 3 labels (clique of 3).
        let code = EinCode::new(
            vec![
                vec!['a', 'b'],
                vec!['b', 'c'],
                vec!['c', 'd'],
                vec!['d', 'a'],
            ],
            vec![],
        );
        let sizes = sizes_uniform(&['a', 'b', 'c', 'd'], 2);
        let elim = Treewidth::min_degree()
            .elimination_order(&code, &sizes)
            .unwrap();
        // Eliminating the first cycle label forms a clique over its two
        // neighbors plus itself = 3 binary labels => width 3.0.
        assert_eq!(elim.width, 3.0);
        assert_eq!(elim.order.len(), 4); // all four labels eliminated
    }

    #[test]
    fn test_non_binary_dims_weighted() {
        // Different dimensions must be weighted by log2(size). Triangle with a
        // large shared label should still contract correctly.
        let code = EinCode::new(vec![vec!['a', 'b'], vec!['b', 'c'], vec!['c', 'a']], vec![]);
        let sizes: HashMap<char, usize> = [('a', 2), ('b', 16), ('c', 8)].into_iter().collect();
        let tree = optimize_code(&code, &sizes, &Treewidth::default()).unwrap();
        assert!(tree.is_binary());
        assert_eq!(tree.leaf_count(), 3);
        let cc = contraction_complexity(&tree, &sizes, &code.ixs);
        // The dominant contraction touches all three labels: its node tc is
        // log2(2*16*8) = 1 + 4 + 3 = 8. The total is the log-sum-exp over both
        // pairwise nodes, so slightly above 8 (and well below 9) — confirming
        // dimensions are weighted by log2(size), not counted uniformly.
        assert!(cc.tc > 8.0 && cc.tc < 8.5, "tc = {}", cc.tc);
        // Peak intermediate holds the two largest labels b,c: sc = 4 + 3 = 7.
        assert_eq!(cc.sc, 7.0, "sc = {}", cc.sc);
    }

    #[test]
    fn test_determinism() {
        // Same input must produce byte-identical trees across repeated runs.
        let code = EinCode::new(
            vec![
                vec![0usize, 1],
                vec![1, 2],
                vec![2, 3],
                vec![3, 0],
                vec![0, 2],
            ],
            vec![],
        );
        let sizes: HashMap<usize, usize> = [(0, 2), (1, 3), (2, 4), (3, 5)].into_iter().collect();
        let t1 = optimize_code(&code, &sizes, &Treewidth::default()).unwrap();
        let t2 = optimize_code(&code, &sizes, &Treewidth::default()).unwrap();
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_scalar_output_full_contraction() {
        // Trace-like full contraction to a scalar.
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'i']], vec![]);
        let sizes = sizes_uniform(&['i', 'j'], 4);
        let tree = optimize_code(&code, &sizes, &Treewidth::default()).unwrap();
        match &tree {
            NestedEinsum::Node { eins, .. } => assert!(eins.iy.is_empty()),
            NestedEinsum::Leaf { .. } => panic!("2-tensor network must not be a leaf"),
        }
    }

    #[test]
    fn test_partial_output_kept() {
        // A[i,j] B[j,k] -> [i] : output label i must survive to the root.
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i']);
        let sizes = sizes_uniform(&['i', 'j', 'k'], 3);
        let tree = optimize_code(&code, &sizes, &Treewidth::default()).unwrap();
        match &tree {
            NestedEinsum::Node { eins, .. } => assert_eq!(eins.iy, vec!['i']),
            NestedEinsum::Leaf { .. } => panic!("must not be a leaf"),
        }
    }

    #[test]
    fn test_disconnected_components_merged() {
        // Two independent matrix products with no shared labels.
        let code = EinCode::new(
            vec![
                vec!['a', 'b'],
                vec!['b', 'c'],
                vec!['x', 'y'],
                vec!['y', 'z'],
            ],
            vec!['a', 'c', 'x', 'z'],
        );
        let sizes = sizes_uniform(&['a', 'b', 'c', 'x', 'y', 'z'], 2);
        let tree = optimize_code(&code, &sizes, &Treewidth::default()).unwrap();
        assert!(tree.is_binary());
        assert_eq!(tree.leaf_count(), 4);
    }

    #[test]
    fn test_algorithm_enum_and_constructors() {
        assert_eq!(Treewidth::default().alg, EliminationAlgorithm::MinDegree);
        assert_eq!(
            Treewidth::new(EliminationAlgorithm::MinDegree).alg,
            EliminationAlgorithm::MinDegree
        );
        assert_eq!(Treewidth::min_degree().alg, EliminationAlgorithm::MinDegree);
    }
}
