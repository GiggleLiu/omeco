//! Waist surgery: global cut improvement of a contraction tree's dominant node.
//!
//! Pure time complexity is pinned by the tree's highest-cost contraction — its
//! **waist**. The tensors below that argmax node form one side `A` of a
//! whole-network bipartition `(A, B)`, with the complement on side `B`. TreeSA's
//! local rewrites move one subtree at a time and can struggle to jump between
//! distinct good bipartitions of comparable size, so a search can get stuck with
//! an improvable waist. This pass injects global information exactly there.
//!
//! [`refine`] takes an existing contraction tree and repeatedly:
//!
//! 1. **Extracts the waist.** Walk the tree, find the argmax-cost contraction,
//!    and use its descendant tensors against their complement as `(A, B)`.
//! 2. **Improves the cut globally.** On the tensor hypergraph (ignoring the tree)
//!    run bounded [Fiduccia–Mattheyses][fm] passes — gain is the reduction in
//!    summed `log2` dimensions of straddling labels, with a balance constraint
//!    `|A|` within a slack band — seeded from the current cut and from
//!    boundary-BFS alternatives.
//! 3. **Rebuilds two-sided.** If the candidate cut, promoted to the root, is
//!    cheaper than the incumbent waist node, cold-anneal a subtree for each side
//!    separately (each side's open labels derived by outside-occurrence counting
//!    so the scorer agrees), join them at the root, and accept iff the global
//!    `tc` strictly drops. A candidate tied with the incumbent partition cut can
//!    still pass this no-new-bottleneck gate.
//!
//! If the bounded search finds no cheaper balanced cut, the [`WaistReport`]
//! records a `waist_min` event. This is a search diagnostic, not a proof of
//! global cut optimality.
//!
//! [fm]: https://en.wikipedia.org/wiki/Fiduccia%E2%80%93Mattheyses_algorithm
//!
//! # Example
//!
//! ```
//! use omeco::waist_surgery::refine;
//! use omeco::{contraction_complexity, optimize_code, EinCode, GreedyMethod};
//! use std::collections::HashMap;
//! use std::time::Duration;
//!
//! let code = EinCode::new(
//!     vec![
//!         vec!['a', 'b'],
//!         vec!['b', 'c'],
//!         vec!['c', 'd'],
//!         vec!['d', 'a'],
//!     ],
//!     vec![],
//! );
//! let sizes: HashMap<char, usize> = [('a', 2), ('b', 2), ('c', 2), ('d', 2)].into();
//! let seed = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
//! let (refined, report) = refine(&seed, &code, &sizes, Duration::from_millis(50));
//! // The refined tree is always a valid contraction over the original tensors and
//! // never worse than the seed.
//! let seed_tc = contraction_complexity(&seed, &sizes, &code.ixs).tc;
//! let refined_tc = contraction_complexity(&refined, &sizes, &code.ixs).tc;
//! assert!(refined_tc <= seed_tc + 1e-9);
//! assert_eq!(report.n_original, 4);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::eincode::{EinCode, NestedEinsum};
use crate::expr_tree::{
    apply_rule_mut, tree_complexity, DecompositionType, ExprTree, Rule, ScratchSpace,
};
use crate::{contraction_complexity, optimize_code, GreedyMethod, Label};

/// Resync the incremental linear-tc accumulator from an exact recompute every
/// this many sweeps, to bound floating-point drift.
const RESYNC_SWEEPS: u64 = 512;

/// Cold end of every linear beta schedule.
const B_HI: f64 = 14.0;

/// Cold start of the side-rebuild beta schedule.
const B_LO_COLD: f64 = 2.5;

/// FM balance slack around the waist size `|A|` (fraction).
const FM_SLACK: f64 = 0.18;

/// Maximum FM passes per cut-improvement call.
const FM_MAX_PASSES: usize = 6;

/// Number of coarse super-nodes the top span selects (`S_top = ceil(m/30)`).
const TARGET_TOP: usize = 30;

/// Maximum V-cycles annealing each rebuilt side before the deadline cuts it off.
const REBUILD_VCYCLES: u64 = 3;

/// Cold sweeps per span level while rebuilding a side.
const REBUILD_COLD_SWEEPS: u64 = 60;

/// Stop early after this many consecutive non-improving surgery iterations.
const MAX_STALE_ITERS: u64 = 4;

/// Deterministic RNG seed for the surgery FM/anneal streams.
const RNG_SEED: u64 = 0x0000_0054_c0ff_ee00;

/// Diagnostics from a [`refine`] run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaistReport {
    /// Number of tensors in the input network.
    pub n_original: usize,
    /// Number of waist-surgery iterations attempted.
    pub surgery_calls: u64,
    /// Iterations where bounded FM strictly improved the incumbent partition
    /// under the same cut-weight functional.
    pub cheaper_cuts: u64,
    /// Candidates that passed the no-new-bottleneck gate and entered rebuilding.
    pub rebuild_attempts: u64,
    /// Rebuilds that strictly lowered the global time complexity and were kept.
    pub rebuild_accepts: u64,
    /// Iterations where the bounded FM search found no cheaper comparable cut.
    pub waist_min_hits: u64,
}

// =============================================================================
// Reduced-network hypergraph (label-id space) for waist cut surgery.
// =============================================================================

struct Hyper {
    n: usize,
    /// Per tensor: sorted, deduped label-ids.
    tlabels: Vec<Vec<usize>>,
    /// Per label-id: list of tensors containing it.
    label_tensors: Vec<Vec<usize>>,
    /// Per label-id: log2 dimension.
    log2: Vec<f64>,
    /// Per label-id: true if in the final output.
    is_out: Vec<bool>,
    /// Tensor adjacency (share >=1 label), deduped — for boundary BFS seeds.
    adj: Vec<Vec<usize>>,
}

impl Hyper {
    fn build<L: Label>(
        code: &EinCode<L>,
        label_map: &HashMap<L, usize>,
        log2: &[f64],
        nlab: usize,
    ) -> Self {
        let n = code.ixs.len();
        let inputs = &code.ixs;
        let tlabels: Vec<Vec<usize>> = inputs
            .iter()
            .map(|ix| {
                let mut v: Vec<usize> = ix
                    .iter()
                    .filter_map(|l| label_map.get(l).copied())
                    .collect();
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();
        let mut label_tensors: Vec<Vec<usize>> = vec![Vec::new(); nlab];
        for (t, ls) in tlabels.iter().enumerate() {
            for &l in ls {
                label_tensors[l].push(t);
            }
        }
        let mut is_out = vec![false; nlab];
        for l in &code.iy {
            if let Some(&id) = label_map.get(l) {
                is_out[id] = true;
            }
        }
        // Tensor adjacency via shared labels (skip giant hyperedges to bound cost).
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for lt in &label_tensors {
            if lt.len() <= 64 {
                for i in 0..lt.len() {
                    for j in (i + 1)..lt.len() {
                        adj[lt[i]].push(lt[j]);
                        adj[lt[j]].push(lt[i]);
                    }
                }
            }
        }
        for a in adj.iter_mut() {
            a.sort_unstable();
            a.dedup();
        }
        Hyper {
            n,
            tlabels,
            label_tensors,
            log2: log2.to_vec(),
            is_out,
            adj,
        }
    }

    /// Straddle-cut cost of a bipartition: sum of `log2` dims over labels present
    /// on both sides, plus output labels (which are always in the top-node output).
    fn cut_cost(&self, part: &[bool]) -> f64 {
        let mut cost = 0.0;
        for (l, ts) in self.label_tensors.iter().enumerate() {
            if ts.is_empty() {
                continue;
            }
            let mut a = false;
            let mut b = false;
            for &t in ts {
                if part[t] {
                    a = true;
                } else {
                    b = true;
                }
                if a && b {
                    break;
                }
            }
            if (a && b) || self.is_out[l] {
                cost += self.log2[l];
            }
        }
        cost
    }
}

// =============================================================================
// Fiduccia–Mattheyses bipartition refinement on the tensor hypergraph.
// =============================================================================

/// Improve a bipartition by bounded FM passes. `part[t] == true` means side A.
/// Balance is constrained so `|A|` stays within `[lo, hi]`. Returns the improved
/// partition and its straddle-cut cost.
fn fm_refine(
    hyper: &Hyper,
    mut part: Vec<bool>,
    lo: usize,
    hi: usize,
    start: Instant,
    budget: Duration,
) -> (Vec<bool>, f64) {
    let n = hyper.n;
    let nlab = hyper.log2.len();
    let mut cnt_a = vec![0u32; nlab];
    for (t, &p) in part.iter().enumerate() {
        if p {
            for &l in &hyper.tlabels[t] {
                cnt_a[l] += 1;
            }
        }
    }
    let mut size_a = part.iter().filter(|&&p| p).count();

    // Gain of moving tensor t to the other side = reduction in straddle cost.
    let gain_of = |t: usize, part: &[bool], cnt_a: &[u32]| -> f64 {
        let mut g = 0.0;
        for &l in &hyper.tlabels[t] {
            if hyper.is_out[l] {
                continue; // output labels never leave the top-node output.
            }
            let deg = hyper.label_tensors[l].len() as u32;
            let (cs, ct) = if part[t] {
                (cnt_a[l], deg - cnt_a[l])
            } else {
                (deg - cnt_a[l], cnt_a[l])
            };
            let before = (ct >= 1) as i32;
            let after = (cs >= 2) as i32;
            g += hyper.log2[l] * (before - after) as f64;
        }
        g
    };

    let mut best_cost = hyper.cut_cost(&part);

    for _pass in 0..FM_MAX_PASSES {
        if start.elapsed() >= budget {
            break;
        }
        let mut gain: Vec<f64> = (0..n).map(|t| gain_of(t, &part, &cnt_a)).collect();
        let mut locked = vec![false; n];
        let mut cur_cost = hyper.cut_cost(&part);
        let mut best_seen_cost = cur_cost;
        let mut best_seen_part = part.clone();
        let mut improved_this_pass = false;

        for _step in 0..n {
            if start.elapsed() >= budget {
                break;
            }
            // Pick max-gain unlocked feasible move.
            let mut bv: Option<usize> = None;
            let mut bg = f64::NEG_INFINITY;
            for t in 0..n {
                if locked[t] {
                    continue;
                }
                let feasible = if part[t] { size_a > lo } else { size_a < hi };
                if !feasible {
                    continue;
                }
                if gain[t] > bg {
                    bg = gain[t];
                    bv = Some(t);
                }
            }
            let Some(v) = bv else { break };

            // Apply the move.
            cur_cost -= gain[v];
            if part[v] {
                for &l in &hyper.tlabels[v] {
                    cnt_a[l] -= 1;
                }
                size_a -= 1;
            } else {
                for &l in &hyper.tlabels[v] {
                    cnt_a[l] += 1;
                }
                size_a += 1;
            }
            part[v] = !part[v];
            locked[v] = true;

            // Recompute gains of all vertices sharing a label with v.
            let mut touched: Vec<usize> = Vec::new();
            for &l in &hyper.tlabels[v] {
                if hyper.label_tensors[l].len() > 256 {
                    continue;
                }
                for &u in &hyper.label_tensors[l] {
                    touched.push(u);
                }
            }
            touched.sort_unstable();
            touched.dedup();
            for &u in &touched {
                gain[u] = gain_of(u, &part, &cnt_a);
            }

            if cur_cost < best_seen_cost - 1e-12 {
                best_seen_cost = cur_cost;
                best_seen_part = part.clone();
                improved_this_pass = true;
            }
        }

        // Roll back to the best cut state seen in this pass.
        part = best_seen_part;
        cnt_a.iter_mut().for_each(|c| *c = 0);
        size_a = 0;
        for (t, &p) in part.iter().enumerate() {
            if p {
                size_a += 1;
                for &l in &hyper.tlabels[t] {
                    cnt_a[l] += 1;
                }
            }
        }
        best_cost = best_seen_cost;
        if !improved_this_pass {
            break;
        }
    }
    (part, best_cost)
}

/// Grow a connected region of `target` tensors by BFS from `seed` over the tensor
/// adjacency; the region is side A.
fn bfs_seed(hyper: &Hyper, seed: usize, target: usize) -> Vec<bool> {
    let n = hyper.n;
    let mut part = vec![false; n];
    let mut queue = VecDeque::new();
    queue.push_back(seed);
    part[seed] = true;
    let mut count = 1usize;
    while count < target {
        let Some(t) = queue.pop_front() else { break };
        for &u in &hyper.adj[t] {
            if !part[u] {
                part[u] = true;
                count += 1;
                queue.push_back(u);
                if count >= target {
                    break;
                }
            }
        }
    }
    part
}

// =============================================================================
// Waist extraction and node cost.
// =============================================================================

/// Return the argmax contraction's node cost and descendant tensor ids.
///
/// Those descendants form side A of the whole-network waist bipartition; every
/// other input tensor forms side B. A root argmax has an empty complement, so the
/// caller treats it as a non-actionable waist.
fn extract_waist(tree: &ExprTree, log2_sizes: &[f64]) -> Option<(f64, Vec<usize>)> {
    fn walk(
        tree: &ExprTree,
        log2_sizes: &[f64],
        best: &mut f64,
        best_leaves: &mut Vec<usize>,
    ) -> Vec<usize> {
        match tree {
            ExprTree::Leaf(info) => vec![info.tensor_id.unwrap_or(0)],
            ExprTree::Node { left, right, info } => {
                let lleaves = walk(left, log2_sizes, best, best_leaves);
                let rleaves = walk(right, log2_sizes, best, best_leaves);
                let cost = node_tc(left.labels(), right.labels(), &info.out_dims, log2_sizes);
                let mut leaves = lleaves;
                leaves.extend_from_slice(&rleaves);
                if cost > *best {
                    *best = cost;
                    *best_leaves = leaves.clone();
                }
                leaves
            }
        }
    }
    if matches!(tree, ExprTree::Leaf(_)) {
        return None;
    }
    let mut best = f64::NEG_INFINITY;
    let mut best_leaves = Vec::new();
    walk(tree, log2_sizes, &mut best, &mut best_leaves);
    Some((best, best_leaves))
}

/// Exact contraction-node cost used by the TreeSA objective.
fn node_tc(ix1: &[usize], ix2: &[usize], iy: &[usize], log2_sizes: &[f64]) -> f64 {
    let mut tc: f64 = iy.iter().map(|&l| log2_sizes[l]).sum();
    for &l in ix1 {
        if ix2.contains(&l) && !iy.contains(&l) {
            tc += log2_sizes[l];
        }
    }
    tc
}

// =============================================================================
// Span-gated SA sweep + conversions (label-id space).
// =============================================================================

/// One span-gated SA sweep: only rewrite nodes whose leaf span is `>= min_span`.
fn gated_sweep(
    tree: &mut ExprTree,
    beta: f64,
    min_span: usize,
    log2_sizes: &[f64],
    rng: &mut SmallRng,
    scratch: &mut ScratchSpace,
    s_lin: &mut f64,
) -> usize {
    match tree {
        ExprTree::Leaf(_) => 1,
        ExprTree::Node { left, right, .. } => {
            let ls = gated_sweep(left, beta, min_span, log2_sizes, rng, scratch, s_lin);
            let rs = gated_sweep(right, beta, min_span, log2_sizes, rng, scratch, s_lin);
            let span = ls + rs;
            if span >= min_span {
                let rules = Rule::applicable_rules(tree, DecompositionType::Tree);
                if !rules.is_empty() {
                    let rule = rules[rng.random_range(0..rules.len())];
                    if let Some(diff) = scratch.rule_diff(tree, rule, log2_sizes, false) {
                        let dtc = diff.tc1 - diff.tc0;
                        if dtc <= 0.0 || rng.random::<f64>() < (-beta * dtc).exp() {
                            *s_lin += f64::exp2(diff.tc1) - f64::exp2(diff.tc0);
                            apply_rule_mut(tree, rule, diff.new_labels);
                        }
                    }
                }
            }
            span
        }
    }
}

/// Convert an at-most-binary `NestedEinsum` into an `ExprTree` in label-id space.
/// Unary reductions are fused into their child interface because `ExprTree` is
/// binary-only. `label_map` maps label values to their bit index.
fn nested_to_expr_tree(
    nested: &NestedEinsum<usize>,
    label_map: &HashMap<usize, usize>,
) -> Option<ExprTree> {
    match nested {
        NestedEinsum::Leaf { .. } => None,
        NestedEinsum::Node { args, eins } => match args.as_slice() {
            [child] => {
                let input_dims: Vec<usize> = eins
                    .ixs
                    .first()?
                    .iter()
                    .filter_map(|l| label_map.get(l).copied())
                    .collect();
                let out_dims: Vec<usize> = eins
                    .iy
                    .iter()
                    .filter_map(|l| label_map.get(l).copied())
                    .collect();
                let mut tree = match child {
                    NestedEinsum::Leaf { tensor_index } => {
                        ExprTree::leaf(input_dims.clone(), *tensor_index)
                    }
                    NestedEinsum::Node { .. } => nested_to_expr_tree(child, label_map)?,
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
                let child = |arg: &NestedEinsum<usize>, side: usize| -> Option<ExprTree> {
                    match arg {
                        NestedEinsum::Leaf { tensor_index } => {
                            let out_dims: Vec<usize> = eins.ixs[side]
                                .iter()
                                .filter_map(|l| label_map.get(l).copied())
                                .collect();
                            Some(ExprTree::leaf(out_dims, *tensor_index))
                        }
                        NestedEinsum::Node { .. } => nested_to_expr_tree(arg, label_map),
                    }
                };
                let left = child(left_arg, 0)?;
                let right = child(right_arg, 1)?;
                let out_dims: Vec<usize> = eins
                    .iy
                    .iter()
                    .filter_map(|l| label_map.get(l).copied())
                    .collect();
                Some(ExprTree::node(left, right, out_dims))
            }
            _ => None,
        },
    }
}

/// Convert an `ExprTree` back into a `NestedEinsum`, deriving every node's output
/// by **outside-occurrence counting** over `ixs` (a label is a node's output iff
/// it occurs in a tensor outside the node's subtree or is an open/output label).
/// This makes the emitted eins bodies topology-consistent regardless of any stale
/// `out_dims` cached on the tree — leaves index into `ixs`.
fn expr_to_nested_counted(
    tree: &ExprTree,
    ixs: &[Vec<usize>],
    open: &[usize],
) -> NestedEinsum<usize> {
    // Global occurrence count of each label across all tensors.
    let mut global_count: HashMap<usize, usize> = HashMap::new();
    for ix in ixs {
        for &l in ix {
            *global_count.entry(l).or_insert(0) += 1;
        }
    }
    let open_set: HashSet<usize> = open.iter().copied().collect();

    // Returns (nested, within-subtree counts, this subtree's output labels).
    fn rec(
        tree: &ExprTree,
        ixs: &[Vec<usize>],
        open_set: &HashSet<usize>,
        global_count: &HashMap<usize, usize>,
    ) -> (NestedEinsum<usize>, HashMap<usize, usize>, Vec<usize>) {
        match tree {
            ExprTree::Leaf(info) => {
                let tid = info.tensor_id.unwrap_or(0);
                let labels = ixs.get(tid).cloned().unwrap_or_default();
                let mut within: HashMap<usize, usize> = HashMap::new();
                for &l in &labels {
                    *within.entry(l).or_insert(0) += 1;
                }
                (NestedEinsum::leaf(tid), within, labels)
            }
            ExprTree::Node { left, right, .. } => {
                let (ln, lw, lout) = rec(left, ixs, open_set, global_count);
                let (rn, rw, rout) = rec(right, ixs, open_set, global_count);
                let mut within = lw;
                for (l, c) in rw {
                    *within.entry(l).or_insert(0) += c;
                }
                // A label is an output of this node iff it appears outside the
                // subtree (within < global) or is an open/output label.
                let mut out: Vec<usize> = within
                    .iter()
                    .filter(|(&l, &c)| open_set.contains(&l) || c < global_count[&l])
                    .map(|(&l, _)| l)
                    .collect();
                out.sort_unstable();
                let eins = EinCode::new(vec![lout, rout], out.clone());
                (NestedEinsum::node(vec![ln, rn], eins), within, out)
            }
        }
    }
    rec(tree, ixs, &open_set, &global_count).0
}

/// Remap the leaf tensor indices of a `NestedEinsum` through `map` (leaf i -> map[i]).
fn remap_leaves(tree: &NestedEinsum<usize>, map: &[usize]) -> NestedEinsum<usize> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => NestedEinsum::leaf(map[*tensor_index]),
        NestedEinsum::Node { args, eins } => {
            let new_args = args.iter().map(|a| remap_leaves(a, map)).collect();
            NestedEinsum::node(new_args, eins.clone())
        }
    }
}

// =============================================================================
// The waist refiner.
// =============================================================================

struct Refiner<'a> {
    code: &'a EinCode<usize>,
    sizes: &'a HashMap<usize, usize>,
    hyper: &'a Hyper,
    log2_sizes: &'a [f64],
    start: Instant,
    budget: Duration,
    rng: SmallRng,
    report: WaistReport,
}

impl Refiner<'_> {
    fn out_of_time(&self) -> bool {
        self.start.elapsed() >= self.budget
    }

    /// One waist-surgery pass on `incumbent` (best NestedEinsum, tc `work_tc`).
    /// Returns an accepted (tree, tc) strictly better than `work_tc`, or `None`.
    fn waist_surgery(
        &mut self,
        incumbent: &ExprTree,
        work_tc: f64,
    ) -> Option<(NestedEinsum<usize>, f64)> {
        self.report.surgery_calls += 1;
        let (waist_node_cost, a_leaves) = extract_waist(incumbent, self.log2_sizes)?;
        let n = self.hyper.n;
        if a_leaves.is_empty() || a_leaves.len() >= n {
            return None;
        }
        let mut cur_part = vec![false; n];
        for &t in &a_leaves {
            if t < n {
                cur_part[t] = true;
            }
        }
        let incumbent_cut_cost = self.hyper.cut_cost(&cur_part);
        let target_a = a_leaves.len();
        let lo = ((target_a as f64 * (1.0 - FM_SLACK)).floor() as usize).max(1);
        let hi = ((target_a as f64 * (1.0 + FM_SLACK)).ceil() as usize).min(n - 1);

        // Candidate seeds: current cut + boundary-BFS alternatives.
        let mut seeds: Vec<Vec<bool>> = vec![cur_part];
        let mut deg_order: Vec<usize> = (0..n).collect();
        deg_order.sort_by_key(|&t| std::cmp::Reverse(self.hyper.tlabels[t].len()));
        seeds.push(bfs_seed(self.hyper, deg_order[0], target_a));
        for _ in 0..3 {
            let seed = self.rng.random_range(0..n);
            seeds.push(bfs_seed(self.hyper, seed, target_a));
        }

        let mut best_alt_cost = f64::INFINITY;
        let mut best_alt_part: Option<Vec<bool>> = None;
        for seed in seeds {
            if self.out_of_time() {
                break;
            }
            let (part, _) = fm_refine(self.hyper, seed, lo, hi, self.start, self.budget);
            let sa = part.iter().filter(|&&p| p).count();
            if sa < lo || sa > hi {
                continue;
            }
            // FM's incremental cost can drift when gain recomputation is
            // skipped for giant hyperedges; rescore exactly before candidates
            // are compared against each other and the acceptance gates.
            let cost = self.hyper.cut_cost(&part);
            if cost < best_alt_cost {
                best_alt_cost = cost;
                best_alt_part = Some(part);
            }
        }

        let Some(part) = best_alt_part else {
            if !self.out_of_time() {
                self.report.waist_min_hits += 1;
            }
            return None;
        };
        if best_alt_cost < incumbent_cut_cost - 1e-9 {
            self.report.cheaper_cuts += 1;
        } else {
            self.report.waist_min_hits += 1;
        }
        // The candidate becomes the rebuilt tree's top contraction. Compare that
        // exact top-node cost with the incumbent global bottleneck; it need not
        // strictly improve the incumbent partition cut to justify rebuilding.
        if best_alt_cost >= waist_node_cost - 1e-9 {
            return None;
        }
        self.report.rebuild_attempts += 1;

        // Rebuild both sides from the improved cut.
        if self.out_of_time() {
            return None;
        }
        let rebuilt = self.rebuild(&part)?;
        if self.out_of_time() {
            return None;
        }
        let new_tc = contraction_complexity(&rebuilt, self.sizes, &self.code.ixs).tc;
        if new_tc < work_tc - 1e-9 {
            self.report.rebuild_accepts += 1;
            Some((rebuilt, new_tc))
        } else {
            None
        }
    }

    /// Rebuild the whole tree from a top-level bipartition `part`.
    fn rebuild(&mut self, part: &[bool]) -> Option<NestedEinsum<usize>> {
        let n = self.hyper.n;
        let a_tensors: Vec<usize> = (0..n).filter(|&t| part[t]).collect();
        let b_tensors: Vec<usize> = (0..n).filter(|&t| !part[t]).collect();
        if a_tensors.is_empty() || b_tensors.is_empty() {
            return None;
        }
        let (open_a, open_b) = self.side_open_labels(part);
        let sub_a = self.solve_side(&a_tensors, &open_a)?;
        if self.out_of_time() {
            return None;
        }
        let sub_b = self.solve_side(&b_tensors, &open_b)?;
        let eins = EinCode::new(vec![open_a, open_b], self.code.iy.clone());
        Some(NestedEinsum::node(vec![sub_a, sub_b], eins))
    }

    /// Open label-ids for each side: a label is open on side A iff it occurs in A
    /// and (occurs in B or is an output label). Symmetric for B.
    fn side_open_labels(&self, part: &[bool]) -> (Vec<usize>, Vec<usize>) {
        let nlab = self.hyper.log2.len();
        let mut open_a = Vec::new();
        let mut open_b = Vec::new();
        for l in 0..nlab {
            let ts = &self.hyper.label_tensors[l];
            if ts.is_empty() {
                continue;
            }
            let mut in_a = false;
            let mut in_b = false;
            for &t in ts {
                if part[t] {
                    in_a = true;
                } else {
                    in_b = true;
                }
            }
            let out = self.hyper.is_out[l];
            if in_a && (in_b || out) {
                open_a.push(l);
            }
            if in_b && (in_a || out) {
                open_b.push(l);
            }
        }
        (open_a, open_b)
    }

    /// Solve a side sub-einsum: greedy + a fixed number of cold span-gated anneal
    /// V-cycles. Returns a `NestedEinsum` over original (reduced) tensor indices.
    fn solve_side(&mut self, tensors: &[usize], open: &[usize]) -> Option<NestedEinsum<usize>> {
        if tensors.is_empty() || self.out_of_time() {
            return None;
        }
        if tensors.len() == 1 {
            let tensor_index = tensors[0];
            let input = self.code.ixs.get(tensor_index)?.clone();
            let leaf = NestedEinsum::leaf(tensor_index);
            return if input == open {
                Some(leaf)
            } else {
                Some(NestedEinsum::node(
                    vec![leaf],
                    EinCode::new(vec![input], open.to_vec()),
                ))
            };
        }
        let sub_ixs: Vec<Vec<usize>> = tensors.iter().map(|&t| self.code.ixs[t].clone()).collect();
        let sub_code = EinCode::new(sub_ixs.clone(), open.to_vec());
        let greedy = optimize_code(&sub_code, self.sizes, &GreedyMethod::default())?;
        if self.out_of_time() {
            return None;
        }
        let sub_labels: Vec<usize> = sub_code.unique_labels();
        let sub_label_map: HashMap<usize, usize> = sub_labels
            .iter()
            .enumerate()
            .map(|(i, &l)| (l, i))
            .collect();
        let sub_log2: Vec<f64> = sub_labels
            .iter()
            .map(|&l| (*self.sizes.get(&l).unwrap_or(&1) as f64).log2())
            .collect();

        let mut sub_best = greedy;
        if let Some(mut tree) = nested_to_expr_tree(&sub_best, &sub_label_map) {
            let mut scratch = ScratchSpace::new(sub_labels.len());
            let m = tree.leaf_count();
            let s_top = ((m + TARGET_TOP - 1) / TARGET_TOP).max(2);
            let mut spans: Vec<usize> = Vec::new();
            let mut s = s_top;
            while s > 2 {
                spans.push(s);
                s /= 2;
            }
            spans.push(2);
            let mut s_lin = f64::exp2(tree_complexity(&tree, &sub_log2).0);
            let mut best_lin = s_lin;
            let mut best_tree = tree.clone();
            let mut sweeps: u64 = 0;
            'anneal: for _vc in 0..REBUILD_VCYCLES.max(1) {
                for &span in &spans {
                    let denom = (REBUILD_COLD_SWEEPS.saturating_sub(1)).max(1) as f64;
                    for k in 0..REBUILD_COLD_SWEEPS {
                        if self.out_of_time() {
                            break 'anneal;
                        }
                        let beta = B_LO_COLD + (B_HI - B_LO_COLD) * (k as f64 / denom);
                        gated_sweep(
                            &mut tree,
                            beta,
                            span,
                            &sub_log2,
                            &mut self.rng,
                            &mut scratch,
                            &mut s_lin,
                        );
                        sweeps += 1;
                        if sweeps % RESYNC_SWEEPS == 0 {
                            s_lin = f64::exp2(tree_complexity(&tree, &sub_log2).0);
                        }
                    }
                    s_lin = f64::exp2(tree_complexity(&tree, &sub_log2).0);
                    if s_lin < best_lin - 1e-9 {
                        best_lin = s_lin;
                        best_tree = tree.clone();
                    }
                }
                tree = best_tree.clone();
                s_lin = best_lin;
            }
            sub_best = expr_to_nested_counted(&best_tree, &sub_ixs, open);
        }
        Some(remap_leaves(&sub_best, tensors))
    }
}

/// Refine a contraction tree by repeated waist surgery within a wall-clock budget.
///
/// `tree` is any valid contraction tree over `code`'s tensors (e.g. a greedy or
/// TreeSA result). The returned tree is over the same original tensor indices and
/// is **never worse** than the input by time complexity. See the [module
/// documentation](self) for the algorithm. `budget` is checked within FM and
/// annealing loops and between rebuild stages; an individual synchronous greedy
/// initialization already in progress cannot be interrupted. The pass also
/// returns early once it can no longer improve.
///
/// The RNG seed is fixed, but how much work fits inside `budget` depends on
/// wall-clock speed, so results are **not** reproducible across machines or
/// loads — only the never-worse guarantee is.
///
/// # Example
///
/// ```
/// use omeco::waist_surgery::refine;
/// use omeco::{optimize_code, EinCode, GreedyMethod};
/// use std::collections::HashMap;
/// use std::time::Duration;
///
/// let code = EinCode::new(
///     vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
///     vec!['i', 'l'],
/// );
/// let sizes: HashMap<char, usize> = [('i', 2), ('j', 2), ('k', 2), ('l', 2)].into();
/// let seed = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
/// let (refined, report) = refine(&seed, &code, &sizes, Duration::from_millis(20));
/// assert_eq!(refined.leaf_count(), 3);
/// assert!(report.surgery_calls >= 1);
/// ```
pub fn refine<L: Label>(
    tree: &NestedEinsum<L>,
    code: &EinCode<L>,
    sizes: &HashMap<L, usize>,
    budget: Duration,
) -> (NestedEinsum<L>, WaistReport) {
    let n = code.num_tensors();
    let mut report = WaistReport {
        n_original: n,
        surgery_calls: 0,
        cheaper_cuts: 0,
        rebuild_attempts: 0,
        rebuild_accepts: 0,
        waist_min_hits: 0,
    };
    // Nothing to do for trivial networks.
    if n < 3 {
        return (tree.clone(), report);
    }

    // Build the label-id space shared by the ExprTree, tc, FM, and conversions.
    let labels: Vec<L> = code.unique_labels();
    let label_map: HashMap<L, usize> = labels
        .iter()
        .enumerate()
        .map(|(i, l)| (l.clone(), i))
        .collect();
    let log2_sizes: Vec<f64> = labels
        .iter()
        .map(|l| (*sizes.get(l).unwrap_or(&1) as f64).log2())
        .collect();
    let hyper = Hyper::build(code, &label_map, &log2_sizes, labels.len());

    // Work entirely in usize label-id space internally.
    let inputs = &code.ixs;
    let id_ixs: Vec<Vec<usize>> = inputs
        .iter()
        .map(|ix| {
            ix.iter()
                .filter_map(|l| label_map.get(l).copied())
                .collect()
        })
        .collect();
    let outputs = &code.iy;
    let id_iy: Vec<usize> = outputs
        .iter()
        .filter_map(|l| label_map.get(l).copied())
        .collect();
    let id_code = EinCode::new(id_ixs.clone(), id_iy.clone());
    let id_sizes: HashMap<usize, usize> = labels
        .iter()
        .enumerate()
        .map(|(i, l)| (i, *sizes.get(l).unwrap_or(&1)))
        .collect();

    // Convert the seed into id-space NestedEinsum, then ExprTree.
    let id_label_map: HashMap<usize, usize> = (0..labels.len()).map(|i| (i, i)).collect();
    let seed_id = relabel_nested(tree, &label_map);
    let mut best = seed_id.clone();
    let mut best_tc = contraction_complexity(&best, &id_sizes, &id_ixs).tc;

    let mut refiner = Refiner {
        code: &id_code,
        sizes: &id_sizes,
        hyper: &hyper,
        log2_sizes: &log2_sizes,
        start: Instant::now(),
        budget,
        rng: SmallRng::seed_from_u64(RNG_SEED),
        report,
    };

    let mut stale: u64 = 0;
    while !refiner.out_of_time() && stale < MAX_STALE_ITERS {
        let Some(incumbent) = nested_to_expr_tree(&best, &id_label_map) else {
            break;
        };
        match refiner.waist_surgery(&incumbent, best_tc) {
            Some((new_tree, new_tc)) if new_tc < best_tc - 1e-9 => {
                best = new_tree;
                best_tc = new_tc;
                stale = 0;
            }
            _ => stale += 1,
        }
    }
    report = refiner.report;

    // Map id-space leaves/eins back to the original label space.
    let best_l = restore_nested(&best, &labels);
    (best_l, report)
}

/// Relabel a `NestedEinsum<L>` into id-space (`usize` labels via `label_map`),
/// preserving leaf tensor indices.
fn relabel_nested<L: Label>(
    tree: &NestedEinsum<L>,
    label_map: &HashMap<L, usize>,
) -> NestedEinsum<usize> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => NestedEinsum::leaf(*tensor_index),
        NestedEinsum::Node { args, eins } => {
            let map_ls = |ls: &[L]| -> Vec<usize> {
                ls.iter()
                    .filter_map(|l| label_map.get(l).copied())
                    .collect()
            };
            let ixs: Vec<Vec<usize>> = eins.ixs.iter().map(|ix| map_ls(ix)).collect();
            let iy = map_ls(&eins.iy);
            let new_args = args.iter().map(|a| relabel_nested(a, label_map)).collect();
            NestedEinsum::node(new_args, EinCode::new(ixs, iy))
        }
    }
}

/// Inverse of [`relabel_nested`]: map id-space labels back to `L` via `labels`.
fn restore_nested<L: Label>(tree: &NestedEinsum<usize>, labels: &[L]) -> NestedEinsum<L> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => NestedEinsum::leaf(*tensor_index),
        NestedEinsum::Node { args, eins } => {
            let map_ls =
                |ls: &[usize]| -> Vec<L> { ls.iter().map(|&i| labels[i].clone()).collect() };
            let ixs: Vec<Vec<L>> = eins.ixs.iter().map(|ix| map_ls(ix)).collect();
            let iy = map_ls(&eins.iy);
            let new_args = args.iter().map(|a| restore_nested(a, labels)).collect();
            NestedEinsum::node(new_args, EinCode::new(ixs, iy))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{optimize_code, Treewidth};

    fn uniform_sizes(code: &EinCode<usize>, d: usize) -> HashMap<usize, usize> {
        code.unique_labels().into_iter().map(|l| (l, d)).collect()
    }

    /// Build a 2D periodic grid tensor network (each bond a distinct label).
    fn grid(rows: usize, cols: usize) -> EinCode<usize> {
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

    #[test]
    fn test_refine_never_worse_than_seed() {
        let code = grid(4, 4);
        let sizes = uniform_sizes(&code, 2);
        let seed = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
        let seed_tc = contraction_complexity(&seed, &sizes, &code.ixs).tc;
        let (refined, report) = refine(&seed, &code, &sizes, Duration::from_millis(500));
        let refined_tc = contraction_complexity(&refined, &sizes, &code.ixs).tc;
        assert!(
            refined_tc <= seed_tc + 1e-9,
            "refined_tc={refined_tc} seed_tc={seed_tc}"
        );
        assert_eq!(report.n_original, code.num_tensors());
        assert!(report.surgery_calls >= 1);
    }

    #[test]
    fn test_refine_preserves_leaf_permutation() {
        let code = grid(4, 4);
        let sizes = uniform_sizes(&code, 2);
        let seed = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
        let (refined, _) = refine(&seed, &code, &sizes, Duration::from_millis(300));
        let mut leaves = refined.leaf_indices();
        leaves.sort_unstable();
        assert_eq!(leaves, (0..code.num_tensors()).collect::<Vec<_>>());
    }

    #[test]
    fn test_refine_trivial_network_is_identity() {
        let code = EinCode::new(vec![vec![0usize, 1], vec![1, 2]], vec![0, 2]);
        let sizes = uniform_sizes(&code, 4);
        let seed = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
        let (refined, report) = refine(&seed, &code, &sizes, Duration::from_millis(50));
        assert_eq!(refined.leaf_count(), 2);
        assert_eq!(report.surgery_calls, 0);
    }

    #[test]
    fn test_extract_waist_finds_non_root_argmax() {
        let right = ExprTree::node(
            ExprTree::leaf(vec![0], 1),
            ExprTree::leaf(vec![1], 2),
            vec![0, 1],
        );
        let root = ExprTree::node(ExprTree::leaf(vec![0], 0), right, vec![]);

        assert_eq!(extract_waist(&root, &[1.0, 1.0]), Some((2.0, vec![1, 2])));
    }

    #[test]
    fn test_fm_refine_observes_expired_budget() {
        let code = grid(2, 2);
        let labels = code.unique_labels();
        let label_map: HashMap<usize, usize> =
            labels.iter().enumerate().map(|(i, &l)| (l, i)).collect();
        let log2 = vec![1.0; labels.len()];
        let hyper = Hyper::build(&code, &label_map, &log2, labels.len());
        let initial = vec![true, true, false, false];
        let initial_cost = hyper.cut_cost(&initial);

        let (part, cost) = fm_refine(
            &hyper,
            initial.clone(),
            1,
            3,
            Instant::now(),
            Duration::ZERO,
        );

        assert_eq!(part, initial);
        assert_eq!(cost, initial_cost);
    }

    #[test]
    fn test_expired_search_is_not_reported_as_waist_minimum() {
        let code = grid(2, 2);
        let sizes = uniform_sizes(&code, 2);
        let labels = code.unique_labels();
        let label_map: HashMap<usize, usize> =
            labels.iter().enumerate().map(|(i, &l)| (l, i)).collect();
        let log2: Vec<f64> = labels
            .iter()
            .map(|label| (sizes[label] as f64).log2())
            .collect();
        let hyper = Hyper::build(&code, &label_map, &log2, labels.len());
        let seed = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
        let incumbent = nested_to_expr_tree(&seed, &label_map).unwrap();
        let mut refiner = Refiner {
            code: &code,
            sizes: &sizes,
            hyper: &hyper,
            log2_sizes: &log2,
            start: Instant::now(),
            budget: Duration::ZERO,
            rng: SmallRng::seed_from_u64(RNG_SEED),
            report: WaistReport {
                n_original: code.num_tensors(),
                surgery_calls: 0,
                cheaper_cuts: 0,
                rebuild_attempts: 0,
                rebuild_accepts: 0,
                waist_min_hits: 0,
            },
        };

        assert!(refiner.waist_surgery(&incumbent, f64::INFINITY).is_none());
        assert_eq!(refiner.report.waist_min_hits, 0);
    }

    #[test]
    fn test_singleton_side_materializes_private_leg_reduction() {
        let code = EinCode::new(vec![vec![0usize, 1], vec![0, 2], vec![0, 3]], vec![]);
        let sizes: HashMap<usize, usize> = [(0, 2), (1, 8), (2, 2), (3, 2)].into();
        let label_map: HashMap<usize, usize> = (0..4).map(|label| (label, label)).collect();
        let log2: Vec<f64> = (0..4).map(|label| (sizes[&label] as f64).log2()).collect();
        let hyper = Hyper::build(&code, &label_map, &log2, 4);
        let mut refiner = Refiner {
            code: &code,
            sizes: &sizes,
            hyper: &hyper,
            log2_sizes: &log2,
            start: Instant::now(),
            budget: Duration::from_secs(1),
            rng: SmallRng::seed_from_u64(RNG_SEED),
            report: WaistReport {
                n_original: code.num_tensors(),
                surgery_calls: 0,
                cheaper_cuts: 0,
                rebuild_attempts: 0,
                rebuild_accepts: 0,
                waist_min_hits: 0,
            },
        };

        let side = refiner.solve_side(&[0], &[0]).unwrap();

        match &side {
            NestedEinsum::Node { args, eins } => {
                assert_eq!(args.len(), 1);
                assert_eq!(eins.ixs, vec![vec![0, 1]]);
                assert_eq!(eins.iy, vec![0]);
                assert_eq!(args[0].output_labels(&code.ixs), eins.ixs[0]);
            }
            NestedEinsum::Leaf { .. } => panic!("private leg requires a unary reduction"),
        }
        let cc = contraction_complexity(&side, &sizes, &code.ixs);
        assert!((cc.tc - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_surgery_rebuilds_a_strictly_cheaper_argmax_cut() {
        // Four tensors on a ring. The first argmax node contains alternating
        // vertices {0,2}; its complement is {1,3}, cutting every bond. A
        // contiguous whole-network cut is cheaper.
        let code = EinCode::new(
            vec![vec![0usize, 3], vec![0, 1], vec![1, 2], vec![2, 3]],
            vec![],
        );
        let sizes = uniform_sizes(&code, 2);
        let label_map: HashMap<usize, usize> = (0..4).map(|label| (label, label)).collect();
        let log2 = vec![1.0; 4];
        let hyper = Hyper::build(&code, &label_map, &log2, 4);
        let left = ExprTree::node(
            ExprTree::leaf(code.ixs[0].clone(), 0),
            ExprTree::leaf(code.ixs[2].clone(), 2),
            vec![0, 1, 2, 3],
        );
        let right = ExprTree::node(
            ExprTree::leaf(code.ixs[1].clone(), 1),
            ExprTree::leaf(code.ixs[3].clone(), 3),
            vec![0, 1, 2, 3],
        );
        let incumbent = ExprTree::node(left, right, vec![]);
        let mut refiner = Refiner {
            code: &code,
            sizes: &sizes,
            hyper: &hyper,
            log2_sizes: &log2,
            start: Instant::now(),
            budget: Duration::from_secs(2),
            rng: SmallRng::seed_from_u64(RNG_SEED),
            report: WaistReport {
                n_original: code.num_tensors(),
                surgery_calls: 0,
                cheaper_cuts: 0,
                rebuild_attempts: 0,
                rebuild_accepts: 0,
                waist_min_hits: 0,
            },
        };

        let (rebuilt, rebuilt_tc) = refiner
            .waist_surgery(&incumbent, f64::INFINITY)
            .expect("alternating ring cut should be replaced");

        assert!(rebuilt_tc.is_finite());
        assert_eq!(rebuilt.leaf_count(), code.num_tensors());
        assert_eq!(refiner.report.cheaper_cuts, 1);
        assert_eq!(refiner.report.rebuild_attempts, 1);
        assert_eq!(refiner.report.rebuild_accepts, 1);
        let mut leaves = rebuilt.leaf_indices();
        leaves.sort_unstable();
        assert_eq!(leaves, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_tied_cut_can_pass_no_new_bottleneck_gate() {
        // A four-cycle has minimum bisection cost two. The incumbent waist
        // subtree {0,1} has that minimum cut, but its contraction node costs
        // three because it also contracts the internal 0 bond. The paper
        // algorithm may therefore rebuild a tied alternative: promoted to the
        // root it cannot reproduce the old bottleneck.
        let code = EinCode::new(
            vec![vec![0usize, 1], vec![0, 2], vec![1, 3], vec![2, 3]],
            vec![],
        );
        let sizes = uniform_sizes(&code, 2);
        let label_map: HashMap<usize, usize> = (0..4).map(|label| (label, label)).collect();
        let log2 = vec![1.0; 4];
        let hyper = Hyper::build(&code, &label_map, &log2, 4);
        let left = ExprTree::node(
            ExprTree::leaf(code.ixs[0].clone(), 0),
            ExprTree::leaf(code.ixs[1].clone(), 1),
            vec![1, 2],
        );
        let right = ExprTree::node(
            ExprTree::leaf(code.ixs[2].clone(), 2),
            ExprTree::leaf(code.ixs[3].clone(), 3),
            vec![1, 2],
        );
        let incumbent = ExprTree::node(left, right, vec![]);
        assert_eq!(extract_waist(&incumbent, &log2), Some((3.0, vec![0, 1])));

        let mut refiner = Refiner {
            code: &code,
            sizes: &sizes,
            hyper: &hyper,
            log2_sizes: &log2,
            start: Instant::now(),
            budget: Duration::from_secs(2),
            rng: SmallRng::seed_from_u64(RNG_SEED),
            report: WaistReport {
                n_original: code.num_tensors(),
                surgery_calls: 0,
                cheaper_cuts: 0,
                rebuild_attempts: 0,
                rebuild_accepts: 0,
                waist_min_hits: 0,
            },
        };

        assert!(refiner.waist_surgery(&incumbent, f64::INFINITY).is_some());
        assert_eq!(refiner.report.cheaper_cuts, 0);
        assert_eq!(refiner.report.waist_min_hits, 1);
        assert_eq!(refiner.report.rebuild_attempts, 1);
        assert_eq!(refiner.report.rebuild_accepts, 1);
    }

    #[test]
    fn test_conversion_and_partition_edge_cases() {
        let label_map: HashMap<usize, usize> = (0..3).map(|label| (label, label)).collect();
        assert!(nested_to_expr_tree(&NestedEinsum::leaf(0), &label_map).is_none());
        assert_eq!(extract_waist(&ExprTree::leaf(vec![0], 0), &[1.0]), None);

        let nary = NestedEinsum::node(
            vec![
                NestedEinsum::leaf(0),
                NestedEinsum::leaf(1),
                NestedEinsum::leaf(2),
            ],
            EinCode::new(vec![vec![0], vec![1], vec![2]], vec![]),
        );
        assert!(nested_to_expr_tree(&nary, &label_map).is_none());

        let binary = NestedEinsum::node(
            vec![NestedEinsum::leaf(0), NestedEinsum::leaf(1)],
            EinCode::new(vec![vec![0, 1], vec![1, 2]], vec![0, 2]),
        );
        let unary = NestedEinsum::node(vec![binary], EinCode::new(vec![vec![0, 2]], vec![0]));
        let converted = nested_to_expr_tree(&unary, &label_map)
            .expect("unary node around a binary child should be fused");
        assert_eq!(converted.labels(), &[0]);

        let root = ExprTree::node(
            ExprTree::node(
                ExprTree::leaf(vec![0], 0),
                ExprTree::leaf(vec![1], 1),
                vec![0, 1],
            ),
            ExprTree::leaf(vec![2], 2),
            vec![],
        );
        assert_eq!(extract_waist(&root, &[1.0; 3]), Some((2.0, vec![0, 1])));
    }

    #[test]
    fn test_root_argmax_is_non_actionable() {
        let code = EinCode::new(vec![vec![0usize], vec![1], vec![2]], vec![0, 1, 2]);
        let sizes = uniform_sizes(&code, 2);
        let label_map: HashMap<usize, usize> = (0..3).map(|label| (label, label)).collect();
        let log2 = vec![1.0; 3];
        let hyper = Hyper::build(&code, &label_map, &log2, 3);
        let right = ExprTree::node(
            ExprTree::leaf(vec![1], 1),
            ExprTree::leaf(vec![2], 2),
            vec![1, 2],
        );
        let incumbent = ExprTree::node(ExprTree::leaf(vec![0], 0), right, vec![0, 1, 2]);
        assert_eq!(extract_waist(&incumbent, &log2), Some((3.0, vec![0, 1, 2])));

        let mut refiner = Refiner {
            code: &code,
            sizes: &sizes,
            hyper: &hyper,
            log2_sizes: &log2,
            start: Instant::now(),
            budget: Duration::from_secs(1),
            rng: SmallRng::seed_from_u64(RNG_SEED),
            report: WaistReport {
                n_original: code.num_tensors(),
                surgery_calls: 0,
                cheaper_cuts: 0,
                rebuild_attempts: 0,
                rebuild_accepts: 0,
                waist_min_hits: 0,
            },
        };

        assert!(refiner.waist_surgery(&incumbent, 3.0).is_none());
        assert_eq!(refiner.report.surgery_calls, 1);
        assert_eq!(refiner.report.rebuild_attempts, 0);
    }

    #[test]
    fn test_public_refine_rejects_non_binary_seed_without_mutation() {
        let code = EinCode::new(vec![vec![0usize], vec![1], vec![2]], vec![0, 1, 2]);
        let sizes = uniform_sizes(&code, 2);
        let seed = NestedEinsum::node(
            vec![
                NestedEinsum::leaf(0),
                NestedEinsum::leaf(1),
                NestedEinsum::leaf(2),
            ],
            code.clone(),
        );

        let (refined, report) = refine(&seed, &code, &sizes, Duration::from_secs(1));

        assert_eq!(refined, seed);
        assert_eq!(report.surgery_calls, 0);
    }

    #[test]
    fn test_hyper_and_refiner_defensive_branches() {
        // Exercise output labels, an unused interned label, and the giant-edge
        // adjacency guard in one compact hypergraph.
        let code = EinCode::new(vec![vec![0usize]; 65], vec![0]);
        let sizes: HashMap<usize, usize> = [(0, 2), (1, 1)].into();
        let label_map: HashMap<usize, usize> = [(0, 0), (1, 1)].into();
        let log2 = vec![1.0, 0.0];
        let hyper = Hyper::build(&code, &label_map, &log2, 2);
        let mut part = vec![false; 65];
        part[0] = true;
        assert_eq!(hyper.cut_cost(&part), 1.0);
        assert!(hyper.adj.iter().all(Vec::is_empty));

        let mut refiner = Refiner {
            code: &code,
            sizes: &sizes,
            hyper: &hyper,
            log2_sizes: &log2,
            start: Instant::now(),
            budget: Duration::from_secs(1),
            rng: SmallRng::seed_from_u64(RNG_SEED),
            report: WaistReport {
                n_original: code.num_tensors(),
                surgery_calls: 0,
                cheaper_cuts: 0,
                rebuild_attempts: 0,
                rebuild_accepts: 0,
                waist_min_hits: 0,
            },
        };
        assert!(refiner.rebuild(&[true; 65]).is_none());
        assert!(refiner.solve_side(&[], &[]).is_none());
        assert!(matches!(
            refiner.solve_side(&[0], &[0]),
            Some(NestedEinsum::Leaf { tensor_index: 0 })
        ));
        let (open_a, open_b) = refiner.side_open_labels(&part);
        assert_eq!(open_a, vec![0]);
        assert_eq!(open_b, vec![0]);
    }

    #[test]
    fn test_public_refine_accepts_better_ring_partition() {
        let code = EinCode::new(
            vec![vec![0usize, 3], vec![0, 1], vec![1, 2], vec![2, 3]],
            vec![],
        );
        let sizes = uniform_sizes(&code, 2);
        let left = NestedEinsum::node(
            vec![NestedEinsum::leaf(0), NestedEinsum::leaf(2)],
            EinCode::new(
                vec![code.ixs[0].clone(), code.ixs[2].clone()],
                vec![0, 1, 2, 3],
            ),
        );
        let right = NestedEinsum::node(
            vec![NestedEinsum::leaf(1), NestedEinsum::leaf(3)],
            EinCode::new(
                vec![code.ixs[1].clone(), code.ixs[3].clone()],
                vec![0, 1, 2, 3],
            ),
        );
        let seed = NestedEinsum::node(
            vec![left, right],
            EinCode::new(vec![vec![0, 1, 2, 3], vec![0, 1, 2, 3]], vec![]),
        );
        let seed_tc = contraction_complexity(&seed, &sizes, &code.ixs).tc;

        let (refined, report) = refine(&seed, &code, &sizes, Duration::from_secs(2));
        let refined_tc = contraction_complexity(&refined, &sizes, &code.ixs).tc;

        assert!(report.rebuild_accepts >= 1);
        assert!(refined_tc < seed_tc - 1e-9);
        assert_eq!(refined.leaf_count(), code.num_tensors());
    }

    #[test]
    fn test_refine_accepts_treewidth_unary_nodes() {
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

        let (refined, report) = refine(&seed, &code, &sizes, Duration::from_millis(50));

        assert!(report.surgery_calls >= 1);
        assert_eq!(refined.leaf_count(), code.num_tensors());
        assert_eq!(refined.output_labels(&code.ixs), code.iy);
    }

    #[test]
    fn test_expr_to_nested_counted_hyperedge_output() {
        // A label shared by three tensors, two of which sit under one node: the
        // node's output must keep that label (it still occurs outside), which
        // outside-occurrence counting gets right.
        let ixs = vec![vec![0usize, 1], vec![0, 2], vec![0, 3]];
        let open = vec![1, 2, 3];
        // Tree: ((t0 t1) t2), leaves index into ixs.
        let inner = ExprTree::node(ExprTree::leaf(vec![], 0), ExprTree::leaf(vec![], 1), vec![]);
        let root = ExprTree::node(inner, ExprTree::leaf(vec![], 2), vec![]);
        let nested = expr_to_nested_counted(&root, &ixs, &open);
        // The inner node contracts t0,t1 which share label 0; label 0 also appears
        // in t2 (outside), so it must be an output of the inner node.
        if let NestedEinsum::Node { args, .. } = &nested {
            if let NestedEinsum::Node { eins, .. } = &args[0] {
                assert!(eins.iy.contains(&0), "inner node must keep shared label 0");
            } else {
                panic!("expected inner node");
            }
        } else {
            panic!("expected root node");
        }
        // The scorer over the topology agrees with the emitted bodies.
        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2), (3, 2)].into();
        let cc = contraction_complexity(&nested, &sizes, &ixs);
        assert!(cc.tc.is_finite());
    }
}
