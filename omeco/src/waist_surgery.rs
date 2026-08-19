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

use std::cmp::Reverse;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
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

/// Labels with more member tensors than this cap skip incremental gain
/// recomputation after a move (their neighbours keep a stale gain until the
/// next pass); the final acceptance rescore in the caller stays exact.
const FM_GAIN_UPDATE_DEG_CAP: usize = 256;

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

/// Initialization strategy for rebuilding each side of a surgery cut.
///
/// [`RebuildMode::Greedy`] preserves the historical behavior. The opt-in
/// [`RebuildMode::WarmRestricted`] variant starts from the incumbent tree
/// restricted to the tensors assigned to that side.
///
/// # Example
///
/// ```
/// use omeco::waist_surgery::RebuildMode;
///
/// assert_eq!(RebuildMode::default(), RebuildMode::Greedy);
/// let mode = RebuildMode::WarmRestricted;
/// assert_ne!(mode, RebuildMode::default());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RebuildMode {
    /// Initialize each rebuilt side with the greedy optimizer.
    #[default]
    Greedy,
    /// Initialize each rebuilt side from the restricted incumbent tree.
    WarmRestricted,
}

/// Region of the incumbent tree replaced by waist surgery.
///
/// [`SurgeryScope::Root`] preserves the historical whole-network rebuild.
/// [`SurgeryScope::Local`] rebuilds and splices only a bounded ancestor of the
/// waist node.
///
/// # Example
///
/// ```
/// use omeco::waist_surgery::SurgeryScope;
///
/// assert_eq!(SurgeryScope::default(), SurgeryScope::Root);
/// let scope = SurgeryScope::Local;
/// assert_ne!(scope, SurgeryScope::default());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SurgeryScope {
    /// Promote the improved bipartition to the root of the full network.
    #[default]
    Root,
    /// Rebuild only a bounded ancestor subtree around the waist.
    Local,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SurgeryOptions {
    pub(crate) rebuild: RebuildMode,
    pub(crate) scope: SurgeryScope,
}

#[derive(Clone, Copy, Debug, Default)]
struct RefineOptions {
    capture_trace: bool,
    surgery: SurgeryOptions,
}

/// One completed like-for-like waist-cut comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaistCallTrace {
    /// Exact cut weight of the incumbent waist bipartition.
    pub incumbent_cut_cost: f64,
    /// Exact cut weight of the best comparable-balance FM candidate.
    pub best_alt_cut_cost: f64,
    /// Cost of the incumbent tree's highest-cost contraction node.
    pub waist_node_cost: f64,
    /// Whether the candidate passed the no-new-bottleneck gate.
    pub rebuild_attempted: bool,
    /// Whether rebuilding strictly reduced the independently rescored total cost.
    pub rebuild_accepted: bool,
}

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

/// Cached, timer-free proposal state for TreeSA's global surgery update.
///
/// Construction is linear in the network and happens once per TreeSA trial;
/// each proposal then moves one boundary tensor across the waist with a
/// leaf prune-and-regraft (SPR) edit. Unaffected subtrees retain their exact
/// topology.
pub(crate) struct WaistUpdate {
    code: EinCode<usize>,
    hyper: Hyper,
    log2_sizes: Vec<f64>,
    label_map: HashMap<usize, usize>,
}

impl WaistUpdate {
    pub(crate) fn new(ixs: &[Vec<usize>], iy: &[usize], log2_sizes: &[f64]) -> Self {
        let code = EinCode::new(ixs.to_vec(), iy.to_vec());
        let label_map: HashMap<usize, usize> = (0..log2_sizes.len()).map(|i| (i, i)).collect();
        let hyper = Hyper::build(&code, &label_map, log2_sizes, log2_sizes.len());
        Self {
            code,
            hyper,
            log2_sizes: log2_sizes.to_vec(),
            label_map,
        }
    }

    /// Propose one global waist update. The proposal itself neither accepts nor
    /// anneals; TreeSA applies its ordinary Metropolis test at the current beta.
    pub(crate) fn propose<R: Rng>(&self, incumbent: &ExprTree, rng: &mut R) -> Option<ExprTree> {
        let (_, a_leaves) = extract_waist(incumbent, &self.log2_sizes)?;
        let n = self.hyper.n;
        if a_leaves.is_empty() || a_leaves.len() >= n {
            return None;
        }
        let mut current = vec![false; n];
        for tensor in a_leaves {
            if tensor < n {
                current[tensor] = true;
            }
        }
        let target_a = current.iter().filter(|&&side| side).count();
        let lo = ((target_a as f64 * (1.0 - FM_SLACK)).floor() as usize).max(1);
        let hi = ((target_a as f64 * (1.0 + FM_SLACK)).ceil() as usize).min(n - 1);

        // One sampled surgery event performs one leaf-SPR move. Repeated update
        // opportunities supply the iteration; there is no hidden FM loop.
        let part = single_cut_move(&self.hyper, current.clone(), lo, hi, rng)?;
        let size_a = part.iter().filter(|&&side| side).count();
        if size_a < lo || size_a > hi {
            return None;
        }

        let moved_tensor = current
            .iter()
            .zip(&part)
            .position(|(before, after)| before != after)?;
        let target_side = part[moved_tensor];
        let (pruned, moved_leaf) = detach_leaf(incumbent, moved_tensor)?;
        let attachment = choose_attachment(
            &pruned,
            &part,
            target_side,
            &self.hyper.tlabels[moved_tensor],
            &self.hyper,
            rng,
        )?;
        let grafted = graft_leaf(pruned, &attachment, moved_leaf)?;

        // Re-derive every cached interface from the new topology. This is an
        // exact normalization step, not a topology rebuild: SPR is the only
        // structural mutation above.
        let nested = expr_to_nested_counted(&grafted, &self.code.ixs, &self.code.iy);
        nested_to_expr_tree(&nested, &self.label_map)
    }
}

// =============================================================================
// Fiduccia–Mattheyses bipartition refinement on the tensor hypergraph.
// =============================================================================

/// Map a gain to a `u64` that orders like the float. `+ 0.0` normalizes `-0.0`
/// to `+0.0` so numerically equal gains get equal keys; gains are finite sums
/// of `log2` dimensions, never NaN.
#[inline]
fn gain_key(g: f64) -> u64 {
    let bits = (g + 0.0).to_bits();
    if bits >> 63 == 0 {
        bits | (1 << 63)
    } else {
        !bits
    }
}

/// Improve a bipartition by bounded FM passes. `part[t] == true` means side A.
/// Balance is constrained so `|A|` stays within `[lo, hi]`. Returns the improved
/// partition and its straddle-cut cost.
fn fm_refine(
    hyper: &Hyper,
    part: Vec<bool>,
    lo: usize,
    hi: usize,
    start: Instant,
    budget: Duration,
) -> (Vec<bool>, f64) {
    fm_refine_core(hyper, part, lo, hi, FM_MAX_PASSES, Some((start, budget)))
}

fn fm_refine_core(
    hyper: &Hyper,
    mut part: Vec<bool>,
    lo: usize,
    hi: usize,
    max_passes: usize,
    deadline: Option<(Instant, Duration)>,
) -> (Vec<bool>, f64) {
    let out_of_time = || deadline.is_some_and(|(start, budget)| start.elapsed() >= budget);
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

    for _pass in 0..max_passes {
        if out_of_time() {
            break;
        }
        let mut gain: Vec<f64> = (0..n).map(|t| gain_of(t, &part, &cnt_a)).collect();
        let mut locked = vec![false; n];
        // Unlocked move candidates by current side. The maximum entry is
        // (max gain, then smallest vertex) — the same move a linear scan over
        // `gain` picks — at O(log n) per query/update instead of O(n) per step.
        let mut side_a: BTreeSet<(u64, Reverse<usize>)> = BTreeSet::new();
        let mut side_b: BTreeSet<(u64, Reverse<usize>)> = BTreeSet::new();
        for t in 0..n {
            let entry = (gain_key(gain[t]), Reverse(t));
            if part[t] {
                side_a.insert(entry);
            } else {
                side_b.insert(entry);
            }
        }
        let mut cur_cost = hyper.cut_cost(&part);
        let mut best_seen_cost = cur_cost;
        let mut best_seen_part = part.clone();
        let mut improved_this_pass = false;

        for _step in 0..n {
            if out_of_time() {
                break;
            }
            // Pick max-gain unlocked feasible move.
            let cand_a = if size_a > lo {
                side_a.last().copied()
            } else {
                None
            };
            let cand_b = if size_a < hi {
                side_b.last().copied()
            } else {
                None
            };
            let best = match (cand_a, cand_b) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            };
            let Some((_, Reverse(v))) = best else { break };
            let ventry = (gain_key(gain[v]), Reverse(v));
            if part[v] {
                side_a.remove(&ventry);
            } else {
                side_b.remove(&ventry);
            }

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
                if hyper.label_tensors[l].len() > FM_GAIN_UPDATE_DEG_CAP {
                    continue;
                }
                for &u in &hyper.label_tensors[l] {
                    touched.push(u);
                }
            }
            touched.sort_unstable();
            touched.dedup();
            for &u in &touched {
                let new_gain = gain_of(u, &part, &cnt_a);
                if !locked[u] {
                    let set = if part[u] { &mut side_a } else { &mut side_b };
                    set.remove(&(gain_key(gain[u]), Reverse(u)));
                    set.insert((gain_key(new_gain), Reverse(u)));
                }
                gain[u] = new_gain;
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

/// Make one bounded FM-style move across the current waist. Candidates are
/// boundary tensors whose move respects the balance band; one of the eight
/// highest-gain candidates is sampled to retain exploration. Whole-tree
/// Metropolis acceptance is performed by TreeSA, not here.
fn single_cut_move<R: Rng>(
    hyper: &Hyper,
    mut part: Vec<bool>,
    lo: usize,
    hi: usize,
    rng: &mut R,
) -> Option<Vec<bool>> {
    let mut cnt_a = vec![0u32; hyper.log2.len()];
    for (tensor, &in_a) in part.iter().enumerate() {
        if in_a {
            for &label in &hyper.tlabels[tensor] {
                cnt_a[label] += 1;
            }
        }
    }
    let size_a = part.iter().filter(|&&in_a| in_a).count();
    let mut top: Vec<(f64, usize)> = Vec::with_capacity(9);
    for (tensor, &tensor_in_a) in part.iter().enumerate().take(hyper.n) {
        if (tensor_in_a && size_a <= lo) || (!tensor_in_a && size_a >= hi) {
            continue;
        }
        let boundary = hyper.tlabels[tensor].iter().any(|&label| {
            let degree = hyper.label_tensors[label].len() as u32;
            cnt_a[label] > 0 && cnt_a[label] < degree
        });
        if !boundary {
            continue;
        }
        let mut gain = 0.0;
        for &label in &hyper.tlabels[tensor] {
            if hyper.is_out[label] {
                continue;
            }
            let degree = hyper.label_tensors[label].len() as u32;
            let (same, other) = if tensor_in_a {
                (cnt_a[label], degree - cnt_a[label])
            } else {
                (degree - cnt_a[label], cnt_a[label])
            };
            gain += hyper.log2[label] * ((other >= 1) as i32 - (same >= 2) as i32) as f64;
        }
        top.push((gain, tensor));
        top.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        top.truncate(8);
    }
    if top.is_empty() {
        return None;
    }
    let (_, tensor) = top[rng.random_range(0..top.len())];
    part[tensor] = !part[tensor];
    Some(part)
}

/// Detach one leaf and suppress the unary branch left at its former parent.
/// All other subtrees are cloned without changing their relative topology.
fn detach_leaf(tree: &ExprTree, tensor: usize) -> Option<(ExprTree, ExprTree)> {
    fn rec(tree: &ExprTree, tensor: usize) -> Option<(Option<ExprTree>, ExprTree)> {
        match tree {
            ExprTree::Leaf(info) => (info.tensor_id == Some(tensor)).then(|| (None, tree.clone())),
            ExprTree::Node { left, right, info } => {
                if let Some((new_left, leaf)) = rec(left, tensor) {
                    let pruned = match new_left {
                        Some(left) => {
                            ExprTree::node(left, (**right).clone(), info.out_dims.clone())
                        }
                        None => (**right).clone(),
                    };
                    return Some((Some(pruned), leaf));
                }
                let (new_right, leaf) = rec(right, tensor)?;
                let pruned = match new_right {
                    Some(right) => ExprTree::node((**left).clone(), right, info.out_dims.clone()),
                    None => (**left).clone(),
                };
                Some((Some(pruned), leaf))
            }
        }
    }

    let (pruned, leaf) = rec(tree, tensor)?;
    Some((pruned?, leaf))
}

#[derive(Debug)]
struct AttachmentCandidate {
    shared_weight: f64,
    span: usize,
    path: Vec<bool>,
}

/// Choose a connected attachment edge wholly inside the tensor's destination
/// side. Higher shared-index weight is preferred; smaller subtrees break ties
/// so the edit stays local. Sampling among the top eight retains exploration.
fn choose_attachment<R: Rng>(
    tree: &ExprTree,
    part: &[bool],
    target_side: bool,
    moved_labels: &[usize],
    hyper: &Hyper,
    rng: &mut R,
) -> Option<Vec<bool>> {
    fn visit(
        tree: &ExprTree,
        part: &[bool],
        target_side: bool,
        moved_labels: &[usize],
        hyper: &Hyper,
        path: &mut Vec<bool>,
        top: &mut Vec<AttachmentCandidate>,
    ) -> (bool, usize) {
        let (all_target, span) = match tree {
            ExprTree::Leaf(info) => (
                info.tensor_id.and_then(|tensor| part.get(tensor).copied()) == Some(target_side),
                1,
            ),
            ExprTree::Node { left, right, .. } => {
                path.push(false);
                let (left_target, left_span) =
                    visit(left, part, target_side, moved_labels, hyper, path, top);
                path.pop();
                path.push(true);
                let (right_target, right_span) =
                    visit(right, part, target_side, moved_labels, hyper, path, top);
                path.pop();
                (left_target && right_target, left_span + right_span)
            }
        };

        if all_target {
            let shared_weight: f64 = moved_labels
                .iter()
                .filter(|&&label| !hyper.is_out[label] && tree.labels().contains(&label))
                .map(|&label| hyper.log2[label])
                .sum();
            if shared_weight > 0.0 {
                top.push(AttachmentCandidate {
                    shared_weight,
                    span,
                    path: path.clone(),
                });
                top.sort_by(|a, b| {
                    b.shared_weight
                        .total_cmp(&a.shared_weight)
                        .then_with(|| a.span.cmp(&b.span))
                        .then_with(|| a.path.cmp(&b.path))
                });
                top.truncate(8);
            }
        }
        (all_target, span)
    }

    let mut top = Vec::with_capacity(9);
    visit(
        tree,
        part,
        target_side,
        moved_labels,
        hyper,
        &mut Vec::new(),
        &mut top,
    );
    if top.is_empty() {
        None
    } else {
        Some(top.swap_remove(rng.random_range(0..top.len())).path)
    }
}

/// Attach `leaf` as a sibling of the subtree at `path`.
fn graft_leaf(tree: ExprTree, path: &[bool], leaf: ExprTree) -> Option<ExprTree> {
    if path.is_empty() {
        return Some(ExprTree::node(tree, leaf, Vec::new()));
    }
    let ExprTree::Node { left, right, info } = tree else {
        return None;
    };
    if path[0] {
        let right = graft_leaf(*right, &path[1..], leaf)?;
        Some(ExprTree::node(*left, right, info.out_dims))
    } else {
        let left = graft_leaf(*left, &path[1..], leaf)?;
        Some(ExprTree::node(left, *right, info.out_dims))
    }
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

/// Return the waist together with its root-relative child path (`false` = left).
fn extract_waist_location(
    tree: &ExprTree,
    log2_sizes: &[f64],
) -> Option<(f64, Vec<usize>, Vec<bool>)> {
    fn walk(
        tree: &ExprTree,
        log2_sizes: &[f64],
        path: &mut Vec<bool>,
        best: &mut f64,
        best_leaves: &mut Vec<usize>,
        best_path: &mut Vec<bool>,
    ) -> Vec<usize> {
        match tree {
            ExprTree::Leaf(info) => vec![info.tensor_id.unwrap_or(0)],
            ExprTree::Node { left, right, info } => {
                path.push(false);
                let left_leaves = walk(left, log2_sizes, path, best, best_leaves, best_path);
                path.pop();
                path.push(true);
                let right_leaves = walk(right, log2_sizes, path, best, best_leaves, best_path);
                path.pop();
                let cost = node_tc(left.labels(), right.labels(), &info.out_dims, log2_sizes);
                let mut leaves = left_leaves;
                leaves.extend_from_slice(&right_leaves);
                if cost > *best {
                    *best = cost;
                    *best_leaves = leaves.clone();
                    *best_path = path.clone();
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
    let mut best_path = Vec::new();
    walk(
        tree,
        log2_sizes,
        &mut Vec::new(),
        &mut best,
        &mut best_leaves,
        &mut best_path,
    );
    Some((best, best_leaves, best_path))
}

fn subtree_at_path<'a>(tree: &'a ExprTree, path: &[bool]) -> Option<&'a ExprTree> {
    let mut current = tree;
    for &right_child in path {
        current = match current {
            ExprTree::Leaf(_) => return None,
            ExprTree::Node { left, right, .. } => {
                if right_child {
                    right
                } else {
                    left
                }
            }
        };
    }
    Some(current)
}

fn local_scope_path(tree: &ExprTree, waist_path: &[bool], waist_size: usize) -> Vec<bool> {
    let target = tree.leaf_count().min(waist_size.saturating_mul(2));
    for depth in (0..=waist_path.len()).rev() {
        let candidate = &waist_path[..depth];
        if subtree_at_path(tree, candidate).is_some_and(|subtree| subtree.leaf_count() >= target) {
            return candidate.to_vec();
        }
    }
    Vec::new()
}

fn replace_subtree_at_path(
    tree: &ExprTree,
    path: &[bool],
    replacement: &ExprTree,
) -> Option<ExprTree> {
    let Some((&right_child, rest)) = path.split_first() else {
        return Some(replacement.clone());
    };
    let ExprTree::Node { left, right, info } = tree else {
        return None;
    };
    if right_child {
        Some(ExprTree::node(
            (**left).clone(),
            replace_subtree_at_path(right, rest, replacement)?,
            info.out_dims.clone(),
        ))
    } else {
        Some(ExprTree::node(
            replace_subtree_at_path(left, rest, replacement)?,
            (**right).clone(),
            info.out_dims.clone(),
        ))
    }
}

/// Restrict a binary tree to `keep`, suppressing internal nodes made unary.
fn restrict_expr_tree(tree: &ExprTree, keep: &HashSet<usize>) -> Option<ExprTree> {
    match tree {
        ExprTree::Leaf(info) => info
            .tensor_id
            .filter(|tensor| keep.contains(tensor))
            .map(|_| tree.clone()),
        ExprTree::Node { left, right, info } => {
            match (
                restrict_expr_tree(left, keep),
                restrict_expr_tree(right, keep),
            ) {
                (Some(left), Some(right)) => {
                    Some(ExprTree::node(left, right, info.out_dims.clone()))
                }
                (Some(child), None) | (None, Some(child)) => Some(child),
                (None, None) => None,
            }
        }
    }
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
pub(crate) fn gated_sweep(
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

fn side_open_labels(hyper: &Hyper, part: &[bool]) -> (Vec<usize>, Vec<usize>) {
    let mut open_a = Vec::new();
    let mut open_b = Vec::new();
    for (label, tensors) in hyper.label_tensors.iter().enumerate() {
        if tensors.is_empty() {
            continue;
        }
        let mut in_a = false;
        let mut in_b = false;
        for &tensor in tensors {
            if part[tensor] {
                in_a = true;
            } else {
                in_b = true;
            }
        }
        let output = hyper.is_out[label];
        if in_a && (in_b || output) {
            open_a.push(label);
        }
        if in_b && (in_a || output) {
            open_b.push(label);
        }
    }
    (open_a, open_b)
}

fn scope_open_labels(hyper: &Hyper, tensors: &[usize]) -> Vec<usize> {
    let inside: HashSet<usize> = tensors.iter().copied().collect();
    hyper
        .label_tensors
        .iter()
        .enumerate()
        .filter(|(label, members)| {
            let occurs_inside = members.iter().any(|tensor| inside.contains(tensor));
            let occurs_outside = members.iter().any(|tensor| !inside.contains(tensor));
            occurs_inside && (occurs_outside || hyper.is_out[*label])
        })
        .map(|(label, _)| label)
        .collect()
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
    /// Maximum number of surgery iterations to start (0 = uncapped). Checked
    /// against `report.surgery_calls` before each iteration begins.
    max_iters: u64,
    rng: SmallRng,
    report: WaistReport,
    /// Opt-in paper diagnostic. Keeping it outside `WaistReport` preserves the
    /// public report's historical `Copy + Eq` API and avoids collecting trace
    /// data during ordinary timed refinement.
    capture_trace: bool,
    last_call_trace: Option<WaistCallTrace>,
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
        self.waist_surgery_opts(
            incumbent,
            work_tc,
            RebuildMode::default(),
            SurgeryScope::default(),
        )
    }

    /// Configurable form of [`Self::waist_surgery`].
    fn waist_surgery_opts(
        &mut self,
        incumbent: &ExprTree,
        work_tc: f64,
        rebuild_mode: RebuildMode,
        scope: SurgeryScope,
    ) -> Option<(NestedEinsum<usize>, f64)> {
        if scope == SurgeryScope::Local {
            if let Some((waist_node_cost, a_leaves, waist_path)) =
                extract_waist_location(incumbent, self.log2_sizes)
            {
                let scope_path = if a_leaves.len().saturating_mul(2) >= self.hyper.n {
                    Vec::new()
                } else {
                    local_scope_path(incumbent, &waist_path, a_leaves.len())
                };
                if !scope_path.is_empty() {
                    return self.waist_surgery_local(
                        incumbent,
                        work_tc,
                        waist_node_cost,
                        &a_leaves,
                        &scope_path,
                        rebuild_mode,
                    );
                }
            }
        }
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
        if self.capture_trace {
            self.last_call_trace = Some(WaistCallTrace {
                incumbent_cut_cost,
                best_alt_cut_cost: best_alt_cost,
                waist_node_cost,
                rebuild_attempted: false,
                rebuild_accepted: false,
            });
        }
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
        if let Some(trace) = &mut self.last_call_trace {
            trace.rebuild_attempted = true;
        }

        // Rebuild both sides from the improved cut.
        if self.out_of_time() {
            return None;
        }
        let rebuilt = match rebuild_mode {
            RebuildMode::Greedy => self.rebuild(&part)?,
            RebuildMode::WarmRestricted => self.rebuild_warm(&part, incumbent)?,
        };
        if self.out_of_time() {
            return None;
        }
        let new_tc = contraction_complexity(&rebuilt, self.sizes, &self.code.ixs).tc;
        if new_tc < work_tc - 1e-9 {
            self.report.rebuild_accepts += 1;
            if let Some(trace) = &mut self.last_call_trace {
                trace.rebuild_accepted = true;
            }
            Some((rebuilt, new_tc))
        } else {
            None
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn waist_surgery_local(
        &mut self,
        incumbent: &ExprTree,
        work_tc: f64,
        waist_node_cost: f64,
        a_leaves: &[usize],
        scope_path: &[bool],
        rebuild_mode: RebuildMode,
    ) -> Option<(NestedEinsum<usize>, f64)> {
        self.report.surgery_calls += 1;
        let scope_tree = subtree_at_path(incumbent, scope_path)?;
        let scope_tensors = scope_tree.leaf_ids();
        let scope_n = scope_tensors.len();
        if scope_n < 2 || a_leaves.is_empty() || a_leaves.len() >= scope_n {
            return None;
        }
        let scope_open = scope_open_labels(self.hyper, &scope_tensors);
        let scope_code = EinCode::new(
            scope_tensors
                .iter()
                .map(|&tensor| self.code.ixs.get(tensor).cloned())
                .collect::<Option<Vec<_>>>()?,
            scope_open.clone(),
        );
        let label_map: HashMap<usize, usize> = (0..self.log2_sizes.len())
            .map(|label| (label, label))
            .collect();
        let scope_hyper = Hyper::build(
            &scope_code,
            &label_map,
            self.log2_sizes,
            self.log2_sizes.len(),
        );
        let a_set: HashSet<usize> = a_leaves.iter().copied().collect();
        let cur_part: Vec<bool> = scope_tensors
            .iter()
            .map(|tensor| a_set.contains(tensor))
            .collect();
        let incumbent_cut_cost = scope_hyper.cut_cost(&cur_part);
        let target_a = a_leaves.len();
        let lo = ((target_a as f64 * (1.0 - FM_SLACK)).floor() as usize).max(1);
        let hi = ((target_a as f64 * (1.0 + FM_SLACK)).ceil() as usize).min(scope_n - 1);

        let mut seeds = vec![cur_part];
        let mut deg_order: Vec<usize> = (0..scope_n).collect();
        deg_order.sort_by_key(|&tensor| Reverse(scope_hyper.tlabels[tensor].len()));
        seeds.push(bfs_seed(&scope_hyper, deg_order[0], target_a));
        for _ in 0..3 {
            let seed = self.rng.random_range(0..scope_n);
            seeds.push(bfs_seed(&scope_hyper, seed, target_a));
        }

        let mut best_alt_cost = f64::INFINITY;
        let mut best_alt_part = None;
        for seed in seeds {
            if self.out_of_time() {
                break;
            }
            let (part, _) = fm_refine(&scope_hyper, seed, lo, hi, self.start, self.budget);
            let size_a = part.iter().filter(|&&side| side).count();
            if size_a < lo || size_a > hi {
                continue;
            }
            let cost = scope_hyper.cut_cost(&part);
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
        if self.capture_trace {
            self.last_call_trace = Some(WaistCallTrace {
                incumbent_cut_cost,
                best_alt_cut_cost: best_alt_cost,
                waist_node_cost,
                rebuild_attempted: false,
                rebuild_accepted: false,
            });
        }
        if best_alt_cost < incumbent_cut_cost - 1e-9 {
            self.report.cheaper_cuts += 1;
        } else {
            self.report.waist_min_hits += 1;
        }
        if best_alt_cost >= waist_node_cost - 1e-9 {
            return None;
        }
        self.report.rebuild_attempts += 1;
        if let Some(trace) = &mut self.last_call_trace {
            trace.rebuild_attempted = true;
        }
        if self.out_of_time() {
            return None;
        }
        let rebuilt = self.rebuild_local(
            &part,
            &scope_hyper,
            &scope_tensors,
            &scope_open,
            incumbent,
            scope_path,
            rebuild_mode,
        )?;
        if self.out_of_time() {
            return None;
        }
        let new_tc = contraction_complexity(&rebuilt, self.sizes, &self.code.ixs).tc;
        if new_tc < work_tc - 1e-9 {
            self.report.rebuild_accepts += 1;
            if let Some(trace) = &mut self.last_call_trace {
                trace.rebuild_accepted = true;
            }
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

    /// Root-scoped rebuild whose side initializers retain incumbent topology.
    fn rebuild_warm(&mut self, part: &[bool], incumbent: &ExprTree) -> Option<NestedEinsum<usize>> {
        let n = self.hyper.n;
        let a_tensors: Vec<usize> = (0..n).filter(|&tensor| part[tensor]).collect();
        let b_tensors: Vec<usize> = (0..n).filter(|&tensor| !part[tensor]).collect();
        if a_tensors.is_empty() || b_tensors.is_empty() {
            return None;
        }
        let (open_a, open_b) = self.side_open_labels(part);
        let sub_a = self.solve_side_warm(&a_tensors, &open_a, incumbent)?;
        if self.out_of_time() {
            return None;
        }
        let sub_b = self.solve_side_warm(&b_tensors, &open_b, incumbent)?;
        let eins = EinCode::new(vec![open_a, open_b], self.code.iy.clone());
        Some(NestedEinsum::node(vec![sub_a, sub_b], eins))
    }

    #[allow(clippy::too_many_arguments)]
    fn rebuild_local(
        &mut self,
        part: &[bool],
        scope_hyper: &Hyper,
        scope_tensors: &[usize],
        scope_open: &[usize],
        incumbent: &ExprTree,
        scope_path: &[bool],
        rebuild_mode: RebuildMode,
    ) -> Option<NestedEinsum<usize>> {
        let a_tensors: Vec<usize> = scope_tensors
            .iter()
            .zip(part)
            .filter_map(|(&tensor, &side)| side.then_some(tensor))
            .collect();
        let b_tensors: Vec<usize> = scope_tensors
            .iter()
            .zip(part)
            .filter_map(|(&tensor, &side)| (!side).then_some(tensor))
            .collect();
        if a_tensors.is_empty() || b_tensors.is_empty() {
            return None;
        }
        let (open_a, open_b) = side_open_labels(scope_hyper, part);
        let solve = |refiner: &mut Self, tensors: &[usize], open: &[usize]| match rebuild_mode {
            RebuildMode::Greedy => refiner.solve_side(tensors, open),
            RebuildMode::WarmRestricted => refiner.solve_side_warm(tensors, open, incumbent),
        };
        let sub_a = solve(self, &a_tensors, &open_a)?;
        if self.out_of_time() {
            return None;
        }
        let sub_b = solve(self, &b_tensors, &open_b)?;
        let rebuilt_scope = NestedEinsum::node(
            vec![sub_a, sub_b],
            EinCode::new(vec![open_a, open_b], scope_open.to_vec()),
        );
        let label_map: HashMap<usize, usize> = (0..self.log2_sizes.len())
            .map(|label| (label, label))
            .collect();
        let replacement = nested_to_expr_tree(&rebuilt_scope, &label_map)?;
        let spliced = replace_subtree_at_path(incumbent, scope_path, &replacement)?;
        Some(expr_to_nested_counted(
            &spliced,
            &self.code.ixs,
            &self.code.iy,
        ))
    }

    /// Open label-ids for each side: a label is open on side A iff it occurs in A
    /// and (occurs in B or is an output label). Symmetric for B.
    fn side_open_labels(&self, part: &[bool]) -> (Vec<usize>, Vec<usize>) {
        side_open_labels(self.hyper, part)
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

    /// Warm-restricted side solve. Restriction failure falls back to the
    /// historical greedy initializer; the cold V-cycle schedule is unchanged.
    fn solve_side_warm(
        &mut self,
        tensors: &[usize],
        open: &[usize],
        incumbent: &ExprTree,
    ) -> Option<NestedEinsum<usize>> {
        if tensors.len() <= 1 {
            return self.solve_side(tensors, open);
        }
        if self.out_of_time() {
            return None;
        }
        let keep: HashSet<usize> = tensors.iter().copied().collect();
        let Some(restricted) = restrict_expr_tree(incumbent, &keep) else {
            return self.solve_side(tensors, open);
        };
        let mut restricted_leaves = restricted.leaf_ids();
        let mut expected_leaves = tensors.to_vec();
        restricted_leaves.sort_unstable();
        expected_leaves.sort_unstable();
        if restricted_leaves != expected_leaves {
            return self.solve_side(tensors, open);
        }

        let sub_ixs: Vec<Vec<usize>> = tensors
            .iter()
            .filter_map(|&tensor| self.code.ixs.get(tensor).cloned())
            .collect();
        if sub_ixs.len() != tensors.len() {
            return self.solve_side(tensors, open);
        }
        let sub_code = EinCode::new(sub_ixs, open.to_vec());
        let sub_labels = sub_code.unique_labels();
        let sub_label_map: HashMap<usize, usize> = sub_labels
            .iter()
            .enumerate()
            .map(|(index, &label)| (label, index))
            .collect();
        let sub_log2: Vec<f64> = sub_labels
            .iter()
            .map(|label| (*self.sizes.get(label).unwrap_or(&1) as f64).log2())
            .collect();
        let normalized = expr_to_nested_counted(&restricted, &self.code.ixs, open);
        let Some(mut tree) = nested_to_expr_tree(&normalized, &sub_label_map) else {
            return self.solve_side(tensors, open);
        };
        if self.out_of_time() {
            return None;
        }

        let mut scratch = ScratchSpace::new(sub_labels.len());
        let m = tree.leaf_count();
        let s_top = ((m + TARGET_TOP - 1) / TARGET_TOP).max(2);
        let mut spans = Vec::new();
        let mut span = s_top;
        while span > 2 {
            spans.push(span);
            span /= 2;
        }
        spans.push(2);
        let mut s_lin = f64::exp2(tree_complexity(&tree, &sub_log2).0);
        let mut best_lin = s_lin;
        let mut best_tree = tree.clone();
        let mut sweeps = 0_u64;
        'anneal: for _ in 0..REBUILD_VCYCLES.max(1) {
            for &span in &spans {
                let denominator = (REBUILD_COLD_SWEEPS.saturating_sub(1)).max(1) as f64;
                for sweep in 0..REBUILD_COLD_SWEEPS {
                    if self.out_of_time() {
                        break 'anneal;
                    }
                    let beta = B_LO_COLD + (B_HI - B_LO_COLD) * (sweep as f64 / denominator);
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
        Some(expr_to_nested_counted(&best_tree, &self.code.ixs, open))
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
/// loads — only the never-worse guarantee is. For a fully deterministic,
/// machine-independent cap use [`refine_capped`] with a positive `max_iters`
/// (this function is a thin delegate: `refine_capped(tree, code, sizes,
/// budget, 0)`, i.e. `0` = uncapped).
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
    refine_capped(tree, code, sizes, budget, 0)
}

/// Refine a contraction tree by repeated waist surgery, identical to
/// [`refine`] except that at most `max_iters` surgery iterations are started
/// (`0` = uncapped, matching `refine`'s behavior exactly).
///
/// Each iteration increments [`WaistReport::surgery_calls`] once, before
/// doing any work; capping stops the loop **before** starting an iteration
/// once that counter has reached `max_iters`, so the guarantee is exactly
/// `report.surgery_calls <= max_iters` whenever `max_iters > 0`.
///
/// Note that `max_iters` bounds *iterations*, not wall-clock time: `budget`
/// still applies within each iteration (FM and rebuild-annealing loops), so
/// pass a generous budget (e.g. [`Duration::MAX`]) if you want the iteration
/// count to be the only limiting factor — that combination is fully
/// deterministic and machine-independent, unlike a positive `budget` alone.
///
/// # Example
///
/// ```
/// use omeco::waist_surgery::refine_capped;
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
/// let (refined, report) = refine_capped(&seed, &code, &sizes, Duration::MAX, 2);
/// assert_eq!(refined.leaf_count(), 3);
/// assert!(report.surgery_calls <= 2);
/// ```
pub fn refine_capped<L: Label>(
    tree: &NestedEinsum<L>,
    code: &EinCode<L>,
    sizes: &HashMap<L, usize>,
    budget: Duration,
    max_iters: u64,
) -> (NestedEinsum<L>, WaistReport) {
    refine_capped_seeded(tree, code, sizes, budget, max_iters, RNG_SEED)
}

/// Iteration-capped refinement with a caller-selected deterministic RNG stream.
///
/// The public standalone API intentionally keeps its historical fixed seed.
/// Interleaved TreeSA rounds use distinct seeds so repeated calls on an
/// unchanged incumbent do not replay the identical FM/BFS proposal forever.
pub(crate) fn refine_capped_seeded<L: Label>(
    tree: &NestedEinsum<L>,
    code: &EinCode<L>,
    sizes: &HashMap<L, usize>,
    budget: Duration,
    max_iters: u64,
    rng_seed: u64,
) -> (NestedEinsum<L>, WaistReport) {
    let (tree, report, _) = refine_capped_seeded_impl(
        tree,
        code,
        sizes,
        budget,
        max_iters,
        rng_seed,
        RefineOptions::default(),
    );
    (tree, report)
}

/// One-call fixed-work refinement with an exact waist-cut diagnostic.
///
/// The returned [`WaistCallTrace`] is last-write-wins across surgery calls, so
/// it only stays consistent with the round-level [`WaistReport`] flags when at
/// most one refinement iteration runs; callers must pass `max_iters <= 1`.
pub(crate) fn refine_capped_seeded_with_trace<L: Label>(
    tree: &NestedEinsum<L>,
    code: &EinCode<L>,
    sizes: &HashMap<L, usize>,
    budget: Duration,
    max_iters: u64,
    rng_seed: u64,
) -> (NestedEinsum<L>, WaistReport, Option<WaistCallTrace>) {
    refine_capped_seeded_with_trace_opts(
        tree,
        code,
        sizes,
        budget,
        max_iters,
        rng_seed,
        SurgeryOptions::default(),
    )
}

/// One-call fixed-work refinement with caller-selected opt-in surgery modes.
pub(crate) fn refine_capped_seeded_with_trace_opts<L: Label>(
    tree: &NestedEinsum<L>,
    code: &EinCode<L>,
    sizes: &HashMap<L, usize>,
    budget: Duration,
    max_iters: u64,
    rng_seed: u64,
    surgery: SurgeryOptions,
) -> (NestedEinsum<L>, WaistReport, Option<WaistCallTrace>) {
    debug_assert!(
        max_iters <= 1,
        "last-call trace semantics require max_iters <= 1, got {max_iters}"
    );
    refine_capped_seeded_impl(
        tree,
        code,
        sizes,
        budget,
        max_iters,
        rng_seed,
        RefineOptions {
            capture_trace: true,
            surgery,
        },
    )
}

fn refine_capped_seeded_impl<L: Label>(
    tree: &NestedEinsum<L>,
    code: &EinCode<L>,
    sizes: &HashMap<L, usize>,
    budget: Duration,
    max_iters: u64,
    rng_seed: u64,
    options: RefineOptions,
) -> (NestedEinsum<L>, WaistReport, Option<WaistCallTrace>) {
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
        return (tree.clone(), report, None);
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
        max_iters,
        rng: SmallRng::seed_from_u64(rng_seed),
        report,
        capture_trace: options.capture_trace,
        last_call_trace: None,
    };

    let mut stale: u64 = 0;
    while !refiner.out_of_time()
        && stale < MAX_STALE_ITERS
        && (refiner.max_iters == 0 || refiner.report.surgery_calls < refiner.max_iters)
    {
        let Some(incumbent) = nested_to_expr_tree(&best, &id_label_map) else {
            break;
        };
        let candidate = if options.surgery.rebuild == RebuildMode::default()
            && options.surgery.scope == SurgeryScope::default()
        {
            refiner.waist_surgery(&incumbent, best_tc)
        } else {
            refiner.waist_surgery_opts(
                &incumbent,
                best_tc,
                options.surgery.rebuild,
                options.surgery.scope,
            )
        };
        match candidate {
            Some((new_tree, new_tc)) if new_tc < best_tc - 1e-9 => {
                best = new_tree;
                best_tc = new_tc;
                stale = 0;
            }
            _ => stale += 1,
        }
    }
    report = refiner.report;
    let call_trace = refiner.last_call_trace;

    // Map id-space leaves/eins back to the original label space.
    let best_l = restore_nested(&best, &labels);
    (best_l, report, call_trace)
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

    /// Reference FM with the original linear-scan move selection (max gain,
    /// then smallest vertex, feasibility by side). `fm_refine` must match this
    /// exactly whatever selection data structure it uses internally.
    fn fm_refine_reference(
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

        let gain_of = |t: usize, part: &[bool], cnt_a: &[u32]| -> f64 {
            let mut g = 0.0;
            for &l in &hyper.tlabels[t] {
                if hyper.is_out[l] {
                    continue;
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

    #[test]
    #[ignore]
    fn perf_fm_refine_large() {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        let n_tensors = 20000usize;
        let n_labels = 4000usize;
        let ixs: Vec<Vec<usize>> = (0..n_tensors)
            .map(|_| {
                let mut v: Vec<usize> = (0..3).map(|_| rng.random_range(0..n_labels)).collect();
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();
        let code = EinCode::new(ixs, Vec::new());
        let label_map: HashMap<usize, usize> = (0..n_labels).map(|l| (l, l)).collect();
        let log2: Vec<f64> = (0..n_labels).map(|_| 1.0).collect();
        let hyper = Hyper::build(&code, &label_map, &log2, n_labels);
        let part: Vec<bool> = (0..n_tensors).map(|t| t % 2 == 0).collect();
        let lo = n_tensors / 4;
        let hi = 3 * n_tensors / 4;
        let budget = Duration::from_secs(3600);
        let t0 = Instant::now();
        let (p1, c1) = fm_refine(&hyper, part.clone(), lo, hi, Instant::now(), budget);
        let new_t = t0.elapsed();
        let t1 = Instant::now();
        let (p2, c2) = fm_refine_reference(&hyper, part, lo, hi, Instant::now(), budget);
        let ref_t = t1.elapsed();
        assert_eq!(p1, p2);
        assert!((c1 - c2).abs() < 1e-9);
        println!("n=20000: new {new_t:?} vs reference {ref_t:?}");
    }

    #[test]
    fn test_fm_refine_matches_reference_selection() {
        use rand::Rng;
        use rand::SeedableRng;

        for seed in 0..80u64 {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
            // Occasionally exceed the gain-update degree cap with one label
            // shared by every tensor, to exercise the stale-gain path too.
            let big = seed % 10 == 0;
            let n_tensors = if big { 300 } else { rng.random_range(6..=40) };
            let n_labels = if big { 40 } else { rng.random_range(4..=12) };
            let mut ixs: Vec<Vec<usize>> = (0..n_tensors)
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
            if big {
                for ix in ixs.iter_mut() {
                    if !ix.contains(&0) {
                        ix.insert(0, 0);
                    }
                }
            }
            let iy: Vec<usize> = (0..n_labels)
                .filter(|_| rng.random_range(0..4) == 0)
                .collect();
            let code = EinCode::new(ixs, iy);
            let label_map: HashMap<usize, usize> = (0..n_labels).map(|l| (l, l)).collect();
            let log2: Vec<f64> = (0..n_labels)
                .map(|_| (rng.random_range(1..=4) as f64).log2())
                .collect();
            let hyper = Hyper::build(&code, &label_map, &log2, n_labels);

            let part: Vec<bool> = (0..n_tensors)
                .map(|_| rng.random_range(0..2) == 0)
                .collect();
            let size_a = part.iter().filter(|&&p| p).count();
            let lo = size_a.saturating_sub(n_tensors / 4).max(1);
            let hi = (size_a + n_tensors / 4).min(n_tensors - 1);
            if lo > hi {
                continue;
            }

            let start = Instant::now();
            let budget = Duration::from_secs(3600);
            let (part_new, cost_new) = fm_refine(&hyper, part.clone(), lo, hi, start, budget);
            let (part_ref, cost_ref) =
                fm_refine_reference(&hyper, part.clone(), lo, hi, start, budget);
            assert_eq!(
                part_new, part_ref,
                "seed {seed}: fm_refine diverged from reference selection"
            );
            assert!(
                (cost_new - cost_ref).abs() < 1e-12,
                "seed {seed}: cost {cost_new} != reference {cost_ref}"
            );
        }
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

    fn test_refiner<'a>(
        code: &'a EinCode<usize>,
        sizes: &'a HashMap<usize, usize>,
        hyper: &'a Hyper,
        log2: &'a [f64],
    ) -> Refiner<'a> {
        Refiner {
            code,
            sizes,
            hyper,
            log2_sizes: log2,
            start: Instant::now(),
            budget: Duration::from_secs(30),
            max_iters: 1,
            rng: SmallRng::seed_from_u64(RNG_SEED),
            report: WaistReport {
                n_original: code.num_tensors(),
                surgery_calls: 0,
                cheaper_cuts: 0,
                rebuild_attempts: 0,
                rebuild_accepts: 0,
                waist_min_hits: 0,
            },
            capture_trace: true,
            last_call_trace: None,
        }
    }

    fn assert_nested_interfaces(tree: &NestedEinsum<usize>, code: &EinCode<usize>) -> Vec<usize> {
        match tree {
            NestedEinsum::Leaf { tensor_index } => code.ixs[*tensor_index].clone(),
            NestedEinsum::Node { args, eins } => {
                assert_eq!(args.len(), 2);
                let child_outputs: Vec<Vec<usize>> = args
                    .iter()
                    .map(|child| assert_nested_interfaces(child, code))
                    .collect();
                assert_eq!(eins.ixs, child_outputs);
                eins.iy.clone()
            }
        }
    }

    #[test]
    fn test_warm_restriction_preserves_leaf_set_and_interfaces() {
        let code = grid(3, 3);
        let sizes = uniform_sizes(&code, 2);
        let seed = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
        let label_map: HashMap<usize, usize> = code
            .unique_labels()
            .into_iter()
            .map(|label| (label, label))
            .collect();
        let incumbent = nested_to_expr_tree(&seed, &label_map).unwrap();
        let tensors = vec![0, 1, 3, 4];
        let keep: HashSet<usize> = tensors.iter().copied().collect();
        let restricted = restrict_expr_tree(&incumbent, &keep).unwrap();
        let log2 = vec![1.0; code.unique_labels().len()];
        let hyper = Hyper::build(&code, &label_map, &log2, log2.len());
        let open = scope_open_labels(&hyper, &tensors);
        let nested = expr_to_nested_counted(&restricted, &code.ixs, &open);

        let mut leaves = nested.leaf_indices();
        leaves.sort_unstable();
        assert_eq!(leaves, tensors);
        assert_eq!(assert_nested_interfaces(&nested, &code), open);
        assert!(nested.is_binary());
    }

    #[test]
    fn test_warm_restricted_seed_is_no_worse_than_greedy_on_optimized_half() {
        let code = grid(2, 4);
        let sizes = uniform_sizes(&code, 2);
        let labels = code.unique_labels();
        let label_map: HashMap<usize, usize> =
            labels.iter().copied().map(|label| (label, label)).collect();
        let log2 = vec![1.0; labels.len()];
        let hyper = Hyper::build(&code, &label_map, &log2, labels.len());
        let part = vec![true, true, true, true, false, false, false, false];
        let (open_a, open_b) = side_open_labels(&hyper, &part);
        let side_tree = |tensors: &[usize], open: &[usize]| {
            let sub_code = EinCode::new(
                tensors
                    .iter()
                    .map(|&tensor| code.ixs[tensor].clone())
                    .collect(),
                open.to_vec(),
            );
            let local = optimize_code(&sub_code, &sizes, &GreedyMethod::default()).unwrap();
            remap_leaves(&local, tensors)
        };
        let a_tensors = vec![0, 1, 2, 3];
        let b_tensors = vec![4, 5, 6, 7];
        let greedy_a = side_tree(&a_tensors, &open_a);
        let incumbent_nested = NestedEinsum::node(
            vec![greedy_a.clone(), side_tree(&b_tensors, &open_b)],
            EinCode::new(vec![open_a.clone(), open_b], code.iy.clone()),
        );
        let incumbent = nested_to_expr_tree(&incumbent_nested, &label_map).unwrap();
        let mut refiner = test_refiner(&code, &sizes, &hyper, &log2);
        let warm = refiner
            .solve_side_warm(&a_tensors, &open_a, &incumbent)
            .unwrap();
        let warm_tc = contraction_complexity(&warm, &sizes, &code.ixs).tc;
        let greedy_tc = contraction_complexity(&greedy_a, &sizes, &code.ixs).tc;

        assert!(warm_tc <= greedy_tc + 1e-9);
        assert!(warm_tc.is_finite());
        assert_eq!(assert_nested_interfaces(&warm, &code), open_a);
    }

    #[test]
    fn test_local_scope_splice_preserves_outside_subtree_byte_for_byte() {
        let waist = ExprTree::node(
            ExprTree::leaf(vec![0, 1, 2], 0),
            ExprTree::leaf(vec![0, 1, 3], 1),
            vec![2, 3],
        );
        let inside_sibling = ExprTree::node(
            ExprTree::leaf(vec![2], 2),
            ExprTree::leaf(vec![3], 3),
            vec![2, 3],
        );
        let scope = ExprTree::node(waist, inside_sibling, vec![4]);
        let outside = ExprTree::node(
            ExprTree::node(
                ExprTree::leaf(vec![4], 4),
                ExprTree::leaf(vec![5], 5),
                vec![4, 5],
            ),
            ExprTree::node(
                ExprTree::leaf(vec![6], 6),
                ExprTree::leaf(vec![7], 7),
                vec![6, 7],
            ),
            vec![4],
        );
        let tree = ExprTree::node(scope, outside, vec![]);
        let (_, waist_leaves, waist_path) = extract_waist_location(&tree, &[1.0; 8]).unwrap();
        assert_eq!(waist_leaves, vec![0, 1]);
        assert_eq!(waist_path, vec![false, false]);
        let scope_path = local_scope_path(&tree, &waist_path, waist_leaves.len());
        assert_eq!(scope_path, vec![false]);
        let outside_before = format!("{:?}", subtree_at_path(&tree, &[true]).unwrap());
        let replacement = ExprTree::node(
            ExprTree::node(
                ExprTree::leaf(vec![2], 2),
                ExprTree::leaf(vec![0, 1, 2], 0),
                vec![0, 1],
            ),
            ExprTree::node(
                ExprTree::leaf(vec![0, 1, 3], 1),
                ExprTree::leaf(vec![3], 3),
                vec![0, 1],
            ),
            vec![4],
        );
        let spliced = replace_subtree_at_path(&tree, &scope_path, &replacement).unwrap();
        let outside_after = format!("{:?}", subtree_at_path(&spliced, &[true]).unwrap());

        assert_eq!(outside_after, outside_before);
        let mut spliced_leaves = spliced.leaf_ids();
        let mut original_leaves = tree.leaf_ids();
        spliced_leaves.sort_unstable();
        original_leaves.sort_unstable();
        assert_eq!(spliced_leaves, original_leaves);
    }

    #[test]
    fn test_opt_in_variants_preserve_permutation_and_strict_acceptance_gate() {
        let code = grid(4, 4);
        let sizes = uniform_sizes(&code, 2);
        let seed = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
        let seed_tc = contraction_complexity(&seed, &sizes, &code.ixs).tc;
        for rebuild in [RebuildMode::Greedy, RebuildMode::WarmRestricted] {
            let (refined, report, _) = refine_capped_seeded_with_trace_opts(
                &seed,
                &code,
                &sizes,
                Duration::MAX,
                1,
                RNG_SEED,
                SurgeryOptions {
                    rebuild,
                    scope: SurgeryScope::Local,
                },
            );
            let refined_tc = contraction_complexity(&refined, &sizes, &code.ixs).tc;
            assert!(refined_tc <= seed_tc + 1e-9);
            assert!(refined_tc.is_finite());
            if report.rebuild_accepts > 0 {
                assert!(refined_tc < seed_tc - 1e-9);
            }
            let mut leaves = refined.leaf_indices();
            leaves.sort_unstable();
            assert_eq!(leaves, (0..code.num_tensors()).collect::<Vec<_>>());
        }
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

    /// `refine_capped` with a generous budget and a small `max_iters` must
    /// stop starting new surgery iterations once the counter reaches the cap.
    #[test]
    fn test_refine_capped_bounds_surgery_calls() {
        let code = grid(4, 4);
        let sizes = uniform_sizes(&code, 2);
        let seed = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
        let (_refined, report) = refine_capped(&seed, &code, &sizes, Duration::from_secs(3600), 3);
        assert!(
            report.surgery_calls <= 3,
            "surgery_calls={} > max_iters=3",
            report.surgery_calls
        );
    }

    /// `refine(...)` must be byte-identical to `refine_capped(..., 0)`: the
    /// former is documented as a thin delegate to the latter.
    #[test]
    fn test_refine_delegates_uncapped() {
        let code = EinCode::new(
            vec![vec![0usize, 1], vec![1, 2], vec![2, 3], vec![3, 4]],
            vec![0, 4],
        );
        let sizes = uniform_sizes(&code, 2);
        let seed = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
        // A generous, identical budget on both sides: this tiny network's
        // surgery loop terminates deterministically (MAX_STALE_ITERS) well
        // inside it, so timing jitter cannot make the two calls diverge.
        let (via_refine, report_refine) = refine(&seed, &code, &sizes, Duration::from_secs(3600));
        let (via_capped, report_capped) =
            refine_capped(&seed, &code, &sizes, Duration::from_secs(3600), 0);
        assert_eq!(format!("{via_refine:?}"), format!("{via_capped:?}"));
        assert_eq!(report_refine, report_capped);
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
            max_iters: 0,
            rng: SmallRng::seed_from_u64(RNG_SEED),
            report: WaistReport {
                n_original: code.num_tensors(),
                surgery_calls: 0,
                cheaper_cuts: 0,
                rebuild_attempts: 0,
                rebuild_accepts: 0,
                waist_min_hits: 0,
            },
            capture_trace: true,
            last_call_trace: None,
        };

        assert!(refiner.waist_surgery(&incumbent, f64::INFINITY).is_none());
        assert_eq!(refiner.report.waist_min_hits, 0);
        assert!(refiner.last_call_trace.is_none());
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
            max_iters: 0,
            rng: SmallRng::seed_from_u64(RNG_SEED),
            report: WaistReport {
                n_original: code.num_tensors(),
                surgery_calls: 0,
                cheaper_cuts: 0,
                rebuild_attempts: 0,
                rebuild_accepts: 0,
                waist_min_hits: 0,
            },
            capture_trace: false,
            last_call_trace: None,
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
            max_iters: 0,
            rng: SmallRng::seed_from_u64(RNG_SEED),
            report: WaistReport {
                n_original: code.num_tensors(),
                surgery_calls: 0,
                cheaper_cuts: 0,
                rebuild_attempts: 0,
                rebuild_accepts: 0,
                waist_min_hits: 0,
            },
            capture_trace: true,
            last_call_trace: None,
        };

        let (rebuilt, rebuilt_tc) = refiner
            .waist_surgery(&incumbent, f64::INFINITY)
            .expect("alternating ring cut should be replaced");

        assert!(rebuilt_tc.is_finite());
        assert_eq!(rebuilt.leaf_count(), code.num_tensors());
        assert_eq!(refiner.report.cheaper_cuts, 1);
        assert_eq!(refiner.report.rebuild_attempts, 1);
        assert_eq!(refiner.report.rebuild_accepts, 1);
        let trace = refiner.last_call_trace.expect("missing call trace");
        assert!(trace.best_alt_cut_cost < trace.incumbent_cut_cost);
        assert!(trace.rebuild_attempted);
        assert!(trace.rebuild_accepted);
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
            max_iters: 0,
            rng: SmallRng::seed_from_u64(RNG_SEED),
            report: WaistReport {
                n_original: code.num_tensors(),
                surgery_calls: 0,
                cheaper_cuts: 0,
                rebuild_attempts: 0,
                rebuild_accepts: 0,
                waist_min_hits: 0,
            },
            capture_trace: true,
            last_call_trace: None,
        };

        assert!(refiner.waist_surgery(&incumbent, f64::INFINITY).is_some());
        assert_eq!(refiner.report.cheaper_cuts, 0);
        assert_eq!(refiner.report.waist_min_hits, 1);
        assert_eq!(refiner.report.rebuild_attempts, 1);
        assert_eq!(refiner.report.rebuild_accepts, 1);
        let trace = refiner.last_call_trace.expect("missing call trace");
        assert_eq!(trace.best_alt_cut_cost, trace.incumbent_cut_cost);
        assert!(trace.rebuild_attempted);
        assert!(trace.rebuild_accepted);
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
            max_iters: 0,
            rng: SmallRng::seed_from_u64(RNG_SEED),
            report: WaistReport {
                n_original: code.num_tensors(),
                surgery_calls: 0,
                cheaper_cuts: 0,
                rebuild_attempts: 0,
                rebuild_accepts: 0,
                waist_min_hits: 0,
            },
            capture_trace: false,
            last_call_trace: None,
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
            max_iters: 0,
            rng: SmallRng::seed_from_u64(RNG_SEED),
            report: WaistReport {
                n_original: code.num_tensors(),
                surgery_calls: 0,
                cheaper_cuts: 0,
                rebuild_attempts: 0,
                rebuild_accepts: 0,
                waist_min_hits: 0,
            },
            capture_trace: false,
            last_call_trace: None,
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

        // Iteration-capped rather than wall-clock budgeted: a 50 ms budget is
        // not enough to complete a surgery call under coverage instrumentation,
        // which made `surgery_calls >= 1` fail intermittently in CI. Capping at
        // one iteration asserts the same thing deterministically.
        let (refined, report) = refine_capped(&seed, &code, &sizes, Duration::MAX, 1);

        assert_eq!(report.surgery_calls, 1);
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

    #[test]
    fn test_single_cut_move_flips_exactly_one_feasible_boundary_tensor() {
        let code = EinCode::new(
            vec![vec![0usize, 3], vec![0, 1], vec![1, 2], vec![2, 3]],
            vec![],
        );
        let label_map: HashMap<usize, usize> = (0..4).map(|label| (label, label)).collect();
        let hyper = Hyper::build(&code, &label_map, &[1.0; 4], 4);
        let before = vec![true, true, false, false];
        let mut rng = SmallRng::seed_from_u64(7);
        let after = single_cut_move(&hyper, before.clone(), 1, 3, &mut rng)
            .expect("ring cut has feasible boundary moves");

        let changed: Vec<usize> = before
            .iter()
            .zip(&after)
            .enumerate()
            .filter_map(|(tensor, (a, b))| (a != b).then_some(tensor))
            .collect();
        assert_eq!(changed.len(), 1);
        let size_a = after.iter().filter(|&&in_a| in_a).count();
        assert!((1..=3).contains(&size_a));

        let moved = changed[0];
        let was_boundary = hyper.tlabels[moved].iter().any(|&label| {
            let members = &hyper.label_tensors[label];
            members.iter().any(|&tensor| before[tensor])
                && members.iter().any(|&tensor| !before[tensor])
        });
        assert!(was_boundary);
    }

    #[test]
    fn test_single_cut_move_respects_balance_outputs_and_boundaries() {
        let ring = EinCode::new(
            vec![vec![0usize, 3], vec![0, 1], vec![1, 2], vec![2, 3]],
            vec![1],
        );
        let label_map: HashMap<usize, usize> = (0..4).map(|label| (label, label)).collect();
        let hyper = Hyper::build(&ring, &label_map, &[1.0; 4], 4);
        let part = vec![true, true, false, false];
        let mut rng = SmallRng::seed_from_u64(9);

        // At an exact-size balance constraint neither side may move, even
        // though the ring has boundary tensors.
        assert!(single_cut_move(&hyper, part.clone(), 2, 2, &mut rng).is_none());

        // With slack, output label 1 is encountered but excluded from gain;
        // one of the remaining boundary moves is still proposed.
        assert!(single_cut_move(&hyper, part, 1, 3, &mut rng).is_some());

        // Two disconnected components have no boundary tensor at this cut.
        let disconnected = EinCode::new(
            vec![vec![0usize], vec![0], vec![1], vec![1]],
            Vec::<usize>::new(),
        );
        let hyper = Hyper::build(&disconnected, &label_map, &[1.0; 4], 4);
        assert!(single_cut_move(&hyper, vec![true, true, false, false], 1, 3, &mut rng).is_none());
    }

    #[test]
    fn test_attachment_and_graft_reject_disconnected_targets() {
        let code = EinCode::new(vec![vec![0usize], vec![0]], Vec::<usize>::new());
        let label_map: HashMap<usize, usize> = [(0, 0)].into();
        let hyper = Hyper::build(&code, &label_map, &[1.0], 1);
        let mut rng = SmallRng::seed_from_u64(13);
        let leaf = ExprTree::leaf(vec![0], 0);

        assert_eq!(
            choose_attachment(&leaf, &[true, false], true, &[0], &hyper, &mut rng),
            Some(Vec::new())
        );
        assert!(choose_attachment(&leaf, &[true, false], true, &[], &hyper, &mut rng).is_none());
        assert!(graft_leaf(leaf, &[false], ExprTree::leaf(vec![0], 1)).is_none());
    }

    #[test]
    fn test_waist_update_proposal_preserves_leaf_permutation_and_interface() {
        let code = EinCode::new(
            vec![vec![0usize, 3], vec![0, 1], vec![1, 2], vec![2, 3]],
            vec![],
        );
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
        let update = WaistUpdate::new(&code.ixs, &code.iy, &[1.0; 4]);
        let mut rng = SmallRng::seed_from_u64(7);
        let candidate = update
            .propose(&incumbent, &mut rng)
            .expect("alternating ring waist has a boundary move");

        let mut leaves = candidate.leaf_ids();
        leaves.sort_unstable();
        assert_eq!(leaves, vec![0, 1, 2, 3]);
        assert_eq!(candidate.labels(), code.iy);
        let (tc, sc, rw) = tree_complexity(&candidate, &[1.0; 4]);
        assert!(tc.is_finite() && sc.is_finite() && rw.is_finite());
    }

    #[test]
    fn test_leaf_spr_preserves_subtrees_off_the_two_edit_paths() {
        fn pair(a: usize, b: usize) -> ExprTree {
            ExprTree::node(
                ExprTree::leaf(vec![a], a),
                ExprTree::leaf(vec![b], b),
                vec![a, b],
            )
        }
        fn subtree_at<'a>(tree: &'a ExprTree, path: &[bool]) -> &'a ExprTree {
            if path.is_empty() {
                return tree;
            }
            let ExprTree::Node { left, right, .. } = tree else {
                panic!("path entered a leaf");
            };
            subtree_at(if path[0] { right } else { left }, &path[1..])
        }

        let before = ExprTree::node(
            ExprTree::node(pair(0, 1), pair(2, 3), vec![0, 1, 2, 3]),
            ExprTree::node(pair(4, 5), pair(6, 7), vec![4, 5, 6, 7]),
            vec![],
        );
        let untouched_23 = format!("{:?}", subtree_at(&before, &[false, true]));
        let untouched_67 = format!("{:?}", subtree_at(&before, &[true, true]));

        let (pruned, leaf) = detach_leaf(&before, 0).expect("leaf 0 exists");
        // After suppressing (0,1), leaf 4 remains at R-L-L.
        let after =
            graft_leaf(pruned, &[true, false, false], leaf).expect("target attachment path exists");

        assert_eq!(
            format!("{:?}", subtree_at(&after, &[false, true])),
            untouched_23
        );
        assert_eq!(
            format!("{:?}", subtree_at(&after, &[true, true])),
            untouched_67
        );
        let mut leaves = after.leaf_ids();
        leaves.sort_unstable();
        assert_eq!(leaves, (0..8).collect::<Vec<_>>());
    }
}
