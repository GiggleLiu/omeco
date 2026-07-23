//! Attempt entry point for the autoresearch validator (attempt-038).
//!
//! Contract: `attempt <graph.json> <budget_ms> <out.json>` — read an einsum
//! graph, search for a contraction order within the wall-clock budget, and keep
//! the best tree found (by pure time complexity, `tc`) written atomically to
//! `out.json` in omeco `writejson` format.
//!
//! HYPEREDGE-AWARE pipeline (attempt-038). Targets graphs whose labels are
//! shared by MANY tensors (e.g. graphical-model partition functions such as
//! `dbn_13` and `nqueens_28`), where a shared label is summed out only when its
//! LAST holder is contracted. Pairwise-greedy seeds and uniform SA moves are
//! misaligned with that geometry. The pipeline:
//!
//!   (a) Seed by VARIABLE ELIMINATION over the label hypergraph. A lazy priority
//!       queue drives the elimination order under a min-cost (min weighted fill)
//!       or fewest-holders (min-degree) heuristic; each eliminated label's holder
//!       group is contracted by a local greedy min-union pairwise order. The
//!       resulting topology is converted to a `NestedEinsum` by exact
//!       outside-occurrence counting, so the measured tc equals the scored tc.
//!       The VE orders race a plain greedy seed.
//!
//!   (b) Route by regime. VE FITS when its seed beats greedy — the hyperedge-
//!       dominated case (e.g. dbn_13): the VE basin is excellent, so refine it
//!       with a warm SA — LINEAR beta cool-down (0.12..14), systematic post-order
//!       sweeps as the base layer, plus a light degree-bias (extra rule attempts
//!       at nodes holding high-hyperedge-degree labels, on top of the systematic
//!       pass; pure targeting is blacklisted), pure intensify from the incumbent.
//!       When VE does NOT fit (e.g. nqueens_28, where the greedy seed is a bad
//!       trap and VE is too slow to complete on ~4k labels), fall back to the
//!       proven library TreeSA under an anytime doubling schedule, predictively
//!       gated so no uninterruptible round overruns the budget.
//!
//! Score is pure `tc` (sc ignored). Single-threaded compute: the work runs on one
//! worker thread with a large stack (deep VE trees exceed the default stack); the
//! main thread blocks on the join, so CPU never exceeds wall. No per-instance
//! constants; behaviour is invariant under instance relabeling. Every improvement
//! is written eagerly and atomically (tmp file + rename).

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use omeco::expr_tree::{apply_rule_mut, tree_complexity, ExprTree, Rule, ScratchSpace};
use omeco::json::writejson;
use omeco::treesa::{prepare_warm_anneal, warm_exprtree_to_nested};
use omeco::{contraction_complexity, optimize_code, EinCode, GreedyMethod, NestedEinsum, TreeSA};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

/// Resync the incremental linear-tc accumulator from an exact recompute every
/// this many sweeps, to bound floating-point drift.
const RESYNC_SWEEPS: u64 = 512;

#[derive(Debug, Deserialize)]
struct GraphData {
    ixs: Vec<Vec<usize>>,
    iy: Vec<usize>,
    sizes: HashMap<String, usize>,
}

/// Atomically write `tree` to `out_path` (tmp file + rename) so the polling
/// validator never observes a partially-written file.
fn write_atomic(
    out_path: &str,
    tree: &NestedEinsum<usize>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tmp = format!("{out_path}.tmp");
    writejson(&tmp, tree)?;
    std::fs::rename(&tmp, out_path)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: attempt <graph.json> <budget_ms> <out.json>");
        std::process::exit(2);
    }
    // Run everything on one worker thread with a large stack: variable-
    // elimination trees (and the SA sweeps / json writer that walk them) recurse
    // to depth O(n), which overflows the default 8 MB stack at n~4k. The main
    // thread blocks on the join, so this stays single-threaded in CPU.
    let handle = std::thread::Builder::new()
        .stack_size(1 << 30)
        .spawn(move || run(args))?;
    match handle.join() {
        Ok(r) => r,
        Err(_) => Err("worker thread panicked".into()),
    }
}

fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let start = Instant::now();
    let budget_ms: f64 = args[2].parse()?;
    let out_path = args[3].clone();

    let graph: GraphData = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let sizes: HashMap<usize, usize> = graph
        .sizes
        .iter()
        .map(|(k, v)| Ok::<_, std::num::ParseIntError>((k.parse::<usize>()?, *v)))
        .collect::<Result<_, _>>()?;
    let code: EinCode<usize> = EinCode::new(graph.ixs.clone(), graph.iy.clone());

    let elapsed_ms = || start.elapsed().as_secs_f64() * 1e3;
    let deadline_ms = budget_ms * 0.97;
    let tc_of = |tree: &NestedEinsum<usize>| contraction_complexity(tree, &sizes, &code.ixs).tc;

    // ---- Intern labels to a dense id space [0, m). --------------------------
    let hg = HyperGraph::build(&graph.ixs, &graph.iy, &sizes);

    // ---- Deterministic greedy: written immediately (always-valid fallback). --
    let mut best = optimize_code(&code, &sizes, &GreedyMethod::default())
        .ok_or("greedy optimizer returned no tree")?;
    let mut best_tc = tc_of(&best);
    write_atomic(&out_path, &best)?;
    let tc_greedy = best_tc;

    // ---- (a) Variable-elimination seeds. ------------------------------------
    // Time-boxed: VE never eats the SA budget. Each order is itself
    // deadline-abortable, so a single large-graph order cannot overrun.
    let mut rng = SmallRng::seed_from_u64(0x0000_0038_c0ff_ee00);
    let mut tc_ve = f64::INFINITY;
    let ve_deadline = (budget_ms * 0.15).min(deadline_ms);
    for &heur in &[Heur::MinDegree, Heur::MinCost] {
        if elapsed_ms() >= ve_deadline {
            break;
        }
        let topo = hg.ve_order(heur, 0.0, &mut rng, &start, ve_deadline);
        let tree = hg.build_nested(&topo);
        let tc = tc_of(&tree);
        if tc < tc_ve {
            tc_ve = tc;
        }
        if tc < best_tc - 1e-9 {
            best = tree;
            best_tc = tc;
            write_atomic(&out_path, &best)?;
        }
        // If the first (cheap min-degree) VE order fails to beat greedy, VE does
        // not fit this instance's geometry — don't spend more of the budget on it.
        if tc_ve > tc_greedy + 1e-9 {
            break;
        }
    }
    eprintln!(
        "tc_greedy={tc_greedy:.4} tc_ve={tc_ve:.4} best_seed={best_tc:.4} t={:.0}ms",
        elapsed_ms()
    );

    // Route by regime. VE FITS when its seed beat greedy — the hyperedge-
    // dominated case (e.g. dbn): refine the excellent VE basin with the warm
    // per-sweep-interruptible intensify anneal. Otherwise VE misfits the geometry
    // and the warm anneal gets trapped by the greedy seed (measured on nqueens);
    // fall back to the proven library TreeSA (a faster, better-explored full
    // anneal), predictively gated so no uninterruptible round overruns.
    let ve_fits = tc_ve <= tc_greedy + 1e-9;

    if ve_fits {
        if code.num_tensors() >= 3 && elapsed_ms() < deadline_ms {
            warm_anneal(
                &code,
                &sizes,
                &hg,
                &out_path,
                &mut best,
                &mut best_tc,
                true,
                &start,
                deadline_ms,
            )?;
        }
    } else {
        treesa_doubling(
            &code,
            &sizes,
            &out_path,
            &mut best,
            &mut best_tc,
            tc_of,
            &start,
            budget_ms,
        )?;
    }

    eprintln!(
        "t_final={:.0}ms tc_final={best_tc:.4} ve_fits={ve_fits}",
        elapsed_ms()
    );
    Ok(())
}

/// Proven anytime TreeSA with a doubling `niters` schedule (the library's
/// optimized full anneal, far faster per unit work than the hand-rolled sweep on
/// large deep trees). Each `optimize_code` round is uninterruptible, so the next
/// round starts only when its PREDICTED cost (rounds roughly double) fits before
/// the hard deadline — no round overruns the budget. Best-by-tc is kept and
/// written atomically.
#[allow(clippy::too_many_arguments)]
fn treesa_doubling(
    code: &EinCode<usize>,
    sizes: &HashMap<usize, usize>,
    out_path: &str,
    best: &mut NestedEinsum<usize>,
    best_tc: &mut f64,
    tc_of: impl Fn(&NestedEinsum<usize>) -> f64,
    start: &Instant,
    budget_ms: f64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let elapsed_ms = || start.elapsed().as_secs_f64() * 1e3;
    // Leave headroom for a final round that runs a little longer than predicted,
    // plus the atomic write of a large tree.
    let hard = budget_ms * 0.92;
    // Deeper anneals (larger niters) give better single results; independent
    // rounds give more stochastic tries. Grow niters by doubling while a doubled
    // round still fits; once it would overrun, size each further round to the
    // LARGEST niters whose predicted cost fits the remaining budget (using the
    // measured cost-per-niter), so the whole budget is spent on useful anneals
    // instead of idling. Every round is a fresh random init; best-by-tc is kept.
    let mut niters = 8usize;
    let mut cost_per_niter = 0.0f64; // ms per niter, measured
    let mut rounds = 0u64;
    loop {
        let remaining = hard - elapsed_ms();
        if remaining <= 0.0 {
            break;
        }
        // Choose this round's niters.
        if rounds > 0 {
            let predicted_double = cost_per_niter * (niters * 2) as f64;
            if predicted_double <= remaining {
                niters = (niters * 2).min(400);
            } else if cost_per_niter > 0.0 {
                // Fit the largest round into the remaining time (with margin).
                let fit = ((remaining * 0.85) / cost_per_niter) as usize;
                if fit < 4 {
                    break; // not even a minimal round fits
                }
                niters = fit.min(400);
            }
        }
        let round_start = Instant::now();
        let ts = TreeSA::default()
            .with_ntrials(1)
            .with_niters(niters)
            .with_sc_target(f64::INFINITY);
        let Some(tree) = optimize_code(code, sizes, &ts) else {
            break;
        };
        let tc = tc_of(&tree);
        if tc < *best_tc - 1e-9 {
            *best = tree;
            *best_tc = tc;
            write_atomic(out_path, best)?;
            eprintln!(
                "t={:.0}ms tc={tc:.4} (treesa niters={niters})",
                elapsed_ms()
            );
        }
        let round_ms = round_start.elapsed().as_secs_f64() * 1e3;
        cost_per_niter = round_ms / niters as f64;
        rounds += 1;
    }
    eprintln!("treesa_rounds={rounds}");
    Ok(())
}

// =============================================================================
// Hypergraph / variable-elimination seeding
// =============================================================================

/// Elimination-order heuristic over the label hypergraph.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Heur {
    /// Min weighted fill: eliminate the label whose holder group merges into the
    /// smallest tensor (min union of live-label sizes).
    MinCost,
    /// Fewest holders first (min-degree): eliminate the label held by the fewest
    /// active tensors.
    MinDegree,
}

/// A binary topology over leaf tensor indices (no einsum metadata; outputs are
/// derived later by outside-occurrence counting).
enum TopoTree {
    Leaf(usize),
    Node(Box<TopoTree>, Box<TopoTree>),
}

/// Interned view of the einsum: dense label ids, per-leaf id sets, `iy` id set,
/// per-id log2 sizes and hyperedge degrees (holder counts).
struct HyperGraph {
    /// Original label for each id (id -> label).
    id_label: Vec<usize>,
    /// Per-leaf sorted-unique label ids.
    leaf_ids: Vec<Vec<u32>>,
    /// `iy` label ids.
    iy_ids: HashSet<u32>,
    /// log2 of each id's dimension.
    log2: Vec<f64>,
    /// Number of leaves holding each id (hyperedge degree).
    total_count: Vec<u32>,
    /// Number of leaves.
    n: usize,
}

impl HyperGraph {
    fn build(ixs: &[Vec<usize>], iy: &[usize], sizes: &HashMap<usize, usize>) -> Self {
        let mut id_of: HashMap<usize, u32> = HashMap::new();
        let mut id_label: Vec<usize> = Vec::new();
        let mut log2: Vec<f64> = Vec::new();
        let intern = |label: usize,
                      id_of: &mut HashMap<usize, u32>,
                      id_label: &mut Vec<usize>,
                      log2: &mut Vec<f64>|
         -> u32 {
            if let Some(&id) = id_of.get(&label) {
                id
            } else {
                let id = id_label.len() as u32;
                id_of.insert(label, id);
                id_label.push(label);
                log2.push((sizes.get(&label).copied().unwrap_or(1) as f64).log2());
                id
            }
        };
        let leaf_ids: Vec<Vec<u32>> = ixs
            .iter()
            .map(|ix| {
                let mut v: Vec<u32> = ix
                    .iter()
                    .map(|&l| intern(l, &mut id_of, &mut id_label, &mut log2))
                    .collect();
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();
        let iy_ids: HashSet<u32> = iy
            .iter()
            .map(|&l| intern(l, &mut id_of, &mut id_label, &mut log2))
            .collect();
        let m = id_label.len();
        let mut total_count = vec![0u32; m];
        for v in &leaf_ids {
            for &id in v {
                total_count[id as usize] += 1;
            }
        }
        HyperGraph {
            n: leaf_ids.len(),
            id_label,
            leaf_ids,
            iy_ids,
            log2,
            total_count,
        }
    }

    /// log2-size sum over a sorted id set (the "size" of a tensor with that
    /// label set, in log2 scale).
    fn set_cost(&self, ids: &[u32]) -> f64 {
        ids.iter().map(|&id| self.log2[id as usize]).sum()
    }

    /// Build a variable-elimination contraction topology.
    ///
    /// `jitter` (>= 0) adds a fixed per-label random offset to the priority key,
    /// diversifying the order across restarts; `0.0` is deterministic.
    fn ve_order(
        &self,
        heur: Heur,
        jitter: f64,
        rng: &mut SmallRng,
        start: &Instant,
        deadline_ms: f64,
    ) -> TopoTree {
        let m = self.id_label.len();
        let n = self.n;

        // Fixed per-label priority offset (consistent between push and the
        // stale-check recompute, so lazy deletion stays correct).
        let offset: Vec<f64> = (0..m)
            .map(|_| {
                if jitter > 0.0 {
                    jitter * rng.random::<f64>()
                } else {
                    0.0
                }
            })
            .collect();

        // Active tensors: tid -> (sorted live ids, topology).
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

        // Current priority score of a label (lower = eliminate sooner).
        let score = |id: u32, live: &HashMap<usize, Vec<u32>>, holders: &[HashSet<usize>]| -> f64 {
            let hs = &holders[id as usize];
            let base = match heur {
                Heur::MinDegree => hs.len() as f64,
                Heur::MinCost => {
                    if hs.len() <= 1 {
                        // singleton / no-op: sum it out first.
                        f64::NEG_INFINITY
                    } else {
                        let mut u: HashSet<u32> = HashSet::new();
                        for &t in hs {
                            for &l in &live[&t] {
                                u.insert(l);
                            }
                        }
                        u.iter().map(|&l| self.log2[l as usize]).sum()
                    }
                }
            };
            base + offset[id as usize]
        };

        // Lazy priority queue of (score, id); on pop we recompute and skip stale
        // entries (a fresh entry is pushed whenever a score changes).
        let mut heap: std::collections::BinaryHeap<(std::cmp::Reverse<OrdF64>, u32)> =
            std::collections::BinaryHeap::new();
        for id in 0..m as u32 {
            if !eliminated[id as usize] && !holders[id as usize].is_empty() {
                heap.push((std::cmp::Reverse(OrdF64(score(id, &live, &holders))), id));
            }
        }

        let mut pops: u64 = 0;
        while let Some((std::cmp::Reverse(OrdF64(s)), id)) = heap.pop() {
            pops += 1;
            // Abort on the deadline: remaining active tensors are chain-merged
            // below, yielding a valid (if suboptimal) tree we may discard.
            if pops % 256 == 0 && start.elapsed().as_secs_f64() * 1e3 >= deadline_ms {
                break;
            }
            if eliminated[id as usize] || holders[id as usize].is_empty() {
                continue;
            }
            let cur = score(id, &live, &holders);
            if (cur - s).abs() > 1e-9 {
                // stale (a fresh entry exists); skip.
                continue;
            }
            // Eliminate `id`: merge its holder group into one tensor.
            let group: Vec<usize> = holders[id as usize].iter().copied().collect();
            let group_set: HashSet<usize> = group.iter().copied().collect();

            let mut members: Vec<(Vec<u32>, TopoTree)> = Vec::with_capacity(group.len());
            for &t in &group {
                let l = live.remove(&t).unwrap();
                let tp = topo.remove(&t).unwrap();
                members.push((l, tp));
            }
            let (live_union, merged_topo) = self.merge_group(members);

            // New live set for the merged tensor: drop `id`; keep a label iff it
            // is an output label, or it still has a holder outside this group.
            let mut new_live: Vec<u32> = Vec::with_capacity(live_union.len());
            let mut dropped: Vec<u32> = Vec::new();
            for &l in &live_union {
                if l == id {
                    continue;
                }
                let outside = holders[l as usize].iter().any(|t| !group_set.contains(t));
                if self.iy_ids.contains(&l) || outside {
                    new_live.push(l);
                } else {
                    dropped.push(l);
                }
            }

            // Update holders: remove group members everywhere in the union.
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
            eliminated[id as usize] = true;
            holders[id as usize].clear();
            for &l in &dropped {
                eliminated[l as usize] = true;
                holders[l as usize].clear();
            }

            live.insert(tnew, new_live.clone());
            topo.insert(tnew, merged_topo);

            // Re-push labels whose score may have changed.
            for &l in &new_live {
                if !eliminated[l as usize] && !holders[l as usize].is_empty() {
                    heap.push((std::cmp::Reverse(OrdF64(score(l, &live, &holders))), l));
                }
            }
        }

        // Merge any remaining active tensors (disconnected components / scalars)
        // into a single root, smallest-first.
        let mut remaining: Vec<(Vec<u32>, TopoTree)> = topo
            .into_iter()
            .map(|(t, tp)| (live.remove(&t).unwrap(), tp))
            .collect();
        if remaining.is_empty() {
            // Degenerate: no tensors. Should not happen (n >= 1).
            return TopoTree::Leaf(0);
        }
        remaining.sort_by(|a, b| {
            self.set_cost(&a.0)
                .partial_cmp(&self.set_cost(&b.0))
                .unwrap()
        });
        let (_, root) = self.merge_group(remaining);
        root
    }

    /// Contract a group of tensors into one, choosing a local greedy min-union
    /// pairwise order. Returns the merged live set (union of members' live sets)
    /// and the binary topology. For large groups, falls back to a size-ordered
    /// chain to bound the O(k^2) pair search.
    fn merge_group(&self, mut members: Vec<(Vec<u32>, TopoTree)>) -> (Vec<u32>, TopoTree) {
        if members.len() == 1 {
            return members.pop().unwrap();
        }
        if members.len() > 12 {
            // Chain smallest-first (cheap, avoids O(k^2) blow-up on big groups).
            members.sort_by(|a, b| {
                self.set_cost(&b.0)
                    .partial_cmp(&self.set_cost(&a.0))
                    .unwrap()
            });
            let mut acc = members.pop().unwrap();
            while let Some(next) = members.pop() {
                let u = sorted_union(&acc.0, &next.0);
                let node = TopoTree::Node(Box::new(acc.1), Box::new(next.1));
                acc = (u, node);
            }
            return acc;
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
            // Remove j first (higher index) then i.
            let (lj, tj) = members.remove(j);
            let (li, ti) = members.remove(i);
            let u = sorted_union(&li, &lj);
            let node = TopoTree::Node(Box::new(ti), Box::new(tj));
            members.push((u, node));
        }
        members.pop().unwrap()
    }

    /// Convert a topology into a `NestedEinsum`, deriving each node's output by
    /// exact outside-occurrence counting (matches the validator scorer): a label
    /// is in a node's output iff it appears in `iy` or in some leaf outside the
    /// node's subtree. Returns the tree and the subtree's output label→count map.
    fn build_nested(&self, topo: &TopoTree) -> NestedEinsum<usize> {
        let (tree, _) = self.build_inner(topo);
        tree
    }

    fn build_inner(&self, topo: &TopoTree) -> (NestedEinsum<usize>, HashMap<u32, u32>) {
        match topo {
            TopoTree::Leaf(i) => {
                // Leaf output = its full label set (scorer returns set(labels)).
                let mut counts = HashMap::with_capacity(self.leaf_ids[*i].len());
                for &id in &self.leaf_ids[*i] {
                    counts.insert(id, 1u32);
                }
                (NestedEinsum::leaf(*i), counts)
            }
            TopoTree::Node(l, r) => {
                let (ltree, lc) = self.build_inner(l);
                let (rtree, rc) = self.build_inner(r);
                let left_out: Vec<usize> =
                    lc.keys().map(|&id| self.id_label[id as usize]).collect();
                let right_out: Vec<usize> =
                    rc.keys().map(|&id| self.id_label[id as usize]).collect();

                // Union of children output ids with subtree counts.
                let mut counts: HashMap<u32, u32> = lc;
                for (id, c) in rc {
                    *counts.entry(id).or_insert(0) += c;
                }
                // Node output: keep labels appearing outside the subtree or in iy.
                let mut node_out: Vec<usize> = Vec::new();
                let mut out_counts: HashMap<u32, u32> = HashMap::with_capacity(counts.len());
                for (&id, &sub) in &counts {
                    let outside = self.total_count[id as usize] > sub;
                    if outside || self.iy_ids.contains(&id) {
                        node_out.push(self.id_label[id as usize]);
                        out_counts.insert(id, sub);
                    }
                }
                let eins = EinCode::new(vec![left_out, right_out], node_out);
                (NestedEinsum::node(vec![ltree, rtree], eins), out_counts)
            }
        }
    }
}

/// Sorted-set union of two sorted-unique id slices.
fn sorted_union(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
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

/// Total-ordered f64 wrapper for the priority queue (finite scores only).
#[derive(Clone, Copy, PartialEq)]
struct OrdF64(f64);
impl Eq for OrdF64 {}
impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

// =============================================================================
// Warm simulated annealing (linear beta; systematic sweeps + degree bias)
// =============================================================================

/// One SA sweep: post-order traversal (mutate current node, then recurse).
/// `s_lin` tracks total tc in linear (2^tc) space, updated per accepted rewrite.
/// `deg`/`bias` add extra rule attempts at nodes holding high-degree labels
/// (the systematic single attempt per node is always the base layer).
#[allow(clippy::too_many_arguments)]
fn sweep(
    tree: &mut ExprTree,
    beta: f64,
    log2_sizes: &[f64],
    deg: &[f64],
    deg_max: f64,
    bias: f64,
    rng: &mut SmallRng,
    scratch: &mut ScratchSpace,
    s_lin: &mut f64,
) {
    // Number of rule attempts at this node: 1 (systematic base) plus a
    // degree-biased extra count for nodes carrying high-hyperedge-degree labels.
    let mut attempts = 1u32;
    if bias > 0.0 && deg_max > 0.0 {
        let node_deg = tree.labels().iter().map(|&l| deg[l]).fold(0.0, f64::max);
        attempts += (bias * (node_deg / deg_max)).round() as u32;
    }
    for _ in 0..attempts {
        let rules = Rule::applicable_rules(tree, omeco::expr_tree::DecompositionType::Tree);
        if rules.is_empty() {
            break;
        }
        let rule = rules[rng.random_range(0..rules.len())];
        if let Some(diff) = scratch.rule_diff(tree, rule, log2_sizes, false) {
            let dtc = diff.tc1 - diff.tc0;
            if dtc <= 0.0 || rng.random::<f64>() < (-beta * dtc).exp() {
                *s_lin += f64::exp2(diff.tc1) - f64::exp2(diff.tc0);
                apply_rule_mut(tree, rule, diff.new_labels);
            }
        }
    }
    if let ExprTree::Node { left, right, .. } = tree {
        sweep(
            left, beta, log2_sizes, deg, deg_max, bias, rng, scratch, s_lin,
        );
        sweep(
            right, beta, log2_sizes, deg, deg_max, bias, rng, scratch, s_lin,
        );
    }
}

/// Warm anneal driver. Repeated LINEAR cool-down cycles from the incumbent best;
/// on stagnation, diversify with a fresh randomized variable-elimination seed.
#[allow(clippy::too_many_arguments)]
fn warm_anneal(
    code: &EinCode<usize>,
    sizes: &HashMap<usize, usize>,
    hg: &HyperGraph,
    out_path: &str,
    best: &mut NestedEinsum<usize>,
    best_tc: &mut f64,
    use_ve_diversify: bool,
    start: &Instant,
    deadline_ms: f64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let elapsed_ms = || start.elapsed().as_secs_f64() * 1e3;

    let Some(ctx) = prepare_warm_anneal(code, sizes, best) else {
        return Ok(());
    };
    let log2_sizes = ctx.log2_sizes;
    let labels = ctx.labels;
    let mut tree = ctx.tree;
    let mut scratch = ScratchSpace::new(ctx.nedge);
    let mut rng = SmallRng::seed_from_u64(0x0000_0038_c0ff_ee11);

    // Per-label-id hyperedge degree, aligned to the warm-anneal label ids
    // (`labels[id]` -> original label). Used for the SA move bias.
    let orig_deg: HashMap<usize, f64> = hg
        .id_label
        .iter()
        .zip(hg.total_count.iter())
        .map(|(&lab, &c)| (lab, c as f64))
        .collect();
    let deg: Vec<f64> = labels
        .iter()
        .map(|l| orig_deg.get(l).copied().unwrap_or(0.0))
        .collect();
    let deg_max = deg.iter().cloned().fold(0.0, f64::max);
    // Light bias: at most ~`bias` extra attempts at the highest-degree node.
    let bias = 1.0f64;

    let mut best_expr = tree.clone();
    let mut best_lin = f64::exp2(*best_tc);
    let mut s_lin: f64;

    let n = code.num_tensors();

    // Re-heat temperature (size-scaled) and cold end of each cool-down.
    let reheat_default = (n as f64 * 1.6e-4).clamp(0.04, 0.12);
    let b_lo = reheat_default;
    let b_hi = 14.0;
    let n_levels = 120u64;
    let ni_short = ((n as f64 / 23.0).round() as u64).clamp(8, 16).max(1);
    let ni_long = 100u64;
    // Stagnation threshold before switching to a diversify (fresh full anneal).
    // In the hyperedge-fit regime the VE seed is an excellent basin and diversify
    // (a structurally different, worse seed) only wastes budget that pure
    // intensify refinement uses to descend further, so we never diversify there.
    let stag_threshold = if use_ve_diversify {
        u64::MAX
    } else if n <= 350 {
        3
    } else {
        16
    };

    let mut cycle_idx: u64 = 0;
    let mut sweeps: u64 = 0;
    let mut fresh_cycles: u64 = 0;
    let mut since_improve: u64 = 0;
    // Measured wall time of the most recent sweep (ms); used as a stop margin so
    // a slow final sweep cannot overrun the deadline.
    let mut last_sweep_ms: f64 = 0.0;
    // Randomized-VE seed heuristics cycled through on diversification.
    let heurs = [Heur::MinCost, Heur::MinDegree];
    let alphas = [0.0f64, 0.25, 0.5, 0.75, 1.0];

    'outer: loop {
        let diversify = since_improve >= stag_threshold;
        let (cycle_blo, cycle_ni) = if diversify {
            // Fresh structurally-different basin. When VE seeds are competitive
            // for this instance (the hyperedge-dominated regime), diversify with
            // a fresh RANDOMIZED variable-elimination seed; otherwise fall back to
            // the proven randomized-greedy restart (attempt-032), which is the
            // right generic escape when VE does not fit the geometry.
            let fresh = if use_ve_diversify {
                let heur = heurs[(fresh_cycles as usize) % heurs.len()];
                let topo = hg.ve_order(heur, 0.75, &mut rng, start, deadline_ms);
                Some(hg.build_nested(&topo))
            } else {
                let alpha = alphas[rng.random_range(0..alphas.len())];
                let temp = 0.05 + rng.random::<f64>() * 0.3;
                optimize_code(code, sizes, &GreedyMethod::new(alpha, temp))
            };
            match fresh.and_then(|f| prepare_warm_anneal(code, sizes, &f)) {
                Some(fctx) => {
                    tree = fctx.tree;
                    s_lin = f64::exp2(tree_complexity(&tree, &log2_sizes).0);
                    fresh_cycles += 1;
                    (0.01, ni_long)
                }
                None => {
                    tree = best_expr.clone();
                    s_lin = best_lin;
                    (b_lo, ni_short)
                }
            }
        } else {
            tree = best_expr.clone();
            s_lin = best_lin;
            (b_lo, ni_short)
        };
        cycle_idx += 1;

        let cycle_sweeps = n_levels * cycle_ni;
        let dbeta = if n_levels > 1 {
            (b_hi - cycle_blo) / (n_levels - 1) as f64
        } else {
            0.0
        };
        let mut improved = false;

        for step in 0..cycle_sweeps {
            // Don't START a sweep that would overrun: on big trees a single
            // sweep costs 100s of ms (occasionally seconds), so leave a margin of
            // the last sweep's measured duration before the deadline.
            if elapsed_ms() + last_sweep_ms >= deadline_ms {
                break 'outer;
            }
            sweeps += 1;
            // Linear cool-down: beta = cycle_blo + dbeta*level.
            let level = step / cycle_ni;
            let beta = cycle_blo + dbeta * level as f64;

            let sweep_t0 = start.elapsed().as_secs_f64() * 1e3;
            sweep(
                &mut tree,
                beta,
                &log2_sizes,
                &deg,
                deg_max,
                bias,
                &mut rng,
                &mut scratch,
                &mut s_lin,
            );
            last_sweep_ms =
                (start.elapsed().as_secs_f64() * 1e3 - sweep_t0).max(last_sweep_ms * 0.5);

            if sweeps % RESYNC_SWEEPS == 0 {
                s_lin = f64::exp2(tree_complexity(&tree, &log2_sizes).0);
            }

            if s_lin < best_lin - 1e-9 {
                let exact = tree_complexity(&tree, &log2_sizes).0;
                s_lin = f64::exp2(exact);
                if exact < *best_tc - 1e-9 {
                    *best_tc = exact;
                    best_lin = s_lin;
                    best_expr = tree.clone();
                    *best = warm_exprtree_to_nested(&tree, code, &labels);
                    write_atomic(out_path, best)?;
                    eprintln!("t={:.0}ms tc={exact:.4}", elapsed_ms());
                    improved = true;
                }
            }

            // Secondary guard (the pre-sweep margin check above is primary).
            if elapsed_ms() >= deadline_ms {
                break 'outer;
            }
        }

        if improved {
            since_improve = 0;
        } else {
            since_improve += 1;
        }
    }
    eprintln!("anneal_sweeps={sweeps} cycles={cycle_idx} fresh={fresh_cycles}");
    Ok(())
}
