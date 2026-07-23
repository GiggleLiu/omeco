//! Attempt entry point for the autoresearch validator (attempt-053).
//!
//! Contract: `attempt <graph.json> <budget_ms> <out.json>` — read an einsum
//! graph, search for a contraction order within the wall-clock budget, and keep
//! the best tree found (by pure time complexity `tc`) written atomically to
//! `out.json` in omeco `writejson` format.
//!
//! BOUNDED / cheap-first variable elimination (attempt-053, parent 038).
//! attempt-038 showed a full min-cost variable-elimination (VE) seed WINS on the
//! dense 44-label `dbn_13` hypergraph but FAILS on `nqueens_28` (4086 labels):
//! unbounded VE walks into the treewidth blow-up (tc_ve = 714 vs tc_greedy =
//! 384) and is too slow to finish. This attempt keeps the full VE seed (the dbn
//! winner) but adds a BOUNDED path for large-treewidth instances:
//!
//!   (a) PEEL only the cheap labels — eliminate labels in min-cost order while
//!       the factor an elimination creates stays under a cost cap (the tree-like
//!       periphery: arity-1 tensors, degree-<=2 labels, and other no-regret
//!       eliminations). Stop before the hard core, so no factor blows up and
//!       peeling is fast even at 4k labels. This partitions the original leaves
//!       into super-tensors, each carrying a contraction subtree + live-label
//!       set, and leaves the hard core un-eliminated.
//!   (b) Hand the DEFERRED residual network (the hard-core super-tensors only, a
//!       much smaller problem) to the proven library TreeSA, which anneals a
//!       few-hundred-tensor residual far better than the full graph.
//!   (c) SPLICE each super-tensor's peel subtree back under the residual tree.
//!       The scorer recomputes tc from tree TOPOLOGY alone (node einsum metadata
//!       is ignored), so the spliced tree scores exactly as its topology
//!       deserves; outputs are re-derived by exact outside-occurrence counting.
//!
//! RESULT (see LOG.md): the bounded-peel scaling idea is FALSIFIED — peeling is
//! fast (<260 ms at 4k labels) but peel+residual-TreeSA loses to full-graph
//! TreeSA at every cap (nqueens: >=188 vs 134), because good large-treewidth
//! orders interleave core and periphery and a fixed peel boundary removes that
//! freedom. So `auto` keeps only the winners: greedy (immediate fallback), the
//! deterministic full VE seed (dbn winner), and a full-graph TreeSA anytime
//! doubling run. The peel lane survives behind `MODE=peel` for attribution only.
//! Best-by-`tc` wins. Pure-tc objective (sc ignored). Single-threaded compute:
//! all work runs on one worker
//! thread with a large stack (deep trees exceed the default 8 MB stack at n~4k);
//! the main thread blocks on the join, so CPU never exceeds wall. No per-instance
//! constants; behaviour is invariant under relabeling. Every improvement is
//! written eagerly and atomically (tmp file + rename).

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use omeco::json::writejson;
use omeco::{contraction_complexity, optimize_code, EinCode, GreedyMethod, NestedEinsum, TreeSA};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

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
    // elimination and residual trees (and the json writer that walks them)
    // recurse to depth O(n), overflowing the default 8 MB stack at n~4k. The main
    // thread blocks on the join, so CPU stays single-threaded.
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
    let deadline_ms = budget_ms * 0.95;
    let tc_of = |tree: &NestedEinsum<usize>| contraction_complexity(tree, &sizes, &code.ixs).tc;

    let hg = HyperGraph::build(&graph.ixs, &graph.iy, &sizes);
    let n = code.num_tensors();

    // Optional experiment knobs (default OFF: production behaviour is the `auto`
    // route with an adaptive cap — no per-instance constants). `MODE`=full|peel
    // forces a single lane for head-to-head attribution; `PEEL_CAP` overrides the
    // adaptive cost cap for a local sweep.
    let mode = std::env::var("MODE").unwrap_or_else(|_| "auto".into());
    let cap_override = std::env::var("PEEL_CAP")
        .ok()
        .and_then(|s| s.parse::<f64>().ok());

    // ---- Deterministic greedy: written immediately (always-valid fallback). --
    let mut best = optimize_code(&code, &sizes, &GreedyMethod::default())
        .ok_or("greedy optimizer returned no tree")?;
    let mut best_tc = tc_of(&best);
    write_atomic(&out_path, &best)?;
    let tc_greedy = best_tc;

    // ---- Full VE seed (attempt-038 min-cost): the dbn winner. ---------------
    // Deadline-boxed so a large graph's VE cannot eat the budget; each order is
    // itself abortable. If it beats greedy, VE fits this geometry (dense small
    // hypergraph); if not, it is discarded and the peel/TreeSA lanes take over.
    // Small box: VE only wins on small dense hypergraphs (dbn: 44 labels, VE
    // finishes in <100 ms); on large-label instances it aborts early and is
    // discarded, so a tight box avoids wasting the TreeSA budget.
    let ve_deadline = (budget_ms * 0.06).min(deadline_ms);
    let mut rng = SmallRng::seed_from_u64(0x0000_0053_c0ff_ee00);
    let ve_topo = hg.ve_order(0.0, &mut rng, &start, ve_deadline);
    let ve_tree = hg.build_nested(&ve_topo);
    let tc_ve = tc_of(&ve_tree);
    if tc_ve < best_tc - 1e-9 {
        best = ve_tree;
        best_tc = tc_ve;
        write_atomic(&out_path, &best)?;
    }

    // ---- Cheap-first PEEL: partition leaves into hard-core super-tensors. ----
    // Adaptive cap: peel any elimination whose factor stays below a fraction of
    // the greedy tc frontier — this removes the tree-like periphery (arity-1,
    // degree-<=2, and other cheap eliminations) without approaching the treewidth
    // blow-up. `PEEL_CAP` overrides for local sweeps.
    let cap = cap_override.unwrap_or_else(|| (tc_greedy * 0.30).clamp(6.0, 60.0));
    let peel = hg.peel(cap, &start, (budget_ms * 0.30).min(deadline_ms));
    let k = peel.residual.len();
    let live_labels: HashSet<u32> = peel
        .residual
        .iter()
        .flat_map(|(live, _)| live.iter().copied())
        .collect();
    eprintln!(
        "n={n} n_labels={} | greedy={tc_greedy:.4} ve={tc_ve:.4} best_seed={best_tc:.4} \
         | peel: cap={cap:.1} residual_tensors={k} residual_labels={} peel_ms={:.0} \
         | t={:.0}ms",
        hg.id_label.len(),
        live_labels.len(),
        peel.peel_ms,
        elapsed_ms()
    );

    // Route. The bounded-peel hypothesis was FALSIFIED empirically (see LOG):
    // separating the periphery from the hard core with fixed peel boundaries is
    // consistently worse than full-graph TreeSA, because good orders on
    // large-treewidth instances interleave core and peripheral contractions and
    // peeling removes that freedom (nqueens: peel residual-TreeSA >= 188 vs full
    // 134 at equal budget). So `auto` routes to the proven full-graph TreeSA; the
    // peel lane is retained only for attribution via `MODE=peel`.
    let _ = n;
    let use_peel = match mode.as_str() {
        "peel" => k >= 2,
        _ => false, // "full" and "auto": full-graph TreeSA
    };

    if use_peel {
        // Residual first (its smaller graph anneals better/faster), then a
        // shorter full-graph safety pass with whatever budget remains.
        let split = if mode == "peel" { 0.99 } else { 0.80 };
        let residual_deadline = (budget_ms * split).min(deadline_ms);
        residual_treesa(
            &hg,
            &peel,
            &sizes,
            &out_path,
            &mut best,
            &mut best_tc,
            tc_of,
            &start,
            residual_deadline,
        )?;
        if mode != "peel" {
            treesa_doubling(
                &code,
                &sizes,
                &out_path,
                &mut best,
                &mut best_tc,
                tc_of,
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
            deadline_ms,
        )?;
    }

    eprintln!(
        "t_final={:.0}ms tc_final={best_tc:.4} use_peel={use_peel}",
        elapsed_ms()
    );
    Ok(())
}

/// Anytime library TreeSA on the RESIDUAL network (peel super-tensors as
/// leaves), splicing each residual leaf back to its peel subtree. Doubling
/// `niters`; each round predictively gated so none overruns the deadline.
#[allow(clippy::too_many_arguments)]
fn residual_treesa(
    hg: &HyperGraph,
    peel: &Peel,
    sizes: &HashMap<usize, usize>,
    out_path: &str,
    best: &mut NestedEinsum<usize>,
    best_tc: &mut f64,
    tc_of: impl Fn(&NestedEinsum<usize>) -> f64,
    start: &Instant,
    deadline_ms: f64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let elapsed_ms = || start.elapsed().as_secs_f64() * 1e3;
    let k = peel.residual.len();
    // Residual EinCode: each super-tensor's live labels (original label space).
    let ixs: Vec<Vec<usize>> = peel
        .residual
        .iter()
        .map(|(live, _)| live.iter().map(|&id| hg.id_label[id as usize]).collect())
        .collect();
    let iy: Vec<usize> = hg
        .iy_ids
        .iter()
        .map(|&id| hg.id_label[id as usize])
        .collect();
    let rcode: EinCode<usize> = EinCode::new(ixs, iy);

    let hard = deadline_ms;
    let mut niters = 8usize;
    let mut cost_per_niter = 0.0f64;
    let mut rounds = 0u64;
    loop {
        let remaining = hard - elapsed_ms();
        if remaining <= 0.0 {
            break;
        }
        if rounds > 0 {
            let predicted_double = cost_per_niter * (niters * 2) as f64;
            if predicted_double <= remaining {
                niters = (niters * 2).min(400);
            } else if cost_per_niter > 0.0 {
                let fit = ((remaining * 0.85) / cost_per_niter) as usize;
                if fit < 4 {
                    break;
                }
                niters = fit.min(400);
            }
        }
        let round_start = Instant::now();
        let ts = TreeSA::default()
            .with_ntrials(1)
            .with_niters(niters)
            .with_sc_target(f64::INFINITY);
        let Some(rtree) = optimize_code(&rcode, sizes, &ts) else {
            break;
        };
        // Splice residual topology -> full topology over original leaves.
        let mut supers: Vec<Option<TopoTree>> =
            peel.residual.iter().map(|(_, t)| Some(t.clone())).collect();
        let topo = splice(&rtree, &mut supers);
        let tree = hg.build_nested(&topo);
        let tc = tc_of(&tree);
        if tc < *best_tc - 1e-9 {
            *best = tree;
            *best_tc = tc;
            write_atomic(out_path, best)?;
            eprintln!(
                "t={:.0}ms tc={tc:.4} (residual niters={niters} k={k})",
                elapsed_ms()
            );
        }
        let round_ms = round_start.elapsed().as_secs_f64() * 1e3;
        cost_per_niter = round_ms / niters as f64;
        rounds += 1;
    }
    eprintln!("residual_rounds={rounds}");
    Ok(())
}

/// Proven anytime full-graph TreeSA with a doubling `niters` schedule. Each
/// `optimize_code` round is uninterruptible, so the next round starts only when
/// its predicted cost fits before the deadline. Best-by-tc kept, written
/// atomically.
#[allow(clippy::too_many_arguments)]
fn treesa_doubling(
    code: &EinCode<usize>,
    sizes: &HashMap<usize, usize>,
    out_path: &str,
    best: &mut NestedEinsum<usize>,
    best_tc: &mut f64,
    tc_of: impl Fn(&NestedEinsum<usize>) -> f64,
    start: &Instant,
    deadline_ms: f64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let elapsed_ms = || start.elapsed().as_secs_f64() * 1e3;
    let hard = deadline_ms;
    let mut niters = 8usize;
    let mut cost_per_niter = 0.0f64;
    let mut rounds = 0u64;
    loop {
        let remaining = hard - elapsed_ms();
        if remaining <= 0.0 {
            break;
        }
        if rounds > 0 {
            let predicted_double = cost_per_niter * (niters * 2) as f64;
            if predicted_double <= remaining {
                niters = (niters * 2).min(400);
            } else if cost_per_niter > 0.0 {
                let fit = ((remaining * 0.85) / cost_per_niter) as usize;
                if fit < 4 {
                    break;
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
                "t={:.0}ms tc={tc:.4} (full treesa niters={niters})",
                elapsed_ms()
            );
        }
        let round_ms = round_start.elapsed().as_secs_f64() * 1e3;
        cost_per_niter = round_ms / niters as f64;
        rounds += 1;
    }
    eprintln!("full_treesa_rounds={rounds}");
    Ok(())
}

/// Convert a residual `NestedEinsum` (leaves 0..k-1 = super-tensor ids) into a
/// `TopoTree` over ORIGINAL leaves by moving each super-tensor's peel subtree
/// into its residual leaf (each appears exactly once).
fn splice(ne: &NestedEinsum<usize>, supers: &mut [Option<TopoTree>]) -> TopoTree {
    match ne {
        NestedEinsum::Leaf { tensor_index } => supers[*tensor_index]
            .take()
            .unwrap_or(TopoTree::Leaf(usize::MAX)),
        NestedEinsum::Node { args, .. } => {
            // TreeSA emits binary trees; fold defensively for any arity.
            let mut it = args.iter();
            let first = it.next().map(|a| splice(a, supers));
            let mut acc = first.unwrap_or(TopoTree::Leaf(usize::MAX));
            for a in it {
                let r = splice(a, supers);
                acc = TopoTree::Node(Box::new(acc), Box::new(r));
            }
            acc
        }
    }
}

// =============================================================================
// Hypergraph / variable-elimination seeding (ported from attempt-038)
// =============================================================================

/// A binary topology over leaf tensor indices (no einsum metadata; outputs are
/// derived later by outside-occurrence counting).
enum TopoTree {
    Leaf(usize),
    Node(Box<TopoTree>, Box<TopoTree>),
}

impl Clone for TopoTree {
    fn clone(&self) -> Self {
        match self {
            TopoTree::Leaf(i) => TopoTree::Leaf(*i),
            TopoTree::Node(l, r) => {
                TopoTree::Node(Box::new((**l).clone()), Box::new((**r).clone()))
            }
        }
    }
}

/// Result of a cheap-first peel: the residual super-tensor partition (each a
/// live-label set + contraction subtree over original leaves) and timing.
struct Peel {
    residual: Vec<(Vec<u32>, TopoTree)>,
    peel_ms: f64,
}

/// Interned view of the einsum: dense label ids, per-leaf id sets, `iy` id set,
/// per-id log2 sizes and hyperedge degrees (holder counts).
struct HyperGraph {
    id_label: Vec<usize>,
    leaf_ids: Vec<Vec<u32>>,
    iy_ids: HashSet<u32>,
    log2: Vec<f64>,
    total_count: Vec<u32>,
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

    /// log2-size sum over a sorted id set.
    fn set_cost(&self, ids: &[u32]) -> f64 {
        ids.iter().map(|&id| self.log2[id as usize]).sum()
    }

    /// State shared by the full VE order and the bounded peel: active tensors,
    /// their live label sets and topologies, per-label holder sets.
    #[allow(clippy::type_complexity)]
    fn init_ve_state(
        &self,
    ) -> (
        HashMap<usize, Vec<u32>>,
        HashMap<usize, TopoTree>,
        Vec<HashSet<usize>>,
        Vec<bool>,
    ) {
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
        (live, topo, holders, eliminated)
    }

    /// The min-cost score of eliminating `id`: log2-size of the factor formed by
    /// unioning the live labels of all its holders (`NEG_INFINITY` for a
    /// singleton, eliminated first). `offset` diversifies across restarts.
    fn score(
        &self,
        id: u32,
        live: &HashMap<usize, Vec<u32>>,
        holders: &[HashSet<usize>],
        offset: &[f64],
    ) -> f64 {
        let hs = &holders[id as usize];
        let base = if hs.len() <= 1 {
            f64::NEG_INFINITY
        } else {
            let mut u: HashSet<u32> = HashSet::new();
            for &t in hs {
                for &l in &live[&t] {
                    u.insert(l);
                }
            }
            u.iter().map(|&l| self.log2[l as usize]).sum()
        };
        base + offset[id as usize]
    }

    /// Eliminate label `id`: merge its holder group into one super-tensor. Mutates
    /// the VE state and re-pushes affected labels. Returns the new tensor id.
    #[allow(clippy::too_many_arguments)]
    fn eliminate(
        &self,
        id: u32,
        live: &mut HashMap<usize, Vec<u32>>,
        topo: &mut HashMap<usize, TopoTree>,
        holders: &mut [HashSet<usize>],
        eliminated: &mut [bool],
        next_tid: &mut usize,
        heap: &mut std::collections::BinaryHeap<(std::cmp::Reverse<OrdF64>, u32)>,
        offset: &[f64],
    ) {
        let group: Vec<usize> = holders[id as usize].iter().copied().collect();
        let group_set: HashSet<usize> = group.iter().copied().collect();

        let mut members: Vec<(Vec<u32>, TopoTree)> = Vec::with_capacity(group.len());
        for &t in &group {
            let l = live.remove(&t).unwrap();
            let tp = topo.remove(&t).unwrap();
            members.push((l, tp));
        }
        let (live_union, merged_topo) = self.merge_group(members);

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

        let tnew = *next_tid;
        *next_tid += 1;
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

        for &l in &new_live {
            if !eliminated[l as usize] && !holders[l as usize].is_empty() {
                heap.push((
                    std::cmp::Reverse(OrdF64(self.score(l, live, holders, offset))),
                    l,
                ));
            }
        }
    }

    /// Build a full variable-elimination contraction topology (min-cost order),
    /// merging any remaining components smallest-first at the end. `jitter` (>=0)
    /// adds a per-label random priority offset; `0.0` is deterministic.
    fn ve_order(
        &self,
        jitter: f64,
        rng: &mut SmallRng,
        start: &Instant,
        deadline_ms: f64,
    ) -> TopoTree {
        let m = self.id_label.len();
        let offset: Vec<f64> = (0..m)
            .map(|_| {
                if jitter > 0.0 {
                    jitter * rng.random::<f64>()
                } else {
                    0.0
                }
            })
            .collect();
        let (mut live, mut topo, mut holders, mut eliminated) = self.init_ve_state();
        let mut next_tid = self.n;

        let mut heap: std::collections::BinaryHeap<(std::cmp::Reverse<OrdF64>, u32)> =
            std::collections::BinaryHeap::new();
        for id in 0..m as u32 {
            if !eliminated[id as usize] && !holders[id as usize].is_empty() {
                heap.push((
                    std::cmp::Reverse(OrdF64(self.score(id, &live, &holders, &offset))),
                    id,
                ));
            }
        }

        let mut pops: u64 = 0;
        while let Some((std::cmp::Reverse(OrdF64(s)), id)) = heap.pop() {
            pops += 1;
            if pops % 256 == 0 && start.elapsed().as_secs_f64() * 1e3 >= deadline_ms {
                break;
            }
            if eliminated[id as usize] || holders[id as usize].is_empty() {
                continue;
            }
            let cur = self.score(id, &live, &holders, &offset);
            if (cur - s).abs() > 1e-9 {
                continue;
            }
            self.eliminate(
                id,
                &mut live,
                &mut topo,
                &mut holders,
                &mut eliminated,
                &mut next_tid,
                &mut heap,
                &offset,
            );
        }

        let mut remaining: Vec<(Vec<u32>, TopoTree)> = topo
            .into_iter()
            .map(|(t, tp)| (live.remove(&t).unwrap(), tp))
            .collect();
        if remaining.is_empty() {
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

    /// Cheap-first BOUNDED peel: eliminate labels in min-cost order while the
    /// factor an elimination creates has log2-size <= `cap`. Stops when the
    /// cheapest available elimination exceeds the cap (only the hard core
    /// remains). Returns the residual super-tensor partition (each a live-label
    /// set + contraction subtree over original leaves).
    fn peel(&self, cap: f64, start: &Instant, deadline_ms: f64) -> Peel {
        let t0 = start.elapsed().as_secs_f64() * 1e3;
        let m = self.id_label.len();
        let offset = vec![0.0f64; m];
        let (mut live, mut topo, mut holders, mut eliminated) = self.init_ve_state();
        let mut next_tid = self.n;

        let mut heap: std::collections::BinaryHeap<(std::cmp::Reverse<OrdF64>, u32)> =
            std::collections::BinaryHeap::new();
        for id in 0..m as u32 {
            if !eliminated[id as usize] && !holders[id as usize].is_empty() {
                heap.push((
                    std::cmp::Reverse(OrdF64(self.score(id, &live, &holders, &offset))),
                    id,
                ));
            }
        }

        let mut pops: u64 = 0;
        while let Some((std::cmp::Reverse(OrdF64(s)), id)) = heap.pop() {
            pops += 1;
            if pops % 256 == 0 && start.elapsed().as_secs_f64() * 1e3 >= deadline_ms {
                break;
            }
            if eliminated[id as usize] || holders[id as usize].is_empty() {
                continue;
            }
            let cur = self.score(id, &live, &holders, &offset);
            if (cur - s).abs() > 1e-9 {
                continue;
            }
            // Bounded gate: singletons (score NEG_INFINITY) are always free;
            // otherwise stop the whole peel once the cheapest factor exceeds cap.
            if cur.is_finite() && cur > cap {
                break;
            }
            self.eliminate(
                id,
                &mut live,
                &mut topo,
                &mut holders,
                &mut eliminated,
                &mut next_tid,
                &mut heap,
                &offset,
            );
        }

        let residual: Vec<(Vec<u32>, TopoTree)> = topo
            .into_iter()
            .map(|(t, tp)| (live.remove(&t).unwrap(), tp))
            .collect();
        Peel {
            residual,
            peel_ms: start.elapsed().as_secs_f64() * 1e3 - t0,
        }
    }

    /// Contract a group of tensors into one, choosing a local greedy min-union
    /// pairwise order (size-ordered chain for large groups to bound O(k^2)).
    fn merge_group(&self, mut members: Vec<(Vec<u32>, TopoTree)>) -> (Vec<u32>, TopoTree) {
        if members.len() == 1 {
            return members.pop().unwrap();
        }
        if members.len() > 12 {
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
            let (lj, tj) = members.remove(j);
            let (li, ti) = members.remove(i);
            let u = sorted_union(&li, &lj);
            let node = TopoTree::Node(Box::new(ti), Box::new(tj));
            members.push((u, node));
        }
        members.pop().unwrap()
    }

    /// Convert a topology into a `NestedEinsum`, deriving each node's output by
    /// exact outside-occurrence counting (matches the validator scorer).
    fn build_nested(&self, topo: &TopoTree) -> NestedEinsum<usize> {
        let (tree, _) = self.build_inner(topo);
        tree
    }

    fn build_inner(&self, topo: &TopoTree) -> (NestedEinsum<usize>, HashMap<u32, u32>) {
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
                let left_out: Vec<usize> =
                    lc.keys().map(|&id| self.id_label[id as usize]).collect();
                let right_out: Vec<usize> =
                    rc.keys().map(|&id| self.id_label[id as usize]).collect();

                let mut counts: HashMap<u32, u32> = lc;
                for (id, c) in rc {
                    *counts.entry(id).or_insert(0) += c;
                }
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
