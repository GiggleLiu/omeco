//! Attempt entry point for the autoresearch validator (attempt-054).
//!
//! Contract: `attempt <graph.json> <budget_ms> <out.json>` — read an einsum
//! graph, search for a contraction order within the wall-clock budget, and keep
//! the best tree found (by pure time complexity, `tc`) written EAGERLY and
//! ATOMICALLY (tmp file + rename) to `out.json` in omeco `writejson` format. The
//! emitted tree's leaves are ALWAYS a permutation of the ORIGINAL 0..n-1.
//!
//! # WAIST SURGERY: global cut improvement of the dominant contraction
//!
//! Pure tc is pinned by the single most expensive contraction — the tree's
//! WAIST, a bipartition (A, B) of the tensors. TreeSA's O(1) rewrites move one
//! leaf at a time and cannot jump between distinct good bipartitions of the same
//! size. Waist surgery injects global information exactly there:
//!
//!   1. SEED + SIMPLIFY. Rank-non-increasing fusion (attempt-039) shrinks the
//!      network; a greedy portfolio seeds the incumbent (written immediately).
//!   2. BASE ENGINE. The 052-style span-gated basin-hopping ratchet: a warm
//!      coarse kick on a clone, then a cold span-gated refinement ladder;
//!      adopt iff it beats the incumbent.
//!   3. WAIST SURGERY (interleaved). Extract the argmax-cost node's leaf
//!      bipartition (A, B). Improve THAT CUT globally on the tensor hypergraph,
//!      ignoring the tree: bounded Fiduccia–Mattheyses passes (gain = change in
//!      summed log2 dims of straddling labels; balance |A| ± slack), seeded from
//!      the current cut and from boundary-BFS alternatives. If a strictly
//!      cheaper comparable-balance cut is found, REBUILD: cold-anneal a subtree
//!      for each side separately (open labels = new cut + each side's external
//!      labels, via outside-occurrence counting so the scorer agrees), join,
//!      splice, and accept iff global tc drops. Iterate as the waist moves.
//!
//! Falsification (equally valuable): if FM/BFS finds NO cheaper balanced cut at
//! the waist of a record-quality incumbent, that instance's waist is a globally
//! minimal balanced cut — a certificate-flavored statement, printed as
//! `WAIST-MIN: instance=... waist_cost=... alternatives_tried=... best_alt=...`.
//!
//! Score is pure `tc` (sc ignored). Single-threaded compute (a large-stack
//! worker carries deep trees; the main thread blocks on join). No per-instance
//! constants; behaviour is invariant under instance relabeling. LINEAR schedules.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use omeco::expr_tree::{
    apply_rule_mut, tree_complexity, DecompositionType, ExprTree, Rule, ScratchSpace,
};
use omeco::json::writejson;
use omeco::{contraction_complexity, optimize_code, EinCode, GreedyMethod, NestedEinsum};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

/// Resync the incremental linear-tc accumulator from an exact recompute every
/// this many sweeps, to bound floating-point drift.
const RESYNC_SWEEPS: u64 = 512;

/// Check the wall clock only every this many sweeps.
const CLOCK_EVERY: u64 = 8;

/// Minimum wall-clock gap between atomic disk writes (validator polls ~0.2s).
const WRITE_EVERY_MS: f64 = 150.0;

/// Number of coarse super-nodes the top span selects (`S_top = ceil(n/30)`).
const TARGET_TOP: usize = 30;

/// Cold end of every linear beta schedule.
const B_HI: f64 = 14.0;

/// FM balance slack around the waist size |A| (fraction).
const FM_SLACK: f64 = 0.18;

/// Maximum FM passes per cut-improvement call.
const FM_MAX_PASSES: usize = 6;

#[derive(Debug, Deserialize)]
struct GraphData {
    ixs: Vec<Vec<usize>>,
    iy: Vec<usize>,
    sizes: HashMap<String, usize>,
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Atomically write `tree` to `out_path` (tmp file + rename).
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
    // Deep trees (SA sweeps / json writer / DFS) recurse to depth O(n); run on a
    // large-stack worker. The main thread blocks on join (single-threaded CPU).
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
    let instance = std::path::Path::new(&args[1])
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let graph: GraphData = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let sizes: HashMap<usize, usize> = graph
        .sizes
        .iter()
        .map(|(k, v)| Ok::<_, std::num::ParseIntError>((k.parse::<usize>()?, *v)))
        .collect::<Result<_, _>>()?;
    let orig_code: EinCode<usize> = EinCode::new(graph.ixs.clone(), graph.iy.clone());
    let n_orig = orig_code.num_tensors();

    let elapsed_ms = || start.elapsed().as_secs_f64() * 1e3;
    let deadline_ms = budget_ms * 0.97;

    if n_orig == 0 {
        return Err("empty einsum".into());
    }

    // ---- SIMPLIFY: rank-non-increasing fusion + splice-back (attempt-039). ----
    let simp = simplify(&orig_code);
    let subtrees = simp.subtrees;
    let code = simp.code;
    let n = code.num_tensors();
    let fusible = 1.0 - (n as f64) / (n_orig as f64);

    let reduced_ixs = code.ixs.clone();
    let tc_reduced =
        |tree: &NestedEinsum<usize>| contraction_complexity(tree, &sizes, &reduced_ixs).tc;
    let tc_full =
        |tree: &NestedEinsum<usize>| contraction_complexity(tree, &sizes, &orig_code.ixs).tc;

    let subtrees_ref = &subtrees;
    let emit =
        |tree: &NestedEinsum<usize>| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let full = splice(tree, subtrees_ref);
            write_atomic(&out_path, &full)
        };

    // ---- Label-id space (shared by ExprTree, tc, FM, and I/O). ----------------
    let labels: Vec<usize> = code.unique_labels();
    let label_map: HashMap<usize, usize> =
        labels.iter().enumerate().map(|(i, &l)| (l, i)).collect();
    let log2_sizes: Vec<f64> = labels
        .iter()
        .map(|&l| (*sizes.get(&l).unwrap_or(&1) as f64).log2())
        .collect();

    eprintln!(
        "probe: instance={instance} n_orig={n_orig} n={n} fusible={fusible:.4} nlabels={} t={:.0}ms",
        labels.len(),
        elapsed_ms()
    );

    // ---- Greedy seed portfolio (always; written immediately). -----------------
    let mut best = optimize_code(&code, &sizes, &GreedyMethod::default())
        .ok_or("greedy optimizer returned no tree")?;
    let mut best_tc = tc_reduced(&best);
    emit(&best)?;

    if n < 3 {
        let full = splice(&best, &subtrees);
        eprintln!(
            "t_final={:.0}ms tc_full={:.4} (trivial)",
            elapsed_ms(),
            tc_full(&full)
        );
        return Ok(());
    }

    let seed_deadline = (budget_ms * 0.04).min(500.0);
    let alphas = [0.0f64, 0.25, 0.5, 0.75, 1.0];
    let temps = [0.03f64, 0.1, 0.3];
    let mut combo = 0usize;
    while elapsed_ms() < seed_deadline {
        let alpha = alphas[combo % alphas.len()];
        let temp = temps[(combo / alphas.len()) % temps.len()];
        combo += 1;
        if let Some(tree) = optimize_code(&code, &sizes, &GreedyMethod::new(alpha, temp)) {
            let tc = tc_reduced(&tree);
            if tc < best_tc - 1e-9 {
                best = tree;
                best_tc = tc;
                emit(&best)?;
            }
        }
    }
    eprintln!("seed_greedy tc={best_tc:.4} t={:.0}ms", elapsed_ms());

    // ---- Build the reduced-network hypergraph for waist surgery. --------------
    let hyper = Hyper::build(&code, &label_map, &log2_sizes, labels.len());

    // ---- Interleaved base engine + waist surgery. -----------------------------
    let mut surgeon = Surgeon {
        code: &code,
        sizes: &sizes,
        subtrees: &subtrees,
        hyper: &hyper,
        original_ixs: &graph.ixs,
        inverse_map: &labels,
        openedges: &code.iy,
        log2_sizes: &log2_sizes,
        out_path: &out_path,
        start: &start,
        deadline_ms,
        rng: SmallRng::seed_from_u64(0x0000_0054_c0ff_ee00),
        scratch: ScratchSpace::new(labels.len()),
        best: &mut best,
        best_tc: &mut best_tc,
        sweeps: 0,
        last_write_ms: elapsed_ms(),
        surgery_calls: 0,
        cheaper_cuts: 0,
        rebuild_accepts: 0,
        waist_min_hits: 0,
        last_waist_cost: 0.0,
        last_best_alt: 0.0,
        alts_tried: 0,
    };
    surgeon.run(n)?;

    let final_tc = *surgeon.best_tc;
    let (calls, cheaper, accepts, wmin) = (
        surgeon.surgery_calls,
        surgeon.cheaper_cuts,
        surgeon.rebuild_accepts,
        surgeon.waist_min_hits,
    );
    let last_waist_cost = surgeon.last_waist_cost;
    let last_best_alt = surgeon.last_best_alt;
    let alts_tried = surgeon.alts_tried;

    let full = splice(&best, &subtrees);
    eprintln!(
        "t_final={:.0}ms tc_reduced={final_tc:.4} tc_full={:.4} leaves={}",
        elapsed_ms(),
        tc_full(&full),
        full.leaf_count(),
    );
    eprintln!(
        "surgery summary: calls={calls} cheaper_cuts={cheaper} rebuild_accepts={accepts} waist_min_hits={wmin}"
    );
    // Falsification certificate: if NO surgery call ever found a cut cheaper than
    // the waist, the frontier's waist is a globally minimal balanced cut here.
    if calls > 0 && cheaper == 0 {
        eprintln!(
            "WAIST-MIN: instance={instance} waist_cost={last_waist_cost:.4} alternatives_tried={alts_tried} best_alt={last_best_alt:.4}"
        );
    }
    Ok(())
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
    /// Tensor adjacency (share ≥1 label), deduped — for boundary BFS seeds.
    adj: Vec<Vec<usize>>,
}

impl Hyper {
    fn build(
        code: &EinCode<usize>,
        label_map: &HashMap<usize, usize>,
        log2: &[f64],
        nlab: usize,
    ) -> Self {
        let n = code.ixs.len();
        let tlabels: Vec<Vec<usize>> = code
            .ixs
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
            if lt.len() > 64 {
                continue;
            }
            for i in 0..lt.len() {
                for j in (i + 1)..lt.len() {
                    adj[lt[i]].push(lt[j]);
                    adj[lt[j]].push(lt[i]);
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

    /// Straddle-cut cost of a bipartition: sum of log2 dims over labels present
    /// on BOTH sides. Output labels are always in the top-node output, so they
    /// add a constant; we include them for a faithful "top-node cost" reading.
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
/// Balance is constrained so |A| stays within [lo, hi]. Returns the improved
/// partition and its straddle-cut cost. Gains are the real reduction in summed
/// log2 dims of straddling (non-output) labels.
fn fm_refine(hyper: &Hyper, mut part: Vec<bool>, lo: usize, hi: usize) -> (Vec<bool>, f64) {
    let n = hyper.n;
    let nlab = hyper.log2.len();
    // cnt_a[l] = number of tensors of label l currently on side A.
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
            // straddle_before = (ct>=1); straddle_after = (cs>=2).
            let before = (ct >= 1) as i32;
            let after = (cs >= 2) as i32;
            g += hyper.log2[l] * (before - after) as f64;
        }
        g
    };

    let mut best_cost = hyper.cut_cost(&part);

    for _pass in 0..FM_MAX_PASSES {
        let mut gain: Vec<f64> = (0..n).map(|t| gain_of(t, &part, &cnt_a)).collect();
        let mut locked = vec![false; n];
        let mut cur_cost = hyper.cut_cost(&part);
        let mut best_seen_cost = cur_cost;
        let mut best_seen_part = part.clone();
        let mut improved_this_pass = false;

        for _step in 0..n {
            // Pick max-gain unlocked feasible move.
            let mut bv: Option<usize> = None;
            let mut bg = f64::NEG_INFINITY;
            for t in 0..n {
                if locked[t] {
                    continue;
                }
                // Feasibility: moving keeps |A| in [lo, hi].
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

/// Grow a connected region of `target` tensors by BFS from `seed` over the
/// tensor adjacency; the region is side A.
fn bfs_seed(hyper: &Hyper, seed: usize, target: usize) -> Vec<bool> {
    let n = hyper.n;
    let mut part = vec![false; n];
    let mut queue = std::collections::VecDeque::new();
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
// Waist extraction from an ExprTree.
// =============================================================================

/// Walk the tree, return (max node cost, leaf-tensor-ids under the argmax node).
fn extract_waist(tree: &ExprTree, log2_sizes: &[f64]) -> (f64, Vec<usize>) {
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
    let mut best = f64::NEG_INFINITY;
    let mut best_leaves = Vec::new();
    walk(tree, log2_sizes, &mut best, &mut best_leaves);
    (best, best_leaves)
}

/// Node tc = output_size + contracted labels (matches `tcscrw`).
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
// The interleaved base engine + waist surgeon.
// =============================================================================

struct Surgeon<'a> {
    code: &'a EinCode<usize>,
    sizes: &'a HashMap<usize, usize>,
    subtrees: &'a [NestedEinsum<usize>],
    hyper: &'a Hyper,
    original_ixs: &'a [Vec<usize>],
    inverse_map: &'a [usize],
    openedges: &'a [usize],
    log2_sizes: &'a [f64],
    out_path: &'a str,
    start: &'a Instant,
    deadline_ms: f64,
    rng: SmallRng,
    scratch: ScratchSpace,
    best: &'a mut NestedEinsum<usize>,
    best_tc: &'a mut f64,
    sweeps: u64,
    last_write_ms: f64,
    surgery_calls: u64,
    cheaper_cuts: u64,
    rebuild_accepts: u64,
    waist_min_hits: u64,
    last_waist_cost: f64,
    last_best_alt: f64,
    alts_tried: u64,
}

impl Surgeon<'_> {
    fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1e3
    }

    /// Emit the current global best (splice to original indices), rate-limited.
    fn emit(&mut self, force: bool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let now = self.elapsed_ms();
        if force || now - self.last_write_ms >= WRITE_EVERY_MS {
            let full = splice(self.best, self.subtrees);
            write_atomic(self.out_path, &full)?;
            self.last_write_ms = now;
        }
        Ok(())
    }

    fn run(&mut self, n: usize) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Span-gate levels: S_top ~= n/TARGET_TOP down to 2 by halving.
        let s_top = ((n + TARGET_TOP - 1) / TARGET_TOP).max(2);
        let mut span_levels: Vec<usize> = Vec::new();
        let mut s = s_top;
        while s > 2 {
            span_levels.push(s);
            s /= 2;
        }
        span_levels.push(2);

        let b_hi = env_f64("ATT_BHI", B_HI);
        let b_lo_cold = env_f64("ATT_BLO_COLD", 2.5);
        let b_kick_lo = env_f64("ATT_BKICK", 0.05);
        let cold_sweeps = env_u64("ATT_SW_COLD", 80).max(1);
        let kick_sweeps = env_u64("ATT_SW_KICK", 300).max(1);

        // Working incumbent as an ExprTree (label-id space).
        let mut incumbent = match nested_to_expr_tree(self.best, self.inverse_map) {
            Some(t) => t,
            None => return Ok(()),
        };
        let mut work_tc = tree_complexity(&incumbent, self.log2_sizes).0;
        if work_tc < *self.best_tc - 1e-9 {
            *self.best_tc = work_tc;
        }

        // Warm up: one cold ladder so the base engine has a refined starting waist.
        {
            let mut tree = incumbent.clone();
            let mut s_lin = f64::exp2(work_tc);
            for &span in span_levels.iter() {
                self.run_level(&mut tree, &mut s_lin, span, b_lo_cold, b_hi, cold_sweeps)?;
                if self.elapsed_ms() >= self.deadline_ms {
                    break;
                }
            }
            let tc = tree_complexity(&tree, self.log2_sizes).0;
            if tc < work_tc - 1e-9 {
                incumbent = tree;
                work_tc = tc;
                self.adopt(&incumbent, work_tc)?;
            }
        }

        let mut iter: u64 = 0;
        let mut since_improve: u64 = 0;
        loop {
            if self.elapsed_ms() >= self.deadline_ms {
                break;
            }
            iter += 1;

            // ---- WAIST SURGERY on the current incumbent. ----
            if let Some((new_tree, new_tc)) = self.waist_surgery(&incumbent, work_tc, iter)? {
                if new_tc < work_tc - 1e-9 {
                    incumbent = new_tree;
                    work_tc = new_tc;
                    since_improve = 0;
                }
            }
            if self.elapsed_ms() >= self.deadline_ms {
                break;
            }

            // ---- BASE ENGINE: 052-style span-gated ratchet on a clone. ----
            let mut tree = incumbent.clone();
            let mut s_lin = f64::exp2(work_tc);
            let tc_incumbent = work_tc;

            self.run_level(&mut tree, &mut s_lin, s_top, b_kick_lo, b_hi, kick_sweeps)?;
            for &span in span_levels.iter() {
                self.run_level(&mut tree, &mut s_lin, span, b_lo_cold, b_hi, cold_sweeps)?;
                if self.elapsed_ms() >= self.deadline_ms {
                    break;
                }
            }
            let tc_after = tree_complexity(&tree, self.log2_sizes).0;
            if tc_after < work_tc - 1e-9 {
                incumbent = tree;
                work_tc = tc_after;
                self.adopt(&incumbent, work_tc)?;
                since_improve = 0;
            } else {
                since_improve += 1;
            }

            eprintln!(
                "t={:.0}ms iter={iter} incumb={tc_incumbent:.4} ratchet={tc_after:.4} best={:.4} since_improve={since_improve}",
                self.elapsed_ms(),
                *self.best_tc,
            );
        }
        // Final forced flush: guarantee the on-disk file is the tracked best
        // (rate-limited `emit(false)` may have skipped the last improvement).
        self.emit(true)?;
        Ok(())
    }

    /// Adopt `tree` (label-id ExprTree) as the global best if it beats it.
    fn adopt(
        &mut self,
        tree: &ExprTree,
        tc: f64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if tc < *self.best_tc - 1e-9 {
            *self.best_tc = tc;
            *self.best =
                expr_tree_to_nested(tree, self.original_ixs, self.inverse_map, self.openedges, 0);
            self.emit(false)?;
        }
        Ok(())
    }

    /// Run one span-gated cooling level over `tree`, emitting global improvements.
    fn run_level(
        &mut self,
        tree: &mut ExprTree,
        s_lin: &mut f64,
        min_span: usize,
        b_lo: f64,
        b_hi: f64,
        sweeps: u64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let denom = (sweeps.saturating_sub(1)).max(1) as f64;
        for k in 0..sweeps {
            self.sweeps += 1;
            let beta = b_lo + (b_hi - b_lo) * (k as f64 / denom);
            gated_sweep(
                tree,
                beta,
                min_span,
                self.log2_sizes,
                &mut self.rng,
                &mut self.scratch,
                s_lin,
            );
            if self.sweeps % RESYNC_SWEEPS == 0 {
                *s_lin = f64::exp2(tree_complexity(tree, self.log2_sizes).0);
            }
            if self.sweeps % CLOCK_EVERY == 0 {
                if *s_lin < f64::exp2(*self.best_tc) - 1e-9 {
                    let exact = tree_complexity(tree, self.log2_sizes).0;
                    *s_lin = f64::exp2(exact);
                    self.adopt(tree, exact)?;
                }
                if self.elapsed_ms() >= self.deadline_ms {
                    break;
                }
            }
        }
        if *s_lin < f64::exp2(*self.best_tc) - 1e-9 {
            let exact = tree_complexity(tree, self.log2_sizes).0;
            *s_lin = f64::exp2(exact);
            self.adopt(tree, exact)?;
        }
        Ok(())
    }

    /// One waist-surgery pass: extract the waist cut, improve it globally via FM
    /// seeded from the current cut and boundary-BFS alternatives, and if a
    /// strictly cheaper comparable-balance cut is found, rebuild both sides and
    /// accept iff global tc drops. Returns the accepted (ExprTree, tc) or None.
    fn waist_surgery(
        &mut self,
        incumbent: &ExprTree,
        work_tc: f64,
        iter: u64,
    ) -> Result<Option<(ExprTree, f64)>, Box<dyn std::error::Error + Send + Sync>> {
        self.surgery_calls += 1;
        let (waist_cost, a_leaves) = extract_waist(incumbent, self.log2_sizes);
        let n = self.hyper.n;
        if a_leaves.is_empty() || a_leaves.len() >= n {
            return Ok(None);
        }
        // Current cut as a partition (side A = waist leaf set).
        let mut cur_part = vec![false; n];
        for &t in &a_leaves {
            if t < n {
                cur_part[t] = true;
            }
        }
        let target_a = a_leaves.len();
        let lo = ((target_a as f64 * (1.0 - FM_SLACK)).floor() as usize).max(1);
        let hi = ((target_a as f64 * (1.0 + FM_SLACK)).ceil() as usize).min(n - 1);

        // Candidate seeds: current cut + boundary-BFS alternatives.
        let mut seeds: Vec<Vec<bool>> = vec![cur_part.clone()];
        let mut deg_order: Vec<usize> = (0..n).collect();
        deg_order.sort_by_key(|&t| std::cmp::Reverse(self.hyper.tlabels[t].len()));
        seeds.push(bfs_seed(self.hyper, deg_order[0], target_a));
        for _ in 0..3 {
            let seed = self.rng.random_range(0..n);
            seeds.push(bfs_seed(self.hyper, seed, target_a));
        }

        let mut best_alt_cost = f64::INFINITY;
        let mut best_alt_part: Option<Vec<bool>> = None;
        let mut tried = 0u64;
        for seed in seeds {
            if self.elapsed_ms() >= self.deadline_ms {
                break;
            }
            let (part, cost) = fm_refine(self.hyper, seed, lo, hi);
            tried += 1;
            let sa = part.iter().filter(|&&p| p).count();
            if sa < lo || sa > hi {
                continue; // out of the comparable-balance band.
            }
            if cost < best_alt_cost {
                best_alt_cost = cost;
                best_alt_part = Some(part);
            }
        }

        self.last_waist_cost = waist_cost;
        self.last_best_alt = best_alt_cost;
        self.alts_tried += tried;
        let gap = waist_cost - best_alt_cost;

        // Revalidation instrumentation (paper-repo issue #1): the historical
        // fields mix cost definitions (waist_cost is a contraction-NODE cost,
        // best_alt a bipartition CUT cost). Log the incumbent partition's cut
        // cost under the SAME functional as best_alt so the figure can compare
        // like with like. Behavior is unchanged.
        let cur_cut = self.hyper.cut_cost(&cur_part);
        let gap_cut = cur_cut - best_alt_cost;

        // Near-clique ("safe") separator note (Tamaki-style).
        let mut safe_note = String::new();
        if let Some(part) = &best_alt_part {
            let (nsep, clique_frac) = self.separator_cliqueness(part);
            safe_note = format!(" sep_labels={nsep} clique_frac={clique_frac:.2}");
        }

        eprintln!(
            "t={:.0}ms iter={iter} WAIST cost={waist_cost:.4} cur_cut={cur_cut:.4} best_alt={best_alt_cost:.4} gap={gap:.4} gap_cut={gap_cut:.4} tried={tried}{safe_note}",
            self.elapsed_ms(),
        );

        if best_alt_cost >= waist_cost - 1e-9 || best_alt_part.is_none() {
            self.waist_min_hits += 1;
            return Ok(None);
        }
        self.cheaper_cuts += 1;
        let part = best_alt_part.unwrap();

        // ---- REBUILD both sides from the improved cut. ----
        let Some(rebuilt) = self.rebuild(&part)? else {
            return Ok(None);
        };
        let new_tc = contraction_complexity(&rebuilt, self.sizes, &self.code.ixs).tc;
        if new_tc < work_tc - 1e-9 {
            self.rebuild_accepts += 1;
            *self.best_tc = self.best_tc.min(new_tc);
            *self.best = rebuilt.clone();
            self.emit(true)?;
            eprintln!(
                "t={:.0}ms iter={iter} REBUILD ACCEPT new_tc={new_tc:.4} (was {work_tc:.4})",
                self.elapsed_ms(),
            );
            if let Some(e) = nested_to_expr_tree(&rebuilt, self.inverse_map) {
                return Ok(Some((e, new_tc)));
            }
        } else {
            eprintln!(
                "t={:.0}ms iter={iter} REBUILD reject new_tc={new_tc:.4} (was {work_tc:.4})",
                self.elapsed_ms(),
            );
        }
        Ok(None)
    }

    /// Rebuild the whole tree from a top-level bipartition `part`: cold-anneal a
    /// subtree for each side (open labels = straddle + that side's output
    /// labels, via outside-occurrence counting), then join at the root.
    fn rebuild(
        &mut self,
        part: &[bool],
    ) -> Result<Option<NestedEinsum<usize>>, Box<dyn std::error::Error + Send + Sync>> {
        let n = self.hyper.n;
        let a_tensors: Vec<usize> = (0..n).filter(|&t| part[t]).collect();
        let b_tensors: Vec<usize> = (0..n).filter(|&t| !part[t]).collect();
        if a_tensors.is_empty() || b_tensors.is_empty() {
            return Ok(None);
        }
        let (open_a_ids, open_b_ids) = self.side_open_labels(part);
        let open_a: Vec<usize> = open_a_ids.iter().map(|&i| self.inverse_map[i]).collect();
        let open_b: Vec<usize> = open_b_ids.iter().map(|&i| self.inverse_map[i]).collect();

        // Fixed V-cycle count => reproducible rebuild trajectory (no wall-clock
        // dependence). 3 V-cycles anneal each half to near-incumbent quality.
        let vcycles = env_u64("ATT_SUB_VC", 3);
        let sub_a = self.solve_side(&a_tensors, &open_a, vcycles)?;
        let sub_b = self.solve_side(&b_tensors, &open_b, vcycles)?;
        let (Some(sub_a), Some(sub_b)) = (sub_a, sub_b) else {
            return Ok(None);
        };

        let eins = EinCode::new(vec![open_a, open_b], self.code.iy.clone());
        Ok(Some(NestedEinsum::node(vec![sub_a, sub_b], eins)))
    }

    /// Open label-ids for each side: a label is open on side A iff it occurs in
    /// A and (occurs in B or is an output label). Symmetric for B.
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

    /// Solve a side sub-einsum: greedy + cold span-gated anneal for a FIXED
    /// number of V-cycles (`vcycles`). Fixed work (not wall-clock-boxed) keeps
    /// the RNG stream and hence the improvement trajectory reproducible across
    /// runs, collapsing the run-to-run variance of the rebuild. Returns a
    /// NestedEinsum over ORIGINAL (reduced) tensor indices.
    fn solve_side(
        &mut self,
        tensors: &[usize],
        open: &[usize],
        vcycles: u64,
    ) -> Result<Option<NestedEinsum<usize>>, Box<dyn std::error::Error + Send + Sync>> {
        if tensors.is_empty() {
            return Ok(None);
        }
        if tensors.len() == 1 {
            return Ok(Some(NestedEinsum::leaf(tensors[0])));
        }
        let sub_ixs: Vec<Vec<usize>> = tensors.iter().map(|&t| self.code.ixs[t].clone()).collect();
        let sub_code = EinCode::new(sub_ixs, open.to_vec());
        let Some(greedy) = optimize_code(&sub_code, self.sizes, &GreedyMethod::default()) else {
            return Ok(None);
        };
        let sub_labels: Vec<usize> = sub_code.unique_labels();
        let sub_log2: Vec<f64> = sub_labels
            .iter()
            .map(|&l| (*self.sizes.get(&l).unwrap_or(&1) as f64).log2())
            .collect();
        let mut sub_best = greedy;
        if let Some(mut tree) = nested_to_expr_tree(&sub_best, &sub_labels) {
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
            let cold_sweeps = 60u64;
            for _vc in 0..vcycles.max(1) {
                for &span in &spans {
                    let denom = (cold_sweeps.saturating_sub(1)).max(1) as f64;
                    for k in 0..cold_sweeps {
                        let beta = 2.5 + (B_HI - 2.5) * (k as f64 / denom);
                        gated_sweep(
                            &mut tree,
                            beta,
                            span,
                            &sub_log2,
                            &mut self.rng,
                            &mut scratch,
                            &mut s_lin,
                        );
                    }
                    s_lin = f64::exp2(tree_complexity(&tree, &sub_log2).0);
                    if s_lin < best_lin - 1e-9 {
                        best_lin = s_lin;
                        best_tree = tree.clone();
                    }
                }
                // Restart each V-cycle from the best tree seen so far.
                tree = best_tree.clone();
                s_lin = best_lin;
            }
            let sub_ixs2: Vec<Vec<usize>> =
                tensors.iter().map(|&t| self.code.ixs[t].clone()).collect();
            sub_best = expr_tree_to_nested(&best_tree, &sub_ixs2, &sub_labels, open, 0);
        }
        let remapped = remap_leaves(&sub_best, tensors);
        Ok(Some(remapped))
    }

    /// Cheap near-clique metric on the separator implied by a cut: the straddle
    /// labels are the separator; report their count and the fraction of straddle
    /// label pairs that co-occur in a common tensor (clique-like => safe).
    fn separator_cliqueness(&self, part: &[bool]) -> (usize, f64) {
        let mut sep: Vec<usize> = Vec::new();
        for (l, ts) in self.hyper.label_tensors.iter().enumerate() {
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
            }
            if a && b {
                sep.push(l);
            }
        }
        let k = sep.len();
        if k < 2 {
            return (k, 1.0);
        }
        if k > 400 {
            return (k, 0.0); // too large to be a clique separator; skip the O(k^2) scan.
        }
        let sepset: HashSet<usize> = sep.iter().copied().collect();
        let total = (k * (k - 1) / 2) as u64;
        let mut seen_pair: HashSet<(usize, usize)> = HashSet::new();
        for tl in &self.hyper.tlabels {
            let local: Vec<usize> = tl.iter().copied().filter(|l| sepset.contains(l)).collect();
            for i in 0..local.len() {
                for j in (i + 1)..local.len() {
                    let key = (local[i].min(local[j]), local[i].max(local[j]));
                    seen_pair.insert(key);
                }
            }
        }
        (k, seen_pair.len() as f64 / total as f64)
    }
}

// =============================================================================
// SIMPLIFY: rank-non-increasing structural simplification (attempt-039).
// =============================================================================

struct Simplified {
    code: EinCode<usize>,
    subtrees: Vec<NestedEinsum<usize>>,
}

fn simplify(code: &EinCode<usize>) -> Simplified {
    let n = code.ixs.len();
    let out_set: HashSet<usize> = code.iy.iter().copied().collect();

    let mut labels: Vec<HashSet<usize>> = code
        .ixs
        .iter()
        .map(|ix| ix.iter().copied().collect())
        .collect();
    let mut label_vec: Vec<Vec<usize>> = code.ixs.clone();
    let mut subtree: Vec<Option<NestedEinsum<usize>>> =
        (0..n).map(|i| Some(NestedEinsum::leaf(i))).collect();
    let mut alive: Vec<bool> = vec![true; n];

    let mut lab2t: HashMap<usize, HashSet<usize>> = HashMap::new();
    for (i, ls) in labels.iter().enumerate() {
        for &l in ls {
            lab2t.entry(l).or_default().insert(i);
        }
    }

    let survivors = |labels: &[HashSet<usize>],
                     lab2t: &HashMap<usize, HashSet<usize>>,
                     a: usize,
                     b: usize|
     -> Vec<usize> {
        let mut out: Vec<usize> = Vec::new();
        for &l in labels[a].iter().chain(labels[b].iter()) {
            if out.contains(&l) {
                continue;
            }
            let occ = lab2t.get(&l).map(|s| s.len()).unwrap_or(0);
            let mut outside = occ;
            if labels[a].contains(&l) {
                outside -= 1;
            }
            if labels[b].contains(&l) {
                outside -= 1;
            }
            if outside > 0 || out_set.contains(&l) {
                out.push(l);
            }
        }
        out
    };

    let mut queue: std::collections::VecDeque<usize> = (0..n).collect();
    let mut in_queue: Vec<bool> = vec![true; n];

    while let Some(t) = queue.pop_front() {
        in_queue[t] = false;
        if !alive[t] {
            continue;
        }
        let mut lab_list: Vec<usize> = labels[t].iter().copied().collect();
        lab_list.sort_unstable();
        let mut chosen: Option<usize> = None;
        'find: for &l in &lab_list {
            let mut neigh: Vec<usize> = lab2t
                .get(&l)
                .map(|s| s.iter().copied().filter(|&x| x != t).collect())
                .unwrap_or_default();
            neigh.sort_unstable();
            for u in neigh {
                if !alive[u] {
                    continue;
                }
                let surv = survivors(&labels, &lab2t, t, u);
                if surv.len() <= labels[t].len().max(labels[u].len()) {
                    chosen = Some(u);
                    break 'find;
                }
            }
        }
        let Some(u) = chosen else { continue };

        let surv = survivors(&labels, &lab2t, t, u);
        let eins = EinCode::new(
            vec![label_vec[t].clone(), label_vec[u].clone()],
            surv.clone(),
        );
        let child_t = subtree[t].take().expect("live tensor has subtree");
        let child_u = subtree[u].take().expect("live tensor has subtree");
        subtree[t] = Some(NestedEinsum::node(vec![child_t, child_u], eins));

        for &l in &labels[u] {
            if let Some(s) = lab2t.get_mut(&l) {
                s.remove(&u);
            }
        }
        for &l in &labels[t] {
            if let Some(s) = lab2t.get_mut(&l) {
                s.remove(&t);
            }
        }
        let mut to_requeue: HashSet<usize> = HashSet::new();
        let new_labels: HashSet<usize> = surv.iter().copied().collect();
        for &l in &new_labels {
            if let Some(s) = lab2t.get(&l) {
                for &x in s {
                    if x != t && x != u && alive[x] {
                        to_requeue.insert(x);
                    }
                }
            }
        }
        for &l in &new_labels {
            lab2t.entry(l).or_default().insert(t);
        }
        labels[t] = new_labels;
        let mut nv: Vec<usize> = surv;
        nv.sort_unstable();
        label_vec[t] = nv;
        alive[u] = false;

        if !in_queue[t] {
            in_queue[t] = true;
            queue.push_back(t);
        }
        for x in to_requeue {
            if !in_queue[x] {
                in_queue[x] = true;
                queue.push_back(x);
            }
        }
    }

    let mut reduced_ixs: Vec<Vec<usize>> = Vec::new();
    let mut subtrees: Vec<NestedEinsum<usize>> = Vec::new();
    for i in 0..n {
        if alive[i] {
            reduced_ixs.push(label_vec[i].clone());
            subtrees.push(subtree[i].take().expect("live tensor has subtree"));
        }
    }
    Simplified {
        code: EinCode::new(reduced_ixs, code.iy.clone()),
        subtrees,
    }
}

/// Replace each leaf `k` of a reduced-network tree by super-tensor `k`'s subtree.
fn splice(tree: &NestedEinsum<usize>, subtrees: &[NestedEinsum<usize>]) -> NestedEinsum<usize> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => subtrees[*tensor_index].clone(),
        NestedEinsum::Node { args, eins } => {
            let new_args = args.iter().map(|a| splice(a, subtrees)).collect();
            NestedEinsum::node(new_args, eins.clone())
        }
    }
}

/// Remap the leaf tensor indices of a NestedEinsum through `map` (leaf i -> map[i]).
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
// Span-gated SA sweep (052).
// =============================================================================

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

// =============================================================================
// NestedEinsum <-> ExprTree conversions (label-id space) (052).
// =============================================================================

fn nested_to_expr_tree(nested: &NestedEinsum<usize>, inverse_map: &[usize]) -> Option<ExprTree> {
    let label_map: HashMap<usize, usize> = inverse_map
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i))
        .collect();
    nested_to_expr_tree_inner(nested, &label_map)
}

fn nested_to_expr_tree_inner(
    nested: &NestedEinsum<usize>,
    label_map: &HashMap<usize, usize>,
) -> Option<ExprTree> {
    match nested {
        NestedEinsum::Leaf { .. } => None,
        NestedEinsum::Node { args, eins } => {
            if args.len() != 2 {
                return None;
            }
            let left = match &args[0] {
                NestedEinsum::Leaf { tensor_index } => {
                    let out_dims: Vec<usize> = eins.ixs[0]
                        .iter()
                        .filter_map(|l| label_map.get(l).copied())
                        .collect();
                    ExprTree::leaf(out_dims, *tensor_index)
                }
                NestedEinsum::Node { .. } => nested_to_expr_tree_inner(&args[0], label_map)?,
            };
            let right = match &args[1] {
                NestedEinsum::Leaf { tensor_index } => {
                    let out_dims: Vec<usize> = eins.ixs[1]
                        .iter()
                        .filter_map(|l| label_map.get(l).copied())
                        .collect();
                    ExprTree::leaf(out_dims, *tensor_index)
                }
                NestedEinsum::Node { .. } => nested_to_expr_tree_inner(&args[1], label_map)?,
            };
            let out_dims: Vec<usize> = eins
                .iy
                .iter()
                .filter_map(|l| label_map.get(l).copied())
                .collect();
            Some(ExprTree::node(left, right, out_dims))
        }
    }
}

fn expr_tree_to_nested(
    tree: &ExprTree,
    original_ixs: &[Vec<usize>],
    inverse_map: &[usize],
    openedges: &[usize],
    level: usize,
) -> NestedEinsum<usize> {
    match tree {
        ExprTree::Leaf(info) => NestedEinsum::leaf(info.tensor_id.unwrap_or(0)),
        ExprTree::Node { left, right, info } => {
            let left_nested =
                expr_tree_to_nested(left, original_ixs, inverse_map, openedges, level + 1);
            let right_nested =
                expr_tree_to_nested(right, original_ixs, inverse_map, openedges, level + 1);
            let iy: Vec<usize> = if level == 0 {
                openedges.to_vec()
            } else {
                info.out_dims.iter().map(|&i| inverse_map[i]).collect()
            };
            let left_labels = child_labels(&left_nested, original_ixs);
            let right_labels = child_labels(&right_nested, original_ixs);
            let eins = EinCode::new(vec![left_labels, right_labels], iy);
            NestedEinsum::node(vec![left_nested, right_nested], eins)
        }
    }
}

fn child_labels(nested: &NestedEinsum<usize>, original_ixs: &[Vec<usize>]) -> Vec<usize> {
    match nested {
        NestedEinsum::Leaf { tensor_index } => {
            original_ixs.get(*tensor_index).cloned().unwrap_or_default()
        }
        NestedEinsum::Node { eins, .. } => eins.iy.clone(),
    }
}
