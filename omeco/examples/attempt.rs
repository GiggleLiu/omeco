//! Attempt entry point for the autoresearch validator (attempt-063).
//!
//! Contract: `attempt <graph.json> <budget_ms> <out.json>` — read an einsum
//! graph, search for a contraction order within the wall-clock budget, and
//! keep the best tree found (by pure time complexity, `tc`) written to
//! `out.json` in omeco `writejson` format. Every improvement is written
//! EAGERLY and ATOMICALLY (tmp file + rename) the instant it is found.
//!
//! # Confirmation-robust early descent on attempt-061
//!
//! The attempt-061 pipeline and targeted waist-band temperature are unchanged.
//! This attempt only makes its early descent easier for an external poller to
//! observe: force a portfolio-end snapshot before annealing, halve the first
//! two band-recompute epochs, and bypass the 150 ms write throttle for every
//! improvement in the first five seconds. `ATT_PARENT=1` bypasses all three
//! changes and recovers pure attempt-061 behavior.
//!
//!   1. SIMPLIFY and SEED. Deterministic + fixed-seed Boltzmann-randomized
//!      greedy portfolio;
//!      the best is written immediately (a valid tree always exists) and
//!      converted to the working `ExprTree`.
//!   2. BASIN-HOP. Each cycle clones the incumbent, runs a long warm anneal
//!      over its coarse nodes only (span >= S_top, beta 0.05 -> 14 over 300
//!      sweeps; finer structure is frozen so the warm phase reorganizes blocks
//!      without shredding them), then runs the full COLD refinement ladder
//!      (span S_top, S_top/2, …, 2; linear beta 2.5 -> 14 per level). The warm
//!      anneal runs on a *clone*, so the incumbent never regresses — the ladder
//!      must beat it to be adopted. NB the warm coarse anneal is load-bearing
//!      (contra 034's "just a perturbation" reading): shortening it to a brief
//!      60-sweep kick plateaus ksg at 41.8; the full 300-sweep anneal reaches
//!      36.9.
//!   3. ITERATE until the deadline; on prolonged stagnation, escape via a fresh
//!      randomized-greedy FLAT warm anneal (a structurally different basin, and
//!      a built-in flat-SA control).
//!
//! Energy is pure `dtc` (sc ignored, sc_target = infinity). The tc of the whole
//! tree is tracked in linear (2^tc) space and updated by exactly the two nodes
//! each accepted local rewrite changes, resynced periodically to bound drift.
//!
//! Single-threaded: no Rayon trials are launched (the harness sets
//! `RAYON_NUM_THREADS=1` and rejects CPU > 1.3x wall). No per-instance
//! constants — every knob is a function of the tensor count `n` (env overrides
//! exist only for local counter-tuning and are never set by the harness).
//! Behaviour is identical under instance relabeling.

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::time::Instant;

use omeco::expr_tree::{
    apply_rule_mut, tree_complexity, DecompositionType, ExprTree, Rule, ScratchSpace,
};
use omeco::greedy::{tree_to_nested_einsum, ContractionTree};
use omeco::incidence_list::{ContractionDims, IncidenceList};
use omeco::json::writejson;
use omeco::{
    contraction_complexity, optimize_code, simplify, splice, EinCode, GreedyMethod, NestedEinsum,
};
use priority_queue::PriorityQueue;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

/// Resync the incrementally-maintained linear tc accumulator from an exact
/// recompute every this many sweeps, to bound floating-point drift.
const RESYNC_SWEEPS: u64 = 512;

/// Number of coarse super-nodes the top span selects. `S_top = ceil(n/30)`
/// gates the ladder top and the perturbation kick to roughly the coarsest ~30
/// nodes, independent of n.
const TARGET_TOP: usize = 30;

/// Check the wall clock only every this many sweeps (keeps overhead low).
const CLOCK_EVERY: u64 = 8;

/// Minimum wall-clock gap between atomic disk writes. The validator polls
/// `out.json` on its own 0.2 s clock, so writing more often is pure waste; on
/// large-boundary trees an unthrottled write-per-improvement during descent
/// costs seconds of serialize/rename I/O. In-memory best tracking stays exact.
const WRITE_EVERY_MS: f64 = 150.0;

/// During confirmation-sensitive early descent, every improvement is written
/// without the normal rate limit for this long. This is a fixed observation
/// window, not an instance-specific search knob.
const EARLY_WRITE_MS: f64 = 5_000.0;

#[derive(Debug, Deserialize)]
struct GraphData {
    ixs: Vec<Vec<usize>>,
    iy: Vec<usize>,
    sizes: HashMap<String, usize>,
}

/// Read an `f64` tuning knob from the environment, falling back to `default`.
fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Read a `u64` tuning knob from the environment, falling back to `default`.
fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_bool(key: &str) -> bool {
    std::env::var(key).is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

/// Width of the heated band in node-cost bits. The plateaus are the
/// pre-registered c={1,2,4} sensitivity values, selected only from reduced n.
fn default_band_bits(n: usize) -> f64 {
    if n < 256 {
        1.0
    } else if n < 2_048 {
        2.0
    } else {
        4.0
    }
}

/// Recompute the band after O(sqrt(n)) sweeps, with bounded bookkeeping cost.
fn default_epoch_sweeps(n: usize) -> u64 {
    (n as f64).sqrt().ceil().clamp(8.0, 32.0) as u64
}

/// Attempt 063 shortens only the first two band-recompute epochs. The parent
/// arm retains attempt 061's epoch length exactly.
fn band_epoch_sweeps(parent: bool, completed_epochs: u64, normal_sweeps: u64) -> u64 {
    if !parent && completed_epochs < 2 {
        normal_sweeps / 2 + normal_sweeps % 2
    } else {
        normal_sweeps
    }
}

/// Warmer per-epoch ramp. Both endpoints scale logarithmically with reduced n.
fn default_band_betas(n: usize) -> (f64, f64) {
    let log_n = (n.max(2) as f64).log2();
    ((0.5 + log_n / 16.0).min(1.25), (4.0 + log_n / 4.0).min(7.0))
}

/// Atomically write `tree` to `out_path` (tmp file + rename) so the polling
/// validator never observes a partially-written file.
fn write_atomic(
    out_path: &str,
    reduced_tree: &NestedEinsum<usize>,
    subtrees: &[NestedEinsum<usize>],
    original_labels: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let tmp = format!("{out_path}.tmp");
    let full_tree = splice(reduced_tree, subtrees);
    let restored = restore_labels(&full_tree, original_labels);
    writejson(&tmp, &restored)?;
    std::fs::rename(&tmp, out_path)?;
    Ok(())
}

fn restore_labels(tree: &NestedEinsum<usize>, original_labels: &[usize]) -> NestedEinsum<usize> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => NestedEinsum::leaf(*tensor_index),
        NestedEinsum::Node { args, eins } => {
            let args = args
                .iter()
                .map(|arg| restore_labels(arg, original_labels))
                .collect();
            let remap =
                |labels: &[usize]| labels.iter().map(|&label| original_labels[label]).collect();
            let eins = EinCode::new(
                eins.ixs.iter().map(|ix| remap(ix)).collect(),
                remap(&eins.iy),
            );
            NestedEinsum::node(args, eins)
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SeedCost(f64);

impl PartialEq for SeedCost {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for SeedCost {}

impl PartialOrd for SeedCost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SeedCost {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
    }
}

type SeedPriority = (SeedCost, Reverse<(usize, usize)>);

fn seed_loss(dims: &ContractionDims<usize>, alpha: f64) -> f64 {
    let output_size = f64::exp2(dims.d01 + dims.d02 + dims.d012);
    let input1_size = f64::exp2(dims.d01 + dims.d12 + dims.d012);
    let input2_size = f64::exp2(dims.d02 + dims.d12 + dims.d012);
    output_size - alpha * (input1_size + input2_size)
}

fn seed_adj_remove(adj: &mut HashMap<usize, HashSet<usize>>, pair: (usize, usize)) {
    if let Some(neighbors) = adj.get_mut(&pair.0) {
        neighbors.remove(&pair.1);
    }
    if let Some(neighbors) = adj.get_mut(&pair.1) {
        neighbors.remove(&pair.0);
    }
}

fn seed_adj_insert(adj: &mut HashMap<usize, HashSet<usize>>, pair: (usize, usize)) {
    adj.entry(pair.0).or_default().insert(pair.1);
    adj.entry(pair.1).or_default().insert(pair.0);
}

fn seed_select_pair<R: Rng>(
    queue: &mut PriorityQueue<(usize, usize), SeedPriority>,
    temperature: f64,
    rng: &mut R,
    adj: &mut HashMap<usize, HashSet<usize>>,
) -> Option<(usize, usize)> {
    let (pair1, priority1) = queue.pop()?;
    seed_adj_remove(adj, pair1);
    if temperature <= 0.0 || queue.is_empty() {
        return Some(pair1);
    }
    let (pair2, priority2) = queue.pop()?;
    seed_adj_remove(adj, pair2);
    let probability = (-(priority2.0 .0 - priority1.0 .0) / temperature).exp();
    if rng.random::<f64>() < probability {
        queue.push(pair1, priority1);
        seed_adj_insert(adj, pair1);
        Some(pair2)
    } else {
        queue.push(pair2, priority2);
        seed_adj_insert(adj, pair2);
        Some(pair1)
    }
}

/// The library's stochastic greedy uses thread entropy. This local equivalent
/// keeps attempt 052's Boltzmann portfolio while honoring the fixed-seed gate.
fn optimize_greedy_seeded<R: Rng>(
    code: &EinCode<usize>,
    sizes: &HashMap<usize, usize>,
    alpha: f64,
    temperature: f64,
    rng: &mut R,
) -> Option<NestedEinsum<usize>> {
    let mut incidence = IncidenceList::<usize, usize>::from_eincode(&code.ixs, &code.iy);
    let original = incidence.clone();
    let log2_sizes: HashMap<usize, f64> = sizes
        .iter()
        .map(|(&label, &size)| (label, (size as f64).log2()))
        .collect();
    let mut vertices: Vec<usize> = incidence.vertices().copied().collect();
    vertices.sort_unstable();
    if vertices.is_empty() {
        return None;
    }
    if vertices.len() == 1 {
        return Some(NestedEinsum::leaf(vertices[0]));
    }

    let mut queue: PriorityQueue<(usize, usize), SeedPriority> = PriorityQueue::new();
    let mut adj: HashMap<usize, HashSet<usize>> = HashMap::new();
    let mut live: BTreeSet<usize> = vertices.iter().copied().collect();
    let mut trees: HashMap<usize, ContractionTree> = vertices
        .iter()
        .map(|&vertex| (vertex, ContractionTree::leaf(vertex)))
        .collect();
    for &left in &vertices {
        let mut neighbors = incidence.neighbors(&left);
        neighbors.sort_unstable();
        for right in neighbors.into_iter().filter(|&right| right > left) {
            let pair = (left, right);
            let dims = ContractionDims::compute(&incidence, &log2_sizes, &left, &right);
            queue.push(pair, (SeedCost(seed_loss(&dims, alpha)), Reverse(pair)));
            seed_adj_insert(&mut adj, pair);
        }
    }

    let mut next_vertex = vertices.last().copied().unwrap_or(0) + 1;
    while incidence.nv() > 1 {
        let (left, right) = if queue.is_empty() {
            let mut iter = live.iter();
            (*iter.next()?, *iter.next()?)
        } else {
            seed_select_pair(&mut queue, temperature, rng, &mut adj)?
        };
        if incidence.edges(&left).is_none() || incidence.edges(&right).is_none() {
            continue;
        }
        let dims = ContractionDims::compute(&incidence, &log2_sizes, &left, &right);
        let joined = ContractionTree::node(trees.remove(&left)?, trees.remove(&right)?);
        let new_vertex = next_vertex;
        next_vertex += 1;
        incidence.set_edges(new_vertex, dims.edges_out);
        incidence.remove_edges(&dims.edges_remove);
        incidence.delete_vertex(&left);
        incidence.delete_vertex(&right);
        live.remove(&left);
        live.remove(&right);
        live.insert(new_vertex);
        trees.insert(new_vertex, joined);

        let mut neighbors = incidence.neighbors(&new_vertex);
        neighbors.sort_unstable();
        for other in neighbors {
            let pair = (new_vertex.min(other), new_vertex.max(other));
            let dims = ContractionDims::compute(&incidence, &log2_sizes, &new_vertex, &other);
            queue.push(pair, (SeedCost(seed_loss(&dims, alpha)), Reverse(pair)));
            seed_adj_insert(&mut adj, pair);
        }

        let mut stale = Vec::new();
        for dead in [left, right] {
            if let Some(partners) = adj.get(&dead) {
                stale.extend(
                    partners
                        .iter()
                        .map(|&other| (dead.min(other), dead.max(other))),
                );
            }
        }
        stale.sort_unstable();
        stale.dedup();
        for pair in stale {
            queue.remove(&pair);
        }
        for dead in [left, right] {
            if let Some(partners) = adj.remove(&dead) {
                for other in partners {
                    if let Some(neighbors) = adj.get_mut(&other) {
                        neighbors.remove(&dead);
                    }
                }
            }
        }
    }

    let tree = trees.into_values().next()?;
    Some(tree_to_nested_einsum(&tree, &original, &code.iy))
}

fn seed_trial_count(n: usize) -> usize {
    ((n.max(2) as f64).log2().ceil() as usize + 5).clamp(6, 15)
}

fn parse_budget_ms(value: &str) -> Result<f64, Box<dyn std::error::Error>> {
    let budget_ms: f64 = value.parse()?;
    if !budget_ms.is_finite() || budget_ms <= 0.0 {
        return Err("budget_ms must be a finite positive number".into());
    }
    Ok(budget_ms)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: attempt <graph.json> <budget_ms> <out.json>");
        std::process::exit(2);
    }
    let start = Instant::now();
    let budget_ms = parse_budget_ms(&args[2])?;
    let out_path = args[3].clone();

    let graph: GraphData = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let input_sizes: HashMap<usize, usize> = graph
        .sizes
        .iter()
        .map(|(k, v)| Ok::<_, std::num::ParseIntError>((k.parse::<usize>()?, *v)))
        .collect::<Result<_, _>>()?;
    // Canonical first-occurrence IDs make simplification and every later tie
    // break invariant to a bijective relabeling of the input index values.
    let mut label_map = HashMap::new();
    let mut original_labels = Vec::new();
    for &label in graph.ixs.iter().flatten().chain(graph.iy.iter()) {
        if let std::collections::hash_map::Entry::Vacant(entry) = label_map.entry(label) {
            let id = original_labels.len();
            entry.insert(id);
            original_labels.push(label);
        }
    }
    let canonicalize = |labels: &[usize]| labels.iter().map(|label| label_map[label]).collect();
    let original_code = EinCode::new(
        graph.ixs.iter().map(|ix| canonicalize(ix)).collect(),
        canonicalize(&graph.iy),
    );
    let sizes: HashMap<usize, usize> = original_labels
        .iter()
        .enumerate()
        .map(|(id, label)| (id, *input_sizes.get(label).unwrap_or(&1)))
        .collect();
    let simplified = simplify(&original_code, &sizes);
    let code = simplified.code;
    let subtrees = simplified.subtrees;
    let n = code.num_tensors();
    let parent = env_bool("ATT_PARENT");

    let elapsed_ms = || start.elapsed().as_secs_f64() * 1e3;
    let deadline_ms = budget_ms * 0.97;
    let tc_of = |tree: &NestedEinsum<usize>| contraction_complexity(tree, &sizes, &code.ixs).tc;
    let full_tc_of = |tree: &NestedEinsum<usize>| {
        contraction_complexity(&splice(tree, &subtrees), &sizes, &original_code.ixs).tc
    };

    // ---- Integer label space (shared by ExprTree, tc, and I/O). --------------
    // `labels[id]` = original label; `log2_sizes[id]` = log2 of that label's
    // dimension. This mirrors omeco's internal convention.
    let labels: Vec<usize> = code.unique_labels();
    let log2_sizes: Vec<f64> = labels
        .iter()
        .map(|&l| (*sizes.get(&l).unwrap_or(&1) as f64).log2())
        .collect();

    // ---- Trivial cases: nothing to anneal. -----------------------------------
    if n == 0 {
        return Err("empty einsum".into());
    }
    if n <= 2 {
        let best = optimize_code(&code, &sizes, &GreedyMethod::default())
            .ok_or("greedy optimizer returned no tree")?;
        write_atomic(&out_path, &best, &subtrees, &original_labels)?;
        return Ok(());
    }

    // ---- Seed portfolio: deterministic + Boltzmann-randomized greedy. --------
    // Deterministic greedy first, written immediately so a valid result always
    // exists regardless of what follows.
    let mut best = optimize_code(&code, &sizes, &GreedyMethod::default())
        .ok_or("greedy optimizer returned no tree")?;
    let mut best_tc = tc_of(&best);
    write_atomic(&out_path, &best, &subtrees, &original_labels)?;

    let alphas = [0.0f64, 0.25, 0.5, 0.75, 1.0];
    let temps = [0.03f64, 0.1, 0.3];
    let mut greedy_rng = SmallRng::seed_from_u64(0x0000_0061_5eed_f00d);
    let mut combo = 0usize;
    while combo < seed_trial_count(n) && elapsed_ms() < deadline_ms {
        let alpha = alphas[combo % alphas.len()];
        let temp = temps[(combo / alphas.len()) % temps.len()];
        combo += 1;
        if let Some(tree) = optimize_greedy_seeded(&code, &sizes, alpha, temp, &mut greedy_rng) {
            let tc = tc_of(&tree);
            if tc < best_tc - 1e-9 {
                best = tree;
                best_tc = tc;
                write_atomic(&out_path, &best, &subtrees, &original_labels)?;
            }
        }
    }
    // Attempt 063 refreshes the portfolio winner immediately before annealing
    // so a fresh confirmation poll cannot depend on the timing of an earlier
    // greedy improvement. ATT_PARENT=1 preserves attempt 061's write stream.
    if !parent {
        write_atomic(&out_path, &best, &subtrees, &original_labels)?;
    }
    let tc_greedy = best_tc;
    eprintln!(
        "t={:.0}ms seed_greedy tc={tc_greedy:.4} full_tc={:.4} (n={n}, {} greedy trials)",
        elapsed_ms(),
        full_tc_of(&best),
        combo + 1
    );

    // ---- Seed the basin-hopper from the greedy portfolio best. ---------------
    // The coarsener of attempt-034 is removed: it seeded strictly worse than
    // greedy on every scale instance and 034 always fell back here anyway.
    let seed_expr = match nested_to_expr_tree(&best, &labels) {
        Some(t) => t,
        None => {
            // n >= 3 greedy always yields internal nodes; this is defensive.
            write_atomic(&out_path, &best, &subtrees, &original_labels)?;
            return Ok(());
        }
    };

    // ---- Scale-structured basin-hopping + targeted waist-band heating. -------
    let band_bits = env_f64("ATT_BAND_BITS", default_band_bits(n)).max(0.0);
    let epoch_sweeps = env_u64("ATT_EPOCH_SWEEPS", default_epoch_sweeps(n)).max(1);
    let (default_band_beta_lo, default_band_beta_hi) = default_band_betas(n);
    let band_beta_lo = env_f64("ATT_BAND_BLO", default_band_beta_lo);
    let band_beta_hi = env_f64("ATT_BAND_BHI", default_band_beta_hi).max(band_beta_lo);
    let max_sweeps = env_u64("ATT_MAX_SWEEPS", u64::MAX);
    let diagnostics = env_bool("ATT_DIAG");
    eprintln!(
        "ATT_CONFIG mode={} n={} c={:.3} epoch_sweeps={} band_beta_lo={:.6} \
         band_beta_hi={:.6} max_sweeps={}",
        if parent { "parent" } else { "band" },
        n,
        band_bits,
        epoch_sweeps,
        band_beta_lo,
        band_beta_hi,
        max_sweeps
    );

    let mut ann = Annealer {
        original_ixs: &code.ixs,
        inverse_map: &labels,
        openedges: &code.iy,
        log2_sizes: &log2_sizes,
        size_dict: &sizes,
        code: &code,
        scratch: ScratchSpace::new(labels.len()),
        rng: SmallRng::seed_from_u64(0x0000_0052_c0ff_ee00),
        greedy_rng,
        out_path: &out_path,
        start: &start,
        deadline_ms,
        best: &mut best,
        best_tc: &mut best_tc,
        subtrees: &subtrees,
        original_labels: &original_labels,
        parent,
        diagnostics,
        band_bits,
        epoch_sweeps,
        band_beta_lo,
        band_beta_hi,
        max_sweeps,
        sweeps: 0,
        band_epochs: 0,
        accepts: 0,
        last_write_ms: elapsed_ms(),
        pending_write: false,
    };
    ann.run(seed_expr, n)?;
    // Forced final flush even if the deadline landed inside the write window.
    write_atomic(&out_path, ann.best, &subtrees, &original_labels)?;
    let (final_tc, final_sweeps, final_accepts) = (full_tc_of(ann.best), ann.sweeps, ann.accepts);

    eprintln!(
        "t={:.0}ms tc_final={final_tc:.4} sweeps={final_sweeps} accepts={final_accepts}",
        elapsed_ms(),
    );
    Ok(())
}

/// Shared state for the basin-hopping annealer.
struct Annealer<'a> {
    original_ixs: &'a [Vec<usize>],
    inverse_map: &'a [usize],
    openedges: &'a [usize],
    log2_sizes: &'a [f64],
    size_dict: &'a HashMap<usize, usize>,
    code: &'a EinCode<usize>,
    scratch: ScratchSpace,
    rng: SmallRng,
    greedy_rng: SmallRng,
    out_path: &'a str,
    start: &'a Instant,
    deadline_ms: f64,
    best: &'a mut NestedEinsum<usize>,
    best_tc: &'a mut f64,
    subtrees: &'a [NestedEinsum<usize>],
    original_labels: &'a [usize],
    parent: bool,
    diagnostics: bool,
    band_bits: f64,
    epoch_sweeps: u64,
    band_beta_lo: f64,
    band_beta_hi: f64,
    max_sweeps: u64,
    sweeps: u64,
    /// Number of completed band-recompute epochs across all levels.
    band_epochs: u64,
    accepts: u64,
    /// Wall-clock ms of the last atomic disk write (writes are rate-limited to
    /// the validator's poll resolution so rapid descent does not trigger an
    /// O(n) serialize-and-rename storm on large-boundary trees).
    last_write_ms: f64,
    /// An in-memory best newer than the most recent atomic disk write.
    pending_write: bool,
}

impl Annealer<'_> {
    fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1e3
    }

    fn exhausted(&self) -> bool {
        self.elapsed_ms() >= self.deadline_ms || self.sweeps >= self.max_sweeps
    }

    /// Drive the basin-hopping schedule until the deadline.
    fn run(&mut self, seed: ExprTree, n: usize) -> Result<(), Box<dyn std::error::Error>> {
        // Span gate levels for the COLD ladder: S_top ~= n/TARGET_TOP down to 2
        // by halving. A node is rewritten only if its leaf-span >= S, so finer
        // structure below the current front stays frozen; lowering S unlocks
        // progressively finer subtrees (the "uncoarsening" that ratchets tc).
        let s_top = ((n + TARGET_TOP - 1) / TARGET_TOP).max(2);
        let mut span_levels: Vec<usize> = Vec::new();
        let mut s = s_top;
        while s > 2 {
            span_levels.push(s);
            s /= 2;
        }
        span_levels.push(2);

        // Beta schedules (linear cool-downs). The kick starts HOT (only coarse
        // nodes are unfrozen, so hot cannot shred fine structure); every ladder
        // level starts COLD so the perturbed coarse structure is refined, not
        // re-melted.
        let b_hi = env_f64("ATT_BHI", 14.0);
        let b_lo_cold = env_f64("ATT_BLO_COLD", 2.5);
        let b_kick_lo = env_f64("ATT_BKICK", 0.05);
        let cold_sweeps = env_u64("ATT_SW_COLD", 80).max(1);
        // The warm coarse anneal is LOAD-BEARING (contra 034's "just a
        // perturbation" reading): a long 0.05->14 anneal over the coarsest
        // nodes does real block reorganization the cold ladder then refines.
        // Probes on ksg (n=5197): 60 sweeps plateaus at 41.8; 300 reaches 36.9.
        let kick_sweeps = env_u64("ATT_SW_KICK", 300).max(1);
        // Fresh-restart diversification after this many cycles with no global
        // improvement (size-scaled: small graphs diversify sooner).
        let stag_threshold = env_u64("ATT_STAG", if n <= 400 { 2 } else { 5 }).max(1);

        // Working incumbent, kept as an ExprTree for cheap restart clones. Its
        // tc (`work_tc`) equals the global best (`*self.best_tc`) here because
        // the seed IS the best; the emitted best is always the global minimum
        // tracked in `*self.best`.
        let mut best_expr = seed;
        let mut work_tc = tree_complexity(&best_expr, self.log2_sizes).0;

        let mut vcycle: u64 = 0;
        let mut since_improve: u64 = 0;
        let mut survived: u64 = 0; // #cycles whose ladder recovered past incumbent

        loop {
            if self.exhausted() {
                break;
            }
            vcycle += 1;

            if since_improve >= stag_threshold {
                // Structurally different basin: a fresh randomized greedy, then
                // a FLAT warm anneal (built-in flat-SA control).
                let alpha = [0.0f64, 0.25, 0.5, 0.75, 1.0][self.rng.random_range(0..5)];
                let temp = 0.05 + self.rng.random::<f64>() * 0.3;
                let (mut tree, mut s_lin) = match optimize_greedy_seeded(
                    self.code,
                    self.size_dict,
                    alpha,
                    temp,
                    &mut self.greedy_rng,
                )
                .and_then(|g| nested_to_expr_tree(&g, self.inverse_map))
                {
                    Some(t) => {
                        let sl = f64::exp2(tree_complexity(&t, self.log2_sizes).0);
                        (t, sl)
                    }
                    None => (best_expr.clone(), f64::exp2(work_tc)),
                };
                since_improve = 0;
                let improved = self.run_level(
                    &mut tree,
                    &mut s_lin,
                    &mut best_expr,
                    &mut work_tc,
                    2,
                    b_kick_lo,
                    b_hi,
                    cold_sweeps,
                    "restart",
                )?;
                eprintln!(
                    "t={:.0}ms vcycle={vcycle} FLAT-control tc={:.4} improved={improved}",
                    self.elapsed_ms(),
                    self.best_tc
                );
                continue;
            }

            // Basin-hop: clone the incumbent, kick its coarse nodes, then run
            // the full cold ladder. The kick is on a clone, so best_expr (the
            // restart point) never regresses.
            let mut tree = best_expr.clone();
            let mut s_lin = f64::exp2(work_tc);
            let tc_incumbent = work_tc;

            // Warm perturbation kick over the coarsest nodes only.
            self.run_level(
                &mut tree,
                &mut s_lin,
                &mut best_expr,
                &mut work_tc,
                s_top,
                b_kick_lo,
                b_hi,
                kick_sweeps,
                "kick",
            )?;
            let tc_after_kick = tree_complexity(&tree, self.log2_sizes).0;

            // Cold refinement ladder.
            let mut cycle_improved = false;
            for &span in span_levels.iter() {
                let improved = self.run_level(
                    &mut tree,
                    &mut s_lin,
                    &mut best_expr,
                    &mut work_tc,
                    span,
                    b_lo_cold,
                    b_hi,
                    cold_sweeps,
                    "cold",
                )?;
                cycle_improved |= improved;
                if self.exhausted() {
                    break;
                }
            }
            let tc_after_ladder = tree_complexity(&tree, self.log2_sizes).0;
            if tc_after_ladder < tc_incumbent - 1e-9 {
                survived += 1;
            }

            eprintln!(
                "t={:.0}ms vcycle={vcycle} incumb={tc_incumbent:.4} kick={tc_after_kick:.4} \
                 ladder={tc_after_ladder:.4} best={:.4} net_gain={:.4} survived={survived}",
                self.elapsed_ms(),
                self.best_tc,
                tc_incumbent - tc_after_ladder,
            );

            if cycle_improved {
                since_improve = 0;
            } else {
                since_improve += 1;
            }
        }
        eprintln!("vcycles={vcycle} survived={survived}");
        Ok(())
    }

    /// Run one span-gated cooling level over `tree`. Returns whether the global
    /// best improved. Emits every improvement eagerly and atomically.
    #[allow(clippy::too_many_arguments)]
    fn run_level(
        &mut self,
        tree: &mut ExprTree,
        s_lin: &mut f64,
        best_expr: &mut ExprTree,
        work_tc: &mut f64,
        min_span: usize,
        b_lo: f64,
        b_hi: f64,
        sweeps: u64,
        phase: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let mut improved = false;
        let denom = (sweeps.saturating_sub(1)).max(1) as f64;
        let mut k = 0_u64;
        while k < sweeps && !self.exhausted() {
            let epoch_sweeps = band_epoch_sweeps(self.parent, self.band_epochs, self.epoch_sweeps);
            let epoch_end = (k + epoch_sweeps).min(sweeps);
            let band = heated_band(tree, self.log2_sizes, self.band_bits);
            let waist_before = max_node_tc(tree, self.log2_sizes);
            let tc_before = tree_complexity(tree, self.log2_sizes).0;
            let mut stats = EpochStats::default();
            let epoch_start = k;
            let epoch_denom = (epoch_end - epoch_start).saturating_sub(1).max(1) as f64;

            while k < epoch_end && !self.exhausted() {
                self.sweeps += 1;
                let beta = b_lo + (b_hi - b_lo) * (k as f64 / denom);
                let band_beta = self.band_beta_lo
                    + (self.band_beta_hi - self.band_beta_lo)
                        * ((k - epoch_start) as f64 / epoch_denom);
                let mut path = Vec::new();
                gated_sweep(
                    tree,
                    beta,
                    band_beta,
                    self.parent,
                    &band.paths,
                    &mut path,
                    min_span,
                    self.log2_sizes,
                    &mut self.rng,
                    &mut self.scratch,
                    s_lin,
                    &mut self.accepts,
                    &mut stats,
                );
                k += 1;

                if self.sweeps % RESYNC_SWEEPS == 0 {
                    *s_lin = f64::exp2(tree_complexity(tree, self.log2_sizes).0);
                }

                // Snapshot every transient best immediately in memory. Disk
                // I/O remains rate-limited independently below.
                if *s_lin < f64::exp2(*self.best_tc) - 1e-9 {
                    let captured = self.capture_current(tree, best_expr, work_tc, s_lin);
                    improved |= captured;
                    if captured && !self.parent && self.elapsed_ms() <= EARLY_WRITE_MS {
                        self.write_best()?;
                    }
                }

                // Rate-limited eager writes of the newest in-memory best.
                if self.sweeps % CLOCK_EVERY == 0 {
                    let now = self.elapsed_ms();
                    if self.pending_write && now - self.last_write_ms >= WRITE_EVERY_MS {
                        self.write_best()?;
                    }
                    if now >= self.deadline_ms {
                        break;
                    }
                }
            }

            self.band_epochs += 1;

            if self.diagnostics {
                let waist_after = max_node_tc(tree, self.log2_sizes);
                let tc_after = tree_complexity(tree, self.log2_sizes).0;
                eprintln!(
                    "ATT_EPOCH mode={} phase={} span={} sweep={} epoch_sweeps={} c={:.3} \
                     band_nodes={} internal_nodes={} waist_before={:.9} waist_after={:.9} \
                     tc_before={:.9} tc_after={:.9} in_proposals={} in_accepts={} \
                     in_net_gain={:.9} in_downhill_gain={:.9} out_proposals={} out_accepts={} \
                     out_net_gain={:.9} out_downhill_gain={:.9}",
                    if self.parent { "parent" } else { "band" },
                    phase,
                    min_span,
                    self.sweeps,
                    k - epoch_start,
                    self.band_bits,
                    band.paths.len(),
                    band.internal_nodes,
                    waist_before,
                    waist_after,
                    tc_before,
                    tc_after,
                    stats.inside.proposals,
                    stats.inside.accepts,
                    stats.inside.net_gain,
                    stats.inside.downhill_gain,
                    stats.outside.proposals,
                    stats.outside.accepts,
                    stats.outside.net_gain,
                    stats.outside.downhill_gain,
                );
            }
        }
        // Capture the endpoint; only the program-final flush may bypass the
        // write-rate window.
        if *s_lin < f64::exp2(*self.best_tc) - 1e-9 {
            let captured = self.capture_current(tree, best_expr, work_tc, s_lin);
            improved |= captured;
            if captured && !self.parent && self.elapsed_ms() <= EARLY_WRITE_MS {
                self.write_best()?;
            }
        }
        if self.pending_write && self.elapsed_ms() - self.last_write_ms >= WRITE_EVERY_MS {
            self.write_best()?;
        }
        Ok(improved)
    }

    /// Snapshot an improved current tree immediately, independent of disk I/O.
    fn capture_current(
        &mut self,
        tree: &ExprTree,
        best_expr: &mut ExprTree,
        work_tc: &mut f64,
        s_lin: &mut f64,
    ) -> bool {
        let exact = tree_complexity(tree, self.log2_sizes).0;
        *s_lin = f64::exp2(exact);
        if exact < *self.best_tc - 1e-9 {
            *self.best_tc = exact;
            *work_tc = exact;
            *best_expr = tree.clone();
            *self.best =
                expr_tree_to_nested(tree, self.original_ixs, self.inverse_map, self.openedges, 0);
            self.pending_write = true;
            return true;
        }
        false
    }

    fn write_best(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        write_atomic(
            self.out_path,
            self.best,
            self.subtrees,
            self.original_labels,
        )?;
        self.last_write_ms = self.elapsed_ms();
        self.pending_write = false;
        Ok(())
    }
}

#[derive(Default)]
struct RegionStats {
    proposals: u64,
    accepts: u64,
    /// Sum of exact whole-tree tc changes over accepted moves.
    net_gain: f64,
    /// Gross downhill improvement, excluding accepted uphill moves.
    downhill_gain: f64,
}

#[derive(Default)]
struct EpochStats {
    inside: RegionStats,
    outside: RegionStats,
}

struct HeatedBand {
    paths: HashSet<Vec<bool>>,
    internal_nodes: usize,
}

fn node_tc(tree: &ExprTree, log2_sizes: &[f64]) -> Option<f64> {
    let ExprTree::Node { left, right, info } = tree else {
        return None;
    };
    let mut tc: f64 = info.out_dims.iter().map(|&label| log2_sizes[label]).sum();
    for &label in left.labels() {
        if right.labels().contains(&label) && !info.out_dims.contains(&label) {
            tc += log2_sizes[label];
        }
    }
    Some(tc)
}

fn max_node_tc(tree: &ExprTree, log2_sizes: &[f64]) -> f64 {
    match tree {
        ExprTree::Leaf(_) => f64::NEG_INFINITY,
        ExprTree::Node { left, right, .. } => node_tc(tree, log2_sizes)
            .unwrap_or(f64::NEG_INFINITY)
            .max(max_node_tc(left, log2_sizes))
            .max(max_node_tc(right, log2_sizes)),
    }
}

/// Snapshot all near-waist nodes and every prefix on their path to the root.
/// Paths are recomputed at every epoch boundary and held fixed within an epoch.
fn heated_band(tree: &ExprTree, log2_sizes: &[f64], width_bits: f64) -> HeatedBand {
    fn collect(
        tree: &ExprTree,
        log2_sizes: &[f64],
        threshold: f64,
        path: &mut Vec<bool>,
        paths: &mut HashSet<Vec<bool>>,
        internal_nodes: &mut usize,
    ) {
        let ExprTree::Node { left, right, .. } = tree else {
            return;
        };
        *internal_nodes += 1;
        if node_tc(tree, log2_sizes).is_some_and(|tc| tc >= threshold - 1e-12) {
            for end in 0..=path.len() {
                paths.insert(path[..end].to_vec());
            }
        }
        path.push(false);
        collect(left, log2_sizes, threshold, path, paths, internal_nodes);
        path.pop();
        path.push(true);
        collect(right, log2_sizes, threshold, path, paths, internal_nodes);
        path.pop();
    }

    let threshold = max_node_tc(tree, log2_sizes) - width_bits;
    let mut paths = HashSet::new();
    let mut internal_nodes = 0;
    collect(
        tree,
        log2_sizes,
        threshold,
        &mut Vec::new(),
        &mut paths,
        &mut internal_nodes,
    );
    HeatedBand {
        paths,
        internal_nodes,
    }
}

/// One span-gated SA sweep with unchanged post-order proposal selection. Only
/// the Metropolis beta differs at paths in the epoch's heated band.
#[allow(clippy::too_many_arguments)]
fn gated_sweep(
    tree: &mut ExprTree,
    beta: f64,
    band_beta: f64,
    parent: bool,
    band: &HashSet<Vec<bool>>,
    path: &mut Vec<bool>,
    min_span: usize,
    log2_sizes: &[f64],
    rng: &mut SmallRng,
    scratch: &mut ScratchSpace,
    s_lin: &mut f64,
    accepts: &mut u64,
    stats: &mut EpochStats,
) -> usize {
    match tree {
        ExprTree::Leaf(_) => 1,
        ExprTree::Node { left, right, .. } => {
            path.push(false);
            let ls = gated_sweep(
                left, beta, band_beta, parent, band, path, min_span, log2_sizes, rng, scratch,
                s_lin, accepts, stats,
            );
            path.pop();
            path.push(true);
            let rs = gated_sweep(
                right, beta, band_beta, parent, band, path, min_span, log2_sizes, rng, scratch,
                s_lin, accepts, stats,
            );
            path.pop();
            let span = ls + rs;
            if span >= min_span {
                let rules = Rule::applicable_rules(tree, DecompositionType::Tree);
                if !rules.is_empty() {
                    let rule = rules[rng.random_range(0..rules.len())];
                    if let Some(diff) = scratch.rule_diff(tree, rule, log2_sizes, false) {
                        let dtc = diff.tc1 - diff.tc0;
                        let in_band = band.contains(path);
                        let region = if in_band {
                            &mut stats.inside
                        } else {
                            &mut stats.outside
                        };
                        region.proposals += 1;
                        let effective_beta = proposal_beta(parent, in_band, beta, band_beta);
                        if dtc <= 0.0 || rng.random::<f64>() < (-effective_beta * dtc).exp() {
                            let tc_before = s_lin.log2();
                            *s_lin += f64::exp2(diff.tc1) - f64::exp2(diff.tc0);
                            let realized_gain = tc_before - s_lin.log2();
                            apply_rule_mut(tree, rule, diff.new_labels);
                            *accepts += 1;
                            region.accepts += 1;
                            region.net_gain += realized_gain;
                            region.downhill_gain += realized_gain.max(0.0);
                        }
                    }
                }
            }
            span
        }
    }
}

fn proposal_beta(parent: bool, in_band: bool, beta: f64, band_beta: f64) -> f64 {
    if parent || !in_band {
        beta
    } else {
        beta.min(band_beta)
    }
}

// ==========================================================================
// NestedEinsum <-> ExprTree conversions (label-id space).
// Reimplemented here because the library keeps them private.
// ==========================================================================

/// Convert a binary `NestedEinsum` into an `ExprTree` in label-id space.
/// Mirrors omeco's private `nested_to_expr_tree_inner`.
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

/// Convert an `ExprTree` (label-id space) back into a `NestedEinsum` over the
/// original labels. Mirrors omeco's private `expr_tree_to_nested`; the root's
/// output is forced to `openedges` (issue #13). Only the topology is scored, so
/// this is always contract-valid regardless of intermediate label bookkeeping.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn band_fixture() -> ExprTree {
        let waist = ExprTree::node(
            ExprTree::leaf(vec![0], 0),
            ExprTree::leaf(vec![0], 1),
            vec![],
        );
        let cold = ExprTree::node(
            ExprTree::leaf(vec![1], 2),
            ExprTree::leaf(vec![1], 3),
            vec![],
        );
        ExprTree::node(waist, cold, vec![])
    }

    #[test]
    fn heated_band_contains_near_waist_and_ancestors_only() {
        let band = heated_band(&band_fixture(), &[5.0, 1.0], 1.0);
        assert_eq!(band.internal_nodes, 3);
        assert_eq!(band.paths.len(), 2);
        assert!(band.paths.contains(&vec![]));
        assert!(band.paths.contains(&vec![false]));
        assert!(!band.paths.contains(&vec![true]));
    }

    #[test]
    fn parent_beta_ignores_band_and_heated_beta_never_cools() {
        assert_eq!(proposal_beta(true, true, 9.0, 0.5), 9.0);
        assert_eq!(proposal_beta(false, false, 9.0, 0.5), 9.0);
        assert_eq!(proposal_beta(false, true, 9.0, 0.5), 0.5);
        assert_eq!(proposal_beta(false, true, 0.1, 0.5), 0.1);
    }

    #[test]
    fn parent_sweep_is_independent_of_band_diagnostics() {
        let mut left = band_fixture();
        let mut right = left.clone();
        let mut left_rng = SmallRng::seed_from_u64(17);
        let mut right_rng = SmallRng::seed_from_u64(17);
        let mut left_scratch = ScratchSpace::new(2);
        let mut right_scratch = ScratchSpace::new(2);
        let mut left_s = f64::exp2(tree_complexity(&left, &[5.0, 1.0]).0);
        let mut right_s = left_s;
        let mut left_accepts = 0;
        let mut right_accepts = 0;
        gated_sweep(
            &mut left,
            2.5,
            0.01,
            true,
            &HashSet::from([vec![]]),
            &mut Vec::new(),
            2,
            &[5.0, 1.0],
            &mut left_rng,
            &mut left_scratch,
            &mut left_s,
            &mut left_accepts,
            &mut EpochStats::default(),
        );
        gated_sweep(
            &mut right,
            2.5,
            100.0,
            true,
            &HashSet::new(),
            &mut Vec::new(),
            2,
            &[5.0, 1.0],
            &mut right_rng,
            &mut right_scratch,
            &mut right_s,
            &mut right_accepts,
            &mut EpochStats::default(),
        );
        assert_eq!(format!("{left:?}"), format!("{right:?}"));
        assert_eq!(left_s, right_s);
        assert_eq!(left_accepts, right_accepts);
    }

    #[test]
    fn budget_must_be_finite_and_positive() {
        assert_eq!(parse_budget_ms("12.5").unwrap(), 12.5);
        for invalid in ["0", "-1", "NaN", "inf", "-inf"] {
            assert!(parse_budget_ms(invalid).is_err(), "accepted {invalid}");
        }
    }

    #[test]
    fn only_first_two_mechanism_epochs_are_halved() {
        assert_eq!(band_epoch_sweeps(false, 0, 15), 8);
        assert_eq!(band_epoch_sweeps(false, 1, 15), 8);
        assert_eq!(band_epoch_sweeps(false, 2, 15), 15);
        assert_eq!(band_epoch_sweeps(true, 0, 15), 15);
        assert_eq!(band_epoch_sweeps(true, 1, 15), 15);
    }
}
