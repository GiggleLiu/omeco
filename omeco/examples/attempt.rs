//! Attempt entry point for the autoresearch validator (attempt-059).
//!
//! Contract: `attempt <graph.json> <budget_ms> <out.json>` — read an einsum
//! graph, search for a contraction order within the wall-clock budget, and
//! keep the best tree found (by pure time complexity, `tc`) written to
//! `out.json` in omeco `writejson` format. Every improvement is written
//! EAGERLY and ATOMICALLY (tmp file + rename) the instant it is found.
//!
//! # Continuous beta(span, t) freeze-out ladder
//!
//! This is attempt-052's simplify -> greedy portfolio -> warm kick -> cold
//! ladder ratchet pipeline with one atomic mechanism change. At each cold
//! ladder level, the hard `span >= min_span` gate is replaced by a front that
//! descends linearly in log2(span) during the same number of sweeps. Nodes ahead
//! of the front have beta=+infinity (only non-regressing moves survive); nodes
//! at the front have beta=2.5; beta rises linearly to 14 over one octave behind
//! the front. `ATT_PARENT=1` restores the hard gate exactly for matched controls.
//!
//!   1. SIMPLIFY + SEED. Deterministically collapse safe local structure, then
//!      run the same deterministic + Boltzmann-randomized greedy portfolio.
//!   2. BASIN-HOP. Each cycle clones the incumbent, runs a LONG warm anneal
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
//! Single-threaded: no Rayon trials are launched. No per-instance constants;
//! every search knob is a constant function of reduced tensor count `n`, except
//! the inherited `S_top=ceil(n/30)`, O(log n) ladder, and stagnation threshold.
//! Behaviour is identical under instance relabeling.

use std::collections::HashMap;
use std::time::Instant;

use omeco::expr_tree::{apply_rule_mut, DecompositionType, ExprTree, Rule, ScratchSpace};
use omeco::json::writejson;
use omeco::{
    contraction_complexity, optimize_code, simplify, splice, EinCode, GreedyMethod, NestedEinsum,
};
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

/// Sweep interval for the matched-work tc(t) diagnostic.
const DIAG_EVERY_SWEEPS: u64 = 40;

/// Minimum wall-clock gap between atomic disk writes. The validator polls
/// `out.json` on its own 0.2 s clock, so writing more often is pure waste; on
/// large-boundary trees an unthrottled write-per-improvement during descent
/// costs seconds of serialize/rename I/O. In-memory best tracking stays exact.
const WRITE_EVERY_MS: f64 = 150.0;

#[derive(Debug, Deserialize)]
struct GraphData {
    ixs: Vec<Vec<usize>>,
    iy: Vec<usize>,
    sizes: HashMap<String, usize>,
}

#[derive(Clone, Copy)]
enum SweepSchedule {
    Hard {
        beta: f64,
        min_span: usize,
    },
    Front {
        front_log2: f64,
        width_log2: f64,
        beta_warm: f64,
        beta_cold: f64,
    },
}

#[derive(Default)]
struct SpanBandStats {
    attempts: u64,
    accepts: u64,
    improving_accepts: u64,
    improving_ahead: u64,
    improving_front: u64,
    improving_behind: u64,
    dtc_gain: f64,
}

struct Checkpoint {
    sweeps: u64,
    tc: f64,
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

/// Atomically write `tree` to `out_path` (tmp file + rename) so the polling
/// validator never observes a partially-written file.
fn write_atomic(
    out_path: &str,
    tree: &NestedEinsum<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let tmp = format!("{out_path}.tmp");
    writejson(&tmp, tree)?;
    std::fs::rename(&tmp, out_path)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: attempt <graph.json> <budget_ms> <out.json>");
        std::process::exit(2);
    }
    let start = Instant::now();
    let budget_ms: f64 = args[2].parse()?;
    if !budget_ms.is_finite() || budget_ms < 0.0 {
        return Err("budget_ms must be finite and non-negative".into());
    }
    let out_path = args[3].clone();

    let graph: GraphData = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let sizes: HashMap<usize, usize> = graph
        .sizes
        .iter()
        .map(|(k, v)| Ok::<_, std::num::ParseIntError>((k.parse::<usize>()?, *v)))
        .collect::<Result<_, _>>()?;
    let original_code: EinCode<usize> = EinCode::new(graph.ixs.clone(), graph.iy.clone());
    let n_original = original_code.num_tensors();

    let elapsed_ms = || start.elapsed().as_secs_f64() * 1e3;
    let deadline_ms = budget_ms * 0.97;
    let tc_of =
        |tree: &NestedEinsum<usize>| contraction_complexity(tree, &sizes, &original_code.ixs).tc;

    if n_original == 0 {
        return Err("empty einsum".into());
    }

    // The shipped baseline starts with the deterministic structural front-end.
    // Search only the reduced network; splice every scored/emitted tree back to
    // original tensor ids so tc and the output contract remain end-to-end exact.
    let simplified = simplify(&original_code, &sizes);
    let code = &simplified.code;
    let n = code.num_tensors();

    // ---- Integer label space (shared by ExprTree, tc, and I/O). --------------
    // `labels[id]` = original label; `log2_sizes[id]` = log2 of that label's
    // dimension. This mirrors omeco's internal convention.
    let labels: Vec<usize> = code.unique_labels();
    let log2_sizes: Vec<f64> = labels
        .iter()
        .map(|&l| (*sizes.get(&l).unwrap_or(&1) as f64).log2())
        .collect();

    // ---- Trivial reduced cases: nothing to anneal. ---------------------------
    if n <= 2 {
        let reduced = optimize_code(code, &sizes, &GreedyMethod::default())
            .ok_or("greedy optimizer returned no tree")?;
        let best = splice(&reduced, &simplified.subtrees);
        write_atomic(&out_path, &best)?;
        return Ok(());
    }

    // ---- Seed portfolio: deterministic + Boltzmann-randomized greedy. --------
    // Deterministic greedy first, written immediately so a valid result always
    // exists regardless of what follows.
    let mut best_reduced = optimize_code(code, &sizes, &GreedyMethod::default())
        .ok_or("greedy optimizer returned no tree")?;
    let mut best = splice(&best_reduced, &simplified.subtrees);
    let mut best_tc = tc_of(&best);
    write_atomic(&out_path, &best)?;

    let seed_deadline = env_f64("ATT_SEED_MS", (budget_ms * 0.04).min(500.0)).max(0.0);
    let alphas = [0.0f64, 0.25, 0.5, 0.75, 1.0];
    let temps = [0.03f64, 0.1, 0.3];
    let mut combo = 0usize;
    while elapsed_ms() < seed_deadline {
        let alpha = alphas[combo % alphas.len()];
        let temp = temps[(combo / alphas.len()) % temps.len()];
        combo += 1;
        if let Some(tree) = optimize_code(code, &sizes, &GreedyMethod::new(alpha, temp)) {
            let full = splice(&tree, &simplified.subtrees);
            let tc = tc_of(&full);
            if tc < best_tc - 1e-9 {
                best_reduced = tree;
                best = full;
                best_tc = tc;
                write_atomic(&out_path, &best)?;
            }
        }
    }
    let tc_greedy = best_tc;
    eprintln!(
        "t={:.0}ms seed_greedy tc={tc_greedy:.4} (n={n}/{n_original}, {} greedy trials)",
        elapsed_ms(),
        combo + 1
    );

    // ---- Seed the basin-hopper from the greedy portfolio best. ---------------
    // The coarsener of attempt-034 is removed: it seeded strictly worse than
    // greedy on every scale instance and 034 always fell back here anyway.
    let seed_expr = match nested_to_expr_tree(&best_reduced, &labels) {
        Some(t) => t,
        None => {
            // n >= 3 greedy always yields internal nodes; this is defensive.
            write_atomic(&out_path, &best)?;
            return Ok(());
        }
    };

    // ---- Scale-structured basin-hopping. -------------------------------------
    let initial_tc = best_tc;
    let mut ann = Annealer {
        reduced_ixs: &code.ixs,
        original_ixs: &graph.ixs,
        subtrees: &simplified.subtrees,
        inverse_map: &labels,
        openedges: &code.iy,
        log2_sizes: &log2_sizes,
        size_dict: &sizes,
        code,
        scratch: ScratchSpace::new(labels.len()),
        rng: SmallRng::seed_from_u64(0x0000_0052_c0ff_ee00),
        out_path: &out_path,
        start: &start,
        deadline_ms,
        best: &mut best,
        best_tc: &mut best_tc,
        sweeps: 0,
        accepts: 0,
        last_write_ms: elapsed_ms(),
        parent: std::env::var("ATT_PARENT").as_deref() == Ok("1"),
        max_sweeps: std::env::var("ATT_MAX_SWEEPS")
            .ok()
            .and_then(|value| value.parse().ok()),
        span_stats: Vec::new(),
        checkpoints: vec![Checkpoint {
            sweeps: 0,
            tc: initial_tc,
        }],
    };
    ann.run(seed_expr, n)?;
    let (final_tc, final_sweeps, final_accepts) = (*ann.best_tc, ann.sweeps, ann.accepts);
    ann.emit_diagnostics(n, n_original);

    eprintln!(
        "t={:.0}ms tc_final={final_tc:.4} sweeps={final_sweeps} accepts={final_accepts}",
        elapsed_ms(),
    );
    // Forced final flush, even when the last improvement was already written.
    write_atomic(&out_path, &best)?;
    Ok(())
}

/// Shared state for the basin-hopping annealer.
struct Annealer<'a> {
    reduced_ixs: &'a [Vec<usize>],
    original_ixs: &'a [Vec<usize>],
    subtrees: &'a [NestedEinsum<usize>],
    inverse_map: &'a [usize],
    openedges: &'a [usize],
    log2_sizes: &'a [f64],
    size_dict: &'a HashMap<usize, usize>,
    code: &'a EinCode<usize>,
    scratch: ScratchSpace,
    rng: SmallRng,
    out_path: &'a str,
    start: &'a Instant,
    deadline_ms: f64,
    best: &'a mut NestedEinsum<usize>,
    best_tc: &'a mut f64,
    sweeps: u64,
    accepts: u64,
    /// Wall-clock ms of the last atomic disk write (writes are rate-limited to
    /// the validator's poll resolution so rapid descent does not trigger an
    /// O(n) serialize-and-rename storm on large-boundary trees).
    last_write_ms: f64,
    parent: bool,
    max_sweeps: Option<u64>,
    span_stats: Vec<SpanBandStats>,
    checkpoints: Vec<Checkpoint>,
}

impl Annealer<'_> {
    fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1e3
    }

    fn at_limit(&self) -> bool {
        self.elapsed_ms() >= self.deadline_ms
            || self
                .max_sweeps
                .is_some_and(|maximum| self.sweeps >= maximum)
    }

    fn exact_tc(&self, tree: &ExprTree) -> f64 {
        let reduced =
            expr_tree_to_nested(tree, self.reduced_ixs, self.inverse_map, self.openedges, 0);
        let full = splice(&reduced, self.subtrees);
        contraction_complexity(&full, self.size_dict, self.original_ixs).tc
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
        let mut work_tc = *self.best_tc;

        let mut vcycle: u64 = 0;
        let mut since_improve: u64 = 0;
        let mut survived: u64 = 0; // #cycles whose ladder recovered past incumbent

        let front_width_log2 = env_f64("ATT_FRONT_WIDTH", 1.0).max(f64::EPSILON);

        eprintln!(
            "mode={} s_top={s_top} levels={} front_width_log2={front_width_log2:.3}",
            if self.parent { "parent" } else { "continuous" },
            span_levels.len()
        );

        loop {
            if self.at_limit() {
                break;
            }
            vcycle += 1;

            if since_improve >= stag_threshold {
                // Structurally different basin: a fresh randomized greedy, then
                // a FLAT warm anneal (built-in flat-SA control).
                let alpha = [0.0f64, 0.25, 0.5, 0.75, 1.0][self.rng.random_range(0..5)];
                let temp = 0.05 + self.rng.random::<f64>() * 0.3;
                let (mut tree, mut s_lin) =
                    match optimize_code(self.code, self.size_dict, &GreedyMethod::new(alpha, temp))
                        .and_then(|g| nested_to_expr_tree(&g, self.inverse_map))
                    {
                        Some(t) => {
                            let sl = f64::exp2(self.exact_tc(&t));
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
                    None,
                    false,
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
                None,
                false,
            )?;
            let tc_after_kick = self.exact_tc(&tree);

            // Cold refinement ladder.
            let mut cycle_improved = false;
            for (level, &span) in span_levels.iter().enumerate() {
                let next_span = span_levels.get(level + 1).copied().unwrap_or(1);
                let improved = self.run_level(
                    &mut tree,
                    &mut s_lin,
                    &mut best_expr,
                    &mut work_tc,
                    span,
                    b_lo_cold,
                    b_hi,
                    cold_sweeps,
                    (!self.parent).then_some((next_span, front_width_log2)),
                    true,
                )?;
                cycle_improved |= improved;
                if self.at_limit() {
                    break;
                }
            }
            let tc_after_ladder = self.exact_tc(&tree);
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
        front: Option<(usize, f64)>,
        record_span_stats: bool,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let mut improved = false;
        let denom = (sweeps.saturating_sub(1)).max(1) as f64;
        for k in 0..sweeps {
            if self
                .max_sweeps
                .is_some_and(|maximum| self.sweeps >= maximum)
            {
                break;
            }
            self.sweeps += 1;
            let beta = b_lo + (b_hi - b_lo) * (k as f64 / denom);
            let schedule = if let Some((next_span, width_log2)) = front {
                let progress = k as f64 / denom;
                let start = (min_span as f64).log2();
                let end = (next_span as f64).log2();
                SweepSchedule::Front {
                    front_log2: start + (end - start) * progress,
                    width_log2,
                    beta_warm: b_lo,
                    beta_cold: b_hi,
                }
            } else {
                SweepSchedule::Hard { beta, min_span }
            };
            gated_sweep(
                tree,
                schedule,
                self.log2_sizes,
                &mut self.rng,
                &mut self.scratch,
                s_lin,
                &mut self.accepts,
                &mut self.span_stats,
                record_span_stats,
            );

            if self.sweeps % RESYNC_SWEEPS == 0 {
                *s_lin = f64::exp2(self.exact_tc(tree));
            }

            // Rate-limited flush: at most one per WRITE_EVERY_MS. The current
            // tree's exact tc is checked against the last-written best; snapshot
            // (clone) and disk write happen only here, never per improving
            // sweep, so rapid descent from a poor seed no longer thrashes the
            // filesystem or the allocator on large-boundary trees.
            if self.sweeps % CLOCK_EVERY == 0 {
                let now = self.elapsed_ms();
                if *s_lin < f64::exp2(*self.best_tc) - 1e-9
                    && now - self.last_write_ms >= WRITE_EVERY_MS
                {
                    improved |= self.flush_current(tree, best_expr, work_tc, s_lin)?;
                }
                if self.max_sweeps.is_some() && self.sweeps % DIAG_EVERY_SWEEPS == 0 {
                    self.checkpoints.push(Checkpoint {
                        sweeps: self.sweeps,
                        tc: *self.best_tc,
                    });
                }
                if self.at_limit() {
                    break;
                }
            }
        }
        // Level-end flush captures the cold-end minimum of this level.
        if *s_lin < f64::exp2(*self.best_tc) - 1e-9 {
            improved |= self.flush_current(tree, best_expr, work_tc, s_lin)?;
        }
        Ok(improved)
    }

    /// If the current `tree` beats the last-written best, write it atomically
    /// and snapshot it as the restart incumbent (`best_expr`/`work_tc`). Returns
    /// whether the global best improved.
    fn flush_current(
        &mut self,
        tree: &ExprTree,
        best_expr: &mut ExprTree,
        work_tc: &mut f64,
        s_lin: &mut f64,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let reduced =
            expr_tree_to_nested(tree, self.reduced_ixs, self.inverse_map, self.openedges, 0);
        let full = splice(&reduced, self.subtrees);
        let exact = contraction_complexity(&full, self.size_dict, self.original_ixs).tc;
        *s_lin = f64::exp2(exact);
        self.last_write_ms = self.elapsed_ms();
        if exact < *self.best_tc - 1e-9 {
            *self.best_tc = exact;
            *work_tc = exact;
            *best_expr = tree.clone();
            *self.best = full;
            write_atomic(self.out_path, self.best)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn emit_diagnostics(&mut self, n: usize, n_original: usize) {
        if self
            .checkpoints
            .last()
            .map_or(true, |point| point.sweeps != self.sweeps)
        {
            self.checkpoints.push(Checkpoint {
                sweeps: self.sweeps,
                tc: *self.best_tc,
            });
        }
        let bands: Vec<_> = self
            .span_stats
            .iter()
            .enumerate()
            .filter(|(_, band)| band.attempts > 0)
            .map(|(log2_lo, band)| {
                serde_json::json!({
                    "log2_lo": log2_lo,
                    "attempts": band.attempts,
                    "accepts": band.accepts,
                    "improving_accepts": band.improving_accepts,
                    "improving_ahead": band.improving_ahead,
                    "improving_front": band.improving_front,
                    "improving_behind": band.improving_behind,
                    "dtc_gain": band.dtc_gain,
                })
            })
            .collect();
        let checkpoints: Vec<_> = self
            .checkpoints
            .iter()
            .map(|point| serde_json::json!({"sweeps": point.sweeps, "tc": point.tc}))
            .collect();
        let payload = serde_json::json!({
            "mode": if self.parent { "parent" } else { "continuous" },
            "n": n,
            "n_original": n_original,
            "sweeps": self.sweeps,
            "accepts": self.accepts,
            "best_tc": *self.best_tc,
            "bands": bands,
            "checkpoints": checkpoints,
        });
        eprintln!("ATT_DIAG {payload}");
    }
}

/// One scale-temperature SA sweep (post-order). The parent schedule uses its
/// exact hard gate. The attempt-059 schedule visits every internal node and
/// assigns beta from its position relative to the continuous log-span front.
/// Energy is pure `dtc`; `s_lin` is updated by exactly the two changed nodes.
#[allow(clippy::too_many_arguments)]
fn gated_sweep(
    tree: &mut ExprTree,
    schedule: SweepSchedule,
    log2_sizes: &[f64],
    rng: &mut SmallRng,
    scratch: &mut ScratchSpace,
    s_lin: &mut f64,
    accepts: &mut u64,
    stats: &mut Vec<SpanBandStats>,
    record_stats: bool,
) -> usize {
    match tree {
        ExprTree::Leaf(_) => 1,
        ExprTree::Node { left, right, .. } => {
            let ls = gated_sweep(
                left,
                schedule,
                log2_sizes,
                rng,
                scratch,
                s_lin,
                accepts,
                stats,
                record_stats,
            );
            let rs = gated_sweep(
                right,
                schedule,
                log2_sizes,
                rng,
                scratch,
                s_lin,
                accepts,
                stats,
                record_stats,
            );
            let span = ls + rs;
            let node_log2 = (span as f64).log2();
            let selected = match schedule {
                SweepSchedule::Hard { beta, min_span } => {
                    (span >= min_span).then_some((beta, 2_usize))
                }
                SweepSchedule::Front {
                    front_log2,
                    width_log2,
                    beta_warm,
                    beta_cold,
                } => {
                    let distance = node_log2 - front_log2;
                    if distance < 0.0 {
                        Some((f64::INFINITY, 0))
                    } else if distance < width_log2 {
                        let beta = beta_warm + (beta_cold - beta_warm) * distance / width_log2;
                        Some((beta, 1))
                    } else {
                        Some((beta_cold, 2))
                    }
                }
            };
            if let Some((beta, region)) = selected {
                let rules = Rule::applicable_rules(tree, DecompositionType::Tree);
                if !rules.is_empty() {
                    let band = if record_stats {
                        let band_index = span.ilog2() as usize;
                        if stats.len() <= band_index {
                            stats.resize_with(band_index + 1, SpanBandStats::default);
                        }
                        let band = &mut stats[band_index];
                        band.attempts += 1;
                        Some(band)
                    } else {
                        None
                    };
                    let rule = rules[rng.random_range(0..rules.len())];
                    if let Some(diff) = scratch.rule_diff(tree, rule, log2_sizes, false) {
                        let dtc = diff.tc1 - diff.tc0;
                        if dtc <= 0.0 || rng.random::<f64>() < (-beta * dtc).exp() {
                            *s_lin += f64::exp2(diff.tc1) - f64::exp2(diff.tc0);
                            apply_rule_mut(tree, rule, diff.new_labels);
                            *accepts += 1;
                            if let Some(band) = band {
                                band.accepts += 1;
                                if dtc < -1e-12 {
                                    band.improving_accepts += 1;
                                    band.dtc_gain += -dtc;
                                    match region {
                                        0 => band.improving_ahead += 1,
                                        1 => band.improving_front += 1,
                                        _ => band.improving_behind += 1,
                                    }
                                }
                            }
                        }
                    }
                }
            }
            span
        }
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
