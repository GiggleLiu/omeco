//! Attempt entry point for the autoresearch validator (attempt-027).
//!
//! Contract: `attempt <graph.json> <budget_ms> <out.json>` — read an einsum
//! graph, search for a contraction order within the wall-clock budget, and
//! write the best tree in `writejson` format before the deadline (anytime:
//! write early, improve in place).
//!
//! Strategy (this attempt's novelty): PROFILE-AWARE simulated annealing. Pure
//! tc — no sc cap. The scored tc of a tree is `log2 Σ_v 2^{cost_v}` over the
//! per-contraction costs. attempt-022 showed the frontier is width-optimal
//! (max cost is already minimal) so the residual tc is the PROFILE: how many
//! contractions sit near the maximum. Plain-tc annealing (γ=1) has almost no
//! gradient on a single near-max node (shaving one changes tc by ~1/count).
//! This attempt keeps the proven LOCAL-dtc gradient as the workhorse (it drives
//! the tree to the width-optimal frontier, tc≈40 on reg3_250) and ADDS a
//! profile-aware "peak-pressure" term that sharpens the search on the near-max
//! nodes that dominate tc:
//!
//!     E_move = Δtc_local + κ · Δpeak,   peak(c) = 2^{α·(c − c_max)},
//!
//! where c_max is the current maximum node cost. `peak` is ≈0 for bulk nodes
//! (no extra pressure — pure local dtc there) and ≈1 with a steep gradient for
//! near-max nodes, so κ concentrates annealing pressure exactly on the frontier
//! overhead. α is the sharpening (the (γ−1) analogue). Moves are also TARGETED
//! at the top-cost decile (proposal mass ≈0.7 there). Both mechanisms engage
//! only in the cold tail of each anneal cycle, after the plain local-dtc
//! gradient has done the bulk descent.
//!
//! Pipeline: (1) an anytime omeco-TreeSA warm start (~35% of budget, sc_target
//! =∞) provides a strong, ROBUST floor and removes catastrophic failed-descent
//! runs; (2) the population of contraction trees is seeded from that floor and
//! run through a cyclic-cool profile-aware SA (NCYCLES independent descents).
//! The global best is ALWAYS tracked/emitted by TRUE tc (`tree_complexity`), so
//! the worst case is exactly the TreeSA floor. ATTEMPT_MODE selects the
//! mechanism for clean attribution: "control" (κ=0, uniform proposal),
//! "target" (targeting only, κ=0), "profile" (default: targeting + κ). Tunable
//! via ATTEMPT_{KAPPA,ALPHA,BETAHI,TREESA}.
//!
//! Reuses omeco's public `expr_tree` primitives (four rotation rules,
//! `rule_diff`, `apply_rule_mut`, `tree_complexity`, `tcscrw`). Single thread.

use std::collections::HashMap;
use std::time::Instant;

use omeco::expr_tree::{
    apply_rule_mut, tcscrw, tree_complexity, DecompositionType, ExprTree, Rule, ScratchSpace,
};
use omeco::json::writejson;
use omeco::{contraction_complexity, optimize_code, EinCode, GreedyMethod, NestedEinsum, TreeSA};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GraphData {
    #[serde(default)]
    name: String,
    ixs: Vec<Vec<usize>>,
    iy: Vec<usize>,
    sizes: HashMap<String, usize>,
}

/// Population size.
const POP: usize = 4;
/// Full-rotation wall-clock target (ms). Round slice ≈ this / POP.
const ROTATION_MS: f64 = 1200.0;
/// A replica is re-seeded from the best when it lags by more than this in tc.
const LAG_THRESHOLD: f64 = 0.4;
/// Number of anneal cycles (each a full β cool-down) across the budget. Multiple
/// cool-downs = multiple independent descent attempts = robustness (a single
/// monotonic cool gives one basin attempt and high variance on 560-node
/// instances; cf. sibling 017).
const NCYCLES: f64 = 4.0;
/// Within each cycle, the profile mechanisms (targeting + κ peak-pressure) engage
/// only once the cycle is this cold (cycle progress ≥ this): the plain local-dtc
/// gradient does the bulk descent first, then profile pressure refines the top.
const PROFILE_START: f64 = 0.45;
/// Legacy first-window control fraction (used only for the in-binary attribution
/// snapshot; the clean control is ATTEMPT_MODE=control on a full run).
const CONTROL_FRAC: f64 = 0.40;
/// Peak-pressure sharpening: peak(c) = 2^{ALPHA·(c − c_max)}. Larger ⇒ pressure
/// confined tighter to the very top of the profile.
const ALPHA: f64 = 2.0;
/// Peak-pressure weight at the end of the profile ramp (a gentle BIAS: large κ
/// accepts tc-increasing profile-flattening moves and steers out of the good
/// basin — see LOG).
const KAPPA_MAX: f64 = 0.3;
/// Monotonic geometric β bounds across the whole budget (slow single cool, as in
/// the strongest known sibling; population + clone-best supplies diversity).
const BETA_LO: f64 = 0.02;
const BETA_HI: f64 = 30.0;
/// Targeted-proposal probability for a NON-top-decile node (top-decile = 1.0).
/// Chosen so top-decile nodes carry ≈0.7 of the proposal mass:
/// 0.1·1 / (0.1·1 + 0.9·q) = 0.7  ⇒  q = 1/21.
const OFF_DECILE_Q: f64 = 1.0 / 21.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: attempt <graph.json> <budget_ms> <out.json>");
        std::process::exit(2);
    }
    let start = Instant::now();
    let budget_ms: f64 = args[2].parse()?;
    let out_path = &args[3];

    // Leave headroom for the final write; validator kills at budget * 1.05.
    let deadline_ms = budget_ms * 0.92;
    let over = |st: &Instant| st.elapsed().as_secs_f64() * 1e3 >= deadline_ms;

    let graph: GraphData = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let sizes: HashMap<usize, usize> = graph
        .sizes
        .iter()
        .map(|(k, v)| Ok::<_, std::num::ParseIntError>((k.parse::<usize>()?, *v)))
        .collect::<Result<_, _>>()?;
    let code: EinCode<usize> = EinCode::new(graph.ixs.clone(), graph.iy.clone());
    let n = code.ixs.len();

    // ---- 1. Deterministic greedy seed, written immediately as a fallback. ---
    let greedy = optimize_code(&code, &sizes, &GreedyMethod::default())
        .ok_or("greedy optimizer returned no tree")?;
    writejson(out_path, &greedy)?;

    // Trivial cases: nothing to optimize.
    if n <= 2 {
        return Ok(());
    }

    // ---- 2. Compact label map and log2 sizes. -------------------------------
    let labels = code.unique_labels();
    let label_map: HashMap<usize, usize> = labels
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, l)| (l, i))
        .collect();
    let inverse_map: Vec<usize> = labels.clone();
    let log2_sizes: Vec<f64> = labels.iter().map(|l| (sizes[l] as f64).log2()).collect();
    let nedge = labels.len();

    let Some(greedy_tree) = nested_to_expr(&greedy, &label_map) else {
        return Ok(());
    };

    // Seed from n plus wall-clock entropy so repeated local runs are genuine
    // independent samples (the validator relabels per run anyway).
    let entropy = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut rng =
        SmallRng::seed_from_u64(0xC027u64 ^ (n as u64).wrapping_mul(2654435761) ^ entropy);
    let mut scratch = ScratchSpace::new(nedge);
    let debug = std::env::var("ATTEMPT_DEBUG").is_ok();
    // Mechanism selector for clean attribution (default = profile, the scored
    // strategy). "control": pure local-dtc + uniform proposal for the whole
    // budget. "target": warmup then targeting only (κ=0). "profile": warmup then
    // targeting + peak-pressure κ ramp. ATTEMPT_CONTROL=1 aliases "control".
    let mode = std::env::var("ATTEMPT_MODE").unwrap_or_default();
    let mode = if std::env::var("ATTEMPT_CONTROL").is_ok() {
        "control"
    } else if mode.is_empty() {
        "profile"
    } else {
        mode.as_str()
    };
    let use_kappa = mode == "profile";
    let use_target = mode == "profile" || mode == "target";
    // Tunable knobs (env overrides for local sweeps; defaults below).
    let envf = |k: &str, d: f64| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(d)
    };
    let kappa_max = envf("ATTEMPT_KAPPA", KAPPA_MAX);
    let alpha = envf("ATTEMPT_ALPHA", ALPHA);
    let control_frac = envf("ATTEMPT_WARMUP", CONTROL_FRAC);
    let beta_hi = envf("ATTEMPT_BETAHI", BETA_HI);
    // Fraction of budget spent on the omeco-TreeSA warm start (a strong, robust
    // floor that removes catastrophic failed-descent runs; the profile SA then
    // refines it, and the true-tc best-gate guarantees worst-case = TreeSA).
    let warmstart_frac = envf("ATTEMPT_TREESA", 0.35);
    let _ = graph.name;

    // ---- 3. omeco-TreeSA warm start (anytime doubling niters), emitted. -----
    // Pure tc: sc_target = +∞. Keeps the best tree by true tc; converts it to an
    // ExprTree seed for the population.
    let cc_of = |t: &NestedEinsum<usize>| contraction_complexity(t, &sizes, &code.ixs);
    let ws_deadline_ms = deadline_ms * warmstart_frac;
    let mut ts_best = greedy.clone();
    let mut ts_best_tc = cc_of(&greedy).tc;
    let mut niters = 10usize;
    while (start.elapsed().as_secs_f64() * 1e3) < ws_deadline_ms {
        let round = Instant::now();
        let treesa = TreeSA::default()
            .with_ntrials(1)
            .with_niters(niters)
            .with_sc_target(f64::INFINITY);
        let Some(tree) = optimize_code(&code, &sizes, &treesa) else {
            break;
        };
        let tc = cc_of(&tree).tc;
        if tc < ts_best_tc - 1e-9 {
            ts_best_tc = tc;
            ts_best = tree;
            writejson(out_path, &ts_best)?;
        }
        let round_ms = round.elapsed().as_secs_f64() * 1e3;
        let remaining = ws_deadline_ms - start.elapsed().as_secs_f64() * 1e3;
        if round_ms > remaining {
            break;
        }
        niters = (niters * 2).min(400);
    }
    let ts_best_tree = nested_to_expr(&ts_best, &label_map).unwrap_or_else(|| greedy_tree.clone());
    if debug {
        eprintln!(
            "[dbg] treesa warm start: tc={:.4} ({:.1}s)",
            ts_best_tc,
            start.elapsed().as_secs_f64()
        );
    }

    // ---- 4. Seed the population from the TreeSA floor + diverse explorers. ---
    // Replica 0 = TreeSA best (exploit). Replicas 1..P = perturbed clones of it
    // (refine nearby) plus stochastic-greedy diversity.
    let mut replicas: Vec<Replica> = Vec::with_capacity(POP);
    replicas.push(Replica::new(ts_best_tree.clone(), &log2_sizes));
    {
        let mut clone = ts_best_tree.clone();
        for _ in 0..rng.random_range(3..=7) {
            random_mutate(&mut clone, &log2_sizes, &mut scratch, &mut rng);
        }
        replicas.push(Replica::new(clone, &log2_sizes));
    }
    for &t in &[2.0_f64, 4.0] {
        let seed_tree = optimize_code(&code, &sizes, &GreedyMethod::stochastic(t))
            .and_then(|nested| nested_to_expr(&nested, &label_map))
            .unwrap_or_else(|| ts_best_tree.clone());
        replicas.push(Replica::new(seed_tree, &log2_sizes));
    }

    // ---- 5. Global incumbent (on disk), tracked by TRUE tc. -----------------
    let mut best = ts_best_tree.clone();
    let mut best_tc = tc_of(&best, &log2_sizes);
    let mut control_best_tc = f64::INFINITY; // best at end of control phase

    macro_rules! consider {
        ($r:expr) => {{
            let r = &$r;
            if r.best_tc < best_tc - 1e-9 {
                best = r.best_tree.clone();
                best_tc = r.best_tc;
                let nested = expr_to_nested(&best, &code.ixs, &inverse_map, &code.iy, 0);
                writejson(out_path, &nested)?;
            }
        }};
    }
    for r in replicas.iter() {
        consider!(r);
    }

    // ---- 5. Size the round slice to the budget. -----------------------------
    let mut costs: Vec<f64> = Vec::with_capacity(n);
    let per_sweep_ms = {
        let mut probe = greedy_tree.clone();
        let mut probe_scratch = ScratchSpace::new(nedge);
        let t0 = Instant::now();
        for _ in 0..4 {
            let (cmax, theta) = profile(&probe, &log2_sizes, &mut costs);
            sweep(
                &mut probe,
                3.0,
                0.0,
                cmax,
                alpha,
                false,
                theta,
                &log2_sizes,
                &mut probe_scratch,
                &mut rng,
            );
        }
        (t0.elapsed().as_secs_f64() * 1e3 / 4.0).max(1e-4)
    };
    let slice_ms = ROTATION_MS / POP as f64;
    let sweeps_per_slice = ((slice_ms / per_sweep_ms) as usize).max(1);

    if debug {
        eprintln!(
            "[dbg] n={} nedge={} greedy_tc={:.3} per_sweep_ms={:.4} sweeps/slice={}",
            n, nedge, best_tc, per_sweep_ms, sweeps_per_slice
        );
    }

    // ---- 6. Monotonic β cool + profile-arm κ ramp, indexed by elapsed. ------
    // f = elapsed / deadline ∈ [0,1]. β geom-cools BETA_LO→BETA_HI over the
    // whole budget (population + clone-best supply diversity). The first
    // CONTROL_FRAC is the plain-tc control arm (κ=0, uniform proposal); after
    // that κ ramps 0→KAPPA_MAX and moves target the top-cost decile.
    // Cyclic β cool-down (NCYCLES independent descent attempts). The profile
    // mechanisms engage only in the cold tail of each cycle (pc ≥ PROFILE_START),
    // after the plain local-dtc gradient has done the bulk descent. The very
    // first cycle stays plain the whole way (a warm control-arm baseline whose
    // best is snapshotted for attribution); ATTEMPT_MODE=control keeps κ=0 and
    // targeting off for the whole budget.
    let schedule = |st: &Instant| -> (f64, f64, bool) {
        let f = (st.elapsed().as_secs_f64() * 1e3 / deadline_ms).clamp(0.0, 1.0);
        let pc = (f * NCYCLES).fract();
        let beta = BETA_LO * (beta_hi / BETA_LO).powf(pc);
        let first_cycle = f < 1.0 / NCYCLES;
        if first_cycle || pc < PROFILE_START {
            (beta, 0.0, false) // plain local-dtc bulk descent
        } else {
            let p = (pc - PROFILE_START) / (1.0 - PROFILE_START);
            let kappa = if use_kappa { kappa_max * p } else { 0.0 };
            (beta, kappa, use_target)
        }
    };
    let _ = control_frac;

    let mut prev_control = true;

    // ---- 7. Round-robin profile-aware anneal. -------------------------------
    while !over(&start) {
        let (beta, kappa, targeted) = schedule(&start);

        // Snapshot the control-arm best once, at the control→profile boundary.
        if prev_control && targeted {
            control_best_tc = best_tc;
            prev_control = false;
            if debug {
                eprintln!("[dbg] control phase end: best_tc={:.4}", control_best_tc);
            }
        }

        for r in replicas.iter_mut() {
            if over(&start) {
                break;
            }
            for _ in 0..sweeps_per_slice {
                let (cmax, theta) = profile(&r.tree, &log2_sizes, &mut costs);
                sweep(
                    &mut r.tree,
                    beta,
                    kappa,
                    cmax,
                    alpha,
                    targeted,
                    theta,
                    &log2_sizes,
                    &mut scratch,
                    &mut rng,
                );
            }
            r.refresh(&log2_sizes);
        }
        for r in replicas.iter() {
            consider!(r);
        }
        if over(&start) {
            break;
        }

        // Exploitation-by-cloning: replace the worst replica with a strong
        // mutation of the best whenever the worst lags by > LAG_THRESHOLD tc.
        let (best_idx, worst_idx) = best_worst(&replicas);
        if best_idx != worst_idx
            && replicas[worst_idx].cur_tc > replicas[best_idx].cur_tc + LAG_THRESHOLD
        {
            let mut clone = replicas[best_idx].tree.clone();
            let k = rng.random_range(4..=9);
            for _ in 0..k {
                random_mutate(&mut clone, &log2_sizes, &mut scratch, &mut rng);
            }
            replicas[worst_idx] = Replica::new(clone, &log2_sizes);
            consider!(replicas[worst_idx]);
        }
    }

    // ---- 8. Attribution summary (stderr; not part of the emitted result). ---
    if debug {
        let (near_1, near_05) = near_max_counts(&best, &log2_sizes, &mut costs);
        eprintln!(
            "[dbg] FINAL best_tc={:.4} control_best_tc={:.4} profile_gain={:.4} \
             nodes_within_1.0_of_max={} within_0.5={}",
            best_tc,
            control_best_tc,
            control_best_tc - best_tc,
            near_1,
            near_05
        );
    }

    Ok(())
}

/// One member of the population: its live annealing state plus the best-visited
/// tree (by TRUE tc) along its own trajectory.
struct Replica {
    tree: ExprTree,
    cur_tc: f64,
    best_tree: ExprTree,
    best_tc: f64,
}

impl Replica {
    fn new(tree: ExprTree, log2_sizes: &[f64]) -> Self {
        let tc = tc_of(&tree, log2_sizes);
        Replica {
            best_tree: tree.clone(),
            tree,
            cur_tc: tc,
            best_tc: tc,
        }
    }

    /// Recompute the live tc and update the best-visited state.
    fn refresh(&mut self, log2_sizes: &[f64]) {
        let tc = tc_of(&self.tree, log2_sizes);
        self.cur_tc = tc;
        if tc < self.best_tc - 1e-9 {
            self.best_tc = tc;
            self.best_tree = self.tree.clone();
        }
    }
}

/// Indices of the best (lowest live tc) and worst (highest live tc) replicas.
fn best_worst(replicas: &[Replica]) -> (usize, usize) {
    let mut best_idx = 0;
    let mut worst_idx = 0;
    for (i, r) in replicas.iter().enumerate() {
        if r.cur_tc < replicas[best_idx].cur_tc - 1e-12 {
            best_idx = i;
        }
        if r.cur_tc > replicas[worst_idx].cur_tc + 1e-12 {
            worst_idx = i;
        }
    }
    (best_idx, worst_idx)
}

/// True total tc of a tree (matches the validator scorer).
#[inline]
fn tc_of(tree: &ExprTree, log2_sizes: &[f64]) -> f64 {
    tree_complexity(tree, log2_sizes).0
}

/// Cost (per-contraction tc) of one internal node = tc over the union of both
/// children's labels. Leaves have cost −∞.
#[inline]
fn node_cost(tree: &ExprTree, log2_sizes: &[f64]) -> f64 {
    match tree {
        ExprTree::Leaf(_) => f64::NEG_INFINITY,
        ExprTree::Node { left, right, info } => {
            tcscrw(
                left.labels(),
                right.labels(),
                &info.out_dims,
                log2_sizes,
                false,
            )
            .0
        }
    }
}

/// Collect every internal node's cost into `costs` (cleared first).
fn collect_costs(tree: &ExprTree, log2_sizes: &[f64], costs: &mut Vec<f64>) {
    if let ExprTree::Node { left, right, info } = tree {
        let c = tcscrw(
            left.labels(),
            right.labels(),
            &info.out_dims,
            log2_sizes,
            false,
        )
        .0;
        costs.push(c);
        collect_costs(left, log2_sizes, costs);
        collect_costs(right, log2_sizes, costs);
    }
}

/// Compute the maximum node cost c_max (for the peak-pressure term) and the
/// top-decile cost threshold θ (90th-percentile node cost; top decile =
/// cost ≥ θ) for a tree. `costs` is a reusable scratch buffer.
fn profile(tree: &ExprTree, log2_sizes: &[f64], costs: &mut Vec<f64>) -> (f64, f64) {
    costs.clear();
    collect_costs(tree, log2_sizes, costs);
    if costs.is_empty() {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    let cmax = costs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sorted = costs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() as f64) * 0.9).floor() as usize;
    let theta = sorted[idx.min(sorted.len() - 1)];
    (cmax, theta)
}

/// Peak-pressure weight of a node cost: 2^{ALPHA·(c − c_max)}. ≈0 for bulk
/// nodes, ≈1 with steep gradient for near-max nodes.
#[inline]
fn peak(c: f64, cmax: f64, alpha: f64) -> f64 {
    (alpha * (c - cmax)).exp2()
}

/// One post-order sweep. The acceptance energy is the proven LOCAL dtc plus a
/// profile-aware peak-pressure term:
///
///     ΔE = (tc1 − tc0) + κ · [peak(new_a)+peak(new_b) − peak(old_a)−peak(old_b)]
///
/// (κ=0 ⇒ exactly the standard local-dtc TreeSA energy). When `targeted`, a move
/// is proposed at a node with probability 1 if its cost is in the top decile
/// (cost ≥ θ) and `OFF_DECILE_Q` otherwise; when not targeted every node is
/// proposed (uniform). Emission/best-tracking uses TRUE tc elsewhere.
#[allow(clippy::too_many_arguments)]
fn sweep(
    tree: &mut ExprTree,
    beta: f64,
    kappa: f64,
    cmax: f64,
    alpha: f64,
    targeted: bool,
    theta: f64,
    log2_sizes: &[f64],
    scratch: &mut ScratchSpace,
    rng: &mut SmallRng,
) {
    let propose = if !targeted {
        true
    } else {
        node_cost(tree, log2_sizes) >= theta || rng.random::<f64>() < OFF_DECILE_Q
    };

    if propose {
        let rules = Rule::applicable_rules(tree, DecompositionType::Tree);
        if !rules.is_empty() {
            let rule = rules[rng.random_range(0..rules.len())];
            if let Some(diff) = scratch.rule_diff(tree, rule, log2_sizes, false) {
                let mut d_energy = diff.tc1 - diff.tc0;
                if kappa != 0.0 {
                    if let Some((old_a, old_b, new_a, new_b)) =
                        affected_costs(tree, rule, &diff.new_labels, log2_sizes)
                    {
                        let dpeak = peak(new_a, cmax, alpha) + peak(new_b, cmax, alpha)
                            - peak(old_a, cmax, alpha)
                            - peak(old_b, cmax, alpha);
                        d_energy += kappa * dpeak;
                    }
                }
                if d_energy <= 0.0 || rng.random::<f64>() < (-beta * d_energy).exp() {
                    apply_rule_mut(tree, rule, diff.new_labels);
                }
            }
        }
    }

    if let ExprTree::Node { left, right, .. } = tree {
        sweep(
            left, beta, kappa, cmax, alpha, targeted, theta, log2_sizes, scratch, rng,
        );
        sweep(
            right, beta, kappa, cmax, alpha, targeted, theta, log2_sizes, scratch, rng,
        );
    }
}

/// Individual costs of the two nodes whose contraction changes under `rule`,
/// before and after the rotation, as (old_child, old_node, new_child, new_node).
/// Mirrors the internal cost arithmetic of `ScratchSpace::rule_diff` exactly,
/// using the `new_labels` it returned. Returns None for Rule5 (no cost change)
/// or a structurally inapplicable rule.
fn affected_costs(
    tree: &ExprTree,
    rule: Rule,
    new_labels: &[usize],
    log2_sizes: &[f64],
) -> Option<(f64, f64, f64, f64)> {
    let ExprTree::Node { left, right, info } = tree else {
        return None;
    };
    let d = &info.out_dims;
    let tc = |x: &[usize], y: &[usize], z: &[usize]| tcscrw(x, y, z, log2_sizes, false).0;
    match rule {
        Rule::Rule1 | Rule::Rule2 => {
            let ExprTree::Node {
                left: a,
                right: b,
                info: ab_info,
            } = left.as_ref()
            else {
                return None;
            };
            let c = right;
            let ab = &ab_info.out_dims;
            let old_child = tc(a.labels(), b.labels(), ab);
            let old_node = tc(ab, c.labels(), d);
            let (new_child, new_node) = match rule {
                Rule::Rule1 => (
                    tc(a.labels(), c.labels(), new_labels),
                    tc(new_labels, b.labels(), d),
                ),
                _ => (
                    tc(c.labels(), b.labels(), new_labels),
                    tc(new_labels, a.labels(), d),
                ),
            };
            Some((old_child, old_node, new_child, new_node))
        }
        Rule::Rule3 | Rule::Rule4 => {
            let ExprTree::Node {
                left: b,
                right: c,
                info: bc_info,
            } = right.as_ref()
            else {
                return None;
            };
            let a = left;
            let bc = &bc_info.out_dims;
            let old_child = tc(b.labels(), c.labels(), bc);
            let old_node = tc(a.labels(), bc, d);
            let (new_child, new_node) = match rule {
                Rule::Rule3 => (
                    tc(a.labels(), c.labels(), new_labels),
                    tc(b.labels(), new_labels, d),
                ),
                _ => (
                    tc(b.labels(), a.labels(), new_labels),
                    tc(c.labels(), new_labels, d),
                ),
            };
            Some((old_child, old_node, new_child, new_node))
        }
        // Rule5 swaps children: node cost is symmetric ⇒ unchanged.
        Rule::Rule5 => None,
    }
}

/// Apply one random rotation somewhere in the tree (a perturbation kick),
/// accepted unconditionally. Descends via a random walk so deep nodes are
/// reached. Returns true if a mutation was applied.
fn random_mutate(
    tree: &mut ExprTree,
    log2_sizes: &[f64],
    scratch: &mut ScratchSpace,
    rng: &mut SmallRng,
) -> bool {
    let (go_left, go_right) = match tree {
        ExprTree::Leaf(_) => return false,
        ExprTree::Node { left, right, .. } => (!left.is_leaf(), !right.is_leaf()),
    };
    let choice = rng.random_range(0..3);
    if choice == 1 && go_left {
        if let ExprTree::Node { left, .. } = tree {
            if random_mutate(left, log2_sizes, scratch, rng) {
                return true;
            }
        }
    } else if choice == 2 && go_right {
        if let ExprTree::Node { right, .. } = tree {
            if random_mutate(right, log2_sizes, scratch, rng) {
                return true;
            }
        }
    }
    let rules = Rule::applicable_rules(tree, DecompositionType::Tree);
    if rules.is_empty() {
        return false;
    }
    let rule = rules[rng.random_range(0..rules.len())];
    if let Some(diff) = scratch.rule_diff(tree, rule, log2_sizes, false) {
        apply_rule_mut(tree, rule, diff.new_labels);
        return true;
    }
    false
}

/// Number of internal-node costs within 1.0 and within 0.5 of the max cost.
fn near_max_counts(tree: &ExprTree, log2_sizes: &[f64], costs: &mut Vec<f64>) -> (usize, usize) {
    costs.clear();
    collect_costs(tree, log2_sizes, costs);
    let mx = costs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let within_1 = costs.iter().filter(|&&c| c >= mx - 1.0 - 1e-9).count();
    let within_05 = costs.iter().filter(|&&c| c >= mx - 0.5 - 1e-9).count();
    (within_1, within_05)
}

/// Convert a greedy `NestedEinsum` into a mutable `ExprTree`.
fn nested_to_expr(
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
                    let out_dims: Vec<usize> = eins.ixs[0].iter().map(|l| label_map[l]).collect();
                    ExprTree::leaf(out_dims, *tensor_index)
                }
                NestedEinsum::Node { .. } => nested_to_expr(&args[0], label_map)?,
            };
            let right = match &args[1] {
                NestedEinsum::Leaf { tensor_index } => {
                    let out_dims: Vec<usize> = eins.ixs[1].iter().map(|l| label_map[l]).collect();
                    ExprTree::leaf(out_dims, *tensor_index)
                }
                NestedEinsum::Node { .. } => nested_to_expr(&args[1], label_map)?,
            };
            let out_dims: Vec<usize> = eins.iy.iter().map(|l| label_map[l]).collect();
            Some(ExprTree::node(left, right, out_dims))
        }
    }
}

/// Convert an `ExprTree` back to a `NestedEinsum` for `writejson`.
fn expr_to_nested(
    tree: &ExprTree,
    original_ixs: &[Vec<usize>],
    inverse_map: &[usize],
    openedges: &[usize],
    level: usize,
) -> NestedEinsum<usize> {
    match tree {
        ExprTree::Leaf(info) => NestedEinsum::leaf(info.tensor_id.unwrap_or(0)),
        ExprTree::Node { left, right, info } => {
            let left_nested = expr_to_nested(left, original_ixs, inverse_map, openedges, level + 1);
            let right_nested =
                expr_to_nested(right, original_ixs, inverse_map, openedges, level + 1);
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
