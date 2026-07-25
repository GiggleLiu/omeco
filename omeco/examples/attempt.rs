//! Attempt entry point for the autoresearch validator (attempt-058).
//!
//! Contract: `attempt <graph.json> <budget_ms> <out.json>` — read an einsum
//! graph, search for a contraction order within the wall-clock budget, and keep
//! the best tree found (by pure time complexity `tc`) written atomically to
//! `out.json` in omeco `writejson` format.
//!
//! ADAPTIVE WIDTH-CAPPED PEEL for annealer-immobile scales (attempt-058, parent
//! 053). attempt-053 peeled the cheap periphery under ONE fixed cost cap, handed
//! the residual to library TreeSA, and spliced the peel subtrees back — and was
//! FALSIFIED on nqueens_28 (4086 labels), where full-graph TreeSA is competitive
//! (134) and peel+residual-TreeSA lost at every cap (>=188): good large-treewidth
//! orders interleave core and periphery, and a fixed peel boundary removes that
//! freedom.
//!
//! NEW REGIME. On the UAI relational instances the annealer is IMMOBILE: at
//! 30k-70k tensors a full-graph anneal barely completes a sweep and is stuck at
//! tc~202/sc=200 at every budget 90-900s (double the width-100 optimum). Here the
//! alternative to peeling is not "better interleaving" but "no optimization at
//! all". This attempt answers the 053 objection two ways: the cap is a LADDER,
//! not a point; and the full-graph anneal still RACES as a fallback, protecting
//! the falsified regime.
//!
//! LADDER (Phase A, <=40% of budget). Rungs are an escalating sequence of
//! factor-width caps; the TOP rung is cap = infinity, i.e. full min-cost VE run
//! to completion (the attempt-038 seed). Each rung: peel the periphery in
//! min-cost order while each new factor stays under the cap -> hand the residual
//! core to the library annealer -> splice each peel subtree back -> score;
//! best full tree by measured tc wins. A residual is annealed only when it is
//! small enough that the annealer can move it (<=3000 tensors); larger residuals
//! are recorded but left to VE / the full-graph fallback. The infinity rung (full
//! VE) reaches tc~109 on uai_relational_4 (30400 tensors, width 100) and tc~24 on
//! uai_relational_5 (70000 tensors, width 10) in well under a second, where the
//! full-graph anneal is frozen at 202.
//!
//! FLOOR. The library greedy is ~O(n·deg^2) and does not finish at 30k+ tensors,
//! so it is run only for n<=6000; at scale the cap=infinity rung (VE) IS the
//! first always-valid tree — a proper elimination tree, cheap to build and score,
//! unlike a degenerate chain over an un-eliminated core. TreeSA's own Greedy
//! initializer has the same cost as the greedy, so anneals on >4000-tensor graphs
//! use Random init.
//!
//! Phase B: the remaining budget goes to whichever arm is winning, with a
//! full-graph TreeSA doubling run racing so the result is never worse than the
//! base annealer on expanders / small large-treewidth cores where peeling is
//! useless (the falsified-053 protection). The fallback is GATED to n<=8000: above
//! that the annealer is immobile (only ever ~202 on the relational cores, never
//! beating VE) and one uninterruptible niters round would overrun the budget, so
//! it is skipped and the process ends as soon as VE has won. Best-by-`tc` wins
//! (sc ignored). Single-threaded on one worker thread
//! with a large stack (deep trees exceed the default 8 MB stack at scale); the
//! main thread blocks on the join, so CPU never exceeds wall. No per-instance
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

    let m = hg.id_label.len();

    // Attribution knobs (default OFF; production is `auto`). MODE=ve|peel|full
    // forces a single lane; PEEL_CAP pins one finite cap for a local sweep.
    let mode = std::env::var("MODE").unwrap_or_else(|_| "auto".into());
    let cap_override = std::env::var("PEEL_CAP")
        .ok()
        .and_then(|s| s.parse::<f64>().ok());

    // TreeSA's Greedy initializer calls the library greedy, which is ~O(n·deg^2)
    // and does NOT finish at 30k+ tensors (the relational instances). Above this
    // size both the full-graph and residual anneals switch to the Random
    // initializer so a sweep can actually start; below it the better Greedy start
    // is kept. `mobile` is the (smaller) size at which annealing can still improve
    // the order at all — the whole hypothesis is peeling the core below it.
    let big_n = |cnt: usize| cnt > 4000;
    let mobile = |cnt: usize| cnt <= 3000;
    let greedy_ok = n <= 6000;

    // =====================================================================
    // Adaptive width-capped peel LADDER (Phase A, <=40% of budget). The rungs
    // are an escalating sequence of factor-width caps; the TOP rung is
    // cap = infinity, i.e. full min-cost variable elimination run to
    // completion (the attempt-038 VE seed). Each rung peels the periphery,
    // hands the residual core to the library annealer, splices the peel
    // subtrees back, and scores; the best full tree by measured tc is kept.
    //
    // Why this is NOT the falsified 053: 053 used ONE fixed cap and always
    // deferred to TreeSA, in a regime (nqueens, 4k tensors) where full-graph
    // TreeSA is COMPETITIVE, so removing interleaving freedom only hurt. Here
    // the regime is annealer-IMMOBILE (30k-70k tensors; full-graph anneal
    // stuck at tc~202 at every budget). The ladder is adaptive AND its top
    // rung is unbounded VE, which reaches the width-~100/~10 optimum in <1s
    // where the annealer cannot move; and the full-graph anneal still RACES
    // as a fallback so the falsified regime is protected.
    // =====================================================================
    let phase_a_deadline = (budget_ms * 0.40).min(deadline_ms);

    // Rung 0 = cap infinity: full min-cost VE run to completion. This is the
    // primary winner on the relational instances AND the scale floor: a proper
    // elimination tree is cheap to BUILD and SCORE (unlike a degenerate chain
    // over an un-eliminated core), reaching tc~109/~24 in under a second where
    // the full-graph anneal is frozen. Boxed so a blow-up order (expander)
    // cannot eat Phase A — and the library greedy is ~O(n*deg^2) and does not
    // finish at 30k+ tensors, so at scale VE is the first always-valid tree.
    let ve_box = phase_a_deadline.min(elapsed_ms() + budget_ms * 0.25);
    let mut rng = SmallRng::seed_from_u64(0x0000_0058_c0ff_ee00);
    let ve_topo = hg.ve_order(0.0, &mut rng, &start, ve_box);
    let mut best = hg.build_nested(&ve_topo);
    let mut best_tc = tc_of(&best);
    let ve_tc = best_tc;
    let mut best_source = "ve(cap=inf)";
    write_atomic(&out_path, &best)?;
    eprintln!(
        "t={:.0}ms tc={best_tc:.4} (rung cap=inf / full VE)",
        elapsed_ms()
    );

    // Library greedy: good and tested, but only for n<=6000 (it hangs at scale).
    // Kept if it beats VE — e.g. expanders, where VE blows up.
    if greedy_ok {
        if let Some(g) = optimize_code(&code, &sizes, &GreedyMethod::default()) {
            let gtc = tc_of(&g);
            if gtc < best_tc - 1e-9 {
                best = g;
                best_tc = gtc;
                best_source = "greedy";
                write_atomic(&out_path, &best)?;
            }
        }
    }
    let tc_floor = best_tc;

    // Finite rungs: probe a geometric cap ladder to find w0 = the smallest cap
    // that already peels >=50% of the tensors, then evaluate {0.8 w0, w0,
    // 1.25 w0}. A rung's residual is annealed only when it is `mobile` (small
    // enough that the annealer can actually improve it) — on the relational core
    // the peel cannot shrink below ~10k, the annealer is as stuck there as on the
    // full graph, so those rungs are recorded but not annealed (VE already wins).
    let mut best_peel: Option<(f64, Peel)> = None;
    let mut peel_stats: Vec<(u32, usize)> = Vec::new();
    if mode == "auto" || mode == "peel" {
        let probe_caps: Vec<f64> = match cap_override {
            Some(c) => vec![c],
            None => vec![6.0, 10.0, 16.0, 26.0, 42.0],
        };
        let mut w0 = *probe_caps.last().unwrap();
        let mut best_frac = f64::INFINITY;
        let mut w0_set = false;
        for &c in &probe_caps {
            if elapsed_ms() >= phase_a_deadline {
                break;
            }
            let p = hg.peel(c, &start, phase_a_deadline);
            let k = p.residual.len();
            peel_stats.push((c as u32, k));
            eprintln!(
                "t_peel={:.0}ms cap={c:.0} peeled={}/{n} residual={k}",
                p.peel_ms,
                n - k
            );
            let frac = k as f64 / n as f64;
            best_frac = best_frac.min(frac);
            if !w0_set && frac <= 0.5 {
                w0 = c;
                w0_set = true;
            }
        }
        let peel_useful = best_frac <= 0.6;
        eprintln!(
            "n={n} n_labels={m} floor={tc_floor:.4} ve={ve_tc:.4} best={best_tc:.4} \
             | probe w0={w0:.1} best_frac={best_frac:.3} useful={peel_useful} \
             caps_residual={peel_stats:?} | t={:.0}ms",
            elapsed_ms()
        );

        if peel_useful || mode == "peel" {
            let mut caps: Vec<f64> = match cap_override {
                Some(_) => vec![w0],
                None => vec![(0.8 * w0).max(4.0), w0, 1.25 * w0],
            };
            caps.sort_by(|a, b| a.partial_cmp(b).unwrap());
            caps.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
            for (i, &c) in caps.iter().enumerate() {
                if elapsed_ms() >= phase_a_deadline {
                    break;
                }
                let p = hg.peel(c, &start, phase_a_deadline);
                let rk = p.residual.len();
                // Only anneal a residual the annealer can move; otherwise VE /
                // full-graph fallback own this instance.
                if !mobile(rk) && mode != "peel" {
                    eprintln!(
                        "t={:.0}ms rung {}/{} cap={c:.1} residual={rk} (immobile, skip)",
                        elapsed_ms(),
                        i + 1,
                        caps.len()
                    );
                    continue;
                }
                let per_rung =
                    ((phase_a_deadline - elapsed_ms()) / (caps.len() - i) as f64).max(0.0);
                let rung_deadline = (elapsed_ms() + per_rung).min(phase_a_deadline);
                let before = best_tc;
                let init_random = big_n(rk);
                let rtc = residual_treesa(
                    &hg,
                    &p,
                    &sizes,
                    &out_path,
                    &mut best,
                    &mut best_tc,
                    tc_of,
                    &start,
                    rung_deadline,
                    init_random,
                )?;
                if best_tc < before - 1e-9 {
                    best_source = "residual";
                }
                eprintln!(
                    "t={:.0}ms rung {}/{} cap={c:.1} residual={rk} rtc={rtc:.4}",
                    elapsed_ms(),
                    i + 1,
                    caps.len()
                );
                match &best_peel {
                    Some((btc, _)) if *btc <= rtc => {}
                    _ => best_peel = Some((rtc, p)),
                }
            }
        }
    }

    // =====================================================================
    // Phase B: hand the remaining budget to whichever arm is winning, with the
    // full-graph TreeSA fallback racing so we never end up worse than the base
    // annealer where peeling/VE is useless (expanders, small large-treewidth
    // cores) — this is the protection for the falsified-053 regime. The fallback
    // is GATED to n<=full_treesa_max: above that the annealer is immobile (it
    // only ever reaches ~202 on the relational cores, never beating VE) AND a
    // single niters round is uninterruptible and would overrun the budget by
    // minutes, so running it is pure downside. best-by-tc, emitted eagerly.
    // =====================================================================
    let residual_won = best_source == "residual" && best_peel.is_some();
    // A full-graph niters round is uninterruptible; only start one where it fits.
    let full_treesa_max = 8000usize;
    let run_full = |n: usize| n <= full_treesa_max;

    match mode.as_str() {
        "peel" => {
            if let Some((_, p)) = &best_peel {
                let init_random = big_n(p.residual.len());
                residual_treesa(
                    &hg,
                    p,
                    &sizes,
                    &out_path,
                    &mut best,
                    &mut best_tc,
                    tc_of,
                    &start,
                    deadline_ms,
                    init_random,
                )?;
            }
        }
        "ve" => {}
        "full" => {
            treesa_doubling(
                &code,
                &sizes,
                &out_path,
                &mut best,
                &mut best_tc,
                tc_of,
                &start,
                deadline_ms,
                big_n(n),
            )?;
        }
        _ => {
            // auto: winning arm gets the larger share; fallback races when it fits.
            if residual_won {
                if let Some((_, p)) = &best_peel {
                    let init_random = big_n(p.residual.len());
                    let rd = if run_full(n) {
                        (elapsed_ms() + (deadline_ms - elapsed_ms()) * 0.70).min(deadline_ms)
                    } else {
                        deadline_ms
                    };
                    residual_treesa(
                        &hg,
                        p,
                        &sizes,
                        &out_path,
                        &mut best,
                        &mut best_tc,
                        tc_of,
                        &start,
                        rd,
                        init_random,
                    )?;
                }
            }
            if run_full(n) {
                treesa_doubling(
                    &code,
                    &sizes,
                    &out_path,
                    &mut best,
                    &mut best_tc,
                    tc_of,
                    &start,
                    deadline_ms,
                    big_n(n),
                )?;
            }
        }
    }

    eprintln!(
        "t_final={:.0}ms tc_final={best_tc:.4} source={best_source} residual_won={residual_won}",
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
    init_random: bool,
) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
    let elapsed_ms = || start.elapsed().as_secs_f64() * 1e3;
    let k = peel.residual.len();
    let mut local_best = f64::INFINITY;
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
        let mut ts = TreeSA::default()
            .with_ntrials(1)
            .with_niters(niters)
            .with_sc_target(f64::INFINITY);
        // Greedy init calls the library greedy (hangs at 30k+); above that size
        // fall back to Random init so a sweep can start at all.
        ts.initializer = if init_random {
            omeco::Initializer::Random
        } else {
            omeco::Initializer::Greedy
        };
        let Some(rtree) = optimize_code(&rcode, sizes, &ts) else {
            break;
        };
        // Splice residual topology -> full topology over original leaves.
        let mut supers: Vec<Option<TopoTree>> =
            peel.residual.iter().map(|(_, t)| Some(t.clone())).collect();
        let topo = splice(&rtree, &mut supers);
        let tree = hg.build_nested(&topo);
        let tc = tc_of(&tree);
        local_best = local_best.min(tc);
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
    Ok(local_best)
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
    init_random: bool,
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
        let mut ts = TreeSA::default()
            .with_ntrials(1)
            .with_niters(niters)
            .with_sc_target(f64::INFINITY);
        ts.initializer = if init_random {
            omeco::Initializer::Random
        } else {
            omeco::Initializer::Greedy
        };
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
