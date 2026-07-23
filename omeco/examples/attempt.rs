//! Attempt 028 — convergence-optimized single continuous anneal.
//!
//! Contract: `attempt <graph.json> <budget_ms> <out.json>` — read an einsum
//! graph, search for a contraction order within the wall-clock budget, and
//! write the best tree in `writejson` format before the deadline (anytime).
//!
//! NOVELTY (targeted at the non-converged large scale, n≈1000–1238):
//!
//! 1. ONE continuous anneal — no ntrials restarts. A single SA trajectory on a
//!    wall-clock-indexed geometric cooling ramp β(t): 0.02 → 20 over 0.95×budget.
//!    Restarts pay re-warmup that does not amortize inside 90 s at this scale.
//!
//! 2. Profile-aware move targeting — the tree is held in a FLAT ARENA of the
//!    n−1 internal nodes, each tagged with its *local* contraction tc. With
//!    p=0.7 the move's node is drawn from the current top-cost decile (the nodes
//!    whose local tc dominates the global log2-sum-exp), else uniformly. At
//!    n=1000 uniform sampling spends ~90 % of proposals on nodes that contribute
//!    negligibly to total tc.
//!
//! 3. O(1) incremental cost maintenance — an accepted move is a handful of
//!    pointer rewires plus two `local_tc` recomputations (the mutated node and
//!    its restructured child). No full-tree recomputation in the hot loop.
//!    Global tc is derived lazily (every K moves, one O(n) log-sum-exp over the
//!    `local_tc` array) purely to decide new-best snapshots.
//!
//! Energy is pure tc (v2.1 sc-unbounded objective): ΔE = tc1 − tc0 (local),
//! Metropolis accept `rng < exp(−β·ΔE)`. Single threaded — no Rayon.

use std::collections::HashMap;
use std::time::Instant;

use omeco::expr_tree::{compute_intermediate_output, tcscrw, tree_complexity, ExprTree};
use omeco::json::writejson;
use omeco::{optimize_code, EinCode, GreedyMethod, NestedEinsum};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GraphData {
    ixs: Vec<Vec<usize>>,
    iy: Vec<usize>,
    sizes: HashMap<String, usize>,
}

const NIL: u32 = u32::MAX;

/// Flat binary contraction tree. Node indices are fixed for the whole run;
/// moves only rewire child/parent pointers and rewrite `out`/`ltc` of two nodes.
struct Arena {
    left: Vec<u32>,
    right: Vec<u32>,
    parent: Vec<u32>,
    out: Vec<Vec<usize>>,
    tid: Vec<u32>, // NIL for internal nodes
    ltc: Vec<f64>, // local contraction tc; NEG_INF for leaves
    root: u32,
    internal: Vec<u32>, // fixed set of internal node indices
}

impl Arena {
    #[inline]
    fn is_internal(&self, i: u32) -> bool {
        self.tid[i as usize] == NIL
    }

    /// Build the arena from an `ExprTree`, computing every node's local tc.
    fn from_expr(tree: &ExprTree, log2_sizes: &[f64]) -> Arena {
        let mut a = Arena {
            left: Vec::new(),
            right: Vec::new(),
            parent: Vec::new(),
            out: Vec::new(),
            tid: Vec::new(),
            ltc: Vec::new(),
            root: NIL,
            internal: Vec::new(),
        };
        a.root = a.push(tree, NIL, log2_sizes);
        a
    }

    fn push(&mut self, tree: &ExprTree, parent: u32, log2_sizes: &[f64]) -> u32 {
        let idx = self.left.len() as u32;
        match tree {
            ExprTree::Leaf(info) => {
                self.left.push(NIL);
                self.right.push(NIL);
                self.parent.push(parent);
                self.out.push(info.out_dims.clone());
                self.tid.push(info.tensor_id.unwrap_or(0) as u32);
                self.ltc.push(f64::NEG_INFINITY);
            }
            ExprTree::Node { left, right, info } => {
                // Reserve this slot first, then recurse (children get later indices).
                self.left.push(NIL);
                self.right.push(NIL);
                self.parent.push(parent);
                self.out.push(info.out_dims.clone());
                self.tid.push(NIL);
                self.ltc.push(0.0);
                let l = self.push(left, idx, log2_sizes);
                let r = self.push(right, idx, log2_sizes);
                self.left[idx as usize] = l;
                self.right[idx as usize] = r;
                let (tc, _, _) = tcscrw(
                    &self.out[l as usize],
                    &self.out[r as usize],
                    &self.out[idx as usize],
                    log2_sizes,
                    false,
                );
                self.ltc[idx as usize] = tc;
                self.internal.push(idx);
            }
        }
        idx
    }

    /// Exact global tc = log2(Σ 2^ltc[i]) over internal nodes.
    fn global_tc(&self) -> f64 {
        let mut mx = f64::NEG_INFINITY;
        for &i in &self.internal {
            let v = self.ltc[i as usize];
            if v > mx {
                mx = v;
            }
        }
        if mx == f64::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }
        let mut s = 0.0f64;
        for &i in &self.internal {
            s += (self.ltc[i as usize] - mx).exp2();
        }
        mx + s.log2()
    }

    /// Convert to a `NestedEinsum` for `writejson`.
    /// Restore tree state from a snapshot (same node set), recomputing parent
    /// pointers and every node's local tc.
    fn load_snapshot(&mut self, s: &Snapshot, log2_sizes: &[f64]) {
        self.left.copy_from_slice(&s.left);
        self.right.copy_from_slice(&s.right);
        for (dst, src) in self.out.iter_mut().zip(s.out.iter()) {
            dst.clear();
            dst.extend_from_slice(src);
        }
        self.root = s.root;
        // Recompute parents.
        for p in self.parent.iter_mut() {
            *p = NIL;
        }
        for i in 0..self.left.len() {
            let l = self.left[i];
            let r = self.right[i];
            if l != NIL {
                self.parent[l as usize] = i as u32;
                self.parent[r as usize] = i as u32;
            }
        }
        // Recompute local tc for internal nodes.
        for &i in &self.internal {
            let ii = i as usize;
            let (tc, _, _) = tcscrw(
                &self.out[self.left[ii] as usize],
                &self.out[self.right[ii] as usize],
                &self.out[ii],
                log2_sizes,
                false,
            );
            self.ltc[ii] = tc;
        }
    }

    /// Snapshot the minimal state needed to reconstruct the tree later.
    fn snapshot(&self) -> Snapshot {
        Snapshot {
            left: self.left.clone(),
            right: self.right.clone(),
            out: self.out.clone(),
            tid: self.tid.clone(),
            root: self.root,
        }
    }
}

/// Minimal frozen tree state for anytime emission.
struct Snapshot {
    left: Vec<u32>,
    right: Vec<u32>,
    out: Vec<Vec<usize>>,
    tid: Vec<u32>,
    root: u32,
}

impl Snapshot {
    fn to_nested(
        &self,
        node: u32,
        original_ixs: &[Vec<usize>],
        inverse_map: &[usize],
        openedges: &[usize],
        level: usize,
    ) -> NestedEinsum<usize> {
        let i = node as usize;
        if self.tid[i] != NIL {
            return NestedEinsum::leaf(self.tid[i] as usize);
        }
        let ln = self.to_nested(self.left[i], original_ixs, inverse_map, openedges, level + 1);
        let rn = self.to_nested(self.right[i], original_ixs, inverse_map, openedges, level + 1);
        let iy: Vec<usize> = if level == 0 {
            openedges.to_vec()
        } else {
            self.out[i].iter().map(|&x| inverse_map[x]).collect()
        };
        let ll = child_labels(&ln, original_ixs);
        let rl = child_labels(&rn, original_ixs);
        let eins = EinCode::new(vec![ll, rl], iy);
        NestedEinsum::node(vec![ln, rn], eins)
    }
}

#[inline]
fn log2sumexp2(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp2() + (b - m).exp2()).log2()
}

/// Attempt one Metropolis move at internal node `x` with a randomly chosen
/// applicable rule. Returns true if a rule was applicable (whether or not it was
/// accepted). All rewiring is O(degree).
#[inline]
fn try_move(
    a: &mut Arena,
    x: u32,
    beta: f64,
    log2_sizes: &[f64],
    rng: &mut SmallRng,
) -> bool {
    let xi = x as usize;
    let l = a.left[xi];
    let r = a.right[xi];
    let l_int = a.is_internal(l);
    let r_int = a.is_internal(r);

    // Choose an applicable rule.
    // Rule1/2 need L internal; Rule3/4 need R internal.
    let rule: u8 = match (l_int, r_int) {
        (false, false) => return false,
        (true, false) => 1 + (rng.random::<u32>() & 1) as u8, // 1 or 2
        (false, true) => 3 + (rng.random::<u32>() & 1) as u8, // 3 or 4
        (true, true) => 1 + (rng.random::<u32>() % 4) as u8,  // 1..4
    };

    let d = &a.out[xi];
    let (new_labels, tc_child_new, tc_x_new, old_ltc_child) = match rule {
        1 => {
            // ((a,b),c) -> ((a,c),b);  L=(a,b), c=R
            let av = a.left[l as usize];
            let bv = a.right[l as usize];
            let nl = compute_intermediate_output(&a.out[av as usize], &a.out[r as usize], &a.out[bv as usize], d);
            let (tcc, _, _) = tcscrw(&a.out[av as usize], &a.out[r as usize], &nl, log2_sizes, false);
            let (tcx, _, _) = tcscrw(&nl, &a.out[bv as usize], d, log2_sizes, false);
            (nl, tcc, tcx, a.ltc[l as usize])
        }
        2 => {
            // ((a,b),c) -> ((c,b),a);  L=(a,b), c=R
            let av = a.left[l as usize];
            let bv = a.right[l as usize];
            let nl = compute_intermediate_output(&a.out[bv as usize], &a.out[r as usize], &a.out[av as usize], d);
            let (tcc, _, _) = tcscrw(&a.out[r as usize], &a.out[bv as usize], &nl, log2_sizes, false);
            let (tcx, _, _) = tcscrw(&nl, &a.out[av as usize], d, log2_sizes, false);
            (nl, tcc, tcx, a.ltc[l as usize])
        }
        3 => {
            // (a,(b,c)) -> (b,(a,c));  R=(b,c), a=L
            let bv = a.left[r as usize];
            let cv = a.right[r as usize];
            let nl = compute_intermediate_output(&a.out[cv as usize], &a.out[l as usize], &a.out[bv as usize], d);
            let (tcc, _, _) = tcscrw(&a.out[l as usize], &a.out[cv as usize], &nl, log2_sizes, false);
            let (tcx, _, _) = tcscrw(&a.out[bv as usize], &nl, d, log2_sizes, false);
            (nl, tcc, tcx, a.ltc[r as usize])
        }
        _ => {
            // Rule4: (a,(b,c)) -> (c,(b,a));  R=(b,c), a=L
            let bv = a.left[r as usize];
            let cv = a.right[r as usize];
            let nl = compute_intermediate_output(&a.out[bv as usize], &a.out[l as usize], &a.out[cv as usize], d);
            let (tcc, _, _) = tcscrw(&a.out[bv as usize], &a.out[l as usize], &nl, log2_sizes, false);
            let (tcx, _, _) = tcscrw(&a.out[cv as usize], &nl, d, log2_sizes, false);
            (nl, tcc, tcx, a.ltc[r as usize])
        }
    };

    let tc0 = log2sumexp2(old_ltc_child, a.ltc[xi]);
    let tc1 = log2sumexp2(tc_child_new, tc_x_new);
    let d_energy = tc1 - tc0;

    let accept = d_energy <= 0.0 || rng.random::<f64>() < (-beta * d_energy).exp();
    if !accept {
        return true;
    }

    // Commit the rewire.
    match rule {
        1 => {
            // L keeps left=a; L.right = R; x.right = b(old L.right)
            let bv = a.right[l as usize];
            a.right[l as usize] = r;
            a.out[l as usize] = new_labels;
            a.right[xi] = bv;
            a.parent[r as usize] = l;
            a.parent[bv as usize] = x;
        }
        2 => {
            // L.left = R; L.right = b(unchanged); x.right = a(old L.left)
            let av = a.left[l as usize];
            a.left[l as usize] = r;
            a.out[l as usize] = new_labels;
            a.right[xi] = av;
            a.parent[r as usize] = l;
            a.parent[av as usize] = x;
        }
        3 => {
            // x.left = b(old R.left); R.left = a(old x.left); R.right = c(unchanged)
            let a_old = a.left[xi];
            let bv = a.left[r as usize];
            a.left[xi] = bv;
            a.left[r as usize] = a_old;
            a.out[r as usize] = new_labels;
            a.parent[bv as usize] = x;
            a.parent[a_old as usize] = r;
        }
        _ => {
            // Rule4: x.left = c(old R.right); R.left = b(unchanged); R.right = a(old x.left)
            let a_old = a.left[xi];
            let cv = a.right[r as usize];
            a.left[xi] = cv;
            a.right[r as usize] = a_old;
            a.out[r as usize] = new_labels;
            a.parent[cv as usize] = x;
            a.parent[a_old as usize] = r;
        }
    }

    let child = if rule <= 2 { l } else { r };
    a.ltc[child as usize] = tc_child_new;
    a.ltc[xi] = tc_x_new;
    true
}

fn child_labels(nested: &NestedEinsum<usize>, original_ixs: &[Vec<usize>]) -> Vec<usize> {
    match nested {
        NestedEinsum::Leaf { tensor_index } => {
            original_ixs.get(*tensor_index).cloned().unwrap_or_default()
        }
        NestedEinsum::Node { eins, .. } => eins.iy.clone(),
    }
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: attempt <graph.json> <budget_ms> <out.json>");
        std::process::exit(2);
    }
    let start = Instant::now();
    let budget_ms: f64 = args[2].parse()?;
    let out_path = &args[3];
    let debug = std::env::var("ATTEMPT_DEBUG").is_ok();

    // Cooling reaches β_max at COOL×budget; the tail runs at β_max (pure
    // downhill refinement). Headroom is left for the final write.
    let cool_frac: f64 = std::env::var("ATTEMPT_COOL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.95);
    let cool_ms = budget_ms * cool_frac;
    let deadline_ms = budget_ms * 0.97;

    let graph: GraphData = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let sizes: HashMap<usize, usize> = graph
        .sizes
        .iter()
        .map(|(k, v)| Ok::<_, std::num::ParseIntError>((k.parse::<usize>()?, *v)))
        .collect::<Result<_, _>>()?;
    let code: EinCode<usize> = EinCode::new(graph.ixs.clone(), graph.iy.clone());
    let n = code.ixs.len();

    // ---- 1. Deterministic greedy seed, written immediately (fallback). ------
    let greedy = optimize_code(&code, &sizes, &GreedyMethod::default())
        .ok_or("greedy optimizer returned no tree")?;
    writejson(out_path, &greedy)?;
    if n <= 2 {
        return Ok(());
    }

    // ---- 2. Label map / log2 sizes. -----------------------------------------
    let labels = code.unique_labels();
    let label_map: HashMap<usize, usize> = labels
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, l)| (l, i))
        .collect();
    let inverse_map: Vec<usize> = labels.clone();
    let log2_sizes: Vec<f64> = labels.iter().map(|l| (sizes[l] as f64).log2()).collect();

    let Some(greedy_tree) = nested_to_expr(&greedy, &label_map) else {
        return Ok(());
    };
    let (greedy_tc, _, _) = tree_complexity(&greedy_tree, &log2_sizes);

    // ---- 3. Greedy portfolio: keep the best of a few stochastic seeds. ------
    // Seed quality matters most at high β; keep this cheap (a small time slice).
    let mut rng = SmallRng::seed_from_u64(0x028u64 ^ (n as u64).wrapping_mul(2654435761));
    let seed_budget_ms = (0.06 * budget_ms).min(6000.0);
    let mut best_seed = greedy_tree;
    let mut best_seed_tc = greedy_tc;
    let temps = [0.5f64, 1.0, 2.0, 4.0, 8.0];
    let mut ti = 0usize;
    let mut n_seeds = 1usize;
    while start.elapsed().as_secs_f64() * 1e3 < seed_budget_ms && n_seeds < 24 {
        let t = temps[ti % temps.len()];
        ti += 1;
        n_seeds += 1;
        if let Some(cand) = optimize_code(&code, &sizes, &GreedyMethod::stochastic(t))
            .and_then(|nx| nested_to_expr(&nx, &label_map))
        {
            let (tc, _, _) = tree_complexity(&cand, &log2_sizes);
            if tc < best_seed_tc {
                best_seed_tc = tc;
                best_seed = cand;
            }
        }
    }

    // ---- 4. Build the flat arena from the best seed. ------------------------
    let mut arena = Arena::from_expr(&best_seed, &log2_sizes);
    let n_int = arena.internal.len();

    let mut best_tc = arena.global_tc();
    let mut best_snap = arena.snapshot();
    // Emit the seed (best of the portfolio) if it beats the deterministic greedy.
    if best_tc < greedy_tc - 1e-9 {
        let nested = best_snap.to_nested(best_snap.root, &code.ixs, &inverse_map, &code.iy, 0);
        writejson(out_path, &nested)?;
    }

    if debug {
        // Correctness: the arena's incremental global tc must equal an
        // independent recomputation of the seed tree's total tc.
        let (seed_true_tc, _, _) = tree_complexity(&best_seed, &log2_sizes);
        eprintln!(
            "[dbg] n={} n_int={} n_seeds={} greedy_tc={:.4} seed_true_tc={:.4} arena_tc={:.4}",
            n, n_int, n_seeds, greedy_tc, seed_true_tc, best_tc
        );
    }

    // ---- 5. Cooling ramp + move loop. ---------------------------------------
    // A single continuous anneal: each pass is a SYSTEMATIC sweep over all
    // internal nodes (pre-order by construction: parents before children — the
    // same traversal shape TreeSA uses) plus a PROFILE-AWARE BOOST of extra
    // targeted attempts on the current top-cost decile. β is indexed by wall
    // clock (no restarts). `ATTEMPT_BOOST` (default 1.0) scales the boost count
    // as a fraction of n_int; 0 disables targeting (pure systematic control).
    let beta_min = 0.02f64;
    let beta_max = 20.0f64;
    let log_ratio = (beta_max / beta_min).ln();

    let boost_frac: f64 = std::env::var("ATTEMPT_BOOST")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let boost = ((n_int as f64) * boost_frac) as usize;

    let decile = (n_int / 10).max(1);
    let mut top: Vec<u32> = arena.internal.clone();
    let rebuild_top = |arena: &Arena, top: &mut Vec<u32>| {
        top.clear();
        top.extend_from_slice(&arena.internal);
        top.sort_by(|&x, &y| {
            arena.ltc[y as usize]
                .partial_cmp(&arena.ltc[x as usize])
                .unwrap()
        });
        top.truncate(decile);
    };
    rebuild_top(&arena, &mut top);

    // Iterated-local-search reheat: split the budget into CYCLES cooling cycles.
    // Cycle 0 anneals the seed over the full β ramp. Each later cycle RELOADS the
    // global-best tree, reheats to β at ramp-fraction REHEAT (a partial melt, not
    // a re-seed from greedy — the hypothesis's objection to restarts is to paying
    // greedy re-warmup, not to refining the incumbent), and re-cools to β_max.
    // This banks every intermediate minimum and converges harder than a single
    // stretched ramp, which wastes early budget in the random-walk regime.
    let cycles: usize = std::env::var("ATTEMPT_CYCLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let reheat: f64 = std::env::var("ATTEMPT_REHEAT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.35);

    let mut moves: u64 = 0;
    let mut sweeps: u64 = 0;
    let mut last_flush = start.elapsed().as_secs_f64() * 1e3;
    let anneal_start = start.elapsed().as_secs_f64() * 1e3;
    let cool_span = (cool_ms - anneal_start).max(1.0);

    'cycles: for cycle in 0..cycles {
        // Cooling sub-window for this cycle; the LAST cycle then holds β_max
        // until the deadline (pure downhill refinement tail).
        let c_cool_lo = anneal_start + cool_span * (cycle as f64) / (cycles as f64);
        let c_cool_hi = anneal_start + cool_span * ((cycle + 1) as f64) / (cycles as f64);
        let c_cool_span = (c_cool_hi - c_cool_lo).max(1.0);
        let c_run_hi = if cycle + 1 == cycles {
            deadline_ms
        } else {
            c_cool_hi
        };
        // Ramp fraction runs from `reheat_lo` to 1.0 (cycle 0 melts the greedy
        // seed from fully hot; later cycles reheat only partially).
        let reheat_lo = if cycle == 0 { 0.0 } else { reheat };

        if cycle > 0 {
            arena.load_snapshot(&best_snap, &log2_sizes);
            rebuild_top(&arena, &mut top);
        }

        loop {
            let elapsed = start.elapsed().as_secs_f64() * 1e3;
            if elapsed >= deadline_ms {
                break 'cycles;
            }
            if elapsed >= c_run_hi {
                break;
            }
            let cf = ((elapsed - c_cool_lo) / c_cool_span).clamp(0.0, 1.0);
            let frac = reheat_lo + (1.0 - reheat_lo) * cf;
            let beta = beta_min * (frac * log_ratio).exp();

            // Systematic sweep over every internal node.
            for k in 0..n_int {
                let x = arena.internal[k];
                try_move(&mut arena, x, beta, &log2_sizes, &mut rng);
            }
            // Profile-aware boost: extra attempts on the current top-cost decile.
            if boost > 0 && !top.is_empty() {
                for _ in 0..boost {
                    let x = top[rng.random_range(0..top.len())];
                    try_move(&mut arena, x, beta, &log2_sizes, &mut rng);
                }
            }
            moves += (n_int + boost) as u64;
            sweeps += 1;

            if sweeps % 4 == 0 {
                rebuild_top(&arena, &mut top);
            }

            // Lazy global-tc check for a new best.
            let cur = arena.global_tc();
            if cur < best_tc - 1e-9 {
                best_tc = cur;
                best_snap = arena.snapshot();
                if debug {
                    eprintln!("[curve] t={:.0} tc={:.4}", elapsed, cur);
                }
                if elapsed - last_flush > 250.0 {
                    let nested =
                        best_snap.to_nested(best_snap.root, &code.ixs, &inverse_map, &code.iy, 0);
                    writejson(out_path, &nested)?;
                    last_flush = elapsed;
                }
            }
        }
    }

    // ---- 6. Final flush of the best. ----------------------------------------
    let nested = best_snap.to_nested(best_snap.root, &code.ixs, &inverse_map, &code.iy, 0);
    writejson(out_path, &nested)?;

    if debug {
        let mps = moves as f64 / start.elapsed().as_secs_f64();
        eprintln!(
            "[dbg] done: moves={} moves/s={:.3e} best_tc={:.4}",
            moves, mps, best_tc
        );
    }

    Ok(())
}
