//! Autoresearch attempt 060: activity-based worklist freezing.
//!
//! Contract: `attempt <graph.json> <budget_ms> <out.json>`. This is the
//! attempt-052 simplify/seed/kick/cold-ladder ratchet with one atomic change:
//! cold sweeps revisit only the radius-`d(n)` neighborhood of rewrites accepted
//! by the preceding sweep, with a full refresh every `k(n)` sweeps. Set
//! `ATT_PARENT=1` to recover the parent full-sweep scheduler exactly.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use omeco::expr_tree::{
    apply_rule_mut, tree_complexity, DecompositionType, ExprTree, Rule, ScratchSpace,
};
use omeco::json::writejson;
use omeco::{
    contraction_complexity, optimize_code, simplify, splice, EinCode, GreedyMethod, NestedEinsum,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

const RESYNC_SWEEPS: u64 = 512;
const TARGET_TOP: usize = 30;
const CLOCK_EVERY: u64 = 8;
const WRITE_EVERY_MS: f64 = 150.0;

#[derive(Debug, Deserialize)]
struct GraphData {
    ixs: Vec<Vec<usize>>,
    iy: Vec<usize>,
    sizes: HashMap<String, usize>,
}

#[derive(Debug, Serialize)]
struct ActivitySample {
    t_ms: u64,
    span: usize,
    sweep: u64,
    full: bool,
    attempted: u64,
    accepted: u64,
    active_fraction: f64,
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_flag(key: &str) -> bool {
    std::env::var_os(key).is_some_and(|value| value == "1")
}

fn write_atomic(
    out_path: &str,
    tree: &NestedEinsum<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let tmp = format!("{out_path}.tmp");
    writejson(&tmp, tree)?;
    std::fs::rename(tmp, out_path)?;
    Ok(())
}

fn ceil_log2(n: usize) -> usize {
    if n <= 1 {
        0
    } else {
        usize::BITS as usize - (n - 1).leading_zeros() as usize
    }
}

/// Tree-distance radius: grows by one every four powers of two in reduced n.
fn activity_radius(n: usize) -> usize {
    (ceil_log2(n) / 4).max(2)
}

/// Full-refresh cadence: sqrt(n), bounded by the fixed 80-sweep cold level.
fn refresh_every(n: usize) -> u64 {
    ((n as f64).sqrt().ceil() as u64).clamp(8, 32)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: attempt <graph.json> <budget_ms> <out.json>");
        std::process::exit(2);
    }
    let start = Instant::now();
    let budget_ms: f64 = args[2].parse()?;
    if !budget_ms.is_finite() || budget_ms <= 0.0 {
        return Err("budget_ms must be finite and positive".into());
    }
    let out_path = args[3].clone();
    let graph: GraphData = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let sizes: HashMap<usize, usize> = graph
        .sizes
        .iter()
        .map(|(key, value)| Ok::<_, std::num::ParseIntError>((key.parse()?, *value)))
        .collect::<Result<_, _>>()?;
    let original_code = EinCode::new(graph.ixs.clone(), graph.iy.clone());
    let n_original = original_code.num_tensors();
    if n_original == 0 {
        return Err("empty einsum".into());
    }

    // The current library pipeline simplifies before search and splices every
    // emitted reduced tree back into the original einsum schema.
    let simplified = simplify(&original_code, &sizes);
    let code = simplified.code;
    let subtrees = simplified.subtrees;
    let n = code.num_tensors();
    if n == 0 {
        return Err("simplification produced an empty einsum".into());
    }
    let full_tree = |reduced: &NestedEinsum<usize>| splice(reduced, &subtrees);
    let tc_of =
        |full: &NestedEinsum<usize>| contraction_complexity(full, &sizes, &original_code.ixs).tc;

    if n == 1 {
        let best = full_tree(&NestedEinsum::leaf(0));
        write_atomic(&out_path, &best)?;
        return Ok(());
    }

    // Deterministic greedy first, then the unchanged randomized portfolio.
    let mut best_reduced = optimize_code(&code, &sizes, &GreedyMethod::default())
        .ok_or("greedy optimizer returned no tree")?;
    let mut best = full_tree(&best_reduced);
    let mut best_tc = tc_of(&best);
    write_atomic(&out_path, &best)?;

    let elapsed_ms = || start.elapsed().as_secs_f64() * 1e3;
    let seed_deadline = (budget_ms * 0.04).min(500.0);
    let alphas = [0.0_f64, 0.25, 0.5, 0.75, 1.0];
    let temps = [0.03_f64, 0.1, 0.3];
    let mut combo = 0_usize;
    while elapsed_ms() < seed_deadline {
        let alpha = alphas[combo % alphas.len()];
        let temp = temps[(combo / alphas.len()) % temps.len()];
        combo += 1;
        if let Some(reduced) = optimize_code(&code, &sizes, &GreedyMethod::new(alpha, temp)) {
            let full = full_tree(&reduced);
            let tc = tc_of(&full);
            if tc < best_tc - 1e-9 {
                best_reduced = reduced;
                best = full;
                best_tc = tc;
                write_atomic(&out_path, &best)?;
            }
        }
    }
    eprintln!(
        "t={:.0}ms seed_greedy tc={best_tc:.4} (n={n_original}->{n}, {} greedy trials)",
        elapsed_ms(),
        combo + 1
    );

    if n == 2 {
        write_atomic(&out_path, &best)?;
        return Ok(());
    }
    let labels = code.unique_labels();
    let log2_sizes: Vec<f64> = labels
        .iter()
        .map(|label| (*sizes.get(label).unwrap_or(&1) as f64).log2())
        .collect();
    let Some(seed_expr) = nested_to_expr_tree(&best_reduced, &labels) else {
        write_atomic(&out_path, &best)?;
        return Ok(());
    };

    let mut ann = Annealer {
        original_ixs: &original_code.ixs,
        reduced_ixs: &code.ixs,
        inverse_map: &labels,
        reduced_openedges: &code.iy,
        log2_sizes: &log2_sizes,
        size_dict: &sizes,
        code: &code,
        subtrees: &subtrees,
        scratch: ScratchSpace::new(labels.len()),
        rng: SmallRng::seed_from_u64(0x0000_0052_c0ff_ee00),
        out_path: &out_path,
        start: &start,
        deadline_ms: budget_ms * 0.97,
        best: &mut best,
        best_tc: &mut best_tc,
        parent_mode: env_flag("ATT_PARENT"),
        radius: activity_radius(n),
        refresh_every: refresh_every(n),
        sweeps: 0,
        accepts: 0,
        node_attempts: 0,
        cold_sweeps: 0,
        cold_full_sweeps: 0,
        cold_active_sweeps: 0,
        cold_attempts: 0,
        cold_accepts: 0,
        active_fraction_sum: 0.0,
        activity_samples: Vec::new(),
        last_write_ms: elapsed_ms(),
    };
    ann.run(seed_expr, n)?;
    // Forced final flush: even if the last improvement fell inside the 150 ms
    // write throttle, out.json ends at the in-memory best.
    write_atomic(ann.out_path, ann.best)?;
    ann.print_summary(n_original, n)?;
    Ok(())
}

struct Annealer<'a> {
    original_ixs: &'a [Vec<usize>],
    reduced_ixs: &'a [Vec<usize>],
    inverse_map: &'a [usize],
    reduced_openedges: &'a [usize],
    log2_sizes: &'a [f64],
    size_dict: &'a HashMap<usize, usize>,
    code: &'a EinCode<usize>,
    subtrees: &'a [NestedEinsum<usize>],
    scratch: ScratchSpace,
    rng: SmallRng,
    out_path: &'a str,
    start: &'a Instant,
    deadline_ms: f64,
    best: &'a mut NestedEinsum<usize>,
    best_tc: &'a mut f64,
    parent_mode: bool,
    radius: usize,
    refresh_every: u64,
    sweeps: u64,
    accepts: u64,
    node_attempts: u64,
    cold_sweeps: u64,
    cold_full_sweeps: u64,
    cold_active_sweeps: u64,
    cold_attempts: u64,
    cold_accepts: u64,
    active_fraction_sum: f64,
    activity_samples: Vec<ActivitySample>,
    last_write_ms: f64,
}

impl Annealer<'_> {
    fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1e3
    }

    fn run(&mut self, seed: ExprTree, n: usize) -> Result<(), Box<dyn std::error::Error>> {
        let s_top = ((n + TARGET_TOP - 1) / TARGET_TOP).max(2);
        let mut span_levels = Vec::new();
        let mut span = s_top;
        while span > 2 {
            span_levels.push(span);
            span /= 2;
        }
        span_levels.push(2);

        let b_hi = env_f64("ATT_BHI", 14.0);
        let b_lo_cold = env_f64("ATT_BLO_COLD", 2.5);
        let b_kick_lo = env_f64("ATT_BKICK", 0.05);
        let cold_sweeps = env_u64("ATT_SW_COLD", 80).max(1);
        let kick_sweeps = env_u64("ATT_SW_KICK", 300).max(1);
        let stag_threshold = env_u64("ATT_STAG", if n <= 400 { 2 } else { 5 }).max(1);

        let mut best_expr = seed;
        let mut work_tc = tree_complexity(&best_expr, self.log2_sizes).0;
        let mut vcycle = 0_u64;
        let mut since_improve = 0_u64;
        let mut survived = 0_u64;

        while self.elapsed_ms() < self.deadline_ms {
            vcycle += 1;
            if since_improve >= stag_threshold {
                let alpha = [0.0_f64, 0.25, 0.5, 0.75, 1.0][self.rng.random_range(0..5)];
                let temp = 0.05 + self.rng.random::<f64>() * 0.3;
                let (mut tree, mut s_lin) = self.fresh_expr(alpha, temp).map_or_else(
                    || (best_expr.clone(), f64::exp2(work_tc)),
                    |tree| {
                        let linear = f64::exp2(tree_complexity(&tree, self.log2_sizes).0);
                        (tree, linear)
                    },
                );
                let mut arena = MetaArena::from_tree(&tree);
                since_improve = 0;
                let improved = self.run_level(
                    &mut tree,
                    &mut arena,
                    &mut s_lin,
                    &mut best_expr,
                    &mut work_tc,
                    2,
                    b_kick_lo,
                    b_hi,
                    cold_sweeps,
                    false,
                )?;
                eprintln!(
                    "t={:.0}ms vcycle={vcycle} FLAT-control tc={:.4} improved={improved}",
                    self.elapsed_ms(),
                    self.best_tc
                );
                continue;
            }

            let mut tree = best_expr.clone();
            let mut arena = MetaArena::from_tree(&tree);
            let mut s_lin = f64::exp2(work_tc);
            let tc_incumbent = work_tc;
            self.run_level(
                &mut tree,
                &mut arena,
                &mut s_lin,
                &mut best_expr,
                &mut work_tc,
                s_top,
                b_kick_lo,
                b_hi,
                kick_sweeps,
                false,
            )?;
            let tc_after_kick = tree_complexity(&tree, self.log2_sizes).0;

            let mut cycle_improved = false;
            for &level_span in &span_levels {
                cycle_improved |= self.run_level(
                    &mut tree,
                    &mut arena,
                    &mut s_lin,
                    &mut best_expr,
                    &mut work_tc,
                    level_span,
                    b_lo_cold,
                    b_hi,
                    cold_sweeps,
                    true,
                )?;
                if self.elapsed_ms() >= self.deadline_ms {
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
                tc_incumbent - tc_after_ladder
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

    fn fresh_expr(&self, alpha: f64, temp: f64) -> Option<ExprTree> {
        optimize_code(self.code, self.size_dict, &GreedyMethod::new(alpha, temp))
            .and_then(|tree| nested_to_expr_tree(&tree, self.inverse_map))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_level(
        &mut self,
        tree: &mut ExprTree,
        arena: &mut MetaArena,
        s_lin: &mut f64,
        best_expr: &mut ExprTree,
        work_tc: &mut f64,
        min_span: usize,
        b_lo: f64,
        b_hi: f64,
        sweeps: u64,
        cold: bool,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let mut improved = false;
        let denominator = sweeps.saturating_sub(1).max(1) as f64;
        let mut activity = SparseSet::new(arena.nodes.len());
        let mut closure = SparseSet::new(arena.nodes.len());

        for level_sweep in 0..sweeps {
            self.sweeps += 1;
            let beta = b_lo + (b_hi - b_lo) * (level_sweep as f64 / denominator);
            let full = !cold || self.parent_mode || level_sweep % self.refresh_every == 0;
            let active_fraction = if full {
                1.0
            } else {
                activity.len() as f64 / arena.internal_nodes.max(1) as f64
            };
            let record_below = f64::exp2(*work_tc);
            let stats = if full {
                gated_sweep_full(
                    tree,
                    arena,
                    beta,
                    min_span,
                    self.log2_sizes,
                    &mut self.rng,
                    &mut self.scratch,
                    s_lin,
                    record_below,
                )
            } else {
                gated_sweep_active(
                    tree,
                    arena,
                    &activity,
                    beta,
                    min_span,
                    self.log2_sizes,
                    &mut self.rng,
                    &mut self.scratch,
                    s_lin,
                    record_below,
                    &mut closure,
                )
            };
            arena.expand_activity(&stats.accepted_nodes, self.radius, &mut activity);
            self.accepts += stats.accepted_nodes.len() as u64;
            self.node_attempts += stats.attempted;
            if let Some(best_after) = stats.best_after_accepts {
                let snapshot = rollback_suffix(tree, arena, &stats.moves[best_after..]);
                improved |= self.consider_candidate(&snapshot, best_expr, work_tc)?;
            }

            if cold {
                self.cold_sweeps += 1;
                self.cold_attempts += stats.attempted;
                self.cold_accepts += stats.accepted_nodes.len() as u64;
                if full {
                    self.cold_full_sweeps += 1;
                } else {
                    self.cold_active_sweeps += 1;
                }
                self.active_fraction_sum += active_fraction;
                if level_sweep % CLOCK_EVERY == CLOCK_EVERY - 1 || (full && !self.parent_mode) {
                    self.activity_samples.push(ActivitySample {
                        t_ms: self.elapsed_ms().round() as u64,
                        span: min_span,
                        sweep: level_sweep,
                        full,
                        attempted: stats.attempted,
                        accepted: stats.accepted_nodes.len() as u64,
                        active_fraction,
                    });
                }
            }

            if self.sweeps % RESYNC_SWEEPS == 0 {
                *s_lin = f64::exp2(tree_complexity(tree, self.log2_sizes).0);
            }
            if self.sweeps % CLOCK_EVERY == 0 {
                let now = self.elapsed_ms();
                if now >= self.deadline_ms {
                    break;
                }
            }
        }
        if *s_lin < f64::exp2(*work_tc) - 1e-9 {
            improved |= self.consider_candidate(tree, best_expr, work_tc)?;
        }
        Ok(improved)
    }

    /// Retain every observed minimum immediately in memory. Only serialization
    /// is rate-limited; `main` forces the final pending best to disk.
    fn consider_candidate(
        &mut self,
        tree: &ExprTree,
        best_expr: &mut ExprTree,
        work_tc: &mut f64,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let reduced_tc = tree_complexity(tree, self.log2_sizes).0;
        let reduced = expr_tree_to_nested(
            tree,
            self.reduced_ixs,
            self.inverse_map,
            self.reduced_openedges,
            0,
        );
        let full = splice(&reduced, self.subtrees);
        let exact = contraction_complexity(&full, self.size_dict, self.original_ixs).tc;
        if exact < *self.best_tc - 1e-9 {
            *self.best_tc = exact;
            *work_tc = reduced_tc;
            *best_expr = tree.clone();
            *self.best = full;
            let now = self.elapsed_ms();
            if now - self.last_write_ms >= WRITE_EVERY_MS {
                write_atomic(self.out_path, self.best)?;
                self.last_write_ms = now;
            }
            return Ok(true);
        }
        Ok(false)
    }

    fn print_summary(
        &self,
        n_original: usize,
        n_reduced: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let elapsed_ms = self.elapsed_ms();
        let active_fraction_mean = if self.cold_sweeps == 0 {
            1.0
        } else {
            self.active_fraction_sum / self.cold_sweeps as f64
        };
        let summary = serde_json::json!({
            "mode": if self.parent_mode { "parent" } else { "active" },
            "n_original": n_original,
            "n_reduced": n_reduced,
            "radius": self.radius,
            "refresh_every": self.refresh_every,
            "elapsed_ms": elapsed_ms,
            "tc_final": *self.best_tc,
            "sweeps": self.sweeps,
            "accepts": self.accepts,
            "node_attempts": self.node_attempts,
            "sweeps_per_sec": self.sweeps as f64 * 1000.0 / elapsed_ms,
            "attempts_per_sec": self.node_attempts as f64 * 1000.0 / elapsed_ms,
            "accepts_per_sec": self.accepts as f64 * 1000.0 / elapsed_ms,
            "cold_sweeps": self.cold_sweeps,
            "cold_full_sweeps": self.cold_full_sweeps,
            "cold_active_sweeps": self.cold_active_sweeps,
            "cold_attempts": self.cold_attempts,
            "cold_accepts": self.cold_accepts,
            "active_fraction_mean": active_fraction_mean,
            "activity": self.activity_samples,
        });
        eprintln!("ATT_DIAG {}", serde_json::to_string(&summary)?);
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct MetaNode {
    parent: Option<usize>,
    left: Option<usize>,
    right: Option<usize>,
    span: usize,
}

#[derive(Clone)]
struct MetaArena {
    nodes: Vec<MetaNode>,
    root: usize,
    internal_nodes: usize,
}

impl MetaArena {
    fn from_tree(tree: &ExprTree) -> Self {
        fn build(tree: &ExprTree, parent: Option<usize>, nodes: &mut Vec<MetaNode>) -> usize {
            let id = nodes.len();
            nodes.push(MetaNode {
                parent,
                left: None,
                right: None,
                span: 1,
            });
            if let ExprTree::Node { left, right, .. } = tree {
                let left_id = build(left, Some(id), nodes);
                let right_id = build(right, Some(id), nodes);
                nodes[id].left = Some(left_id);
                nodes[id].right = Some(right_id);
                nodes[id].span = nodes[left_id].span + nodes[right_id].span;
            }
            id
        }
        let mut nodes = Vec::with_capacity(tree.leaf_count().saturating_mul(2).saturating_sub(1));
        let root = build(tree, None, &mut nodes);
        Self {
            nodes,
            root,
            internal_nodes: tree.leaf_count().saturating_sub(1),
        }
    }

    fn is_internal(&self, id: usize) -> bool {
        self.nodes[id].left.is_some()
    }

    fn apply_rule(&mut self, id: usize, rule: Rule) {
        let left = self.nodes[id].left.expect("rule applied to leaf metadata");
        let right = self.nodes[id].right.expect("rule applied to leaf metadata");
        match rule {
            Rule::Rule1 => {
                let moved = self.nodes[left].right.expect("Rule1 requires left node");
                self.nodes[left].right = Some(right);
                self.nodes[right].parent = Some(left);
                self.nodes[id].right = Some(moved);
                self.nodes[moved].parent = Some(id);
                self.nodes[left].span =
                    self.nodes[self.nodes[left].left.unwrap()].span + self.nodes[right].span;
            }
            Rule::Rule2 => {
                let moved = self.nodes[left].left.expect("Rule2 requires left node");
                self.nodes[left].left = Some(right);
                self.nodes[right].parent = Some(left);
                self.nodes[id].right = Some(moved);
                self.nodes[moved].parent = Some(id);
                self.nodes[left].span =
                    self.nodes[right].span + self.nodes[self.nodes[left].right.unwrap()].span;
            }
            Rule::Rule3 => {
                let moved = self.nodes[right].left.expect("Rule3 requires right node");
                self.nodes[id].left = Some(moved);
                self.nodes[moved].parent = Some(id);
                self.nodes[right].left = Some(left);
                self.nodes[left].parent = Some(right);
                self.nodes[right].span =
                    self.nodes[left].span + self.nodes[self.nodes[right].right.unwrap()].span;
            }
            Rule::Rule4 => {
                let moved = self.nodes[right].right.expect("Rule4 requires right node");
                self.nodes[id].left = Some(moved);
                self.nodes[moved].parent = Some(id);
                self.nodes[right].right = Some(left);
                self.nodes[left].parent = Some(right);
                self.nodes[right].span =
                    self.nodes[self.nodes[right].left.unwrap()].span + self.nodes[left].span;
            }
            Rule::Rule5 => {
                self.nodes[id].left = Some(right);
                self.nodes[id].right = Some(left);
            }
        }
    }

    fn expand_activity(&self, accepted: &[usize], radius: usize, active: &mut SparseSet) {
        active.clear();
        let mut queue = VecDeque::new();
        for &id in accepted {
            if active.insert(id) {
                queue.push_back((id, 0_usize));
            }
        }
        while let Some((id, distance)) = queue.pop_front() {
            if distance == radius {
                continue;
            }
            let node = self.nodes[id];
            for neighbor in [node.parent, node.left, node.right].into_iter().flatten() {
                if self.is_internal(neighbor) && active.insert(neighbor) {
                    queue.push_back((neighbor, distance + 1));
                }
            }
        }
    }

    fn active_closure(&self, active: &SparseSet, closure: &mut SparseSet) {
        closure.clear();
        for &id in active.ids() {
            closure.insert(id);
            let mut current = self.nodes[id].parent;
            while let Some(parent) = current {
                if !closure.insert(parent) {
                    break;
                }
                current = self.nodes[parent].parent;
            }
        }
    }

    fn path_to(&self, mut id: usize) -> Vec<bool> {
        let mut reversed = Vec::new();
        while let Some(parent) = self.nodes[id].parent {
            reversed.push(self.nodes[parent].right == Some(id));
            id = parent;
        }
        reversed.reverse();
        reversed
    }
}

/// Generation-stamped sparse membership set. Clearing and iteration are
/// O(active), while membership remains O(1); allocation is once per level.
struct SparseSet {
    marks: Vec<u64>,
    generation: u64,
    ids: Vec<usize>,
}

impl SparseSet {
    fn new(capacity: usize) -> Self {
        Self {
            marks: vec![0; capacity],
            generation: 1,
            ids: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.ids.clear();
        self.generation = self.generation.checked_add(1).unwrap_or_else(|| {
            self.marks.fill(0);
            1
        });
    }

    fn insert(&mut self, id: usize) -> bool {
        if self.contains(id) {
            return false;
        }
        self.marks[id] = self.generation;
        self.ids.push(id);
        true
    }

    fn contains(&self, id: usize) -> bool {
        self.marks[id] == self.generation
    }

    fn ids(&self) -> &[usize] {
        &self.ids
    }

    fn len(&self) -> usize {
        self.ids.len()
    }
}

struct SweepStats {
    attempted: u64,
    accepted_nodes: Vec<usize>,
    moves: Vec<AcceptedMove>,
    best_linear: f64,
    best_after_accepts: Option<usize>,
}

struct AcceptedMove {
    node: usize,
    rule: Rule,
    old_labels: Vec<usize>,
}

#[allow(clippy::too_many_arguments)]
fn gated_sweep_full(
    tree: &mut ExprTree,
    arena: &mut MetaArena,
    beta: f64,
    min_span: usize,
    log2_sizes: &[f64],
    rng: &mut SmallRng,
    scratch: &mut ScratchSpace,
    s_lin: &mut f64,
    record_below: f64,
) -> SweepStats {
    let mut stats = SweepStats {
        attempted: 0,
        accepted_nodes: Vec::new(),
        moves: Vec::new(),
        best_linear: record_below,
        best_after_accepts: None,
    };
    sweep_inner(
        tree, arena.root, arena, None, None, beta, min_span, log2_sizes, rng, scratch, s_lin,
        &mut stats,
    );
    stats
}

#[allow(clippy::too_many_arguments)]
fn gated_sweep_active(
    tree: &mut ExprTree,
    arena: &mut MetaArena,
    active: &SparseSet,
    beta: f64,
    min_span: usize,
    log2_sizes: &[f64],
    rng: &mut SmallRng,
    scratch: &mut ScratchSpace,
    s_lin: &mut f64,
    record_below: f64,
    closure: &mut SparseSet,
) -> SweepStats {
    arena.active_closure(active, closure);
    let mut stats = SweepStats {
        attempted: 0,
        accepted_nodes: Vec::new(),
        moves: Vec::new(),
        best_linear: record_below,
        best_after_accepts: None,
    };
    sweep_inner(
        tree,
        arena.root,
        arena,
        Some(active),
        Some(closure),
        beta,
        min_span,
        log2_sizes,
        rng,
        scratch,
        s_lin,
        &mut stats,
    );
    stats
}

#[allow(clippy::too_many_arguments)]
fn sweep_inner(
    tree: &mut ExprTree,
    meta_id: usize,
    arena: &mut MetaArena,
    active: Option<&SparseSet>,
    closure: Option<&SparseSet>,
    beta: f64,
    min_span: usize,
    log2_sizes: &[f64],
    rng: &mut SmallRng,
    scratch: &mut ScratchSpace,
    s_lin: &mut f64,
    stats: &mut SweepStats,
) -> usize {
    if closure.is_some_and(|set| !set.contains(meta_id)) {
        return arena.nodes[meta_id].span;
    }
    let ExprTree::Node { left, right, .. } = tree else {
        return 1;
    };
    let left_id = arena.nodes[meta_id].left.unwrap();
    let right_id = arena.nodes[meta_id].right.unwrap();
    let left_span = sweep_inner(
        left, left_id, arena, active, closure, beta, min_span, log2_sizes, rng, scratch, s_lin,
        stats,
    );
    let right_span = sweep_inner(
        right, right_id, arena, active, closure, beta, min_span, log2_sizes, rng, scratch, s_lin,
        stats,
    );
    let span = left_span + right_span;
    debug_assert_eq!(span, arena.nodes[meta_id].span);
    if span >= min_span && active.map_or(true, |set| set.contains(meta_id)) {
        let rules = Rule::applicable_rules(tree, DecompositionType::Tree);
        if !rules.is_empty() {
            stats.attempted += 1;
            let rule = rules[rng.random_range(0..rules.len())];
            if let Some(diff) = scratch.rule_diff(tree, rule, log2_sizes, false) {
                let dtc = diff.tc1 - diff.tc0;
                if dtc <= 0.0 || rng.random::<f64>() < (-beta * dtc).exp() {
                    let old_labels = changed_child_labels(tree, rule);
                    *s_lin += f64::exp2(diff.tc1) - f64::exp2(diff.tc0);
                    apply_rule_mut(tree, rule, diff.new_labels);
                    arena.apply_rule(meta_id, rule);
                    stats.accepted_nodes.push(meta_id);
                    stats.moves.push(AcceptedMove {
                        node: meta_id,
                        rule,
                        old_labels,
                    });
                    if *s_lin < stats.best_linear - 1e-9 {
                        stats.best_linear = *s_lin;
                        stats.best_after_accepts = Some(stats.moves.len());
                    }
                }
            }
        }
    }
    span
}

fn changed_child_labels(tree: &ExprTree, rule: Rule) -> Vec<usize> {
    let ExprTree::Node { left, right, info } = tree else {
        return Vec::new();
    };
    match rule {
        Rule::Rule1 | Rule::Rule2 => left.info().out_dims.clone(),
        Rule::Rule3 | Rule::Rule4 => right.info().out_dims.clone(),
        Rule::Rule5 => info.out_dims.clone(),
    }
}

fn subtree_at_path_mut<'a>(mut tree: &'a mut ExprTree, path: &[bool]) -> &'a mut ExprTree {
    for &right_child in path {
        let ExprTree::Node { left, right, .. } = tree else {
            panic!("metadata path entered a leaf");
        };
        tree = if right_child { right } else { left };
    }
    tree
}

/// Reconstruct the exact whole-tree state at a transient minimum by undoing the
/// accepted-move suffix. Rules 1--4 are involutions; `old_labels` restores the
/// one intermediate interface each rule changed.
fn rollback_suffix(tree: &ExprTree, arena: &MetaArena, suffix: &[AcceptedMove]) -> ExprTree {
    let mut snapshot = tree.clone();
    let mut snapshot_arena = arena.clone();
    for accepted in suffix.iter().rev() {
        let path = snapshot_arena.path_to(accepted.node);
        apply_rule_mut(
            subtree_at_path_mut(&mut snapshot, &path),
            accepted.rule,
            accepted.old_labels.clone(),
        );
        snapshot_arena.apply_rule(accepted.node, accepted.rule);
    }
    snapshot
}

fn nested_to_expr_tree(nested: &NestedEinsum<usize>, inverse_map: &[usize]) -> Option<ExprTree> {
    let label_map: HashMap<usize, usize> = inverse_map
        .iter()
        .enumerate()
        .map(|(index, label)| (*label, index))
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
            let child = |nested: &NestedEinsum<usize>, labels: &[usize]| match nested {
                NestedEinsum::Leaf { tensor_index } => Some(ExprTree::leaf(
                    labels
                        .iter()
                        .filter_map(|label| label_map.get(label).copied())
                        .collect(),
                    *tensor_index,
                )),
                NestedEinsum::Node { .. } => nested_to_expr_tree_inner(nested, label_map),
            };
            let left = child(&args[0], &eins.ixs[0])?;
            let right = child(&args[1], &eins.ixs[1])?;
            let out_dims = eins
                .iy
                .iter()
                .filter_map(|label| label_map.get(label).copied())
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
            let iy = if level == 0 {
                openedges.to_vec()
            } else {
                info.out_dims
                    .iter()
                    .map(|index| inverse_map[*index])
                    .collect()
            };
            let eins = EinCode::new(
                vec![
                    child_labels(&left_nested, original_ixs),
                    child_labels(&right_nested, original_ixs),
                ],
                iy,
            );
            NestedEinsum::node(vec![left_nested, right_nested], eins)
        }
    }
}

fn child_labels(nested: &NestedEinsum<usize>, original_ixs: &[Vec<usize>]) -> Vec<usize> {
    match nested {
        NestedEinsum::Leaf { tensor_index } => original_ixs[*tensor_index].clone(),
        NestedEinsum::Node { eins, .. } => eins.iy.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(id: usize) -> ExprTree {
        ExprTree::leaf(Vec::new(), id)
    }

    fn node(left: ExprTree, right: ExprTree) -> ExprTree {
        ExprTree::node(left, right, Vec::new())
    }

    fn validate_arena(tree: &ExprTree, arena: &MetaArena) {
        fn visit(
            tree: &ExprTree,
            id: usize,
            parent: Option<usize>,
            arena: &MetaArena,
            seen: &mut [bool],
        ) -> usize {
            assert!(!seen[id], "metadata node visited twice");
            seen[id] = true;
            let meta = arena.nodes[id];
            assert_eq!(meta.parent, parent);
            let span = match tree {
                ExprTree::Leaf(_) => {
                    assert_eq!((meta.left, meta.right), (None, None));
                    1
                }
                ExprTree::Node { left, right, .. } => {
                    let left_id = meta.left.expect("missing metadata left child");
                    let right_id = meta.right.expect("missing metadata right child");
                    visit(left, left_id, Some(id), arena, seen)
                        + visit(right, right_id, Some(id), arena, seen)
                }
            };
            assert_eq!(meta.span, span);
            span
        }

        let mut seen = vec![false; arena.nodes.len()];
        assert_eq!(
            visit(tree, arena.root, None, arena, &mut seen),
            tree.leaf_count()
        );
        assert!(seen.into_iter().all(|value| value));
    }

    #[test]
    fn arena_tracks_all_tree_rules() {
        for rule in [Rule::Rule1, Rule::Rule2] {
            let mut tree = node(node(leaf(0), leaf(1)), leaf(2));
            let mut arena = MetaArena::from_tree(&tree);
            apply_rule_mut(&mut tree, rule, Vec::new());
            arena.apply_rule(arena.root, rule);
            validate_arena(&tree, &arena);
        }
        for rule in [Rule::Rule3, Rule::Rule4] {
            let mut tree = node(leaf(0), node(leaf(1), leaf(2)));
            let mut arena = MetaArena::from_tree(&tree);
            apply_rule_mut(&mut tree, rule, Vec::new());
            arena.apply_rule(arena.root, rule);
            validate_arena(&tree, &arena);
        }
    }

    #[test]
    fn activity_expands_by_exact_tree_distance() {
        let tree = node(node(leaf(0), leaf(1)), node(leaf(2), leaf(3)));
        let arena = MetaArena::from_tree(&tree);
        let left = arena.nodes[arena.root].left.unwrap();
        let right = arena.nodes[arena.root].right.unwrap();
        let mut active = SparseSet::new(arena.nodes.len());

        arena.expand_activity(&[left], 1, &mut active);
        assert!(active.contains(left));
        assert!(active.contains(arena.root));
        assert!(!active.contains(right));
        assert_eq!(active.len(), 2);

        arena.expand_activity(&[left], 2, &mut active);
        assert!(active.contains(right));
        assert_eq!(active.len(), 3);
    }

    #[test]
    fn rollback_reconstructs_whole_tree_before_move_suffix() {
        let mut tree = node(node(leaf(0), leaf(1)), node(leaf(2), leaf(3)));
        let original = format!("{tree:?}");
        let mut arena = MetaArena::from_tree(&tree);
        let mut moves = Vec::new();
        for rule in [Rule::Rule1, Rule::Rule2] {
            let old_labels = changed_child_labels(&tree, rule);
            apply_rule_mut(&mut tree, rule, Vec::new());
            arena.apply_rule(arena.root, rule);
            moves.push(AcceptedMove {
                node: arena.root,
                rule,
                old_labels,
            });
        }

        let restored = rollback_suffix(&tree, &arena, &moves);
        assert_eq!(format!("{restored:?}"), original);
    }

    #[test]
    fn all_active_matches_full_sweep_and_rng() {
        let tree = node(node(leaf(0), leaf(1)), node(leaf(2), leaf(3)));
        let mut full_tree = tree.clone();
        let mut active_tree = tree;
        let mut full_arena = MetaArena::from_tree(&full_tree);
        let mut active_arena = MetaArena::from_tree(&active_tree);
        let mut active = SparseSet::new(active_arena.nodes.len());
        for id in 0..active_arena.nodes.len() {
            if active_arena.is_internal(id) {
                active.insert(id);
            }
        }
        let mut closure = SparseSet::new(active_arena.nodes.len());
        let mut full_rng = SmallRng::seed_from_u64(17);
        let mut active_rng = SmallRng::seed_from_u64(17);
        let mut full_scratch = ScratchSpace::new(0);
        let mut active_scratch = ScratchSpace::new(0);
        let mut full_linear = f64::exp2(tree_complexity(&full_tree, &[]).0);
        let mut active_linear = full_linear;

        let full_stats = gated_sweep_full(
            &mut full_tree,
            &mut full_arena,
            1.0,
            2,
            &[],
            &mut full_rng,
            &mut full_scratch,
            &mut full_linear,
            f64::NEG_INFINITY,
        );
        let active_stats = gated_sweep_active(
            &mut active_tree,
            &mut active_arena,
            &active,
            1.0,
            2,
            &[],
            &mut active_rng,
            &mut active_scratch,
            &mut active_linear,
            f64::NEG_INFINITY,
            &mut closure,
        );

        assert_eq!(format!("{full_tree:?}"), format!("{active_tree:?}"));
        assert_eq!(full_linear.to_bits(), active_linear.to_bits());
        assert_eq!(full_stats.attempted, active_stats.attempted);
        assert_eq!(full_stats.accepted_nodes, active_stats.accepted_nodes);
        assert_eq!(full_rng.random::<u64>(), active_rng.random::<u64>());
        validate_arena(&full_tree, &full_arena);
        validate_arena(&active_tree, &active_arena);
    }
}
