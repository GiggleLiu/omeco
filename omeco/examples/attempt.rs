//! Attempt 029 — structure-seeded refinement at non-converged scale.
//!
//! Contract: `attempt <graph.json> <budget_ms> <out.json>` — read an einsum
//! graph, search for a contraction order within the wall-clock budget, and
//! write the best tree found in `writejson` format before the deadline.
//!
//! Pipeline:
//!  1. GreedyMethod seed, written immediately so a valid tree always exists.
//!  2. Structured seed by instance shape:
//!     - RQC lattice (rank-1 boundary + rank-4 gates): contraction-front
//!       (MPS spacetime-sweep) trees over several BFS orders (reuse of the
//!       sibling-019 machinery), best-by-tc kept.
//!     - 3-regular / generic large graph: recursive FM min-cut bisection with
//!       greedy leaves (reuse of the sibling-005 machinery).
//!     The best seed by tc is adopted.
//!  3. Continuous simulated-annealing refinement OF THE SEED, built directly
//!     on omeco's public `expr_tree` primitives (rule-diff incremental deltas,
//!     no rebuild-from-greedy restarts). Pure-tc objective (sc unbounded).
//!     Global best-by-true-tc kept; anytime atomic writes; reheat cycles fill
//!     the full budget.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use omeco::expr_tree::{
    apply_rule_mut, tree_complexity, DecompositionType, ExprTree, Rule, ScratchSpace,
};
use omeco::json::writejson;
use omeco::{
    contraction_complexity, optimize_code, EinCode, ExhaustiveSearch, GreedyMethod, NestedEinsum,
    TreeSA,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GraphData {
    #[serde(default)]
    name: String,
    ixs: Vec<Vec<usize>>,
    iy: Vec<usize>,
    sizes: HashMap<String, usize>,
}

// ---------------------------------------------------------------------------
// Small deterministic xorshift RNG (single-threaded, no external deps).
// ---------------------------------------------------------------------------
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x2545_F491_4F6C_DD1D)
            | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % (n as u64)) as usize
        }
    }
    /// Uniform f64 in [0, 1).
    fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Atomic anytime write: serialize to a sibling temp file then rename.
fn atomic_write(
    out_path: &str,
    tree: &NestedEinsum<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let tmp = format!("{out_path}.tmp");
    writejson(&tmp, tree)?;
    std::fs::rename(&tmp, out_path)?;
    Ok(())
}

/// Deduplicate a tensor's labels while preserving first-seen order.
fn dedup(labels: &[usize]) -> Vec<usize> {
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(labels.len());
    for &l in labels {
        if seen.insert(l) {
            out.push(l);
        }
    }
    out
}

// ===========================================================================
// Structured contraction-front (MPS spacetime-sweep) seeds — sibling 019.
// ===========================================================================

/// Tensor adjacency graph: tensors sharing an index label are neighbours.
fn build_adjacency(ixs: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut label_to_tensors: HashMap<usize, Vec<usize>> = HashMap::new();
    for (ti, ix) in ixs.iter().enumerate() {
        for &l in &dedup(ix) {
            label_to_tensors.entry(l).or_default().push(ti);
        }
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); ixs.len()];
    for tensors in label_to_tensors.values() {
        for i in 0..tensors.len() {
            for j in (i + 1)..tensors.len() {
                let (a, b) = (tensors[i], tensors[j]);
                adj[a].push(b);
                adj[b].push(a);
            }
        }
    }
    for nb in adj.iter_mut() {
        nb.sort_unstable();
        nb.dedup();
    }
    adj
}

/// BFS distance from a single source over the tensor adjacency graph.
fn bfs_dist(n: usize, adj: &[Vec<usize>], source: usize) -> Vec<usize> {
    let mut dist = vec![usize::MAX; n];
    let mut queue = VecDeque::new();
    dist[source] = 0;
    queue.push_back(source);
    while let Some(u) = queue.pop_front() {
        for &v in &adj[u] {
            if dist[v] == usize::MAX {
                dist[v] = dist[u] + 1;
                queue.push_back(v);
            }
        }
    }
    dist
}

/// BFS ordering from a set of source tensors (increasing distance, discovery
/// order ties). Disconnected tensors are appended in index order.
fn bfs_order(n: usize, adj: &[Vec<usize>], sources: &[usize]) -> Vec<usize> {
    let mut dist = vec![usize::MAX; n];
    let mut queue = VecDeque::new();
    for &s in sources {
        if dist[s] == usize::MAX {
            dist[s] = 0;
            queue.push_back(s);
        }
    }
    let mut order = Vec::with_capacity(n);
    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &v in &adj[u] {
            if dist[v] == usize::MAX {
                dist[v] = dist[u] + 1;
                queue.push_back(v);
            }
        }
    }
    for (t, &d) in dist.iter().enumerate() {
        if d == usize::MAX {
            order.push(t);
        }
    }
    order
}

/// Build a BALANCED binary contraction tree over a tensor order by recursively
/// halving the order and merging the two halves. Depth is log(n) (unlike the
/// caterpillar front tree), which keeps the intermediate-label sets small on
/// most nodes and makes the tree cheap to anneal, while contiguous halves keep
/// the sweep's spatial-locality structure. Returns the covering `Item`.
fn build_balanced(ctx: &Ctx, order: &[usize]) -> Item {
    if order.len() == 1 {
        return ctx.item_from_tensor(order[0]);
    }
    let mid = order.len() / 2;
    let left = build_balanced(ctx, &order[..mid]);
    let right = build_balanced(ctx, &order[mid..]);
    merge_two(ctx, &left, &right)
}

/// Build a BLOCKED "front" tree along a tensor order: split the order into
/// contiguous blocks of `w` tensors, build a balanced tree inside each block,
/// then combine the blocks left-to-right into a running front. This keeps the
/// low-sc MPS spacetime-sweep structure (the front carries the spatial cut)
/// while capping the height at ~ n/w + log2(w) — a plain caterpillar (w = 1) is
/// ~n deep, which would blow the validator's JSON-parse recursion limit.
fn build_blocked_front(ctx: &Ctx, order: &[usize], w: usize) -> Item {
    let w = w.max(1);
    let mut blocks = order.chunks(w).map(|c| build_balanced(ctx, c));
    let mut cur = blocks.next().expect("non-empty order");
    for b in blocks {
        cur = merge_two(ctx, &cur, &b);
    }
    cur
}

/// Height (longest root-to-leaf path) of a nested tree; used to keep the output
/// under the validator's recursive-JSON-parse depth limit.
fn nested_height(t: &NestedEinsum<usize>) -> usize {
    match t {
        NestedEinsum::Leaf { .. } => 1,
        NestedEinsum::Node { args, .. } => {
            1 + args.iter().map(nested_height).max().unwrap_or(0)
        }
    }
}

/// Build the best structured seed (by tc) for a layered-circuit instance: a
/// balanced binary tree over several candidate spacetime-sweep orders. Returns
/// `None` if the instance is not a layered circuit.
fn structured_seed(
    ctx: &Ctx,
    ixs: &[Vec<usize>],
    _iy: &[usize],
    sizes: &HashMap<usize, usize>,
    code: &EinCode<usize>,
) -> Option<(NestedEinsum<usize>, f64)> {
    let n = ixs.len();
    let boundary: Vec<usize> = (0..n).filter(|&t| dedup(&ixs[t]).len() == 1).collect();
    let gates: Vec<usize> = (0..n).filter(|&t| dedup(&ixs[t]).len() == 4).collect();
    if boundary.len() < 4 || gates.len() < 4 {
        return None;
    }

    let adj = build_adjacency(ixs);

    let mut orders: Vec<Vec<usize>> = Vec::new();
    // (a) flat temporal sweeps from each temporal end.
    {
        let dist0 = bfs_dist(n, &adj, boundary[0]);
        let max_bd = boundary
            .iter()
            .map(|&b| dist0[b])
            .filter(|&d| d != usize::MAX)
            .max()
            .unwrap_or(0);
        let thresh = max_bd / 2;
        let near: Vec<usize> = boundary
            .iter()
            .copied()
            .filter(|&b| dist0[b] != usize::MAX && dist0[b] <= thresh)
            .collect();
        let far: Vec<usize> = boundary
            .iter()
            .copied()
            .filter(|&b| dist0[b] != usize::MAX && dist0[b] > thresh)
            .collect();
        if !near.is_empty() {
            orders.push(bfs_order(n, &adj, &near));
        }
        if !far.is_empty() {
            orders.push(bfs_order(n, &adj, &far));
        }
    }
    // (b) bidirectional multi-source sweep.
    orders.push(bfs_order(n, &adj, &boundary));
    // (c) directed cone sweeps from single sources spread across the boundary.
    let n_sources = boundary.len().min(6);
    for k in 0..n_sources {
        let src = boundary[k * boundary.len() / n_sources];
        orders.push(bfs_order(n, &adj, &[src]));
    }
    // reverses (front swept from the opposite side)
    let mut reversed: Vec<Vec<usize>> = orders
        .iter()
        .map(|o| {
            let mut r = o.clone();
            r.reverse();
            r
        })
        .collect();
    orders.append(&mut reversed);

    // Block width for the blocked-front seed: chosen so the tree height
    // (~ n/w + log2 w) stays well under the validator's JSON recursion limit
    // (height ≲ 470 ⇒ safe). w ≥ n/300 keeps n/w ≲ 300.
    let n_over = (n / 300) + 1;
    let widths = [n_over.max(4), n_over.max(8), n_over.max(16)];

    let mut best: Option<(NestedEinsum<usize>, f64)> = None;
    for order in &orders {
        // Blocked MPS fronts at a few block widths, plus the balanced tree;
        // keep the lowest-tc structured tree overall. All are shallow enough to
        // serialise safely.
        let mut candidates: Vec<NestedEinsum<usize>> =
            widths.iter().map(|&w| build_blocked_front(ctx, order, w).tree).collect();
        candidates.push(build_balanced(ctx, order).tree);
        for tree in candidates {
            let cc = contraction_complexity(&tree, sizes, &code.ixs);
            if !cc.tc.is_finite() {
                continue;
            }
            if best.as_ref().map(|(_, t)| cc.tc < *t).unwrap_or(true) {
                best = Some((tree, cc.tc));
            }
        }
    }
    best
}

// ===========================================================================
// Recursive FM min-cut bisection seed — sibling 005.
// ===========================================================================

const BASE_THRESHOLD: usize = 12;

#[derive(Clone)]
struct Item {
    cover: Vec<usize>,
    labels: Vec<usize>,
    tree: NestedEinsum<usize>,
}

struct Ctx {
    ixs: Vec<Vec<usize>>,
    log2: HashMap<usize, f64>,
    global_count: HashMap<usize, u32>,
    iy_set: HashSet<usize>,
}

impl Ctx {
    fn log2_of(&self, l: usize) -> f64 {
        self.log2.get(&l).copied().unwrap_or(0.0)
    }
    fn open_labels(&self, cover: &[usize]) -> Vec<usize> {
        let mut local: HashMap<usize, u32> = HashMap::new();
        for &t in cover {
            let mut seen = HashSet::new();
            for &l in &self.ixs[t] {
                if seen.insert(l) {
                    *local.entry(l).or_insert(0) += 1;
                }
            }
        }
        let mut out: Vec<usize> = Vec::new();
        for (&l, &c) in &local {
            let g = self.global_count.get(&l).copied().unwrap_or(c);
            if self.iy_set.contains(&l) || g > c {
                out.push(l);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
    fn item_from_tensor(&self, t: usize) -> Item {
        let mut labels = self.ixs[t].clone();
        labels.sort_unstable();
        labels.dedup();
        Item {
            cover: vec![t],
            labels,
            tree: NestedEinsum::leaf(t),
        }
    }
}

fn splice(tree: &NestedEinsum<usize>, items: &[Item]) -> NestedEinsum<usize> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => items[*tensor_index].tree.clone(),
        NestedEinsum::Node { args, eins } => {
            let new_args = args.iter().map(|a| splice(a, items)).collect();
            NestedEinsum::node(new_args, eins.clone())
        }
    }
}

fn solve_base(ctx: &Ctx, items: &[Item], sizes: &HashMap<usize, usize>) -> NestedEinsum<usize> {
    if items.len() == 1 {
        return items[0].tree.clone();
    }
    let local_ixs: Vec<Vec<usize>> = items.iter().map(|it| it.labels.clone()).collect();
    let mut cover: Vec<usize> = Vec::new();
    for it in items {
        cover.extend_from_slice(&it.cover);
    }
    let local_iy = ctx.open_labels(&cover);
    let code = EinCode::new(local_ixs, local_iy);
    let local_tree = optimize_code(&code, sizes, &ExhaustiveSearch::default())
        .or_else(|| optimize_code(&code, sizes, &GreedyMethod::default()));
    match local_tree {
        Some(t) => splice(&t, items),
        None => fold_left(ctx, items),
    }
}

fn fold_left(ctx: &Ctx, items: &[Item]) -> NestedEinsum<usize> {
    let mut cur = items[0].clone();
    for it in &items[1..] {
        cur = merge_two(ctx, &cur, it);
    }
    cur.tree
}

fn merge_two(ctx: &Ctx, a: &Item, b: &Item) -> Item {
    let mut cover = a.cover.clone();
    cover.extend_from_slice(&b.cover);
    cover.sort_unstable();
    cover.dedup();
    let out = ctx.open_labels(&cover);
    let eins = EinCode::new(vec![a.labels.clone(), b.labels.clone()], out.clone());
    let tree = NestedEinsum::node(vec![a.tree.clone(), b.tree.clone()], eins);
    Item {
        cover,
        labels: out,
        tree,
    }
}

fn bipartition(ctx: &Ctx, items: &[Item], rng: &mut Rng) -> Vec<bool> {
    let n = items.len();
    let min_side = (((n as f64) * 0.35).floor() as usize).max(1);
    let mut side = bfs_initial(ctx, items, rng, min_side);

    let mut count_a: HashMap<usize, u32> = HashMap::new();
    let mut count_b: HashMap<usize, u32> = HashMap::new();
    let rebuild =
        |side: &[bool], count_a: &mut HashMap<usize, u32>, count_b: &mut HashMap<usize, u32>| {
            count_a.clear();
            count_b.clear();
            for (i, it) in items.iter().enumerate() {
                let tgt = if side[i] { &mut *count_b } else { &mut *count_a };
                for &l in &it.labels {
                    *tgt.entry(l).or_insert(0) += 1;
                }
            }
        };
    rebuild(&side, &mut count_a, &mut count_b);

    let cut_of = |ca: &HashMap<usize, u32>, cb: &HashMap<usize, u32>| -> f64 {
        let mut s = 0.0;
        for (&l, &a) in ca {
            if a > 0 && cb.get(&l).copied().unwrap_or(0) > 0 {
                s += ctx.log2_of(l);
            }
        }
        s
    };

    for _pass in 0..6 {
        let mut improved = false;
        let mut size_a = side.iter().filter(|&&s| !s).count();
        let mut size_b = n - size_a;
        let mut locked = vec![false; n];
        let base_cut = cut_of(&count_a, &count_b);
        let mut best_cut = base_cut;
        let mut best_side = side.clone();
        let mut running_cut = base_cut;

        for _step in 0..n {
            let mut best_gain = f64::NEG_INFINITY;
            let mut best_i: Option<usize> = None;
            for i in 0..n {
                if locked[i] {
                    continue;
                }
                if side[i] {
                    if size_b <= min_side {
                        continue;
                    }
                } else if size_a <= min_side {
                    continue;
                }
                let g = move_gain(ctx, &items[i], side[i], &count_a, &count_b);
                if g > best_gain {
                    best_gain = g;
                    best_i = Some(i);
                }
            }
            let Some(i) = best_i else { break };
            apply_move(&items[i], side[i], &mut count_a, &mut count_b);
            if side[i] {
                size_b -= 1;
                size_a += 1;
            } else {
                size_a -= 1;
                size_b += 1;
            }
            side[i] = !side[i];
            locked[i] = true;
            running_cut -= best_gain;
            if running_cut < best_cut - 1e-12 {
                best_cut = running_cut;
                best_side = side.clone();
                improved = true;
            }
        }
        side = best_side;
        rebuild(&side, &mut count_a, &mut count_b);
        if !improved {
            break;
        }
    }

    let sa = side.iter().filter(|&&s| !s).count();
    if sa == 0 || sa == n {
        for (i, s) in side.iter_mut().enumerate() {
            *s = i >= n / 2;
        }
    }
    side
}

fn move_gain(
    ctx: &Ctx,
    item: &Item,
    on_b: bool,
    count_a: &HashMap<usize, u32>,
    count_b: &HashMap<usize, u32>,
) -> f64 {
    let mut gain = 0.0;
    for &l in &item.labels {
        let a = count_a.get(&l).copied().unwrap_or(0);
        let b = count_b.get(&l).copied().unwrap_or(0);
        let (from, to) = if on_b { (b, a) } else { (a, b) };
        let crossing_before = from > 0 && to > 0;
        let from_after = from.saturating_sub(1);
        let to_after = to + 1;
        let crossing_after = from_after > 0 && to_after > 0;
        let w = ctx.log2_of(l);
        if crossing_before {
            gain += w;
        }
        if crossing_after {
            gain -= w;
        }
    }
    gain
}

fn apply_move(
    item: &Item,
    on_b: bool,
    count_a: &mut HashMap<usize, u32>,
    count_b: &mut HashMap<usize, u32>,
) {
    for &l in &item.labels {
        if on_b {
            if let Some(c) = count_b.get_mut(&l) {
                *c = c.saturating_sub(1);
            }
            *count_a.entry(l).or_insert(0) += 1;
        } else {
            if let Some(c) = count_a.get_mut(&l) {
                *c = c.saturating_sub(1);
            }
            *count_b.entry(l).or_insert(0) += 1;
        }
    }
}

fn bfs_initial(ctx: &Ctx, items: &[Item], rng: &mut Rng, min_side: usize) -> Vec<bool> {
    let n = items.len();
    let target = n / 2;
    let mut side = vec![false; n];

    let mut label_items: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, it) in items.iter().enumerate() {
        for &l in &it.labels {
            label_items.entry(l).or_default().push(i);
        }
    }

    let seed = rng.below(n);
    side[seed] = true;
    let mut in_b = 1usize;

    let mut connection = vec![0.0f64; n];
    for &l in &items[seed].labels {
        let w = ctx.log2_of(l);
        if let Some(neis) = label_items.get(&l) {
            for &j in neis {
                connection[j] += w;
            }
        }
    }

    while in_b < target {
        let mut best_j: Option<usize> = None;
        let mut best_c = f64::NEG_INFINITY;
        for j in 0..n {
            if side[j] {
                continue;
            }
            let c = connection[j];
            if c > best_c || (c == best_c && rng.below(2) == 0) {
                best_c = c;
                best_j = Some(j);
            }
        }
        let Some(j) = best_j else { break };
        side[j] = true;
        in_b += 1;
        for &l in &items[j].labels {
            let w = ctx.log2_of(l);
            if let Some(neis) = label_items.get(&l) {
                for &k in neis {
                    connection[k] += w;
                }
            }
        }
    }

    let mut size_b = in_b;
    let mut size_a = n - in_b;
    let mut idx = 0;
    while size_b < min_side && idx < n {
        if !side[idx] {
            side[idx] = true;
            size_b += 1;
            size_a -= 1;
        }
        idx += 1;
    }
    idx = 0;
    while size_a < min_side && idx < n {
        if side[idx] {
            side[idx] = false;
            size_a += 1;
            size_b = size_b.saturating_sub(1);
        }
        idx += 1;
    }
    let _ = size_b;
    side
}

fn solve(
    ctx: &Ctx,
    items: Vec<Item>,
    rng: &mut Rng,
    depth: usize,
    sizes: &HashMap<usize, usize>,
) -> NestedEinsum<usize> {
    if items.len() == 1 {
        return items.into_iter().next().unwrap().tree;
    }
    if items.len() <= BASE_THRESHOLD {
        return solve_base(ctx, &items, sizes);
    }
    if depth > 4 * items.len() + 64 {
        return solve_base(ctx, &items, sizes);
    }

    let side = bipartition(ctx, &items, rng);
    let mut side_a: Vec<Item> = Vec::new();
    let mut side_b: Vec<Item> = Vec::new();
    for (i, it) in items.into_iter().enumerate() {
        if side[i] {
            side_b.push(it);
        } else {
            side_a.push(it);
        }
    }

    let (child, parent) = if side_a.len() >= 2 && side_b.len() >= 2 {
        if side_a.len() <= side_b.len() {
            (side_a, side_b)
        } else {
            (side_b, side_a)
        }
    } else if side_a.len() >= 2 {
        (side_a, side_b)
    } else {
        (side_b, side_a)
    };

    let child_tree = solve(ctx, child.clone(), rng, depth + 1, sizes);
    let mut child_cover: Vec<usize> = Vec::new();
    for it in &child {
        child_cover.extend_from_slice(&it.cover);
    }
    child_cover.sort_unstable();
    child_cover.dedup();
    let child_labels = ctx.open_labels(&child_cover);
    let child_item = Item {
        cover: child_cover,
        labels: child_labels,
        tree: child_tree,
    };

    let mut next = parent;
    next.push(child_item);
    solve(ctx, next, rng, depth + 1, sizes)
}

fn build_ctx(graph: &GraphData, sizes: &HashMap<usize, usize>) -> Ctx {
    let log2: HashMap<usize, f64> = sizes
        .iter()
        .map(|(&k, &v)| (k, (v.max(1) as f64).log2()))
        .collect();
    let mut global_count: HashMap<usize, u32> = HashMap::new();
    for ix in &graph.ixs {
        let mut seen = HashSet::new();
        for &l in ix {
            if seen.insert(l) {
                *global_count.entry(l).or_insert(0) += 1;
            }
        }
    }
    let iy_set: HashSet<usize> = graph.iy.iter().copied().collect();
    Ctx {
        ixs: graph.ixs.clone(),
        log2,
        global_count,
        iy_set,
    }
}

/// Build the recursive-bisection seed with a fresh RNG; return it with its tc
/// if it is a valid full-coverage tree.
fn bisection_seed(
    ctx: &Ctx,
    n: usize,
    seed: u64,
    sizes: &HashMap<usize, usize>,
    code: &EinCode<usize>,
) -> Option<(NestedEinsum<usize>, f64)> {
    let mut rng = Rng::new(seed);
    let items0: Vec<Item> = (0..n).map(|t| ctx.item_from_tensor(t)).collect();
    let tree = solve(ctx, items0, &mut rng, 0, sizes);
    let mut leaves = tree.leaf_indices();
    leaves.sort_unstable();
    if leaves.len() != n || !leaves.iter().enumerate().all(|(i, &v)| i == v) {
        return None;
    }
    let tc = contraction_complexity(&tree, sizes, &code.ixs).tc;
    Some((tree, tc))
}

// ===========================================================================
// Seed-refining simulated annealing on omeco's public ExprTree primitives.
// ===========================================================================

/// Compact-label view of the problem needed by the SA and the tree conversions.
struct SaCtx {
    /// original label -> compact int index
    label_map: HashMap<usize, usize>,
    /// compact int index -> original label
    inverse: Vec<usize>,
    /// log2 size per compact int index
    log2_sizes: Vec<f64>,
    nedge: usize,
}

fn build_sa_ctx(ixs: &[Vec<usize>], iy: &[usize], sizes: &HashMap<usize, usize>) -> SaCtx {
    let mut inverse: Vec<usize> = Vec::new();
    let mut label_map: HashMap<usize, usize> = HashMap::new();
    for ix in ixs {
        for &l in ix {
            if !label_map.contains_key(&l) {
                label_map.insert(l, inverse.len());
                inverse.push(l);
            }
        }
    }
    for &l in iy {
        if !label_map.contains_key(&l) {
            label_map.insert(l, inverse.len());
            inverse.push(l);
        }
    }
    let log2_sizes: Vec<f64> = inverse
        .iter()
        .map(|l| (*sizes.get(l).unwrap_or(&1) as f64).log2())
        .collect();
    let nedge = inverse.len();
    SaCtx {
        label_map,
        inverse,
        log2_sizes,
        nedge,
    }
}

/// Convert a (strictly binary) NestedEinsum into an ExprTree, replicating the
/// library's `nested_to_expr_tree` conversion (leaf children take their labels
/// from the parent einsum's `ixs[i]`). Returns `None` if the top is not a
/// binary node or a non-binary node is encountered.
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
            let child = |i: usize| -> Option<ExprTree> {
                match &args[i] {
                    NestedEinsum::Leaf { tensor_index } => {
                        let out_dims: Vec<usize> =
                            eins.ixs[i].iter().map(|l| label_map[l]).collect();
                        Some(ExprTree::leaf(out_dims, *tensor_index))
                    }
                    NestedEinsum::Node { .. } => nested_to_expr(&args[i], label_map),
                }
            };
            let left = child(0)?;
            let right = child(1)?;
            let out_dims: Vec<usize> = eins.iy.iter().map(|l| label_map[l]).collect();
            Some(ExprTree::node(left, right, out_dims))
        }
    }
}

/// Convert an ExprTree back to a NestedEinsum, replicating the library's
/// `expr_tree_to_nested` (root uses the requested open edges).
fn expr_to_nested(
    tree: &ExprTree,
    original_ixs: &[Vec<usize>],
    inverse: &[usize],
    openedges: &[usize],
    level: usize,
) -> NestedEinsum<usize> {
    match tree {
        ExprTree::Leaf(info) => NestedEinsum::leaf(info.tensor_id.unwrap_or(0)),
        ExprTree::Node { left, right, info } => {
            let l = expr_to_nested(left, original_ixs, inverse, openedges, level + 1);
            let r = expr_to_nested(right, original_ixs, inverse, openedges, level + 1);
            let iy: Vec<usize> = if level == 0 {
                openedges.to_vec()
            } else {
                info.out_dims.iter().map(|&i| inverse[i]).collect()
            };
            let ll = child_labels(&l, original_ixs);
            let rl = child_labels(&r, original_ixs);
            let eins = EinCode::new(vec![ll, rl], iy);
            NestedEinsum::node(vec![l, r], eins)
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

/// Space complexity of a single node's output labels.
#[inline]
fn node_sc(out_dims: &[usize], log2_sizes: &[f64]) -> f64 {
    out_dims.iter().map(|&l| log2_sizes[l]).sum()
}

/// Local space complexity at a node for a rule (matches the library `_sc`).
#[inline]
fn local_sc(tree: &ExprTree, rule: Rule, log2_sizes: &[f64]) -> f64 {
    match tree {
        ExprTree::Leaf(info) => node_sc(&info.out_dims, log2_sizes),
        ExprTree::Node { left, right, info } => {
            let tree_sc = node_sc(&info.out_dims, log2_sizes);
            let child_sc = match rule {
                Rule::Rule1 | Rule::Rule2 => node_sc(left.labels(), log2_sizes),
                Rule::Rule3 | Rule::Rule4 | Rule::Rule5 => node_sc(right.labels(), log2_sizes),
            };
            tree_sc.max(child_sc)
        }
    }
}

/// One SA sweep over the whole tree (post-order, mutation-first). Replicates
/// the library's `optimize_subtree_mut`: energy is `dtc` plus, when the local
/// sc exceeds `sc_target`, `sc_weight * dsc`. On these large closed graphs the
/// sc term is always active, so the search jointly minimises tc and sc — a much
/// stronger gradient than pure tc, and the sc/tc correlation drives tc down.
#[allow(clippy::too_many_arguments)]
fn sa_sweep(
    tree: &mut ExprTree,
    beta: f64,
    log2_sizes: &[f64],
    sc_target: f64,
    sc_weight: f64,
    rng: &mut Rng,
    scratch: &mut ScratchSpace,
) {
    let rules = Rule::applicable_rules(tree, DecompositionType::Tree);
    if !rules.is_empty() {
        let rule = rules[rng.below(rules.len())];
        if let Some(diff) = scratch.rule_diff(tree, rule, log2_sizes, false) {
            let dtc = diff.tc1 - diff.tc0;
            let sc = local_sc(tree, rule, log2_sizes);
            let sc_after = sc.max(sc + diff.dsc);
            let d_energy = if sc_after > sc_target {
                sc_weight * diff.dsc + dtc
            } else {
                dtc
            };
            if d_energy <= 0.0 || rng.f64() < (-beta * d_energy).exp() {
                apply_rule_mut(tree, rule, diff.new_labels);
            }
        }
    }
    if let ExprTree::Node { left, right, .. } = tree {
        sa_sweep(left, beta, log2_sizes, sc_target, sc_weight, rng, scratch);
        sa_sweep(right, beta, log2_sizes, sc_target, sc_weight, rng, scratch);
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
    let deadline_ms = budget_ms * 0.9;

    let graph: GraphData = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let sizes: HashMap<usize, usize> = graph
        .sizes
        .iter()
        .map(|(k, v)| Ok::<_, std::num::ParseIntError>((k.parse::<usize>()?, *v)))
        .collect::<Result<_, _>>()?;
    let code: EinCode<usize> = EinCode::new(graph.ixs.clone(), graph.iy.clone());
    let ixs = &graph.ixs;
    let iy = &graph.iy;
    let n = ixs.len();
    let _ = &graph.name;

    let tc_of = |tree: &NestedEinsum<usize>| contraction_complexity(tree, &sizes, &code.ixs).tc;

    // The validator's scorer parses the tree JSON with Python's default
    // recursion limit (~1000) and recomputes cost with recursive walks; each
    // tree level nests ~2 JSON levels, so JSON depth ≈ 2·height. Empirically a
    // height-475 tree parses (baseline) and a height-544 tree raises
    // RecursionError (rejected). Cap at the proven-safe 475; every tree of
    // lesser height has strictly smaller JSON depth, so it is safe too.
    const HEIGHT_CAP: usize = 475;
    let ctx = build_ctx(&graph, &sizes);
    // Guaranteed-shallow fallback (balanced over all tensors, height ~log2 n).
    let safe_floor = build_balanced(&ctx, &(0..n).collect::<Vec<_>>()).tree;

    // ---- 1. greedy seed, written immediately (anytime safety net) ----
    let greedy = optimize_code(&code, &sizes, &GreedyMethod::default())
        .ok_or("greedy optimizer returned no tree")?;
    let greedy_tc = tc_of(&greedy);
    // Adopt greedy only if it is shallow enough to serialise; else the safe
    // balanced floor.
    let (mut best, mut best_tc) = if nested_height(&greedy) <= HEIGHT_CAP {
        (greedy, greedy_tc)
    } else {
        let f = safe_floor.clone();
        let t = tc_of(&f);
        (f, t)
    };
    atomic_write(out_path, &best)?;
    eprintln!(
        "[029] greedy seed: tc={greedy_tc:.4} height={} (n={n}) adopted_tc={best_tc:.4}",
        nested_height(&best)
    );

    // Experiment mode is enabled by any attribution toggle; otherwise the
    // DEFAULT path runs the proven library-TreeSA doubling loop from greedy.
    // Rationale (see LOG.md): at the 90 s budget the plain greedy+anneal already
    // converges to record-level on both targets and produces serialisable
    // (shallow-enough) trees, while the structured-seed + warm-refinement search
    // — although it reaches comparable tc quickly — trends toward trees deeper
    // than the validator's JSON-parse recursion limit, and on the expander it is
    // strictly worse. So seeding gives no net benefit at this budget and the
    // default must not regress below the reference.
    let experiment = ["A029_WARM", "A029_NOSEED", "A029_SCT", "A029_EXPERIMENT"]
        .iter()
        .any(|k| std::env::var(k).is_ok());

    if !experiment {
        // Library TreeSA, ntrials=1, doubling niters, pure tc (sc unbounded);
        // each round a fresh full anneal from greedy, keep the best safe tree.
        let mut niters = 5usize;
        loop {
            if start.elapsed().as_secs_f64() * 1e3 >= deadline_ms {
                break;
            }
            let round_start = Instant::now();
            let treesa = TreeSA::default()
                .with_ntrials(1)
                .with_niters(niters)
                .with_sc_target(f64::INFINITY);
            let Some(tree) = optimize_code(&code, &sizes, &treesa) else {
                break;
            };
            let tc = tc_of(&tree);
            if tc < best_tc - 1e-12 && nested_height(&tree) <= HEIGHT_CAP {
                best = tree;
                best_tc = tc;
                atomic_write(out_path, &best)?;
            }
            let round_ms = round_start.elapsed().as_secs_f64() * 1e3;
            let remaining = deadline_ms - start.elapsed().as_secs_f64() * 1e3;
            if round_ms * 2.0 > remaining {
                if round_ms > remaining {
                    break;
                }
            } else {
                niters = (niters * 2).min(400);
            }
        }
        atomic_write(out_path, &best)?;
        let final_cc = contraction_complexity(&best, &sizes, &code.ixs);
        eprintln!(
            "[029] final (default): greedy_tc={greedy_tc:.4} final_tc={:.4} final_sc={:.4} height={}",
            final_cc.tc,
            final_cc.sc,
            nested_height(&best)
        );
        return Ok(());
    }

    // ---- 2. structured seed by instance shape (EXPERIMENT mode) ----
    let mut seed_tc = greedy_tc;
    let mut seed_kind = "greedy";
    // Control toggle: A029_NOSEED=1 disables the structured seed (SA from
    // greedy) so the seed-vs-anneal attribution can be measured directly.
    let noseed = std::env::var("A029_NOSEED").map(|v| v == "1").unwrap_or(false);

    if noseed {
        eprintln!("[029] A029_NOSEED set: skipping structured seed (SA from greedy)");
    } else if let Some((tree, tc)) = structured_seed(&ctx, ixs, iy, &sizes, &code) {
        eprintln!("[029] RQC balanced-sweep seed: tc={tc:.4}");
        if tc < best_tc {
            best = tree;
            best_tc = tc;
        }
        seed_tc = tc;
        seed_kind = "rqc-balanced-sweep";
    } else if !noseed && n > BASE_THRESHOLD {
        // Recursive FM bisection (a few restarts, cheap vs the 90 s SA budget).
        let mut best_bis: Option<f64> = None;
        for s in 1..=3u64 {
            if start.elapsed().as_secs_f64() * 1e3 >= deadline_ms * 0.3 {
                break;
            }
            if let Some((tree, tc)) = bisection_seed(&ctx, n, s, &sizes, &code) {
                if best_bis.map(|b| tc < b).unwrap_or(true) {
                    best_bis = Some(tc);
                }
                if tc < best_tc {
                    best = tree;
                    best_tc = tc;
                }
            }
        }
        if let Some(tc) = best_bis {
            eprintln!("[029] FM-bisection seed: tc={tc:.4}");
            seed_tc = tc;
            seed_kind = "fm-bisection";
        }
    }
    atomic_write(out_path, &best)?;
    eprintln!("[029] adopted seed: kind={seed_kind} seed_tc={seed_tc:.4} best_tc={best_tc:.4}");

    // ---- 3. continuous SA refinement of the seed (pure tc) ----
    let sactx = build_sa_ctx(ixs, iy, &sizes);
    // Convert the adopted seed to an ExprTree; fall back to greedy if the seed
    // is not convertible (should not happen for these binary seeds).
    let seed_expr = nested_to_expr(&best, &sactx.label_map).or_else(|| {
        optimize_code(&code, &sizes, &GreedyMethod::default())
            .and_then(|g| nested_to_expr(&g, &sactx.label_map))
    });

    if let Some(seed_expr) = seed_expr {
        let mut scratch = ScratchSpace::new(sactx.nedge);
        let mut rng = Rng::new(0x00C0_FFEE ^ (n as u64));

        // Library-default annealing schedule (β 0.01 → ~15 in steps of 0.05).
        // Same driver as `treesa_tuned`: each round is a FRESH full-schedule
        // anneal from the seed with `niters` per temperature, `niters` doubling
        // (5 → 400) while a round still fits the remaining budget, keeping the
        // global best. The only difference from the reference is the init tree
        // (our structured seed vs greedy), which isolates seed-vs-anneal.
        //
        // sc_target selects the objective: +inf = pure tc (matches the
        // reference / the pure-tc leaderboard); a finite value adds the joint
        // sc term. Overridable via A029_SCT for attribution runs.
        let betas: Vec<f64> = (0..300).map(|i| 0.01 + 0.05 * i as f64).collect();
        let sc_target = std::env::var("A029_SCT")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(f64::INFINITY);
        let sc_weight = 1.0f64;

        // Warm mode: a single CONTINUOUS anneal from the seed over only the
        // cold tail β ≥ β0, so the hot phase never scrambles the seed
        // structure. This is the refinement that can actually exploit a good
        // seed — local descent that preserves it — as opposed to the
        // full-schedule rounds that wash the seed out.
        //
        // Auto-selected by instance shape: for a layered-circuit (RQC) seed the
        // structured tree + warm refinement dominates greedy + full anneal
        // (measured ~106.5 vs ~129 on rqc_97_m24); for an expander (reg3) the
        // full anneal wins, so warm is disabled and the driver falls through to
        // the doubling-niters fresh-schedule loop (which matches the reference
        // when seeded from greedy). `A029_WARM=β0` overrides the β0 / forces
        // warm for attribution runs.
        let structured_warm = seed_kind.starts_with("rqc");
        let warm = std::env::var("A029_WARM")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .or(if structured_warm { Some(2.0) } else { None });

        let mut best_expr_tc = tree_complexity(&seed_expr, &sactx.log2_sizes).0;
        if best_expr_tc < best_tc - 1e-12 {
            let cand = expr_to_nested(&seed_expr, ixs, &sactx.inverse, iy, 0);
            if nested_height(&cand) <= HEIGHT_CAP {
                best = cand;
                best_tc = best_expr_tc;
                atomic_write(out_path, &best)?;
            }
        }

        if let Some(b0) = warm {
            let betas_warm: Vec<f64> = betas.iter().copied().filter(|&b| b >= b0).collect();
            let mut work = seed_expr.clone();
            let mut niters = 20usize;
            let mut sweeps = 0u64;
            let mut done = false;
            while !done {
                for &beta in &betas_warm {
                    for _ in 0..niters {
                        if start.elapsed().as_secs_f64() * 1e3 >= deadline_ms {
                            done = true;
                            break;
                        }
                        sa_sweep(
                            &mut work,
                            beta,
                            &sactx.log2_sizes,
                            sc_target,
                            sc_weight,
                            &mut rng,
                            &mut scratch,
                        );
                        sweeps += 1;
                    }
                    if done {
                        break;
                    }
                }
                let tc = tree_complexity(&work, &sactx.log2_sizes).0;
                if tc < best_expr_tc - 1e-12 {
                    best_expr_tc = tc;
                }
                if tc < best_tc - 1e-12 {
                    // Only adopt/write trees shallow enough to serialise safely.
                    let cand = expr_to_nested(&work, ixs, &sactx.inverse, iy, 0);
                    if nested_height(&cand) <= HEIGHT_CAP {
                        best = cand;
                        best_tc = tc;
                        atomic_write(out_path, &best)?;
                    }
                }
                niters = (niters * 2).min(400);
            }
            atomic_write(out_path, &best)?;
            eprintln!("[029] WARM SA done: b0={b0} sweeps={sweeps} best_expr_tc={best_expr_tc:.4}");
            let final_cc = contraction_complexity(&best, &sizes, &code.ixs);
            eprintln!(
                "[029] final: seed_kind={seed_kind} greedy_tc={greedy_tc:.4} seed_tc={seed_tc:.4} final_tc={:.4} final_sc={:.4}",
                final_cc.tc, final_cc.sc
            );
            return Ok(());
        }

        let mut niters = 5usize;
        let mut sweeps = 0u64;
        let mut rounds = 0u64;
        loop {
            if start.elapsed().as_secs_f64() * 1e3 >= deadline_ms {
                break;
            }
            let round_start = Instant::now();
            let mut work = seed_expr.clone();
            let mut aborted = false;
            'sched: for &beta in &betas {
                for _ in 0..niters {
                    if start.elapsed().as_secs_f64() * 1e3 >= deadline_ms {
                        aborted = true;
                        break 'sched;
                    }
                    sa_sweep(
                        &mut work,
                        beta,
                        &sactx.log2_sizes,
                        sc_target,
                        sc_weight,
                        &mut rng,
                        &mut scratch,
                    );
                    sweeps += 1;
                }
            }
            rounds += 1;
            // Only a completed schedule is a fair (cold) tree to adopt.
            if !aborted {
                let tc = tree_complexity(&work, &sactx.log2_sizes).0;
                if tc < best_expr_tc - 1e-12 {
                    best_expr_tc = tc;
                }
                if tc < best_tc - 1e-12 {
                    let cand = expr_to_nested(&work, ixs, &sactx.inverse, iy, 0);
                    if nested_height(&cand) <= HEIGHT_CAP {
                        best = cand;
                        best_tc = tc;
                        atomic_write(out_path, &best)?;
                    }
                }
            }
            let round_ms = round_start.elapsed().as_secs_f64() * 1e3;
            let remaining = deadline_ms - start.elapsed().as_secs_f64() * 1e3;
            if round_ms * 2.0 > remaining {
                if round_ms > remaining {
                    break;
                }
            } else {
                niters = (niters * 2).min(400);
            }
        }
        eprintln!("[029] SA done: rounds={rounds} sweeps={sweeps} niters={niters} best_expr_tc={best_expr_tc:.4}");
    }

    // Final safety write.
    atomic_write(out_path, &best)?;
    let final_cc = contraction_complexity(&best, &sizes, &code.ixs);
    eprintln!(
        "[029] final: seed_kind={seed_kind} greedy_tc={greedy_tc:.4} seed_tc={seed_tc:.4} final_tc={:.4} final_sc={:.4}",
        final_cc.tc, final_cc.sc
    );
    Ok(())
}
