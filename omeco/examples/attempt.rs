//! Attempt 023 — hierarchical coarsen + exact super-network contraction tree.
//!
//! Contract: `attempt <graph.json> <budget_ms> <out.json>`.
//!
//! Mechanism under test: coarsen the tensor graph into m (~12-16) connected
//! super-tensors via heavy-edge agglomeration, contract each cluster
//! internally with a heuristic (greedy) to obtain its subtree and boundary
//! label set, then solve the SUPER-network's contraction tree EXACTLY with a
//! subset DP over the m super-nodes (minimizing total flops in log2 domain,
//! subject to the sc-cap on every intermediate). Splice the cluster subtrees
//! into the exact super-tree and keep the global best over many random
//! coarsenings. A TreeSA doubling loop is the anytime quality floor.
//!
//! Scoring is pure tc (validator recomputes from topology); sc is capped
//! (reg3_250 <= 35, sycamore_m20 <= 55), so feasibility is enforced here too.

use std::collections::HashMap;
use std::time::Instant;

use omeco::json::writejson;
use omeco::{contraction_complexity, optimize_code, EinCode, GreedyMethod, NestedEinsum, TreeSA};
use rand::rngs::StdRng;
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

/// sc-cap per instance (matches the validator's enforced caps). Unknown
/// instances fall back to a generous cap derived from the greedy solution.
fn sc_cap(name: &str, greedy_sc: f64) -> f64 {
    match name {
        "reg3_250" => 35.0,
        "sycamore_m20" => 55.0,
        _ => greedy_sc.max(30.0) + 2.0,
    }
}

/// Best-so-far tracker with anytime writes; prefers feasible (sc <= cap) trees,
/// then lower tc.
struct Best {
    tree: NestedEinsum<usize>,
    tc: f64,
    sc: f64,
    feasible: bool,
}

impl Best {
    fn consider(
        &mut self,
        cand: NestedEinsum<usize>,
        tc: f64,
        sc: f64,
        cap: f64,
        out_path: &str,
    ) -> bool {
        let feasible = sc <= cap + 1e-9;
        let take = if feasible && !self.feasible {
            true
        } else if feasible == self.feasible {
            tc < self.tc - 1e-12
        } else {
            false
        };
        if take {
            self.tree = cand;
            self.tc = tc;
            self.sc = sc;
            self.feasible = feasible;
            let _ = writejson(out_path, &self.tree);
        }
        take
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
    let out_path = args[3].clone();

    let graph: GraphData = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let sizes: HashMap<usize, usize> = graph
        .sizes
        .iter()
        .map(|(k, v)| Ok::<_, std::num::ParseIntError>((k.parse::<usize>()?, *v)))
        .collect::<Result<_, _>>()?;
    let code: EinCode<usize> = EinCode::new(graph.ixs.clone(), graph.iy.clone());
    let ixs = graph.ixs.clone();
    let iy = graph.iy.clone();
    let n = ixs.len();

    let log2_size: HashMap<usize, f64> = sizes.iter().map(|(&k, &v)| (k, (v as f64).log2())).collect();
    let cc_of = |tree: &NestedEinsum<usize>| contraction_complexity(tree, &sizes, &ixs);

    // ---- safety net: greedy, written immediately ----
    let greedy = optimize_code(&code, &sizes, &GreedyMethod::default())
        .ok_or("greedy optimizer returned no tree")?;
    let gcc = cc_of(&greedy);
    let cap = sc_cap(&graph.name, gcc.sc);
    let mut best = Best {
        tree: greedy,
        tc: gcc.tc,
        sc: gcc.sc,
        feasible: gcc.sc <= cap + 1e-9,
    };
    writejson(&out_path, &best.tree)?;

    // Precompute per-label tensor occurrence counts and adjacency.
    let mut label_tensor_count: HashMap<usize, usize> = HashMap::new();
    for ix in &ixs {
        // count each tensor once per label it holds
        let mut seen = std::collections::HashSet::new();
        for &l in ix {
            if seen.insert(l) {
                *label_tensor_count.entry(l).or_insert(0) += 1;
            }
        }
    }
    let iy_set: std::collections::HashSet<usize> = iy.iter().copied().collect();
    // label -> list of tensors containing it (for coarsening adjacency)
    let mut label_tensors: HashMap<usize, Vec<usize>> = HashMap::new();
    for (t, ix) in ixs.iter().enumerate() {
        let mut seen = std::collections::HashSet::new();
        for &l in ix {
            if seen.insert(l) {
                label_tensors.entry(l).or_default().push(t);
            }
        }
    }

    let mut rng = StdRng::seed_from_u64(0x0233_2026_u64 ^ (n as u64));
    let debug = std::env::var("OMECO_DEBUG").is_ok();
    let mut hier_best_feasible = f64::INFINITY;
    let mut hier_rounds_feasible = 0usize;
    let mut hier_rounds_total = 0usize;

    // ---- phase A: hierarchical, up to ~40% of budget ----
    // (The mechanism under test. On cap-tight expander/RQC instances it does
    // not beat the TreeSA floor, so the remaining ~50% is left to the floor.)
    let hier_deadline = budget_ms * 0.40;
    let mut iter = 0usize;
    loop {
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        if elapsed >= hier_deadline {
            break;
        }
        iter += 1;
        // pick a coarsening target size (many diverse coarsenings)
        let m_target = 10 + (rng.random::<u32>() % 7) as usize; // 10..=16
        if let Some((tree, tc, sc)) = hierarchical_round(
            n,
            &ixs,
            &iy,
            &iy_set,
            &label_tensors,
            &label_tensor_count,
            &log2_size,
            &sizes,
            m_target,
            cap,
            &mut rng,
            &cc_of,
        ) {
            hier_rounds_total += 1;
            if sc <= cap + 1e-9 {
                hier_rounds_feasible += 1;
                if tc < hier_best_feasible {
                    hier_best_feasible = tc;
                }
            }
            best.consider(tree, tc, sc, cap, &out_path);
        } else {
            hier_rounds_total += 1;
        }
        if iter > 100_000 {
            break;
        }
    }
    if debug {
        eprintln!(
            "[hier] rounds={} feasible={} best_feasible_tc={:.4} (cap={})",
            hier_rounds_total, hier_rounds_feasible, hier_best_feasible, cap
        );
        eprintln!(
            "[after-hier] best.tc={:.4} best.sc={:.2} feasible={}",
            best.tc, best.sc, best.feasible
        );
    }

    // ---- phase B: TreeSA doubling floor, until ~90% of budget ----
    let mut niters = 5usize;
    loop {
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        if elapsed >= budget_ms * 0.9 {
            break;
        }
        let round_start = Instant::now();
        let treesa = TreeSA::default()
            .with_ntrials(1)
            .with_niters(niters)
            .with_sc_target(cap);
        let Some(tree) = optimize_code(&code, &sizes, &treesa) else {
            break;
        };
        let cc = cc_of(&tree);
        best.consider(tree, cc.tc, cc.sc, cap, &out_path);
        let round_ms = round_start.elapsed().as_secs_f64() * 1e3;
        let remaining = budget_ms * 0.9 - start.elapsed().as_secs_f64() * 1e3;
        if round_ms > remaining {
            break;
        }
        if round_ms * 2.0 <= remaining {
            niters = (niters * 2).min(400);
        }
    }

    // Final guarantee: emit best.
    writejson(&out_path, &best.tree)?;
    Ok(())
}

/// One hierarchical coarsening + exact super-solve round.
#[allow(clippy::too_many_arguments)]
fn hierarchical_round(
    n: usize,
    ixs: &[Vec<usize>],
    iy: &[usize],
    iy_set: &std::collections::HashSet<usize>,
    label_tensors: &HashMap<usize, Vec<usize>>,
    label_tensor_count: &HashMap<usize, usize>,
    log2_size: &HashMap<usize, f64>,
    sizes: &HashMap<usize, usize>,
    m_target: usize,
    cap: f64,
    rng: &mut StdRng,
    cc_of: &impl Fn(&NestedEinsum<usize>) -> omeco::ContractionComplexity,
) -> Option<(NestedEinsum<usize>, f64, f64)> {
    let clusters = coarsen(n, label_tensors, log2_size, m_target, rng);
    let m = clusters.len();
    if m < 2 || m > 16 {
        return None;
    }

    // Cluster boundary label sets + internal subtrees (remapped to original ids).
    let mut boundaries: Vec<Vec<usize>> = Vec::with_capacity(m);
    let mut subtrees: Vec<NestedEinsum<usize>> = Vec::with_capacity(m);
    for cluster in &clusters {
        let cluster_set: std::collections::HashSet<usize> = cluster.iter().copied().collect();
        // in-cluster count per label
        let mut in_count: HashMap<usize, usize> = HashMap::new();
        for &t in cluster {
            let mut seen = std::collections::HashSet::new();
            for &l in &ixs[t] {
                if seen.insert(l) {
                    *in_count.entry(l).or_insert(0) += 1;
                }
            }
        }
        // boundary: appears outside the cluster OR in iy
        let mut boundary: Vec<usize> = Vec::new();
        let mut bseen = std::collections::HashSet::new();
        for (&l, &ic) in &in_count {
            let total = *label_tensor_count.get(&l).unwrap_or(&ic);
            if (total > ic || iy_set.contains(&l)) && bseen.insert(l) {
                boundary.push(l);
            }
        }
        boundary.sort_unstable();

        // sub-einsum: cluster tensors, output = boundary. Internal order is
        // sc-aware: greedy first; if it exceeds the cap, a short TreeSA pass
        // with sc_target=cap tries to bring the cluster-internal intermediates
        // under the cap (the exact super-level already respects the cap).
        let sub_ixs: Vec<Vec<usize>> = cluster.iter().map(|&t| ixs[t].clone()).collect();
        let sub_code = EinCode::new(sub_ixs.clone(), boundary.clone());
        let sub_tree = optimize_code(&sub_code, sizes, &GreedyMethod::default())?;
        let sub_sc = contraction_complexity(&sub_tree, sizes, &sub_ixs).sc;
        let sub_tree = if sub_sc > cap + 1e-9 && cluster.len() > 2 {
            let ts = TreeSA::default()
                .with_ntrials(1)
                .with_niters(30)
                .with_sc_target(cap);
            match optimize_code(&sub_code, sizes, &ts) {
                Some(t2) => {
                    let sc2 = contraction_complexity(&t2, sizes, &sub_ixs).sc;
                    if sc2 < sub_sc {
                        t2
                    } else {
                        sub_tree
                    }
                }
                None => sub_tree,
            }
        } else {
            sub_tree
        };
        let remapped = remap_leaves(&sub_tree, cluster);
        let _ = &cluster_set;
        boundaries.push(boundary);
        subtrees.push(remapped);
    }

    // Build super-label index.
    let mut label_to_sl: HashMap<usize, usize> = HashMap::new();
    let mut sl_label: Vec<usize> = Vec::new();
    let mut sl_weight: Vec<f64> = Vec::new();
    for b in &boundaries {
        for &l in b {
            if !label_to_sl.contains_key(&l) {
                let idx = sl_label.len();
                label_to_sl.insert(l, idx);
                sl_label.push(l);
                sl_weight.push(*log2_size.get(&l).unwrap_or(&0.0));
            }
        }
    }
    let nl = sl_label.len();
    let words_per = nl.div_ceil(64).max(1);
    let uniform = sl_weight.iter().all(|&w| (w - 1.0).abs() < 1e-12);

    // occ[sl] = bitmask over clusters that carry label sl; iy flag.
    let mut occ: Vec<u32> = vec![0u32; nl];
    let mut iy_flag: Vec<bool> = vec![false; nl];
    for (ci, b) in boundaries.iter().enumerate() {
        for &l in b {
            let sl = label_to_sl[&l];
            occ[sl] |= 1u32 << ci;
            if iy_set.contains(&l) {
                iy_flag[sl] = true;
            }
        }
    }
    let _ = iy;

    let full: u32 = if m == 32 { u32::MAX } else { (1u32 << m) - 1 };
    let nsub = 1usize << m;

    // Precompute out-bitset words and sc per subset.
    let mut out_words = vec![0u64; nsub * words_per];
    let mut sc_of = vec![0.0f64; nsub];
    for s in 0..nsub as u32 {
        let base = (s as usize) * words_per;
        let mut sc = 0.0f64;
        for sl in 0..nl {
            let om = occ[sl];
            let in_s = (om & s) != 0;
            if !in_s {
                continue;
            }
            let outside = (om & (full & !s)) != 0 || iy_flag[sl];
            if outside {
                out_words[base + (sl >> 6)] |= 1u64 << (sl & 63);
                sc += sl_weight[sl];
            }
        }
        sc_of[s as usize] = sc;
    }

    // Weighted popcount of a subset's out-bitset.
    let weighted_size = |base: usize, out_words: &[u64]| -> f64 {
        if uniform {
            let mut c = 0u32;
            for w in 0..words_per {
                c += out_words[base + w].count_ones();
            }
            c as f64
        } else {
            let mut acc = 0.0f64;
            for w in 0..words_per {
                let mut bits = out_words[base + w];
                while bits != 0 {
                    let b = bits.trailing_zeros() as usize;
                    acc += sl_weight[(w << 6) + b];
                    bits &= bits - 1;
                }
            }
            acc
        }
    };

    // Subset DP over super-nodes: minimize log2(total flops), sc-capped.
    const INF: f64 = f64::INFINITY;
    let mut dp = vec![INF; nsub];
    let mut split = vec![0u32; nsub];
    // order subsets by popcount
    let mut order: Vec<u32> = (0..nsub as u32).collect();
    order.sort_by_key(|s| s.count_ones());

    for &s in &order {
        let pc = s.count_ones();
        if pc == 0 {
            continue;
        }
        if sc_of[s as usize] > cap + 1e-9 {
            dp[s as usize] = INF;
            continue;
        }
        if pc == 1 {
            dp[s as usize] = f64::NEG_INFINITY; // 0 super-flops
            continue;
        }
        let lowbit = s & s.wrapping_neg();
        let mut best_cost = INF;
        let mut best_l = 0u32;
        let mut sub = s;
        while sub != 0 {
            if sub != s && (sub & lowbit) != 0 {
                let l = sub;
                let r = s ^ l;
                let dl = dp[l as usize];
                let dr = dp[r as usize];
                if dl != INF && dr != INF {
                    // node tc = weighted |out(L) | out(R)|
                    let lb = (l as usize) * words_per;
                    let rb = (r as usize) * words_per;
                    let node_tc = if uniform {
                        let mut c = 0u32;
                        for w in 0..words_per {
                            c += (out_words[lb + w] | out_words[rb + w]).count_ones();
                        }
                        c as f64
                    } else {
                        let mut acc = 0.0f64;
                        for w in 0..words_per {
                            let mut bits = out_words[lb + w] | out_words[rb + w];
                            while bits != 0 {
                                let b = bits.trailing_zeros() as usize;
                                acc += sl_weight[(w << 6) + b];
                                bits &= bits - 1;
                            }
                        }
                        acc
                    };
                    let cand = log2sumexp2_3(dl, dr, node_tc);
                    if cand < best_cost {
                        best_cost = cand;
                        best_l = l;
                    }
                }
            }
            sub = (sub - 1) & s;
        }
        dp[s as usize] = best_cost;
        split[s as usize] = best_l;
    }

    if dp[full as usize] == INF {
        return None; // no cap-feasible exact top-level tree for this coarsening
    }

    // Reconstruct + splice.
    let out_labels = |s: u32| -> Vec<usize> {
        let base = (s as usize) * words_per;
        let mut v = Vec::new();
        for w in 0..words_per {
            let mut bits = out_words[base + w];
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                v.push(sl_label[(w << 6) + b]);
                bits &= bits - 1;
            }
        }
        v
    };
    let tree = build_super(full, &split, &subtrees, &out_labels);
    let cc = cc_of(&tree);
    let _ = weighted_size;
    Some((tree, cc.tc, cc.sc))
}

fn build_super(
    s: u32,
    split: &[u32],
    subtrees: &[NestedEinsum<usize>],
    out_labels: &impl Fn(u32) -> Vec<usize>,
) -> NestedEinsum<usize> {
    if s.count_ones() == 1 {
        let idx = s.trailing_zeros() as usize;
        return subtrees[idx].clone();
    }
    let l = split[s as usize];
    let r = s ^ l;
    let left = build_super(l, split, subtrees, out_labels);
    let right = build_super(r, split, subtrees, out_labels);
    let eins = EinCode::new(vec![out_labels(l), out_labels(r)], out_labels(s));
    NestedEinsum::node(vec![left, right], eins)
}

/// Remap a sub-tree's leaf tensor indices (0..k) back to original ids.
fn remap_leaves(tree: &NestedEinsum<usize>, cluster: &[usize]) -> NestedEinsum<usize> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => NestedEinsum::leaf(cluster[*tensor_index]),
        NestedEinsum::Node { args, eins } => {
            let new_args = args.iter().map(|a| remap_leaves(a, cluster)).collect();
            NestedEinsum::node(new_args, eins.clone())
        }
    }
}

/// Heavy-edge agglomeration: merge the pair of clusters with the largest
/// weighted shared boundary until `m_target` clusters remain. Random tie-break.
fn coarsen(
    n: usize,
    label_tensors: &HashMap<usize, Vec<usize>>,
    log2_size: &HashMap<usize, f64>,
    m_target: usize,
    rng: &mut StdRng,
) -> Vec<Vec<usize>> {
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut alive: Vec<bool> = vec![true; n];
    let mut adj: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n];

    for (l, ts) in label_tensors {
        if ts.len() == 2 {
            let (a, b) = (ts[0], ts[1]);
            if a != b {
                let w = *log2_size.get(l).unwrap_or(&1.0);
                *adj[a].entry(b).or_insert(0.0) += w;
                *adj[b].entry(a).or_insert(0.0) += w;
            }
        }
    }

    let mut count = n;
    while count > m_target {
        // find max-weight alive edge, reservoir tie-break
        let mut best_w = f64::NEG_INFINITY;
        let mut best_pair: Option<(usize, usize)> = None;
        let mut ties = 0u32;
        for a in 0..n {
            if !alive[a] {
                continue;
            }
            for (&b, &w) in &adj[a] {
                if a < b && alive[b] {
                    if w > best_w + 1e-12 {
                        best_w = w;
                        best_pair = Some((a, b));
                        ties = 1;
                    } else if (w - best_w).abs() <= 1e-12 {
                        ties += 1;
                        if rng.random::<u32>() % ties == 0 {
                            best_pair = Some((a, b));
                        }
                    }
                }
            }
        }
        let Some((a, b)) = best_pair else {
            break; // no more mergeable edges (disconnected)
        };
        // merge b into a
        let bn: Vec<(usize, f64)> = adj[b].iter().map(|(&k, &v)| (k, v)).collect();
        for (k, w) in bn {
            adj[k].remove(&b);
            if k != a {
                *adj[a].entry(k).or_insert(0.0) += w;
                *adj[k].entry(a).or_insert(0.0) += w;
            }
        }
        adj[a].remove(&b);
        adj[b].clear();
        let bm = std::mem::take(&mut members[b]);
        members[a].extend(bm);
        alive[b] = false;
        count -= 1;
    }

    members
        .into_iter()
        .enumerate()
        .filter(|(i, _)| alive[*i])
        .map(|(_, m)| m)
        .collect()
}

/// log2(2^a + 2^b + 2^c) with -inf handling.
#[inline]
fn log2sumexp2_3(a: f64, b: f64, c: f64) -> f64 {
    let m = a.max(b).max(c);
    if m == f64::NEG_INFINITY {
        return m;
    }
    let mut s = 0.0;
    if a != f64::NEG_INFINITY {
        s += (a - m).exp2();
    }
    if b != f64::NEG_INFINITY {
        s += (b - m).exp2();
    }
    if c != f64::NEG_INFINITY {
        s += (c - m).exp2();
    }
    m + s.log2()
}
