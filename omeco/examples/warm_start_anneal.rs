//! Drive a custom simulated-annealing loop from a greedy seed via the warm-start API.
//!
//! Usage: `warm_start_anneal <graph.json> [sweeps]`
//!
//! This demonstrates [`omeco::treesa::prepare_warm_anneal`] /
//! [`omeco::treesa::warm_exprtree_to_nested`]: greedy-seed a tree, hand off to a
//! caller-owned anneal loop over the public `expr_tree` rewrite utilities, then
//! convert the annealed tree back to a `NestedEinsum`. Unlike `optimize_treesa`,
//! the caller controls the schedule, stopping rule, and acceptance — the point of
//! the warm-start API. Reads an omeco-style JSON graph with integer labels.

use std::collections::HashMap;

use omeco::expr_tree::{apply_rule_mut, tree_complexity, DecompositionType, Rule, ScratchSpace};
use omeco::treesa::{prepare_warm_anneal, warm_exprtree_to_nested};
use omeco::{contraction_complexity, optimize_code, EinCode, GreedyMethod};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GraphData {
    ixs: Vec<Vec<usize>>,
    iy: Vec<usize>,
    sizes: HashMap<String, usize>,
}

/// One recursive SA sweep: at every node try one random applicable rewrite,
/// accepting downhill moves always and uphill moves with the Metropolis rule.
fn sweep(
    tree: &mut omeco::expr_tree::ExprTree,
    beta: f64,
    log2_sizes: &[f64],
    rng: &mut SmallRng,
    scratch: &mut ScratchSpace,
) {
    let rules = Rule::applicable_rules(tree, DecompositionType::Tree);
    if !rules.is_empty() {
        let rule = rules[rng.random_range(0..rules.len())];
        if let Some(diff) = scratch.rule_diff(tree, rule, log2_sizes, false) {
            let dtc = diff.tc1 - diff.tc0;
            if dtc <= 0.0 || rng.random::<f64>() < (-beta * dtc).exp() {
                apply_rule_mut(tree, rule, diff.new_labels);
            }
        }
    }
    if let omeco::expr_tree::ExprTree::Node { left, right, .. } = tree {
        sweep(left, beta, log2_sizes, rng, scratch);
        sweep(right, beta, log2_sizes, rng, scratch);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: warm_start_anneal <graph.json> [sweeps]");
        std::process::exit(2);
    }
    let sweeps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2000);

    let graph: GraphData = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let sizes: HashMap<usize, usize> = graph
        .sizes
        .iter()
        .map(|(k, v)| Ok::<_, std::num::ParseIntError>((k.parse::<usize>()?, *v)))
        .collect::<Result<_, _>>()?;
    let code = EinCode::new(graph.ixs.clone(), graph.iy.clone());

    let seed = optimize_code(&code, &sizes, &GreedyMethod::default())
        .ok_or("greedy optimizer returned no tree")?;
    let seed_tc = contraction_complexity(&seed, &sizes, &code.ixs).tc;

    let Some(ctx) = prepare_warm_anneal(&code, &sizes, &seed) else {
        println!("seed has nothing to anneal (single tensor)");
        return Ok(());
    };
    let mut tree = ctx.tree;
    let mut scratch = ScratchSpace::new(ctx.nedge);
    let mut rng = SmallRng::seed_from_u64(42);

    // A simple linear inverse-temperature ramp from warm to cold.
    let (b_lo, b_hi) = (0.05_f64, 14.0_f64);
    for k in 0..sweeps {
        let beta = b_lo + (b_hi - b_lo) * (k as f64 / (sweeps.max(2) - 1) as f64);
        sweep(&mut tree, beta, &ctx.log2_sizes, &mut rng, &mut scratch);
    }

    let refined = warm_exprtree_to_nested(&tree, &code, &ctx.labels);
    let refined_tc = contraction_complexity(&refined, &sizes, &code.ixs).tc;
    println!(
        "sweeps={sweeps} seed_tc={seed_tc:.4} warm_tc={refined_tc:.4} (final tree tc={:.4})",
        tree_complexity(&tree, &ctx.log2_sizes).0,
    );
    Ok(())
}
