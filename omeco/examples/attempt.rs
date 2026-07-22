//! Attempt entry point for the autoresearch validator.
//!
//! Contract: `attempt <graph.json> <budget_ms> <out.json>` — read an einsum
//! graph, search for a contraction order within the wall-clock budget, and
//! write the best tree found in `writejson` format before the deadline.
//!
//! This baseline implementation seeds with GreedyMethod (written immediately
//! so a valid result always exists), then spends the remaining budget on
//! TreeSA rounds with growing iteration counts, keeping the best tree by tc.

use std::collections::HashMap;
use std::time::Instant;

use omeco::json::writejson;
use omeco::{contraction_complexity, optimize_code, EinCode, GreedyMethod, NestedEinsum, TreeSA};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GraphData {
    ixs: Vec<Vec<usize>>,
    iy: Vec<usize>,
    sizes: HashMap<String, usize>,
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

    let graph: GraphData = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let sizes: HashMap<usize, usize> = graph
        .sizes
        .iter()
        .map(|(k, v)| Ok::<_, std::num::ParseIntError>((k.parse::<usize>()?, *v)))
        .collect::<Result<_, _>>()?;
    let code: EinCode<usize> = EinCode::new(graph.ixs.clone(), graph.iy.clone());

    let tc_of = |tree: &NestedEinsum<usize>| contraction_complexity(tree, &sizes, &code.ixs).tc;

    let mut best = optimize_code(&code, &sizes, &GreedyMethod::default())
        .ok_or("greedy optimizer returned no tree")?;
    let mut best_tc = tc_of(&best);
    writejson(out_path, &best)?;

    // Spend the rest of the budget on TreeSA rounds, doubling niters while at
    // least half the elapsed round time still fits in the remaining budget.
    let mut niters = 5usize;
    loop {
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        if elapsed >= budget_ms * 0.85 {
            break;
        }
        let round_start = Instant::now();
        let treesa = TreeSA::default().with_ntrials(1).with_niters(niters);
        let Some(tree) = optimize_code(&code, &sizes, &treesa) else {
            break;
        };
        let tc = tc_of(&tree);
        if tc < best_tc {
            best = tree;
            best_tc = tc;
            writejson(out_path, &best)?;
        }
        let round_ms = round_start.elapsed().as_secs_f64() * 1e3;
        let remaining = budget_ms * 0.85 - start.elapsed().as_secs_f64() * 1e3;
        if round_ms * 2.0 > remaining {
            break;
        }
        niters = (niters * 2).min(200);
    }
    Ok(())
}
