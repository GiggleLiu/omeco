//! sc_target sweep for the paper's discovery figure.
//! usage: sweep_sctarget <graph.json> <budget_ms> <sc_target|inf>
//! Runs the anytime doubling TreeSA loop at the given sc_target and prints
//! the best tc/sc found as JSON.

use std::collections::HashMap;
use std::time::Instant;

use omeco::{contraction_complexity, optimize_code, EinCode, GreedyMethod, TreeSA};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GraphData {
    ixs: Vec<Vec<usize>>,
    iy: Vec<usize>,
    sizes: HashMap<String, usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let start = Instant::now();
    let budget_ms: f64 = args[2].parse()?;
    let sc_target: f64 = if args[3] == "inf" { f64::INFINITY } else { args[3].parse()? };

    let graph: GraphData = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let sizes: HashMap<usize, usize> = graph
        .sizes
        .iter()
        .map(|(k, v)| Ok::<_, std::num::ParseIntError>((k.parse::<usize>()?, *v)))
        .collect::<Result<_, _>>()?;
    let code: EinCode<usize> = EinCode::new(graph.ixs.clone(), graph.iy.clone());

    let mut best = optimize_code(&code, &sizes, &GreedyMethod::default())
        .ok_or("greedy failed")?;
    let mut best_cc = contraction_complexity(&best, &sizes, &code.ixs);
    let mut niters = 5usize;
    while (start.elapsed().as_secs_f64() * 1e3) < budget_ms * 0.9 {
        let treesa = TreeSA::default()
            .with_ntrials(1)
            .with_niters(niters)
            .with_sc_target(sc_target);
        let Some(tree) = optimize_code(&code, &sizes, &treesa) else { break };
        let cc = contraction_complexity(&tree, &sizes, &code.ixs);
        if cc.tc < best_cc.tc {
            best = tree;
            best_cc = cc;
        }
        niters = (niters * 2).min(400);
    }
    let _ = &best;
    println!("{{\"tc\": {}, \"sc\": {}}}", best_cc.tc, best_cc.sc);
    Ok(())
}
