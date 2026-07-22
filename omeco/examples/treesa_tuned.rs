//! Strengthened reference for the autoresearch validator: baseline TreeSA
//! with `sc_target` tuned to the instance's sc cap (instead of the default
//! 20) plus the same anytime doubling schedule as the baseline attempt.
//!
//! Same CLI contract as `attempt`: `treesa_tuned <graph.json> <budget_ms>
//! <out.json>`. Exists to answer: how much of the cycle-2 record gains was
//! reference mis-tuning vs genuine mechanism contribution?

use std::collections::HashMap;
use std::time::Instant;

use omeco::json::writejson;
use omeco::{contraction_complexity, optimize_code, EinCode, GreedyMethod, NestedEinsum, TreeSA};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GraphData {
    #[serde(default)]
    name: String,
    ixs: Vec<Vec<usize>>,
    iy: Vec<usize>,
    sizes: HashMap<String, usize>,
}

fn sc_cap(name: &str, greedy_sc: f64) -> f64 {
    match name {
        "reg3_250" => 35.0,
        "sycamore_m20" => 55.0,
        _ => greedy_sc.max(30.0) + 2.0,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: treesa_tuned <graph.json> <budget_ms> <out.json>");
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

    let cc_of = |tree: &NestedEinsum<usize>| contraction_complexity(tree, &sizes, &code.ixs);

    let mut best = optimize_code(&code, &sizes, &GreedyMethod::default())
        .ok_or("greedy optimizer returned no tree")?;
    let mut best_cc = cc_of(&best);
    writejson(out_path, &best)?;
    let cap = sc_cap(&graph.name, best_cc.sc);

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
        let feasible = cc.sc <= cap;
        let best_feasible = best_cc.sc <= cap;
        // Prefer feasible trees; among equally feasible, lower tc wins.
        if (feasible && !best_feasible)
            || (feasible == best_feasible && cc.tc < best_cc.tc)
        {
            best = tree;
            best_cc = cc;
            writejson(out_path, &best)?;
        }
        let round_ms = round_start.elapsed().as_secs_f64() * 1e3;
        let remaining = budget_ms * 0.9 - start.elapsed().as_secs_f64() * 1e3;
        if round_ms * 2.0 > remaining {
            // keep annealing at the largest niters that fits until budget ends
            if round_ms > remaining {
                break;
            }
        } else {
            niters = (niters * 2).min(400);
        }
    }
    Ok(())
}
