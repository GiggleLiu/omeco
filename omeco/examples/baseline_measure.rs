//! Measure baseline GreedyMethod/TreeSA complexity and runtime on one graph.
//!
//! `baseline_measure <graph.json> <ntrials> <niters>` prints a JSON object
//! with greedy/treesa tc, sc, rwc and mean wall-clock (3 runs), matching the
//! protocol of the benchmark example. Used by the autoresearch validator to
//! produce baselines for holdout instances.

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
    if args.len() != 4 {
        eprintln!("usage: baseline_measure <graph.json> <ntrials> <niters>");
        std::process::exit(2);
    }
    let graph: GraphData = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let ntrials: usize = args[2].parse()?;
    let niters: usize = args[3].parse()?;

    let sizes: HashMap<usize, usize> = graph
        .sizes
        .iter()
        .map(|(k, v)| Ok::<_, std::num::ParseIntError>((k.parse::<usize>()?, *v)))
        .collect::<Result<_, _>>()?;
    let code: EinCode<usize> = EinCode::new(graph.ixs.clone(), graph.iy.clone());

    let mut out = serde_json::Map::new();
    for (name, runs) in [("greedy", 3usize), ("treesa", 3usize)] {
        let mut best = None;
        let mut total_ms = 0.0;
        for _ in 0..runs {
            let start = Instant::now();
            let tree = if name == "greedy" {
                optimize_code(&code, &sizes, &GreedyMethod::default())
            } else {
                let treesa = TreeSA::default().with_ntrials(ntrials).with_niters(niters);
                optimize_code(&code, &sizes, &treesa)
            }
            .ok_or("optimizer returned no tree")?;
            total_ms += start.elapsed().as_secs_f64() * 1e3;
            best = Some(tree);
        }
        let tree = best.ok_or("no result")?;
        let cc = contraction_complexity(&tree, &sizes, &code.ixs);
        out.insert(
            name.to_string(),
            serde_json::json!({"tc": cc.tc, "sc": cc.sc, "rwc": cc.rwc,
                               "avg_ms": total_ms / runs as f64}),
        );
    }
    println!("{}", serde_json::Value::Object(out));
    Ok(())
}
