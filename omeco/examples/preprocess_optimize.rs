//! Simplify-then-optimize a tensor network read from an omeco-style JSON graph.
//!
//! Usage: `preprocess_optimize <graph.json> [out.json]`
//!
//! The graph JSON has the shape `{ "ixs": [[..],..], "iy": [..], "sizes": {..} }`
//! with integer labels (as strings in `sizes`). This runs the deterministic
//! rank-non-increasing simplification front-end, optimizes the reduced network
//! with the greedy method, splices the collapsed structure back, and prints the
//! shrink statistics and the spliced tree's time complexity. With an `out.json`
//! argument it writes the resulting contraction tree in omeco `writejson` format.

use std::collections::HashMap;

use omeco::json::writejson;
use omeco::preprocess::simplify_then_optimize;
use omeco::{contraction_complexity, EinCode, GreedyMethod};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GraphData {
    ixs: Vec<Vec<usize>>,
    iy: Vec<usize>,
    sizes: HashMap<String, usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: preprocess_optimize <graph.json> [out.json]");
        std::process::exit(2);
    }
    let graph: GraphData = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let sizes: HashMap<usize, usize> = graph
        .sizes
        .iter()
        .map(|(k, v)| Ok::<_, std::num::ParseIntError>((k.parse::<usize>()?, *v)))
        .collect::<Result<_, _>>()?;
    let code = EinCode::new(graph.ixs.clone(), graph.iy.clone());

    let (tree, report) = simplify_then_optimize(&code, &sizes, &GreedyMethod::default())
        .ok_or("optimizer returned no tree")?;
    let cc = contraction_complexity(&tree, &sizes, &code.ixs);

    println!(
        "n_original={} n_reduced={} shrink={:.4} tc={:.4} sc={:.4} leaves={}",
        report.n_original,
        report.n_reduced,
        report.shrink,
        cc.tc,
        cc.sc,
        tree.leaf_count(),
    );

    if let Some(out) = args.get(2) {
        writejson(out, &tree)?;
        println!("wrote {out}");
    }
    Ok(())
}
