//! Refine a greedy-seeded contraction tree with the waist-surgery pass.
//!
//! Usage: `waist_refine <graph.json> [budget_ms] [out.json]`
//!
//! Reads an omeco-style JSON graph (`{ "ixs", "iy", "sizes" }` with integer
//! labels), builds a greedy seed, runs [`omeco::waist_surgery::refine`] for the
//! given wall-clock budget (default 20000 ms), and prints the seed and refined
//! time complexities plus the surgery diagnostics. With an `out.json` argument it
//! writes the refined contraction tree in omeco `writejson` format.

use std::collections::HashMap;
use std::time::Duration;

use omeco::json::writejson;
use omeco::waist_surgery::refine;
use omeco::{contraction_complexity, optimize_code, EinCode, GreedyMethod};
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
        eprintln!("usage: waist_refine <graph.json> [budget_ms] [out.json]");
        std::process::exit(2);
    }
    let budget_ms: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20_000);

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

    let (refined, report) = refine(&seed, &code, &sizes, Duration::from_millis(budget_ms));
    let cc = contraction_complexity(&refined, &sizes, &code.ixs);

    println!(
        "n={} seed_tc={:.4} refined_tc={:.4} sc={:.4}",
        report.n_original, seed_tc, cc.tc, cc.sc,
    );
    println!(
        "surgery: calls={} cheaper_cuts={} rebuild_accepts={} waist_min_hits={}",
        report.surgery_calls, report.cheaper_cuts, report.rebuild_accepts, report.waist_min_hits,
    );

    if let Some(out) = args.get(3) {
        writejson(out, &refined)?;
        println!("wrote {out}");
    }
    Ok(())
}
