//! Rust benchmark for omeco contraction order optimization.
//!
//! Run with: cargo run --release --example benchmark -p omeco

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use omeco::{contraction_complexity, optimize_code, EinCode, GreedyMethod, TreeSA};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct GraphData {
    name: String,
    description: String,
    ixs: Vec<Vec<usize>>,
    iy: Vec<usize>,
    sizes: HashMap<String, usize>,
}

#[derive(Debug, Serialize)]
struct MethodResult {
    tc: f64,
    sc: f64,
    rwc: f64,
    avg_ms: f64,
    total_ms: f64,
    runs: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    ntrials: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    niters: Option<usize>,
}

#[derive(Debug, Serialize)]
struct BenchmarkResult {
    name: String,
    description: String,
    tensors: usize,
    indices: usize,
    greedy: MethodResult,
    treesa: MethodResult,
}

fn load_graph(path: &Path) -> Result<GraphData, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let graph: GraphData = serde_json::from_str(&content)?;
    Ok(graph)
}

fn benchmark_graph(graph: &GraphData, ntrials: usize, niters: usize) -> BenchmarkResult {
    // Convert sizes from String keys to usize keys
    let sizes: HashMap<usize, usize> = graph
        .sizes
        .iter()
        .map(|(k, v)| (k.parse::<usize>().unwrap(), *v))
        .collect();

    let code: EinCode<usize> = EinCode::new(graph.ixs.clone(), graph.iy.clone());

    println!("{}", "=".repeat(70));
    println!("Benchmark: {}", graph.name);
    println!("  Description: {}", graph.description);
    println!("  Tensors: {}", graph.ixs.len());
    println!("  Indices: {}", sizes.len());
    println!();

    // ========== Greedy ==========
    println!("GreedyMethod:");
    let greedy = GreedyMethod::default();

    // Warmup
    let _ = optimize_code(&code, &sizes, &greedy);

    // Benchmark
    let start = Instant::now();
    let mut greedy_result = None;
    for _ in 0..10 {
        greedy_result = optimize_code(&code, &sizes, &greedy);
    }
    let greedy_time = start.elapsed().as_secs_f64();

    let greedy_tree = greedy_result.unwrap();
    let greedy_c = contraction_complexity(&greedy_tree, &sizes, &code.ixs);

    println!("  Time complexity (tc):       {:.6}", greedy_c.tc);
    println!("  Space complexity (sc):      {:.6}", greedy_c.sc);
    println!("  Read-write complexity (rwc): {:.6}", greedy_c.rwc);
    println!(
        "  Execution time (10 runs):   {:.2} ms",
        greedy_time * 1000.0
    );
    println!(
        "  Average per run:            {:.4} ms",
        greedy_time / 10.0 * 1000.0
    );
    println!();

    // ========== TreeSA ==========
    println!("TreeSA (ntrials={}, niters={}):", ntrials, niters);
    let treesa = TreeSA::default().with_ntrials(ntrials).with_niters(niters);

    // Warmup
    let _ = optimize_code(&code, &sizes, &treesa);

    // Benchmark
    let start = Instant::now();
    let mut treesa_result = None;
    for _ in 0..3 {
        treesa_result = optimize_code(&code, &sizes, &treesa);
    }
    let treesa_time = start.elapsed().as_secs_f64();

    let treesa_tree = treesa_result.unwrap();
    let treesa_c = contraction_complexity(&treesa_tree, &sizes, &code.ixs);

    println!("  Time complexity (tc):       {:.6}", treesa_c.tc);
    println!("  Space complexity (sc):      {:.6}", treesa_c.sc);
    println!("  Read-write complexity (rwc): {:.6}", treesa_c.rwc);
    println!(
        "  Execution time (3 runs):    {:.2} ms",
        treesa_time * 1000.0
    );
    println!(
        "  Average per run:            {:.2} ms",
        treesa_time / 3.0 * 1000.0
    );
    println!();

    // ========== Improvement ==========
    let tc_improvement = greedy_c.tc - treesa_c.tc;
    let sc_improvement = greedy_c.sc - treesa_c.sc;
    println!("  Improvement over Greedy:");
    println!(
        "    tc reduction: {:.2} ({:.1}%)",
        tc_improvement,
        tc_improvement / greedy_c.tc * 100.0
    );
    println!("    sc reduction: {:.2}", sc_improvement);
    println!();

    BenchmarkResult {
        name: graph.name.clone(),
        description: graph.description.clone(),
        tensors: graph.ixs.len(),
        indices: sizes.len(),
        greedy: MethodResult {
            tc: greedy_c.tc,
            sc: greedy_c.sc,
            rwc: greedy_c.rwc,
            avg_ms: greedy_time / 10.0 * 1000.0,
            total_ms: greedy_time * 1000.0,
            runs: 10,
            ntrials: None,
            niters: None,
        },
        treesa: MethodResult {
            tc: treesa_c.tc,
            sc: treesa_c.sc,
            rwc: treesa_c.rwc,
            avg_ms: treesa_time / 3.0 * 1000.0,
            total_ms: treesa_time * 1000.0,
            runs: 3,
            ntrials: Some(ntrials),
            niters: Some(niters),
        },
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!();
    println!("{}", "=".repeat(70));
    println!("Rust TreeSA Benchmark");
    println!("Backend: omeco (native Rust)");
    println!("{}", "=".repeat(70));
    println!();

    let graphs_dir = Path::new("benchmarks/graphs");
    let results_dir = Path::new("benchmarks/results");
    fs::create_dir_all(results_dir)?;

    let mut results = Vec::new();

    // Define which graphs to benchmark and their SA parameters
    let benchmarks = vec![
        ("chain_10.json", 1, 50),
        ("chain_20.json", 1, 50),
        ("grid_4x4.json", 1, 100),
        ("grid_5x5.json", 1, 100),
        ("grid_6x6.json", 1, 100),
        ("petersen.json", 1, 50),
        ("reg3_50.json", 1, 100),
        ("reg3_100.json", 1, 100),
        ("reg3_250.json", 1, 100),
    ];

    for (filename, ntrials, niters) in benchmarks {
        let path = graphs_dir.join(filename);
        if !path.exists() {
            println!("Skipping {} (not found)", filename);
            continue;
        }

        let graph = load_graph(&path)?;
        let result = benchmark_graph(&graph, ntrials, niters);
        results.push(result);
    }

    // ========== Summary Table ==========
    println!("{}", "=".repeat(70));
    println!("SUMMARY");
    println!("{}", "=".repeat(70));
    println!();
    println!(
        "{:<15} {:>8} {:>8} │ {:>10} {:>10} │ {:>10} {:>10}",
        "Graph", "Tensors", "Indices", "Greedy tc", "TreeSA tc", "Greedy ms", "TreeSA ms"
    );
    println!(
        "{}",
        "─".repeat(15)
            + &"─".repeat(8)
            + &"─".repeat(8)
            + "─┼"
            + &"─".repeat(10)
            + &"─".repeat(10)
            + "─┼"
            + &"─".repeat(10)
            + &"─".repeat(10)
    );
    for r in &results {
        println!(
            "{:<15} {:>8} {:>8} │ {:>10.2} {:>10.2} │ {:>10.3} {:>10.2}",
            r.name,
            r.tensors,
            r.indices,
            r.greedy.tc,
            r.treesa.tc,
            r.greedy.avg_ms,
            r.treesa.avg_ms
        );
    }
    println!();

    // ========== Save Results ==========
    // Save combined results
    let output = serde_json::json!({
        "language": "rust",
        "backend": "omeco (native Rust)",
        "results": results.iter().map(|r| (r.name.clone(), r)).collect::<HashMap<_, _>>(),
    });

    let output_path = results_dir.join("rust_results.json");
    fs::write(&output_path, serde_json::to_string_pretty(&output)?)?;

    println!("Results saved to:");
    println!("  {}", output_path.display());

    Ok(())
}
