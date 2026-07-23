//! measure <graph.json> <tree.json> — recompute tc/sc of an emitted tree.
use omeco::json::{readjson, ContractionOrder};
use omeco::{contraction_complexity, NestedEinsum};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct GraphData {
    ixs: Vec<Vec<usize>>,
    #[allow(dead_code)]
    iy: Vec<usize>,
    sizes: HashMap<String, usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let graph: GraphData = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let sizes: HashMap<usize, usize> = graph
        .sizes
        .iter()
        .map(|(k, v)| (k.parse::<usize>().unwrap(), *v))
        .collect();
    let order: ContractionOrder<usize> = readjson(&args[2])?;
    let tree: NestedEinsum<usize> = match order {
        ContractionOrder::Nested(t) => t,
        ContractionOrder::Sliced(_) => return Err("sliced tree not supported".into()),
    };
    let cc = contraction_complexity(&tree, &sizes, &graph.ixs);
    println!("tc={:.4} sc={:.4} rwc={:.4}", cc.tc, cc.sc, cc.rwc);
    Ok(())
}
