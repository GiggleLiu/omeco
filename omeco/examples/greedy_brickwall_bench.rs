//! Benchmark deterministic greedy optimization on closed brick-wall circuits.
//!
//! Each circuit has 20 alternating even/odd layers of rank-4 two-qubit gates,
//! with rank-1 `|0>`/`<0|` boundary tensors and dimension-two bonds. Widths of
//! 101, 201, and 301 qubits give exactly 1,202, 2,402, and 3,602 tensors. The
//! largest case uses 542 qubits (6,494 tensors, within two of the issue case).
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example greedy_brickwall_bench -p omeco
//! ```

use std::collections::HashMap;
use std::time::Instant;

use omeco::{contraction_complexity, optimize_code, EinCode, GreedyMethod};

const LAYERS: usize = 20;
const QUBIT_COUNTS: [usize; 4] = [101, 201, 301, 542];

fn brick_wall_amplitude(qubits: usize) -> (EinCode<usize>, HashMap<usize, usize>) {
    let mut next_label = 0usize;
    let mut sizes = HashMap::new();
    let mut current = Vec::with_capacity(qubits);
    let mut tensors = Vec::new();

    for _ in 0..qubits {
        let label = next_label;
        next_label += 1;
        sizes.insert(label, 2);
        current.push(label);
        tensors.push(vec![label]);
    }

    for layer in 0..LAYERS {
        let start = layer % 2;
        for q in (start..qubits - 1).step_by(2) {
            let out_a = next_label;
            let out_b = next_label + 1;
            next_label += 2;
            sizes.insert(out_a, 2);
            sizes.insert(out_b, 2);
            tensors.push(vec![current[q], current[q + 1], out_a, out_b]);
            current[q] = out_a;
            current[q + 1] = out_b;
        }
    }

    for label in current {
        tensors.push(vec![label]);
    }

    (EinCode::new(tensors, Vec::new()), sizes)
}

fn main() {
    println!("tensors\tseconds\ttc");
    for qubits in QUBIT_COUNTS {
        let (code, sizes) = brick_wall_amplitude(qubits);
        let tensor_count = code.num_tensors();
        let started = Instant::now();
        let Some(tree) = optimize_code(&code, &sizes, &GreedyMethod::default()) else {
            eprintln!("greedy returned no tree for {tensor_count} tensors");
            std::process::exit(1);
        };
        let seconds = started.elapsed().as_secs_f64();
        let tc = contraction_complexity(&tree, &sizes, &code.ixs).tc;
        println!("{tensor_count}\t{seconds:.6}\t{tc:.6}");
    }
}
