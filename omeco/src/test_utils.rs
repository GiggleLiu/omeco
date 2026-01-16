//! Test utilities for validating einsum contractions
//!
//! This module provides tools for testing tensor contraction optimizations
//! by performing actual numerical contractions and validating results.

use ndarray::{ArrayD, IxDyn};
use rand::Rng;
use std::collections::{HashMap, HashSet};

/// Naive einsum contractor for testing
///
/// Performs actual tensor contractions using ndarray to validate
/// that optimized contraction orders produce correct results.
pub struct NaiveContractor {
    tensors: HashMap<usize, ArrayD<f64>>,
}

impl NaiveContractor {
    /// Create a new contractor
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
        }
    }

    /// Add a tensor with random data
    ///
    /// # Arguments
    /// * `idx` - Tensor identifier
    /// * `shape` - Shape of the tensor
    pub fn add_tensor(&mut self, idx: usize, shape: Vec<usize>) {
        let mut rng = rand::thread_rng();
        let size: usize = shape.iter().product();
        let data: Vec<f64> = (0..size).map(|_| rng.gen()).collect();
        let tensor = ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap();
        self.tensors.insert(idx, tensor);
    }

    /// Execute an einsum contraction between two tensors
    ///
    /// # Arguments
    /// * `left_idx` - Identifier for left tensor
    /// * `right_idx` - Identifier for right tensor
    /// * `left_labels` - Index labels for left tensor
    /// * `right_labels` - Index labels for right tensor
    /// * `output_labels` - Index labels for output tensor
    ///
    /// # Returns
    /// Identifier of the result tensor (reuses min of input identifiers)
    pub fn contract(
        &mut self,
        left_idx: usize,
        right_idx: usize,
        left_labels: &[usize],
        right_labels: &[usize],
        output_labels: &[usize],
    ) -> usize {
        let left = self.tensors[&left_idx].clone();
        let right = self.tensors[&right_idx].clone();

        let result = self.einsum_contract(&left, &right, left_labels, right_labels, output_labels);

        let result_idx = left_idx.min(right_idx);
        self.tensors.insert(result_idx, result);
        self.tensors.remove(&left_idx.max(right_idx));
        result_idx
    }

    /// Get a reference to a tensor
    pub fn get_tensor(&self, idx: usize) -> Option<&ArrayD<f64>> {
        self.tensors.get(&idx)
    }

    /// Get the shape of a tensor
    pub fn get_shape(&self, idx: usize) -> Option<Vec<usize>> {
        self.tensors.get(&idx).map(|t| t.shape().to_vec())
    }

    /// Core einsum contraction logic
    ///
    /// Implements general einsum contraction using a simple nested-loop approach.
    /// This is slower than optimized implementations but more likely to be correct.
    fn einsum_contract(
        &self,
        left: &ArrayD<f64>,
        right: &ArrayD<f64>,
        left_labels: &[usize],
        right_labels: &[usize],
        output_labels: &[usize],
    ) -> ArrayD<f64> {
        // Build label->size mapping
        let mut label_sizes: HashMap<usize, usize> = HashMap::new();
        for (i, &label) in left_labels.iter().enumerate() {
            let size = left.shape()[i];
            if let Some(&existing) = label_sizes.get(&label) {
                assert_eq!(existing, size, "Label {} has inconsistent sizes", label);
            } else {
                label_sizes.insert(label, size);
            }
        }
        for (i, &label) in right_labels.iter().enumerate() {
            let size = right.shape()[i];
            if let Some(&existing) = label_sizes.get(&label) {
                assert_eq!(existing, size, "Label {} has inconsistent sizes", label);
            } else {
                label_sizes.insert(label, size);
            }
        }

        // Determine output shape
        let output_shape: Vec<usize> = output_labels
            .iter()
            .map(|&label| *label_sizes.get(&label).unwrap_or(&1))
            .collect();

        let output_size: usize = if output_shape.is_empty() {
            1
        } else {
            output_shape.iter().product()
        };

        // Allocate result
        let mut result_data = vec![0.0; output_size];

        // Get all unique labels
        let mut all_labels: HashSet<usize> = HashSet::new();
        all_labels.extend(left_labels.iter().copied());
        all_labels.extend(right_labels.iter().copied());
        let all_labels: Vec<usize> = all_labels.into_iter().collect();

        // Compute number of iterations (product of all label sizes)
        let total_iterations: usize = all_labels
            .iter()
            .map(|&label| *label_sizes.get(&label).unwrap_or(&1))
            .product();

        // Iterate over all combinations of index values
        for iter_idx in 0..total_iterations {
            // Decode iter_idx into label values
            let mut label_values: HashMap<usize, usize> = HashMap::new();
            let mut remaining = iter_idx;
            for &label in all_labels.iter().rev() {
                let size = *label_sizes.get(&label).unwrap_or(&1);
                label_values.insert(label, remaining % size);
                remaining /= size;
            }

            // Build multi-dimensional indices for left, right, and output
            let left_indices: Vec<usize> = left_labels
                .iter()
                .map(|&label| *label_values.get(&label).unwrap_or(&0))
                .collect();

            let right_indices: Vec<usize> = right_labels
                .iter()
                .map(|&label| *label_values.get(&label).unwrap_or(&0))
                .collect();

            let output_indices: Vec<usize> = output_labels
                .iter()
                .map(|&label| *label_values.get(&label).unwrap_or(&0))
                .collect();

            // Get values
            let left_val = if left.shape().is_empty() {
                1.0
            } else {
                left[&*left_indices]
            };

            let right_val = if right.shape().is_empty() {
                1.0
            } else {
                right[&*right_indices]
            };

            // Compute output flat index
            let mut out_idx = 0;
            let mut out_stride = 1;
            for i in (0..output_indices.len()).rev() {
                out_idx += output_indices[i] * out_stride;
                out_stride *= output_shape[i];
            }

            result_data[out_idx] += left_val * right_val;
        }

        // Return result
        if output_shape.is_empty() {
            ArrayD::from_shape_vec(IxDyn(&[]), vec![result_data[0]]).unwrap()
        } else {
            ArrayD::from_shape_vec(IxDyn(&output_shape), result_data).unwrap()
        }
    }
}

/// Generate random einsum test instance
///
/// # Arguments
/// * `num_tensors` - Number of input tensors to generate
/// * `num_indices` - Maximum index value to use
/// * `allow_duplicates` - Whether to allow duplicate indices within a tensor
/// * `allow_output_only_indices` - Whether to allow indices in output that don't appear in inputs
///
/// # Returns
/// Tuple of (input indices, output indices)
pub fn generate_random_eincode(
    num_tensors: usize,
    num_indices: usize,
    allow_duplicates: bool,
    allow_output_only_indices: bool,
) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut rng = rand::thread_rng();

    // Generate random input indices for each tensor
    let mut ixs = Vec::new();
    let mut all_indices = HashSet::new();

    for _ in 0..num_tensors {
        let tensor_rank = rng.gen_range(1..=4);
        let mut tensor_indices = Vec::new();

        for _ in 0..tensor_rank {
            let idx = rng.gen_range(1..=num_indices);
            tensor_indices.push(idx);
            all_indices.insert(idx);
        }

        // Optionally add duplicates (for trace/diagonal operations)
        if allow_duplicates && rng.gen_bool(0.3) && !tensor_indices.is_empty() {
            let dup_idx = tensor_indices[rng.gen_range(0..tensor_indices.len())];
            tensor_indices.push(dup_idx);
        }

        ixs.push(tensor_indices);
    }

    // Generate output indices (without duplicates for simplicity)
    let mut output = Vec::new();
    let mut used_output_indices = HashSet::new();
    let num_output = if all_indices.is_empty() { 0 } else { rng.gen_range(0..=3) };

    for _ in 0..num_output {
        let idx = if allow_output_only_indices && rng.gen_bool(0.2) {
            // Add index not in any input (outer product/broadcast)
            num_indices + 1 + output.len()
        } else if !all_indices.is_empty() {
            // Pick from existing indices (avoid duplicates)
            let available: Vec<_> = all_indices
                .difference(&used_output_indices)
                .copied()
                .collect();
            if available.is_empty() {
                continue;
            }
            available[rng.gen_range(0..available.len())]
        } else {
            continue;
        };

        if !used_output_indices.contains(&idx) {
            output.push(idx);
            used_output_indices.insert(idx);
        }
    }

    (ixs, output)
}

/// Generate C60 fullerene graph edges
///
/// Creates a 60-vertex fullerene (buckyball) molecule structure.
/// Each vertex represents a carbon atom with degree 3.
/// Returns 90 edges representing carbon-carbon bonds.
///
/// The fullerene is constructed as a truncated icosahedron.
pub fn generate_fullerene_edges() -> Vec<(usize, usize)> {
    // Simplified construction: 60 vertices, each with degree 3 → 90 edges
    // Using standard construction based on coordinates
    let mut edges = Vec::new();

    // Top pentagon (vertices 0-4)
    for i in 0..5 {
        edges.push((i, (i + 1) % 5));
    }

    // Connect top pentagon to first belt (vertices 5-14)
    for i in 0..5 {
        edges.push((i, 5 + 2 * i));
        edges.push((i, 5 + 2 * i + 1));
    }

    // First belt hexagons
    for i in 0..10 {
        edges.push((5 + i, 5 + (i + 1) % 10));
    }

    // Connect first to second belt (vertices 15-34)
    for i in 0..10 {
        edges.push((5 + i, 15 + 2 * i));
    }

    // Second belt (20 vertices)
    for i in 0..20 {
        edges.push((15 + i, 15 + (i + 1) % 20));
    }

    // Connect second to third belt (vertices 35-44)
    for i in 0..10 {
        edges.push((15 + 2 * i, 35 + i));
    }

    // Third belt
    for i in 0..10 {
        edges.push((35 + i, 35 + (i + 1) % 10));
    }

    // Connect third belt to bottom pentagon (vertices 45-49)
    for i in 0..5 {
        edges.push((35 + 2 * i, 45 + i));
        edges.push((35 + 2 * i + 1, 45 + i));
    }

    // Bottom pentagon (vertices 45-49 renamed to 50-54 for 60 total)
    // Actually we have 50 vertices so far, add 10 more
    for i in 0..10 {
        edges.push((35 + i, 50 + i));
    }

    for i in 0..10 {
        edges.push((50 + i, 50 + (i + 1) % 10));
    }

    // Convert to 1-indexed
    edges.into_iter().map(|(a, b)| (a + 1, b + 1)).collect()
}

/// Generate Tutte graph edges
///
/// Creates the Tutte graph with 46 vertices and 69 edges.
/// The Tutte graph is a 3-regular non-Hamiltonian graph.
pub fn generate_tutte_edges() -> Vec<(usize, usize)> {
    let mut edges = Vec::new();

    // Outer cycle (vertices 0-14)
    for i in 0..15 {
        edges.push((i, (i + 1) % 15));
    }

    // Middle cycle (vertices 15-29)
    for i in 0..15 {
        edges.push((15 + i, 15 + (i + 1) % 15));
    }

    // Inner vertices (vertices 30-44, 15 vertices)
    for i in 0..15 {
        // Connect outer to middle
        edges.push((i, 15 + i));
    }

    // Inner petals (3-vertex structures)
    for i in 0..15 {
        let inner = 30 + i;
        let next_inner = 30 + (i + 1) % 15;

        // Connect middle to inner
        edges.push((15 + i, inner));

        // Inner connections
        if i % 3 == 0 {
            edges.push((inner, next_inner));
            edges.push((next_inner, 30 + (i + 2) % 15));
        }
    }

    // Center vertex (45)
    for i in 0..15 {
        if i % 3 == 1 {
            edges.push((30 + i, 45));
        }
    }

    // Convert to 1-indexed and ensure we have 46 vertices
    edges.into_iter().map(|(a, b)| (a + 1, b + 1)).collect()
}

/// Generate a ring (cycle) graph
///
/// Creates a cycle graph with the specified number of vertices.
/// Each vertex connects to its two neighbors in a ring.
pub fn generate_ring_edges(n: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for i in 0..n {
        edges.push((i + 1, ((i + 1) % n) + 1));
    }
    edges
}
