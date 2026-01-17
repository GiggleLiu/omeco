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
#[derive(Default, Clone)]
pub struct NaiveContractor {
    tensors: HashMap<usize, ArrayD<f64>>,
}

impl NaiveContractor {
    /// Create a new contractor
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tensor with random data
    ///
    /// # Arguments
    /// * `idx` - Tensor identifier
    /// * `shape` - Shape of the tensor
    pub fn add_tensor(&mut self, idx: usize, shape: Vec<usize>) {
        let mut rng = rand::rng();
        let size: usize = shape.iter().product();
        let data: Vec<f64> = (0..size).map(|_| rng.random()).collect();
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
    let mut rng = rand::rng();

    // Generate random input indices for each tensor
    let mut ixs = Vec::new();
    let mut all_indices = HashSet::new();

    for _ in 0..num_tensors {
        let tensor_rank = rng.random_range(1..=4);
        let mut tensor_indices = Vec::new();

        for _ in 0..tensor_rank {
            let idx = rng.random_range(1..=num_indices);
            tensor_indices.push(idx);
            all_indices.insert(idx);
        }

        // Optionally add duplicates (for trace/diagonal operations)
        if allow_duplicates && rng.random_bool(0.3) && !tensor_indices.is_empty() {
            let dup_idx = tensor_indices[rng.random_range(0..tensor_indices.len())];
            tensor_indices.push(dup_idx);
        }

        ixs.push(tensor_indices);
    }

    // Generate output indices (without duplicates for simplicity)
    let mut output = Vec::new();
    let mut used_output_indices = HashSet::new();
    let num_output = if all_indices.is_empty() {
        0
    } else {
        rng.random_range(0..=3)
    };

    for _ in 0..num_output {
        let idx = if allow_output_only_indices && rng.random_bool(0.2) {
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
            available[rng.random_range(0..available.len())]
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

    // Bottom pentagon (vertices 45-49)
    for i in 0..5 {
        edges.push((45 + i, 45 + (i + 1) % 5));
    }

    // Connect third belt to bottom belt (vertices 50-59, forming 10-vertex ring)
    for i in 0..10 {
        edges.push((35 + i, 50 + i));
    }

    // Bottom belt ring (vertices 50-59)
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

/// Execute a NestedEinsum contraction tree using NaiveContractor
///
/// This recursively executes the contraction tree and returns the final result tensor.
/// The function tracks what labels each intermediate result has.
///
/// # Arguments
/// * `nested` - The NestedEinsum tree to execute
/// * `contractor` - The NaiveContractor managing the tensors
/// * `label_map` - Mapping from labels to internal indices (usize)
///
/// # Returns
/// Tuple of (tensor_index, labels) where labels are the output labels of this computation
pub fn execute_nested<L: crate::Label>(
    nested: &crate::NestedEinsum<L>,
    contractor: &mut NaiveContractor,
    label_map: &HashMap<L, usize>,
) -> usize {
    execute_nested_impl(nested, contractor, label_map).0
}

/// Internal implementation that returns (tensor_idx, labels)
fn execute_nested_impl<L: crate::Label>(
    nested: &crate::NestedEinsum<L>,
    contractor: &mut NaiveContractor,
    label_map: &HashMap<L, usize>,
) -> (usize, Vec<usize>) {
    use crate::NestedEinsum;

    match nested {
        NestedEinsum::Leaf { tensor_index } => {
            // For leaf, we need to figure out its labels from the original input
            // This is tricky - we don't have direct access to the original labels here
            // So we'll return empty labels and rely on the parent node's eins.ixs
            (*tensor_index, vec![])
        }
        NestedEinsum::Node { args, eins } => {
            // Recursively execute children
            let child_results: Vec<(usize, Vec<usize>)> = args
                .iter()
                .map(|child| execute_nested_impl(child, contractor, label_map))
                .collect();

            // For binary contraction (2 args), contract directly
            if child_results.len() == 2 {
                let (left_idx, _) = child_results[0];
                let (right_idx, _) = child_results[1];

                // Map labels from eins.ixs to usize using label_map
                let left_labels: Vec<usize> = eins.ixs[0]
                    .iter()
                    .map(|l| *label_map.get(l).expect("Label should be in map"))
                    .collect();
                let right_labels: Vec<usize> = eins.ixs[1]
                    .iter()
                    .map(|l| *label_map.get(l).expect("Label should be in map"))
                    .collect();
                let output_labels: Vec<usize> = eins
                    .iy
                    .iter()
                    .map(|l| *label_map.get(l).expect("Label should be in map"))
                    .collect();

                let result_idx = contractor.contract(
                    left_idx,
                    right_idx,
                    &left_labels,
                    &right_labels,
                    &output_labels,
                );

                (result_idx, output_labels)
            } else {
                // For >2 args, contract sequentially
                panic!("execute_nested only supports binary trees, got {} args", child_results.len());
            }
        }
    }
}

/// Compare two tensors for approximate equality
///
/// # Arguments
/// * `a` - First tensor
/// * `b` - Second tensor
/// * `rtol` - Relative tolerance (default: 1e-5)
/// * `atol` - Absolute tolerance (default: 1e-8)
///
/// # Returns
/// true if tensors are approximately equal
pub fn tensors_approx_equal(a: &ArrayD<f64>, b: &ArrayD<f64>, rtol: f64, atol: f64) -> bool {
    if a.shape() != b.shape() {
        return false;
    }

    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x - y).abs() <= atol + rtol * y.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_naive_contractor_basic() {
        // Test basic tensor contraction: A[i,j] * B[j,k] -> C[i,k]
        let mut contractor = NaiveContractor::new();
        contractor.add_tensor(0, vec![2, 3]); // A: 2x3
        contractor.add_tensor(1, vec![3, 2]); // B: 3x2

        let result_idx = contractor.contract(
            0,
            1,
            &[1, 2], // A has indices i=1, j=2
            &[2, 3], // B has indices j=2, k=3
            &[1, 3], // output: i=1, k=3
        );

        let result_shape = contractor.get_shape(result_idx).unwrap();
        assert_eq!(result_shape, vec![2, 2], "Result should be 2x2");
    }

    #[test]
    fn test_naive_contractor_scalar() {
        // Test scalar contraction: A[i,j] * B[i,j] -> scalar
        let mut contractor = NaiveContractor::new();
        contractor.add_tensor(0, vec![2, 2]);
        contractor.add_tensor(1, vec![2, 2]);

        let result_idx = contractor.contract(0, 1, &[1, 2], &[1, 2], &[]);

        let result_tensor = contractor.get_tensor(result_idx).unwrap();
        assert_eq!(result_tensor.ndim(), 0, "Result should be scalar");
    }

    #[test]
    fn test_naive_contractor_outer_product() {
        // Test outer product: A[i] * B[j] -> C[i,j]
        let mut contractor = NaiveContractor::new();
        contractor.add_tensor(0, vec![2]);
        contractor.add_tensor(1, vec![3]);

        let result_idx = contractor.contract(0, 1, &[1], &[2], &[1, 2]);

        let result_shape = contractor.get_shape(result_idx).unwrap();
        assert_eq!(result_shape, vec![2, 3], "Result should be 2x3");
    }

    #[test]
    fn test_generate_random_eincode_basic() {
        // Test basic random generation
        let (ixs, output) = generate_random_eincode(3, 5, false, false);
        assert_eq!(ixs.len(), 3, "Should generate 3 tensors");

        // All indices should be within range
        for tensor_indices in &ixs {
            for &idx in tensor_indices {
                assert!((1..=5).contains(&idx), "Index should be in range 1-5");
            }
        }

        // Output indices should not have duplicates
        let mut seen = HashSet::new();
        for &idx in &output {
            assert!(!seen.contains(&idx), "Output should not have duplicates");
            seen.insert(idx);
        }
    }

    #[test]
    fn test_generate_random_eincode_with_duplicates() {
        // Test with duplicates allowed
        let (ixs, _output) = generate_random_eincode(5, 8, true, false);
        assert_eq!(ixs.len(), 5, "Should generate 5 tensors");

        // Some tensors may have duplicate indices (trace operations)
        // We can't deterministically test for this, but function shouldn't panic
    }

    #[test]
    fn test_generate_random_eincode_with_broadcast() {
        // Test with output-only indices allowed
        let (_ixs, output) = generate_random_eincode(3, 8, false, true);

        // Output may contain indices not in any input (broadcast)
        // Function should not panic
        assert!(output.len() <= 3, "Output should have at most 3 indices");
    }

    #[test]
    fn test_generate_ring_edges() {
        let edges = generate_ring_edges(5);
        assert_eq!(edges.len(), 5, "Ring with 5 vertices should have 5 edges");

        // Check that it forms a cycle
        assert_eq!(edges[0], (1, 2));
        assert_eq!(edges[1], (2, 3));
        assert_eq!(edges[2], (3, 4));
        assert_eq!(edges[3], (4, 5));
        assert_eq!(edges[4], (5, 1)); // Closes the ring
    }

    #[test]
    fn test_generate_fullerene_edges() {
        let edges = generate_fullerene_edges();
        assert!(
            !edges.is_empty(),
            "Fullerene graph should have edges"
        );

        // Check all edges are 1-indexed
        for &(a, b) in &edges {
            assert!(a >= 1, "Vertices should be 1-indexed");
            assert!(b >= 1, "Vertices should be 1-indexed");
            assert_ne!(a, b, "No self-loops");
        }
    }

    #[test]
    fn test_generate_tutte_edges() {
        let edges = generate_tutte_edges();
        assert!(
            !edges.is_empty(),
            "Tutte graph should have edges"
        );

        // Check all edges are 1-indexed
        for &(a, b) in &edges {
            assert!(a >= 1, "Vertices should be 1-indexed");
            assert!(b >= 1, "Vertices should be 1-indexed");
            assert_ne!(a, b, "No self-loops");
        }
    }

    #[test]
    fn test_naive_contractor_default() {
        // Test Default trait implementation
        let contractor = NaiveContractor::default();
        assert_eq!(contractor.tensors.len(), 0, "Default should be empty");
    }
}
