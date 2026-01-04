//! Greedy contraction order optimizer.
//!
//! The greedy algorithm iteratively contracts the tensor pair with the
//! minimum cost until all tensors are contracted into one.

use crate::eincode::{log2_size_dict, EinCode, NestedEinsum};
use crate::incidence_list::{ContractionDims, IncidenceList};
use crate::Label;
use priority_queue::PriorityQueue;
use rand::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;

/// A binary contraction tree built during greedy optimization.
#[derive(Debug, Clone)]
pub enum ContractionTree {
    /// A leaf representing an input tensor.
    Leaf(usize),
    /// A contraction of two subtrees.
    Node {
        left: Box<ContractionTree>,
        right: Box<ContractionTree>,
    },
}

impl ContractionTree {
    /// Create a leaf node.
    pub fn leaf(idx: usize) -> Self {
        Self::Leaf(idx)
    }

    /// Create an internal node.
    pub fn node(left: ContractionTree, right: ContractionTree) -> Self {
        Self::Node {
            left: Box::new(left),
            right: Box::new(right),
        }
    }
}

/// Configuration for the greedy optimizer.
#[derive(Debug, Clone)]
pub struct GreedyMethod {
    /// Weight balancing output size vs input size reduction.
    /// - α = 0.0: Minimize output tensor size (default)
    /// - α = 1.0: Maximize input tensor size reduction
    pub alpha: f64,
    /// Temperature for stochastic selection.
    /// - temperature = 0.0: Deterministic greedy (default)
    /// - temperature > 0.0: Boltzmann sampling
    pub temperature: f64,
}

impl Default for GreedyMethod {
    fn default() -> Self {
        Self {
            alpha: 0.0,
            temperature: 0.0,
        }
    }
}

impl GreedyMethod {
    /// Create a new greedy method with custom parameters.
    pub fn new(alpha: f64, temperature: f64) -> Self {
        Self { alpha, temperature }
    }

    /// Create a stochastic greedy method with given temperature.
    pub fn stochastic(temperature: f64) -> Self {
        Self {
            alpha: 0.0,
            temperature,
        }
    }
}

/// Cost value wrapper for the priority queue (min-heap behavior).
#[derive(Debug, Clone, Copy)]
struct Cost(f64);

impl PartialEq for Cost {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Cost {}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cost {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
    }
}

/// Compute the greedy loss function for contracting two tensors.
///
/// Loss = size(output) - α * (size(input1) + size(input2))
/// where sizes are in linear scale (2^log2_size).
fn greedy_loss(dims: &ContractionDims<impl Clone + Eq + std::hash::Hash>, alpha: f64) -> f64 {
    let output_size = 2_f64.powf(dims.d01 + dims.d02 + dims.d012);
    let input1_size = 2_f64.powf(dims.d01 + dims.d12 + dims.d012);
    let input2_size = 2_f64.powf(dims.d02 + dims.d12 + dims.d012);
    output_size - alpha * (input1_size + input2_size)
}

/// Result of the greedy optimization.
#[derive(Debug, Clone)]
pub struct GreedyResult<E> {
    /// The contraction tree
    pub tree: ContractionTree,
    /// Log2 time complexities for each contraction step
    pub log2_tcs: Vec<f64>,
    /// Log2 space complexities for each contraction step
    pub log2_scs: Vec<f64>,
    /// Final output edges
    pub output_edges: Vec<E>,
}

/// Run the greedy contraction algorithm.
pub fn tree_greedy<E: Label>(
    il: &IncidenceList<usize, E>,
    log2_sizes: &HashMap<E, f64>,
    alpha: f64,
    temperature: f64,
) -> Option<GreedyResult<E>> {
    let mut il = il.clone();
    let n = il.nv();

    if n == 0 {
        return None;
    }

    if n == 1 {
        let v = *il.vertices().next()?;
        return Some(GreedyResult {
            tree: ContractionTree::leaf(v),
            log2_tcs: Vec::new(),
            log2_scs: Vec::new(),
            output_edges: il.edges(&v).cloned().unwrap_or_default(),
        });
    }

    let mut rng = rand::rng();
    let mut log2_tcs = Vec::new();
    let mut log2_scs = Vec::new();

    // Map vertex to its current tree
    let mut trees: HashMap<usize, ContractionTree> = il
        .vertices()
        .map(|&v| (v, ContractionTree::leaf(v)))
        .collect();

    // Initialize priority queue with all pairs
    let mut pq = PriorityQueue::new();
    let vertices: Vec<usize> = il.vertices().cloned().collect();

    for (i, &vi) in vertices.iter().enumerate() {
        for &vj in &vertices[i + 1..] {
            if il.are_neighbors(&vi, &vj) {
                let dims = ContractionDims::compute(&il, log2_sizes, &vi, &vj);
                let loss = greedy_loss(&dims, alpha);
                pq.push((vi.min(vj), vi.max(vj)), Cost(loss));
            }
        }
    }

    // Track which vertex represents merged tensors
    let mut next_vertex = vertices.iter().max().copied().unwrap_or(0) + 1;

    // Main greedy loop
    while il.nv() > 1 && !pq.is_empty() {
        // Select pair to contract
        let (pair, _) = select_pair(&mut pq, temperature, &mut rng)?;
        let (vi, vj) = pair;

        // Check if both vertices still exist
        if il.edges(&vi).is_none() || il.edges(&vj).is_none() {
            continue;
        }

        // Compute contraction dimensions
        let dims = ContractionDims::compute(&il, log2_sizes, &vi, &vj);

        // Record complexity
        log2_tcs.push(dims.time_complexity());
        log2_scs.push(dims.space_complexity());

        // Build new tree
        let tree_i = trees.remove(&vi)?;
        let tree_j = trees.remove(&vj)?;
        let new_tree = ContractionTree::node(tree_i, tree_j);

        // Contract in the incidence list
        let new_v = next_vertex;
        next_vertex += 1;

        // Set edges for the new vertex (output edges of the contraction)
        il.set_edges(new_v, dims.edges_out.clone());

        // Remove contracted edges
        il.remove_edges(&dims.edges_remove);

        // Delete old vertices
        il.delete_vertex(&vi);
        il.delete_vertex(&vj);

        // Store the new tree
        trees.insert(new_v, new_tree);

        // Update costs for neighbors of the new vertex
        for neighbor in il.neighbors(&new_v) {
            let pair_key = (new_v.min(neighbor), new_v.max(neighbor));
            let new_dims = ContractionDims::compute(&il, log2_sizes, &new_v, &neighbor);
            let loss = greedy_loss(&new_dims, alpha);
            pq.push(pair_key, Cost(loss));
        }
    }

    // Get the final tree
    let final_tree = trees.into_values().next()?;
    let output_edges = il
        .vertices()
        .next()
        .and_then(|v| il.edges(v).cloned())
        .unwrap_or_default();

    Some(GreedyResult {
        tree: final_tree,
        log2_tcs,
        log2_scs,
        output_edges,
    })
}

/// Select the next pair to contract from the priority queue.
fn select_pair<R: Rng>(
    pq: &mut PriorityQueue<(usize, usize), Cost>,
    temperature: f64,
    rng: &mut R,
) -> Option<((usize, usize), Cost)> {
    if pq.is_empty() {
        return None;
    }

    let (pair1, cost1) = pq.pop()?;

    if temperature <= 0.0 || pq.is_empty() {
        return Some((pair1, cost1));
    }

    // Boltzmann sampling: consider the second-best option
    let (pair2, cost2) = pq.pop()?;

    // Probability of accepting the worse option
    let delta = cost2.0 - cost1.0;
    let prob = (-delta / temperature).exp();

    if rng.random::<f64>() < prob {
        // Accept the second option
        pq.push(pair1, cost1);
        Some((pair2, cost2))
    } else {
        // Keep the first option
        pq.push(pair2, cost2);
        Some((pair1, cost1))
    }
}

/// Convert a contraction tree to a NestedEinsum.
pub fn tree_to_nested_einsum<L: Label>(
    tree: &ContractionTree,
    original_ixs: &[Vec<L>],
    output_iy: &[L],
) -> NestedEinsum<L> {
    // First, collect all leaf indices to build the mapping
    let mut leaf_labels: HashMap<usize, Vec<L>> = HashMap::new();
    collect_leaf_labels(tree, original_ixs, &mut leaf_labels);

    // Then recursively build the nested einsum
    build_nested(tree, &leaf_labels, output_iy)
}

fn collect_leaf_labels<L: Label>(
    tree: &ContractionTree,
    original_ixs: &[Vec<L>],
    labels: &mut HashMap<usize, Vec<L>>,
) {
    match tree {
        ContractionTree::Leaf(idx) => {
            if let Some(ix) = original_ixs.get(*idx) {
                labels.insert(*idx, ix.clone());
            }
        }
        ContractionTree::Node { left, right } => {
            collect_leaf_labels(left, original_ixs, labels);
            collect_leaf_labels(right, original_ixs, labels);
        }
    }
}

fn build_nested<L: Label>(
    tree: &ContractionTree,
    leaf_labels: &HashMap<usize, Vec<L>>,
    final_output: &[L],
) -> NestedEinsum<L> {
    match tree {
        ContractionTree::Leaf(idx) => NestedEinsum::leaf(*idx),
        ContractionTree::Node { left, right } => {
            // Get labels from children
            let left_labels = get_subtree_labels(left, leaf_labels);
            let right_labels = get_subtree_labels(right, leaf_labels);

            // Compute output labels for this contraction
            let output_labels =
                compute_contraction_output(&left_labels, &right_labels, final_output);

            // Build children
            let left_nested = build_nested(left, leaf_labels, final_output);
            let right_nested = build_nested(right, leaf_labels, final_output);

            // Create the einsum code for this contraction
            let eins = EinCode::new(vec![left_labels, right_labels], output_labels);

            NestedEinsum::node(vec![left_nested, right_nested], eins)
        }
    }
}

fn get_subtree_labels<L: Label>(
    tree: &ContractionTree,
    leaf_labels: &HashMap<usize, Vec<L>>,
) -> Vec<L> {
    match tree {
        ContractionTree::Leaf(idx) => leaf_labels.get(idx).cloned().unwrap_or_default(),
        ContractionTree::Node { left, right } => {
            let left_labels = get_subtree_labels(left, leaf_labels);
            let right_labels = get_subtree_labels(right, leaf_labels);
            compute_contraction_output(&left_labels, &right_labels, &[])
        }
    }
}

fn compute_contraction_output<L: Label>(left: &[L], right: &[L], final_output: &[L]) -> Vec<L> {
    use std::collections::HashSet;

    let left_set: HashSet<_> = left.iter().cloned().collect();
    let right_set: HashSet<_> = right.iter().cloned().collect();
    let final_set: HashSet<_> = final_output.iter().cloned().collect();

    let mut output = Vec::new();

    // Include labels that appear in only one input (external)
    // or appear in both inputs and are in the final output
    for l in left {
        if (!right_set.contains(l) || final_set.contains(l)) && !output.contains(l) {
            output.push(l.clone());
        }
    }
    for l in right {
        if (!left_set.contains(l) || final_set.contains(l)) && !output.contains(l) {
            output.push(l.clone());
        }
    }

    output
}

/// Optimize an EinCode using the greedy method.
pub fn optimize_greedy<L: Label>(
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    config: &GreedyMethod,
) -> Option<NestedEinsum<L>> {
    let il: IncidenceList<usize, L> = IncidenceList::<usize, L>::from_eincode(&code.ixs, &code.iy);
    let log2_sizes = log2_size_dict(size_dict);

    let result = tree_greedy(&il, &log2_sizes, config.alpha, config.temperature)?;
    Some(tree_to_nested_einsum(&result.tree, &code.ixs, &code.iy))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_method_default() {
        let method = GreedyMethod::default();
        assert_eq!(method.alpha, 0.0);
        assert_eq!(method.temperature, 0.0);
    }

    #[test]
    fn test_greedy_method_new() {
        let method = GreedyMethod::new(0.5, 1.0);
        assert_eq!(method.alpha, 0.5);
        assert_eq!(method.temperature, 1.0);
    }

    #[test]
    fn test_greedy_method_stochastic() {
        let method = GreedyMethod::stochastic(2.5);
        assert_eq!(method.alpha, 0.0);
        assert_eq!(method.temperature, 2.5);
    }

    #[test]
    fn test_contraction_tree_leaf() {
        let leaf = ContractionTree::leaf(42);
        assert!(matches!(leaf, ContractionTree::Leaf(42)));
    }

    #[test]
    fn test_contraction_tree_node() {
        let left = ContractionTree::leaf(0);
        let right = ContractionTree::leaf(1);
        let node = ContractionTree::node(left, right);
        assert!(matches!(node, ContractionTree::Node { .. }));
    }

    #[test]
    fn test_greedy_empty() {
        let il: IncidenceList<usize, char> = IncidenceList::new(HashMap::new(), vec![]);
        let log2_sizes: HashMap<char, f64> = HashMap::new();

        let result = tree_greedy(&il, &log2_sizes, 0.0, 0.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_greedy_single_tensor() {
        let ixs = vec![vec!['i', 'j']];
        let iy = vec!['i', 'j'];
        let il = IncidenceList::<usize, char>::from_eincode(&ixs, &iy);

        let mut log2_sizes = HashMap::new();
        log2_sizes.insert('i', 2.0);
        log2_sizes.insert('j', 2.0);

        let result = tree_greedy(&il, &log2_sizes, 0.0, 0.0);
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(matches!(result.tree, ContractionTree::Leaf(0)));
    }

    #[test]
    fn test_greedy_two_tensors() {
        let ixs = vec![vec!['i', 'j'], vec!['j', 'k']];
        let iy = vec!['i', 'k'];
        let il = IncidenceList::<usize, char>::from_eincode(&ixs, &iy);

        let mut log2_sizes = HashMap::new();
        log2_sizes.insert('i', 2.0);
        log2_sizes.insert('j', 3.0);
        log2_sizes.insert('k', 2.0);

        let result = tree_greedy(&il, &log2_sizes, 0.0, 0.0);
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(matches!(result.tree, ContractionTree::Node { .. }));
        assert_eq!(result.log2_tcs.len(), 1);
        assert_eq!(result.log2_scs.len(), 1);
    }

    #[test]
    fn test_greedy_chain() {
        // Chain: A[i,j] * B[j,k] * C[k,l] -> [i,l]
        let ixs = vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']];
        let iy = vec!['i', 'l'];
        let il = IncidenceList::<usize, char>::from_eincode(&ixs, &iy);

        let mut log2_sizes = HashMap::new();
        log2_sizes.insert('i', 2.0);
        log2_sizes.insert('j', 3.0);
        log2_sizes.insert('k', 3.0);
        log2_sizes.insert('l', 2.0);

        let result = tree_greedy(&il, &log2_sizes, 0.0, 0.0);
        assert!(result.is_some());
        let result = result.unwrap();
        // Should have 2 contractions for 3 tensors
        assert_eq!(result.log2_tcs.len(), 2);
    }

    #[test]
    fn test_greedy_with_alpha() {
        let ixs = vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']];
        let iy = vec!['i', 'l'];
        let il = IncidenceList::<usize, char>::from_eincode(&ixs, &iy);

        let mut log2_sizes = HashMap::new();
        log2_sizes.insert('i', 2.0);
        log2_sizes.insert('j', 3.0);
        log2_sizes.insert('k', 3.0);
        log2_sizes.insert('l', 2.0);

        // Test with alpha = 0.5
        let result = tree_greedy(&il, &log2_sizes, 0.5, 0.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_greedy_with_temperature() {
        let ixs = vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']];
        let iy = vec!['i', 'l'];
        let il = IncidenceList::<usize, char>::from_eincode(&ixs, &iy);

        let mut log2_sizes = HashMap::new();
        log2_sizes.insert('i', 2.0);
        log2_sizes.insert('j', 3.0);
        log2_sizes.insert('k', 3.0);
        log2_sizes.insert('l', 2.0);

        // Test with positive temperature (stochastic)
        let result = tree_greedy(&il, &log2_sizes, 0.0, 1.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_optimize_greedy() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 4);

        let config = GreedyMethod::default();
        let result = optimize_greedy(&code, &size_dict, &config);

        assert!(result.is_some());
        let nested = result.unwrap();
        assert!(nested.is_binary());
        assert_eq!(nested.leaf_count(), 2);
    }

    #[test]
    fn test_optimize_greedy_stochastic() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let mut size_dict = HashMap::new();
        size_dict.insert('i', 4);
        size_dict.insert('j', 8);
        size_dict.insert('k', 4);

        let config = GreedyMethod::stochastic(1.0);
        let result = optimize_greedy(&code, &size_dict, &config);

        assert!(result.is_some());
    }

    #[test]
    fn test_tree_to_nested_einsum() {
        let tree = ContractionTree::node(ContractionTree::leaf(0), ContractionTree::leaf(1));
        let ixs = vec![vec!['i', 'j'], vec!['j', 'k']];
        let iy = vec!['i', 'k'];

        let nested = tree_to_nested_einsum(&tree, &ixs, &iy);
        assert!(nested.is_binary());
        assert_eq!(nested.leaf_count(), 2);
    }

    #[test]
    fn test_tree_to_nested_einsum_chain() {
        // ((0,1),2)
        let inner = ContractionTree::node(ContractionTree::leaf(0), ContractionTree::leaf(1));
        let tree = ContractionTree::node(inner, ContractionTree::leaf(2));
        let ixs = vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']];
        let iy = vec!['i', 'l'];

        let nested = tree_to_nested_einsum(&tree, &ixs, &iy);
        assert!(nested.is_binary());
        assert_eq!(nested.leaf_count(), 3);
    }

    #[test]
    fn test_cost_ordering() {
        // Test that Cost implements correct min-heap ordering
        let cost1 = Cost(1.0);
        let cost2 = Cost(2.0);

        // Lower cost should have higher priority (reverse ordering)
        assert!(cost1 > cost2);
        assert!(cost2 < cost1);
        assert!(cost1 == Cost(1.0));
    }

    #[test]
    fn test_greedy_disconnected_tensors() {
        // Two tensors that don't share any indices
        let ixs = vec![vec!['i', 'j'], vec!['k', 'l']];
        let iy = vec!['i', 'j', 'k', 'l'];
        let il = IncidenceList::<usize, char>::from_eincode(&ixs, &iy);

        let mut log2_sizes = HashMap::new();
        log2_sizes.insert('i', 2.0);
        log2_sizes.insert('j', 2.0);
        log2_sizes.insert('k', 2.0);
        log2_sizes.insert('l', 2.0);

        let result = tree_greedy(&il, &log2_sizes, 0.0, 0.0);
        // Even disconnected tensors should produce a result
        assert!(result.is_some());
    }

    #[test]
    fn test_greedy_trace() {
        // Trace operation: contract all indices
        let ixs = vec![vec!['i', 'j'], vec!['j', 'i']];
        let iy = vec![];
        let il = IncidenceList::<usize, char>::from_eincode(&ixs, &iy);

        let mut log2_sizes = HashMap::new();
        log2_sizes.insert('i', 2.0);
        log2_sizes.insert('j', 2.0);

        let result = tree_greedy(&il, &log2_sizes, 0.0, 0.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_compute_contraction_output_no_final() {
        // Test contraction output with empty final output
        let output = compute_contraction_output(&['i', 'j'], &['j', 'k'], &[]);
        assert!(output.contains(&'i'));
        assert!(output.contains(&'k'));
        assert!(!output.contains(&'j')); // contracted
    }

    #[test]
    fn test_compute_contraction_output_with_batched() {
        // Test with batched index (appears in both inputs and output)
        let output =
            compute_contraction_output(&['i', 'j', 'b'], &['j', 'k', 'b'], &['i', 'k', 'b']);
        assert!(output.contains(&'i'));
        assert!(output.contains(&'k'));
        assert!(output.contains(&'b')); // batched, kept
        assert!(!output.contains(&'j')); // contracted
    }
}
