//! Expression tree for simulated annealing optimization.
//!
//! The ExprTree represents a binary contraction tree with mutable structure
//! that can be modified by applying local mutation rules.

use crate::utils::{fast_log2sumexp2, fast_log2sumexp2_3};

/// Cached complexity values for a subtree.
#[derive(Debug, Clone, Copy, Default)]
pub struct CachedComplexity {
    /// Time complexity (log2)
    pub tc: f64,
    /// Space complexity (log2)
    pub sc: f64,
    /// Read-write complexity (log2)
    pub rw: f64,
}

/// Information about an expression tree node.
#[derive(Debug, Clone)]
pub struct ExprInfo {
    /// Output dimension labels (as integer indices)
    pub out_dims: Vec<usize>,
    /// Tensor ID if this is a leaf node, None otherwise
    pub tensor_id: Option<usize>,
    /// Cached complexity values for this subtree (computed lazily)
    pub cached: Option<CachedComplexity>,
}

impl ExprInfo {
    /// Create info for an internal node.
    pub fn internal(out_dims: Vec<usize>) -> Self {
        Self {
            out_dims,
            tensor_id: None,
            cached: None,
        }
    }

    /// Create info for a leaf node.
    pub fn leaf(out_dims: Vec<usize>, tensor_id: usize) -> Self {
        Self {
            out_dims,
            tensor_id: Some(tensor_id),
            cached: None,
        }
    }
}

/// A mutable binary expression tree for simulated annealing.
///
/// Unlike NestedEinsum, this structure is designed for efficient
/// tree mutations during the optimization process.
#[derive(Debug, Clone)]
pub enum ExprTree {
    /// A leaf node representing an input tensor.
    Leaf(ExprInfo),
    /// An internal node with left and right children.
    Node {
        left: Box<ExprTree>,
        right: Box<ExprTree>,
        info: ExprInfo,
    },
}

impl ExprTree {
    /// Create a leaf node.
    pub fn leaf(out_dims: Vec<usize>, tensor_id: usize) -> Self {
        Self::Leaf(ExprInfo::leaf(out_dims, tensor_id))
    }

    /// Create an internal node.
    pub fn node(left: ExprTree, right: ExprTree, out_dims: Vec<usize>) -> Self {
        Self::Node {
            left: Box::new(left),
            right: Box::new(right),
            info: ExprInfo::internal(out_dims),
        }
    }

    /// Check if this is a leaf node.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf(_))
    }

    /// Get the output labels for this subtree.
    pub fn labels(&self) -> &[usize] {
        match self {
            Self::Leaf(info) | Self::Node { info, .. } => &info.out_dims,
        }
    }

    /// Get the tensor ID if this is a leaf.
    pub fn tensor_id(&self) -> Option<usize> {
        match self {
            Self::Leaf(info) => info.tensor_id,
            Self::Node { .. } => None,
        }
    }

    /// Get the info for this node.
    pub fn info(&self) -> &ExprInfo {
        match self {
            Self::Leaf(info) | Self::Node { info, .. } => info,
        }
    }

    /// Get mutable info for this node.
    pub fn info_mut(&mut self) -> &mut ExprInfo {
        match self {
            Self::Leaf(info) | Self::Node { info, .. } => info,
        }
    }

    /// Count the number of leaves in this tree.
    pub fn leaf_count(&self) -> usize {
        match self {
            Self::Leaf(_) => 1,
            Self::Node { left, right, .. } => left.leaf_count() + right.leaf_count(),
        }
    }

    /// Get all leaf tensor IDs in depth-first order.
    pub fn leaf_ids(&self) -> Vec<usize> {
        let mut ids = Vec::new();
        self.collect_leaf_ids(&mut ids);
        ids
    }

    fn collect_leaf_ids(&self, ids: &mut Vec<usize>) {
        match self {
            Self::Leaf(info) => {
                if let Some(id) = info.tensor_id {
                    ids.push(id);
                }
            }
            Self::Node { left, right, .. } => {
                left.collect_leaf_ids(ids);
                right.collect_leaf_ids(ids);
            }
        }
    }
}

/// Decomposition type for the tree structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecompositionType {
    /// Full binary tree (allows any tree structure)
    #[default]
    Tree,
    /// Path decomposition (linear chain, right child is always a leaf)
    Path,
}

/// The mutation rules that can be applied to an ExprTree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// `((a,b),c) → ((a,c),b)`: Swap b and c
    Rule1,
    /// `((a,b),c) → ((c,b),a)`: Swap a and c
    Rule2,
    /// `(a,(b,c)) → (b,(a,c))`: Swap a and b
    Rule3,
    /// `(a,(b,c)) → (c,(b,a))`: Swap a and c
    Rule4,
    /// `(a,b) → (b,a)`: Simple swap (PathDecomp only)
    Rule5,
}

/// Pre-allocated rule arrays to avoid heap allocations
const RULES_NONE: &[Rule] = &[];
const RULES_1_2: &[Rule] = &[Rule::Rule1, Rule::Rule2];
const RULES_3_4: &[Rule] = &[Rule::Rule3, Rule::Rule4];
const RULES_1_2_3_4: &[Rule] = &[Rule::Rule1, Rule::Rule2, Rule::Rule3, Rule::Rule4];
const RULES_5: &[Rule] = &[Rule::Rule5];
const RULES_1: &[Rule] = &[Rule::Rule1];

impl Rule {
    /// Get the applicable rules for a tree node given the decomposition type.
    /// Returns a static slice to avoid heap allocation.
    #[inline]
    pub fn applicable_rules(tree: &ExprTree, decomp: DecompositionType) -> &'static [Rule] {
        match tree {
            ExprTree::Leaf(_) => RULES_NONE,
            ExprTree::Node { left, right, .. } => {
                let left_is_leaf = left.is_leaf();
                let right_is_leaf = right.is_leaf();

                match decomp {
                    DecompositionType::Tree => {
                        if left_is_leaf && right_is_leaf {
                            RULES_NONE
                        } else if right_is_leaf {
                            RULES_1_2
                        } else if left_is_leaf {
                            RULES_3_4
                        } else {
                            RULES_1_2_3_4
                        }
                    }
                    DecompositionType::Path => {
                        if left_is_leaf {
                            RULES_5
                        } else {
                            RULES_1
                        }
                    }
                }
            }
        }
    }
}

/// Check if a slice contains an element (linear search, fast for small slices)
#[inline(always)]
fn slice_contains(slice: &[usize], elem: usize) -> bool {
    slice.contains(&elem)
}

/// Compute the time, space, and read-write complexity for a single contraction.
///
/// Returns (tc, sc, rw) where:
/// - tc: log2 of FLOP count
/// - sc: log2 of output tensor size
/// - rw: log2 of total read-write operations
///
/// Uses linear search instead of HashSet for small index arrays (typical tensor dimensions).
#[inline]
pub fn tcscrw(
    ix1: &[usize],
    ix2: &[usize],
    iy: &[usize],
    log2_sizes: &[f64],
    compute_rw: bool,
) -> (f64, f64, f64) {
    // Size of input 1
    let sc1: f64 = ix1.iter().map(|&l| log2_sizes[l]).sum();
    // Size of input 2
    let sc2: f64 = ix2.iter().map(|&l| log2_sizes[l]).sum();
    // Size of output
    let sc: f64 = iy.iter().map(|&l| log2_sizes[l]).sum();

    // Time complexity = output size + contracted indices
    // Use linear search (faster for typical small tensor dimensions < 20)
    let mut tc = sc;
    for &l in ix1 {
        if slice_contains(ix2, l) && !slice_contains(iy, l) {
            tc += log2_sizes[l];
        }
    }

    // Read-write complexity
    let rw = if compute_rw {
        fast_log2sumexp2_3(sc, sc1, sc2)
    } else {
        0.0
    };

    (tc, sc, rw)
}

/// Compute the output labels for a contraction of two tensors.
///
/// Uses linear search instead of HashSet for small index arrays.
#[inline]
pub fn contraction_output(ix1: &[usize], ix2: &[usize], final_output: &[usize]) -> Vec<usize> {
    // Pre-allocate with reasonable capacity
    let mut output = Vec::with_capacity(ix1.len() + ix2.len());

    // Include labels that:
    // 1. Appear in only one input (external edges)
    // 2. Appear in both inputs AND in the final output (batched indices)
    for &l in ix1 {
        if (!slice_contains(ix2, l) || slice_contains(final_output, l))
            && !slice_contains(&output, l)
        {
            output.push(l);
        }
    }
    for &l in ix2 {
        if (!slice_contains(ix1, l) || slice_contains(final_output, l))
            && !slice_contains(&output, l)
        {
            output.push(l);
        }
    }

    output
}

/// Compute the total tree complexity (time, space, read-write).
pub fn tree_complexity(tree: &ExprTree, log2_sizes: &[f64]) -> (f64, f64, f64) {
    // Check if we have cached complexity
    if let Some(cached) = tree.info().cached {
        return (cached.tc, cached.sc, cached.rw);
    }

    match tree {
        ExprTree::Leaf(info) => {
            let sc: f64 = info.out_dims.iter().map(|&l| log2_sizes[l]).sum();
            (f64::NEG_INFINITY, sc, f64::NEG_INFINITY)
        }
        ExprTree::Node { left, right, info } => {
            let (tcl, scl, rwl) = tree_complexity(left, log2_sizes);
            let (tcr, scr, rwr) = tree_complexity(right, log2_sizes);
            let (tc, sc, rw) = tcscrw(
                left.labels(),
                right.labels(),
                &info.out_dims,
                log2_sizes,
                true,
            );

            (
                fast_log2sumexp2_3(tc, tcl, tcr),
                sc.max(scl).max(scr),
                fast_log2sumexp2_3(rw, rwl, rwr),
            )
        }
    }
}

/// Compute and cache tree complexity, returning a mutable reference for updates.
pub fn tree_complexity_cached(tree: &mut ExprTree, log2_sizes: &[f64]) -> CachedComplexity {
    // If already cached, return it
    if let Some(cached) = tree.info().cached {
        return cached;
    }

    let (tc, sc, rw) = match tree {
        ExprTree::Leaf(info) => {
            let sc: f64 = info.out_dims.iter().map(|&l| log2_sizes[l]).sum();
            (f64::NEG_INFINITY, sc, f64::NEG_INFINITY)
        }
        ExprTree::Node { left, right, info } => {
            let left_cached = tree_complexity_cached(left, log2_sizes);
            let right_cached = tree_complexity_cached(right, log2_sizes);
            let (tc, sc, rw) = tcscrw(
                left.labels(),
                right.labels(),
                &info.out_dims,
                log2_sizes,
                true,
            );

            (
                fast_log2sumexp2_3(tc, left_cached.tc, right_cached.tc),
                sc.max(left_cached.sc).max(right_cached.sc),
                fast_log2sumexp2_3(rw, left_cached.rw, right_cached.rw),
            )
        }
    };

    let cached = CachedComplexity { tc, sc, rw };
    tree.info_mut().cached = Some(cached);
    cached
}

/// Quickly compute only the space complexity for a tree.
/// This is an optimized version that skips tc and rw computation.
#[inline]
pub fn tree_sc_only(tree: &ExprTree, log2_sizes: &[f64]) -> f64 {
    match tree {
        ExprTree::Leaf(info) => info.out_dims.iter().map(|&l| log2_sizes[l]).sum(),
        ExprTree::Node { left, right, info } => {
            let scl = tree_sc_only(left, log2_sizes);
            let scr = tree_sc_only(right, log2_sizes);
            let sc: f64 = info.out_dims.iter().map(|&l| log2_sizes[l]).sum();
            sc.max(scl).max(scr)
        }
    }
}

/// Result of computing complexity difference for a rule application.
#[derive(Debug, Clone)]
pub struct RuleDiff {
    /// Time complexity before
    pub tc0: f64,
    /// Time complexity after
    pub tc1: f64,
    /// Space complexity change (after - before)
    pub dsc: f64,
    /// Read-write complexity before
    pub rw0: f64,
    /// Read-write complexity after
    pub rw1: f64,
    /// New output labels for the modified subtree
    pub new_labels: Vec<usize>,
}

/// Compute the complexity difference for applying a rule to a tree.
///
/// This computes the local change in complexity without recomputing the entire tree.
pub fn rule_diff(
    tree: &ExprTree,
    rule: Rule,
    log2_sizes: &[f64],
    compute_rw: bool,
) -> Option<RuleDiff> {
    match tree {
        ExprTree::Leaf(_) => None,
        ExprTree::Node { left, right, info } => {
            let d = &info.out_dims;

            match rule {
                Rule::Rule1 | Rule::Rule2 => {
                    // ((a,b),c) structure - left is a Node
                    match left.as_ref() {
                        ExprTree::Node {
                            left: a,
                            right: b,
                            info: ab_info,
                        } => {
                            let c = right;
                            let ab = &ab_info.out_dims;

                            // Old: (a,b)->ab, (ab,c)->d
                            let (tc_ab, sc_ab, rw_ab) =
                                tcscrw(a.labels(), b.labels(), ab, log2_sizes, compute_rw);
                            let (tc_d, sc_d, rw_d) =
                                tcscrw(ab, c.labels(), d, log2_sizes, compute_rw);
                            let tc0 = fast_log2sumexp2(tc_ab, tc_d);
                            let sc0 = sc_ab.max(sc_d);
                            let rw0 = if compute_rw {
                                fast_log2sumexp2(rw_ab, rw_d)
                            } else {
                                0.0
                            };

                            // New structure depends on rule
                            let new_labels = match rule {
                                Rule::Rule1 => {
                                    // ((a,c),b) -> ac_labels
                                    contraction_output(a.labels(), c.labels(), d)
                                }
                                Rule::Rule2 => {
                                    // ((c,b),a) -> cb_labels
                                    contraction_output(c.labels(), b.labels(), d)
                                }
                                _ => unreachable!(),
                            };

                            // Compute new complexity
                            let (tc_new_left, sc_new_left, rw_new_left) = match rule {
                                Rule::Rule1 => tcscrw(
                                    a.labels(),
                                    c.labels(),
                                    &new_labels,
                                    log2_sizes,
                                    compute_rw,
                                ),
                                Rule::Rule2 => tcscrw(
                                    c.labels(),
                                    b.labels(),
                                    &new_labels,
                                    log2_sizes,
                                    compute_rw,
                                ),
                                _ => unreachable!(),
                            };

                            let (tc_new_d, sc_new_d, rw_new_d) = match rule {
                                Rule::Rule1 => {
                                    tcscrw(&new_labels, b.labels(), d, log2_sizes, compute_rw)
                                }
                                Rule::Rule2 => {
                                    tcscrw(&new_labels, a.labels(), d, log2_sizes, compute_rw)
                                }
                                _ => unreachable!(),
                            };

                            let tc1 = fast_log2sumexp2(tc_new_left, tc_new_d);
                            let sc1 = sc_new_left.max(sc_new_d);
                            let rw1 = if compute_rw {
                                fast_log2sumexp2(rw_new_left, rw_new_d)
                            } else {
                                0.0
                            };

                            Some(RuleDiff {
                                tc0,
                                tc1,
                                dsc: sc1 - sc0,
                                rw0,
                                rw1,
                                new_labels,
                            })
                        }
                        _ => None,
                    }
                }
                Rule::Rule3 | Rule::Rule4 => {
                    // (a,(b,c)) structure - right is a Node
                    match right.as_ref() {
                        ExprTree::Node {
                            left: b,
                            right: c,
                            info: bc_info,
                        } => {
                            let a = left;
                            let bc = &bc_info.out_dims;

                            // Old: (b,c)->bc, (a,bc)->d
                            let (tc_bc, sc_bc, rw_bc) =
                                tcscrw(b.labels(), c.labels(), bc, log2_sizes, compute_rw);
                            let (tc_d, sc_d, rw_d) =
                                tcscrw(a.labels(), bc, d, log2_sizes, compute_rw);
                            let tc0 = fast_log2sumexp2(tc_bc, tc_d);
                            let sc0 = sc_bc.max(sc_d);
                            let rw0 = if compute_rw {
                                fast_log2sumexp2(rw_bc, rw_d)
                            } else {
                                0.0
                            };

                            // New structure depends on rule
                            let new_labels = match rule {
                                Rule::Rule3 => {
                                    // (b,(a,c)) -> ac_labels
                                    contraction_output(a.labels(), c.labels(), d)
                                }
                                Rule::Rule4 => {
                                    // (c,(b,a)) -> ba_labels
                                    contraction_output(b.labels(), a.labels(), d)
                                }
                                _ => unreachable!(),
                            };

                            // Compute new complexity
                            let (tc_new_right, sc_new_right, rw_new_right) = match rule {
                                Rule::Rule3 => tcscrw(
                                    a.labels(),
                                    c.labels(),
                                    &new_labels,
                                    log2_sizes,
                                    compute_rw,
                                ),
                                Rule::Rule4 => tcscrw(
                                    b.labels(),
                                    a.labels(),
                                    &new_labels,
                                    log2_sizes,
                                    compute_rw,
                                ),
                                _ => unreachable!(),
                            };

                            let (tc_new_d, sc_new_d, rw_new_d) = match rule {
                                Rule::Rule3 => {
                                    tcscrw(b.labels(), &new_labels, d, log2_sizes, compute_rw)
                                }
                                Rule::Rule4 => {
                                    tcscrw(c.labels(), &new_labels, d, log2_sizes, compute_rw)
                                }
                                _ => unreachable!(),
                            };

                            let tc1 = fast_log2sumexp2(tc_new_right, tc_new_d);
                            let sc1 = sc_new_right.max(sc_new_d);
                            let rw1 = if compute_rw {
                                fast_log2sumexp2(rw_new_right, rw_new_d)
                            } else {
                                0.0
                            };

                            Some(RuleDiff {
                                tc0,
                                tc1,
                                dsc: sc1 - sc0,
                                rw0,
                                rw1,
                                new_labels,
                            })
                        }
                        _ => None,
                    }
                }
                Rule::Rule5 => {
                    // Simple swap (a,b) -> (b,a)
                    // This doesn't change complexity, just the order
                    Some(RuleDiff {
                        tc0: 0.0,
                        tc1: 0.0,
                        dsc: 0.0,
                        rw0: 0.0,
                        rw1: 0.0,
                        new_labels: info.out_dims.clone(),
                    })
                }
            }
        }
    }
}

/// Apply a mutation rule to a tree, returning the modified tree.
pub fn apply_rule(tree: ExprTree, rule: Rule, new_labels: Vec<usize>) -> ExprTree {
    match tree {
        ExprTree::Leaf(_) => tree,
        ExprTree::Node {
            left,
            right,
            mut info,
        } => {
            match rule {
                Rule::Rule1 => {
                    // ((a,b),c) → ((a,c),b)
                    match *left {
                        ExprTree::Node {
                            left: a, right: b, ..
                        } => {
                            let new_left = ExprTree::Node {
                                left: a,
                                right,
                                info: ExprInfo::internal(new_labels),
                            };
                            ExprTree::Node {
                                left: Box::new(new_left),
                                right: b,
                                info,
                            }
                        }
                        _ => ExprTree::Node { left, right, info },
                    }
                }
                Rule::Rule2 => {
                    // ((a,b),c) → ((c,b),a)
                    match *left {
                        ExprTree::Node {
                            left: a, right: b, ..
                        } => {
                            let new_left = ExprTree::Node {
                                left: right,
                                right: b,
                                info: ExprInfo::internal(new_labels),
                            };
                            ExprTree::Node {
                                left: Box::new(new_left),
                                right: a,
                                info,
                            }
                        }
                        _ => ExprTree::Node { left, right, info },
                    }
                }
                Rule::Rule3 => {
                    // (a,(b,c)) → (b,(a,c))
                    match *right {
                        ExprTree::Node {
                            left: b, right: c, ..
                        } => {
                            let new_right = ExprTree::Node {
                                left,
                                right: c,
                                info: ExprInfo::internal(new_labels),
                            };
                            ExprTree::Node {
                                left: b,
                                right: Box::new(new_right),
                                info,
                            }
                        }
                        _ => ExprTree::Node { left, right, info },
                    }
                }
                Rule::Rule4 => {
                    // (a,(b,c)) → (c,(b,a))
                    match *right {
                        ExprTree::Node {
                            left: b, right: c, ..
                        } => {
                            let new_right = ExprTree::Node {
                                left: b,
                                right: left,
                                info: ExprInfo::internal(new_labels),
                            };
                            ExprTree::Node {
                                left: c,
                                right: Box::new(new_right),
                                info,
                            }
                        }
                        _ => ExprTree::Node { left, right, info },
                    }
                }
                Rule::Rule5 => {
                    // (a,b) → (b,a)
                    info.out_dims = new_labels;
                    ExprTree::Node {
                        left: right,
                        right: left,
                        info,
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_tree() -> ExprTree {
        // ((0,1),2) where 0:[i,j], 1:[j,k], 2:[k,l]
        let leaf0 = ExprTree::leaf(vec![0, 1], 0); // i,j
        let leaf1 = ExprTree::leaf(vec![1, 2], 1); // j,k
        let leaf2 = ExprTree::leaf(vec![2, 3], 2); // k,l

        let inner = ExprTree::node(leaf0, leaf1, vec![0, 2]); // i,k
        ExprTree::node(inner, leaf2, vec![0, 3]) // i,l
    }

    #[test]
    fn test_expr_tree_leaf() {
        let leaf = ExprTree::leaf(vec![0, 1], 0);
        assert!(leaf.is_leaf());
        assert_eq!(leaf.tensor_id(), Some(0));
        assert_eq!(leaf.labels(), &[0, 1]);
    }

    #[test]
    fn test_expr_tree_node() {
        let tree = simple_tree();
        assert!(!tree.is_leaf());
        assert_eq!(tree.leaf_count(), 3);
        assert_eq!(tree.leaf_ids(), vec![0, 1, 2]);
    }

    #[test]
    fn test_applicable_rules_tree_decomp() {
        let tree = simple_tree();
        let rules = Rule::applicable_rules(&tree, DecompositionType::Tree);
        // Left child is a Node, right is Leaf -> Rules 1 and 2
        assert!(rules.contains(&Rule::Rule1));
        assert!(rules.contains(&Rule::Rule2));
        assert!(!rules.contains(&Rule::Rule3));
    }

    #[test]
    fn test_tcscrw() {
        let log2_sizes = vec![2.0, 3.0, 3.0, 2.0]; // i=4, j=8, k=8, l=4

        // Contract [i,j] with [j,k] -> [i,k]
        let (tc, sc, _rw) = tcscrw(&[0, 1], &[1, 2], &[0, 2], &log2_sizes, true);

        // tc = output + contracted = (i+k) + j = 2+3+3 = 8
        assert!((tc - 8.0).abs() < 1e-10);
        // sc = i + k = 2 + 3 = 5
        assert!((sc - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_contraction_output() {
        let output = contraction_output(&[0, 1], &[1, 2], &[0, 3]);
        // i and k should be in output (j is contracted)
        assert!(output.contains(&0)); // i
        assert!(output.contains(&2)); // k
        assert!(!output.contains(&1)); // j is contracted
    }

    #[test]
    fn test_tree_complexity() {
        let leaf0 = ExprTree::leaf(vec![0, 1], 0);
        let leaf1 = ExprTree::leaf(vec![1, 2], 1);
        let tree = ExprTree::node(leaf0, leaf1, vec![0, 2]);

        let log2_sizes = vec![2.0, 3.0, 2.0]; // i=4, j=8, k=4

        let (tc, sc, _rw) = tree_complexity(&tree, &log2_sizes);

        // tc = log2(2^(i+k+j)) = i+k+j = 2+2+3 = 7
        assert!((tc - 7.0).abs() < 1e-10);
        // sc = max(output, input1, input2) = max(i+k, i+j, j+k) = max(4, 5, 5) = 5
        assert!((sc - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_rule_diff_rule1() {
        // ((0,1),2) structure for Rule1/Rule2
        let tree = simple_tree();
        let log2_sizes = vec![2.0, 3.0, 3.0, 2.0];

        let diff = rule_diff(&tree, Rule::Rule1, &log2_sizes, true);
        assert!(diff.is_some());
        let diff = diff.unwrap();
        // The diff should have valid complexity values
        assert!(diff.tc0 > 0.0);
        assert!(diff.tc1 > 0.0);
    }

    #[test]
    fn test_rule_diff_rule2() {
        let tree = simple_tree();
        let log2_sizes = vec![2.0, 3.0, 3.0, 2.0];

        let diff = rule_diff(&tree, Rule::Rule2, &log2_sizes, false);
        assert!(diff.is_some());
        let diff = diff.unwrap();
        assert!(diff.tc0 > 0.0);
        assert!(diff.tc1 > 0.0);
        // rw should be 0 when compute_rw is false
        assert_eq!(diff.rw0, 0.0);
        assert_eq!(diff.rw1, 0.0);
    }

    #[test]
    fn test_rule_diff_rule3_rule4() {
        // (0,(1,2)) structure for Rule3/Rule4
        let leaf0 = ExprTree::leaf(vec![0, 1], 0); // i,j
        let leaf1 = ExprTree::leaf(vec![1, 2], 1); // j,k
        let leaf2 = ExprTree::leaf(vec![2, 3], 2); // k,l

        let inner = ExprTree::node(leaf1, leaf2, vec![1, 3]); // j,l
        let tree = ExprTree::node(leaf0, inner, vec![0, 3]); // i,l

        let log2_sizes = vec![2.0, 3.0, 3.0, 2.0];

        let diff3 = rule_diff(&tree, Rule::Rule3, &log2_sizes, true);
        assert!(diff3.is_some());

        let diff4 = rule_diff(&tree, Rule::Rule4, &log2_sizes, true);
        assert!(diff4.is_some());
    }

    #[test]
    fn test_rule_diff_rule5() {
        // ((0,1),(2,3)) structure for Rule5
        let leaf0 = ExprTree::leaf(vec![0, 1], 0);
        let leaf1 = ExprTree::leaf(vec![1, 2], 1);
        let leaf2 = ExprTree::leaf(vec![2, 3], 2);
        let leaf3 = ExprTree::leaf(vec![3, 4], 3);

        let left = ExprTree::node(leaf0, leaf1, vec![0, 2]);
        let right = ExprTree::node(leaf2, leaf3, vec![2, 4]);
        let tree = ExprTree::node(left, right, vec![0, 4]);

        let log2_sizes = vec![2.0, 3.0, 3.0, 3.0, 2.0];

        let diff = rule_diff(&tree, Rule::Rule5, &log2_sizes, true);
        assert!(diff.is_some());
    }

    #[test]
    fn test_apply_rule1() {
        let tree = simple_tree();
        let log2_sizes = vec![2.0, 3.0, 3.0, 2.0];

        let diff = rule_diff(&tree, Rule::Rule1, &log2_sizes, false).unwrap();
        let new_tree = apply_rule(tree, Rule::Rule1, diff.new_labels);

        assert_eq!(new_tree.leaf_count(), 3);
        assert!(!new_tree.is_leaf());
    }

    #[test]
    fn test_apply_rule3() {
        // (0,(1,2)) structure
        let leaf0 = ExprTree::leaf(vec![0, 1], 0);
        let leaf1 = ExprTree::leaf(vec![1, 2], 1);
        let leaf2 = ExprTree::leaf(vec![2, 3], 2);

        let inner = ExprTree::node(leaf1, leaf2, vec![1, 3]);
        let tree = ExprTree::node(leaf0, inner, vec![0, 3]);

        let log2_sizes = vec![2.0, 3.0, 3.0, 2.0];
        let diff = rule_diff(&tree, Rule::Rule3, &log2_sizes, false).unwrap();
        let new_tree = apply_rule(tree, Rule::Rule3, diff.new_labels);

        assert_eq!(new_tree.leaf_count(), 3);
    }

    #[test]
    fn test_apply_rule5() {
        // ((0,1),(2,3)) structure
        let leaf0 = ExprTree::leaf(vec![0, 1], 0);
        let leaf1 = ExprTree::leaf(vec![1, 2], 1);
        let leaf2 = ExprTree::leaf(vec![2, 3], 2);
        let leaf3 = ExprTree::leaf(vec![3, 4], 3);

        let left = ExprTree::node(leaf0, leaf1, vec![0, 2]);
        let right = ExprTree::node(leaf2, leaf3, vec![2, 4]);
        let tree = ExprTree::node(left, right, vec![0, 4]);

        let log2_sizes = vec![2.0, 3.0, 3.0, 3.0, 2.0];
        let diff = rule_diff(&tree, Rule::Rule5, &log2_sizes, false).unwrap();
        let new_tree = apply_rule(tree, Rule::Rule5, diff.new_labels);

        assert_eq!(new_tree.leaf_count(), 4);
    }

    #[test]
    fn test_applicable_rules_path_decomp() {
        // For path decomposition: ((0,1),2) - left is node, right is leaf -> Rule1
        let tree = simple_tree();
        let rules = Rule::applicable_rules(&tree, DecompositionType::Path);
        assert!(rules.contains(&Rule::Rule1));
        assert!(!rules.contains(&Rule::Rule5));
    }

    #[test]
    fn test_applicable_rules_path_decomp_left_leaf() {
        // (0,(1,2)) - left is leaf, right is node -> Rule5
        let leaf0 = ExprTree::leaf(vec![0, 1], 0);
        let leaf1 = ExprTree::leaf(vec![1, 2], 1);
        let leaf2 = ExprTree::leaf(vec![2, 3], 2);
        let inner = ExprTree::node(leaf1, leaf2, vec![1, 3]);
        let tree = ExprTree::node(leaf0, inner, vec![0, 3]);

        let rules = Rule::applicable_rules(&tree, DecompositionType::Path);
        assert!(rules.contains(&Rule::Rule5));
        assert!(!rules.contains(&Rule::Rule1));
    }

    #[test]
    fn test_rule_diff_on_leaf() {
        let leaf = ExprTree::leaf(vec![0, 1], 0);
        let log2_sizes = vec![2.0, 3.0];

        let diff = rule_diff(&leaf, Rule::Rule1, &log2_sizes, false);
        assert!(diff.is_none());
    }

    #[test]
    fn test_tree_complexity_single_leaf() {
        let leaf = ExprTree::leaf(vec![0, 1], 0);
        let log2_sizes = vec![2.0, 3.0];

        let (tc, sc, rw) = tree_complexity(&leaf, &log2_sizes);
        // Single leaf has no contractions
        assert!(tc < 1e-10 || tc == f64::NEG_INFINITY);
        assert!((sc - 5.0).abs() < 1e-10); // i+j = 2+3 = 5
        assert!(rw < 1e-10 || rw == f64::NEG_INFINITY);
    }

    #[test]
    fn test_applicable_rules_both_children_nodes() {
        // ((0,1),(2,3)) - both children are nodes -> Rules 1,2,3,4
        let leaf0 = ExprTree::leaf(vec![0, 1], 0);
        let leaf1 = ExprTree::leaf(vec![1, 2], 1);
        let leaf2 = ExprTree::leaf(vec![2, 3], 2);
        let leaf3 = ExprTree::leaf(vec![3, 4], 3);

        let left = ExprTree::node(leaf0, leaf1, vec![0, 2]);
        let right = ExprTree::node(leaf2, leaf3, vec![2, 4]);
        let tree = ExprTree::node(left, right, vec![0, 4]);

        let rules = Rule::applicable_rules(&tree, DecompositionType::Tree);
        assert!(rules.contains(&Rule::Rule1));
        assert!(rules.contains(&Rule::Rule2));
        assert!(rules.contains(&Rule::Rule3));
        assert!(rules.contains(&Rule::Rule4));
        // Rule5 not returned for Tree decomposition
        assert!(!rules.contains(&Rule::Rule5));
    }

    #[test]
    fn test_contraction_output_all_contracted() {
        // When all indices are contracted
        let output = contraction_output(&[0, 1], &[0, 1], &[]);
        assert!(output.is_empty());
    }

    #[test]
    fn test_tcscrw_no_rw() {
        let log2_sizes = vec![2.0, 3.0, 2.0];
        let (tc, sc, rw) = tcscrw(&[0, 1], &[1, 2], &[0, 2], &log2_sizes, false);

        assert!(tc > 0.0);
        assert!(sc > 0.0);
        assert_eq!(rw, 0.0);
    }

    #[test]
    fn test_expr_info_internal() {
        let info = ExprInfo::internal(vec![0, 1, 2]);
        assert_eq!(info.out_dims, vec![0, 1, 2]);
        assert!(info.tensor_id.is_none());
        assert!(info.cached.is_none());
    }

    #[test]
    fn test_expr_info_leaf() {
        let info = ExprInfo::leaf(vec![0, 1], 42);
        assert_eq!(info.out_dims, vec![0, 1]);
        assert_eq!(info.tensor_id, Some(42));
        assert!(info.cached.is_none());
    }

    #[test]
    fn test_expr_tree_info() {
        let leaf = ExprTree::leaf(vec![0, 1], 0);
        let info = leaf.info();
        assert_eq!(info.out_dims, vec![0, 1]);
        assert_eq!(info.tensor_id, Some(0));
    }

    #[test]
    fn test_expr_tree_info_mut() {
        let mut leaf = ExprTree::leaf(vec![0, 1], 0);
        {
            let info = leaf.info_mut();
            info.out_dims = vec![2, 3];
        }
        assert_eq!(leaf.labels(), &[2, 3]);
    }

    #[test]
    fn test_tree_complexity_cached() {
        let leaf0 = ExprTree::leaf(vec![0, 1], 0);
        let leaf1 = ExprTree::leaf(vec![1, 2], 1);
        let mut tree = ExprTree::node(leaf0, leaf1, vec![0, 2]);
        let log2_sizes = vec![2.0, 3.0, 2.0];

        // First call computes and caches
        let cached1 = tree_complexity_cached(&mut tree, &log2_sizes);
        assert!(cached1.tc > 0.0);
        assert!(cached1.sc > 0.0);

        // Second call returns cached value
        let cached2 = tree_complexity_cached(&mut tree, &log2_sizes);
        assert!((cached1.tc - cached2.tc).abs() < 1e-10);
        assert!((cached1.sc - cached2.sc).abs() < 1e-10);
    }

    #[test]
    fn test_tree_complexity_cached_node() {
        let leaf0 = ExprTree::leaf(vec![0, 1], 0);
        let leaf1 = ExprTree::leaf(vec![1, 2], 1);
        let mut tree = ExprTree::node(leaf0, leaf1, vec![0, 2]);

        let log2_sizes = vec![2.0, 3.0, 2.0];

        let cached = tree_complexity_cached(&mut tree, &log2_sizes);
        assert!(cached.tc > 0.0);
        assert!(cached.sc > 0.0);
    }

    #[test]
    fn test_tree_sc_only() {
        let leaf0 = ExprTree::leaf(vec![0, 1], 0);
        let leaf1 = ExprTree::leaf(vec![1, 2], 1);
        let tree = ExprTree::node(leaf0, leaf1, vec![0, 2]);

        let log2_sizes = vec![2.0, 3.0, 2.0];
        let sc = tree_sc_only(&tree, &log2_sizes);

        // Same as full tree_complexity sc
        let (_, full_sc, _) = tree_complexity(&tree, &log2_sizes);
        assert!((sc - full_sc).abs() < 1e-10);
    }

    #[test]
    fn test_tree_sc_only_leaf() {
        let leaf = ExprTree::leaf(vec![0, 1], 0);
        let log2_sizes = vec![2.0, 3.0];

        let sc = tree_sc_only(&leaf, &log2_sizes);
        assert!((sc - 5.0).abs() < 1e-10); // 2 + 3
    }

    #[test]
    fn test_apply_rule_on_leaf() {
        // Applying rule on leaf should return the leaf unchanged
        let leaf = ExprTree::leaf(vec![0, 1], 0);
        let result = apply_rule(leaf.clone(), Rule::Rule1, vec![]);

        assert!(result.is_leaf());
    }

    #[test]
    fn test_apply_rule2() {
        // Test Rule2: ((a,b),c) → ((c,b),a)
        let tree = simple_tree();
        let log2_sizes = vec![2.0, 3.0, 3.0, 2.0];

        let diff = rule_diff(&tree, Rule::Rule2, &log2_sizes, false).unwrap();
        let new_tree = apply_rule(tree, Rule::Rule2, diff.new_labels);

        assert_eq!(new_tree.leaf_count(), 3);
    }

    #[test]
    fn test_apply_rule4() {
        // Test Rule4: (a,(b,c)) → (c,(b,a))
        let leaf0 = ExprTree::leaf(vec![0, 1], 0);
        let leaf1 = ExprTree::leaf(vec![1, 2], 1);
        let leaf2 = ExprTree::leaf(vec![2, 3], 2);

        let inner = ExprTree::node(leaf1, leaf2, vec![1, 3]);
        let tree = ExprTree::node(leaf0, inner, vec![0, 3]);

        let log2_sizes = vec![2.0, 3.0, 3.0, 2.0];
        let diff = rule_diff(&tree, Rule::Rule4, &log2_sizes, false).unwrap();
        let new_tree = apply_rule(tree, Rule::Rule4, diff.new_labels);

        assert_eq!(new_tree.leaf_count(), 3);
    }

    #[test]
    fn test_apply_rule_wrong_structure() {
        // Apply Rule1/Rule2 when left child is a leaf (should return unchanged)
        let leaf0 = ExprTree::leaf(vec![0, 1], 0);
        let leaf1 = ExprTree::leaf(vec![1, 2], 1);
        let tree = ExprTree::node(leaf0.clone(), leaf1.clone(), vec![0, 2]);

        // Tree is (leaf, leaf) - Rule1 expects ((a,b),c)
        let result = apply_rule(tree, Rule::Rule1, vec![]);
        // Should return unchanged since left is not a Node
        assert_eq!(result.leaf_count(), 2);
    }

    #[test]
    fn test_apply_rule3_wrong_structure() {
        // Apply Rule3/Rule4 when right child is a leaf
        let leaf0 = ExprTree::leaf(vec![0, 1], 0);
        let leaf1 = ExprTree::leaf(vec![1, 2], 1);
        let tree = ExprTree::node(leaf0, leaf1, vec![0, 2]);

        // Tree is (leaf, leaf) - Rule3 expects (a,(b,c))
        let result = apply_rule(tree, Rule::Rule3, vec![]);
        assert_eq!(result.leaf_count(), 2);
    }

    #[test]
    fn test_rule_diff_wrong_structure_rule1() {
        // Rule1/Rule2 on a tree where left is a leaf
        let leaf0 = ExprTree::leaf(vec![0, 1], 0);
        let leaf1 = ExprTree::leaf(vec![1, 2], 1);
        let tree = ExprTree::node(leaf0, leaf1, vec![0, 2]);

        let log2_sizes = vec![2.0, 3.0, 2.0];
        let diff = rule_diff(&tree, Rule::Rule1, &log2_sizes, false);
        assert!(diff.is_none()); // Cannot apply Rule1 to (leaf, leaf)
    }

    #[test]
    fn test_rule_diff_wrong_structure_rule3() {
        // Rule3/Rule4 on a tree where right is a leaf
        let tree = simple_tree(); // ((leaf,leaf), leaf)

        let log2_sizes = vec![2.0, 3.0, 3.0, 2.0];
        let diff = rule_diff(&tree, Rule::Rule3, &log2_sizes, false);
        assert!(diff.is_none()); // Cannot apply Rule3 because right is leaf
    }

    #[test]
    fn test_applicable_rules_both_leaves() {
        let leaf0 = ExprTree::leaf(vec![0, 1], 0);
        let leaf1 = ExprTree::leaf(vec![1, 2], 1);
        let tree = ExprTree::node(leaf0, leaf1, vec![0, 2]);

        // Both children are leaves - no rules applicable for Tree decomp
        let rules = Rule::applicable_rules(&tree, DecompositionType::Tree);
        assert!(rules.is_empty());

        // For Path decomp with left as leaf -> Rule5
        let rules_path = Rule::applicable_rules(&tree, DecompositionType::Path);
        assert!(rules_path.contains(&Rule::Rule5));
    }

    #[test]
    fn test_applicable_rules_right_is_node() {
        // (leaf, (leaf, leaf)) structure
        let leaf0 = ExprTree::leaf(vec![0, 1], 0);
        let leaf1 = ExprTree::leaf(vec![1, 2], 1);
        let leaf2 = ExprTree::leaf(vec![2, 3], 2);
        let inner = ExprTree::node(leaf1, leaf2, vec![1, 3]);
        let tree = ExprTree::node(leaf0, inner, vec![0, 3]);

        let rules = Rule::applicable_rules(&tree, DecompositionType::Tree);
        // Left is leaf, right is node -> Rules 3,4
        assert!(rules.contains(&Rule::Rule3));
        assert!(rules.contains(&Rule::Rule4));
        assert!(!rules.contains(&Rule::Rule1));
        assert!(!rules.contains(&Rule::Rule2));
    }

    #[test]
    fn test_tree_complexity_with_cached() {
        let leaf0 = ExprTree::leaf(vec![0, 1], 0);
        let leaf1 = ExprTree::leaf(vec![1, 2], 1);
        let mut tree = ExprTree::node(leaf0, leaf1, vec![0, 2]);

        let log2_sizes = vec![2.0, 3.0, 2.0];

        // First cache the complexity
        tree_complexity_cached(&mut tree, &log2_sizes);

        // Now tree_complexity should return cached values
        let (tc, sc, rw) = tree_complexity(&tree, &log2_sizes);
        assert!(tc > 0.0);
        assert!(sc > 0.0);
        assert!(rw > 0.0 || rw == f64::NEG_INFINITY);
    }

    #[test]
    fn test_leaf_ids_deep_tree() {
        // Create a deeper tree and verify leaf order
        let leaf0 = ExprTree::leaf(vec![0], 0);
        let leaf1 = ExprTree::leaf(vec![1], 1);
        let leaf2 = ExprTree::leaf(vec![2], 2);
        let leaf3 = ExprTree::leaf(vec![3], 3);

        let inner1 = ExprTree::node(leaf0, leaf1, vec![0, 1]);
        let inner2 = ExprTree::node(leaf2, leaf3, vec![2, 3]);
        let tree = ExprTree::node(inner1, inner2, vec![0, 1, 2, 3]);

        let ids = tree.leaf_ids();
        assert_eq!(ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_contraction_output_duplicates() {
        // Test that output doesn't contain duplicates
        let output = contraction_output(&[0, 1, 2], &[1, 2, 3], &[0, 3]);

        // Count occurrences
        let mut counts = std::collections::HashMap::new();
        for &x in &output {
            *counts.entry(x).or_insert(0) += 1;
        }

        // No duplicates
        for (_, count) in counts {
            assert_eq!(count, 1);
        }
    }
}
