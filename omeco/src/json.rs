//! JSON serialization for tensor network structures.
//!
//! Provides functions to read and write NestedEinsum and SlicedEinsum
//! structures to JSON format, compatible with Julia's OMEinsumContractionOrders.jl.
//!
//! # Example
//!
//! ```rust
//! use omeco::{EinCode, GreedyMethod, optimize_code};
//! use omeco::json::{writejson, readjson};
//! use std::collections::HashMap;
//!
//! let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
//! let sizes: HashMap<char, usize> = [('i', 2), ('j', 3), ('k', 2)].into();
//! let tree = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
//!
//! // Write to file (using temp directory for cross-platform compatibility)
//! let path = std::env::temp_dir().join("omeco_test.json");
//! writejson(&path, &tree).unwrap();
//!
//! // Read back
//! let loaded: omeco::json::ContractionOrder<char> = readjson(&path).unwrap();
//! ```

use crate::eincode::{EinCode, NestedEinsum, SlicedEinsum};
use crate::Label;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Error type for JSON operations.
#[derive(Debug, thiserror::Error)]
pub enum JsonError {
    /// IO error during file operations.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON parsing error.
    #[error("JSON parse error: {0}")]
    Parse(#[from] serde_json::Error),
}

/// Label type indicator for Julia compatibility.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LabelType {
    /// Character labels (e.g., 'i', 'j', 'k')
    Char,
    /// 64-bit integer labels
    Int64,
    /// Generic integer labels
    Int,
}

/// A contraction order that can be either NestedEinsum or SlicedEinsum.
///
/// This enum is returned by `readjson` to handle both types uniformly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractionOrder<L: Label> {
    /// A nested einsum contraction order.
    Nested(NestedEinsum<L>),
    /// A sliced einsum contraction order.
    Sliced(SlicedEinsum<L>),
}

impl<L: Label> ContractionOrder<L> {
    /// Check if this is a nested einsum.
    pub fn is_nested(&self) -> bool {
        matches!(self, ContractionOrder::Nested(_))
    }

    /// Check if this is a sliced einsum.
    pub fn is_sliced(&self) -> bool {
        matches!(self, ContractionOrder::Sliced(_))
    }

    /// Get the nested einsum if this is one.
    pub fn as_nested(&self) -> Option<&NestedEinsum<L>> {
        match self {
            ContractionOrder::Nested(n) => Some(n),
            ContractionOrder::Sliced(_) => None,
        }
    }

    /// Get the sliced einsum if this is one.
    pub fn as_sliced(&self) -> Option<&SlicedEinsum<L>> {
        match self {
            ContractionOrder::Nested(_) => None,
            ContractionOrder::Sliced(s) => Some(s),
        }
    }

    /// Convert to nested einsum, returning None if this is sliced.
    pub fn into_nested(self) -> Option<NestedEinsum<L>> {
        match self {
            ContractionOrder::Nested(n) => Some(n),
            ContractionOrder::Sliced(_) => None,
        }
    }

    /// Convert to sliced einsum, returning None if this is nested.
    pub fn into_sliced(self) -> Option<SlicedEinsum<L>> {
        match self {
            ContractionOrder::Nested(_) => None,
            ContractionOrder::Sliced(s) => Some(s),
        }
    }
}

impl<L: Label> From<NestedEinsum<L>> for ContractionOrder<L> {
    fn from(n: NestedEinsum<L>) -> Self {
        ContractionOrder::Nested(n)
    }
}

impl<L: Label> From<SlicedEinsum<L>> for ContractionOrder<L> {
    fn from(s: SlicedEinsum<L>) -> Self {
        ContractionOrder::Sliced(s)
    }
}

/// JSON format for contraction orders (Julia-compatible).
///
/// Format:
/// ```json
/// {
///   "label-type": "Char",
///   "inputs": [["i", "j"], ["j", "k"]],
///   "output": ["i", "k"],
///   "tree": { ... },
///   "slices": ["j"]  // optional, only for SlicedEinsum
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct ContractionOrderJson<L: Label> {
    /// Label type indicator
    label_type: LabelType,
    /// Original input tensor indices
    inputs: Vec<Vec<L>>,
    /// Requested output indices
    output: Vec<L>,
    /// The contraction tree
    tree: NestedEinsumTree<L>,
    /// Sliced indices (optional, only for SlicedEinsum)
    #[serde(skip_serializing_if = "Option::is_none")]
    slices: Option<Vec<L>>,
}

/// Tree node structure matching Julia's JSON format.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NestedEinsumTree<L: Label> {
    /// A leaf node representing an input tensor.
    Leaf {
        /// Always true for leaf nodes
        isleaf: bool,
        /// Index into the inputs array (0-based)
        #[serde(rename = "tensorindex")]
        tensor_index: usize,
    },
    /// An internal node representing a contraction.
    Node {
        /// Always false for internal nodes
        isleaf: bool,
        /// Child nodes
        args: Vec<NestedEinsumTree<L>>,
        /// The einsum operation at this node
        eins: EinCodeJson<L>,
    },
}

/// JSON-compatible EinCode representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EinCodeJson<L: Label> {
    /// Input indices for each child
    pub ixs: Vec<Vec<L>>,
    /// Output indices
    pub iy: Vec<L>,
}

// ============================================================================
// Conversion implementations
// ============================================================================

impl<L: Label> From<&NestedEinsum<L>> for NestedEinsumTree<L> {
    fn from(nested: &NestedEinsum<L>) -> Self {
        match nested {
            NestedEinsum::Leaf { tensor_index } => NestedEinsumTree::Leaf {
                isleaf: true,
                tensor_index: *tensor_index,
            },
            NestedEinsum::Node { args, eins } => NestedEinsumTree::Node {
                isleaf: false,
                args: args.iter().map(|a| a.into()).collect(),
                eins: EinCodeJson {
                    ixs: eins.ixs.clone(),
                    iy: eins.iy.clone(),
                },
            },
        }
    }
}

impl<L: Label> From<NestedEinsumTree<L>> for NestedEinsum<L> {
    fn from(tree: NestedEinsumTree<L>) -> Self {
        match tree {
            NestedEinsumTree::Leaf { tensor_index, .. } => NestedEinsum::leaf(tensor_index),
            NestedEinsumTree::Node { args, eins, .. } => {
                let nested_args: Vec<NestedEinsum<L>> =
                    args.into_iter().map(|a| a.into()).collect();
                NestedEinsum::node(nested_args, EinCode::new(eins.ixs, eins.iy))
            }
        }
    }
}

// ============================================================================
// Trait for types that can be written to JSON
// ============================================================================

/// Trait for contraction orders that can be serialized to JSON.
///
/// This trait is implemented for `NestedEinsum` and `SlicedEinsum`,
/// allowing unified use of `writejson` and `to_json_string`.
pub trait ToJson<L: Label + Serialize> {
    /// Convert to a JSON value.
    fn to_json_value(&self) -> Result<serde_json::Value, JsonError>;
}

impl<L: Label + Serialize> ToJson<L> for NestedEinsum<L> {
    fn to_json_value(&self) -> Result<serde_json::Value, JsonError> {
        let json = self.to_contraction_order_json();
        Ok(serde_json::to_value(&json)?)
    }
}

impl<L: Label + Serialize> ToJson<L> for SlicedEinsum<L> {
    fn to_json_value(&self) -> Result<serde_json::Value, JsonError> {
        let json = self.to_contraction_order_json();
        Ok(serde_json::to_value(&json)?)
    }
}

// Internal helper to create ContractionOrderJson from NestedEinsum
impl<L: Label> NestedEinsum<L> {
    fn to_contraction_order_json(&self) -> ContractionOrderJson<L> {
        let original_ixs = self.to_eincode_ixs();
        let original_iy = self.root_output();

        ContractionOrderJson {
            label_type: detect_label_type::<L>(),
            inputs: original_ixs,
            output: original_iy,
            tree: self.into(),
            slices: None,
        }
    }
}

// Internal helper to create ContractionOrderJson from SlicedEinsum
impl<L: Label> SlicedEinsum<L> {
    fn to_contraction_order_json(&self) -> ContractionOrderJson<L> {
        let original_ixs = self.eins.to_eincode_ixs();
        let original_iy = self.eins.root_output();

        ContractionOrderJson {
            label_type: detect_label_type::<L>(),
            inputs: original_ixs,
            output: original_iy,
            tree: (&self.eins).into(),
            slices: Some(self.slicing.clone()),
        }
    }
}

// Helper methods for NestedEinsum
impl<L: Label> NestedEinsum<L> {
    /// Extract original input indices from the tree.
    fn to_eincode_ixs(&self) -> Vec<Vec<L>> {
        let mut ixs: Vec<(usize, Vec<L>)> = Vec::new();
        self.collect_leaf_ixs(&mut ixs);
        ixs.sort_by_key(|(idx, _)| *idx);
        ixs.into_iter().map(|(_, ix)| ix).collect()
    }

    fn collect_leaf_ixs(&self, ixs: &mut Vec<(usize, Vec<L>)>) {
        match self {
            NestedEinsum::Leaf { tensor_index } => {
                // For leaves, we need to get the indices from the parent's eins
                // This is handled at the Node level
                if ixs.iter().all(|(idx, _)| *idx != *tensor_index) {
                    ixs.push((*tensor_index, vec![]));
                }
            }
            NestedEinsum::Node { args, eins } => {
                for (i, arg) in args.iter().enumerate() {
                    if arg.is_leaf() {
                        if let NestedEinsum::Leaf { tensor_index } = arg {
                            ixs.push((*tensor_index, eins.ixs[i].clone()));
                        }
                    } else {
                        arg.collect_leaf_ixs(ixs);
                    }
                }
            }
        }
    }

    /// Get the root output indices.
    fn root_output(&self) -> Vec<L> {
        match self {
            NestedEinsum::Leaf { .. } => vec![],
            NestedEinsum::Node { eins, .. } => eins.iy.clone(),
        }
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Write a contraction order to a JSON file.
///
/// Works with both `NestedEinsum` and `SlicedEinsum`.
///
/// # Arguments
/// * `path` - Path to the output file
/// * `order` - The contraction order to write
///
/// # Example
/// ```rust,ignore
/// use omeco::json::writejson;
///
/// writejson("output.json", &tree)?;
/// writejson("sliced.json", &sliced_tree)?;
/// ```
pub fn writejson<L, T, P>(path: P, order: &T) -> Result<(), JsonError>
where
    L: Label + Serialize,
    T: ToJson<L>,
    P: AsRef<Path>,
{
    let json_str = to_json_string(order)?;
    std::fs::write(path, json_str)?;
    Ok(())
}

/// Write a contraction order to a JSON string.
///
/// Works with both `NestedEinsum` and `SlicedEinsum`.
pub fn to_json_string<L, T>(order: &T) -> Result<String, JsonError>
where
    L: Label + Serialize,
    T: ToJson<L>,
{
    let value = order.to_json_value()?;
    Ok(serde_json::to_string_pretty(&value)?)
}

/// Read a contraction order from a JSON file.
///
/// Returns a `ContractionOrder` that can be either `NestedEinsum` or `SlicedEinsum`.
///
/// # Arguments
/// * `path` - Path to the input file
///
/// # Returns
/// A `ContractionOrder` enum containing either the nested or sliced result.
pub fn readjson<L, P>(path: P) -> Result<ContractionOrder<L>, JsonError>
where
    L: Label + for<'de> Deserialize<'de>,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let json: ContractionOrderJson<L> = serde_json::from_reader(reader)?;
    Ok(json_to_contraction_order(json))
}

/// Read a contraction order from a JSON string.
///
/// Returns a `ContractionOrder` that can be either `NestedEinsum` or `SlicedEinsum`.
pub fn from_json_string<L>(s: &str) -> Result<ContractionOrder<L>, JsonError>
where
    L: Label + for<'de> Deserialize<'de>,
{
    let json: ContractionOrderJson<L> = serde_json::from_str(s)?;
    Ok(json_to_contraction_order(json))
}

fn json_to_contraction_order<L: Label>(json: ContractionOrderJson<L>) -> ContractionOrder<L> {
    let tree: NestedEinsum<L> = json.tree.into();

    match json.slices {
        Some(slices) => ContractionOrder::Sliced(SlicedEinsum::new(slices, tree)),
        None => ContractionOrder::Nested(tree),
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Detect the label type for JSON metadata.
fn detect_label_type<L: Label>() -> LabelType {
    let type_name = std::any::type_name::<L>();
    if type_name.contains("char") {
        LabelType::Char
    } else {
        LabelType::Int64
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eincode::uniform_size_dict;
    use crate::greedy::{optimize_greedy, GreedyMethod};
    use std::collections::HashMap;
    use tempfile::NamedTempFile;

    #[test]
    fn test_writejson_readjson_nested_char() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let sizes = uniform_size_dict(&code, 4);
        let tree = optimize_greedy(&code, &sizes, &GreedyMethod::default()).unwrap();

        let temp = NamedTempFile::new().unwrap();
        writejson(temp.path(), &tree).unwrap();

        let loaded: ContractionOrder<char> = readjson(temp.path()).unwrap();

        assert!(loaded.is_nested());
        let loaded_tree = loaded.into_nested().unwrap();
        assert_eq!(loaded_tree.leaf_count(), tree.leaf_count());
    }

    #[test]
    fn test_writejson_readjson_nested_usize() {
        let code = EinCode::new(vec![vec![1usize, 2], vec![2, 3]], vec![1, 3]);
        let sizes: HashMap<usize, usize> = [(1, 4), (2, 8), (3, 4)].into();
        let tree = optimize_greedy(&code, &sizes, &GreedyMethod::default()).unwrap();

        let temp = NamedTempFile::new().unwrap();
        writejson(temp.path(), &tree).unwrap();

        let loaded: ContractionOrder<usize> = readjson(temp.path()).unwrap();

        assert!(loaded.is_nested());
        let loaded_tree = loaded.into_nested().unwrap();
        assert_eq!(loaded_tree.leaf_count(), tree.leaf_count());
    }

    #[test]
    fn test_writejson_readjson_sliced() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let sizes = uniform_size_dict(&code, 8);
        let tree = optimize_greedy(&code, &sizes, &GreedyMethod::default()).unwrap();
        let sliced = SlicedEinsum::new(vec!['j'], tree);

        let temp = NamedTempFile::new().unwrap();
        writejson(temp.path(), &sliced).unwrap();

        let loaded: ContractionOrder<char> = readjson(temp.path()).unwrap();

        assert!(loaded.is_sliced());
        let loaded_sliced = loaded.into_sliced().unwrap();
        assert_eq!(loaded_sliced.slicing, vec!['j']);
    }

    #[test]
    fn test_to_json_string_from_json_string() {
        let code = EinCode::new(vec![vec!['a', 'b'], vec!['b', 'c']], vec!['a', 'c']);
        let sizes = uniform_size_dict(&code, 2);
        let tree = optimize_greedy(&code, &sizes, &GreedyMethod::default()).unwrap();

        let json_str = to_json_string(&tree).unwrap();
        assert!(json_str.contains("label-type"));
        assert!(json_str.contains("inputs"));
        assert!(json_str.contains("tree"));

        let loaded: ContractionOrder<char> = from_json_string(&json_str).unwrap();
        assert!(loaded.is_nested());
    }

    #[test]
    fn test_json_format_julia_compatible() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let sizes = uniform_size_dict(&code, 2);
        let tree = optimize_greedy(&code, &sizes, &GreedyMethod::default()).unwrap();

        let json_str = to_json_string(&tree).unwrap();

        // Parse as generic JSON to check structure
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        assert!(v.get("label-type").is_some(), "Should have label-type");
        assert!(v.get("inputs").is_some(), "Should have inputs");
        assert!(v.get("output").is_some(), "Should have output");
        assert!(v.get("tree").is_some(), "Should have tree");
        assert!(
            v.get("slices").is_none(),
            "Should not have slices for NestedEinsum"
        );
    }

    #[test]
    fn test_sliced_json_has_slices_field() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);
        let sizes = uniform_size_dict(&code, 8);
        let tree = optimize_greedy(&code, &sizes, &GreedyMethod::default()).unwrap();
        let sliced = SlicedEinsum::new(vec!['j'], tree);

        let json_str = to_json_string(&sliced).unwrap();

        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert!(
            v.get("slices").is_some(),
            "Should have slices for SlicedEinsum"
        );
    }

    #[test]
    fn test_deep_tree_json() {
        let code = EinCode::new(
            vec![
                vec!['a', 'b'],
                vec!['b', 'c'],
                vec!['c', 'd'],
                vec!['d', 'e'],
            ],
            vec!['a', 'e'],
        );
        let sizes = uniform_size_dict(&code, 2);
        let tree = optimize_greedy(&code, &sizes, &GreedyMethod::default()).unwrap();

        let json_str = to_json_string(&tree).unwrap();
        let loaded: ContractionOrder<char> = from_json_string(&json_str).unwrap();

        let loaded_tree = loaded.into_nested().unwrap();
        assert_eq!(loaded_tree.leaf_count(), 4);
        assert!(loaded_tree.is_binary());
    }

    #[test]
    fn test_scalar_output_json() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'i']], vec![]);
        let sizes = uniform_size_dict(&code, 3);
        let tree = optimize_greedy(&code, &sizes, &GreedyMethod::default()).unwrap();

        let json_str = to_json_string(&tree).unwrap();
        let loaded: ContractionOrder<char> = from_json_string(&json_str).unwrap();

        assert!(loaded.is_nested());
    }

    #[test]
    fn test_label_type_detection() {
        assert_eq!(detect_label_type::<char>(), LabelType::Char);
        assert_eq!(detect_label_type::<usize>(), LabelType::Int64);
        assert_eq!(detect_label_type::<i64>(), LabelType::Int64);
    }

    #[test]
    fn test_contraction_order_enum() {
        let tree: NestedEinsum<char> = NestedEinsum::leaf(0);
        let order: ContractionOrder<char> = tree.clone().into();

        assert!(order.is_nested());
        assert!(!order.is_sliced());
        assert!(order.as_nested().is_some());
        assert!(order.as_sliced().is_none());

        // Test into_nested on nested
        let nested_tree = order.into_nested().unwrap();
        assert_eq!(nested_tree.leaf_count(), 1);

        let sliced = SlicedEinsum::new(vec!['i'], tree.clone());
        let order: ContractionOrder<char> = sliced.into();

        assert!(!order.is_nested());
        assert!(order.is_sliced());
        assert!(order.as_nested().is_none());
        assert!(order.as_sliced().is_some());

        // Test into_sliced on sliced
        let sliced_tree = order.into_sliced().unwrap();
        assert_eq!(sliced_tree.slicing, vec!['i']);

        // Test into_nested on sliced (returns None)
        let order2: ContractionOrder<char> = SlicedEinsum::new(vec!['j'], tree.clone()).into();
        assert!(order2.into_nested().is_none());

        // Test into_sliced on nested (returns None)
        let order3: ContractionOrder<char> = tree.into();
        assert!(order3.into_sliced().is_none());
    }

    #[test]
    fn test_single_leaf_json() {
        // Test serializing a single leaf (edge case for coverage)
        let leaf: NestedEinsum<char> = NestedEinsum::leaf(0);

        // Note: A single leaf doesn't have an eins, so root_output returns vec![]
        // This tests the Leaf branch of root_output (line 311)
        let json_value = leaf.to_json_value().unwrap();
        assert!(json_value.get("output").is_some());
    }
}
