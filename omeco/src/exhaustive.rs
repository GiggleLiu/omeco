//! Exact contraction-order optimizer using exhaustive dynamic programming.
//!
//! `ExhaustiveSearch` is intended for small tensor networks where the optimal
//! total contraction FLOP count is more important than optimizer runtime.

use crate::{EinCode, Label, NestedEinsum};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

type Mask = u128;

/// Exact contraction-order optimizer.
///
/// This optimizer minimizes total FLOP count within each connected component.
/// Disconnected components are then combined with outer products from smallest
/// output size to largest.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExhaustiveSearch {
    /// Print progress information during the search.
    pub verbose: bool,
}

impl ExhaustiveSearch {
    /// Create a new exhaustive optimizer.
    pub fn new(verbose: bool) -> Self {
        Self { verbose }
    }
}

/// Errors reported by [`optimize_exhaustive`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExhaustiveSearchError {
    /// The input has no tensors.
    #[error("ExhaustiveSearch requires at least one tensor")]
    EmptyInput,
    /// The exact bitset implementation currently supports up to 128 tensors.
    #[error("ExhaustiveSearch supports at most 128 tensors, got {0}")]
    TooManyTensors(usize),
    /// The exact bitset implementation currently supports up to 128 labels.
    #[error("ExhaustiveSearch supports at most 128 unique labels, got {0}")]
    TooManyLabels(usize),
    /// An output label is not present in any input tensor.
    #[error("output label {0} does not appear in the input tensors")]
    InvalidOutputLabel(String),
    /// A single tensor repeats a label, which would require a unary trace.
    #[error(
        "partial traces are not supported: label {label} appears more than once in tensor {tensor}"
    )]
    PartialTrace { tensor: usize, label: String },
    /// A summed index appears in only one tensor, which would require a unary sum.
    #[error("dangling summed indices are not supported: label {0} appears in only one tensor and is not an output label")]
    DanglingSummedIndex(String),
    /// A connected component could not be contracted under the supported scope.
    #[error("could not construct a connected binary contraction tree")]
    NoContractionTree,
    /// The configured or inferred dimensions overflowed `usize`.
    #[error("contraction cost overflowed usize")]
    CostOverflow,
}

#[derive(Debug, Clone)]
enum SearchTree {
    Leaf(usize),
    Node(Box<SearchTree>, Box<SearchTree>),
}

#[derive(Debug, Clone)]
struct DpEntry {
    cost: usize,
    tree: SearchTree,
}

#[derive(Debug, Clone)]
struct ComponentResult<L: Label> {
    tensor_mask: Mask,
    labels: Vec<L>,
    tree: NestedEinsum<L>,
    output_size: usize,
}

struct SearchContext<'a, L: Label> {
    code: &'a EinCode<L>,
    size_dict: &'a HashMap<L, usize>,
    labels: Vec<L>,
    label_tensor_masks: Vec<Mask>,
    output_label_mask: Mask,
    full_tensor_mask: Mask,
}

/// Optimize an [`EinCode`] with exact dynamic programming.
pub fn optimize_exhaustive<L: Label>(
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    config: &ExhaustiveSearch,
) -> Result<NestedEinsum<L>, ExhaustiveSearchError> {
    let n = code.num_tensors();
    if n == 0 {
        return Err(ExhaustiveSearchError::EmptyInput);
    }
    if n > 128 {
        return Err(ExhaustiveSearchError::TooManyTensors(n));
    }

    if n == 1 {
        return Ok(NestedEinsum::leaf(0));
    }
    if n == 2 {
        return Ok(NestedEinsum::node(
            vec![NestedEinsum::leaf(0), NestedEinsum::leaf(1)],
            code.clone(),
        ));
    }

    validate_scope(code)?;
    let ctx = SearchContext::new(code, size_dict)?;
    let components = connected_components(&ctx);

    if config.verbose {
        eprintln!(
            "ExhaustiveSearch: {} tensors, {} labels, {} connected component(s)",
            n,
            ctx.labels.len(),
            components.len()
        );
    }

    let mut results = Vec::with_capacity(components.len());
    for component in components {
        results.push(optimize_component(&ctx, component)?);
    }

    combine_components(&ctx, results)
}

impl<'a, L: Label> SearchContext<'a, L> {
    fn new(
        code: &'a EinCode<L>,
        size_dict: &'a HashMap<L, usize>,
    ) -> Result<Self, ExhaustiveSearchError> {
        let labels = code.unique_labels();
        if labels.len() > 128 {
            return Err(ExhaustiveSearchError::TooManyLabels(labels.len()));
        }

        let label_to_pos: HashMap<L, usize> = labels
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, label)| (label, i))
            .collect();
        let mut label_tensor_masks = vec![0; labels.len()];

        for (tensor, ix) in code.ixs.iter().enumerate() {
            let bit = bit(tensor);
            let mut seen = HashSet::new();
            for label in ix {
                if seen.insert(label) {
                    let pos = label_to_pos[label];
                    label_tensor_masks[pos] |= bit;
                }
            }
        }

        let mut output_label_mask = 0;
        for label in &code.iy {
            let pos = label_to_pos
                .get(label)
                .copied()
                .ok_or_else(|| ExhaustiveSearchError::InvalidOutputLabel(format!("{label:?}")))?;
            output_label_mask |= bit(pos);
        }

        Ok(Self {
            code,
            size_dict,
            labels,
            label_tensor_masks,
            output_label_mask,
            full_tensor_mask: first_n_bits(code.num_tensors()),
        })
    }

    fn open_label_mask(&self, tensor_mask: Mask) -> Mask {
        let outside = self.full_tensor_mask & !tensor_mask;
        let mut labels = 0;

        for (label_pos, &label_tensors) in self.label_tensor_masks.iter().enumerate() {
            if label_tensors & tensor_mask == 0 {
                continue;
            }

            let is_output = self.output_label_mask & bit(label_pos) != 0;
            let appears_outside = label_tensors & outside != 0;
            if is_output || appears_outside {
                labels |= bit(label_pos);
            }
        }

        labels
    }

    fn labels_from_mask(&self, label_mask: Mask) -> Vec<L> {
        self.labels
            .iter()
            .enumerate()
            .filter_map(|(i, label)| {
                if label_mask & bit(i) != 0 {
                    Some(label.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    fn root_labels_for_mask(&self, tensor_mask: Mask) -> Vec<L> {
        if tensor_mask == self.full_tensor_mask {
            return self.code.iy.clone();
        }

        let open_mask = self.open_label_mask(tensor_mask);
        self.code
            .iy
            .iter()
            .filter(|label| {
                self.labels
                    .iter()
                    .position(|candidate| candidate == *label)
                    .is_some_and(|pos| open_mask & bit(pos) != 0)
            })
            .cloned()
            .collect()
    }

    fn label_mask_size(&self, label_mask: Mask) -> Result<usize, ExhaustiveSearchError> {
        let mut size = 1usize;
        for (i, label) in self.labels.iter().enumerate() {
            if label_mask & bit(i) == 0 {
                continue;
            }
            let dim = self.size_dict.get(label).copied().unwrap_or(1);
            size = size
                .checked_mul(dim)
                .ok_or(ExhaustiveSearchError::CostOverflow)?;
        }
        Ok(size)
    }
}

fn validate_scope<L: Label>(code: &EinCode<L>) -> Result<(), ExhaustiveSearchError> {
    let output_labels: HashSet<_> = code.iy.iter().cloned().collect();
    let mut occurrence_counts: HashMap<L, usize> = HashMap::new();

    for (tensor, ix) in code.ixs.iter().enumerate() {
        let mut seen = HashSet::new();
        for label in ix {
            if !seen.insert(label.clone()) {
                return Err(ExhaustiveSearchError::PartialTrace {
                    tensor,
                    label: format!("{label:?}"),
                });
            }
        }

        for label in seen {
            *occurrence_counts.entry(label).or_insert(0) += 1;
        }
    }

    for label in &code.iy {
        if !occurrence_counts.contains_key(label) {
            return Err(ExhaustiveSearchError::InvalidOutputLabel(format!(
                "{label:?}"
            )));
        }
    }

    for (label, count) in occurrence_counts {
        if count == 1 && !output_labels.contains(&label) {
            return Err(ExhaustiveSearchError::DanglingSummedIndex(format!(
                "{label:?}"
            )));
        }
    }

    Ok(())
}

fn connected_components<L: Label>(ctx: &SearchContext<'_, L>) -> Vec<Mask> {
    let mut components = Vec::new();
    let mut unvisited = ctx.full_tensor_mask;

    while unvisited != 0 {
        let start = unvisited & unvisited.wrapping_neg();
        let mut component = 0;
        let mut frontier = start;
        unvisited &= !start;

        while frontier != 0 {
            let tensor_bit = frontier & frontier.wrapping_neg();
            frontier &= !tensor_bit;
            component |= tensor_bit;

            let mut neighbors = 0;
            for &label_tensors in &ctx.label_tensor_masks {
                if label_tensors & tensor_bit != 0 {
                    neighbors |= label_tensors;
                }
            }

            let new_neighbors = neighbors & unvisited;
            frontier |= new_neighbors;
            unvisited &= !new_neighbors;
        }

        components.push(component);
    }

    components
}

fn optimize_component<L: Label>(
    ctx: &SearchContext<'_, L>,
    component: Mask,
) -> Result<ComponentResult<L>, ExhaustiveSearchError> {
    if component.count_ones() == 1 {
        let tensor = singleton_index(component);
        let labels = ctx.root_labels_for_mask(component);
        let output_size = ctx.label_mask_size(ctx.open_label_mask(component))?;
        return Ok(ComponentResult {
            tensor_mask: component,
            labels,
            tree: NestedEinsum::leaf(tensor),
            output_size,
        });
    }

    let component_size = component.count_ones() as usize;
    let mut by_size: Vec<HashMap<Mask, DpEntry>> =
        (0..=component_size).map(|_| HashMap::new()).collect();

    for tensor in bits(component) {
        let mask = bit(tensor);
        by_size[1].insert(
            mask,
            DpEntry {
                cost: 0,
                tree: SearchTree::Leaf(tensor),
            },
        );
    }

    for size in 2..=component_size {
        for subset in submasks_with_size(component, size) {
            let mut best: Option<DpEntry> = None;
            let anchor = subset & subset.wrapping_neg();
            let mut left = (subset - 1) & subset;

            while left != 0 {
                let right = subset ^ left;
                if right != 0 && left & anchor != 0 {
                    let left_size = left.count_ones() as usize;
                    let right_size = right.count_ones() as usize;

                    if let (Some(left_entry), Some(right_entry)) = (
                        by_size[left_size].get(&left),
                        by_size[right_size].get(&right),
                    ) {
                        let shared_labels = ctx.open_label_mask(left) & ctx.open_label_mask(right);
                        if shared_labels != 0 {
                            let merge_label_mask =
                                ctx.open_label_mask(left) | ctx.open_label_mask(right);
                            let merge_cost = ctx.label_mask_size(merge_label_mask)?;
                            let cost = left_entry
                                .cost
                                .checked_add(right_entry.cost)
                                .and_then(|c| c.checked_add(merge_cost))
                                .ok_or(ExhaustiveSearchError::CostOverflow)?;

                            if best.as_ref().map_or(true, |entry| cost < entry.cost) {
                                best = Some(DpEntry {
                                    cost,
                                    tree: SearchTree::Node(
                                        Box::new(left_entry.tree.clone()),
                                        Box::new(right_entry.tree.clone()),
                                    ),
                                });
                            }
                        }
                    }
                }
                left = (left - 1) & subset;
            }

            if let Some(entry) = best {
                by_size[size].insert(subset, entry);
            }
        }
    }

    let entry = by_size[component_size]
        .get(&component)
        .ok_or(ExhaustiveSearchError::NoContractionTree)?;
    let labels = ctx.root_labels_for_mask(component);
    let tree = build_nested(ctx, &entry.tree, component, true);
    let output_size = ctx.label_mask_size(ctx.open_label_mask(component))?;

    Ok(ComponentResult {
        tensor_mask: component,
        labels,
        tree,
        output_size,
    })
}

fn build_nested<L: Label>(
    ctx: &SearchContext<'_, L>,
    tree: &SearchTree,
    tensor_mask: Mask,
    force_root_labels: bool,
) -> NestedEinsum<L> {
    match tree {
        SearchTree::Leaf(tensor) => NestedEinsum::leaf(*tensor),
        SearchTree::Node(left, right) => {
            let left_mask = tree_tensor_mask(left);
            let right_mask = tensor_mask ^ left_mask;
            let left_nested = build_nested(ctx, left, left_mask, false);
            let right_nested = build_nested(ctx, right, right_mask, false);
            let left_labels = left_nested.output_labels(&ctx.code.ixs);
            let right_labels = right_nested.output_labels(&ctx.code.ixs);
            let output = if force_root_labels {
                ctx.root_labels_for_mask(tensor_mask)
            } else {
                ctx.labels_from_mask(ctx.open_label_mask(tensor_mask))
            };

            NestedEinsum::node(
                vec![left_nested, right_nested],
                EinCode::new(vec![left_labels, right_labels], output),
            )
        }
    }
}

fn combine_components<L: Label>(
    ctx: &SearchContext<'_, L>,
    mut results: Vec<ComponentResult<L>>,
) -> Result<NestedEinsum<L>, ExhaustiveSearchError> {
    if results.is_empty() {
        return Err(ExhaustiveSearchError::NoContractionTree);
    }

    while results.len() > 1 {
        results.sort_by_key(|result| result.output_size);
        let left = results.remove(0);
        let right = results.remove(0);
        let tensor_mask = left.tensor_mask | right.tensor_mask;
        let output = if tensor_mask == ctx.full_tensor_mask {
            ctx.code.iy.clone()
        } else {
            ctx.root_labels_for_mask(tensor_mask)
        };
        let output_size = ctx.label_mask_size(ctx.open_label_mask(tensor_mask))?;
        let tree = NestedEinsum::node(
            vec![left.tree, right.tree],
            EinCode::new(vec![left.labels, right.labels], output.clone()),
        );

        results.push(ComponentResult {
            tensor_mask,
            labels: output,
            tree,
            output_size,
        });
    }

    Ok(results.pop().unwrap().tree)
}

fn tree_tensor_mask(tree: &SearchTree) -> Mask {
    match tree {
        SearchTree::Leaf(tensor) => bit(*tensor),
        SearchTree::Node(left, right) => tree_tensor_mask(left) | tree_tensor_mask(right),
    }
}

fn first_n_bits(n: usize) -> Mask {
    if n == 128 {
        Mask::MAX
    } else {
        (1u128 << n) - 1
    }
}

fn bit(pos: usize) -> Mask {
    1u128 << pos
}

fn singleton_index(mask: Mask) -> usize {
    mask.trailing_zeros() as usize
}

fn bits(mask: Mask) -> impl Iterator<Item = usize> {
    (0..128).filter(move |&i| mask & bit(i) != 0)
}

fn submasks_with_size(mask: Mask, size: usize) -> impl Iterator<Item = Mask> {
    let mut submasks = Vec::new();
    let mut submask = mask;
    while submask != 0 {
        if submask.count_ones() as usize == size {
            submasks.push(submask);
        }
        submask = (submask - 1) & mask;
    }
    submasks.into_iter()
}
