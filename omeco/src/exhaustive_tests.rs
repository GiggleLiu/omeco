use crate::complexity::nested_flop;
use crate::test_utils::{execute_nested, tensors_approx_equal, NaiveContractor};
use crate::{
    optimize_code, optimize_exhaustive, EinCode, ExhaustiveSearch, GreedyMethod, NestedEinsum,
};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
enum TestTree {
    Leaf(usize),
    Node(Box<TestTree>, Box<TestTree>),
}

fn sizes(values: &[(usize, usize)]) -> HashMap<usize, usize> {
    values.iter().copied().collect()
}

fn set_of_tree(tree: &TestTree) -> HashSet<usize> {
    match tree {
        TestTree::Leaf(i) => HashSet::from([*i]),
        TestTree::Node(left, right) => {
            let mut set = set_of_tree(left);
            set.extend(set_of_tree(right));
            set
        }
    }
}

fn open_labels(ixs: &[Vec<usize>], iy: &[usize], tensors: &HashSet<usize>) -> Vec<usize> {
    let output: HashSet<_> = iy.iter().copied().collect();
    let mut result = Vec::new();
    let mut seen = HashSet::new();

    for &tensor in tensors {
        for &label in &ixs[tensor] {
            if seen.contains(&label) {
                continue;
            }

            let appears_outside = ixs
                .iter()
                .enumerate()
                .any(|(i, ix)| !tensors.contains(&i) && ix.contains(&label));
            if output.contains(&label) || appears_outside {
                seen.insert(label);
                result.push(label);
            }
        }
    }

    result.sort_unstable();
    result
}

fn test_tree_to_nested(
    tree: &TestTree,
    ixs: &[Vec<usize>],
    iy: &[usize],
    full_set: &HashSet<usize>,
) -> NestedEinsum<usize> {
    match tree {
        TestTree::Leaf(i) => NestedEinsum::leaf(*i),
        TestTree::Node(left, right) => {
            let left_nested = test_tree_to_nested(left, ixs, iy, full_set);
            let right_nested = test_tree_to_nested(right, ixs, iy, full_set);
            let left_set = set_of_tree(left);
            let right_set = set_of_tree(right);
            let mut merged = left_set.clone();
            merged.extend(right_set);

            let left_labels = left_nested.output_labels(ixs);
            let right_labels = right_nested.output_labels(ixs);
            let output = if &merged == full_set {
                iy.to_vec()
            } else {
                open_labels(ixs, iy, &merged)
            };

            NestedEinsum::node(
                vec![left_nested, right_nested],
                EinCode::new(vec![left_labels, right_labels], output),
            )
        }
    }
}

fn all_binary_trees(leaves: &[usize]) -> Vec<TestTree> {
    if leaves.len() == 1 {
        return vec![TestTree::Leaf(leaves[0])];
    }

    let first = leaves[0];
    let rest = &leaves[1..];
    let mut trees = Vec::new();

    for mask in 0usize..(1usize << rest.len()) {
        let mut left = vec![first];
        let mut right = Vec::new();

        for (bit, &leaf) in rest.iter().enumerate() {
            if (mask & (1usize << bit)) == 0 {
                left.push(leaf);
            } else {
                right.push(leaf);
            }
        }

        if right.is_empty() {
            continue;
        }

        for left_tree in all_binary_trees(&left) {
            for right_tree in all_binary_trees(&right) {
                trees.push(TestTree::Node(
                    Box::new(left_tree.clone()),
                    Box::new(right_tree),
                ));
            }
        }
    }

    trees
}

fn brute_force_min_flop(code: &EinCode<usize>, size_dict: &HashMap<usize, usize>) -> usize {
    let leaves: Vec<_> = (0..code.num_tensors()).collect();
    let full_set: HashSet<_> = leaves.iter().copied().collect();

    all_binary_trees(&leaves)
        .iter()
        .map(|tree| {
            let nested = test_tree_to_nested(tree, &code.ixs, &code.iy, &full_set);
            nested_flop(&nested, size_dict)
        })
        .min()
        .unwrap()
}

fn assert_exhaustive_is_optimal(code: EinCode<usize>, size_dict: HashMap<usize, usize>) {
    let nested = optimize_exhaustive(&code, &size_dict, &ExhaustiveSearch::default()).unwrap();
    assert!(nested.is_binary());
    assert_eq!(
        nested_flop(&nested, &size_dict),
        brute_force_min_flop(&code, &size_dict)
    );
}

#[test]
fn exhaustive_matches_bruteforce_on_five_tensor_network() {
    let code = EinCode::new(
        vec![
            vec![0, 1],
            vec![0, 2, 3],
            vec![1, 2, 4, 5],
            vec![4],
            vec![3, 5],
        ],
        vec![],
    );
    let size_dict = sizes(&[(0, 2), (1, 4), (2, 8), (3, 16), (4, 32), (5, 64)]);

    assert_exhaustive_is_optimal(code, size_dict);
}

#[test]
fn exhaustive_finds_matrix_chain_optimum() {
    let code = EinCode::new(vec![vec![0, 1], vec![1, 2], vec![2, 3]], vec![0, 3]);
    let size_dict = sizes(&[(0, 2), (1, 3), (2, 4), (3, 5)]);

    assert_exhaustive_is_optimal(code, size_dict);
}

#[test]
fn exhaustive_preserves_numerical_result() {
    let code = EinCode::new(
        vec![
            vec![0, 1],
            vec![0, 2, 3],
            vec![1, 2, 4, 5],
            vec![4],
            vec![3, 5],
        ],
        vec![],
    );
    let size_dict = sizes(&[(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)]);
    let exact = optimize_exhaustive(&code, &size_dict, &ExhaustiveSearch::default()).unwrap();
    let greedy = optimize_code(&code, &size_dict, &GreedyMethod::default()).unwrap();

    let mut contractor = NaiveContractor::new();
    for (tensor, labels) in code.ixs.iter().enumerate() {
        contractor.add_tensor(
            tensor,
            labels.iter().map(|label| size_dict[label]).collect(),
        );
    }
    let label_map: HashMap<_, _> = code
        .unique_labels()
        .into_iter()
        .map(|label| (label, label))
        .collect();

    let mut exact_contractor = contractor.clone();
    let exact_idx = execute_nested(&exact, &mut exact_contractor, &label_map);
    let greedy_idx = execute_nested(&greedy, &mut contractor, &label_map);
    let exact_result = exact_contractor.get_tensor(exact_idx).unwrap();
    let greedy_result = contractor.get_tensor(greedy_idx).unwrap();

    assert!(tensors_approx_equal(
        exact_result,
        greedy_result,
        1e-9,
        1e-12
    ));
}

#[test]
fn exhaustive_preserves_root_output_order() {
    let code = EinCode::new(vec![vec![0, 1], vec![1, 2], vec![2, 3]], vec![3, 0]);
    let size_dict = sizes(&[(0, 2), (1, 3), (2, 4), (3, 5)]);
    let nested = optimize_exhaustive(&code, &size_dict, &ExhaustiveSearch::default()).unwrap();

    assert_eq!(nested.output_labels(&code.ixs), vec![3, 0]);
}

#[test]
fn exhaustive_combines_disconnected_components_by_outer_product() {
    let code = EinCode::new(
        vec![vec![0, 1], vec![1, 2], vec![3, 4], vec![4, 5]],
        vec![0, 2, 3, 5],
    );
    let size_dict = sizes(&[(0, 2), (1, 3), (2, 5), (3, 7), (4, 11), (5, 13)]);

    assert_exhaustive_is_optimal(code, size_dict);
}

#[test]
fn exhaustive_handles_trivial_one_and_two_tensor_inputs() {
    let one = EinCode::new(vec![vec![0, 1]], vec![0, 1]);
    let two = EinCode::new(vec![vec![0, 1], vec![1, 2]], vec![0, 2]);
    let size_dict = sizes(&[(0, 2), (1, 3), (2, 5)]);

    let one_nested = optimize_exhaustive(&one, &size_dict, &ExhaustiveSearch::default()).unwrap();
    let two_nested = optimize_exhaustive(&two, &size_dict, &ExhaustiveSearch::default()).unwrap();

    assert_eq!(one_nested.leaf_count(), 1);
    assert_eq!(two_nested.leaf_count(), 2);
    assert_eq!(two_nested.output_labels(&two.ixs), two.iy);
}

#[test]
fn exhaustive_supports_hyperedges_summed_out() {
    let code = EinCode::new(vec![vec![0, 1], vec![0, 2], vec![0, 3]], vec![1, 2, 3]);
    let size_dict = sizes(&[(0, 5), (1, 2), (2, 3), (3, 7)]);

    assert_exhaustive_is_optimal(code, size_dict);
}

#[test]
fn exhaustive_supports_batch_diagonal_output_indices() {
    let code = EinCode::new(vec![vec![0, 1], vec![0, 2], vec![2, 3]], vec![0, 1, 3]);
    let size_dict = sizes(&[(0, 2), (1, 3), (2, 5), (3, 7)]);

    assert_exhaustive_is_optimal(code, size_dict);
}

#[test]
fn exhaustive_handles_dimension_one_contracted_indices() {
    let code = EinCode::new(
        vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 4]],
        vec![0, 4],
    );
    let size_dict = sizes(&[(0, 3), (1, 1), (2, 5), (3, 1), (4, 7)]);

    assert_exhaustive_is_optimal(code, size_dict);
}

#[test]
fn exhaustive_handles_all_dimension_one_indices() {
    let code = EinCode::new(vec![vec![0, 1], vec![1, 2], vec![2, 3]], vec![0, 3]);
    let size_dict = sizes(&[(0, 1), (1, 1), (2, 1), (3, 1)]);

    assert_exhaustive_is_optimal(code, size_dict);
}

#[test]
fn exhaustive_handles_dimension_one_output_indices() {
    let code = EinCode::new(vec![vec![0, 1], vec![1, 2], vec![2, 3]], vec![0, 3]);
    let size_dict = sizes(&[(0, 1), (1, 4), (2, 1), (3, 1)]);

    assert_exhaustive_is_optimal(code, size_dict);
}

#[test]
fn exhaustive_reports_scope_errors_for_partial_trace_and_dangling_sums() {
    let partial_trace = EinCode::new(vec![vec![0, 0, 1], vec![1, 2], vec![2, 3]], vec![3]);
    let dangling_sum = EinCode::new(vec![vec![0, 1], vec![1, 2], vec![3, 4]], vec![0, 2]);
    let size_dict = sizes(&[(0, 2), (1, 3), (2, 5), (3, 7), (4, 11)]);

    assert!(optimize_exhaustive(&partial_trace, &size_dict, &ExhaustiveSearch::default()).is_err());
    assert!(optimize_exhaustive(&dangling_sum, &size_dict, &ExhaustiveSearch::default()).is_err());
}

#[test]
fn exhaustive_supports_char_labels_through_optimizer_trait() {
    let code = EinCode::new(
        vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
        vec!['i', 'l'],
    );
    let size_dict = HashMap::from([('i', 2), ('j', 3), ('k', 4), ('l', 5)]);

    let nested = optimize_code(&code, &size_dict, &ExhaustiveSearch::default()).unwrap();

    assert!(nested.is_binary());
    assert_eq!(nested.output_labels(&code.ixs), code.iy);
}

#[test]
fn exhaustive_trait_returns_none_for_unsupported_scope() {
    let code = EinCode::new(vec![vec![0, 0, 1], vec![1, 2], vec![2, 3]], vec![3]);
    let size_dict = sizes(&[(0, 2), (1, 3), (2, 5), (3, 7)]);

    assert!(optimize_code(&code, &size_dict, &ExhaustiveSearch::default()).is_none());
}
