"""
Example: Using omeco with PyTorch for tensor network contraction.

Demonstrates: optimize_code, slice_code, and contraction with PyTorch.
"""

import torch
from omeco import (
    optimize_code, slice_code, contraction_complexity, sliced_complexity,
    GreedyMethod, TreeSA, TreeSASlicer, ScoreFunction, NestedEinsum,
)


def contract(tree: NestedEinsum, tensors: list[torch.Tensor]) -> torch.Tensor:
    """Contract tensors according to the optimized tree."""
    return _contract_recursive(tree.to_dict(), tensors)


def _contract_recursive(tree_dict: dict, tensors: list[torch.Tensor]) -> torch.Tensor:
    if "tensor_index" in tree_dict:
        return tensors[tree_dict["tensor_index"]]
    args = [_contract_recursive(arg, tensors) for arg in tree_dict["args"]]
    return _einsum_int(tree_dict["eins"]["ixs"], tree_dict["eins"]["iy"], args)


def _einsum_int(ixs: list[list[int]], iy: list[int], tensors: list[torch.Tensor]) -> torch.Tensor:
    """Execute einsum with integer index labels."""
    all_labels = set(sum(ixs, []) + iy)
    label_map = {l: chr(ord('a') + i) for i, l in enumerate(sorted(all_labels))}
    inputs = ",".join("".join(label_map[l] for l in ix) for ix in ixs)
    output = "".join(label_map[l] for l in iy)
    return torch.einsum(f"{inputs}->{output}", *tensors)


def main():
    # 4x4 grid tensor network (contracts to scalar)
    grid_size, bond_dim = 4, 8
    
    def edge_idx(r, c, d):
        return r * (grid_size * 2) + c * 2 + d
    
    ixs = []
    for r in range(grid_size):
        for c in range(grid_size):
            idx = []
            if c > 0: idx.append(edge_idx(r, c-1, 0))
            if c < grid_size-1: idx.append(edge_idx(r, c, 0))
            if r > 0: idx.append(edge_idx(r-1, c, 1))
            if r < grid_size-1: idx.append(edge_idx(r, c, 1))
            ixs.append(idx)
    
    out = []
    sizes = {i: bond_dim for i in set(sum(ixs, []))}
    tensors = [torch.randn(*[sizes[i] for i in ix]) for ix in ixs]

    # 1. Optimize contraction order (compare Greedy vs TreeSA)
    tree_greedy = optimize_code(ixs, out, sizes, GreedyMethod())
    tree_treesa = optimize_code(ixs, out, sizes, TreeSA.fast())
    
    c_greedy = contraction_complexity(tree_greedy, ixs, sizes)
    c_treesa = contraction_complexity(tree_treesa, ixs, sizes)
    
    print(f"Greedy: tc=2^{c_greedy.tc:.1f}, sc=2^{c_greedy.sc:.1f}")
    print(f"TreeSA: tc=2^{c_treesa.tc:.1f}, sc=2^{c_treesa.sc:.1f}")

    # 2. Slice to reduce memory (trade space for time)
    slicer = TreeSASlicer.fast(score=ScoreFunction(sc_target=9.0))
    sliced = slice_code(tree_treesa, ixs, sizes, slicer)
    c_sliced = sliced_complexity(sliced, ixs, sizes)
    
    print(f"Sliced: tc=2^{c_sliced.tc:.1f}, sc=2^{c_sliced.sc:.1f}, sliced={sliced.slicing()}")

    # 3. Contract and verify
    result = contract(tree_treesa, tensors)
    expected = _einsum_int(ixs, out, tensors)
    
    print(f"Result: {result.item():.6f}, error: {abs(result-expected).item()/abs(expected).item():.1e}")


if __name__ == "__main__":
    main()
