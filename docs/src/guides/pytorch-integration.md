# PyTorch Integration

Use omeco to optimize tensor contractions in PyTorch.

## Overview

omeco optimizes the **contraction order**, then you execute with PyTorch tensors.

**Workflow**:
1. Define network structure (indices and sizes)
2. Optimize with omeco → get contraction tree
3. Execute tree with PyTorch tensors

## Basic Example

```python
import torch
from omeco import optimize_code

# Define tensor network structure
ixs = [[0, 1], [1, 2], [2, 3]]  # Matrix chain
out = [0, 3]
sizes = {0: 100, 1: 200, 2: 50, 3: 100}

# Optimize contraction order
tree = optimize_code(ixs, out, sizes)
print(f"Optimized tree:\n{tree}")

# Create PyTorch tensors
tensors = [
    torch.randn(100, 200),  # A[0,1]
    torch.randn(200, 50),   # B[1,2]
    torch.randn(50, 100),   # C[2,3]
]

# Execute contraction (see full example below)
result = execute_tree(tree, tensors)
```

## Complete Example

See [examples/pytorch_tensor_network_example.py](https://github.com/GiggleLiu/omeco/blob/master/examples/pytorch_tensor_network_example.py) for a full working example with:
- Tree traversal and execution
- Verification against direct einsum
- 4×4 grid tensor network
- Memory reduction via slicing

## Converting Tree to Dict

```python
tree_dict = tree.to_dict()

# Dictionary structure:
# Leaf: {"tensor_index": int}
# Node: {"args": [children], "eins": {"ixs": [[int]], "iy": [int]}}
```

## Executing the Tree

```python
def execute_tree(tree_dict, tensors):
    """Recursively execute contraction tree."""
    if "tensor_index" in tree_dict:
        # Leaf node - return tensor
        return tensors[tree_dict["tensor_index"]]
    
    # Internal node - execute children then contract
    args = [execute_tree(child, tensors) for child in tree_dict["args"]]
    eins = tree_dict["eins"]
    
    # Build einsum string
    ixs_str = [indices_to_str(ix) for ix in eins["ixs"]]
    iy_str = indices_to_str(eins["iy"])
    einsum_eq = f"{','.join(ixs_str)}->{iy_str}"
    
    # Execute with PyTorch
    return torch.einsum(einsum_eq, *args)

def indices_to_str(indices):
    """Convert integer indices to letters."""
    return ''.join(chr(ord('a') + i) for i in indices)
```

## GPU Execution

```python
# Move tensors to GPU
tensors_gpu = [t.cuda() for t in tensors]

# Execute on GPU
result = execute_tree(tree.to_dict(), tensors_gpu)
```

## With Autograd

```python
# Tensors with gradients
tensors = [torch.randn(100, 200, requires_grad=True) for _ in range(3)]

# Execute
result = execute_tree(tree.to_dict(), tensors)

# Backward pass
result.sum().backward()

# Gradients available
print(tensors[0].grad)
```

## Next Steps

- [Score Function](./score-function.md) - Optimize for your hardware
- [GPU Optimization](./gpu-optimization.md) - Maximize GPU performance
- [Examples](https://github.com/GiggleLiu/omeco/tree/master/examples) - Complete code
