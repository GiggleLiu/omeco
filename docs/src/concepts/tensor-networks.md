# Tensor Networks

## What are Tensor Networks?

A **tensor network** is a collection of multidimensional arrays (tensors) connected by shared indices. Contracting a tensor network means computing their products and sums according to the Einstein summation convention.

## Einstein Summation

The einsum notation specifies which indices to sum over:

```
einsum("ij,jk->ik", A, B)
```

This means:
- `A` has shape `[i, j]`
- `B` has shape `[j, k]`
- Sum over shared index `j`
- Result has shape `[i, k]`

Equivalent to matrix multiplication: `C[i,k] = Σⱼ A[i,j] * B[j,k]`

## Common Patterns

### Matrix Multiplication

```python
# C = A × B
einsum("ij,jk->ik", A, B)
```

### Batch Matrix Multiplication

```python
# Multiple matrix multiplications at once
einsum("bij,bjk->bik", A, B)
```

### Trace

```python
# Sum of diagonal elements
einsum("ii->", A)
```

### Outer Product

```python
# C[i,j] = A[i] * B[j]
einsum("i,j->ij", a, b)
```

### Tensor Contraction with Multiple Indices

```python
# Contract over indices j and k
einsum("ijk,klj->il", A, B)
```

## Tensor Networks in Practice

### Quantum Circuit Simulation

Quantum gates are represented as tensors, and circuit evaluation becomes a tensor network contraction:

```python
# 3-qubit quantum circuit
gates = [
    [[...]]  # Gate on qubits 0,1
    [[...]]  # Gate on qubits 1,2
    [[...]]  # Gate on qubits 0,2
]

# Each qubit index appears multiple times
# Contracting computes the circuit output
```

### Neural Networks

Einsum operations appear in:
- Attention mechanisms: `einsum("bqd,bkd->bqk", Q, K)`
- Tensor decompositions: Tucker, CP, tensor train
- Graph neural networks

### Scientific Computing

- **Physics**: Partition functions, path integrals
- **Chemistry**: Molecular orbital calculations
- **Mathematics**: Graph polynomials, combinatorial problems

## Contraction Order Matters

Given tensors A, B, C to contract, we can do:

1. `(A × B) × C`
2. `A × (B × C)`
3. `(A × C) × B`

These give the same result but have **vastly different costs**.

Example with shapes `A[10,100]`, `B[100,20]`, `C[20,5]`:

| Order | FLOPs | Peak Memory |
|-------|-------|-------------|
| `(A×B)×C` | 20,000 + 1,000 = 21,000 | 200 |
| `A×(B×C)` | 10,000 + 50,000 = 60,000 | 2,000 |

The first is **3x faster** and uses **10x less memory**!

## Representing Tensor Networks

### As Lists of Index Lists

```python
# Matrix chain A×B×C
ixs = [
    [0, 1],  # A has indices [i, j]
    [1, 2],  # B has indices [j, k]
    [2, 3],  # C has indices [k, l]
]
out = [0, 3]  # Output has indices [i, l]
```

### As Graphs

Each tensor is a node, shared indices are edges:

```
A[i,j] --- j --- B[j,k] --- k --- C[k,l]
  i                                   l
```

Contracting removes edges and merges nodes.

## Next Steps

- [Contraction Order Problem](./contraction-order.md) - Why optimization is needed
- [Complexity Metrics](./complexity-metrics.md) - How to measure cost
- [Quick Start](../quick-start.md) - Optimize your first tensor network
