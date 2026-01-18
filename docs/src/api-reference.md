# API Reference

Quick reference for omeco's Python and Rust APIs.

## Python API

Full Python API documentation is available in the package docstrings. This page provides a quick overview.

### Optimization Functions

#### `optimize_code`

```python
def optimize_code(
    ixs: List[List[int]],
    out: List[int],
    sizes: Dict[int, int],
    optimizer: Optional[Optimizer] = None
) -> NestedEinsum
```

Optimize tensor contraction order.

**Parameters**:
- `ixs`: List of index lists for each tensor (e.g., `[[0, 1], [1, 2]]`)
- `out`: Output indices (e.g., `[0, 2]`)
- `sizes`: Dictionary mapping indices to dimensions (e.g., `{0: 10, 1: 20, 2: 10}`)
- `optimizer`: Optimizer instance (default: `GreedyMethod()`)

**Returns**: `NestedEinsum` representing the optimized contraction tree

**Example**:
```python
from omeco import optimize_code, TreeSA

tree = optimize_code(
    ixs=[[0, 1], [1, 2], [2, 3]],
    out=[0, 3],
    sizes={0: 10, 1: 20, 2: 30, 3: 10},
    optimizer=TreeSA.fast()
)
```

#### `optimize_greedy`

```python
def optimize_greedy(
    ixs: List[List[int]],
    out: List[int],
    sizes: Dict[int, int]
) -> NestedEinsum
```

Optimize using greedy method (shorthand for `optimize_code` with `GreedyMethod()`).

**Example**:
```python
from omeco import optimize_greedy

tree = optimize_greedy(
    ixs=[[0, 1], [1, 2]],
    out=[0, 2],
    sizes={0: 10, 1: 20, 2: 10}
)
```

### Slicing Functions

#### `slice_code`

```python
def slice_code(
    tree: NestedEinsum,
    ixs: List[List[int]],
    sizes: Dict[int, int],
    slicer: Slicer
) -> SlicedCode
```

Apply slicing to reduce memory usage.

**Parameters**:
- `tree`: Optimized contraction tree from `optimize_code`
- `ixs`: Original index lists
- `sizes`: Dimension sizes
- `slicer`: Slicer instance (e.g., `TreeSASlicer.fast()`)

**Returns**: `SlicedCode` with slicing applied

**Example**:
```python
from omeco import optimize_code, slice_code, TreeSASlicer, ScoreFunction

tree = optimize_code(ixs, out, sizes)

slicer = TreeSASlicer.fast(
    score=ScoreFunction(sc_target=28.0)
)
sliced = slice_code(tree, ixs, sizes, slicer)
```

### Complexity Functions

#### `contraction_complexity`

```python
def contraction_complexity(
    tree: NestedEinsum,
    ixs: List[List[int]],
    sizes: Dict[int, int]
) -> Complexity
```

Calculate complexity metrics for a contraction tree.

**Returns**: `Complexity` object with fields:
- `tc`: Time complexity (log₂ FLOPs)
- `sc`: Space complexity (log₂ elements)
- `rwc`: Read-write complexity (log₂ elements)

**Example**:
```python
from omeco import optimize_code, contraction_complexity

tree = optimize_code(ixs, out, sizes)
comp = contraction_complexity(tree, ixs, sizes)

print(f"Time: 2^{comp.tc:.2f} FLOPs")
print(f"Space: 2^{comp.sc:.2f} elements")
print(f"Read-write: 2^{comp.rwc:.2f} elements")
```

#### `sliced_complexity`

```python
def sliced_complexity(
    sliced: SlicedCode,
    ixs: List[List[int]],
    sizes: Dict[int, int]
) -> Complexity
```

Calculate complexity for sliced code.

**Example**:
```python
from omeco import slice_code, sliced_complexity

sliced = slice_code(tree, ixs, sizes, slicer)
comp = sliced_complexity(sliced, ixs, sizes)
print(f"Sliced space: 2^{comp.sc:.2f}")
```

### Optimizer Classes

#### `GreedyMethod`

```python
class GreedyMethod:
    def __init__(
        self,
        max_iter: Optional[int] = None,
        temperature: float = 0.0,
        max_branches: int = 1
    )
```

Greedy optimization algorithm.

**Parameters**:
- `max_iter`: Maximum iterations (default: unlimited)
- `temperature`: Temperature for stochastic variant (0.0 = deterministic)
- `max_branches`: Number of branches to explore (1 = pure greedy)

**Presets**:
```python
# Default greedy
GreedyMethod()

# Stochastic variant
GreedyMethod(temperature=1.0, max_branches=5)
```

#### `TreeSA`

```python
class TreeSA:
    def __init__(
        self,
        ntrials: int = 1,
        niters: int = 50,
        score: Optional[ScoreFunction] = None
    )
    
    @staticmethod
    def fast(score: Optional[ScoreFunction] = None) -> TreeSA
```

Tree-based simulated annealing optimizer.

**Parameters**:
- `ntrials`: Number of independent trials
- `niters`: Iterations per trial
- `score`: Score function for evaluating candidates

**Presets**:
```python
# Fast preset (good balance)
TreeSA.fast()

# Custom configuration
TreeSA(ntrials=10, niters=100, score=ScoreFunction(sc_target=28.0))
```

#### `TreeSASlicer`

```python
class TreeSASlicer:
    def __init__(
        self,
        ntrials: int = 1,
        niters: int = 50,
        fixed_slices: Optional[List[int]] = None,
        score: Optional[ScoreFunction] = None
    )
    
    @staticmethod
    def fast(
        fixed_slices: Optional[List[int]] = None,
        score: Optional[ScoreFunction] = None
    ) -> TreeSASlicer
```

Simulated annealing slicer.

**Parameters**:
- `ntrials`: Number of trials
- `niters`: Iterations per trial
- `fixed_slices`: Indices that must be sliced
- `score`: Score function

**Example**:
```python
# Automatic slice selection
TreeSASlicer.fast(score=ScoreFunction(sc_target=25.0))

# Force slicing on specific indices
TreeSASlicer.fast(fixed_slices=[1, 2])
```

### Configuration Classes

#### `ScoreFunction`

```python
class ScoreFunction:
    def __init__(
        self,
        tc_weight: float = 1.0,
        sc_weight: float = 1.0,
        rw_weight: float = 0.0,
        sc_target: float = 20.0
    )
```

Configure optimization objectives.

**Parameters**:
- `tc_weight`: Weight for time complexity
- `sc_weight`: Weight for space complexity
- `rw_weight`: Weight for read-write complexity (experimental: try 0.1-1.0 for GPU)
- `sc_target`: Target space complexity (log₂ elements)

**Score Formula**:
```
score = tc_weight × 2^tc + rw_weight × 2^rwc + sc_weight × max(0, 2^sc - 2^sc_target)
```

**Examples**:
```python
# CPU optimization
ScoreFunction(tc_weight=1.0, sc_weight=1.0, rw_weight=0.0, sc_target=28.0)

# GPU optimization (experimental: tune rw_weight empirically)
ScoreFunction(tc_weight=1.0, sc_weight=1.0, rw_weight=0.1, sc_target=30.0)

# Memory-constrained
ScoreFunction(tc_weight=0.1, sc_weight=10.0, rw_weight=0.0, sc_target=25.0)
```

### Data Classes

#### `NestedEinsum`

Represents an optimized contraction tree.

**Methods**:
```python
tree.to_dict() -> dict          # Convert to dictionary
tree.leaf_count() -> int        # Number of leaf tensors
tree.depth() -> int             # Tree depth
str(tree)                       # Pretty-print tree
repr(tree)                      # Summary representation
```

**Dictionary Structure**:
```python
# Leaf node
{"tensor_index": int}

# Internal node
{
    "args": [child1_dict, child2_dict, ...],
    "eins": {
        "ixs": [[int, ...], ...],  # Input indices
        "iy": [int, ...]            # Output indices
    }
}
```

**Example**:
```python
tree = optimize_code(ixs, out, sizes)

# Pretty print
print(tree)
# Output:
# ab, bc -> ac
# ├─ ab
# └─ bc

# Convert to dict for execution
tree_dict = tree.to_dict()
```

#### `SlicedCode`

Represents a contraction with slicing applied.

**Methods**:
```python
sliced.slicing() -> List[int]   # Get sliced indices
```

**Example**:
```python
sliced = slice_code(tree, ixs, sizes, slicer)
print(f"Sliced indices: {sliced.slicing()}")
# Output: [1, 3]
```

#### `Complexity`

Complexity metrics for a contraction.

**Fields**:
```python
comp.tc: float   # Time complexity (log₂ FLOPs)
comp.sc: float   # Space complexity (log₂ elements)
comp.rwc: float  # Read-write complexity (log₂ elements)
```

**Example**:
```python
comp = contraction_complexity(tree, ixs, sizes)

# Interpret log₂ values
flops = 2 ** comp.tc
memory_elements = 2 ** comp.sc
io_operations = 2 ** comp.rwc

print(f"{flops:.2e} FLOPs")
print(f"{memory_elements:.2e} elements in memory")
```

## Rust API

Full Rust API documentation: [docs.rs/omeco](https://docs.rs/omeco)

### Core Types

#### `NestedEinsum<I>`

```rust
pub struct NestedEinsum<I> { /* ... */ }

impl<I> NestedEinsum<I> {
    pub fn leaf(tensor_index: usize) -> Self
    pub fn new(args: Vec<Self>, eins: ContractionOperation<I>) -> Self
    pub fn depth(&self) -> usize
    pub fn leaf_count(&self) -> usize
}
```

#### `ContractionOperation<I>`

```rust
pub struct ContractionOperation<I> {
    pub ixs: Vec<Vec<I>>,  // Input indices
    pub iy: Vec<I>,        // Output indices
}
```

### Optimization Functions

#### `optimize_greedy`

```rust
pub fn optimize_greedy<I>(
    ixs: &[Vec<I>],
    out: &[I],
    sizes: &HashMap<I, usize>
) -> NestedEinsum<I>
where
    I: Hash + Eq + Clone + Ord
```

#### `optimize_code`

```rust
pub fn optimize_code<I, O>(
    ixs: &[Vec<I>],
    out: &[I],
    sizes: &HashMap<I, usize>,
    optimizer: O
) -> NestedEinsum<I>
where
    I: Hash + Eq + Clone + Ord,
    O: Optimizer<I>
```

### Optimizer Traits

#### `Optimizer<I>`

```rust
pub trait Optimizer<I> {
    fn optimize(
        &self,
        ixs: &[Vec<I>],
        out: &[I],
        sizes: &HashMap<I, usize>
    ) -> NestedEinsum<I>;
}
```

### Configuration

#### `ScoreFunction`

```rust
pub struct ScoreFunction {
    pub tc_weight: f64,
    pub sc_weight: f64,
    pub rw_weight: f64,
    pub sc_target: f64,
}

impl ScoreFunction {
    pub fn new(tc_weight: f64, sc_weight: f64, rw_weight: f64, sc_target: f64) -> Self
}
```

## Type Mapping (Python ↔ Rust)

| Python | Rust | Notes |
|--------|------|-------|
| `List[List[int]]` | `Vec<Vec<i64>>` | Index lists |
| `Dict[int, int]` | `HashMap<i64, usize>` | Dimension sizes |
| `NestedEinsum` | `NestedEinsum<i64>` | Contraction tree |
| `ScoreFunction` | `ScoreFunction` | Same fields |
| `Complexity` | `ContractionComplexity` | Metrics |

## Common Patterns

### Optimize with custom score

```python
from omeco import optimize_code, TreeSA, ScoreFunction

score = ScoreFunction(
    tc_weight=1.0,
    sc_weight=1.0,
    rw_weight=0.1,   # Experimental for GPU
    sc_target=30.0
)

tree = optimize_code(
    ixs=[[0, 1], [1, 2]],
    out=[0, 2],
    sizes={0: 10, 1: 20, 2: 10},
    optimizer=TreeSA(score=score)
)
```

### Optimize then slice

```python
from omeco import optimize_code, slice_code, TreeSA, TreeSASlicer, ScoreFunction

# Step 1: Optimize contraction order
tree = optimize_code(ixs, out, sizes, TreeSA.fast())

# Step 2: Apply slicing if needed
slicer = TreeSASlicer.fast(score=ScoreFunction(sc_target=28.0))
sliced = slice_code(tree, ixs, sizes, slicer)
```

### Check complexity

```python
from omeco import optimize_code, contraction_complexity

tree = optimize_code(ixs, out, sizes)
comp = contraction_complexity(tree, ixs, sizes)

print(f"Time: 2^{comp.tc:.2f} FLOPs = {2**comp.tc:.2e} FLOPs")
print(f"Space: 2^{comp.sc:.2f} elements = {2**comp.sc:.2e} elements")

# Memory in GB (float64)
memory_gb = (2 ** comp.sc) * 8 / 1e9
print(f"Memory: {memory_gb:.2f} GB")
```

## Next Steps

- [Quick Start Guide](./quick-start.md) - Get started with examples
- [Score Function Configuration](./guides/score-function.md) - Configure optimization
- [Troubleshooting](./troubleshooting.md) - Common issues and solutions
- [Full Rust Docs](https://docs.rs/omeco) - Complete Rust API documentation
