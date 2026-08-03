<!-- The release history is owned by the repository-root CHANGELOG.md and
     included here verbatim; edit that file, not this page. -->
{{#include ../../CHANGELOG.md}}

## Migration Guides

### Migrating from 0.1.x to 0.2.x

**Major Changes**:
1. **New `optimize_code` function**: Unified API for all optimizers
   ```python
   # Old (0.1.x)
   tree = optimize_greedy(ixs, out, sizes)
   tree = optimize_treesa(ixs, out, sizes)

   # New (0.2.x) - unified interface
   from omeco import optimize_code, GreedyMethod, TreeSA
   tree = optimize_code(ixs, out, sizes)  # Uses GreedyMethod by default
   tree = optimize_code(ixs, out, sizes, TreeSA.fast())
   ```

2. **ScoreFunction configuration**:
   ```python
   # New in 0.2.x
   from omeco import ScoreFunction, TreeSA

   score = ScoreFunction(tc_weight=1.0, sc_weight=1.0, rw_weight=10.0, sc_target=30.0)
   tree = optimize_code(ixs, out, sizes, TreeSA(score=score))
   ```

3. **Slicing support**:
   ```python
   # New in 0.2.x
   from omeco import slice_code, TreeSASlicer

   sliced = slice_code(tree, ixs, sizes, TreeSASlicer.fast())
   ```

**Breaking Changes**:
- Removed `optimize_greedy()` and `optimize_treesa()` from Python exports
- Use `optimize_code(...)` instead with `GreedyMethod()` or `TreeSA()` optimizers
- Rust API unchanged

### Migrating from Julia OMEinsumContractionOrders.jl

**Index Differences**:
- Julia: 1-based indexing
- Rust/Python: 0-based indexing (or use arbitrary hashable types)

```julia
# Julia
ixs = [[1, 2], [2, 3], [3, 1]]
sizes = Dict(1 => 10, 2 => 20, 3 => 10)
```

```python
# Python (0-based)
ixs = [[0, 1], [1, 2], [2, 0]]
sizes = {0: 10, 1: 20, 2: 10}
```

**Function Names**:
| Julia | omeco (Python/Rust) |
|-------|---------------------|
| `optimize_greedy` | `optimize_code(..., GreedyMethod())` |
| `optimize_treesa` | `optimize_code(..., TreeSA.fast())` |
| `contraction_complexity` | `contraction_complexity` |
| `slicing` | `slice_code` |

**API Compatibility**: Most functions have similar signatures and behavior.
