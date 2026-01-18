# Changelog

All notable changes to omeco are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive mdBook documentation following tropical-gemm standard
- Pretty printing for Python `NestedEinsum` with ASCII tree visualization
- PyTorch integration guide and examples
- GPU optimization guide with `rw_weight` configuration
- Slicing strategy guide for memory-constrained environments
- Troubleshooting guide with common issues and solutions
- API reference for Python and Rust APIs
- Performance benchmarks comparing Rust vs Julia
- Algorithm comparison guide (Greedy vs TreeSA)

### Changed
- Migrated documentation from scattered markdown files to structured mdBook
- Improved Python bindings with better `__str__` and `__repr__` methods

### Deprecated
- Legacy `docs/score_function_guide.md` (migrated to mdBook)

## [0.2.1] - 2024-01-XX

### Fixed
- **Issue #6**: Hyperedge index preservation in contraction operations ([PR #7](https://github.com/GiggleLiu/omeco/pull/7))
  - Fixed `contract_tree!` macro to correctly preserve tensor indices during contraction
  - Added regression tests to verify hyperedge handling
  - Ensures contraction order matches input tensor order specified in `ixs`

### Added
- Test suite for hyperedge index preservation
- CI improvements for better test coverage

## [0.2.0] - 2024-01-XX

### Added
- TreeSA (Tree-based Simulated Annealing) optimizer
- `TreeSA.fast()` preset for quick high-quality optimization
- Slicing support with `TreeSASlicer` for memory reduction
- `ScoreFunction` for configurable optimization objectives
- `contraction_complexity` and `sliced_complexity` functions
- Python bindings via PyO3
- `optimize_code` generic function accepting optimizer instances
- Read-write complexity (rwc) metric for GPU optimization

### Changed
- Improved API ergonomics with preset methods
- Better default parameters for optimizers

### Performance
- 1.4-1.5x faster than Julia OMEinsumContractionOrders.jl on benchmarks
- Efficient TreeSA implementation with better exploration

## [0.1.0] - 2023-XX-XX

### Added
- Initial release
- GreedyMethod optimizer
- Basic contraction order optimization
- Support for tensor networks with arbitrary indices
- Complexity calculation (time and space)
- Rust core library
- Basic documentation

### Features
- Greedy algorithm with configurable parameters
- Stochastic variants for improved solutions
- Efficient index handling with generic types
- HashMap-based dimension tracking

## Version Numbering

omeco follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality in a backward compatible manner
- **PATCH** version for backward compatible bug fixes

## Release Process

1. Update version in `Cargo.toml` files
2. Update `CHANGELOG.md` with release notes
3. Tag release: `git tag v0.X.Y`
4. Push tags: `git push --tags`
5. Publish to crates.io: `cargo publish`
6. Publish to PyPI: `maturin publish` (Python bindings)

## Links

- [GitHub Releases](https://github.com/GiggleLiu/omeco/releases)
- [Crates.io](https://crates.io/crates/omeco)
- [PyPI](https://pypi.org/project/omeco/)
- [Documentation](https://gigglueliu.github.io/omeco/)

## Contributing

See [CONTRIBUTING.md](https://github.com/GiggleLiu/omeco/blob/master/CONTRIBUTING.md) for guidelines on contributing to omeco.

## Migration Guides

### Migrating from 0.1.x to 0.2.x

**Major Changes**:
1. **New `optimize_code` function**: Replaces direct optimizer usage
   ```python
   # Old (0.1.x)
   tree = optimize_greedy(ixs, out, sizes)
   
   # New (0.2.x) - still works
   tree = optimize_greedy(ixs, out, sizes)
   
   # New (0.2.x) - recommended for TreeSA
   from omeco import optimize_code, TreeSA
   tree = optimize_code(ixs, out, sizes, TreeSA.fast())
   ```

2. **ScoreFunction configuration**:
   ```python
   # New in 0.2.x
   from omeco import ScoreFunction, TreeSA
   
   score = ScoreFunction(tc_weight=1.0, sc_weight=1.0, rw_weight=0.1, sc_target=30.0)
   tree = optimize_code(ixs, out, sizes, TreeSA(score=score))
   ```

3. **Slicing support**:
   ```python
   # New in 0.2.x
   from omeco import slice_code, TreeSASlicer
   
   sliced = slice_code(tree, ixs, sizes, TreeSASlicer.fast())
   ```

**Breaking Changes**: None - 0.2.x is backward compatible with 0.1.x API

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
| `optimize_greedy` | `optimize_greedy` |
| `optimize_treesa` | `optimize_code(..., TreeSA.fast())` |
| `contraction_complexity` | `contraction_complexity` |
| `slicing` | `slice_code` |

**API Compatibility**: Most functions have similar signatures and behavior.
