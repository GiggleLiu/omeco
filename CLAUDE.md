# OMECO Project Guidelines

## Project Overview
- **OMECO** (One More Einsum Contraction Order) - Rust library for tensor network contraction order optimization
- Rust core with Python bindings via PyO3
- Current Version: 0.2.3

## Development Setup

### Rust
- Edition 2021, MSRV: 1.70
- All Clippy warnings treated as errors (`-D warnings`)

### Python
- Use `uv` for virtual environment management
- Python 3.8-3.14 supported
- Bindings built with PyO3 and Maturin

## Make Commands

### Development (use these regularly)
```bash
make check-all          # Run fmt-check + clippy + test (run before commits)
make test               # Run Rust tests
make build              # Build workspace in debug mode
make build-release      # Build workspace in release mode
make fmt                # Format all code
make fmt-check          # Check formatting without modifying
make clippy             # Run clippy linter (denies warnings)
make clean              # Clean build artifacts
```

### Python Development
```bash
make python-dev         # Build and install Python package locally (for testing)
make python-build       # Build Python wheel for distribution
make python-test        # Run Python tests with pytest
```

### Documentation
```bash
make doc                # Build rustdoc and open in browser
make doc-private        # Build rustdoc including private items
make serve-docs         # Serve rustdoc at http://127.0.0.1:8000/omeco
make install-mdbook     # Install mdBook if not present
make build-book         # Build mdBook documentation
make serve-book         # Serve mdBook at http://127.0.0.1:3000
make clean-book         # Remove generated mdBook files
```

### Benchmarks
```bash
make benchmark          # Run Python vs Julia benchmark
```

### Version Management
```bash
make version            # Show current version
make bump-patch         # Bump patch version (e.g., 0.2.3 -> 0.2.4)
make bump-minor         # Bump minor version (e.g., 0.2.3 -> 0.3.0)
make bump-major         # Bump major version (e.g., 0.2.3 -> 1.0.0)
```

### Release (use in order)
```bash
make bump-patch         # 1. Bump version and commit
make release            # 2. Create git tag and push to origin
make publish-crates     # 3a. Publish Rust crate to crates.io
make publish            # 3b. Publish Python package to PyPI
make publish-all        # 3c. Or publish to both at once
make github-release     # 4. Create GitHub release with auto-generated notes
```

## Code Conventions

### Rust
- No panics/unwraps in production code (only in tests)
- Use `thiserror` for error types
- All public items must have doc comments with examples
- Generic `Label` trait allows `char` or `usize` indices

### Python Bindings
- Use `#[pyclass]` and `#[pymethods]` for PyO3
- Python bindings use `i64` for all indices

### Testing
- Coverage must exceed 95%
- Rust: inline `#[test]` modules in source files
- Python: tests in `omeco-python/tests/`
- Use `criterion` for benchmarks

## Project Structure

```
omeco/              # Core Rust library
omeco-python/       # PyO3 Python bindings
docs/               # mdBook documentation
benchmarks/         # Performance benchmarks
examples/           # Usage examples
```

## Key Files

- `omeco/Cargo.toml` - Main library dependencies
- `omeco-python/pyproject.toml` - Python package config
- `Makefile` - Build automation (primary interface)
- `.github/workflows/` - CI/CD pipelines

## Architecture

- **Optimizers:** GreedyMethod (fast), TreeSA (quality), TreeSASlicer (memory reduction)
- **Complexity Metrics:** Time (tc), Space (sc), Read-write (rwc) - all in log2 scale
- **Parallelism:** TreeSA uses Rayon; control via `RAYON_NUM_THREADS`

## CI Requirements

- Rust tests pass on Windows/Mac/Linux
- Python tests pass
- Coverage >95%
- Clippy with no warnings
- Proper formatting
