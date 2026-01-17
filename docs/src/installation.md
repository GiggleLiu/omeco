# Installation

## Python

### Via pip

The easiest way to install omeco for Python:

```bash
pip install omeco
```

### Via conda (not yet available)

```bash
# Coming soon
conda install -c conda-forge omeco
```

### From source

For development or the latest features:

```bash
git clone https://github.com/GiggleLiu/omeco.git
cd omeco/omeco-python
pip install maturin
maturin develop
```

### Verify Installation

```python
import omeco
print(omeco.__version__)

# Quick test
from omeco import optimize_greedy
tree = optimize_greedy([[0, 1], [1, 2]], [0, 2], {0: 2, 1: 3, 2: 2})
print(tree)
```

Expected output:
```
ab, bc -> ac
├─ tensor_0
└─ tensor_1
```

## Rust

### Via Cargo

Add to your `Cargo.toml`:

```toml
[dependencies]
omeco = "0.2"
```

Or use `cargo add`:

```bash
cargo add omeco
```

### From source

```bash
git clone https://github.com/GiggleLiu/omeco.git
cd omeco
cargo build --release
```

### Verify Installation

Create `src/main.rs`:

```rust
use omeco::{EinCode, GreedyMethod, optimize_code};
use std::collections::HashMap;

fn main() {
    let code = EinCode::new(vec![vec![0, 1], vec![1, 2]], vec![0, 2]);
    let sizes = HashMap::from([(0, 2), (1, 3), (2, 2)]);
    
    let tree = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
    println!("{} leaves, depth {}", tree.leaf_count(), tree.depth());
}
```

Run:
```bash
cargo run
```

## Troubleshooting

### Python: "No module named omeco"

- Ensure you're using the correct Python environment
- Try `pip install --force-reinstall omeco`
- Check `pip list | grep omeco`

### Rust: Compilation errors

- Update Rust: `rustup update`
- Check minimum version: `rustc --version` (need 1.70+)
- Clean and rebuild: `cargo clean && cargo build`

### Installation takes too long

- Python: Pre-built wheels are available for common platforms
- Rust: First build compiles dependencies, subsequent builds are fast

## Next Steps

- [Quick Start](./quick-start.md) - Your first optimization
- [Concepts](./concepts/README.md) - Understand tensor networks
