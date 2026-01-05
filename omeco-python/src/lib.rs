//! Python bindings for omeco tensor network contraction order optimization.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

use omeco::{
    CodeOptimizer, ContractionComplexity, EinCode, GreedyMethod, NestedEinsum, ScoreFunction,
    SlicedEinsum, TreeSA, TreeSASlicer,
};

/// A contraction order represented as a nested einsum tree.
#[pyclass(name = "NestedEinsum")]
#[derive(Clone)]
pub struct PyNestedEinsum {
    inner: NestedEinsum<i64>,
}

#[pymethods]
impl PyNestedEinsum {
    /// Check if this is a leaf node (single tensor).
    fn is_leaf(&self) -> bool {
        self.inner.is_leaf()
    }

    /// Check if the tree is binary.
    fn is_binary(&self) -> bool {
        self.inner.is_binary()
    }

    /// Count the number of leaf nodes (input tensors).
    fn leaf_count(&self) -> usize {
        self.inner.leaf_count()
    }

    /// Get the depth of the tree.
    fn depth(&self) -> usize {
        self.inner.depth()
    }

    /// Get leaf indices in order.
    fn leaf_indices(&self) -> Vec<usize> {
        self.inner.leaf_indices()
    }

    /// Convert to a Python dictionary for traversal.
    ///
    /// Returns a dict with structure:
    /// - For leaf: {"tensor_index": int}
    /// - For node: {"args": [child_dicts], "eins": {"ixs": [[int]], "iy": [int]}}
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        nested_to_dict(py, &self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "NestedEinsum(leaves={}, depth={})",
            self.leaf_count(),
            self.depth()
        )
    }
}

fn nested_to_dict(py: Python<'_>, nested: &NestedEinsum<i64>) -> PyResult<PyObject> {
    use pyo3::types::PyDict;

    let dict = PyDict::new(py);
    match nested {
        NestedEinsum::Leaf { tensor_index } => {
            dict.set_item("tensor_index", *tensor_index)?;
        }
        NestedEinsum::Node { args, eins } => {
            let args_list: Vec<PyObject> = args
                .iter()
                .map(|arg| nested_to_dict(py, arg))
                .collect::<PyResult<_>>()?;
            dict.set_item("args", args_list)?;

            let eins_dict = PyDict::new(py);
            eins_dict.set_item("ixs", &eins.ixs)?;
            eins_dict.set_item("iy", &eins.iy)?;
            dict.set_item("eins", eins_dict)?;
        }
    }
    Ok(dict.into())
}

/// A sliced einsum with indices to loop over.
#[pyclass(name = "SlicedEinsum")]
#[derive(Clone)]
pub struct PySlicedEinsum {
    inner: SlicedEinsum<i64>,
}

#[pymethods]
impl PySlicedEinsum {
    /// Create a new sliced einsum.
    #[new]
    fn new(slicing: Vec<i64>, tree: PyNestedEinsum) -> Self {
        Self {
            inner: SlicedEinsum::new(slicing, tree.inner),
        }
    }

    /// Get the sliced indices.
    fn slicing(&self) -> Vec<i64> {
        self.inner.slicing.clone()
    }

    /// Get the number of sliced indices.
    fn num_slices(&self) -> usize {
        self.inner.num_slices()
    }

    fn __repr__(&self) -> String {
        format!("SlicedEinsum(slicing={:?})", self.inner.slicing)
    }
}

/// Complexity metrics for a contraction.
#[pyclass(name = "ContractionComplexity")]
#[derive(Clone)]
pub struct PyContractionComplexity {
    /// Time complexity (log2 of FLOPs).
    #[pyo3(get)]
    pub tc: f64,
    /// Space complexity (log2 of max intermediate size).
    #[pyo3(get)]
    pub sc: f64,
    /// Read-write complexity (log2 of total I/O).
    #[pyo3(get)]
    pub rwc: f64,
}

#[pymethods]
impl PyContractionComplexity {
    /// Get the total FLOPs.
    fn flops(&self) -> f64 {
        2.0_f64.powf(self.tc)
    }

    /// Get the peak memory in number of elements.
    fn peak_memory(&self) -> f64 {
        2.0_f64.powf(self.sc)
    }

    fn __repr__(&self) -> String {
        format!(
            "ContractionComplexity(tc={:.2}, sc={:.2}, rwc={:.2})",
            self.tc, self.sc, self.rwc
        )
    }
}

impl From<ContractionComplexity> for PyContractionComplexity {
    fn from(c: ContractionComplexity) -> Self {
        Self {
            tc: c.tc,
            sc: c.sc,
            rwc: c.rwc,
        }
    }
}

/// Greedy optimizer for contraction order.
#[pyclass(name = "GreedyMethod")]
#[derive(Clone)]
pub struct PyGreedyMethod {
    inner: GreedyMethod,
}

#[pymethods]
impl PyGreedyMethod {
    /// Create a new greedy optimizer.
    ///
    /// Args:
    ///     alpha: Balance between output size and input size reduction (0.0-1.0).
    ///     temperature: Boltzmann sampling temperature (0.0 = deterministic).
    #[new]
    #[pyo3(signature = (alpha=0.0, temperature=0.0))]
    fn new(alpha: f64, temperature: f64) -> Self {
        Self {
            inner: GreedyMethod::new(alpha, temperature),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GreedyMethod(alpha={}, temperature={})",
            self.inner.alpha, self.inner.temperature
        )
    }
}

/// Simulated annealing optimizer for contraction order.
#[pyclass(name = "TreeSA")]
#[derive(Clone)]
pub struct PyTreeSA {
    inner: TreeSA,
}

#[pymethods]
impl PyTreeSA {
    /// Create a new TreeSA optimizer with default settings.
    #[new]
    fn new() -> Self {
        Self {
            inner: TreeSA::default(),
        }
    }

    /// Create a fast TreeSA configuration (fewer iterations).
    #[staticmethod]
    fn fast() -> Self {
        Self {
            inner: TreeSA::fast(),
        }
    }

    /// Set the space complexity target.
    fn with_sc_target(&self, sc_target: f64) -> Self {
        Self {
            inner: self.inner.clone().with_sc_target(sc_target),
        }
    }

    /// Set the number of parallel trials.
    fn with_ntrials(&self, ntrials: usize) -> Self {
        Self {
            inner: self.inner.clone().with_ntrials(ntrials),
        }
    }

    /// Set the number of iterations per temperature level.
    fn with_niters(&self, niters: usize) -> Self {
        Self {
            inner: self.inner.clone().with_niters(niters),
        }
    }

    /// Set the inverse temperature schedule (betas).
    fn with_betas(&self, betas: Vec<f64>) -> Self {
        Self {
            inner: self.inner.clone().with_betas(betas),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TreeSA(ntrials={}, niters={})",
            self.inner.ntrials, self.inner.niters
        )
    }
}

/// Slicing optimizer for reducing space complexity.
///
/// This optimizer iteratively adds slices to reduce memory requirements,
/// trading time complexity for space complexity.
#[pyclass(name = "TreeSASlicer")]
#[derive(Clone)]
pub struct PyTreeSASlicer {
    inner: TreeSASlicer,
}

#[pymethods]
impl PyTreeSASlicer {
    /// Create a new TreeSASlicer optimizer.
    ///
    /// Args:
    ///     sc_target: Target space complexity (log2 scale). Default: 30.0.
    ///     ntrials: Number of parallel trials. Default: 10.
    ///     niters: Iterations per temperature level. Default: 10.
    ///     optimization_ratio: Ratio for iteration count. Default: 2.0.
    #[new]
    #[pyo3(signature = (sc_target=30.0, ntrials=10, niters=10, optimization_ratio=2.0))]
    fn new(sc_target: f64, ntrials: usize, niters: usize, optimization_ratio: f64) -> Self {
        let score = ScoreFunction::default().with_sc_target(sc_target);
        Self {
            inner: TreeSASlicer {
                score,
                ntrials,
                niters,
                optimization_ratio,
                ..Default::default()
            },
        }
    }

    /// Create a fast TreeSASlicer configuration (fewer iterations).
    #[staticmethod]
    fn fast() -> Self {
        Self {
            inner: TreeSASlicer::fast(),
        }
    }

    /// Set the space complexity target.
    fn with_sc_target(&self, sc_target: f64) -> Self {
        Self {
            inner: self.inner.clone().with_sc_target(sc_target),
        }
    }

    /// Set the number of parallel trials.
    fn with_ntrials(&self, ntrials: usize) -> Self {
        Self {
            inner: self.inner.clone().with_ntrials(ntrials),
        }
    }

    /// Set the number of iterations per temperature level.
    fn with_niters(&self, niters: usize) -> Self {
        Self {
            inner: self.inner.clone().with_niters(niters),
        }
    }

    /// Set the optimization ratio.
    fn with_optimization_ratio(&self, ratio: f64) -> Self {
        Self {
            inner: self.inner.clone().with_optimization_ratio(ratio),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TreeSASlicer(sc_target={:.1}, ntrials={}, niters={})",
            self.inner.score.sc_target, self.inner.ntrials, self.inner.niters
        )
    }
}

/// Optimize the contraction order using greedy method.
///
/// Args:
///     ixs: List of index lists for each tensor (e.g., [[0, 1], [1, 2]]).
///     out: Output indices (e.g., [0, 2]).
///     sizes: Dictionary mapping indices to their dimensions.
///     optimizer: Optimizer to use (GreedyMethod or TreeSA).
///
/// Returns:
///     Optimized contraction tree as NestedEinsum.
#[pyfunction]
#[pyo3(signature = (ixs, out, sizes, optimizer=None))]
fn optimize_greedy(
    ixs: Vec<Vec<i64>>,
    out: Vec<i64>,
    sizes: HashMap<i64, usize>,
    optimizer: Option<PyGreedyMethod>,
) -> PyResult<PyNestedEinsum> {
    let code = EinCode::new(ixs, out);
    let opt = optimizer.unwrap_or_else(|| PyGreedyMethod::new(0.0, 0.0));

    opt.inner
        .optimize(&code, &sizes)
        .map(|inner| PyNestedEinsum { inner })
        .ok_or_else(|| PyValueError::new_err("Optimization failed"))
}

/// Optimize the contraction order using simulated annealing.
///
/// Args:
///     ixs: List of index lists for each tensor.
///     out: Output indices.
///     sizes: Dictionary mapping indices to their dimensions.
///     optimizer: TreeSA optimizer configuration.
///
/// Returns:
///     Optimized contraction tree as NestedEinsum.
#[pyfunction]
#[pyo3(signature = (ixs, out, sizes, optimizer=None))]
fn optimize_treesa(
    ixs: Vec<Vec<i64>>,
    out: Vec<i64>,
    sizes: HashMap<i64, usize>,
    optimizer: Option<PyTreeSA>,
) -> PyResult<PyNestedEinsum> {
    let code = EinCode::new(ixs, out);
    let opt = optimizer.unwrap_or_else(PyTreeSA::new);

    opt.inner
        .optimize(&code, &sizes)
        .map(|inner| PyNestedEinsum { inner })
        .ok_or_else(|| PyValueError::new_err("Optimization failed"))
}

/// Compute the contraction complexity of an optimized tree.
///
/// Args:
///     tree: Optimized contraction tree.
///     ixs: Original index lists for each tensor.
///     sizes: Dictionary mapping indices to their dimensions.
///
/// Returns:
///     ContractionComplexity with tc, sc, and rwc metrics.
#[pyfunction]
fn contraction_complexity(
    tree: &PyNestedEinsum,
    ixs: Vec<Vec<i64>>,
    sizes: HashMap<i64, usize>,
) -> PyContractionComplexity {
    omeco::contraction_complexity(&tree.inner, &sizes, &ixs).into()
}

/// Compute the complexity of a sliced contraction.
///
/// Args:
///     sliced: Sliced einsum.
///     ixs: Original index lists for each tensor.
///     sizes: Dictionary mapping indices to their dimensions.
///
/// Returns:
///     ContractionComplexity with adjusted metrics.
#[pyfunction]
fn sliced_complexity(
    sliced: &PySlicedEinsum,
    ixs: Vec<Vec<i64>>,
    sizes: HashMap<i64, usize>,
) -> PyContractionComplexity {
    omeco::sliced_complexity(&sliced.inner, &sizes, &ixs).into()
}

/// Create a size dictionary with uniform dimensions.
///
/// Args:
///     ixs: List of index lists for each tensor.
///     out: Output indices.
///     size: Dimension for all indices.
///
/// Returns:
///     Dictionary mapping each index to the given size.
#[pyfunction]
fn uniform_size_dict(ixs: Vec<Vec<i64>>, out: Vec<i64>, size: usize) -> HashMap<i64, usize> {
    let code = EinCode::new(ixs, out);
    omeco::uniform_size_dict(&code, size)
}

/// Slice a contraction tree to reduce space complexity.
///
/// This function takes an already-optimized contraction tree and finds indices
/// to slice over, reducing memory requirements at the cost of additional computation.
///
/// Args:
///     tree: Optimized contraction tree (from optimize_code or optimize_treesa).
///     ixs: Original index lists for each tensor.
///     sizes: Dictionary mapping indices to their dimensions.
///     slicer: Slicing optimizer configuration. Defaults to TreeSASlicer().
///
/// Returns:
///     SlicedEinsum with the sliced indices and optimized tree.
///
/// Example:
///     >>> from omeco import optimize_code, slice_code, TreeSASlicer, GreedyMethod
///     >>> ixs = [[0, 1], [1, 2], [2, 3]]
///     >>> out = [0, 3]
///     >>> sizes = {0: 100, 1: 50, 2: 80, 3: 100}
///     >>> tree = optimize_code(ixs, out, sizes, GreedyMethod())
///     >>> sliced = slice_code(tree, ixs, sizes, TreeSASlicer(sc_target=10.0))
///     >>> print(sliced.slicing())  # Indices that will be looped over
#[pyfunction]
#[pyo3(signature = (tree, ixs, sizes, slicer=None))]
fn slice_code(
    tree: &PyNestedEinsum,
    ixs: Vec<Vec<i64>>,
    sizes: HashMap<i64, usize>,
    slicer: Option<PyTreeSASlicer>,
) -> PyResult<PySlicedEinsum> {
    let config = slicer.unwrap_or_else(|| PyTreeSASlicer::new(30.0, 10, 10, 2.0));

    omeco::slice_code(&tree.inner, &sizes, &config.inner, &ixs)
        .map(|inner| PySlicedEinsum { inner })
        .ok_or_else(|| PyValueError::new_err("Slicing failed"))
}

/// Unified optimizer type that can be either GreedyMethod or TreeSA.
#[derive(FromPyObject)]
enum PyOptimizer {
    Greedy(PyGreedyMethod),
    TreeSA(PyTreeSA),
}

/// Optimize the contraction order using the specified optimizer.
///
/// This is the unified interface for contraction order optimization.
///
/// Args:
///     ixs: List of index lists for each tensor (e.g., [[0, 1], [1, 2]]).
///     out: Output indices (e.g., [0, 2]).
///     sizes: Dictionary mapping indices to their dimensions.
///     optimizer: Optimizer to use (GreedyMethod or TreeSA). Defaults to GreedyMethod().
///
/// Returns:
///     Optimized contraction tree as NestedEinsum.
///
/// Example:
///     >>> from omeco import optimize_code, GreedyMethod, TreeSA
///     >>> ixs = [[0, 1], [1, 2], [2, 3]]
///     >>> out = [0, 3]
///     >>> sizes = {0: 100, 1: 50, 2: 80, 3: 100}
///     >>> tree = optimize_code(ixs, out, sizes, GreedyMethod())
///     >>> tree = optimize_code(ixs, out, sizes, TreeSA.fast())
#[pyfunction]
#[pyo3(signature = (ixs, out, sizes, optimizer=None))]
fn optimize_code(
    ixs: Vec<Vec<i64>>,
    out: Vec<i64>,
    sizes: HashMap<i64, usize>,
    optimizer: Option<PyOptimizer>,
) -> PyResult<PyNestedEinsum> {
    let code = EinCode::new(ixs, out);

    let result = match optimizer {
        Some(PyOptimizer::Greedy(opt)) => opt.inner.optimize(&code, &sizes),
        Some(PyOptimizer::TreeSA(opt)) => opt.inner.optimize(&code, &sizes),
        None => GreedyMethod::default().optimize(&code, &sizes),
    };

    result
        .map(|inner| PyNestedEinsum { inner })
        .ok_or_else(|| PyValueError::new_err("Optimization failed"))
}

/// Python module for omeco tensor network contraction order optimization.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNestedEinsum>()?;
    m.add_class::<PySlicedEinsum>()?;
    m.add_class::<PyContractionComplexity>()?;
    m.add_class::<PyGreedyMethod>()?;
    m.add_class::<PyTreeSA>()?;
    m.add_class::<PyTreeSASlicer>()?;
    m.add_function(wrap_pyfunction!(optimize_code, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_greedy, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_treesa, m)?)?;
    m.add_function(wrap_pyfunction!(contraction_complexity, m)?)?;
    m.add_function(wrap_pyfunction!(sliced_complexity, m)?)?;
    m.add_function(wrap_pyfunction!(uniform_size_dict, m)?)?;
    m.add_function(wrap_pyfunction!(slice_code, m)?)?;
    Ok(())
}
