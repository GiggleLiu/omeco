//! Python bindings for omeco tensor network contraction order optimization.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;

use omeco::{
    CodeOptimizer, ContractionComplexity, EinCode, GreedyMethod, NestedEinsum, SlicedEinsum,
    TreeSA,
};

/// A contraction order represented as a nested einsum tree.
#[pyclass(name = "NestedEinsum")]
#[derive(Clone)]
pub struct PyNestedEinsum {
    inner: NestedEinsum<char>,
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

    fn __repr__(&self) -> String {
        format!("NestedEinsum(leaves={}, depth={})", self.leaf_count(), self.depth())
    }
}

/// A sliced einsum with indices to loop over.
#[pyclass(name = "SlicedEinsum")]
#[derive(Clone)]
pub struct PySlicedEinsum {
    inner: SlicedEinsum<char>,
}

#[pymethods]
impl PySlicedEinsum {
    /// Create a new sliced einsum.
    #[new]
    fn new(slicing: Vec<char>, tree: PyNestedEinsum) -> Self {
        Self {
            inner: SlicedEinsum::new(slicing, tree.inner),
        }
    }

    /// Get the sliced indices.
    fn slicing(&self) -> Vec<char> {
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

/// Optimize the contraction order using greedy method.
///
/// Args:
///     ixs: List of index lists for each tensor (e.g., [['i', 'j'], ['j', 'k']]).
///     out: Output indices (e.g., ['i', 'k']).
///     sizes: Dictionary mapping indices to their dimensions.
///     optimizer: Optimizer to use (GreedyMethod or TreeSA).
///
/// Returns:
///     Optimized contraction tree as NestedEinsum.
#[pyfunction]
#[pyo3(signature = (ixs, out, sizes, optimizer=None))]
fn optimize_greedy(
    ixs: Vec<Vec<char>>,
    out: Vec<char>,
    sizes: HashMap<char, usize>,
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
    ixs: Vec<Vec<char>>,
    out: Vec<char>,
    sizes: HashMap<char, usize>,
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
    ixs: Vec<Vec<char>>,
    sizes: HashMap<char, usize>,
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
    ixs: Vec<Vec<char>>,
    sizes: HashMap<char, usize>,
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
fn uniform_size_dict(ixs: Vec<Vec<char>>, out: Vec<char>, size: usize) -> HashMap<char, usize> {
    let code = EinCode::new(ixs, out);
    omeco::uniform_size_dict(&code, size)
}

/// Python module for omeco tensor network contraction order optimization.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNestedEinsum>()?;
    m.add_class::<PySlicedEinsum>()?;
    m.add_class::<PyContractionComplexity>()?;
    m.add_class::<PyGreedyMethod>()?;
    m.add_class::<PyTreeSA>()?;
    m.add_function(wrap_pyfunction!(optimize_greedy, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_treesa, m)?)?;
    m.add_function(wrap_pyfunction!(contraction_complexity, m)?)?;
    m.add_function(wrap_pyfunction!(sliced_complexity, m)?)?;
    m.add_function(wrap_pyfunction!(uniform_size_dict, m)?)?;
    Ok(())
}

