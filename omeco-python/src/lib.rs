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
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
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

fn nested_to_dict(py: Python<'_>, nested: &NestedEinsum<i64>) -> PyResult<Py<PyAny>> {
    use pyo3::types::PyDict;

    let dict = PyDict::new(py);
    match nested {
        NestedEinsum::Leaf { tensor_index } => {
            dict.set_item("tensor_index", *tensor_index)?;
        }
        NestedEinsum::Node { args, eins } => {
            let args_list: Vec<Py<PyAny>> = args
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

/// Score function for evaluating contraction quality.
///
/// The score is computed as:
///     score = tc_weight * 2^tc + rw_weight * 2^rw + sc_weight * max(0, 2^sc - 2^sc_target)
///
/// Where:
/// - tc is the time complexity (log2 of FLOP count)
/// - sc is the space complexity (log2 of max intermediate tensor size)
/// - rw is the read-write complexity (log2 of total I/O operations)
#[pyclass(name = "ScoreFunction")]
#[derive(Clone)]
pub struct PyScoreFunction {
    inner: ScoreFunction,
}

#[pymethods]
impl PyScoreFunction {
    /// Create a new ScoreFunction.
    ///
    /// Args:
    ///     tc_weight: Weight for time complexity (default: 1.0).
    ///     sc_weight: Weight for space complexity penalty (default: 1.0).
    ///     rw_weight: Weight for read-write complexity (default: 0.0).
    ///     sc_target: Target space complexity threshold (default: 20.0).
    ///                Space complexity is only penalized if it exceeds this target.
    #[new]
    #[pyo3(signature = (tc_weight=1.0, sc_weight=1.0, rw_weight=0.0, sc_target=20.0))]
    fn new(tc_weight: f64, sc_weight: f64, rw_weight: f64, sc_target: f64) -> Self {
        Self {
            inner: ScoreFunction::new(tc_weight, sc_weight, rw_weight, sc_target),
        }
    }

    /// Get the time complexity weight.
    #[getter]
    fn tc_weight(&self) -> f64 {
        self.inner.tc_weight
    }

    /// Get the space complexity weight.
    #[getter]
    fn sc_weight(&self) -> f64 {
        self.inner.sc_weight
    }

    /// Get the read-write complexity weight.
    #[getter]
    fn rw_weight(&self) -> f64 {
        self.inner.rw_weight
    }

    /// Get the space complexity target.
    #[getter]
    fn sc_target(&self) -> f64 {
        self.inner.sc_target
    }

    fn __repr__(&self) -> String {
        format!(
            "ScoreFunction(tc_weight={}, sc_weight={}, rw_weight={}, sc_target={})",
            self.inner.tc_weight, self.inner.sc_weight, self.inner.rw_weight, self.inner.sc_target
        )
    }
}

/// Greedy optimizer for contraction order.
///
/// Args:
///     alpha: Balance between output size and input size reduction.
///            For pairwise interaction: L = size(out) - alpha * (size(in1) + size(in2)).
///            Default: 0.0.
///     temperature: Boltzmann sampling temperature. If 0.0, the minimum loss is selected;
///                  for non-zero, selection uses Boltzmann distribution p ~ exp(-loss/temperature).
///                  Default: 0.0.
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
    ///     alpha: Balance between output size and input size reduction (default: 0.0).
    ///     temperature: Boltzmann sampling temperature (default: 0.0 = deterministic).
    #[new]
    #[pyo3(signature = (alpha=0.0, temperature=0.0))]
    fn new(alpha: f64, temperature: f64) -> Self {
        Self {
            inner: GreedyMethod::new(alpha, temperature),
        }
    }

    /// Get the alpha parameter.
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    /// Get the temperature parameter.
    #[getter]
    fn temperature(&self) -> f64 {
        self.inner.temperature
    }

    fn __repr__(&self) -> String {
        format!(
            "GreedyMethod(alpha={}, temperature={})",
            self.inner.alpha, self.inner.temperature
        )
    }
}

/// Simulated annealing optimizer for contraction order.
///
/// Args:
///     ntrials: Number of independent trials to run (default: 10).
///     niters: Iterations per temperature level (default: 50).
///     betas: Inverse temperature schedule. If None, uses default schedule (default: None).
///     score: Score function for evaluating solutions (default: ScoreFunction()).
#[pyclass(name = "TreeSA")]
#[derive(Clone)]
pub struct PyTreeSA {
    inner: TreeSA,
}

#[pymethods]
impl PyTreeSA {
    /// Create a new TreeSA optimizer.
    ///
    /// Args:
    ///     ntrials: Number of independent trials to run (default: 10).
    ///     niters: Iterations per temperature level (default: 50).
    ///     betas: Inverse temperature schedule. If None, uses default schedule.
    ///     score: Score function for evaluating solutions. If None, uses default.
    #[new]
    #[pyo3(signature = (ntrials=10, niters=50, betas=None, score=None))]
    fn new(
        ntrials: usize,
        niters: usize,
        betas: Option<Vec<f64>>,
        score: Option<PyScoreFunction>,
    ) -> Self {
        let default_betas: Vec<f64> = (1..=300).map(|i| 0.01 + 0.05 * i as f64).collect();
        let betas = betas.unwrap_or(default_betas);
        let score = score.map(|s| s.inner).unwrap_or_default();

        Self {
            inner: TreeSA {
                betas,
                ntrials,
                niters,
                score,
                ..Default::default()
            },
        }
    }

    /// Create a fast TreeSA configuration (fewer iterations).
    ///
    /// Args:
    ///     score: Score function for evaluating solutions. If None, uses default.
    #[staticmethod]
    #[pyo3(signature = (score=None))]
    fn fast(score: Option<PyScoreFunction>) -> Self {
        let mut inner = TreeSA::fast();
        if let Some(s) = score {
            inner.score = s.inner;
        }
        Self { inner }
    }

    /// Get the number of trials.
    #[getter]
    fn ntrials(&self) -> usize {
        self.inner.ntrials
    }

    /// Get the number of iterations.
    #[getter]
    fn niters(&self) -> usize {
        self.inner.niters
    }

    /// Get the inverse temperature schedule.
    #[getter]
    fn betas(&self) -> Vec<f64> {
        self.inner.betas.clone()
    }

    /// Get the score function.
    #[getter]
    fn score(&self) -> PyScoreFunction {
        PyScoreFunction {
            inner: self.inner.score.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TreeSA(ntrials={}, niters={}, score={})",
            self.inner.ntrials,
            self.inner.niters,
            PyScoreFunction {
                inner: self.inner.score.clone()
            }
            .__repr__()
        )
    }
}

/// Slicing optimizer for reducing space complexity.
///
/// This optimizer iteratively adds slices to reduce memory requirements,
/// trading time complexity for space complexity.
///
/// Args:
///     ntrials: Number of parallel trials (default: 10).
///     niters: Iterations per temperature level (default: 10).
///     betas: Inverse temperature schedule. If None, uses default (14.0 to 15.0).
///     fixed_slices: List of index labels that must be sliced. These indices will always
///                   be included in the slicing and cannot be removed during optimization.
///     optimization_ratio: Ratio for iteration count (default: 2.0).
///     score: Score function for evaluating solutions (default: ScoreFunction(sc_target=30.0)).
#[pyclass(name = "TreeSASlicer")]
#[derive(Clone)]
pub struct PyTreeSASlicer {
    inner: TreeSASlicer,
    /// Fixed slices as user-provided labels (converted to indices in slice_code)
    fixed_slices_labels: Vec<i64>,
}

#[pymethods]
impl PyTreeSASlicer {
    /// Create a new TreeSASlicer optimizer.
    ///
    /// Args:
    ///     ntrials: Number of parallel trials (default: 10).
    ///     niters: Iterations per temperature level (default: 10).
    ///     betas: Inverse temperature schedule. If None, uses default (14.0 to 15.0).
    ///     fixed_slices: List of index labels that must be sliced (default: []).
    ///     optimization_ratio: Ratio for iteration count (default: 2.0).
    ///     score: Score function for evaluating solutions. If None, uses default with sc_target=30.0.
    #[new]
    #[pyo3(signature = (ntrials=10, niters=10, betas=None, fixed_slices=None, optimization_ratio=2.0, score=None))]
    fn new(
        ntrials: usize,
        niters: usize,
        betas: Option<Vec<f64>>,
        fixed_slices: Option<Vec<i64>>,
        optimization_ratio: f64,
        score: Option<PyScoreFunction>,
    ) -> Self {
        let default_betas: Vec<f64> = (0..=20).map(|i| 14.0 + 0.05 * i as f64).collect();
        let betas = betas.unwrap_or(default_betas);
        let fixed_slices_labels = fixed_slices.unwrap_or_default();
        let score = score
            .map(|s| s.inner)
            .unwrap_or_else(|| ScoreFunction::default().with_sc_target(30.0));
        Self {
            inner: TreeSASlicer {
                betas,
                score,
                ntrials,
                niters,
                optimization_ratio,
                ..Default::default()
            },
            fixed_slices_labels,
        }
    }

    /// Create a fast TreeSASlicer configuration (fewer iterations).
    ///
    /// Args:
    ///     fixed_slices: List of index labels that must be sliced (default: []).
    ///     score: Score function for evaluating solutions. If None, uses default.
    #[staticmethod]
    #[pyo3(signature = (fixed_slices=None, score=None))]
    fn fast(fixed_slices: Option<Vec<i64>>, score: Option<PyScoreFunction>) -> Self {
        let mut inner = TreeSASlicer::fast();
        if let Some(s) = score {
            inner.score = s.inner;
        }
        Self {
            inner,
            fixed_slices_labels: fixed_slices.unwrap_or_default(),
        }
    }

    /// Get the number of trials.
    #[getter]
    fn ntrials(&self) -> usize {
        self.inner.ntrials
    }

    /// Get the number of iterations.
    #[getter]
    fn niters(&self) -> usize {
        self.inner.niters
    }

    /// Get the inverse temperature schedule.
    #[getter]
    fn betas(&self) -> Vec<f64> {
        self.inner.betas.clone()
    }

    /// Get the fixed slices (index labels).
    #[getter]
    fn fixed_slices(&self) -> Vec<i64> {
        self.fixed_slices_labels.clone()
    }

    /// Get the optimization ratio.
    #[getter]
    fn optimization_ratio(&self) -> f64 {
        self.inner.optimization_ratio
    }

    /// Get the score function.
    #[getter]
    fn score(&self) -> PyScoreFunction {
        PyScoreFunction {
            inner: self.inner.score.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TreeSASlicer(ntrials={}, niters={}, fixed_slices={:?}, score={})",
            self.inner.ntrials,
            self.inner.niters,
            self.fixed_slices_labels,
            PyScoreFunction {
                inner: self.inner.score.clone()
            }
            .__repr__()
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
    let opt = optimizer.unwrap_or_else(|| PyTreeSA::new(10, 50, None, None));

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
///     >>> from omeco import optimize_code, slice_code, TreeSASlicer, ScoreFunction, GreedyMethod
///     >>> ixs = [[0, 1], [1, 2], [2, 3]]
///     >>> out = [0, 3]
///     >>> sizes = {0: 100, 1: 50, 2: 80, 3: 100}
///     >>> tree = optimize_code(ixs, out, sizes, GreedyMethod())
///     >>> sliced = slice_code(tree, ixs, sizes, TreeSASlicer(score=ScoreFunction(sc_target=10.0)))
///     >>> print(sliced.slicing())  # Indices that will be looped over
///
///     # With fixed slices:
///     >>> sliced = slice_code(tree, ixs, sizes, TreeSASlicer(fixed_slices=[1, 2]))
#[pyfunction]
#[pyo3(signature = (tree, ixs, sizes, slicer=None))]
fn slice_code(
    tree: &PyNestedEinsum,
    ixs: Vec<Vec<i64>>,
    sizes: HashMap<i64, usize>,
    slicer: Option<PyTreeSASlicer>,
) -> PyResult<PySlicedEinsum> {
    let config = slicer.unwrap_or_else(|| PyTreeSASlicer::new(10, 10, None, None, 2.0, None));

    // Build label map from ixs to convert fixed_slices labels to indices
    let mut all_labels: Vec<i64> = Vec::new();
    for ix in &ixs {
        for &l in ix {
            if !all_labels.contains(&l) {
                all_labels.push(l);
            }
        }
    }
    let label_map: HashMap<i64, usize> = all_labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i))
        .collect();

    // Convert fixed_slices labels to indices
    let fixed_slices_indices: Vec<usize> = config
        .fixed_slices_labels
        .iter()
        .filter_map(|&label| label_map.get(&label).copied())
        .collect();

    // Create a modified config with the converted fixed_slices
    let mut inner_config = config.inner.clone();
    inner_config.fixed_slices = fixed_slices_indices;

    omeco::slice_code(&tree.inner, &sizes, &inner_config, &ixs)
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
    m.add_class::<PyScoreFunction>()?;
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
