//! Python bindings for omeco tensor network contraction order optimization.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

use omeco::{
    CodeOptimizer, ContractionComplexity, EinCode, ExhaustiveSearch, GreedyMethod, NestedEinsum,
    ScoreFunction, SlicedEinsum, TreeSA, TreeSASlicer, Treewidth,
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

    fn __str__(&self) -> String {
        format_nested_tree(&self.inner)
    }

    /// Compute the contraction complexity of this einsum code.
    ///
    /// Args:
    ///     ixs: List of input tensor index lists.
    ///     sizes: Dictionary mapping indices to their dimensions.
    ///
    /// Returns:
    ///     ContractionComplexity with tc, sc, and rwc metrics.
    ///
    /// Example:
    ///     ```python
    ///     tree = optimize_code(ixs, out, sizes)
    ///     comp = tree.complexity(ixs, sizes)
    ///     print(f"Time: 2^{comp.tc:.2f}, Space: 2^{comp.sc:.2f}")
    ///     ```
    fn complexity(
        &self,
        ixs: Vec<Vec<i64>>,
        sizes: HashMap<i64, usize>,
    ) -> PyContractionComplexity {
        omeco::contraction_complexity(&self.inner, &sizes, &ixs).into()
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

/// Format a NestedEinsum as an ASCII tree with box-drawing characters.
fn format_nested_tree(nested: &NestedEinsum<i64>) -> String {
    let mut output = String::new();
    format_tree_recursive(nested, &mut output, "", true, true);
    // Remove trailing newline
    if output.ends_with('\n') {
        output.pop();
    }
    output
}

/// Recursively format the tree with proper indentation and box-drawing.
fn format_tree_recursive(
    nested: &NestedEinsum<i64>,
    output: &mut String,
    prefix: &str,
    is_last: bool,
    is_root: bool,
) {
    match nested {
        NestedEinsum::Leaf { tensor_index } => {
            if !is_root {
                output.push_str(prefix);
                output.push_str(if is_last { "└─ " } else { "├─ " });
            }
            output.push_str(&format!("tensor_{}\n", tensor_index));
        }
        NestedEinsum::Node { args, eins } => {
            // Format the einsum operation
            let ixs_str: Vec<String> = eins.ixs.iter().map(|ix| indices_to_letters(ix)).collect();
            let iy_str = indices_to_letters(&eins.iy);
            let einsum_notation = if iy_str.is_empty() {
                ixs_str.join(", ")
            } else {
                format!("{} -> {}", ixs_str.join(", "), iy_str)
            };

            if !is_root {
                output.push_str(prefix);
                output.push_str(if is_last { "└─ " } else { "├─ " });
            }
            output.push_str(&einsum_notation);
            output.push('\n');

            // Format children
            let child_count = args.len();
            for (i, child) in args.iter().enumerate() {
                let child_is_last = i == child_count - 1;
                let child_prefix = if is_root {
                    String::new()
                } else {
                    format!("{}{}", prefix, if is_last { "   " } else { "│  " })
                };

                format_tree_recursive(child, output, &child_prefix, child_is_last, false);
            }
        }
    }
}

/// Convert integer indices to letter representation (1→a, 2→b, ..., 26→z, 27→aa, ...).
fn indices_to_letters(indices: &[i64]) -> String {
    indices
        .iter()
        .map(|&idx| index_to_letter(idx))
        .collect::<Vec<_>>()
        .join("")
}

/// Convert a single index to letter(s).
fn index_to_letter(idx: i64) -> String {
    if idx <= 0 {
        return format!("_{}", idx);
    }

    let mut result = String::new();
    let mut n = idx;

    while n > 0 {
        let remainder = ((n - 1) % 26) as u8;
        result.insert(0, (b'a' + remainder) as char);
        n = (n - 1) / 26;
    }

    result
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

    /// Get the optimized contraction tree executed for each slice.
    fn tree(&self) -> PyNestedEinsum {
        PyNestedEinsum {
            inner: self.inner.eins.clone(),
        }
    }

    /// Convert the optimized contraction tree to a Python dictionary.
    ///
    /// This returns the same structure as `NestedEinsum.to_dict()`.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        nested_to_dict(py, &self.inner.eins)
    }

    fn __repr__(&self) -> String {
        format!("SlicedEinsum(slicing={:?})", self.inner.slicing)
    }

    /// Compute the contraction complexity of this sliced einsum code.
    ///
    /// The complexity accounts for slicing by setting sliced indices to size 1,
    /// representing the complexity of computing a single slice.
    ///
    /// Args:
    ///     ixs: List of input tensor index lists.
    ///     sizes: Dictionary mapping indices to their dimensions.
    ///
    /// Returns:
    ///     ContractionComplexity with adjusted metrics for sliced computation.
    ///
    /// Example:
    ///     ```python
    ///     sliced = slice_code(tree, ixs, sizes, slicer)
    ///     comp = sliced.complexity(ixs, sizes)
    ///     print(f"Space per slice: 2^{comp.sc:.2f}")
    ///     ```
    fn complexity(
        &self,
        ixs: Vec<Vec<i64>>,
        sizes: HashMap<i64, usize>,
    ) -> PyContractionComplexity {
        omeco::sliced_complexity(&self.inner, &sizes, &ixs).into()
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

/// Exact optimizer for small contraction networks.
///
/// Args:
///     verbose: Print progress information during the exhaustive search.
///              Default: False.
#[pyclass(name = "ExhaustiveSearch")]
#[derive(Clone)]
pub struct PyExhaustiveSearch {
    inner: ExhaustiveSearch,
}

#[pymethods]
impl PyExhaustiveSearch {
    /// Create a new exact optimizer.
    #[new]
    #[pyo3(signature = (verbose=false))]
    fn new(verbose: bool) -> Self {
        Self {
            inner: ExhaustiveSearch::new(verbose),
        }
    }

    /// Get the verbose flag.
    #[getter]
    fn verbose(&self) -> bool {
        self.inner.verbose
    }

    fn __repr__(&self) -> String {
        format!(
            "ExhaustiveSearch(verbose={})",
            if self.inner.verbose { "True" } else { "False" }
        )
    }
}

/// Treewidth-heuristic optimizer for contraction order.
///
/// Computes a weighted minimum-degree elimination order over the index labels
/// (an AMD-style quotient-graph ordering) and replays it into a binary
/// contraction tree. Deterministic and highly scalable — well suited to large
/// structured networks (graphical models, relational instances) where a
/// low-treewidth elimination order beats pairwise-greedy and annealing search.
///
/// Mirrors the ``Treewidth`` optimizer of the Julia package
/// OMEinsumContractionOrders (which backs its orderings with CliqueTrees.jl).
#[pyclass(name = "Treewidth")]
#[derive(Clone)]
pub struct PyTreewidth {
    inner: Treewidth,
}

#[pymethods]
impl PyTreewidth {
    /// Create a new Treewidth optimizer.
    ///
    /// Args:
    ///     algorithm: Elimination-ordering algorithm. Currently only
    ///                "min_degree" (weighted minimum degree) is supported,
    ///                which is the default.
    #[new]
    #[pyo3(signature = (algorithm="min_degree"))]
    fn new(algorithm: &str) -> PyResult<Self> {
        let alg = match algorithm {
            "min_degree" | "mindegree" | "MinDegree" => omeco::EliminationAlgorithm::MinDegree,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown elimination algorithm '{other}' (expected 'min_degree')"
                )))
            }
        };
        Ok(Self {
            inner: Treewidth::new(alg),
        })
    }

    /// Get the elimination algorithm name.
    #[getter]
    fn algorithm(&self) -> String {
        match self.inner.alg {
            omeco::EliminationAlgorithm::MinDegree => "min_degree".to_string(),
            _ => "unknown".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("Treewidth(algorithm={})", self.algorithm())
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
    ///     preprocess: Simplify the network (splice degree-2/parallel tensors) before
    ///                 annealing, then splice the reduced tree back (default: True).
    ///     surgery_budget: Wall-clock seconds for a post-annealing waist-surgery
    ///                     refinement pass; 0.0 disables it (default: 0.0).
    #[new]
    #[pyo3(signature = (ntrials=10, niters=50, betas=None, score=None, preprocess=true, surgery_budget=0.0))]
    fn new(
        ntrials: usize,
        niters: usize,
        betas: Option<Vec<f64>>,
        score: Option<PyScoreFunction>,
        preprocess: bool,
        surgery_budget: f64,
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
                preprocess,
                surgery_budget,
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

    /// Whether the network is simplified before annealing and spliced back after.
    #[getter]
    fn preprocess(&self) -> bool {
        self.inner.preprocess
    }

    /// Wall-clock seconds budgeted for the post-annealing waist-surgery pass
    /// (0.0 disables it).
    #[getter]
    fn surgery_budget(&self) -> f64 {
        self.inner.surgery_budget
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

/// Statistics describing how far the structural simplification pass shrank a
/// network, returned alongside the optimized tree by `simplify_then_optimize`.
#[pyclass(name = "SimplifyReport")]
#[derive(Clone)]
pub struct PySimplifyReport {
    inner: omeco::preprocess::SimplifyReport,
}

#[pymethods]
impl PySimplifyReport {
    /// Number of tensors in the original network.
    #[getter]
    fn n_original(&self) -> usize {
        self.inner.n_original
    }

    /// Number of super-tensors surviving in the reduced network.
    #[getter]
    fn n_reduced(&self) -> usize {
        self.inner.n_reduced
    }

    /// Fraction of tensors removed: `1 - n_reduced / n_original` (0 when empty).
    #[getter]
    fn shrink(&self) -> f64 {
        self.inner.shrink
    }

    fn __repr__(&self) -> String {
        format!(
            "SimplifyReport(n_original={}, n_reduced={}, shrink={:.3})",
            self.inner.n_original, self.inner.n_reduced, self.inner.shrink
        )
    }
}

/// Diagnostics from a `waist_refine` run.
#[pyclass(name = "WaistReport")]
#[derive(Clone)]
pub struct PyWaistReport {
    inner: omeco::waist_surgery::WaistReport,
}

#[pymethods]
impl PyWaistReport {
    /// Number of tensors in the input network.
    #[getter]
    fn n_original(&self) -> usize {
        self.inner.n_original
    }

    /// Number of waist-surgery iterations attempted.
    #[getter]
    fn surgery_calls(&self) -> u64 {
        self.inner.surgery_calls
    }

    /// Iterations where bounded FM strictly improved the incumbent partition
    /// under the same cut-weight functional.
    #[getter]
    fn cheaper_cuts(&self) -> u64 {
        self.inner.cheaper_cuts
    }

    /// Candidates that passed the no-new-bottleneck gate and entered rebuilding.
    #[getter]
    fn rebuild_attempts(&self) -> u64 {
        self.inner.rebuild_attempts
    }

    /// Rebuilds that strictly lowered the global time complexity and were kept.
    #[getter]
    fn rebuild_accepts(&self) -> u64 {
        self.inner.rebuild_accepts
    }

    /// Iterations where the bounded FM search found no cheaper comparable cut.
    #[getter]
    fn waist_min_hits(&self) -> u64 {
        self.inner.waist_min_hits
    }

    fn __repr__(&self) -> String {
        format!(
            "WaistReport(n_original={}, surgery_calls={}, cheaper_cuts={}, rebuild_attempts={}, rebuild_accepts={}, waist_min_hits={})",
            self.inner.n_original,
            self.inner.surgery_calls,
            self.inner.cheaper_cuts,
            self.inner.rebuild_attempts,
            self.inner.rebuild_accepts,
            self.inner.waist_min_hits
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
    let opt = optimizer.unwrap_or_else(|| PyTreeSA::new(10, 50, None, None, true, 0.0));

    opt.inner
        .optimize(&code, &sizes)
        .map(|inner| PyNestedEinsum { inner })
        .ok_or_else(|| PyValueError::new_err("Optimization failed"))
}

/// Simplify a network (splice degree-2/parallel tensors), optimize the reduced
/// network, then splice the reduced tree back into a full contraction tree.
///
/// Only a `GreedyMethod` optimizer is accepted here: the reduced-network
/// optimizer is generic over `CodeOptimizer` on the Rust side, which cannot
/// cross the PyO3 boundary, so Greedy is the documented inner optimizer for
/// this pipeline. TreeSA users get the same simplify/splice behavior via
/// `TreeSA(preprocess=True)` (the default) instead of calling this function
/// directly.
///
/// Args:
///     ixs: List of index lists for each tensor.
///     out: Output indices.
///     sizes: Dictionary mapping indices to their dimensions.
///     optimizer: GreedyMethod optimizer for the reduced network (default: GreedyMethod()).
///
/// Returns:
///     A tuple of (optimized contraction tree, SimplifyReport).
#[pyfunction]
#[pyo3(signature = (ixs, out, sizes, optimizer=None))]
fn simplify_then_optimize(
    ixs: Vec<Vec<i64>>,
    out: Vec<i64>,
    sizes: HashMap<i64, usize>,
    optimizer: Option<PyGreedyMethod>,
) -> PyResult<(PyNestedEinsum, PySimplifyReport)> {
    let code = EinCode::new(ixs, out);
    let opt = optimizer.map(|o| o.inner).unwrap_or_default();
    omeco::preprocess::simplify_then_optimize(&code, &sizes, &opt)
        .map(|(t, r)| (PyNestedEinsum { inner: t }, PySimplifyReport { inner: r }))
        .ok_or_else(|| PyValueError::new_err("optimization failed"))
}

/// Refine a contraction tree with a bounded waist-surgery pass: repeatedly
/// finds and repairs the tree's costliest cut, keeping only rebuilds that
/// strictly lower the global time complexity, within a wall-clock budget.
///
/// Args:
///     tree: Seed contraction tree to refine (e.g. from `optimize_code`).
///     ixs: List of index lists for each tensor.
///     out: Output indices.
///     sizes: Dictionary mapping indices to their dimensions.
///     budget_secs: Wall-clock seconds to spend surgering; must be finite and >= 0.
///
/// Returns:
///     A tuple of (refined contraction tree, WaistReport). The refined tree's
///     time complexity is never worse than the input tree's.
#[pyfunction]
fn waist_refine(
    tree: PyNestedEinsum,
    ixs: Vec<Vec<i64>>,
    out: Vec<i64>,
    sizes: HashMap<i64, usize>,
    budget_secs: f64,
) -> PyResult<(PyNestedEinsum, PyWaistReport)> {
    if !budget_secs.is_finite() || budget_secs < 0.0 {
        return Err(PyValueError::new_err(
            "budget_secs must be a non-negative finite number",
        ));
    }
    let code = EinCode::new(ixs, out);
    let (t, r) = omeco::waist_surgery::refine(
        &tree.inner,
        &code,
        &sizes,
        std::time::Duration::from_secs_f64(budget_secs),
    );
    Ok((PyNestedEinsum { inner: t }, PyWaistReport { inner: r }))
}

/// Optimize the contraction order using the treewidth heuristic.
///
/// Args:
///     ixs: List of index lists for each tensor.
///     out: Output indices.
///     sizes: Dictionary mapping indices to their dimensions.
///     optimizer: Treewidth optimizer configuration.
///
/// Returns:
///     Optimized contraction tree as NestedEinsum.
#[pyfunction]
#[pyo3(signature = (ixs, out, sizes, optimizer=None))]
fn optimize_treewidth(
    ixs: Vec<Vec<i64>>,
    out: Vec<i64>,
    sizes: HashMap<i64, usize>,
    optimizer: Option<PyTreewidth>,
) -> PyResult<PyNestedEinsum> {
    let code = EinCode::new(ixs, out);
    let opt = optimizer.unwrap_or_else(|| PyTreewidth {
        inner: Treewidth::default(),
    });

    omeco::optimize_treewidth(&code, &sizes, &opt.inner)
        .map(|inner| PyNestedEinsum { inner })
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Optimize the contraction order using exact exhaustive search.
///
/// Args:
///     ixs: List of index lists for each tensor.
///     out: Output indices.
///     sizes: Dictionary mapping indices to their dimensions.
///     optimizer: ExhaustiveSearch optimizer configuration.
///
/// Returns:
///     Optimized contraction tree as NestedEinsum.
#[pyfunction]
#[pyo3(signature = (ixs, out, sizes, optimizer=None))]
fn optimize_exhaustive(
    ixs: Vec<Vec<i64>>,
    out: Vec<i64>,
    sizes: HashMap<i64, usize>,
    optimizer: Option<PyExhaustiveSearch>,
) -> PyResult<PyNestedEinsum> {
    let code = EinCode::new(ixs, out);
    let opt = optimizer.unwrap_or_else(|| PyExhaustiveSearch::new(false));

    omeco::optimize_exhaustive(&code, &sizes, &opt.inner)
        .map(|inner| PyNestedEinsum { inner })
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Compute the contraction complexity of an optimized tree.
///
/// .. deprecated:: 0.3.0
///     Use `tree.complexity(ixs, sizes)` instead.
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
    py: Python<'_>,
    tree: &PyNestedEinsum,
    ixs: Vec<Vec<i64>>,
    sizes: HashMap<i64, usize>,
) -> PyResult<PyContractionComplexity> {
    // Emit deprecation warning
    let warnings = py.import("warnings")?;
    warnings.call_method1(
        "warn",
        (
            "contraction_complexity() is deprecated, use tree.complexity(ixs, sizes) instead",
            py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
            2,
        ),
    )?;

    Ok(omeco::contraction_complexity(&tree.inner, &sizes, &ixs).into())
}

/// Compute the complexity of a sliced contraction.
///
/// .. deprecated:: 0.3.0
///     Use `sliced.complexity(ixs, sizes)` instead.
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
    py: Python<'_>,
    sliced: &PySlicedEinsum,
    ixs: Vec<Vec<i64>>,
    sizes: HashMap<i64, usize>,
) -> PyResult<PyContractionComplexity> {
    // Emit deprecation warning
    let warnings = py.import("warnings")?;
    warnings.call_method1(
        "warn",
        (
            "sliced_complexity() is deprecated, use sliced.complexity(ixs, sizes) instead",
            py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
            2,
        ),
    )?;

    Ok(omeco::sliced_complexity(&sliced.inner, &sizes, &ixs).into())
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

/// Unified optimizer type for contraction order optimization.
#[derive(FromPyObject)]
enum PyOptimizer {
    Greedy(PyGreedyMethod),
    Exhaustive(PyExhaustiveSearch),
    TreeSA(PyTreeSA),
    Treewidth(PyTreewidth),
}

/// Optimize the contraction order using the specified optimizer.
///
/// This is the unified interface for contraction order optimization.
///
/// Args:
///     ixs: List of index lists for each tensor (e.g., [[0, 1], [1, 2]]).
///     out: Output indices (e.g., [0, 2]).
///     sizes: Dictionary mapping indices to their dimensions.
///     optimizer: Optimizer to use (GreedyMethod, ExhaustiveSearch, or TreeSA).
///                Defaults to GreedyMethod().
///
/// Returns:
///     Optimized contraction tree as NestedEinsum.
///
/// Example:
///     >>> from omeco import optimize_code, GreedyMethod, ExhaustiveSearch, TreeSA
///     >>> ixs = [[0, 1], [1, 2], [2, 3]]
///     >>> out = [0, 3]
///     >>> sizes = {0: 100, 1: 50, 2: 80, 3: 100}
///     >>> tree = optimize_code(ixs, out, sizes, GreedyMethod())
///     >>> tree = optimize_code(ixs, out, sizes, ExhaustiveSearch())
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
        Some(PyOptimizer::Greedy(opt)) => opt
            .inner
            .optimize(&code, &sizes)
            .ok_or_else(|| PyValueError::new_err("Optimization failed")),
        Some(PyOptimizer::Exhaustive(opt)) => omeco::optimize_exhaustive(&code, &sizes, &opt.inner)
            .map_err(|err| PyValueError::new_err(err.to_string())),
        Some(PyOptimizer::TreeSA(opt)) => opt
            .inner
            .optimize(&code, &sizes)
            .ok_or_else(|| PyValueError::new_err("Optimization failed")),
        Some(PyOptimizer::Treewidth(opt)) => omeco::optimize_treewidth(&code, &sizes, &opt.inner)
            .map_err(|err| PyValueError::new_err(err.to_string())),
        None => GreedyMethod::default()
            .optimize(&code, &sizes)
            .ok_or_else(|| PyValueError::new_err("Optimization failed")),
    };

    result.map(|inner| PyNestedEinsum { inner })
}

/// Python module for omeco tensor network contraction order optimization.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNestedEinsum>()?;
    m.add_class::<PySlicedEinsum>()?;
    m.add_class::<PyContractionComplexity>()?;
    m.add_class::<PyScoreFunction>()?;
    m.add_class::<PyGreedyMethod>()?;
    m.add_class::<PyExhaustiveSearch>()?;
    m.add_class::<PyTreeSA>()?;
    m.add_class::<PyTreewidth>()?;
    m.add_class::<PyTreeSASlicer>()?;
    m.add_class::<PySimplifyReport>()?;
    m.add_class::<PyWaistReport>()?;
    m.add_function(wrap_pyfunction!(optimize_code, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_greedy, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_exhaustive, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_treesa, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_treewidth, m)?)?;
    m.add_function(wrap_pyfunction!(simplify_then_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(waist_refine, m)?)?;
    m.add_function(wrap_pyfunction!(contraction_complexity, m)?)?;
    m.add_function(wrap_pyfunction!(sliced_complexity, m)?)?;
    m.add_function(wrap_pyfunction!(uniform_size_dict, m)?)?;
    m.add_function(wrap_pyfunction!(slice_code, m)?)?;
    Ok(())
}
