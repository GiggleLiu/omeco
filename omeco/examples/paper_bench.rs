//! Deterministic benchmark runner behind the paper's contraction-order tables.
//!
//! Every number in the paper regenerates from the exact current
//! `origin/master` revision with this binary — there is no reference host and
//! no recorded oracle, and artifacts produced from any other revision are not
//! paper data (see the master revision hard gate in
//! `benchmarks/paper/README.md`). A manifest names
//! the instance files and the optimizer *arms*; the runner writes one JSON
//! artifact whose bytes are stable for a fixed (manifest, set, machine).
//!
//! ```text
//! cargo run --release --example paper_bench -p omeco -- \
//!     --manifest benchmarks/paper/manifest.json \
//!     --set ci \
//!     --out /tmp/ci.json \
//!     [--repo-root <dir>]
//! ```
//!
//! Compare two artifacts with `python3 benchmarks/paper/check.py a.json b.json`.
//! See `benchmarks/paper/README.md` for the determinism contract and the
//! manifest override policy.
//!
//! Every user-facing failure (bad flag, missing file, malformed JSON, unknown
//! manifest key) prints a message to stderr and exits with status 2; no user
//! error is allowed to surface as a panic.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use omeco::treesa::anneal_surgery_rounds;
use omeco::{
    contraction_complexity, optimize_code, simplify, splice, EinCode, GreedyMethod, NestedEinsum,
    TreeSA, Treewidth,
};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Manifest schema
//
// Every struct denies unknown fields: a typo in the manifest must abort the
// run, never silently fall back to a default. `serde` names the offending key
// in its error message, which is what the runner prints.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    /// Named benchmark sets, e.g. `ci` and `full`. `BTreeMap` (not `HashMap`)
    /// so that error messages listing the known sets are ordered.
    sets: BTreeMap<String, SetSpec>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SetSpec {
    arms: ArmsSpec,
    instances: Vec<InstanceSpec>,
}

/// The arms a set may request. Modelled as a struct rather than a map so that
/// an unknown arm name (`"treesa_round"`) is a hard error, and so that the
/// per-arm key vocabularies stay distinct — `rounds` is meaningful only for
/// `treesa_rounds` and is rejected elsewhere.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArmsSpec {
    greedy: Option<GreedyArm>,
    treesa: Option<TreeSAArm>,
    treesa_rounds: Option<RoundsArm>,
    treesa_surgery_rule: Option<SurgeryRuleArm>,
}

/// The greedy arm takes no configuration; `{}` is the only legal value.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GreedyArm {}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TreeSAArm {
    ntrials: Option<usize>,
    niters: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RoundsArm {
    ntrials: Option<usize>,
    niters: Option<usize>,
    /// Required: a rounds arm without a round count is a manifest bug, not a
    /// request for zero rounds.
    rounds: u64,
    /// Emit exact per-round waist-cut comparisons for mechanism figures.
    trace_cuts: Option<bool>,
    /// Emit per-round configured-score fields without the full cut trace.
    trace_scores: Option<bool>,
}

impl RoundsArm {
    /// The `waist` object is emitted only for `trace_cuts` sets.
    fn emit_cuts(&self) -> bool {
        self.trace_cuts.unwrap_or(false)
    }

    /// `score_before`/`score_retained` accompany either trace flag: a cut
    /// trace without the score ratchet would be unverifiable.
    fn emit_scores(&self) -> bool {
        self.emit_cuts() || self.trace_scores.unwrap_or(false)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SurgeryRuleArm {
    ntrials: Option<usize>,
    niters: Option<usize>,
    probability: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InstanceSpec {
    name: String,
    /// Repo-root-relative path, resolved against `--repo-root`.
    path: String,
    /// Run the `treewidth` arm on this instance. Required (no default) so that
    /// adding an instance forces an explicit decision.
    treewidth: bool,
    /// Optional deterministic tensor-order permutation for independent runs.
    relabel_seed: Option<u64>,
}

/// Instance file schema, shared with `benchmarks/graphs/*.json`. Extra keys
/// (`name`, `description`) are tolerated — instance files are data, not
/// configuration, so strictness would only get in the way here.
#[derive(Debug, Deserialize)]
struct InstanceData {
    ixs: Vec<Vec<usize>>,
    iy: Vec<usize>,
    /// JSON object keys are strings; labels are parsed back to `usize`.
    sizes: BTreeMap<String, usize>,
}

// ---------------------------------------------------------------------------
// Output schema
//
// Ordered structs only: field order is fixed by the declarations below and the
// row order by an explicit sort, so `to_string_pretty` is byte-stable. No
// `HashMap` may appear anywhere in this section.
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct Output {
    format: u32,
    set: String,
    results: Vec<ResultRow>,
}

#[derive(Debug, Serialize)]
struct ResultRow {
    instance: String,
    arm: String,
    tc: f64,
    sc: f64,
    rwc: f64,
    /// Per-round retained-incumbent trace, emitted by `treesa_rounds` only.
    #[serde(skip_serializing_if = "Option::is_none")]
    curve: Option<Vec<CurvePoint>>,
}

#[derive(Debug, Serialize)]
struct CurvePoint {
    round: u64,
    tc_before: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    score_before: Option<f64>,
    tc_after_surgery: f64,
    tc_after_anneal: f64,
    tc_retained: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    score_retained: Option<f64>,
    surgery_accepted: bool,
    /// Exact like-for-like cut comparison, emitted only when requested by the
    /// manifest's `trace_cuts` flag.
    #[serde(skip_serializing_if = "Option::is_none")]
    waist: Option<WaistPoint>,
}

#[derive(Debug, Serialize)]
struct WaistPoint {
    incumbent_cut_cost: f64,
    best_alt_cut_cost: f64,
    waist_node_cost: f64,
    rebuild_attempted: bool,
    rebuild_accepted: bool,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Report a user-facing failure and exit with the documented status 2.
fn fail(msg: &str) -> ! {
    eprintln!("paper_bench: error: {msg}");
    std::process::exit(2)
}

// ---------------------------------------------------------------------------
// Argument parsing (hand-rolled; the runner takes no third-party CLI dep)
// ---------------------------------------------------------------------------

const USAGE: &str = "usage: paper_bench --manifest <file> --set <name> --out <file> \
                     [--repo-root <dir>]";

struct Args {
    manifest: PathBuf,
    set: String,
    out: PathBuf,
    repo_root: PathBuf,
}

fn parse_args() -> Args {
    let mut manifest: Option<PathBuf> = None;
    let mut set: Option<String> = None;
    let mut out: Option<PathBuf> = None;
    let mut repo_root: Option<PathBuf> = None;

    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        // Every flag takes exactly one value, so the value fetch is shared.
        let mut value = || match argv.next() {
            Some(v) => v,
            None => fail(&format!("flag `{flag}` requires a value\n{USAGE}")),
        };
        match flag.as_str() {
            "--manifest" => manifest = Some(PathBuf::from(value())),
            "--set" => set = Some(value()),
            "--out" => out = Some(PathBuf::from(value())),
            "--repo-root" => repo_root = Some(PathBuf::from(value())),
            "--help" | "-h" => {
                println!("{USAGE}");
                std::process::exit(0)
            }
            other => fail(&format!("unknown flag `{other}`\n{USAGE}")),
        }
    }

    Args {
        manifest: manifest.unwrap_or_else(|| fail(&format!("missing --manifest\n{USAGE}"))),
        set: set.unwrap_or_else(|| fail(&format!("missing --set\n{USAGE}"))),
        out: out.unwrap_or_else(|| fail(&format!("missing --out\n{USAGE}"))),
        repo_root: repo_root.unwrap_or_else(|| PathBuf::from(".")),
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

fn read_json<T: serde::de::DeserializeOwned>(path: &Path, what: &str) -> T {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|e| fail(&format!("cannot read {what} `{}`: {e}", path.display())));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| fail(&format!("invalid {what} `{}`: {e}", path.display())))
}

/// Turn an instance file into the `(code, size_dict)` pair the optimizers take.
fn build_instance(
    mut data: InstanceData,
    path: &Path,
    relabel_seed: Option<u64>,
) -> (EinCode<usize>, HashMap<usize, usize>) {
    let mut sizes = HashMap::with_capacity(data.sizes.len());
    for (label, dim) in &data.sizes {
        let parsed = label.parse::<usize>().unwrap_or_else(|_| {
            fail(&format!(
                "instance `{}`: label `{label}` is not an integer",
                path.display()
            ))
        });
        sizes.insert(parsed, *dim);
    }
    // A label present in `ixs`/`iy` but absent from `sizes` would panic deep
    // inside the optimizers; catch it here so the message names the file.
    for label in data.ixs.iter().flatten().chain(data.iy.iter()) {
        if !sizes.contains_key(label) {
            fail(&format!(
                "instance `{}`: label `{label}` has no entry in `sizes`",
                path.display()
            ));
        }
    }
    if let Some(seed) = relabel_seed {
        permute_tensors(&mut data.ixs, seed);
    }
    (EinCode::new(data.ixs, data.iy), sizes)
}

fn permute_tensors(ixs: &mut [Vec<usize>], seed: u64) {
    let mut rng = SmallRng::seed_from_u64(seed);
    ixs.shuffle(&mut rng);
}

// ---------------------------------------------------------------------------
// Arms
// ---------------------------------------------------------------------------

/// Round to six decimals so the artifact is insensitive to the last few bits
/// of the floating-point accumulation order.
fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

/// `TreeSA::default()` with the manifest's overrides applied and the rounds
/// loop switched off. The `treesa_rounds` arm reuses this to build its seed,
/// which is what makes the two arms comparable at equal `ntrials`/`niters`.
fn treesa_config(ntrials: Option<usize>, niters: Option<usize>) -> TreeSA {
    let base = TreeSA::default();
    TreeSA {
        ntrials: ntrials.unwrap_or(base.ntrials),
        niters: niters.unwrap_or(base.niters),
        surgery_iters: 0,
        ..base
    }
}

fn row(
    instance: &str,
    arm: &str,
    tree: &NestedEinsum<usize>,
    code: &EinCode<usize>,
    sizes: &HashMap<usize, usize>,
    curve: Option<Vec<CurvePoint>>,
) -> ResultRow {
    let cc = contraction_complexity(tree, sizes, &code.ixs);
    ResultRow {
        instance: instance.to_string(),
        arm: arm.to_string(),
        tc: round6(cc.tc),
        sc: round6(cc.sc),
        rwc: round6(cc.rwc),
        curve,
    }
}

/// Run every arm the set requests on one instance.
fn run_instance(
    spec: &InstanceSpec,
    arms: &ArmsSpec,
    code: &EinCode<usize>,
    sizes: &HashMap<usize, usize>,
) -> Vec<ResultRow> {
    let mut rows = Vec::new();
    let optimized = |tree: Option<NestedEinsum<usize>>, arm: &str| match tree {
        Some(t) => t,
        None => fail(&format!(
            "instance `{}`: arm `{arm}` returned no contraction tree",
            spec.name
        )),
    };

    if arms.greedy.is_some() {
        let started = Instant::now();
        let tree = optimized(
            optimize_code(code, sizes, &GreedyMethod::default()),
            "greedy",
        );
        rows.push(row(&spec.name, "greedy", &tree, code, sizes, None));
        report_progress(&spec.name, "greedy", started);
    }

    if let Some(arm) = &arms.treesa {
        let started = Instant::now();
        let config = treesa_config(arm.ntrials, arm.niters);
        let tree = optimized(optimize_code(code, sizes, &config), "treesa");
        rows.push(row(&spec.name, "treesa", &tree, code, sizes, None));
        report_progress(&spec.name, "treesa", started);
    }

    if let Some(arm) = &arms.treesa_rounds {
        let started = Instant::now();
        // Match `optimize_treesa`: simplify, anneal and run the rounds on the
        // reduced hypergraph, then restore the retained contractions. Driving
        // the public stages explicitly exposes the per-round trajectory.
        let simplified = simplify(code, sizes);
        let mut config = treesa_config(arm.ntrials, arm.niters);
        config.preprocess = false;
        let seed = optimized(
            optimize_code(&simplified.code, sizes, &config),
            "treesa_rounds",
        );
        let (reduced, report) =
            anneal_surgery_rounds(&seed, &simplified.code, sizes, &config, arm.rounds);
        let tree = splice(&reduced, &simplified.subtrees);
        let curve = report
            .round_trace
            .iter()
            .map(|trace| CurvePoint {
                round: trace.round,
                tc_before: round6(trace.tc_before),
                score_before: arm.emit_scores().then_some(round6(trace.score_before)),
                tc_after_surgery: round6(trace.tc_after_surgery),
                tc_after_anneal: round6(trace.tc_after_anneal),
                tc_retained: round6(trace.tc_retained),
                score_retained: arm.emit_scores().then_some(round6(trace.score_retained)),
                surgery_accepted: trace.surgery_accepted,
                waist: arm
                    .emit_cuts()
                    .then_some(trace.waist)
                    .flatten()
                    .map(|waist| WaistPoint {
                        incumbent_cut_cost: round6(waist.incumbent_cut_cost),
                        best_alt_cut_cost: round6(waist.best_alt_cut_cost),
                        waist_node_cost: round6(waist.waist_node_cost),
                        rebuild_attempted: waist.rebuild_attempted,
                        rebuild_accepted: waist.rebuild_accepted,
                    }),
            })
            .collect();
        rows.push(row(
            &spec.name,
            "treesa_rounds",
            &tree,
            code,
            sizes,
            Some(curve),
        ));
        report_progress(&spec.name, "treesa_rounds", started);
    }

    if let Some(arm) = &arms.treesa_surgery_rule {
        let started = Instant::now();
        if !arm.probability.is_finite() || !(0.0..=1.0).contains(&arm.probability) {
            fail(&format!(
                "instance `{}`: arm `treesa_surgery_rule` probability must be finite and in [0, 1], got {}",
                spec.name, arm.probability
            ));
        }
        let config =
            treesa_config(arm.ntrials, arm.niters).with_surgery_probability(arm.probability);
        let tree = optimized(optimize_code(code, sizes, &config), "treesa_surgery_rule");
        rows.push(row(
            &spec.name,
            "treesa_surgery_rule",
            &tree,
            code,
            sizes,
            None,
        ));
        report_progress(&spec.name, "treesa_surgery_rule", started);
    }

    if spec.treewidth {
        let started = Instant::now();
        let tree = optimized(
            optimize_code(code, sizes, &Treewidth::default()),
            "treewidth",
        );
        rows.push(row(&spec.name, "treewidth", &tree, code, sizes, None));
        report_progress(&spec.name, "treewidth", started);
    }

    rows
}

/// Progress goes to stderr: stdout and the artifact must stay clean.
fn report_progress(instance: &str, arm: &str, started: Instant) {
    eprintln!(
        "  {instance:<20} {arm:<14} {:.1}s",
        started.elapsed().as_secs_f64()
    );
}

// ---------------------------------------------------------------------------

fn main() {
    let args = parse_args();
    let manifest: Manifest = read_json(&args.manifest, "manifest");

    let set = manifest.sets.get(&args.set).unwrap_or_else(|| {
        let known: Vec<&str> = manifest.sets.keys().map(String::as_str).collect();
        fail(&format!(
            "unknown set `{}`; manifest defines: {}",
            args.set,
            known.join(", ")
        ))
    });

    let started = Instant::now();
    let mut results = Vec::new();
    for spec in &set.instances {
        let path = args.repo_root.join(&spec.path);
        let data: InstanceData = read_json(&path, "instance");
        let (code, sizes) = build_instance(data, &path, spec.relabel_seed);
        results.extend(run_instance(spec, &set.arms, &code, &sizes));
    }

    // Row order must not depend on manifest order or on iteration order
    // anywhere upstream.
    results.sort_by(|a, b| (&a.instance, &a.arm).cmp(&(&b.instance, &b.arm)));

    let output = Output {
        format: 2,
        set: args.set,
        results,
    };
    let mut text = serde_json::to_string_pretty(&output)
        .expect("output structs are plain data and always serialize");
    text.push('\n');
    fs::write(&args.out, text).unwrap_or_else(|e| {
        fail(&format!(
            "cannot write output `{}`: {e}",
            args.out.display()
        ))
    });

    eprintln!(
        "paper_bench: wrote {} ({} results, {:.1}s)",
        args.out.display(),
        output.results.len(),
        started.elapsed().as_secs_f64()
    );
}

#[cfg(test)]
mod tests {
    use super::{permute_tensors, CurvePoint, RoundsArm};

    #[test]
    fn tensor_relabeling_is_seeded_and_deterministic() {
        let original: Vec<Vec<usize>> = (0..32).map(|value| vec![value]).collect();
        let mut first = original.clone();
        let mut again = original.clone();
        let mut other = original.clone();
        permute_tensors(&mut first, 54);
        permute_tensors(&mut again, 54);
        permute_tensors(&mut other, 55);
        assert_eq!(first, again);
        assert_ne!(first, original);
        assert_ne!(first, other);
        first.sort();
        assert_eq!(first, original);
    }

    #[test]
    fn score_fields_follow_either_trace_flag() {
        let arm = |trace_cuts, trace_scores| RoundsArm {
            ntrials: None,
            niters: None,
            rounds: 1,
            trace_cuts,
            trace_scores,
        };

        assert!(!arm(None, None).emit_scores());
        assert!(arm(Some(true), None).emit_scores());
        assert!(arm(None, Some(true)).emit_scores());
        assert!(!arm(Some(false), Some(false)).emit_scores());
        assert!(arm(Some(true), None).emit_cuts());
        assert!(!arm(None, Some(true)).emit_cuts());
    }

    #[test]
    fn trace_scores_are_opt_in_artifact_fields() {
        let point = |score_before, score_retained| CurvePoint {
            round: 0,
            tc_before: 1.0,
            score_before,
            tc_after_surgery: 1.0,
            tc_after_anneal: 1.0,
            tc_retained: 1.0,
            score_retained,
            surgery_accepted: false,
            waist: None,
        };

        let ordinary = serde_json::to_value(point(None, None)).unwrap();
        assert!(ordinary.get("score_before").is_none());
        assert!(ordinary.get("score_retained").is_none());

        let traced = serde_json::to_value(point(Some(2.0), Some(1.5))).unwrap();
        assert_eq!(traced["score_before"], 2.0);
        assert_eq!(traced["score_retained"], 1.5);
    }
}
