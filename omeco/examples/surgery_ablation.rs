//! Deterministic work-matched TreeSA and waist-surgery ablation driver.
//!
//! See `benchmarks/surgery_ablation/README.md` for the protocol and commands.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::time::Instant;

use omeco::treesa::{anneal_refine_rounds, optimize_treesa_seeded, RoundsOptions, RoundsSchedule};
use omeco::waist_surgery::{RebuildMode, SurgeryScope};
use omeco::{
    contraction_complexity, simplify, splice, EinCode, NestedEinsum, ScoreFunction, TreeSA,
};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const DEFAULT_VISITS: u64 = 140_000_000;
const BETA_LEVELS: u64 = 300;
/// Row schema version. Bump when the row layout changes: rows written under an
/// older version are never treated as complete, so a resume re-runs and
/// replaces them instead of silently keeping stale data.
const SCHEMA_VERSION: u32 = 2;
const USAGE: &str = "usage: surgery_ablation --instances <dir> --out <file.jsonl> \
    [--only name,name] [--raw] [--labels N] [--rounds 8,32] \
    [--set all|a|b] [--visits N] [--jobs N]";

#[derive(Debug, Error)]
enum AppError {
    #[error("{0}")]
    Message(String),
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid JSON at {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("worker process exited with {0}")]
    Worker(ExitStatus),
    #[error("large-stack worker panicked")]
    WorkerPanic,
}

type AppResult<T> = Result<T, AppError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArmSet {
    All,
    A,
    B,
}

impl ArmSet {
    fn includes_a(self) -> bool {
        matches!(self, Self::All | Self::A)
    }

    fn includes_b(self) -> bool {
        matches!(self, Self::All | Self::B)
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::A => "a",
            Self::B => "b",
        }
    }
}

#[derive(Clone, Debug)]
struct Args {
    instances: PathBuf,
    out: PathBuf,
    only: Option<Vec<String>>,
    raw: bool,
    labels: usize,
    rounds: Vec<u64>,
    set: ArmSet,
    visits: u64,
    jobs: usize,
    shard_index: usize,
    shard_count: usize,
    resume_from: Option<PathBuf>,
}

fn next_value<I: Iterator<Item = OsString>>(iter: &mut I, flag: &str) -> AppResult<OsString> {
    iter.next()
        .ok_or_else(|| AppError::Message(format!("flag `{flag}` requires a value\n{USAGE}")))
}

fn parse_number<T: std::str::FromStr>(value: OsString, flag: &str) -> AppResult<T> {
    value
        .to_string_lossy()
        .parse()
        .map_err(|_| AppError::Message(format!("invalid value for `{flag}`\n{USAGE}")))
}

fn parse_args_from<I: Iterator<Item = OsString>>(mut iter: I) -> AppResult<Args> {
    let mut instances = None;
    let mut out = None;
    let mut only = None;
    let mut raw = false;
    let mut labels = 5_usize;
    let mut rounds = vec![8, 32];
    let mut set = ArmSet::All;
    let mut visits = DEFAULT_VISITS;
    let mut jobs = 1_usize;
    let mut shard_index = 0_usize;
    let mut shard_count = 1_usize;
    let mut resume_from = None;

    while let Some(flag) = iter.next() {
        let flag = flag.to_string_lossy();
        match flag.as_ref() {
            "--instances" => instances = Some(PathBuf::from(next_value(&mut iter, &flag)?)),
            "--out" => out = Some(PathBuf::from(next_value(&mut iter, &flag)?)),
            "--only" => {
                let value = next_value(&mut iter, &flag)?.to_string_lossy().into_owned();
                only = Some(
                    value
                        .split(',')
                        .filter(|name| !name.is_empty())
                        .map(str::to_owned)
                        .collect(),
                );
            }
            "--raw" => raw = true,
            "--labels" => labels = parse_number(next_value(&mut iter, &flag)?, &flag)?,
            "--rounds" => {
                let value = next_value(&mut iter, &flag)?.to_string_lossy().into_owned();
                rounds = value
                    .split(',')
                    .map(|item| {
                        item.parse::<u64>().map_err(|_| {
                            AppError::Message(format!("invalid value for `--rounds`: {item}"))
                        })
                    })
                    .collect::<AppResult<Vec<_>>>()?;
            }
            "--set" => {
                set = match next_value(&mut iter, &flag)?.to_string_lossy().as_ref() {
                    "all" => ArmSet::All,
                    "a" => ArmSet::A,
                    "b" => ArmSet::B,
                    other => {
                        return Err(AppError::Message(format!(
                            "invalid --set `{other}`; expected all, a, or b"
                        )))
                    }
                };
            }
            "--visits" => visits = parse_number(next_value(&mut iter, &flag)?, &flag)?,
            "--jobs" => jobs = parse_number(next_value(&mut iter, &flag)?, &flag)?,
            "--shard-index" => shard_index = parse_number(next_value(&mut iter, &flag)?, &flag)?,
            "--shard-count" => shard_count = parse_number(next_value(&mut iter, &flag)?, &flag)?,
            "--resume-from" => resume_from = Some(PathBuf::from(next_value(&mut iter, &flag)?)),
            "--help" | "-h" => return Err(AppError::Message(USAGE.to_owned())),
            other => {
                return Err(AppError::Message(format!(
                    "unknown flag `{other}`\n{USAGE}"
                )))
            }
        }
    }
    if labels == 0 || rounds.is_empty() || jobs == 0 || shard_count == 0 {
        return Err(AppError::Message(
            "--labels, --jobs, --shard-count, and the --rounds list must be nonzero".to_owned(),
        ));
    }
    if shard_index >= shard_count {
        return Err(AppError::Message(
            "--shard-index must be smaller than --shard-count".to_owned(),
        ));
    }
    Ok(Args {
        instances: instances
            .ok_or_else(|| AppError::Message(format!("missing --instances\n{USAGE}")))?,
        out: out.ok_or_else(|| AppError::Message(format!("missing --out\n{USAGE}")))?,
        only,
        raw,
        labels,
        rounds,
        set,
        visits,
        jobs,
        shard_index,
        shard_count,
        resume_from,
    })
}

#[derive(Debug, Deserialize)]
struct InstanceData {
    ixs: Vec<Vec<usize>>,
    iy: Vec<usize>,
    sizes: BTreeMap<String, usize>,
}

#[derive(Clone)]
struct Instance {
    name: String,
    code: EinCode<usize>,
    sizes: HashMap<usize, usize>,
}

fn io_error(path: &Path, source: std::io::Error) -> AppError {
    AppError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn load_instances(args: &Args) -> AppResult<Vec<Instance>> {
    let requested: Option<HashSet<&str>> = args
        .only
        .as_ref()
        .map(|names| names.iter().map(String::as_str).collect());
    let entries =
        fs::read_dir(&args.instances).map_err(|error| io_error(&args.instances, error))?;
    let mut paths = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect::<Vec<_>>();
    paths.sort();
    let mut instances = Vec::new();
    for path in paths {
        let Some(name) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        if requested
            .as_ref()
            .is_some_and(|names| !names.contains(name))
        {
            continue;
        }
        let text = fs::read_to_string(&path).map_err(|error| io_error(&path, error))?;
        let data: InstanceData = serde_json::from_str(&text).map_err(|source| AppError::Json {
            path: path.clone(),
            source,
        })?;
        let mut sizes = HashMap::new();
        for (label, dimension) in data.sizes {
            let parsed = label.parse::<usize>().map_err(|_| {
                AppError::Message(format!(
                    "{} has non-integer label `{label}`",
                    path.display()
                ))
            })?;
            sizes.insert(parsed, dimension);
        }
        for label in data.ixs.iter().flatten().chain(&data.iy) {
            if !sizes.contains_key(label) {
                return Err(AppError::Message(format!(
                    "{} is missing a size for label {label}",
                    path.display()
                )));
            }
        }
        instances.push(Instance {
            name: name.to_owned(),
            code: EinCode::new(data.ixs, data.iy),
            sizes,
        });
    }
    if let Some(names) = requested {
        let found: HashSet<&str> = instances
            .iter()
            .map(|instance| instance.name.as_str())
            .collect();
        let mut missing = names.difference(&found).copied().collect::<Vec<_>>();
        missing.sort_unstable();
        if !missing.is_empty() {
            return Err(AppError::Message(format!(
                "requested instances not found: {}",
                missing.join(", ")
            )));
        }
    }
    Ok(instances)
}

fn relabel_instance(instance: &Instance, seed: u64) -> Instance {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut ixs = instance.code.ixs.clone();
    ixs.shuffle(&mut rng);
    let mut old_labels = instance.code.unique_labels();
    old_labels.sort_unstable();
    let mut new_labels: Vec<usize> = (0..old_labels.len()).collect();
    new_labels.shuffle(&mut rng);
    let mapping: HashMap<usize, usize> = old_labels.into_iter().zip(new_labels).collect();
    let map_labels = |labels: &[usize]| -> Vec<usize> {
        labels
            .iter()
            .filter_map(|label| mapping.get(label).copied())
            .collect()
    };
    let ixs = ixs.iter().map(|labels| map_labels(labels)).collect();
    let iy = map_labels(&instance.code.iy);
    let sizes = instance
        .sizes
        .iter()
        .filter_map(|(label, dimension)| mapping.get(label).map(|new| (*new, *dimension)))
        .collect();
    Instance {
        name: instance.name.clone(),
        code: EinCode::new(ixs, iy),
        sizes,
    }
}

#[derive(Debug, Clone, Serialize)]
struct Params {
    raw: bool,
    relabel_seed: u64,
    optimizer_seed: u64,
    beta_levels: usize,
    niters: usize,
    target_visits: u64,
    rounds: Option<u64>,
    surgery: Option<bool>,
    rebuild: Option<&'static str>,
    scope: Option<&'static str>,
    n_original: usize,
    n_optimized: usize,
}

#[derive(Debug, Clone, Serialize)]
struct TraceRow {
    round: u64,
    tc_before: f64,
    tc_after_surgery: f64,
    tc_after_anneal: f64,
    tc_retained: f64,
    surgery_accepted: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ResultRow {
    key: String,
    schema_version: u32,
    instance: String,
    label: String,
    arm: String,
    params: Params,
    tc: f64,
    sc: f64,
    rwc: f64,
    wall_seconds: f64,
    planned_visits: u64,
    fine_tune_sweeps: u64,
    total_node_visits: u64,
    accepted_rebuilds: u64,
    post_splice_guard_triggered: bool,
    round_trace: Vec<TraceRow>,
}

fn result_key(instance: &str, label: &str, arm: &str, raw: bool, visits: u64) -> String {
    format!("{instance}|{label}|{arm}|raw={raw}|target_visits={visits}")
}

fn expected_arms(args: &Args) -> Vec<String> {
    let mut arms = Vec::new();
    if args.set.includes_a() {
        arms.extend(
            ["treesa_x1", "treesa_x2", "treesa_x4", "treesa_x8"]
                .into_iter()
                .map(str::to_owned),
        );
    }
    for rounds in &args.rounds {
        if args.set.includes_b() {
            arms.extend(
                [
                    "cold_only",
                    "surg_greedy_root",
                    "surg_warm_root",
                    "surg_greedy_local",
                    "surg_warm_local",
                ]
                .into_iter()
                .map(|arm| format!("{arm}_r{rounds}")),
            );
        }
        if args.set.includes_a() {
            arms.push(format!("treesa_x1+cold{rounds}"));
        }
    }
    arms
}

fn existing_keys(path: &Path) -> AppResult<HashSet<String>> {
    if !path.exists() {
        return Ok(HashSet::new());
    }
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    let mut keys = HashSet::new();
    for (line_number, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|error| io_error(path, error))?;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value =
            serde_json::from_str(&line).map_err(|source| AppError::Json {
                path: path.to_path_buf(),
                source,
            })?;
        // Rows written under an older schema are stale: do not let their keys
        // mark a group complete, so the group is re-run and replaced.
        if value
            .get("schema_version")
            .and_then(serde_json::Value::as_u64)
            != Some(SCHEMA_VERSION as u64)
        {
            continue;
        }
        let key = value
            .get("key")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                AppError::Message(format!(
                    "{}:{} has no string `key`",
                    path.display(),
                    line_number + 1
                ))
            })?;
        keys.insert(key.to_owned());
    }
    Ok(keys)
}

fn pure_tc_config(niters: usize) -> TreeSA {
    TreeSA {
        betas: (0..BETA_LEVELS)
            .map(|index| 0.01 + 0.05 * index as f64)
            .collect(),
        ntrials: 1,
        niters,
        score: ScoreFunction::time_optimized(),
        preprocess: false,
        surgery_iters: 0,
        surgery_probability: 0.0,
        ..TreeSA::default()
    }
}

fn base_niters(visits: u64, n: usize) -> usize {
    let internal_nodes = n.saturating_sub(1).max(1) as f64;
    ((visits as f64 / (BETA_LEVELS as f64 * internal_nodes)).round() as usize).max(1)
}

fn planned_visits(niters: usize, n: usize) -> u64 {
    BETA_LEVELS
        .saturating_mul(niters as u64)
        .saturating_mul(n.saturating_sub(1) as u64)
}

struct Prepared {
    original: Instance,
    code: EinCode<usize>,
    subtrees: Option<Vec<NestedEinsum<usize>>>,
}

fn prepare(instance: Instance, raw: bool) -> Prepared {
    if raw {
        return Prepared {
            code: instance.code.clone(),
            original: instance,
            subtrees: None,
        };
    }
    let simplified = simplify(&instance.code, &instance.sizes);
    Prepared {
        original: instance,
        code: simplified.code,
        subtrees: Some(simplified.subtrees),
    }
}

fn full_tree(prepared: &Prepared, reduced: &NestedEinsum<usize>) -> NestedEinsum<usize> {
    prepared
        .subtrees
        .as_ref()
        .map_or_else(|| reduced.clone(), |subtrees| splice(reduced, subtrees))
}

fn guard_post_splice_rounds(
    prepared: &Prepared,
    baseline: &NestedEinsum<usize>,
    candidate: &NestedEinsum<usize>,
) -> (NestedEinsum<usize>, bool) {
    let baseline = full_tree(prepared, baseline);
    let candidate = full_tree(prepared, candidate);
    let tc = |tree: &NestedEinsum<usize>| {
        contraction_complexity(tree, &prepared.original.sizes, &prepared.original.code.ixs).tc
    };
    if tc(&candidate) < tc(&baseline) {
        (candidate, false)
    } else {
        (baseline, true)
    }
}

struct GroupContext<'a> {
    prepared: &'a Prepared,
    label: String,
    relabel_seed: u64,
    optimizer_seed: u64,
    target_visits: u64,
    raw: bool,
}

fn make_row(
    context: &GroupContext<'_>,
    arm: String,
    full: &NestedEinsum<usize>,
    niters: usize,
    wall_seconds: f64,
    planned: u64,
    rounds: Option<(u64, &RoundsOptions, &omeco::treesa::RoundsReport, bool)>,
) -> ResultRow {
    let cc = contraction_complexity(
        full,
        &context.prepared.original.sizes,
        &context.prepared.original.code.ixs,
    );
    let (
        round_count,
        surgery,
        rebuild,
        scope,
        fine_sweeps,
        accepted,
        trace,
        post_splice_guard_triggered,
    ) = rounds.map_or(
        (None, None, None, None, 0, 0, Vec::new(), false),
        |(count, opts, report, guard_triggered)| {
            let rebuild = match opts.rebuild {
                RebuildMode::Greedy => "greedy",
                RebuildMode::WarmRestricted => "warm_restricted",
            };
            let scope = match opts.scope {
                SurgeryScope::Root => "root",
                SurgeryScope::Local => "local",
            };
            let trace = report
                .round_trace
                .iter()
                .map(|item| TraceRow {
                    round: item.round,
                    tc_before: item.tc_before,
                    tc_after_surgery: item.tc_after_surgery,
                    tc_after_anneal: item.tc_after_anneal,
                    tc_retained: item.tc_retained,
                    surgery_accepted: item.surgery_accepted,
                })
                .collect();
            (
                Some(count),
                Some(opts.surgery),
                Some(rebuild),
                Some(scope),
                report.fine_tune_sweeps_total,
                report
                    .round_trace
                    .iter()
                    .filter(|item| item.surgery_accepted)
                    .count() as u64,
                trace,
                guard_triggered,
            )
        },
    );
    let fine_visits =
        fine_sweeps.saturating_mul(context.prepared.code.num_tensors().saturating_sub(1) as u64);
    ResultRow {
        key: result_key(
            &context.prepared.original.name,
            &context.label,
            &arm,
            context.raw,
            context.target_visits,
        ),
        schema_version: SCHEMA_VERSION,
        instance: context.prepared.original.name.clone(),
        label: context.label.clone(),
        arm,
        params: Params {
            raw: context.raw,
            relabel_seed: context.relabel_seed,
            optimizer_seed: context.optimizer_seed,
            beta_levels: BETA_LEVELS as usize,
            niters,
            target_visits: context.target_visits,
            rounds: round_count,
            surgery,
            rebuild,
            scope,
            n_original: context.prepared.original.code.num_tensors(),
            n_optimized: context.prepared.code.num_tensors(),
        },
        tc: cc.tc,
        sc: cc.sc,
        rwc: cc.rwc,
        wall_seconds,
        planned_visits: planned,
        fine_tune_sweeps: fine_sweeps,
        total_node_visits: planned.saturating_add(fine_visits),
        accepted_rebuilds: accepted,
        post_splice_guard_triggered,
        round_trace: trace,
    }
}

fn run_round_arm(
    context: &GroupContext<'_>,
    baseline: &NestedEinsum<usize>,
    baseline_niters: usize,
    baseline_wall: f64,
    rounds: u64,
    opts: RoundsOptions,
    arm: String,
) -> ResultRow {
    let config = pure_tc_config(baseline_niters);
    let started = Instant::now();
    let (tree, report) = anneal_refine_rounds(
        baseline,
        &context.prepared.code,
        &context.prepared.original.sizes,
        &config,
        rounds,
        &opts,
    );
    let (reported, post_splice_guard_triggered) =
        guard_post_splice_rounds(context.prepared, baseline, &tree);
    make_row(
        context,
        arm,
        &reported,
        baseline_niters,
        baseline_wall + started.elapsed().as_secs_f64(),
        planned_visits(baseline_niters, context.prepared.code.num_tensors()),
        Some((rounds, &opts, &report, post_splice_guard_triggered)),
    )
}

fn run_group(prepared: Prepared, label_index: usize, args: &Args) -> AppResult<Vec<ResultRow>> {
    let relabel_seed = 5400 + 2 * label_index as u64;
    let optimizer_seed = 7000 + 2 * label_index as u64;
    let label = format!("r{label_index}");
    let n = prepared.code.num_tensors();
    if n == 0 {
        return Err(AppError::Message(format!(
            "{} simplifies to an empty network",
            prepared.original.name
        )));
    }
    let niters = base_niters(args.visits, n);
    let context = GroupContext {
        prepared: &prepared,
        label,
        relabel_seed,
        optimizer_seed,
        target_visits: args.visits,
        raw: args.raw,
    };
    let config = pure_tc_config(niters);
    let baseline_started = Instant::now();
    let baseline = optimize_treesa_seeded(
        &prepared.code,
        &prepared.original.sizes,
        &config,
        optimizer_seed,
    )
    .ok_or_else(|| AppError::Message("TreeSA returned no tree".to_owned()))?;
    let baseline_wall = baseline_started.elapsed().as_secs_f64();
    let mut rows = Vec::new();

    if args.set.includes_a() {
        let baseline_full = full_tree(&prepared, &baseline);
        rows.push(make_row(
            &context,
            "treesa_x1".to_owned(),
            &baseline_full,
            niters,
            baseline_wall,
            planned_visits(niters, n),
            None,
        ));
        for scale in [2_usize, 4, 8] {
            let scaled_niters = niters.saturating_mul(scale);
            let started = Instant::now();
            let tree = optimize_treesa_seeded(
                &prepared.code,
                &prepared.original.sizes,
                &pure_tc_config(scaled_niters),
                optimizer_seed,
            )
            .ok_or_else(|| AppError::Message("TreeSA returned no tree".to_owned()))?;
            let full = full_tree(&prepared, &tree);
            rows.push(make_row(
                &context,
                format!("treesa_x{scale}"),
                &full,
                scaled_niters,
                started.elapsed().as_secs_f64(),
                planned_visits(scaled_niters, n),
                None,
            ));
        }
    }

    for &round_count in &args.rounds {
        let cold_opts = RoundsOptions {
            surgery: false,
            ..RoundsOptions::default()
        };
        let cold = run_round_arm(
            &context,
            &baseline,
            niters,
            baseline_wall,
            round_count,
            cold_opts,
            format!("cold_only_r{round_count}"),
        );
        let work_matched = args.set.includes_a().then(|| {
            let mut row = cold.clone();
            row.arm = format!("treesa_x1+cold{round_count}");
            row.key = result_key(
                &context.prepared.original.name,
                &context.label,
                &row.arm,
                context.raw,
                context.target_visits,
            );
            row
        });
        if args.set.includes_b() {
            rows.push(cold);
        }
        if let Some(row) = work_matched {
            rows.push(row);
        }
        if args.set.includes_b() {
            for (name, rebuild, scope) in [
                ("surg_greedy_root", RebuildMode::Greedy, SurgeryScope::Root),
                (
                    "surg_warm_root",
                    RebuildMode::WarmRestricted,
                    SurgeryScope::Root,
                ),
                (
                    "surg_greedy_local",
                    RebuildMode::Greedy,
                    SurgeryScope::Local,
                ),
                (
                    "surg_warm_local",
                    RebuildMode::WarmRestricted,
                    SurgeryScope::Local,
                ),
            ] {
                rows.push(run_round_arm(
                    &context,
                    &baseline,
                    niters,
                    baseline_wall,
                    round_count,
                    RoundsOptions {
                        surgery: true,
                        rebuild,
                        scope,
                        schedule: RoundsSchedule::Cold,
                    },
                    format!("{name}_r{round_count}"),
                ));
            }
        }
    }
    Ok(rows)
}

/// Rewrite `path` dropping every row whose key is in `keys`. Used before
/// appending fresh rows so a stale-schema or duplicate row can never shadow
/// the newer data for the same key.
fn remove_keys(path: &Path, keys: &HashSet<String>) -> AppResult<()> {
    if !path.exists() || keys.is_empty() {
        return Ok(());
    }
    let text = fs::read_to_string(path).map_err(|error| io_error(path, error))?;
    let mut kept = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value =
            serde_json::from_str(line).map_err(|source| AppError::Json {
                path: path.to_path_buf(),
                source,
            })?;
        let key = value
            .get("key")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                AppError::Message(format!(
                    "{} has a row without a string `key`",
                    path.display()
                ))
            })?;
        if !keys.contains(key) {
            kept.push(line.to_owned());
        }
    }
    let mut file = File::create(path).map_err(|error| io_error(path, error))?;
    for line in kept {
        writeln!(file, "{line}").map_err(|error| io_error(path, error))?;
    }
    file.flush().map_err(|error| io_error(path, error))?;
    Ok(())
}

fn append_rows(path: &Path, rows: &[ResultRow], existing: &mut HashSet<String>) -> AppResult<()> {
    let fresh: Vec<&ResultRow> = rows
        .iter()
        .filter(|row| !existing.contains(&row.key))
        .collect();
    if fresh.is_empty() {
        return Ok(());
    }
    let fresh_keys: HashSet<String> = fresh.iter().map(|row| row.key.clone()).collect();
    remove_keys(path, &fresh_keys)?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| io_error(path, error))?;
    for row in fresh {
        let mut line = serde_json::to_vec(row).map_err(|source| AppError::Json {
            path: path.to_path_buf(),
            source,
        })?;
        line.push(b'\n');
        file.write_all(&line)
            .map_err(|error| io_error(path, error))?;
        file.flush().map_err(|error| io_error(path, error))?;
        existing.insert(row.key.clone());
    }
    Ok(())
}

fn stack_size(num_tensors: usize) -> usize {
    num_tensors
        .saturating_mul(4 * 1024)
        .clamp(32 * 1024 * 1024, 1024 * 1024 * 1024)
}

fn run_serial(args: &Args) -> AppResult<()> {
    let instances = load_instances(args)?;
    let resume_path = args.resume_from.as_deref().unwrap_or(&args.out);
    let mut existing = existing_keys(resume_path)?;
    if args.out != resume_path {
        existing.extend(existing_keys(&args.out)?);
    }
    let total = instances.len().saturating_mul(args.labels);
    let started = Instant::now();
    let mut completed = 0_usize;
    for (instance_index, instance) in instances.into_iter().enumerate() {
        for label_index in 0..args.labels {
            let group_index = instance_index * args.labels + label_index;
            if group_index % args.shard_count != args.shard_index {
                continue;
            }
            let label = format!("r{label_index}");
            let complete = expected_arms(args).iter().all(|arm| {
                existing.contains(&result_key(
                    &instance.name,
                    &label,
                    arm,
                    args.raw,
                    args.visits,
                ))
            });
            if complete {
                eprintln!(
                    "[{}/{}] {} {} already complete",
                    group_index + 1,
                    total,
                    instance.name,
                    label
                );
                continue;
            }
            let relabel_seed = 5400 + 2 * label_index as u64;
            let relabeled = relabel_instance(&instance, relabel_seed);
            let prepared = prepare(relabeled, args.raw);
            let n = prepared.code.num_tensors();
            let child_args = args.clone();
            let worker = std::thread::Builder::new()
                .name("surgery-ablation".to_owned())
                .stack_size(stack_size(n))
                .spawn(move || run_group(prepared, label_index, &child_args))
                .map_err(|error| AppError::Message(format!("cannot spawn worker: {error}")))?;
            let rows = worker.join().map_err(|_| AppError::WorkerPanic)??;
            append_rows(&args.out, &rows, &mut existing)?;
            completed += 1;
            let elapsed = started.elapsed().as_secs_f64();
            let remaining = total.saturating_sub(group_index + 1);
            let eta = if completed > 0 {
                elapsed / completed as f64 * remaining as f64
            } else {
                0.0
            };
            eprintln!(
                "[{}/{}] {} r{} done in {:.1}s; ETA {:.1} min",
                group_index + 1,
                total,
                instance.name,
                label_index,
                elapsed,
                eta / 60.0
            );
        }
    }
    Ok(())
}

fn child_command(args: &Args, part: &Path, index: usize) -> AppResult<Command> {
    let executable = std::env::current_exe()
        .map_err(|error| AppError::Message(format!("cannot locate current executable: {error}")))?;
    let mut command = Command::new(executable);
    command
        .arg("--instances")
        .arg(&args.instances)
        .arg("--out")
        .arg(part)
        .arg("--labels")
        .arg(args.labels.to_string())
        .arg("--rounds")
        .arg(
            args.rounds
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(","),
        )
        .arg("--set")
        .arg(args.set.as_str())
        .arg("--visits")
        .arg(args.visits.to_string())
        .arg("--shard-index")
        .arg(index.to_string())
        .arg("--shard-count")
        .arg(args.jobs.to_string())
        .arg("--resume-from")
        .arg(&args.out);
    if args.raw {
        command.arg("--raw");
    }
    if let Some(only) = &args.only {
        command.arg("--only").arg(only.join(","));
    }
    Ok(command)
}

fn run_parallel(args: &Args) -> AppResult<()> {
    let process_id = std::process::id();
    let mut children = Vec::new();
    let mut parts = Vec::new();
    for index in 0..args.jobs {
        let part = args
            .out
            .with_extension(format!("part.{process_id}.{index}.jsonl"));
        File::create(&part).map_err(|error| io_error(&part, error))?;
        let child = child_command(args, &part, index)?
            .spawn()
            .map_err(|error| AppError::Message(format!("cannot spawn shard {index}: {error}")))?;
        children.push(child);
        parts.push(part);
    }
    for mut child in children {
        let status = child
            .wait()
            .map_err(|error| AppError::Message(format!("cannot wait for worker: {error}")))?;
        if !status.success() {
            return Err(AppError::Worker(status));
        }
    }
    let mut existing = existing_keys(&args.out)?;
    let mut fresh_lines = Vec::new();
    for part in &parts {
        let file = File::open(part).map_err(|error| io_error(part, error))?;
        for line in BufReader::new(file).lines() {
            let line = line.map_err(|error| io_error(part, error))?;
            if line.trim().is_empty() {
                continue;
            }
            let row: ResultRowOwned =
                serde_json::from_str(&line).map_err(|source| AppError::Json {
                    path: part.clone(),
                    source,
                })?;
            if existing.insert(row.key) {
                fresh_lines.push(line);
            }
        }
        fs::remove_file(part).map_err(|error| io_error(part, error))?;
    }
    let fresh_keys: HashSet<String> = fresh_lines
        .iter()
        .filter_map(|line| {
            serde_json::from_str::<ResultRowOwned>(line)
                .ok()
                .map(|row| row.key)
        })
        .collect();
    remove_keys(&args.out, &fresh_keys)?;
    let mut out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.out)
        .map_err(|error| io_error(&args.out, error))?;
    for line in fresh_lines {
        writeln!(out, "{line}").map_err(|error| io_error(&args.out, error))?;
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct ResultRowOwned {
    key: String,
}

fn real_main() -> AppResult<()> {
    let args = parse_args_from(std::env::args_os().skip(1))?;
    if args.jobs > 1 && args.shard_count == 1 {
        run_parallel(&args)
    } else {
        run_serial(&args)
    }
}

fn main() {
    if let Err(error) = real_main() {
        eprintln!("surgery_ablation: error: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relabeling_is_deterministic_and_preserves_dimensions() {
        let instance = Instance {
            name: "tiny".to_owned(),
            code: EinCode::new(vec![vec![10, 20], vec![20, 30], vec![30, 40]], vec![10, 40]),
            sizes: [(10, 2), (20, 3), (30, 5), (40, 7)].into(),
        };
        let first = relabel_instance(&instance, 5400);
        let again = relabel_instance(&instance, 5400);
        let other = relabel_instance(&instance, 5402);
        assert_eq!(first.code, again.code);
        assert_eq!(first.sizes, again.sizes);
        assert_ne!(first.code, other.code);
        let mut dimensions = first.sizes.values().copied().collect::<Vec<_>>();
        dimensions.sort_unstable();
        assert_eq!(dimensions, vec![2, 3, 5, 7]);
    }

    #[test]
    fn visit_budget_rounding_matches_protocol() {
        let niters = base_niters(DEFAULT_VISITS, 101);
        assert_eq!(niters, 4667);
        assert_eq!(planned_visits(niters, 101), 140_010_000);
        assert_eq!(base_niters(1, 1), 1);
    }

    #[test]
    fn result_keys_include_protocol_switches() {
        assert_ne!(
            result_key("petersen", "r0", "treesa_x1", false, 100),
            result_key("petersen", "r0", "treesa_x1", true, 100)
        );
        assert_ne!(
            result_key("petersen", "r0", "treesa_x1", false, 100),
            result_key("petersen", "r0", "treesa_x1", false, 200)
        );
    }

    #[test]
    fn post_splice_guard_keeps_baseline_when_rounds_candidate_is_worse() {
        let code = EinCode::new(
            vec![vec![0usize, 3], vec![0, 1], vec![1, 2], vec![2, 3]],
            vec![],
        );
        let sizes = code
            .unique_labels()
            .into_iter()
            .map(|label| (label, 2))
            .collect();
        let pair = |left: usize, right: usize, output: Vec<usize>| {
            NestedEinsum::node(
                vec![NestedEinsum::leaf(left), NestedEinsum::leaf(right)],
                EinCode::new(
                    vec![code.ixs[left].clone(), code.ixs[right].clone()],
                    output,
                ),
            )
        };
        let baseline = NestedEinsum::node(
            vec![pair(0, 1, vec![1, 3]), pair(2, 3, vec![1, 3])],
            EinCode::new(vec![vec![1, 3], vec![1, 3]], vec![]),
        );
        let worse_candidate = NestedEinsum::node(
            vec![pair(0, 2, vec![0, 1, 2, 3]), pair(1, 3, vec![0, 1, 2, 3])],
            EinCode::new(vec![vec![0, 1, 2, 3], vec![0, 1, 2, 3]], vec![]),
        );
        let prepared = Prepared {
            original: Instance {
                name: "ring".to_owned(),
                code: code.clone(),
                sizes,
            },
            code,
            subtrees: Some((0..4).map(NestedEinsum::leaf).collect()),
        };

        let (selected, triggered) =
            guard_post_splice_rounds(&prepared, &baseline, &worse_candidate);
        assert!(triggered);
        assert_eq!(
            omeco::json::to_json_string(&selected).unwrap(),
            omeco::json::to_json_string(&baseline).unwrap()
        );

        let (selected, triggered) =
            guard_post_splice_rounds(&prepared, &worse_candidate, &baseline);
        assert!(!triggered);
        assert_eq!(
            omeco::json::to_json_string(&selected).unwrap(),
            omeco::json::to_json_string(&baseline).unwrap()
        );
    }
}
