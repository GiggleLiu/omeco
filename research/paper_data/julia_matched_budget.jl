# Matched-budget Julia OMECO baselines, run THROUGH the benchmark repo's own
# harness (runner.jl / run_one) so results land in examples/*/results/ in the
# published schema and are directly comparable to the published rows.
#
# Design: a cost ladder of configs per optimizer; the analysis step selects,
# per optimizer, the best tc among rows with time_elapsed <= the paper budget
# (90 s). Includes TreeSA at BOTH the shipped sc_target=20 default and the
# pure-tc setting (sc_target = 1e3 ~ unbounded) — measuring the sc_target
# cliff on the reference Julia implementation itself.
#
# Usage:
#   julia --project=/Users/liujinguo/jcode/OMEinsumContractionOrdersBenchmark \
#     research/paper_data/julia_matched_budget.jl

const BENCH = "/Users/liujinguo/jcode/OMEinsumContractionOrdersBenchmark"
include(joinpath(BENCH, "runner.jl"))

function paper_optimizers()
    opts = Any[]
    for ntrials in (1, 4, 8), sct in (20.0, 1000.0)
        push!(opts, TreeSA(sc_target=sct, ntrials=ntrials, niters=50,
                           βs=0.01:0.05:15.0))
    end
    push!(opts, HyperND())
    for alg in (MF(), AMF(), MMD())
        push!(opts, Treewidth(alg=alg))
    end
    return opts
end

# Paper instances only (superset of config.toml; d=9..17 for the family trend)
const PAPER_INSTANCES = [
    ("quantumcircuit", "sycamore_53_20_0.json"),
    ("qec", "surfacecode_d=21.json"),
    ("qec", "surfacecode_d=17.json"),
    ("qec", "surfacecode_d=13.json"),
    ("qec", "surfacecode_d=9.json"),
    ("independentset", "ksg.json"),
    ("inference", "DBN_13.json"),
    ("nqueens", "nqueens_n=28.json"),
    ("einsumorg", "qc_qft_27.json"),
]

for (problem, inst) in PAPER_INSTANCES
    for opt in paper_optimizers()
        try
            run_one(joinpath(BENCH, "examples", problem, "codes", inst), opt)
        catch e
            @warn "failed" inst opt exception = e
        end
    end
end
