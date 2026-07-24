# Matched-budget Julia OMECO baselines through the benchmark repo's harness —
# v2, remote (huawei) edition. Changes vs v1:
#   * run_one_fast: same result schema/paramhash/filenames as runner.jl's
#     run_one, but JIT warmup happens ONCE on a tiny einsum instead of running
#     the full instance twice (halves total cost; JIT overhead on the timed
#     run is seconds, negligible vs the 90 s+ budget filter).
#   * budget guard: within each TreeSA ladder (ntrials 1 -> 4 -> 8), skip the
#     higher rung when the extrapolated time cannot fit even the 900 s budget.
#   * instances ordered smallest-first so rows accumulate early.
#
# Usage: julia --project=/root/OMEinsumContractionOrdersBenchmark julia_matched_budget_v2.jl

const BENCH = "/root/OMEinsumContractionOrdersBenchmark"
include(joinpath(BENCH, "runner.jl"))

# --- JIT warmup on a tiny instance, once per optimizer ---------------------
function warmup(optimizer)
    code = OMEinsumContractionOrders.EinCode([[1, 2], [2, 3], [3, 4]], [1, 4])
    sizes = Dict(1 => 2, 2 => 2, 3 => 2, 4 => 2)
    try
        optimize_code(code, sizes, optimizer)
    catch
    end
end

# run_one minus the double-run warmup; identical filename/schema.
function run_one_fast(input_file, optimizer)
    @assert endswith(input_file, ".json")
    _process_labels(ix::Vector) = Vector{Int}(ix)
    js = JSON.parsefile(input_file)
    code = OMEinsumContractionOrders.EinCode(_process_labels.(js["einsum"]["ixs"]),
                                             _process_labels(js["einsum"]["iy"]))
    sizes = Dict([(Base.parse(Int, k) => Int(v)) for (k, v) in js["size"]])
    filename = joinpath(dirname(dirname(input_file)), "results",
                        "$(paramhash((input_file, optimizer))).json")
    if isfile(filename)
        @info "Skipping (exists): $(basename(filename))"
        return -1.0
    end
    mkpath(dirname(filename))
    @info "Running: $(input_file) with $(optimizer)"
    flush(stderr)
    time_elapsed = @elapsed optcode = optimize_code(code, sizes, optimizer)
    cc = OMEinsumContractionOrders.contraction_complexity(optcode, sizes)
    @info "Contraction complexity: $(cc), time cost: $(time_elapsed)s, saving to: $(filename)"
    flush(stderr)
    out = JSON.json(Dict(
        "instance" => input_file,
        "optimizer" => string(typeof(optimizer).name.name),
        "optimizer_config" => optimizer,
        "contraction_complexity" => cc,
        "time_elapsed" => time_elapsed,
    ))
    open(filename, "w") do f
        JSON.write(f, out)
    end
    return time_elapsed
end

treesa(ntrials, sct) = TreeSA(sc_target=sct, ntrials=ntrials, niters=50,
                              βs=0.01:0.05:15.0)

# Smallest-first so rows accumulate early.
const PAPER_INSTANCES = [
    ("einsumorg", "qc_qft_27.json"),
    ("inference", "DBN_13.json"),
    ("qec", "surfacecode_d=9.json"),
    ("qec", "surfacecode_d=13.json"),
    ("qec", "surfacecode_d=17.json"),
    ("qec", "surfacecode_d=21.json"),
    ("quantumcircuit", "sycamore_53_20_0.json"),
    ("nqueens", "nqueens_n=28.json"),
    ("independentset", "ksg.json"),
]

const MAX_USEFUL_S = 1100.0  # nothing above the 900 s budget tier is usable

for (problem, inst) in PAPER_INSTANCES
    path = joinpath(BENCH, "examples", problem, "codes", inst)
    # cheap non-SA baselines first
    for opt in Any[HyperND(), Treewidth(alg=MF()), Treewidth(alg=AMF()),
                   Treewidth(alg=MMD())]
        warmup(opt)
        try
            run_one_fast(path, opt)
        catch e
            @warn "failed" inst opt exception = e
            flush(stderr)
        end
    end
    # TreeSA ladders with budget guard, per sc_target
    for sct in (20.0, 1000.0)
        t1 = 0.0
        for ntrials in (1, 4, 8)
            est = t1 * ntrials  # extrapolate from the ntrials=1 rung
            if ntrials > 1 && est > MAX_USEFUL_S
                @info "SKIP $(inst) TreeSA ntrials=$(ntrials) sc_target=$(sct): est $(round(est))s > $(MAX_USEFUL_S)s"
                flush(stderr)
                continue
            end
            opt = treesa(ntrials, sct)
            warmup(opt)
            t = try
                run_one_fast(path, opt)
            catch e
                @warn "failed" inst opt exception = e
                flush(stderr)
                nothing
            end
            if ntrials == 1 && t isa Real && t > 0
                t1 = t
            end
        end
    end
end
println("JULIA-LADDER-DONE")
