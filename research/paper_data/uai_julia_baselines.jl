# Julia baselines for the UAI-2014 MAR hard-instance batch (huawei).
# Reads the validator-schema instance JSONs and runs:
#   * TensorInference.jl's documented default: TreeSA(ntrials=1, niters=5,
#     betas=0.1:0.1:100)  [what an inference user gets out of the box]
#   * matched-budget ladder rungs: TreeSA sc_target in {20, 1e3}, ntrials in
#     {1, 4}, niters=50, betas=0.01:0.05:15
#   * HyperND, Treewidth(MF)
# Appends JSONL rows {instance, optimizer, tc, sc, time_elapsed}. Resumable:
# existing (instance, optimizer) pairs are skipped.
#
# Usage: julia --project=/root/OMEinsumContractionOrdersBenchmark \
#          uai_julia_baselines.jl <instances_dir> <out.jsonl>

using OMEinsumContractionOrders, OMEinsumContractionOrders.JSON, KaHyPar, Metis
using OMEinsumContractionOrders: MF

const INSTANCES = [
    "uai_DBN_16", "uai_DBN_12", "uai_DBN_14",
    "uai_linkage_15", "uai_linkage_13", "uai_linkage_23", "uai_linkage_17",
    "uai_CSP_11", "uai_Grids_15", "uai_Promedus_14",
]

function optimizers()
    opts = Any[
        ("TI-default-TreeSA", TreeSA(ntrials=1, niters=5, βs=0.1:0.1:100.0)),
        ("HyperND", HyperND()),
        ("Treewidth-MF", Treewidth(alg=MF())),
    ]
    for sct in (20.0, 1000.0), ntrials in (1, 4)
        push!(opts, ("TreeSA-sc$(Int(sct))-nt$(ntrials)",
                     TreeSA(sc_target=sct, ntrials=ntrials, niters=50,
                            βs=0.01:0.05:15.0)))
    end
    return opts
end

function warmup(optimizer)
    code = OMEinsumContractionOrders.EinCode([[1, 2], [2, 3], [3, 4]], [1, 4])
    try
        optimize_code(code, Dict(1 => 2, 2 => 2, 3 => 2, 4 => 2), optimizer)
    catch
    end
end

function main(dir, outpath)
    done = Set{Tuple{String, String}}()
    if isfile(outpath)
        for line in eachline(outpath)
            isempty(strip(line)) && continue
            d = JSON.parse(line)
            push!(done, (d["instance"], d["optimizer"]))
        end
    end
    for inst in INSTANCES
        js = JSON.parsefile(joinpath(dir, "$inst.json"))
        ixs = [Vector{Int}(ix) for ix in js["ixs"]]
        code = OMEinsumContractionOrders.EinCode(ixs, Int[])
        sizes = Dict(Base.parse(Int, k) => Int(v) for (k, v) in js["sizes"])
        for (label, opt) in optimizers()
            (inst, label) in done && continue
            warmup(opt)
            @info "Running: $inst with $label"
            flush(stderr)
            row = try
                t = @elapsed optcode = optimize_code(code, sizes, opt)
                cc = OMEinsumContractionOrders.contraction_complexity(optcode, sizes)
                Dict("instance" => inst, "optimizer" => label, "tc" => cc.tc,
                     "sc" => cc.sc, "time_elapsed" => t,
                     "host" => "huawei-ecs-2core")
            catch e
                @warn "failed" inst label exception = e
                Dict("instance" => inst, "optimizer" => label,
                     "error" => sprint(showerror, e), "host" => "huawei-ecs-2core")
            end
            open(outpath, "a") do f
                println(f, JSON.json(row))
            end
            @info "row: $(JSON.json(row))"
            flush(stderr)
        end
    end
    println("UAI-JULIA-DONE")
end

main(ARGS[1], ARGS[2])
