# Matched-budget (90 s) Julia OMECO baselines on the paper instances.
# Best-of-restarts within the budget per optimizer; pure-tc objective for
# TreeSA (sc_target = Inf) to match the paper's protocol; HyperND/Treewidth
# are deterministic-ish one-shots looped best-of.
# Usage: julia --project=/Users/liujinguo/jcode/OMEinsumContractionOrdersBenchmark \
#          research/paper_data/julia_matched_budget.jl <instance.json> <budget_s> <out.json>

using OMEinsumContractionOrders, OMEinsumContractionOrders.JSON, KaHyPar
using OMEinsumContractionOrders: MF, MMD, AMF

function load_instance(path)
    js = JSON.parsefile(path)
    ixs = [Vector{Int}(ix) for ix in js["ixs"]]
    iy = Vector{Int}(js["iy"])
    sizes = Dict([(Base.parse(Int, k) => Int(v)) for (k, v) in js["sizes"]])
    return OMEinsumContractionOrders.EinCode(ixs, iy), sizes
end

function best_within(code, sizes, budget_s, mkopt)
    deadline = time() + budget_s
    best_tc, best_sc, runs = Inf, Inf, 0
    while time() < deadline
        optcode = optimize_code(code, sizes, mkopt(runs))
        cc = OMEinsumContractionOrders.contraction_complexity(optcode, sizes)
        runs += 1
        if cc.tc < best_tc
            best_tc, best_sc = cc.tc, cc.sc
        end
    end
    return best_tc, best_sc, runs
end

function main()
    inst_path, budget_s, out_path = ARGS[1], Base.parse(Float64, ARGS[2]), ARGS[3]
    code, sizes = load_instance(inst_path)
    results = Dict{String,Any}()
    # warm up JIT on a tiny budget first
    optimize_code(code, sizes, GreedyMethod())

    for (name, mkopt) in [
        ("TreeSA_scinf", r -> TreeSA(sc_target=1000.0, ntrials=1, niters=30,
                                     βs=0.01:0.05:15.0)),
        ("HyperND", r -> HyperND()),
        ("Treewidth_MF", r -> Treewidth(alg=MF())),
    ]
        try
            tc, sc, runs = best_within(code, sizes, budget_s, mkopt)
            results[name] = Dict("tc" => tc, "sc" => sc, "restarts" => runs)
            @info "$name: tc=$tc sc=$sc restarts=$runs"
        catch e
            results[name] = Dict("error" => sprint(showerror, e))
            @warn "$name failed" exception = e
        end
    end
    open(out_path, "w") do io
        JSON.print(io, Dict("instance" => inst_path, "budget_s" => budget_s,
                            "results" => results), 1)
    end
end

main()
