# Densify the Pareto figure: a wide ladder of Julia optimizer configs on the
# two figure instances, one measured run each, appended to a resumable JSONL.
# Every config is wrapped in try/catch so unsupported kwargs or crashes skip.
#
# Usage: julia --project=/root/OMEinsumContractionOrdersBenchmark pareto_sweep.jl \
#          ~/omeco_campaign/targets ~/omeco_campaign/pareto_sweep.jsonl

using OMEinsumContractionOrders, OMEinsumContractionOrders.JSON, KaHyPar, Metis
using OMEinsumContractionOrders: MF, AMF, MMD, LexBFS, BFS, MCS, RCMMD, RCMGL, MCSM

const INSTANCES = ["sycamore_53_20_0", "surfacecode_d21"]

function configs()
    c = Any[]
    push!(c, ("greedy-default", () -> GreedyMethod()))
    push!(c, ("greedy-nrepeat10", () -> GreedyMethod(nrepeat=10)))
    push!(c, ("greedy-nrepeat40", () -> GreedyMethod(nrepeat=40)))
    for (tag, niters, betas) in [
        ("i1-fastbeta", 1, 0.1:0.3:15.0),
        ("i5-fastbeta", 5, 0.1:0.3:15.0),
        ("i5", 5, 0.01:0.05:15.0),
        ("i10", 10, 0.01:0.05:15.0),
        ("i20", 20, 0.01:0.05:15.0),
    ]
        push!(c, ("treesa-scInf-n1-$tag",
                  () -> TreeSA(sc_target=1000.0, ntrials=1, niters=niters, βs=betas)))
    end
    push!(c, ("treesa-scInf-n2-i10",
              () -> TreeSA(sc_target=1000.0, ntrials=2, niters=10, βs=0.01:0.05:15.0)))
    push!(c, ("treesa-sc20-n1-i5",
              () -> TreeSA(sc_target=20.0, ntrials=1, niters=5, βs=0.01:0.05:15.0)))
    push!(c, ("treesa-sc20-n1-i20",
              () -> TreeSA(sc_target=20.0, ntrials=1, niters=20, βs=0.01:0.05:15.0)))
    for alg in [("LexBFS", LexBFS), ("BFS", BFS), ("MCS", MCS),
                ("RCMMD", RCMMD), ("RCMGL", RCMGL), ("MCSM", MCSM)]
        push!(c, ("treewidth-$(alg[1])", () -> Treewidth(alg=alg[2]())))
    end
    return c
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
            push!(done, (d["instance"], d["config"]))
        end
    end
    for inst in INSTANCES
        js = JSON.parsefile(joinpath(dir, "$inst.json"))
        ixs = [Vector{Int}(ix) for ix in js["ixs"]]
        code = OMEinsumContractionOrders.EinCode(ixs, Int[])
        sizes = Dict(Base.parse(Int, k) => Int(v) for (k, v) in js["sizes"])
        for (label, mk) in configs()
            (inst, label) in done && continue
            opt = try
                mk()
            catch e
                @warn "construct failed" label exception = e
                continue
            end
            warmup(opt)
            @info "Running: $inst with $label"
            flush(stderr)
            row = try
                t = @elapsed optcode = optimize_code(code, sizes, opt)
                cc = OMEinsumContractionOrders.contraction_complexity(optcode, sizes)
                Dict("instance" => inst, "config" => label, "tc" => cc.tc,
                     "sc" => cc.sc, "time_elapsed" => t,
                     "host" => "huawei-ecs-2core")
            catch e
                @warn "failed" inst label exception = e
                Dict("instance" => inst, "config" => label,
                     "error" => sprint(showerror, e))
            end
            open(outpath, "a") do f
                println(f, JSON.json(row))
            end
            flush(stderr)
        end
    end
    println("PARETO-SWEEP-DONE")
end

main(ARGS[1], ARGS[2])
