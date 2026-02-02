#!/usr/bin/env julia
"""
Benchmark TreeSA contraction order optimization in Julia.
Uses OMEinsumContractionOrders.jl with shared graph files.
"""

using OMEinsumContractionOrders
using OMEinsum
using Printf
using JSON

const GRAPHS_DIR = joinpath(@__DIR__, "graphs")
const RESULTS_DIR = joinpath(@__DIR__, "results")

function load_graph(name::String)
    """Load a graph from the graphs directory."""
    path = joinpath(GRAPHS_DIR, "$(name).json")
    data = JSON.parsefile(path)

    ixs = [Int.(ix) .+ 1 for ix in data["ixs"]]  # Julia uses 1-based indexing
    iy = Int.(data["iy"]) .+ 1
    sizes = Dict(parse(Int, k) + 1 => v for (k, v) in data["sizes"])
    description = get(data, "description", name)

    return ixs, iy, sizes, description
end

function run_benchmark(name::String; ntrials=1, niters=50)
    ixs, iy, sizes, description = load_graph(name)

    println("=" ^ 70)
    println("Benchmark: $name")
    println("  Description: $description")
    println("  Tensors: $(length(ixs))")
    println("  Indices: $(length(sizes))")
    println()

    # Convert to OMEinsum format
    ixs_tuples = Tuple(Tuple(ix) for ix in ixs)
    iy_tuple = Tuple(iy)
    code = EinCode(ixs_tuples, iy_tuple)

    # ========== Greedy ==========
    println("GreedyMethod:")
    greedy_opt = GreedyMethod()

    # Warmup
    _ = optimize_code(code, sizes, greedy_opt)

    # Benchmark
    greedy_result = nothing
    greedy_time = @elapsed for _ in 1:10
        greedy_result = optimize_code(code, sizes, greedy_opt)
    end
    greedy_cc = contraction_complexity(greedy_result, sizes)
    @printf("  Time complexity (tc):       %.6f\n", greedy_cc.tc)
    @printf("  Space complexity (sc):      %.6f\n", greedy_cc.sc)
    @printf("  Read-write complexity (rwc): %.6f\n", greedy_cc.rwc)
    @printf("  Execution time (10 runs):   %.2f ms\n", greedy_time * 1000)
    @printf("  Average per run:            %.4f ms\n", greedy_time / 10 * 1000)
    println()

    # ========== TreeSA ==========
    println("TreeSA (ntrials=$ntrials, niters=$niters):")
    treesa_opt = TreeSA(ntrials=ntrials, niters=niters, βs=0.01:0.05:15.0)

    # Warmup
    _ = optimize_code(code, sizes, treesa_opt)

    # Benchmark
    treesa_result = nothing
    treesa_time = @elapsed for _ in 1:3
        treesa_result = optimize_code(code, sizes, treesa_opt)
    end
    treesa_cc = contraction_complexity(treesa_result, sizes)
    @printf("  Time complexity (tc):       %.6f\n", treesa_cc.tc)
    @printf("  Space complexity (sc):      %.6f\n", treesa_cc.sc)
    @printf("  Read-write complexity (rwc): %.6f\n", treesa_cc.rwc)
    @printf("  Execution time (3 runs):    %.2f ms\n", treesa_time * 1000)
    @printf("  Average per run:            %.2f ms\n", treesa_time / 3 * 1000)
    println()

    # ========== Improvement ==========
    tc_improvement = greedy_cc.tc - treesa_cc.tc
    sc_improvement = greedy_cc.sc - treesa_cc.sc
    println("  Improvement over Greedy:")
    @printf("    tc reduction: %.2f (%.1f%%)\n", tc_improvement, tc_improvement / greedy_cc.tc * 100)
    @printf("    sc reduction: %.2f\n", sc_improvement)
    println()

    return Dict(
        "name" => name,
        "description" => description,
        "tensors" => length(ixs),
        "indices" => length(sizes),
        "greedy" => Dict(
            "tc" => greedy_cc.tc,
            "sc" => greedy_cc.sc,
            "rwc" => greedy_cc.rwc,
            "avg_ms" => greedy_time / 10 * 1000,
            "total_ms" => greedy_time * 1000,
            "runs" => 10,
        ),
        "treesa" => Dict(
            "ntrials" => ntrials,
            "niters" => niters,
            "tc" => treesa_cc.tc,
            "sc" => treesa_cc.sc,
            "rwc" => treesa_cc.rwc,
            "avg_ms" => treesa_time / 3 * 1000,
            "total_ms" => treesa_time * 1000,
            "runs" => 3,
        ),
    )
end

function main()
    println()
    println("=" ^ 70)
    println("Julia TreeSA Benchmark")
    println("Backend: OMEinsumContractionOrders.jl")
    println("=" ^ 70)
    println()

    mkpath(RESULTS_DIR)

    results = Dict{String, Any}()

    # Define benchmarks: (graph_name, ntrials, niters)
    benchmarks = [
        ("chain_10", 1, 50),
        ("chain_20", 1, 50),
        ("grid_4x4", 1, 100),
        ("grid_5x5", 1, 100),
        ("grid_6x6", 1, 100),
        ("petersen", 1, 50),
        ("reg3_50", 1, 100),
        ("reg3_100", 1, 100),
        ("reg3_250", 1, 100),
    ]

    for (name, ntrials, niters) in benchmarks
        path = joinpath(GRAPHS_DIR, "$(name).json")
        if !isfile(path)
            println("Skipping $name (graph file not found)")
            continue
        end
        results[name] = run_benchmark(name, ntrials=ntrials, niters=niters)
    end

    # ========== Summary Table ==========
    println("=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println()
    @printf("%-15s %8s %8s │ %10s %10s │ %10s %10s\n",
            "Graph", "Tensors", "Indices", "Greedy tc", "TreeSA tc", "Greedy ms", "TreeSA ms")
    println("─" ^ 15 * "─" ^ 8 * "─" ^ 8 * "─┼" * "─" ^ 10 * "─" ^ 10 * "─┼" * "─" ^ 10 * "─" ^ 10)
    for (name, r) in sort(collect(results), by=x->x[1])
        @printf("%-15s %8d %8d │ %10.2f %10.2f │ %10.3f %10.2f\n",
                name, r["tensors"], r["indices"],
                r["greedy"]["tc"], r["treesa"]["tc"],
                r["greedy"]["avg_ms"], r["treesa"]["avg_ms"])
    end
    println()

    # ========== Save Results ==========
    # Save combined results
    output = Dict(
        "language" => "julia",
        "backend" => "OMEinsumContractionOrders.jl",
        "results" => results,
    )
    output_path = joinpath(RESULTS_DIR, "julia_results.json")
    open(output_path, "w") do f
        JSON.print(f, output, 2)
    end

    # Save separate greedy results (old format compatibility)
    greedy_output = Dict(
        "language" => "julia",
        "backend" => "OMEinsumContractionOrders.jl",
        "method" => "greedy",
        "results" => Dict(
            name => Dict(
                "tensors" => r["tensors"],
                "indices" => r["indices"],
                "tc" => r["greedy"]["tc"],
                "sc" => r["greedy"]["sc"],
                "rwc" => r["greedy"]["rwc"],
                "avg_ms" => r["greedy"]["avg_ms"],
            )
            for (name, r) in results
        ),
    )
    open(joinpath(RESULTS_DIR, "results_julia_greedy.json"), "w") do f
        JSON.print(f, greedy_output, 2)
    end

    # Save separate treesa results (old format compatibility)
    treesa_output = Dict(
        "language" => "julia",
        "backend" => "OMEinsumContractionOrders.jl",
        "method" => "treesa",
        "results" => Dict(
            name => Dict(
                "tensors" => r["tensors"],
                "indices" => r["indices"],
                "ntrials" => r["treesa"]["ntrials"],
                "niters" => r["treesa"]["niters"],
                "tc" => r["treesa"]["tc"],
                "sc" => r["treesa"]["sc"],
                "rwc" => r["treesa"]["rwc"],
                "avg_ms" => r["treesa"]["avg_ms"],
            )
            for (name, r) in results
        ),
    )
    open(joinpath(RESULTS_DIR, "results_julia_treesa.json"), "w") do f
        JSON.print(f, treesa_output, 2)
    end

    println("Results saved to:")
    println("  $output_path")
    println("  $(joinpath(RESULTS_DIR, "results_julia_greedy.json"))")
    println("  $(joinpath(RESULTS_DIR, "results_julia_treesa.json"))")
end

main()
