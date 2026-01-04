#!/usr/bin/env julia
"""
Benchmark TreeSA contraction order optimization in Julia.
Uses OMEinsumContractionOrders.jl
"""

using OMEinsumContractionOrders
using OMEinsum
using Printf
using Random
using JSON

# Test cases: increasingly complex tensor networks
function chain_network(n::Int, d::Int)
    """Matrix chain of n matrices"""
    labels = collect(1:n+1)
    ixs = [[labels[i], labels[i+1]] for i in 1:n]
    iy = [labels[1], labels[end]]
    sizes = Dict(l => d for l in labels)
    return ixs, iy, sizes
end

function grid_network(rows::Int, cols::Int, d::Int)
    """2D grid tensor network (like PEPS)"""
    label = 1
    h_edge_map = Dict{Tuple{Int,Int}, Int}()
    v_edge_map = Dict{Tuple{Int,Int}, Int}()
    
    for r in 1:rows
        for c in 1:cols-1
            h_edge_map[(r, c)] = label
            label += 1
        end
    end
    
    for r in 1:rows-1
        for c in 1:cols
            v_edge_map[(r, c)] = label
            label += 1
        end
    end
    
    ixs = Vector{Vector{Int}}()
    sizes = Dict{Int, Int}()
    
    for r in 1:rows
        for c in 1:cols
            tensor_ixs = Int[]
            # Left edge
            if c > 1
                e = h_edge_map[(r, c-1)]
                push!(tensor_ixs, e)
                sizes[e] = d
            end
            # Right edge  
            if c < cols
                e = h_edge_map[(r, c)]
                push!(tensor_ixs, e)
                sizes[e] = d
            end
            # Top edge
            if r > 1
                e = v_edge_map[(r-1, c)]
                push!(tensor_ixs, e)
                sizes[e] = d
            end
            # Bottom edge
            if r < rows
                e = v_edge_map[(r, c)]
                push!(tensor_ixs, e)
                sizes[e] = d
            end
            push!(ixs, tensor_ixs)
        end
    end
    
    iy = Int[]  # scalar output
    return ixs, iy, sizes
end

function random_regular_graph(n::Int, degree::Int, d::Int; seed::Int=42)
    """
    Random regular graph tensor network.
    Each vertex is a tensor with `degree` indices.
    """
    Random.seed!(seed)
    
    # Generate random regular graph using configuration model
    half_edges = Int[]
    for v in 1:n
        for _ in 1:degree
            push!(half_edges, v)
        end
    end
    
    shuffle!(half_edges)
    
    # Pair up half-edges to form edges
    edge_label = 1
    vertex_edges = Dict{Int, Vector{Int}}(v => Int[] for v in 1:n)
    
    for i in 1:2:length(half_edges)
        v1, v2 = half_edges[i], half_edges[i + 1]
        # Skip self-loops
        if v1 != v2
            push!(vertex_edges[v1], edge_label)
            push!(vertex_edges[v2], edge_label)
            edge_label += 1
        end
    end
    
    ixs = [vertex_edges[v] for v in 1:n if !isempty(vertex_edges[v])]
    sizes = Dict(e => d for e in 1:edge_label-1)
    iy = Int[]  # scalar output
    
    return ixs, iy, sizes
end

function run_benchmark(name::String, ixs, iy, sizes; ntrials=1, niters=50)
    println("=" ^ 60)
    println("Benchmark: $name")
    println("  Tensors: $(length(ixs))")
    println("  Indices: $(length(sizes))")
    println()
    
    # Convert to OMEinsum format
    ixs_tuples = Tuple(Tuple(ix) for ix in ixs)
    iy_tuple = Tuple(iy)
    code = EinCode(ixs_tuples, iy_tuple)
    
    # Greedy warmup + benchmark
    println("GreedyMethod:")
    greedy_opt = GreedyMethod()
    
    # Warmup
    _ = optimize_code(code, sizes, greedy_opt)
    
    greedy_result = nothing
    greedy_time = @elapsed for _ in 1:10
        greedy_result = optimize_code(code, sizes, greedy_opt)
    end
    greedy_cc = contraction_complexity(greedy_result, sizes)
    println("  tc=$(round(greedy_cc.tc, digits=2)), sc=$(round(greedy_cc.sc, digits=2)), rwc=$(round(greedy_cc.rwc, digits=2))")
    println("  Time (10 runs): $(round(greedy_time*1000, digits=2))ms, avg: $(round(greedy_time/10*1000, digits=4))ms")
    println()
    
    # TreeSA  
    println("TreeSA (ntrials=$ntrials, niters=$niters):")
    treesa_opt = TreeSA(ntrials=ntrials, niters=niters, βs=0.01:0.05:15.0)  # Same beta schedule as Rust default
    
    # Warmup
    _ = optimize_code(code, sizes, treesa_opt)
    
    treesa_result = nothing
    treesa_time = @elapsed for _ in 1:3
        treesa_result = optimize_code(code, sizes, treesa_opt)
    end
    treesa_cc = contraction_complexity(treesa_result, sizes)
    println("  tc=$(round(treesa_cc.tc, digits=2)), sc=$(round(treesa_cc.sc, digits=2)), rwc=$(round(treesa_cc.rwc, digits=2))")
    println("  Time (3 runs): $(round(treesa_time*1000, digits=2))ms, avg: $(round(treesa_time/3*1000, digits=2))ms")
    println()
    
    return Dict(
        "tensors" => length(ixs),
        "indices" => length(sizes),
        "ntrials" => ntrials,
        "niters" => niters,
        "greedy_avg_ms" => greedy_time/10*1000,
        "greedy_tc" => greedy_cc.tc,
        "greedy_sc" => greedy_cc.sc,
        "greedy_rwc" => greedy_cc.rwc,
        "treesa_avg_ms" => treesa_time/3*1000,
        "treesa_tc" => treesa_cc.tc,
        "treesa_sc" => treesa_cc.sc,
        "treesa_rwc" => treesa_cc.rwc,
    )
end

function main()
    println()
    println("Julia TreeSA Benchmark")
    println("OMEinsumContractionOrders.jl")
    println("=" ^ 60)
    println()
    
    results = Dict{String, Any}()
    
    # Small: matrix chain
    ixs, iy, sizes = chain_network(10, 100)
    results["chain_10"] = run_benchmark("Matrix Chain (n=10)", ixs, iy, sizes, ntrials=1, niters=50)
    
    # Medium: small grid
    ixs, iy, sizes = grid_network(4, 4, 2)
    results["grid_4x4"] = run_benchmark("Grid 4x4", ixs, iy, sizes, ntrials=1, niters=100)
    
    # Large: bigger grid
    ixs, iy, sizes = grid_network(5, 5, 2)
    results["grid_5x5"] = run_benchmark("Grid 5x5", ixs, iy, sizes, ntrials=1, niters=100)
    
    # Random 3-regular graph n=250
    ixs, iy, sizes = random_regular_graph(250, 3, 2)
    results["reg3_250"] = run_benchmark("Random 3-regular n=250", ixs, iy, sizes, ntrials=1, niters=100)
    
    # Summary
    println("=" ^ 60)
    println("Summary (Julia):")
    println("-" ^ 60)
    @printf("%-20s %-15s %-15s %-12s %-12s\n", "Problem", "Greedy (ms)", "TreeSA (ms)", "Greedy tc", "TreeSA tc")
    println("-" ^ 60)
    for name in ["chain_10", "grid_4x4", "grid_5x5", "reg3_250"]
        r = results[name]
        @printf("%-20s %-15.3f %-15.2f %-12.2f %-12.2f\n", name, r["greedy_avg_ms"], r["treesa_avg_ms"], r["greedy_tc"], r["treesa_tc"])
    end
    
    # Save Greedy results to JSON
    greedy_output = Dict(
        "language" => "julia",
        "backend" => "OMEinsumContractionOrders.jl",
        "method" => "greedy",
        "results" => Dict(name => Dict(
            "tensors" => r["tensors"],
            "indices" => r["indices"],
            "avg_ms" => r["greedy_avg_ms"],
            "tc" => r["greedy_tc"],
            "sc" => r["greedy_sc"],
            "rwc" => r["greedy_rwc"],
        ) for (name, r) in results)
    )
    greedy_path = joinpath(@__DIR__, "results_julia_greedy.json")
    open(greedy_path, "w") do f
        JSON.print(f, greedy_output, 2)
    end
    
    # Save TreeSA results to JSON
    treesa_output = Dict(
        "language" => "julia",
        "backend" => "OMEinsumContractionOrders.jl",
        "method" => "treesa",
        "results" => Dict(name => Dict(
            "tensors" => r["tensors"],
            "indices" => r["indices"],
            "ntrials" => r["ntrials"],
            "niters" => r["niters"],
            "avg_ms" => r["treesa_avg_ms"],
            "tc" => r["treesa_tc"],
            "sc" => r["treesa_sc"],
            "rwc" => r["treesa_rwc"],
        ) for (name, r) in results)
    )
    treesa_path = joinpath(@__DIR__, "results_julia_treesa.json")
    open(treesa_path, "w") do f
        JSON.print(f, treesa_output, 2)
    end
    
    println()
    println("Results saved to:")
    println("  $greedy_path")
    println("  $treesa_path")
end

main()
