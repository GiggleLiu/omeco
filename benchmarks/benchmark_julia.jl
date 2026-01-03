#!/usr/bin/env julia
"""
Benchmark TreeSA contraction order optimization in Julia.
Uses OMEinsumContractionOrders.jl
"""

using OMEinsumContractionOrders
using OMEinsum
using Printf

# Test cases: increasingly complex tensor networks
function chain_network(n::Int, d::Int)
    """Matrix chain of n matrices"""
    labels = [Symbol("i$i") for i in 1:n+1]
    ixs = [[labels[i], labels[i+1]] for i in 1:n]
    iy = [labels[1], labels[end]]
    sizes = Dict(l => d for l in labels)
    return ixs, iy, sizes
end

function grid_network(rows::Int, cols::Int, d::Int)
    """2D grid tensor network (like PEPS)"""
    function edge_h(r, c)
        return Symbol("h$(r)_$(c)")  # horizontal
    end
    function edge_v(r, c)
        return Symbol("v$(r)_$(c)")  # vertical
    end
    
    ixs = Vector{Vector{Symbol}}()
    sizes = Dict{Symbol, Int}()
    
    for r in 1:rows
        for c in 1:cols
            tensor_ixs = Symbol[]
            # Left edge
            if c > 1
                e = edge_h(r, c-1)
                push!(tensor_ixs, e)
                sizes[e] = d
            end
            # Right edge  
            if c < cols
                e = edge_h(r, c)
                push!(tensor_ixs, e)
                sizes[e] = d
            end
            # Top edge
            if r > 1
                e = edge_v(r-1, c)
                push!(tensor_ixs, e)
                sizes[e] = d
            end
            # Bottom edge
            if r < rows
                e = edge_v(r, c)
                push!(tensor_ixs, e)
                sizes[e] = d
            end
            push!(ixs, tensor_ixs)
        end
    end
    
    iy = Symbol[]  # scalar output
    return ixs, iy, sizes
end

function run_benchmark(name::String, ixs, iy, sizes; ntrials=10, niters=50)
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
    
    return (greedy_avg=greedy_time/10*1000, treesa_avg=treesa_time/3*1000, greedy_tc=greedy_cc.tc, treesa_tc=treesa_cc.tc)
end

function main()
    println()
    println("Julia TreeSA Benchmark")
    println("OMEinsumContractionOrders.jl")
    println("=" ^ 60)
    println()
    
    results = Dict{String, NamedTuple}()
    
    # Small: matrix chain
    ixs, iy, sizes = chain_network(10, 100)
    results["chain_10"] = run_benchmark("Matrix Chain (n=10)", ixs, iy, sizes)
    
    # Medium: small grid
    ixs, iy, sizes = grid_network(4, 4, 2)
    results["grid_4x4"] = run_benchmark("Grid 4x4", ixs, iy, sizes, ntrials=10, niters=100)
    
    # Large: bigger grid
    ixs, iy, sizes = grid_network(5, 5, 2)
    results["grid_5x5"] = run_benchmark("Grid 5x5", ixs, iy, sizes, ntrials=10, niters=100)
    
    # Summary
    println("=" ^ 60)
    println("Summary (Julia):")
    println("-" ^ 60)
    println("Problem         Greedy (ms)     TreeSA (ms)    ")
    println("-" ^ 60)
    for name in ["chain_10", "grid_4x4", "grid_5x5"]
        r = results[name]
        @printf("%-15s %-15.3f %-15.2f\n", name, r.greedy_avg, r.treesa_avg)
    end
end

main()
