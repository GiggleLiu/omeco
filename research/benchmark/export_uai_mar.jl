# Export UAI-2014 MAR instances from the TensorInference.jl artifact into the
# validator's target JSON schema ({name, description, ixs, iy, sizes}).
#
# Faithful to TensorInference's TensorNetworkModel construction: the einsum is
# [[i] for i in 1:nvars] (unity vertex tensors) + factor scopes, iy = [],
# sizes = FULL cardinalities (evidence is applied at contraction time, not at
# order-optimization time — see src/Core.jl).
#
# Usage: julia --project=$HOME/.julia/dev/TensorInference \
#          research/benchmark/export_uai_mar.jl <outdir>

using TensorInference

const FAMILIES = ["Promedus", "linkage", "Pedigree", "DBN", "Grids", "CSP"]

json_esc(s) = replace(s, "\\" => "\\\\", "\"" => "\\\"")

function write_target(path, name, desc, ixs, sizes)
    open(path, "w") do io
        print(io, "{\"name\": \"", json_esc(name), "\", ")
        print(io, "\"description\": \"", json_esc(desc), "\", ")
        print(io, "\"ixs\": [")
        for (k, ix) in enumerate(ixs)
            k > 1 && print(io, ", ")
            print(io, "[", join(ix, ", "), "]")
        end
        print(io, "], \"iy\": [], \"sizes\": {")
        first = true
        for (k, v) in sort(collect(sizes))
            first || print(io, ", ")
            first = false
            print(io, "\"", k, "\": ", v)
        end
        print(io, "}}")
    end
end

function main(outdir)
    mkpath(outdir)
    artifact_path = TensorInference.get_artifact_path("uai2014")
    mar = joinpath(artifact_path, "MAR")
    rows = []
    for f in sort(readdir(mar))
        endswith(f, ".uai") || continue
        fam = match(r"^([A-Za-z_]+)_\d+\.uai$", f)
        (fam === nothing || !(fam[1] in FAMILIES)) && continue
        name = "uai_" * replace(f, ".uai" => "")
        model = TensorInference.read_model_file(joinpath(mar, f))
        ixs = vcat([[i] for i in 1:model.nvars],
                   [collect(Int, factor.vars) for factor in model.factors])
        sizes = Dict(i => model.cards[i] for i in 1:model.nvars)
        maxcard = maximum(model.cards)
        desc = "UAI-2014 MAR $(replace(f, ".uai" => "")): $(length(ixs)) tensors " *
               "($(model.nvars) unity + $(length(model.factors)) factors), " *
               "$(model.nvars) vars, max card $(maxcard) " *
               "[TensorInference.jl uai2014 artifact, full-card einsum]"
        write_target(joinpath(outdir, "$name.json"), name, desc, ixs, sizes)
        push!(rows, (name, length(ixs), model.nvars, maxcard))
        println("$name  tensors=$(length(ixs)) vars=$(model.nvars) maxcard=$maxcard")
    end
    println("exported $(length(rows)) instances to $outdir")
end

main(length(ARGS) >= 1 ? ARGS[1] : "research/benchmark/uai_mar")
