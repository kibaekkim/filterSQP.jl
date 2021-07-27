using Test

# took this from Ipopt.jl
function runtests(mod)
    for name in names(mod; all = true)
        if !startswith("$(name)", "test_")
            continue
        end
        @testset "$(name)" begin
            getfield(mod, name)()
        end
    end
end

# @testset "NlpExample" begin
#     include("NLP.jl")
# end

@testset "MathOptInterface" begin
    include("MOI_wrapper.jl")
end
