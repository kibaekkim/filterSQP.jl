module TestJuMPExample

using filterSQP
using JuMP
using Test

function test_nlp1()
    u = [2.0, 2.0, 1.0]
    m = Model(filterSQP.Optimizer)
    set_optimizer_attribute(m, "iprint", 0)
    set_optimizer_attribute(m, "use_warm_start", true)

    @variable(m, 0 <= x[i=1:3] <= u[i])
    @variable(m, 0 <= y[i=1:3] <= 1)

    @NLobjective(m, Min, 5*y[1] + 6*y[2] + 8*y[3]
        + 10*x[1] - 7*x[3] - 18*log(x[2]+1)
        - 19.2*log(x[1]-x[2]+1) + 10)

    @NLconstraint(m, 0.8*log(x[2]+1) + 0.96*log(x[1]-x[2]+1) - 0.8*x[3] >= 0)
    @NLconstraint(m, log(x[2]+1) + 1.2*log(x[1]-x[2]+1) - x[3] - 2*y[3] >= -1)
    @constraint(m, x[2] - x[1] <= 0)
    @constraint(m, x[2] - 2*y[1] <= 0)
    @constraint(m, x[1] - x[2] - 2*y[2] <= 0)
    @constraint(m, y[1] + y[2] <= 1)

    optimize!(m)

    @test isapprox(objective_value(m), 0.759; atol = 1e-3)
    @test isapprox(value.(x), [1.147,0.547,1.000]; atol = 1e-3)
    @test isapprox(value.(y), [0.273,0.300,0.000]; atol = 1e-3)

    # warm start
    optimize!(m)

    @test isapprox(objective_value(m), 0.759; atol = 1e-3)
    @test isapprox(value.(x), [1.147,0.547,1.000]; atol = 1e-3)
    @test isapprox(value.(y), [0.273,0.300,0.000]; atol = 1e-3)

    # # Modify the objective function
    # @NLobjective(m, Min, 3*y[1] + 8*y[2] + 10*y[3]
    #     + 10*x[1] - 7*x[3] - 18*log(x[2]+1)
    #     - 19.2*log(x[1]-x[2]+1) + 10)

    # optimize!(m)
end

end

runtests(TestJuMPExample)
