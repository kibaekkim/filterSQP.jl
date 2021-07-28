module TestMOIWrapper

using filterSQP
using MathOptInterface
using Test

const MOI = MathOptInterface

const OPTIMIZER = filterSQP.Optimizer()
MOI.set(OPTIMIZER, MOI.Silent(), true)

# TODO(odow): add features to filterSQP so we can remove some of this caching.
const BRIDGED_OPTIMIZER = MOI.Bridges.full_bridge_optimizer(
    MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        OPTIMIZER,
    ),
    Float64,
)

const CONFIG = MOI.Test.TestConfig(
    atol = 1e-4,
    rtol = 1e-4,
    optimal_status = MOI.LOCALLY_SOLVED,
    infeas_certificates = false,
)

const CONFIG_NO_DUAL = MOI.Test.TestConfig(
    atol = 1e-4,
    rtol = 1e-4,
    optimal_status = MOI.LOCALLY_SOLVED,
    infeas_certificates = false,
    duals = false,  # Don't check dual result!
)

function test_solvername()
    @test MOI.get(OPTIMIZER, MOI.SolverName()) == "filterSQP"
end

function test_supports_default_copy_to()
    @test MOI.Utilities.supports_default_copy_to(OPTIMIZER, false)
    @test !MOI.Utilities.supports_default_copy_to(OPTIMIZER, true)
end

function test_unittest()
    return MOI.Test.unittest(
        BRIDGED_OPTIMIZER,
        CONFIG_NO_DUAL,
        String[
            # VectorOfVariables-in-SecondOrderCone not supported
            "delete_soc_variables",
            # NumberOfThreads not supported
            "number_threads",
            # MOI.Integer not supported.
            "solve_integer_edge_cases",
            # ObjectiveBound not supported.
            "solve_objbound_edge_cases",
            # DualObjectiveValue not supported.
            "solve_result_index",
            # Returns NORM_LIMIT instead of DUAL_INFEASIBLE
            "solve_unbounded_model",
            # MOI.ZeroOne not supported.
            "solve_zero_one_with_bounds_1",
            "solve_zero_one_with_bounds_2",
            "solve_zero_one_with_bounds_3",
            "time_limit_sec",
        ],
    )
end

function test_ConstraintDualStart()
    model = filterSQP.Optimizer()
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)
    l = MOI.add_constraint(model, x[1], MOI.GreaterThan(1.0))
    u = MOI.add_constraint(model, x[1], MOI.LessThan(1.0))
    e = MOI.add_constraint(model, x[2], MOI.EqualTo(1.0))
    c = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, x), 0.0),
        MOI.LessThan(1.5),
    )
    @test MOI.get(model, MOI.ConstraintDualStart(), l) === nothing
    @test MOI.get(model, MOI.ConstraintDualStart(), u) === nothing
    @test MOI.get(model, MOI.ConstraintDualStart(), e) === nothing
    @test MOI.get(model, MOI.ConstraintDualStart(), c) === nothing
    @test MOI.get(model, MOI.NLPBlockDualStart()) === nothing
    MOI.set(model, MOI.ConstraintDualStart(), l, 1.0)
    MOI.set(model, MOI.ConstraintDualStart(), u, -1.0)
    MOI.set(model, MOI.ConstraintDualStart(), e, -1.5)
    MOI.set(model, MOI.ConstraintDualStart(), c, 2.0)
    MOI.set(model, MOI.NLPBlockDualStart(), [1.0, 2.0])
    @test MOI.get(model, MOI.ConstraintDualStart(), l) == 1.0
    @test MOI.get(model, MOI.ConstraintDualStart(), u) == -1.0
    @test MOI.get(model, MOI.ConstraintDualStart(), e) == -1.5
    @test MOI.get(model, MOI.ConstraintDualStart(), c) == 2.0
    @test MOI.get(model, MOI.NLPBlockDualStart()) == [1.0, 2.0]
    MOI.set(model, MOI.ConstraintDualStart(), l, nothing)
    MOI.set(model, MOI.ConstraintDualStart(), u, nothing)
    MOI.set(model, MOI.ConstraintDualStart(), e, nothing)
    MOI.set(model, MOI.ConstraintDualStart(), c, nothing)
    MOI.set(model, MOI.NLPBlockDualStart(), nothing)
    @test MOI.get(model, MOI.ConstraintDualStart(), l) === nothing
    @test MOI.get(model, MOI.ConstraintDualStart(), u) === nothing
    @test MOI.get(model, MOI.ConstraintDualStart(), e) === nothing
    @test MOI.get(model, MOI.ConstraintDualStart(), c) === nothing
    @test MOI.get(model, MOI.NLPBlockDualStart()) === nothing
end

function test_contlinear()
    return MOI.Test.contlineartest(
        BRIDGED_OPTIMIZER,
        CONFIG_NO_DUAL,
        String[
            # FIXME
            "linear10",
            # unbounded instances
            "linear8b",
            "linear8c",
        ],
    )
end

function test_qp()
    return MOI.Test.qptest(BRIDGED_OPTIMIZER, CONFIG)
end

function test_qcp()
    MOI.empty!(BRIDGED_OPTIMIZER)
    return MOI.Test.qcptest(BRIDGED_OPTIMIZER, CONFIG_NO_DUAL)
end

function test_nlptest()
    return MOI.Test.nlptest(
        OPTIMIZER, 
        CONFIG,
        String[
            # Failed to solve the following instances.
            "hs071_hessian_vector_product_test",
            "hs071_no_hessian",
            "feasibility_sense_with_no_objective_and_no_hessian",
            "feasibility_sense_with_objective_and_no_hessian",
        ],
    )
end

function test_getters()
    return MOI.Test.copytest(
        MOI.instantiate(filterSQP.Optimizer, with_bridge_type = Float64),
        MOI.Utilities.Model{Float64}(),
    )
end

function test_boundsettwice()
    MOI.Test.set_lower_bound_twice(OPTIMIZER, Float64)
    return MOI.Test.set_upper_bound_twice(OPTIMIZER, Float64)
end

function test_nametest()
    return MOI.Test.nametest(BRIDGED_OPTIMIZER)
end

function test_validtest()
    return MOI.Test.validtest(BRIDGED_OPTIMIZER)
end

function test_emptytest()
    return MOI.Test.emptytest(BRIDGED_OPTIMIZER)
end

function test_solve_time()
    model = filterSQP.Optimizer()
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    @test isnan(MOI.get(model, MOI.SolveTime()))
    MOI.optimize!(model)
    @test MOI.get(model, MOI.SolveTime()) > 0.0
end

end  # module TestMOIWrapper

runtests(TestMOIWrapper)
# TestMOIWrapper.test_solvername() # PASS
# TestMOIWrapper.test_supports_default_copy_to() # PASS
# TestMOIWrapper.test_unittest() # PASS
# TestMOIWrapper.test_ConstraintDualStart() # PASS
# TestMOIWrapper.test_contlinear() # PASS
# TestMOIWrapper.test_qp() # PASS
# TestMOIWrapper.test_qcp() # PASS
# TestMOIWrapper.test_nlptest() # PASS
# TestMOIWrapper.test_getters() # PASS
# TestMOIWrapper.test_boundsettwice() # PASS
# TestMOIWrapper.test_nametest() # PASS
# TestMOIWrapper.test_validtest() # PASS
# TestMOIWrapper.test_emptytest() # PASS
# TestMOIWrapper.test_solve_time() # PASS
