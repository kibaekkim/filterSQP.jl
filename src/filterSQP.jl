module filterSQP

using Base: UInt8, Float64, String
using Libdl
using LinearAlgebra
using SparseArrays

export libfilter

function __init__()
    try
        global libfilter = Libdl.find_library("libfilter")
        if libfilter == ""
            @warn("Could not load filterSQP shared library. Make sure it is in your library path.")
        end
    catch
        @warn("Failed to initialize the package.")
        rethrow()
    end
end

export createProblem,
    solveProblem,
    FilterSqpProblem

include("parameters.jl")

mutable struct FilterSqpProblem
    n::Int                      # number of variables
    m::Int                      # number of constraints (linear and nonlinear)
    maxa::Int                   # max. nmbrer of entries in a
    ifail::Vector{Int32}        # fail flag: -1 = ON ENTRY: warm start (use ONLY if n, m, k, lws unchanged from previous call)
                                #             0 = successful run
                                #             1 = unbounded NLP detected (f < fmin)
                                #             2 = linear constraints are infeasible
                                #             3 = nonlinear constraints locally infeasible
                                #             4 = h <= eps, but QP is infeasible
                                #             5 = termiation with rho < eps
                                #             6 = termiation with iter > max_iter
                                #             7 = crash in user routine (IEEE error) could not be resolved
                                #             8 = unexpect ifail from QP solver 
                                #             9 = not enough *real* workspace
                                #            10 = not enough *integer* workspace
    x::Vector{Float64}          # initial gues of variables / best solution found
    c::Vector{Float64}          # constraint values at solution
    f::Float64                  # objective value at solution
    fmin::Float64               # lower bound on f(x); unbounded, if f(x) < fmin
    blo::Vector{Float64}        # lower bounds on variables and constraints
    bup::Vector{Float64}        # upper bounds on variables and constraints
    s::Vector{Float64}          # variable scale factors (see scale_mode on settings)
    a::Vector{Float64}          # dense/sparse storage of Jacobian & objective gradient
    la::Vector{Int32}           # integers associated with storage of a
    ws::Vector{Float64}         # *real* workspace
    lws::Vector{Int32}          # *integer* workspace
    lam::Vector{Float64}        # Lagrange multipliers at solution
    cstype::String              # 'N' for Nonlinear; 'L' for Linear constraints
    # user::Vector{Float64}       # *real* user workspace passed through to objfun etc.
    # iuser::Vector{Int32}        # *integer* user workspace passed through to objfun etc.
    istat::Vector{Int32}        # storage for some *integer* statistics of this run (size 14)
    rstat::Vector{Float64}      # storage for some *real* statistics of this run (size 7)

    mult_g::Vector{Float64}     # lagrange multipliers on constraints
    mult_x_L::Vector{Float64}   # lagrange multipliers on lower bounds
    mult_x_U::Vector{Float64}   # lagrange multipliers on upper bounds
    status::Int                 # Final status

    grad_f::Vector{Float64}

    nele_jac::Int
    rows_jac::Vector{Int}
    cols_jac::Vector{Int}
    values_jac::Vector{Float64}
    A::SparseMatrixCSC{Float64, Int64}

    nele_hess::Int
    rows_hess::Vector{Int}
    cols_hess::Vector{Int}
    values_hess::Vector{Float64}
    H::SparseMatrixCSC{Float64, Int64}

    # Callbacks
    eval_f::Union{Function,Nothing}
    eval_g::Union{Function,Nothing}
    eval_grad_f::Union{Function,Nothing}
    eval_jac_g::Union{Function,Nothing}
    eval_h::Union{Function,Nothing}

    par::Parameters

    function FilterSqpProblem(
        n,
        m,
        bl,
        bu,
        cstype,
        nele_jac,
        nele_hess,
        eval_f,
        eval_g,
        eval_grad_f,
        eval_jac_g,
        eval_h,
    )
        par = Parameters()

        # from filterSQP manual
        nprof = 20*n
        par.mxwk = (16*n + 8*m + par.mlp + 8*par.maxf + par.kmax*(par.kmax+9)/2 + 5*n + nprof) * 2
        par.mxiwk = (4*n + 3*m + par.mlp + 100 + par.kmax + 9*n + m) * 2
        ws = zeros(Float64, par.mxwk)
        lws = zeros(Float64, par.mxiwk)

        prob = new(
            n,
            m,
            n + nele_jac,                   # maxa
            [0],                            # ifail
            zeros(Float64, n),              # x
            zeros(Float64, m),              # c
            0.0,                            # f
            -Inf,                           # fmin
            bl,                             # bl
            bu,                             # bu
            ones(Float64, n),               # s
            zeros(Float64, n+nele_jac),     # a
            zeros(Float64, n+nele_jac+m+3), # la
            ws,                             # ws: should the size better be mxwk?
            lws,                            # lws: should the size better be mxiwk?
            zeros(Float64, n+m),            # lam
            cstype,                         # cstype
            zeros(Int32, 14),               # istat
            zeros(Float64, 7),              # rstat
            zeros(Float64, m),              # mult_g
            zeros(Float64, n),              # mult_x_L
            zeros(Float64, n),              # mult_x_U
            0,                              # status
            zeros(Float64, n),              # grad_f
            nele_jac,
            zeros(Int, nele_jac),
            zeros(Int, nele_jac),
            zeros(Float64, nele_jac),
            sparse(1:n, ones(n), zeros(n), n, m+1),
            nele_hess,
            zeros(Int, nele_hess),
            zeros(Int, nele_hess),
            zeros(Float64, nele_hess),
            spzeros(n, n),
            eval_f,
            eval_g,
            eval_grad_f,
            eval_jac_g,
            eval_h,
            par,
        )
        
        return prob
    end
end

# From filter.c
const ApplicationReturnStatus = Dict(
    0 => :Optimal_solution_found,
    1 => :Unbounded_objective,
    2 => :Infeasible_Linear_Constraints,
    3 => :Locally_Infeasible_Nonlinear_Constraints,
    4 => :QP_Infeasible,
    5 => :SQP_Termination_by_eps,
    6 => :SQP_Termination_by_maxiter,
    7 => :Crash_in_user_supplied_routines,
    8 => :Unexpected_ifail_from_QP_solver,
    9 => :Not_enough_REAL_workspace_or_parameter_error,
    10 => :Not_enough_INTEGER_workspace_or_parameter_error,
)

###########################################################################
# Callback wrappers
###########################################################################

function objfun_wrapper(
    x_ptr::Ptr{Cdouble}, 
    n_ptr::Ptr{Cint},
    f_ptr::Ptr{Cdouble}, 
    user::Ptr{Cdouble}, 
    iuser::Ptr{Cint}, 
    userdata::Ptr{Cvoid},
    errflag::Ptr{Cint}
)
    # println(">>>> objfun_wrapper ")
    unsafe_store!(errflag, 1)
    prob = unsafe_pointer_to_objref(userdata)::FilterSqpProblem
    n = unsafe_load(n_ptr)
    x = unsafe_wrap(Array, x_ptr, Int(n))
    f = convert(Float64, prob.eval_f(x))::Float64
    # @show n, x, f
    unsafe_store!(f_ptr, f)
    unsafe_store!(errflag, 0)
    # println("<<<< objfun_wrapper ")
    return
end

function confun_wrapper(
    x_ptr::Ptr{Cdouble}, 
    n_ptr::Ptr{Cint}, 
    m_ptr::Ptr{Cint}, 
    c_ptr::Ptr{Cdouble}, 
    a_ptr::Ptr{Cdouble}, 
    la_ptr::Ptr{Cint}, 
    user::Ptr{Cdouble}, 
    iuser::Ptr{Cint}, 
    userdata::Ptr{Cvoid},
    errflag::Ptr{Cint}
)
    # println(">>>> confun_wrapper ")
    unsafe_store!(errflag, 1)
    prob = unsafe_pointer_to_objref(userdata)::FilterSqpProblem
    n = unsafe_load(n_ptr)
    m = unsafe_load(m_ptr)
    x = unsafe_wrap(Array, x_ptr, Int(n))
    c = unsafe_wrap(Array, c_ptr, Int(m))
    # carr = Vector{Float64}(undef, m)
    # prob.eval_g(x, carr)
    # for i = 1:m
    #     c[i] = carr[i]
    # end
    prob.eval_g(x, c)
    # @show n, m, x, c
    unsafe_store!(errflag, 0)
    # println("<<<< confun_wrapper ")
    return
end

function gradient_wrapper(
    n_ptr::Ptr{Cint}, 
    m_ptr::Ptr{Cint}, 
    mxa_ptr::Ptr{Cint}, 
    x_ptr::Ptr{Cdouble}, 
    a_ptr::Ptr{Cdouble}, 
    la_ptr::Ptr{Cint}, 
    maxa_ptr::Ptr{Cint}, 
    user::Ptr{Cdouble}, 
    iuser::Ptr{Cint}, 
    userdata::Ptr{Cvoid},
    errflag::Ptr{Cint}
)
    # println(">>>> gradient_wrapper ")
    unsafe_store!(errflag, 1)
    prob = unsafe_pointer_to_objref(userdata)::FilterSqpProblem
    n = unsafe_load(n_ptr)
    m = unsafe_load(m_ptr)
    maxa = unsafe_load(maxa_ptr)
    # @show maxa

    x = unsafe_wrap(Array, x_ptr, n)
    a = unsafe_wrap(Array, a_ptr, maxa)
    la = unsafe_wrap(Array, la_ptr, 1 + maxa + m + 2)
    # @show n, m, x

    prob.eval_grad_f(x, prob.grad_f)
    prob.eval_jac_g(x, :Structure, prob.rows_jac, prob.cols_jac, prob.values_jac) # FIXME: why should I get this info again?
    prob.eval_jac_g(x, :Values, prob.rows_jac, prob.cols_jac, prob.values_jac)
    # @show prob.nele_jac
    # @show prob.rows_jac 
    # @show prob.cols_jac 
    # @show prob.values_jac 
    fill!(prob.A.nzval, 0.0)
    for i = 1:n
        prob.A[i,1] = prob.grad_f[i]
    end
    for i = 1:prob.nele_jac
        prob.A[prob.cols_jac[i], 1+prob.rows_jac[i]] += prob.values_jac[i]
    end
    # dropzeros!(A)
    # @show x
    # @show prob.A
    
    nnza = length(prob.A.nzval)
    pjp = nnza + 1
    la[1] = pjp

    for i = 1:nnza
        a[i] = prob.A.nzval[i]
        la[i+1] = prob.A.rowval[i]
    end
    for i = 1:(prob.A.n+1)
        la[pjp+i] = prob.A.colptr[i]
    end
    # @show a
    # @show la
    unsafe_store!(mxa_ptr, nnza)
    unsafe_store!(errflag, 0)
    # println("<<<< gradient_wrapper ")

    return
end

function hessian_wrapper(
    x_ptr::Ptr{Cdouble}, 
    n_ptr::Ptr{Cint}, 
    m_ptr::Ptr{Cint}, 
    phase_ptr::Ptr{Cint}, 
    lam_ptr::Ptr{Cdouble}, 
    ws_ptr::Ptr{Cdouble}, 
    lws_ptr::Ptr{Cint}, 
    user::Ptr{Cdouble}, 
    iuser::Ptr{Cint}, 
    userdata::Ptr{Cvoid},
    l_hess::Ptr{Cint},  # On entry: max. space allowed for Hessian storage in ws.  On exit: actual amount of Hessian storage used in ws
    li_hess::Ptr{Cint}, # On entry: max. space allowed for Hessian storage in lws. On exit: actual amount of Hessian storage used in lws
    errflag::Ptr{Cint}
)

    # println(">>>> hessian_wrapper ")
    unsafe_store!(errflag, 1)
    prob = unsafe_pointer_to_objref(userdata)::FilterSqpProblem
    if prob.eval_h === nothing  # Did the user specify a Hessian?
        return
    end

    n = unsafe_load(n_ptr)
    m = unsafe_load(m_ptr)
    phase = unsafe_load(phase_ptr)

    x = unsafe_wrap(Array, x_ptr, n)
    lam = unsafe_wrap(Array, lam_ptr, n+m)
    ws = unsafe_wrap(Array, ws_ptr, prob.par.mxwk)
    lws = unsafe_wrap(Array, lws_ptr, prob.par.mxiwk)
    # @show x
    # @show lam
    # @show prob.nele_hess
    nnzH = 0

    if prob.nele_hess > 0
        obj_factor = ifelse(phase == 2, 1.0, 0.0)
        prob.eval_h(x, :Structure, prob.rows_hess, prob.cols_hess, obj_factor, lam, prob.values_hess) # FIXME: why should I get this info again?
        prob.eval_h(x, :Values, prob.rows_hess, prob.cols_hess, obj_factor, lam, prob.values_hess)

        # @show prob.rows_hess
        # @show prob.cols_hess
        # @show prob.values_hess
        fill!(prob.H.nzval, 0.0)
        for i = 1:prob.nele_hess
            if prob.rows_hess[i] <= prob.cols_hess[i] && !iszero(prob.values_hess[i])
                prob.H[prob.rows_hess[i], prob.cols_hess[i]] += prob.values_hess[i]
            end
        end
        nnzH = length(prob.H.nzval)
        # @show prob.H

        # store indices and values of Hessian
        if nnzH > 0
            for j = 1:n, i = (prob.H.colptr[j]):(prob.H.colptr[j+1]-1)
                lws[i] = prob.H.rowval[i]
                lws[nnzH+i] = j
                ws[i] = prob.H.nzval[i]
            end
        end
    end

    # save number of Hessian entries
    lws[1] = nnzH

    # set storage requirements for Hessian
    unsafe_store!(l_hess, lws[1])
    unsafe_store!(li_hess, 1 + 2 * lws[1])
    unsafe_store!(errflag, 0)
    # println("<<<< hessian_wrapper ")
    return
end

###########################################################################
# C function wrappers
###########################################################################

function createProblem(
    n::Int,
    x_L::Vector{Float64},
    x_U::Vector{Float64},
    m::Int,
    g_L::Vector{Float64},
    g_U::Vector{Float64},
    cstype::String,
    nele_jac::Int,
    nele_hess::Int,
    eval_f,
    eval_g,
    eval_grad_f,
    eval_jac_g,
    eval_h=nothing,
)
    # @show n, m
    # @show cstype
    # @show nele_jac, nele_hess

    return FilterSqpProblem(
        n,
        m,
        [x_L; g_L],
        [x_U; g_U],
        cstype,
        nele_jac,
        nele_hess,
        eval_f,
        eval_g,
        eval_grad_f,
        eval_jac_g,
        eval_h,
    )
end

function addOption(prob::FilterSqpProblem, keyword::String, value)
    setfield!(prob.par, Symbol(keyword), value)
    return
end

function lam2mult!(prob::FilterSqpProblem)
    for j = 1:prob.n
        # @show prob.x[j], prob.blo[j], prob.bup[j], prob.lam[j]
        prob.mult_x_L[j] = max(0.0, prob.lam[j])
        prob.mult_x_U[j] = min(0.0, prob.lam[j])
    end
    for i = 1:prob.m
        # @show prob.c[i], prob.blo[prob.n+i], prob.bup[prob.n+i], prob.lam[prob.n+i]
        prob.mult_g[i] = prob.lam[prob.n+i]
    end
end

function mult2lam!(prob::FilterSqpProblem)
    for j = 1:prob.n
        prob.lam[j] = prob.mult_x_L[j] + prob.mult_x_U[j]
    end
    for i = 1:prob.m
        prob.lam[prob.n+i] = prob.mult_g[i]
    end
end

function solveProblem(prob::FilterSqpProblem)

    if libfilter == ""
        prob.status = -999
        return prob.status
    end

    objfun_cb = @cfunction(
        objfun_wrapper,
        Cvoid,
        (Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint})
    )
    confun_cb = @cfunction(
        confun_wrapper,
        Cvoid,
        (Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint})
    )
    gradient_cb = @cfunction(
        gradient_wrapper,
        Cvoid,
        (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint})
    )
    hessian_cb = @cfunction(
        hessian_wrapper,
        Cvoid,
        (Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint})
    )

    objval = Ref{Cdouble}(prob.f)

    mult2lam!(prob)
    # @show prob.lam
    # @show prob.blo
    # @show prob.bup

    ccall(
        (:filterSQP, libfilter),
        Cvoid,
        (
            Cint, # n
            Cint, # m
            Cint, # kmax
            Cint, # maxa
            Cint, # maxf
            Cint, # mlp
            Cint, # mxwk
            Cint, # mxiwk
            Cint, # iprint
            Cint, # nout
            Ptr{Cint}, # ifail
            Cdouble, # rho
            Ptr{Cdouble}, # x
            Ptr{Cdouble}, # c
            Ptr{Cdouble}, # f
            Cdouble, # fmin
            Ptr{Cdouble}, # bl
            Ptr{Cdouble}, # bu
            Ptr{Cdouble}, # s
            Ptr{Cdouble}, # a
            Ptr{Cint}, # la
            Ptr{Cdouble}, # ws
            Ptr{Cint}, # lws
            Ptr{Cdouble}, # lam
            Ptr{UInt8}, # cstype
            Ptr{Cdouble}, # user
            Ptr{Cint}, # iuser
            Any, # userdata
            Cint, # maxiter
            Ptr{Cint}, # istat
            Ptr{Cdouble}, # rstat
            Ptr{Cvoid}, # objfun
            Ptr{Cvoid}, # confun
            Ptr{Cvoid}, # gradient
            Ptr{Cvoid}, # hessian
        ),
        prob.n,
        prob.m,
        prob.par.kmax,
        prob.maxa,
        prob.par.maxf,
        prob.par.mlp,
        prob.par.mxwk,
        prob.par.mxiwk,
        prob.par.iprint,
        prob.par.nout,
        prob.ifail,
        prob.par.rho,
        prob.x,
        prob.c,
        objval,
        prob.fmin,
        prob.blo,
        prob.bup,
        prob.s,
        prob.a,
        prob.la,
        prob.ws,
        prob.lws,
        prob.lam,
        prob.cstype,
        [], # user
        [], # iuser
        prob,
        prob.par.max_iter,
        prob.istat,
        prob.rstat,
        objfun_cb,
        confun_cb,
        gradient_cb,
        hessian_cb,
    )

    prob.status = prob.ifail[1]
    prob.ifail[1] = -1
    prob.f = objval[]
    # @show prob.status
    # @show prob.f
    # @show prob.lam
    lam2mult!(prob)

    return prob.status
end

include("MOI_wrapper.jl")

end # module
