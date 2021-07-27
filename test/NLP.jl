module TestNlpExample

using filterSQP
using Test

prob = createProblem(
    6,
    Float64[0,0,0,0,0,0], # x_L
    Float64[2,2,1,1,1,1], # x_U
    6,
    Float64[-1e+9,-1e+9,-1e+9,-1e+9,0,-2], # g_L
    Float64[0,0,0,1,1e+9,1e+9], # g_U
    "LLLLNN", # cstype
    16,
    2000000,
    nothing,
    nothing,
    nothing,
    nothing,
)

prob.maxa = 22
prob.la[1] = 23
prob.a[7] = +1.0; prob.la[8] = 2
prob.a[8] = -1.0; prob.la[9] = 1
prob.a[9] = +1.0; prob.la[10] = 2
prob.a[10] = -2.0; prob.la[11] = 4
prob.a[11] = +1.0; prob.la[12] = 1
prob.a[12] = -1.0; prob.la[13] = 2
prob.a[13] = -2.0; prob.la[14] = 5
prob.a[14] = +1.0; prob.la[15] = 4
prob.a[15] = +1.0; prob.la[16] = 5
prob.la[24] = 1
prob.la[25] = 7
prob.la[26] = 9
prob.la[27] = 11
prob.la[28] = 14
prob.la[29] = 16
prob.la[30] = 19
prob.la[31] = 23

function objfun(
    x_ptr::Ptr{Cdouble}, 
    n_ptr::Ptr{Cint},
    f::Ptr{Cdouble}, 
    user::Ptr{Cdouble}, 
    iuser::Ptr{Cint}, 
    userdata::Ptr{Cvoid},
    errflag::Ptr{Cint}
)
    n = unsafe_load(n_ptr)
    @test n == 6
    x = unsafe_wrap(Array, x_ptr, Int(n))
    fval = 5.0 * x[4] + 6.0 * x[5] + 10.0 * x[1] - 7.0 * x[3] - 18.0 * log(x[2] + 1) - 19.2 * log(x[1] - x[2] + 1) + 10.0
    unsafe_store!(f, fval)
    unsafe_store!(errflag, 0)
    return
end

function confun(
    x_ptr::Ptr{Cdouble}, 
    n_ptr::Ptr{Cint}, 
    m_ptr::Ptr{Cint}, 
    c_ptr::Ptr{Cdouble}, 
    a::Ptr{Cdouble}, 
    la_ptr::Ptr{Cint}, 
    user::Ptr{Cdouble}, 
    iuser_ptr::Ptr{Cint}, 
    userdata::Ptr{Cvoid},
    errflag::Ptr{Cint}
)
    n = unsafe_load(n_ptr)
    m = unsafe_load(m_ptr)
    x = unsafe_wrap(Array, x_ptr, n)
    c = unsafe_wrap(Array, c_ptr, m)
    c[1] = x[2] - x[1]
    c[2] = x[2] - 2 * x[4]
    c[3] = x[1] - x[2] - 2 * x[5]
    c[4] = x[4] + x[5]
    c[5] = 0.8 * log(x[2] + 1) + 0.96 * log(x[1] - x[2] + 1) - 0.8 * x[3]
    c[6] = log(x[2] + 1) + 1.2 * log(x[1] - x[2] + 1) - x[3] - 2 * x[6]
    unsafe_store!(errflag, 0)
    return
end

function gradient(
    n_ptr::Ptr{Cint}, 
    m_ptr::Ptr{Cint}, 
    mxa_ptr::Ptr{Cint}, 
    x_ptr::Ptr{Cdouble}, 
    a_ptr::Ptr{Cdouble}, 
    la_ptr::Ptr{Cint}, 
    maxa_ptr::Ptr{Cint}, 
    user_ptr::Ptr{Cdouble}, 
    iuser_ptr::Ptr{Cint}, 
    userdata::Ptr{Cvoid},
    errflag::Ptr{Cint}
)
    n = unsafe_load(n_ptr)
    m = unsafe_load(m_ptr)
    maxa = unsafe_load(maxa_ptr)
    x = unsafe_wrap(Array, x_ptr, n)
    a = unsafe_wrap(Array, a_ptr, maxa)
    la = unsafe_wrap(Array, la_ptr, 1 + maxa + m + 2)
    iuser = unsafe_wrap(Array, iuser_ptr, 1)

    # here, the number of entries in a is known (hence pjp known)
    pjp = la[1]
    nrow = 0
    a_entries = 0
    la[pjp + 1] = 1

    # compute gradient of objective function
    a[1] = 10.0 - 19.2 / (x[1] - x[2] + 1.0)
    la[2] = 1

    a[2] = -18.0 / (x[2] + 1.0) + 19.2 / (x[1] - x[2] + 1.0)
    la[3] = 2

    a[3] = -7.0
    la[4] = 3

    a[4] = 5.0
    la[5] = 4

    a[5] = 6.0
    la[6] = 5

    a[6] = 8.0
    la[7] = 6

    a_entries += n # 6
    la[pjp + 2] = a_entries + 1
    # NOTE: Always leave n spaces for objective gradient !!!

    # space for linear Jacobian
    a_entries += 9 # 15
    nrow = 1 + iuser[1] # 1 + number of linear constraints

    # evaluate the entries of the Jacobian of c (for nonlinear part)
    # ROW 1
    a_entries += 1 # 16
    a[a_entries] = 0.96 / (x[1] - x[2] + 1.0)
    la[a_entries + 1] = 1

    a_entries += 1 # 17
    a[a_entries] = 0.8 / (x[2] + 1.0) - 0.96 / (x[1] - x[2] + 1.0)
    la[a_entries + 1] = 2

    a_entries += 1 # 18
    a[a_entries] = -0.8
    la[a_entries + 1] = 3

    nrow += 1 # 6
    la[pjp + nrow + 1] = a_entries + 1

    # ROW 2
    a_entries += 1 # 19
    a[a_entries] = 1.2 / (x[1] - x[2] + 1.0)
    la[a_entries + 1] = 1

    a_entries += 1 # 20
    a[a_entries] = 1.0 / (x[2] + 1.0) - 1.2 / (x[1] - x[2] + 1.0)
    la[a_entries + 1] = 2

    a_entries += 1 # 21
    a[a_entries] = -1.0
    la[a_entries + 1] = 3

    a_entries += 1 # 22
    a[a_entries] = -2.0
    la[a_entries + 1] = 6

    nrow += 1 # 7
    la[pjp + nrow + 1] = a_entries + 1

    #= 
    Ahat (n by (1+m)) = [
        1 8  0 11  0 16 19
        2 7  9 12  0 17 20
        3 0  0  0  0 18 21
        4 0 10  0 14  0  0
        5 0  0 13 15  0  0
        6 0  0  0  0  0 22
    ] =#
    @test la == Int32[23, # 1
        1, 2, 3, 4, 5, 6, # 7
        2, 1, 
        2, 4, 
        1, 2, 5, 
        4, 5, # 16
        1, 2, 3, # 19
        1, 2, 3, 6, # 23
        1,
        7, 
        9, 
        11,
        14,
        16,
        19,
        23
    ]

    unsafe_store!(mxa_ptr, a_entries)
    unsafe_store!(errflag, 0)
    return
end

function hessian(
    x_ptr::Ptr{Cdouble}, 
    n_ptr::Ptr{Cint}, 
    m_ptr::Ptr{Cint}, 
    phase_ptr::Ptr{Cint}, 
    lam_ptr::Ptr{Cdouble}, 
    ws_ptr::Ptr{Cdouble}, 
    lws_ptr::Ptr{Cint}, 
    user_ptr::Ptr{Cdouble}, 
    iuser_ptr::Ptr{Cint}, 
    userdata::Ptr{Cvoid},
    l_hess::Ptr{Cint}, 
    li_hess::Ptr{Cint}, 
    errflag::Ptr{Cint}
)
    n = unsafe_load(n_ptr)
    m = unsafe_load(m_ptr)
    phase = unsafe_load(phase_ptr)

    x = unsafe_wrap(Array, x_ptr, n)
    lam = unsafe_wrap(Array, lam_ptr, 12)
    ws = unsafe_wrap(Array, ws_ptr, 2000000)
    lws = unsafe_wrap(Array, lws_ptr, 500000)
    iuser = unsafe_wrap(Array, iuser_ptr, 1)

    # number of linear c/s
    mlin = iuser[1]

    # compute frequent constants
    den1 = (x[1] - x[2] + 1.0)^2
    den2 = (x[2] + 1.0)^2

    # set storage map for Hessian (3 entries here)
    phl = 1
    phr = 1
    phc = phr + 3
    # save number of Hessian entries
    lws[phl] = 3

    # store indices of entries of Hessian
    lws[phr + 1] = 1
    lws[phc + 1] = 1
    lws[phr + 2] = 1
    lws[phc + 2] = 2
    lws[phr + 3] = 2
    lws[phc + 3] = 2

    # calculate objective Hessian only, if phase == 2
    if phase == 2
        ws[1] = +19.2 / den1
        ws[2] = -19.2 / den1
        ws[3] = +19.2 / den1 + 18.0 / den2
    else
        ws[1] = 0.0
        ws[2] = 0.0
        ws[3] = 0.0
    end

    # add constraint Hessian weighted with lambda
    ws[1] -= ( -0.96 * lam[n + mlin + 1] - 1.2 * lam[n + mlin + 2] ) / den1
    ws[2] -= ( +0.96 * lam[n + mlin + 1] + 1.2 * lam[n + mlin + 2] ) / den1
    ws[3] -= ( -0.96 * lam[n + mlin + 1] - 1.2 * lam[n + mlin + 2] ) / den1 + ( -0.8 * lam[n + mlin + 1] - lam[n + mlin + 2] ) / den2

    # set storage requirements for Hessian
    unsafe_store!(l_hess, lws[phl])
    unsafe_store!(li_hess, 1 + 2 * lws[phl])
    
    unsafe_store!(errflag, 0)
    return
end

function solve_test()
    objfun_cb = @cfunction(
        objfun,
        Cvoid,
        (Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint})
    )
    confun_cb = @cfunction(
        confun,
        Cvoid,
        (Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint})
    )
    gradient_cb = @cfunction(
        gradient,
        Cvoid,
        (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint})
    )
    hessian_cb = @cfunction(
        hessian,
        Cvoid,
        (Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint})
    )
    
    objval = Ref{Cdouble}(prob.f)
    ifail = Ref{Int32}(0)
    
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
        ifail,
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
        [],
        [4], # iuser
        prob,
        prob.par.max_iter,
        prob.istat,
        prob.rstat,
        objfun_cb,
        confun_cb,
        gradient_cb,
        hessian_cb,
    )
    
    prob.ifail = ifail[]
    prob.status = ifail[]
    prob.f = objval[]
    
    @show prob.f, prob.x
    @test prob.ifail == 0
    @test isapprox(prob.f, 0.759; atol = 1e-3)
    @test isapprox(prob.x, [1.147,0.547,1.000,0.273, 0.300, 0.000]; atol = 1e-3)
end

solve_test()

end

runtests(TestNlpExample)
