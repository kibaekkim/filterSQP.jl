Base.@kwdef mutable struct Parameters
    kmax::Int = 500                 # max. allowable dimension of null space
    maxf::Int = 100                 # max. length of filter
    mlp::Int = 1000                 # max. level for resolving degeneracy in QP
    mxwk::Int = 2000000             # length of *real* workspace
    mxiwk::Int = 500000             # length of *integer* workspace
    iprint::Int = 1                 # print flag: 0 = quiet (no printing)
                                    #             1 = one line per iteration
                                    #             2 = scalar information printed
                                    #             3 = scalar & vector information printed
                                    #            >3 = as 3, and call QP solver with iprint-3
    nout::Int = 6                   # output channel
    rho::Float64 = 10.0             # initial/final trust region radius
    max_iter::Int = 1000            # Max. number of iterations allowed to SQP solver
    use_warm_start::Bool = false    # whether or not to use warm start
end