"""
Solves the Nikolov and Whited 2014 JF model
Author: Juan F. Imbet
Based on file Nikolov&Whited2014.py from Marcelo
Date: 5/11/2021
"""

using Distributed
addprocs(17) # Get to 18 procs

@everywhere begin
    using Parameters
    using QuantEcon
    using Interpolations
    using Plots
    using Statistics
    using Roots
    using Random
    using ValidatedNumerics
    using NearestNeighbors
    using NLsolve
    using StatsBase
    using LaTeXStrings
    using SharedArrays
end
# @everywhere include("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\utils.jl")

@everywhere function AgencyModel(θ_; mk=2)
    # Model Parameters
    θ = θ_[1] # curvature of production function
    α = θ_[2] # Manager's bonus
    β = θ_[3] # Equity Share
    s = θ_[4] # Private benefit
    δ = θ_[5] # Depreciation
    τ = θ_[6] # Taxes
    r = θ_[7] # risk free rate
    ρ = θ_[8] # Persitence of AR shock
    σ = θ_[9] # Volatility of AR shock
    ϕ = θ_[10] # Fee in the external equity financed
    a = θ_[11] # Capital adjustment cost

    nk  = 25 # Capital grid points
    nϵ  = 11  # ϵ grid points
    nc  = 7 # Cash point
    _nk = 2 # intermediate points in the capital grid
    _nc = 2 # intermediate points in the cash grid

    stdbound = 4

    # Compute the capital stock around the steady state
    k_steady = ((θ*(1-τ))/(r+δ))^(1/(1-θ))

    # K_steady with adjustment costs? Is actually lower than the normal k steady
    #k_steady    = newton1d(k -> r*(1+a*δ*k)-(θ*(1-τ)*k^(θ-1) - δ - 0.5*a*δ^2), 1e-6..Inf)[1].interval.lo
    #k_steady = ((θ)/(r+δ))^(1/(1-θ))
    k_max    = newton1d(k -> (α+β)*(1-τ)*k^θ - δ*β*k, 1e-6..Inf)[1].interval.lo

    k_max = maximum([mk*k_steady, k_max])

    # Create a non linear grid with more curvature around k_steady

    k_min = k_steady*0.01
    # k_grid = NonLinearGrid(k_min, k_steady, k_max, nk)
    # k_grid = k_grid[:,1]
    # kp_grid = []
    # for i_k=1:nk-1
    #     kp_grid = vcat(kp_grid, LinRange(k_grid[i_k], k_grid[i_k + 1], _nk))
    # end

    # kp_grid=unique(kp_grid)
    # plot()
    # scatter!(k_grid, k_grid, label="")

    # nkp = size(kp_grid)[1]
    k_grid  = LinRange(k_min, k_max, nk)
    kp_grid = LinRange(k_min, k_max, nk*_nk)
    nkp     = size(kp_grid)[1]
    mc_ϵ = tauchen(nϵ, ρ, σ, 0, 4)
    Π_ϵ  = mc_ϵ.p
    ϵ_grid = exp.(collect(mc_ϵ.state_values))
    c_max = mk > 2 ? k_steady*(mk-2) : k_steady
    c_grid  = LinRange(0.0, c_max, nc)
    cp_grid = LinRange(0.0, c_max, nc*_nc)
    ncp = size(cp_grid)[1]
    # Compute the rewards
    D=zeros(nk, nkp, nc, ncp, nϵ)
    R=zeros(nk, nkp, nc, ncp, nϵ)
    for (i_k, k) in enumerate(k_grid)
        for (i_kp, kp) in enumerate(kp_grid)
            for (i_c, c) in enumerate(c_grid)
                for (i_cp, cp) in enumerate(cp_grid)
                    for (i_ϵ, ϵ) in enumerate(ϵ_grid)
                        I = kp-(1-δ)*k
                        d = (1-τ)*(1-(α+s))*ϵ*k^θ + δ*k*τ - I - 0.5*a*((I/k)^2)*k  - cp +c*(1+r*(1-τ))*(1-s)
                        D[i_k, i_kp, i_c, i_cp, i_ϵ] = d>=0 ? d : d*(1+ϕ)
                        R[i_k, i_kp, i_c, i_cp, i_ϵ] = (α+s)*ϵ*k^θ + s*c*(1+r) + β*D[i_k, i_kp, i_c, i_cp, i_ϵ]
                    end
                end
            end
        end
    end

    return (α=α, β=β, s=s, δ=δ, a=a, θ=θ, τ=τ,
            r=r, ϕ=ϕ, σ=σ, ρ=ρ, stdbound=stdbound,
            nϵ=nϵ, nk=nk, nc=nc, nkp=nkp, ncp=ncp,
            k_steady=k_steady, k_grid=k_grid,
            kp_grid=kp_grid, Π_ϵ=Π_ϵ, ϵ_grid=ϵ_grid,
            c_grid=c_grid, cp_grid=cp_grid,
            R=R, D=D)
end

@everywhere function howard_step!(U, R, i_kp, i_cp, am,  Λ, Ω_kp, Ω_cp)
    nϵ, nc, nk, nkp, ncp, k_grid, c_grid, ϵ_grid, kp_grid, cp_grid, Π_ϵ = am.nϵ, am.nc, am.nk, am.nkp, am.ncp, am.k_grid, am.c_grid, am.ϵ_grid, am.kp_grid, am.cp_grid, am.Π_ϵ
    r = am.r
    UU = copy(U)
    for (i_ϵ, ϵ) in enumerate(ϵ_grid)
        for (i_k, k) in enumerate(k_grid)
            for (i_c, c) in enumerate(c_grid)
                # Given policy i_kp and i_cp, what is the continuation value?
                _ikp = i_kp[i_k, i_c, i_ϵ]
                _icp = i_cp[i_k, i_c, i_ϵ]
                EV = dot(Π_ϵ[i_ϵ, :], sum(Λ[_ikp, _icp, j]*UU[Ω_kp[_ikp, _icp,j], Ω_cp[_ikp, _icp,j], :] for j=1:3))
                U[i_k, i_c, i_ϵ]     = R[i_k, _ikp, i_c, _icp, i_ϵ] + (1/(1+r))*EV
            end
        end
    end
end


@everywhere function bellman_operator!(U, R, i_kp, i_cp,  am, Λ, Ω_kp, Ω_cp)

    nϵ, nc, nk, nkp, ncp, k_grid, c_grid, ϵ_grid, kp_grid, cp_grid, Π_ϵ = am.nϵ, am.nc, am.nk, am.nkp, am.ncp, am.k_grid, am.c_grid, am.ϵ_grid, am.kp_grid, am.cp_grid, am.Π_ϵ
    r = am.r
    UU= copy(U)
    for (i_ϵ, ϵ) in enumerate(ϵ_grid)
        EV = [dot(Π_ϵ[i_ϵ, :], sum(Λ[_ikp, _icp, j]*UU[Ω_kp[_ikp, _icp,j], Ω_cp[_ikp, _icp,j], :] for j=1:3)) for _ikp=1:nkp, _icp=1:ncp ]
        for (i_k, k) in enumerate(k_grid)
            for (i_c, c) in enumerate(c_grid)
                RHS                  = R[i_k, :, i_c, :, i_ϵ] + (1/(1+r))*EV
                i_kc                 = argmax(RHS)
                U[i_k, i_c, i_ϵ]     = RHS[i_kc]                                       # update U with all the best expected values.
                i_kp[i_k, i_c, i_ϵ]  = i_kc[1]
                i_cp[i_k, i_c, i_ϵ]  = i_kc[2]
            end
        end
    end

end



@everywhere function vfi(am; tol=1E-6, itmax = 10000, verbose=true, H=10)


    if verbose
        println("Solving the model via Value Function Iteration")
        println("Max Iterations: $(itmax)")
        println("Steps in Howard improvement: $(H)")
    end

    Λ, Ω_kp, Ω_cp = coordinates(am); # Store the coordinates of the interpolation in advance to avoid creating
                                     # a new object each time


    # An initial guess is the value function of the steady state
    ϵ_grid, k_grid, c_grid = am.ϵ_grid, am.k_grid, am.c_grid
    kp_grid, cp_grid = am.kp_grid, am.cp_grid
    Π_ϵ = am.Π_ϵ
    R= am.R
    nk, nc, nϵ, nkp, ncp = am.nk, am.nc, am.nϵ, am.nkp, am.ncp
    U    = zeros(nk, nc, nϵ)
    i_kp = ones(Int, nk, nc, nϵ)
    i_cp = ones(Int, nk, nc, nϵ)
    Kp   = zeros(nk, nc, nϵ)
    Cp   = zeros(nk, nc, nϵ)

    U_old  = copy(U)
    Kp_old = copy(Kp)
    Cp_old = copy(Cp)

    for i=1:itmax

        for h=1:H
            howard_step!(U, R, i_kp, i_cp, am,  Λ, Ω_kp, Ω_cp)
        end

        bellman_operator!(U, R, i_kp, i_cp, am, Λ, Ω_kp, Ω_cp)
        Kp = kp_grid[i_kp]
        Cp = cp_grid[i_cp]
        ctol  = maximum(abs.(U-U_old))
        ptolk = maximum(abs.(Kp-Kp_old))
        ptolc = maximum(abs.(Cp-Cp_old))

        verbose && i%50==0 && println("Iteration $i - Tolerance U = $(round(ctol,digits = 6)) - K error = $(round(ptolk,digits = 6)) - C error = $(round(ptolc,digits = 6)) ")

        if (ctol < tol)
            break
        end

        U_old  = copy(U)
        Kp_old = copy(Kp)
        Cp_old = copy(Cp)
    end

    # return also the values themselves

    return U, Kp, Cp, i_kp, i_cp, R
end



@everywhere function policy_functions(Kp, Cp, am)
    @unpack θ, α, β, s, δ, τ, r, ρ, σ, ϕ, a, nk, nϵ, nc, nkp, ncp, Π_ϵ, k_grid, ϵ_grid, c_grid, kp_grid, cp_grid, k_steady = am;

    I_p      = zeros(nk, nc, nϵ)
    I        = zeros(nk, nc, nϵ)
    CRatio_P = zeros(nk, nc, nϵ)
    CF_p     = zeros(nk, nc, nϵ)
    F_p      = zeros(nk, nc, nϵ)
    for i_ϵ=1:nϵ
        for i_c=1:nc
            for i_k=1:nk
                CF_p[i_k,i_c,i_ϵ]       = ((1-τ)*ϵ_grid[i_ϵ]*k_grid[i_k]^θ)/k_grid[i_k]
                I[i_k, i_c, i_ϵ]        = Kp[i_k,i_c,i_ϵ]-(1-δ)*k_grid[i_k]
                I_p[i_k,i_c,i_ϵ]        = (Kp[i_k,i_c,i_ϵ]-(1-δ)*k_grid[i_k])/k_grid[i_k]
                CRatio_P[i_k,i_c,i_ϵ]   = Cp[i_k,i_c,i_ϵ]/(c_grid[i_c]+k_grid[i_k])
                d                       = (1-τ)*(1-(α+s))*ϵ_grid[i_ϵ]*k_grid[i_k]^θ + δ*k_grid[i_k]*τ - I[i_k,i_c,i_ϵ]  - 0.5*a*((I[i_k,i_c,i_ϵ] /k_grid[i_k])^2)*k_grid[i_k]  - Cp[i_k,i_c,i_ϵ] + c_grid[i_c]*(1+r*(1-τ))*(1-s)

                F_p[i_k,i_c,i_ϵ]    = d*(1+(d<0)*ϕ)/(c_grid[i_c]+k_grid[i_k])

            end
        end
    end

    # What is the kstar?
    i_ks = argmin(abs.(k_grid.-k_steady))[1]

    cf_k = ((1-τ)*ϵ_grid*k_steady^θ)/k_steady
    p1=plot()
    plot!(log.(ϵ_grid), cf_k, label = "Low")
    plot!(log.(ϵ_grid), cf_k, label = "Medium")
    plot!(log.(ϵ_grid), cf_k, label = "High")
    ylabel!("Cash Flow / Capital")
    xlabel!("Log productivity shock")

    # Investment over capital
    p2=plot()
    plot!(log.(ϵ_grid), I_p[i_ks, 2, :], label = "Low")
    plot!(log.(ϵ_grid), I_p[i_ks, 3, :], label = "Medium")
    plot!(log.(ϵ_grid), I_p[i_ks, 4, :], label = "High")
    ylabel!("Investment / Capital")
    xlabel!("Log productivity shock")


    # Investment over capital
    p3=plot()
    plot!(log.(ϵ_grid), CRatio_P[i_ks, 2, :], label = "Low")
    plot!(log.(ϵ_grid), CRatio_P[i_ks, 3, :], label = "Medium")
    plot!(log.(ϵ_grid), CRatio_P[i_ks, 4, :], label = "High")
    ylabel!("Cash / Assets")
    xlabel!("Log productivity shock")

    p4=plot()
    plot!(log.(ϵ_grid), F_p[i_ks, 2, :], label = "Low")
    plot!(log.(ϵ_grid), F_p[i_ks, 3, :], label = "Medium")
    plot!(log.(ϵ_grid), F_p[i_ks, 4, :], label = "High")
    ylabel!("External Fin / Assets")
    xlabel!("Log productivity shock")

    plot(p1,p2,p3,p4, layout = (4,1), legendfontsize=5,size = (1000, 1000),show=true)
    #savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\policy_functions")


    # Surface plot
    plot(zlabel=L"\frac{I}{k}")
    plot!(log.(ϵ_grid), k_grid, I_p[:, 2, :], st = :surface, camera=(80,30), label = "Low", size=(1000,1000), c=:blues)
    plot!(log.(ϵ_grid), k_grid, I_p[:, 3, :], st = :surface, camera=(80,30), label = "Medium", size=(1000,1000), c=:blues)
    plot!(log.(ϵ_grid), k_grid, I_p[:, 4, :], st = :surface, camera=(80,30), label = "High", size=(1000,1000), c=:blues)

    xlabel!(L"ln(\epsilon)")
    ylabel!(L"k")
    #savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\investment")
end

@everywhere function simulate_economy(am, Kp, Cp; Terg = 200, Tsim = 100000, N = 50)
    @unpack θ, α, β, s, δ, τ, r, ρ, σ, ϕ, a, nk, nϵ, nc, nkp, ncp, Π_ϵ, k_grid, ϵ_grid, c_grid, kp_grid, cp_grid, k_steady = am;
    # Simulate the shocks
    T = Terg + Tsim +1
    nϵs=100
    mc = tauchen(nϵs, ρ, σ, 0, 4)
    ϵ_grids = mc.state_values
    ϵ_grids = exp.(collect(ϵ_grids))
    mc = MarkovChain(mc.p)

    # Initial points from the stationary distribution
    μ = mc.p^10
    μ = μ[1,:]

    init_points = sample(1:nϵs, Weights(μ), N)
    E  = [simulate(mc, T, init =  init_points[i]) for i=1:N]

    ϵs = [ϵ_grids[e[t]] for t=1:T, e in E]
    i_ϵs = [e[t] for t=1:T, e in E]

    # Simulate how capital and cash change
    Kp_func = extrapolate(
                interpolate(
                    (k_grid, c_grid, ϵ_grid),
                    Kp,
                    (Gridded(Linear()), Gridded(Linear()), Gridded(Linear()))
                ),
                Flat()
    )

    Cp_func = extrapolate(
                interpolate(
                    (k_grid, c_grid, ϵ_grid),
                    Cp,
                    (Gridded(Linear()), Gridded(Linear()), Gridded(Linear()))
                ),
                Flat()
    )
    Ksim = zeros(T, N)
    Csim = zeros(T, N)

    Ksim[1, :] .= k_steady

    for n=1:N
        for t=2:T
            Ksim[t, n] = Kp_func(Ksim[t-1, n], Csim[t-1, n], ϵs[t-1,n])
            Csim[t, n] = Cp_func(Ksim[t-1, n], Csim[t-1, n], ϵs[t-1,n])
        end
    end

    println("Simulation Check")
    println("         lb            min         max        ub")
    println("Capital: $(round(minimum(k_grid), digits=1))           $(round(minimum(Ksim), digits=1))       $(round(maximum(Ksim), digits=1))       $(round(maximum(k_grid), digits=1))")
    println("Cash:    $(round(minimum(c_grid), digits=1))             $(round(minimum(Csim), digits=1))       $(round(maximum(Csim), digits=1))       $(round(maximum(c_grid), digits=1))")



    # p1 = plot(ϵs[:,1], clim = (ϵ_grids[1], ϵ_grids[end]), label = "Shock")
    # p2 = plot(Ksim[:,1], label="Capital")
    # p3 = plot(Csim[:, 1], label="Cash")
    # plot(p1, p2, p3, layout=(3,1), show=true)


    # Remove the burning period
    Ksim=Ksim[(Terg+1):end, :]
    Csim = Csim[(Terg+1):end, :]

    c_a = Csim[2:end, :]./((Csim[1:end-1, :] + Ksim[1:end-1, :]))
    #savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\example_simulation")

    return mean(c_a)


end

@everywhere function compute_moments(θ_)
    am = AgencyModel(θ_; mk=2);

    @time  U, Kp, Cp,  i_kp, i_cp, R = vfi(am; H=10, verbose=false);

    #policy_functions(Kp, Cp, am);
    moments = simulate_economy(am, Kp, Cp);

    return moments

end

@everywhere function single_comparative_statistics(θ_, npos, gridj)
    θ_temp = copy(θ_)
    θ_temp[npos] = gridj
    return compute_moments(θ_temp)
end
function comparative_statistics(θ_)

    grids=[LinRange(0.0, 0.25, 20), #grid_ϕ
           LinRange(0.1, 0.5, 20),  #grid_σ
           LinRange(0.5, 0.75, 20), #grid_ρ
           LinRange(0.5, 0.9, 20),  #grid_θ
           LinRange(0, 1.5, 20),    #grid_a
           LinRange(0.05, 0.2, 20), #grid_δ
           LinRange(0, 0.01, 20),   #grid_α
           LinRange(0, 0.0002, 20), #grid_s
           LinRange(0.01, 0.1, 20)]    #grid_β

    positions_in_θ = [10, 9, 8, 1, 11, 5, 2, 4, 3] # This tells where is the parameter in the vector θ_
    names_graph = [L"Cost of External Financing: $\phi$",
                   L"Volatility of Innovation to $ln(\epsilon)$ : $\sigma$",
                   L"Serial correlation of $ln(\epsilon)$ : $\rho$",
                   L"Curvature of the Profit Function: $\theta$",
                   L"Convex Cost of Capital Adjustment: $a$",
                   L"Depreciation: $\delta$",
                   L"Profit-sharing: $\alpha$ (in bps)",
                   L"Tunneling: $s$ (in bps)",
                   L"Managerial Ownership: $\beta$"]



    c_as=[]
    for (i, grid) in enumerate(grids)
        println("Computing comparative statistics for: $(names_graph[i])")
        c_a=SharedArray{Float64}(size(grid)[1])
        @sync @distributed for j=1:size(c_a)[1]
            c_a[j]=single_comparative_statistics(θ_, positions_in_θ[i], grid[j])
        end
        push!(c_as, c_a)
    end

    i=1
    p1 = plot(label ="")
    plot!(grids[i], c_as[i])
    ylabel!("Cash / Assets")
    xlabel!(names_graph[i])
    savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\cs_$i")

    i=2
    p2 = plot(label ="")
    plot!(grids[i], c_as[i])
    ylabel!("Cash / Assets")
    xlabel!(names_graph[i])
    savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\cs_$i")

    i=3
    p3 = plot(label ="")
    plot!(grids[i], c_as[i])
    ylabel!("Cash / Assets")
    xlabel!(names_graph[i])
    savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\cs_$i")

    i=4
    p4 = plot(label ="")
    plot!(grids[i], c_as[i])
    ylabel!("Cash / Assets")
    xlabel!(names_graph[i])
    savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\cs_$i")

    i=5
    p5 = plot(label ="")
    plot!(grids[i], c_as[i])
    ylabel!("Cash / Assets")
    xlabel!(names_graph[i])
    savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\cs_$i")

    i=6
    p6 = plot(label ="")
    plot!(grids[i], c_as[i])
    ylabel!("Cash / Assets")
    xlabel!(names_graph[i])
    savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\cs_$i")

    i=7
    p7 = plot(label ="")
    plot!(grids[i], c_as[i])
    ylabel!("Cash / Assets")
    xlabel!(names_graph[i])
    savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\cs_$i")

    i=8
    p8 = plot(label ="")
    plot!(grids[i], c_as[i])
    ylabel!("Cash / Assets")
    xlabel!(names_graph[i])
    savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\cs_$i")

    i=9
    p9 = plot(label ="")
    plot!(grids[i], c_as[i] )
    ylabel!("Cash / Assets")
    xlabel!(names_graph[i])
    savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\cs_$i")


end

function main(;debug=false)

    θ_     = zeros(11)

    θ_[1]  =  0.773        # curvature of production function
    θ_[2]  =  0.751/100   # Manager's bonus
    θ_[3]  =  0.051        # Equity Share
    θ_[4]  =  0.101/1000   # Private benefit
    θ_[5]  =  0.130        # Depreciation
    θ_[6]  =  0.2          # Taxes
    θ_[7]  =  0.011        # risk free rate
    θ_[8]  =  0.713        # Persitence of AR shock
    θ_[9]  =  0.262        # Volatility of AR shock
    θ_[10] =  0.043        # Fee in the external equity financed
    θ_[11] =  1.278        # Capital adjustment cost

    comparative_statistics(θ_)
end

main()

#
