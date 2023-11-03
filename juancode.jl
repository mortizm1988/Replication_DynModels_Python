"""
Solves the Nikolov and Whited 2014 JF model
Author: Juan F. Imbet
Based on file Nikolov&Whited2014.py from Marcelo

Date: 5/11/2021

Log:
07/03/2022: New Version to check the interpolation and Howard Improvement, as well as to understand why grid borders hit during simulations
"""

using Distributed
addprocs(10) # Get to 18 procs

@everywhere begin
    using Parameters
    using QuantEcon
    using Interpolations
    using Plots
    using Statistics
    using Roots
    using Random
    using Distributions
    using ValidatedNumerics
    using NearestNeighbors
    using NLsolve
    using StatsBase
    using LaTeXStrings
    using SharedArrays
    using Roots
    using Logging
    using DataFrames
    using FixedEffectModels
    using Feather
    using PrettyTables
    using Metaheuristics
end

@everywhere include("utils.jl")

@everywhere function AgencyModel(θ_)
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
    nϵ  = 5  # ϵ grid points
    nc  = 11 # Cash point
    _nk = 10 # intermediate points in the capital grid
    _nc = 10 # intermediate points in the cash grid

    stdbound = 4

    # Grid for epsilon

    mc_ϵ = tauchen(nϵ, ρ, σ, 0, stdbound)
    Π_ϵ  = mc_ϵ.p
    ϵ_grid = exp.(collect(mc_ϵ.state_values))

    # Compute the capital stock around the steady state
    #k_steady = ((θ)/(r+δ))^(1/(1-θ))
    k_steady = ((θ*(1-τ))/(r*(1+a*δ)-δ*τ+δ+0.5*a*δ^2))^(1/(1-θ))

    # K_steady with adjustment costs? Is actually lower than the normal k steady

    #k_max    = find_zero(k ->  (α+β)*ϵ_grid[5]*k^θ - δ*β*k, (1e-6, Inf))

    k_min    = ((θ*(1-τ)*ϵ_grid[1])/(r*(1+a*δ)-δ*τ+δ+0.5*a*δ^2))^(1/(1-θ)) # A guess for k_min?

    k_max    = ((θ*(1-τ)*ϵ_grid[end-1])/(r*(1+a*δ)-δ*τ+δ+0.5*a*δ^2))^(1/(1-θ)) # A guess for k_max?

    k_grid   = LinRange(k_min, k_max, nk)
    kp_grid  = LinRange(k_min, k_max, nk*_nk)

    nkp     = size(kp_grid)[1]

    c_max   = k_max/2 #4*((θ*(1-τ))/(r*(1+a*δ)-δ*τ+δ+0.5*a*δ^2))^(1/(1-θ))
    c_grid  = LinRange(0.0, c_max, nc)

    cp_grid = LinRange(0.0, c_max, _nc*nc)

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

@everywhere function howard_step!(U, R, i_kp, i_cp, am)
    nϵ, nc, nk, nkp, ncp, k_grid, c_grid, ϵ_grid, kp_grid, cp_grid, Π_ϵ = am.nϵ, am.nc, am.nk, am.nkp, am.ncp, am.k_grid, am.c_grid, am.ϵ_grid, am.kp_grid, am.cp_grid, am.Π_ϵ
    r = am.r
    UU = copy(U)
    U_func = LinearInterpolation((k_grid, c_grid, ϵ_grid), UU, extrapolation_bc=Flat())
    for (i_ϵ, ϵ) in enumerate(ϵ_grid)
        for (i_k, k) in enumerate(k_grid)
            for (i_c, c) in enumerate(c_grid)
                # Given policy i_kp and i_cp, what is the continuation value?
                _ikp = i_kp[i_k, i_c, i_ϵ]
                _icp = i_cp[i_k, i_c, i_ϵ]
                EV = dot(Π_ϵ[i_ϵ, :], [U_func(kp_grid[_ikp], cp_grid[_icp], ϵ_grid[i_ϵp]) for i_ϵp=1:nϵ])
                U[i_k, i_c, i_ϵ]     = R[i_k, _ikp, i_c, _icp, i_ϵ] + (1/(1+r))*EV
            end
        end
    end
end

@everywhere function howard_stepV!(V, D, i_kp, i_cp, am)
    nϵ, nc, nk, nkp, ncp, k_grid, c_grid, ϵ_grid, kp_grid, cp_grid, Π_ϵ = am.nϵ, am.nc, am.nk, am.nkp, am.ncp, am.k_grid, am.c_grid, am.ϵ_grid, am.kp_grid, am.cp_grid, am.Π_ϵ
    r = am.r
    VV = copy(V)
    V_func = LinearInterpolation((k_grid, c_grid, ϵ_grid), VV, extrapolation_bc=Flat())
    for (i_ϵ, ϵ) in enumerate(ϵ_grid)
        for (i_k, k) in enumerate(k_grid)
            for (i_c, c) in enumerate(c_grid)
                # Given policy i_kp and i_cp, what is the continuation value?
                _ikp = i_kp[i_k, i_c, i_ϵ]
                _icp = i_cp[i_k, i_c, i_ϵ]
                EV = dot(Π_ϵ[i_ϵ, :], [V_func(kp_grid[_ikp], cp_grid[_icp], ϵ_grid[i_ϵp]) for i_ϵp=1:nϵ])
                V[i_k, i_c, i_ϵ]     = D[i_k, _ikp, i_c, _icp, i_ϵ] + (1/(1+r))*EV
            end
        end
    end
end




@everywhere function bellman_operator!(U, R, i_kp, i_cp,  am)

    nϵ, nc, nk, nkp, ncp, k_grid, c_grid, ϵ_grid, kp_grid, cp_grid, Π_ϵ = am.nϵ, am.nc, am.nk, am.nkp, am.ncp, am.k_grid, am.c_grid, am.ϵ_grid, am.kp_grid, am.cp_grid, am.Π_ϵ
    r = am.r
    UU= copy(U)

    # Interpolate the value function
    U_func = LinearInterpolation((k_grid, c_grid, ϵ_grid), UU, extrapolation_bc=Flat())
    for (i_ϵ, ϵ) in enumerate(ϵ_grid)
        EV = [dot(Π_ϵ[i_ϵ, :], [U_func(kp_grid[_ikp], cp_grid[_icp], ϵ_grid[i_ϵp]) for i_ϵp=1:nϵ]) for _ikp=1:nkp, _icp=1:ncp]
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



@everywhere function vfi(am; tol=1E-6, itmax = 100000, verbose=true, H=10)

    if verbose
        println("Solving the model via Value Function Iteration")
        println("Max Iterations: $(itmax)")
        println("Steps in Howard improvement: $(H)")
    end
    # An initial guess is the value function of the steady state
    ϵ_grid, k_grid, c_grid = am.ϵ_grid, am.k_grid, am.c_grid
    kp_grid, cp_grid = am.kp_grid, am.cp_grid
    Π_ϵ = am.Π_ϵ
    R= am.R
    D = am.D
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
            howard_step!(U, R, i_kp, i_cp, am)
        end

        bellman_operator!(U, R, i_kp, i_cp, am)

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

    # return also the values themselves, the dividends in each state
    DD = copy(U)
    for (i_ϵ, ϵ) in enumerate(ϵ_grid)
        for (i_c, c) in enumerate(c_grid)
            for (i_k, k) in enumerate(k_grid)
                i_kp_ = i_kp[i_k, i_c, i_ϵ]
                i_cp_ = i_cp[i_k, i_c, i_ϵ]
                DD[i_k, i_c, i_ϵ] = D[i_k, i_kp_, i_c, i_cp_, i_ϵ]
            end
        end
    end


    return U, Kp, Cp, i_kp, i_cp, R, D, DD
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
                F_p[i_k,i_c,i_ϵ]        = d*(1+(d<0)*ϕ)/(c_grid[i_c]+k_grid[i_k])
            end
        end
    end


    # Here we check if it hits the grid on the policy function
    if maximum(Cp) >= maximum(c_grid)
        println("Hitting the grid with β = $β")
    end
    # What is the kstar?
    i_ks = argmin(abs.(k_grid.-k_steady))[1]

    cf_k = ((1-τ)*ϵ_grid*k_steady^θ)/k_steady
    p1=plot(legend=:topleft)
    plot!(log.(ϵ_grid), cf_k, label = "Low",     linestyle = :dash,  color = :black)
    plot!(log.(ϵ_grid), cf_k, label = "Medium",  linestyle = :solid,  color = :black)
    plot!(log.(ϵ_grid), cf_k, label = "High",    linestyle = :dot,  color = :black)
    ylabel!("Cash Flow / Capital")
    xlabel!("Log productivity shock")
    hline!([0], color = :black, label = "")
    # Investment over capital

    p2=plot(legend=:topleft)
    plot!(log.(ϵ_grid), I_p[i_ks, 2, :], label = "Low",    linestyle = :dash,  color = :black)
    plot!(log.(ϵ_grid), I_p[i_ks, 3, :], label = "Medium", linestyle = :solid, color = :black)
    plot!(log.(ϵ_grid), I_p[i_ks, 4, :], label = "High",   linestyle = :dot,   color = :black)
    ylabel!("Investment / Capital")
    xlabel!("Log productivity shock")
    hline!([0], color = :black, label = "")

    # Cash over assets
    p3=plot(legend=:topleft)
    plot!(log.(ϵ_grid), CRatio_P[i_ks, 2, :], label = "Low",     linestyle = :dash,  color = :black)
    plot!(log.(ϵ_grid), CRatio_P[i_ks, 3, :], label = "Medium",  linestyle = :solid,  color = :black)
    plot!(log.(ϵ_grid), CRatio_P[i_ks, 4, :], label = "High",    linestyle = :dot,  color = :black)
    ylabel!("Cash / Assets")
    xlabel!("Log productivity shock")
    hline!([0], color = :black, label = "")


    p4=plot(legend = :topleft)
    plot!(log.(ϵ_grid), F_p[i_ks, 3, :], label = "Low",     linestyle = :dash,  color = :black)
    plot!(log.(ϵ_grid), F_p[i_ks, 5, :], label = "Medium",  linestyle = :solid,  color = :black)
    plot!(log.(ϵ_grid), F_p[i_ks, 7, :], label = "High",    linestyle = :dot,  color = :black)
    ylabel!("External Fin / Assets")
    xlabel!("Log productivity shock")
    hline!([0], color = :black, label = "")

    plot(p1,p2,p3,p4, layout = (4,1), legendfontsize=5,size = (1000, 1000),show=true)
    savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\policy_functions")


    # Surface plot
    plot(zlabel=L"\frac{I}{k}")
    plot!(log.(ϵ_grid), k_grid, I_p[:, 2, :], st = :surface, camera=(80,30), label = "Low", size=(1000,1000), c=:blues)
    plot!(log.(ϵ_grid), k_grid, I_p[:, 3, :], st = :surface, camera=(80,30), label = "Medium", size=(1000,1000), c=:blues)
    plot!(log.(ϵ_grid), k_grid, I_p[:, 4, :], st = :surface, camera=(80,30), label = "High", size=(1000,1000), c=:blues)

    xlabel!(L"ln(\epsilon)")
    ylabel!(L"k")
    #savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\investment")
end


@everywhere function simulate_ϵ(ϵ0, ϕ_0, ϕ_1, σ, z)
    ϵs = zeros(size(z)[1]+1)
    ϵs[1] = ϵ0
    for i=2:size(ϵs)[1]
        ϵs[i] = exp(ϕ_0 + ϕ_1*log(ϵs[i-1])+σ*z[i-1])
    end
    return ϵs
end

@everywhere function simulate_economy(am, Kp, Cp, V, DD; Terg = 20, Tsim = 1000, N = 50, verbose = true)

    @unpack θ, α, β, s, δ, τ, r, ρ, σ, ϕ, a, nk, nϵ, nc, nkp, ncp, Π_ϵ, k_grid, ϵ_grid, c_grid, kp_grid, cp_grid, k_steady = am;
    # Simulate the shocks
    T = Terg + Tsim +1

    Random.seed!(99999)
    # Create some gaussian variables as input for the AR simulations
    d = Normal()
    zs = [rand(d, T-1) for i=1:N]

    #nϵs = nϵ
    #mc = tauchen(nϵs, ρ, σ, 0, 3)
    #ϵ_grids = mc.state_values
    #ϵ_grids = exp.(collect(ϵ_grids))
    #mc = MarkovChain(mc.p)

    # Initial points from the stationary distribution
    #μ = mc.p^10
    #μ = μ[1,:]

    #init_points = sample(1:nϵs, Weights(μ), N)

    ϵs=[simulate_ϵ(1.0, 0, ρ, σ, z) for z in zs]

    #E  = [simulate(mc, T, init =  init_points[i]) for i=1:N]

    #ϵs = [ϵ_grids[e[t]] for t=1:T, e in E]
    #i_ϵs = [e[t] for t=1:T, e in E]

    # Simulate how capital and cash change
    Kp_func = LinearInterpolation((k_grid, c_grid, ϵ_grid), Kp, extrapolation_bc=Flat())
    Cp_func = LinearInterpolation((k_grid, c_grid, ϵ_grid), Cp, extrapolation_bc=Flat())
    V_func  = LinearInterpolation((k_grid, c_grid, ϵ_grid), V, extrapolation_bc=Flat())
    D_func  = LinearInterpolation((k_grid, c_grid, ϵ_grid), DD, extrapolation_bc=Flat())
    Ksim  = zeros(T, N)
    Csim  = zeros(T, N)
    OInc  = zeros(T, N)
    Dsim     = zeros(T, N)
    F_p     = zeros(T, N)
    V_val = zeros(T, N)
    Ksim[1, :] .= k_steady

    for n=1:N
        for t=2:T
            Ksim[t, n]  = Kp_func(Ksim[t-1, n], Csim[t-1, n], ϵs[n][t-1])
            Csim[t, n]  = Cp_func(Ksim[t-1, n], Csim[t-1, n], ϵs[n][t-1])
            V_val[t, n] = V_func(Ksim[t-1, n], Csim[t-1, n], ϵs[n][t-1])
            OInc[t-1, n] = ϵs[n][t-1]*Ksim[t-1, n]^θ
            # Compute dividends and external finance
            Dsim[t-1, n]   =   maximum([D_func(Ksim[t-1, n], Csim[t-1, n], ϵs[n][t-1]),0.0])
            F_p[t-1, n]    = - minimum([D_func(Ksim[t-1, n], Csim[t-1, n], ϵs[n][t-1]) ,0.0])
        end
    end

    verbose && println("Simulation Check")
    verbose && println("         lb            min         max        ub")
    verbose && println("Capital: $(round(minimum(k_grid), digits=1))           $(round(minimum(Ksim), digits=1))       $(round(maximum(Ksim), digits=1))       $(round(maximum(k_grid), digits=1))")
    verbose && println("Cash:    $(round(minimum(c_grid), digits=1))           $(round(minimum(Csim), digits=1))       $(round(maximum(Csim), digits=1))       $(round(maximum(c_grid), digits=1))")

    # Remove the burning period
    Ksim= Ksim[(Terg+1):end, :]
    Csim = Csim[(Terg+1):end, :]
    OInc = OInc[(Terg+1):end, :]
    V_val = V_val[(Terg+1):end, :]
    Dsim  = Dsim[(Terg+1):end, :]
    F_p   = F_p[(Terg+1):end, :]

    c_a = Csim[1:end-1, :]./((Csim[1:end-1, :] .+ Ksim[1:end-1, :]))

    I_k = (Ksim[2:end,:] .- (1-δ)*Ksim[1:end-1, :])./Ksim[1:end-1, :]
    oi_a =  OInc[1:end-1, :] ./((Csim[1:end-1, :] .+ Ksim[1:end-1, :]))

    q = V_val[2:end, :] ./(Csim[1:end-1, :] .+ Ksim[1:end-1, :])

    d_a = Dsim[1:end-1, :] ./(Csim[1:end-1, :] .+ Ksim[1:end-1, :])
    f_a = (1/(1+ϕ))*F_p[1:end-1, :] ./(Csim[1:end-1, :] .+ Ksim[1:end-1, :])

    # We put it all in a dataframe so that we can compute autoregressive coefficients
    df = DataFrame()
    # For the time we repeat a vector from 1:T N times
    df[!, "time"]  = repeat(collect(1:size(Ksim)[1]-1), outer = N)
    df[!, "firm"]  = repeat(collect(1:N), inner = size(Ksim)[1]-1)
    df[!, "oi_a"]  = vec(oi_a)
    df[!, "I_k"]   = vec(I_k)
    df[!, "c_a"]   = vec(c_a)
    df[!, "Foi_a"] = Union{Missing, Float64}[missing for i in eachrow(df)]
    df[!, "FI_k"]  = Union{Missing, Float64}[missing for i in eachrow(df)]
    df[!, "Fc_a"]  = Union{Missing, Float64}[missing for i in eachrow(df)]

    for n=1:N
        temp =  df[df.firm .== n, "oi_a"]
        df[(df.firm .== n) .* (df.time .< size(Ksim)[1]-1), "Foi_a"] = temp[2:end]

        temp =  df[df.firm .== n, "I_k"]
        df[(df.firm .== n) .* (df.time .< size(Ksim)[1]-1), "FI_k"] = temp[2:end]

        temp =  df[df.firm .== n, "c_a"]
        df[(df.firm .== n) .* (df.time .< size(Ksim)[1]-1), "Fc_a"] = temp[2:end]
    end

    fem = FixedEffectModels.reg(df, @formula(Foi_a ~ oi_a + fe(firm)))
    auto_oia = fem.coef[1]

    temp = df[completecases(df), :]
    ser_inv = cor(temp.FI_k, temp.I_k)
    ser_ca  = cor(temp.Fc_a, temp.c_a)


    # returns also a flag in case the grid hits
    flag = 0

    return [mean(I_k),       # First Moment of the investment rate, helps pin δ
            var(I_k),        # The second moment helps identify both the curvature of the profit function, θ, and the investment adjustment cost parameter, a.
            mean(oi_a),      # Helps also to pin down θ
            var(oi_a),       # We match the variance of operating income,
            auto_oia,        # as well as the autoregressive coefficient from a first-order panel autoregression of operating income on lagged operating income.
            ser_inv,         # Serial correlation of investment
            mean(c_a),       # mean cash
            var(c_a),        # variance cash
            ser_ca,          # ser auto corr of cash
            mean(d_a),         # mean of dividends
            var(d_a),          # Variance dividends
            mean(f_a),       # Mean external finance
            var(f_a),        # Variance external finance
            mean(q),         # Mean of Tobin's q
            var(q),          # Var of Tobin's q
            α*mean(oi_a)*100 # Average bonus
            ]
end

@everywhere function compute_moments(θ_; display_moments = false)
    am = AgencyModel(θ_);

    @time  U, Kp, Cp,  i_kp, i_cp, R, D, DD= vfi(am; H=20, verbose=false);

    #policy_functions(Kp, Cp, am);
    V = compute_value_function(D, i_kp, i_cp, am);
    moments = simulate_economy(am, Kp, Cp, V, DD);
    display_moments && verbose_moments(moments)
    return moments
end

#!TODO Use real moments, for now Im using the simulated from NW
@everywhere function Q(moments; real_moments = [0.133, 0.004, 0.936, 0.131, 0.003, 0.549, 0.148, 0.003, 0.707, 1.949, 0.016, 0.024, 0.001, 0.039, 0.001, 0.111], W = I )
    return (moments-real_moments)'*W*(moments-real_moments)
end

@everywhere function obj(θ_)
    am = AgencyModel(θ_);
    U, Kp, Cp,  i_kp, i_cp, R, D = vfi(am; H=20, verbose=false);
    V = compute_value_function(D, i_kp, i_cp, am);
    moments = simulate_economy(am, Kp, Cp, V, D);
    return Q(moments)
end

@everywhere function compute_value_function(D, i_kp, i_cp, am; tol=1e-5, maxit = 1000)
    @unpack θ, α, β, s, δ, τ, r, ρ, σ, ϕ, a, nk, nϵ, nc, nkp, ncp, Π_ϵ, k_grid, ϵ_grid, c_grid, kp_grid, cp_grid, k_steady = am;
    V = zeros(nk, nc, nϵ)
    V_old  = copy(V)
    for i=1:maxit
        howard_stepV!(V, D, i_kp, i_cp, am)
        v_tol = maximum(abs.(V-V_old))

        if (v_tol < tol)
            break
        end

        V_old  = copy(V)
    end
    return V
end



@everywhere function verbose_moments(moments)

df = DataFrame()
names = ["Average cash", "Variance of cash", "Serial correlation of cash", "Average Investment",
         "Variance of Investment", "Serial correlation of Investment", "Average Profits", "Error variance profits",
         "Serial correlation of profits", "Average Tobin's q", "Variance Tobin's q", "Average external finance",
         "Variance external finance", "Average of distributions", "Variance of distributions", "Average bonus"]

order_values = [7, 8, 9, 1, 2, 6, 3, 4, 5, 14, 15, 12, 13, 10, 11, 16]
df[!, "Name"] = names
df[!, "Simulated Moment"] = round.(moments[order_values],  digits=3)
df[!, "Simulated Moment NW2014"] = round.([0.133, 0.004, 0.936, 0.131, 0.003, 0.549, 0.148, 0.003, 0.707, 1.949, 0.016, 0.024, 0.001, 0.039, 0.001, 0.111] ,digits=3)
println("Similarity to Table IV ")
println(pretty_table(df))
end

@everywhere function single_comparative_statistics(θ_, npos, gridj)
    θ_temp = copy(θ_)
    θ_temp[npos] = gridj
    moments = compute_moments(θ_temp)
    return moments[7] # Picks cash
end

@everywhere function comparative_statistics(θ_; N = 20, subset = [])

    grids=[LinRange(0.0, 0.25, N), #grid_ϕ
           LinRange(0.1, 0.4,  N),  #grid_σ
           LinRange(0.5, 0.75, N), #grid_ρ
           LinRange(0.5, 0.8,  N),  #grid_θ
           LinRange(0, 1.5,    N),    #grid_a
           LinRange(0.05, 0.2, N), #grid_δ
           LinRange(0, 100/10000,   N),   #grid_α
           LinRange(0, 2/10000, N), #grid_s
           LinRange(0.04, 0.1, N)]    #grid_β

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
        #if i==9
        println("Computing comparative statistics for: $(names_graph[i])")
        c_a=SharedArray{Float64}(size(grid)[1])
        if i in subset
            @sync @distributed for j=1:size(c_a)[1]
                c_a[j]=single_comparative_statistics(θ_, positions_in_θ[i], grid[j])
            end
        end
        push!(c_as, c_a)
        #end
    end

    i=1
    p1 = plot()
    if i in subset
        p1 = plot(label ="")
        temp_y = copy(c_as[i])
        temp_x = copy(grids[i])
        plot!(temp_x, temp_y, ylims = (0.0,0.5), label ="")
        ylabel!("Cash / Assets")
        xlabel!(names_graph[i])
        savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\cs_$i")
    end

    i=2
    p2 = plot()
    if i in subset
        p2 = plot(label ="")
        temp_y = copy(c_as[i])
        temp_x = copy(grids[i])
        plot!(temp_x, temp_y, ylims = (0.0,0.5), label ="")
        ylabel!("Cash / Assets")
        xlabel!(names_graph[i])
        savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\cs_$i")
    end

    i=3
    p3 = plot()
    if i in subset
        p3 = plot(label ="")
        temp_y = copy(c_as[i])
        temp_x = copy(grids[i])
        plot!(temp_x, temp_y, ylims = (0.0,0.5), label ="")
        ylabel!("Cash / Assets")
        xlabel!(names_graph[i])
        savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\cs_$i")
    end

    i=4
    p4 = plot()
    if i in subset
        p4 = plot(label ="")
        temp_y = copy(c_as[i])
        temp_x = copy(grids[i])
        plot!(temp_x, temp_y, ylims = (0.0,0.5), label ="")
        ylabel!("Cash / Assets")
        xlabel!(names_graph[i])
        savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\cs_$i")
    end

    i=5
    p5 = plot()
    if i in subset
        p5 = plot(label ="")
        temp_y = copy(c_as[i])
        temp_x = copy(grids[i])
        plot!(temp_x, temp_y, ylims = (0.0,0.5), label ="")
        ylabel!("Cash / Assets")
        xlabel!(names_graph[i])
        savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\cs_$i")
    end

    i=6
    p6 = plot()
    if i in subset
        p6 = plot(label ="")
        temp_y = copy(c_as[i])
        temp_x = copy(grids[i])
        plot!(temp_x, temp_y, ylims = (0.0,0.5), label ="")
        ylabel!("Cash / Assets")
        xlabel!(names_graph[i])
        savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\cs_$i")
    end

    i=7
    p7 = plot()
    if i in subset
        p7 = plot(label ="")
        temp_y = copy(c_as[i])
        temp_x = copy(grids[i])
        plot!(temp_x, temp_y, ylims = (0.0,0.5), label ="")
        ylabel!("Cash / Assets")
        xlabel!(names_graph[i])
        savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\cs_$i")
    end

    i=8
    p8 = plot()
    if i in subset
        p8 = plot(label ="")
        temp_y = copy(c_as[i])
        temp_x = copy(grids[i])
        plot!(temp_x, temp_y, ylims = (0.0,0.5), label ="")
        ylabel!("Cash / Assets")
        xlabel!(names_graph[i])
        savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\cs_$i")
    end

    i=9
    p9 = plot()
    if i in subset
        p9 = plot(label ="")
        temp_y = copy(c_as[i])
        temp_x = copy(grids[i])
        plot!(temp_x, temp_y, ylims = (0.0,0.5), label ="")
        ylabel!("Cash / Assets")
        xlabel!(names_graph[i])
        savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\cs_$i")
    end

    plot(p1, p2, p3, p4, p5, p6, p9, p7, p8, layout = (3,3), legend = false, size=(1000,2000))
    savefig("$(pwd())\\Work\\Code Juan\\Nikolov Whited 2014\\comp_stat")
end

"""
    Computes the moments in the data
"""
function moments_data()

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

    @time comparative_statistics(θ_, N = 20, subset = [1, 2,3,4,5,6,7,8,9]) #,
    #compute_moments(θ_)
    # use the bouonds on the grid for the comparative statistics in the SMM parameters that do not enter to the SMM estimation have their bounds binding
    bounds = [0.5 0.0   0.04 0.0     0.130 0.2 0.011 0.5  0.1 0.0  0.0; # Lower bounds,
              0.8 1/100 0.1  2/10000 0.130 0.2 0.011 0.75 0.4 0.25 1.5
    ]


    #result = optimize(obj, bounds, PSO(), logger = (status) -> println(status.best_sol))

    #println(result)
end

main()
