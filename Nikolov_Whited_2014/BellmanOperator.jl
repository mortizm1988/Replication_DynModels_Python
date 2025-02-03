function bellman_operator(U, R, c_value, i_kp, i_cp, ϵ_grid, k_grid, c_grid,param_fin)
    r, _ = param_fin
    df=(1/(1+r))
    UU= copy(U)
    
    for (i_ϵ, ϵ) in enumerate(ϵ_grid)
        for (i_k, k) in enumerate(k_grid)
            for (i_c, c) in enumerate(c_grid)
                RHS                  = R[i_k, :, i_c, :, i_ϵ] + df*c_value[:, :, i_ϵ]
                i_kc                 = argmax(RHS)
                U[i_k, i_c, i_ϵ]     = RHS[i_kc]                                       # update U with all the best expected values.
                i_kp[i_k, i_c, i_ϵ]  = i_kc[1]
                i_cp[i_k, i_c, i_ϵ]  = i_kc[2]
            end
        end
    end
    return U, i_kp,i_cp
end
