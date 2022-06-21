# -*- coding: utf-8 -*-
import numpy as np
import quantecon as qe
from scipy.interpolate import interpn
from multiprocessing import Pool
from itertools import repeat
import matplotlib.pyplot as plt

import value_interation as vi

def model_sim(param_dim,param_ar,kstar,z_prob_mat ,k_vec,c_vec,z_vec, kp_vec, cp_vec, Kp, Cp, N=20, Ttot=5_000,Terg=200):
        """
        # Model simulation.
        """
        mc = qe.MarkovChain(np.transpose(z_prob_mat))
        E = mc.simulate(ts_length=Ttot, num_reps=N).T
        Ksim = np.zeros((Ttot, N))
        Csim = np.zeros((Ttot, N))
        # the simulations start with the value of the central element in k_vec and c_vec,
        Ksim[0, :]=kstar*np.ones((1, N))
        # interpolations [it does not work, because in some simulations "previous_state" falls out the grid points]
        # points = (np.reshape(k_vec, k_vec.size), np.reshape(
        #     c_vec, c_vec.size), np.reshape(z_vec, z_vec.size))
        # for n in range(N):
        #     for t in range(1, Ttot):
        #         if Ksim[t-1, n] < k_vec[0]:
        #             print(
        #                 f"Alert, previus in {t-1} and {n}; Ksim[t-1, n]< k_vec[0]")
        #             print(f"{Ksim[t-1, n]} and {k_vec[0]}")
        #             Ksim[t-1, n] = np.copy(k_vec[0])
        #         elif Ksim[t-1, n] > k_vec[-1]:
        #             print(
        #                 f"alert, previus in {t-1} and {n}; Ksim[t-1, n]> k_vec[-1]")
        #             print(f"{Ksim[t-1, n]} and {k_vec[-1]}")
        #             Ksim[t-1, n] = np.copy(k_vec[-1])
        #         previus_state = [Ksim[t-1, n], Csim[t-1, n], sim_z[t-1, n]]
        #         Ksim[t, n] = interpn(points, Kp, previus_state)
        #         Csim[t, n] = interpn(points, Cp, previus_state)
        given_value = 0
        search_list = []
        absolute_difference_function = lambda list_value : abs(list_value - given_value)
        for n in range(N):
            for t in range(1,Ttot):
                # find previous K
                given_value=Ksim[t-1, n]
                search_list=k_vec
                closest_k=min(search_list, key=absolute_difference_function)
                closest_k_position=np.take(np.where(search_list==closest_k),0)
                # find previous C
                given_value=Csim[t-1, n]
                search_list=c_vec
                closest_c=min(search_list, key=absolute_difference_function)
                closest_c_position=np.take(np.where(search_list==closest_c),0)
                # find previus shock
                id_z = E[t-1,n]
                # compute the respective policy depending of the position in t-1
                Ksim[t, n] = Kp[closest_k_position,closest_c_position, id_z] 
                Csim[t, n] = Cp[closest_k_position,closest_c_position, id_z]
        # Remove the burning period
        Ksim = Ksim[(Terg+1):-1, :]
        Csim = Csim[(Terg+1):-1, :]
        
        
        print()
        print("Quick simulation check: p lb, min(psim), max(psim), p ub \n")
        print(f"K = {k_vec[0]} {np.min(Ksim):,.2f} {np.max(Ksim):,.2f} {k_vec[-1]} \n")
        print(f"C = {c_vec[0]} {np.min(Csim):,.2f} {np.max(Csim):,.2f} {c_vec[-1]} \n")
        print()
       
        return Ksim, Csim

def run_comparative_stats(variable,param_iter):
    
    param_manager     = (0.751/100, 0.051, 0.101/1000) # (α, β, s) 
    param_inv         = (0.130, 0, 1.278, 0.773 , 0.2) # (δ, λ, a, θ, τ)   
    param_fin         = (0.011, 0.043)                 # (r, ϕ=0.043)
    param_ar          = (0, 0.262, 0.713, 4)           # (μ, σ, ρ, stdbound)
    _nk               = 10                              # intermediate points in the capital grid
    _nc               = 10                              # intermediate points in the cash grid
    (dimc, dimk, dimz) = (11, 25, 5)
    param_dim         = (dimz,dimk,dimc,dimk*_nk,dimc*_nc)         # wrapping dimensional parameters: (dimz, dimk, dimc, dimkp, dimcp)
    if variable =='ϕ':
        param_fin                   = (0.011, param_iter)                 # (r, ϕ)
    elif variable=='θ':     
        param_inv                   = (0.130, 0, 1.278, param_iter, 0.2)  # (δ, λ, a, θ, τ)
    elif variable=='a':     
        param_inv                   = (0.130, 0, param_iter, 0.773 , 0.2) # (δ, λ, a, θ, τ)
    elif variable=='δ':     
        param_inv                   = (param_iter, 0, 1.278, 0.773 , 0.2)  # (δ, λ, a, θ, τ)
    elif variable=='σ':
        param_ar                    = (0, param_iter, 0.713, 4)           # (μ, σ, ρ, stdbound)
    elif variable=='ρ':
        param_ar                    = (0, 0.262, param_iter, 4)           # (μ, σ, ρ, stdbound)
    elif variable=='α':
        param_manager               = (param_iter,0.051, 0.101/1000)      # (α, β, s)
    elif variable=='s':
        param_manager               = (0.751/100,0.051, param_iter)       # (α, β, s)
    elif variable=='β':
        param_manager               = (0.751/100,param_iter, 0.101/1000)  # (α, β, s)      
    else:
        print("not known variable")
        breakpoint()      
    z_vec, z_prob_mat                  = vi.trans_matrix(param_ar,param_dim)   
    k_vec, kp_vec, c_vec, cp_vec, kstar,grid_points,grid_to_interp= vi.set_vec(param_inv, param_fin, param_dim, param_manager,z_vec)
    R, D                               = vi.rewards_grids(param_manager, param_inv, param_fin, param_dim, z_vec, k_vec, c_vec, kp_vec, cp_vec)  
    Upol,Kpol,Cpol,i_kpol,i_cpol       = vi.value_iteration(param_dim,param_fin,R,z_prob_mat,k_vec,c_vec,z_vec,kp_vec,cp_vec,grid_points,grid_to_interp)
    sim_K, sim_C                         = model_sim(param_dim,param_ar,kstar,z_prob_mat ,k_vec,c_vec,z_vec, kp_vec, cp_vec, Kpol, Cpol)
    sim_Cratio                           = sim_C/(sim_K+sim_C)       
    sim_Cratio_av                        = np.mean(sim_Cratio)
    print(f"average cash ratio {variable}= {param_iter}: {sim_Cratio_av}")
    return sim_Cratio_av

def parallel_simulation(num_values_per_param):
    # Financing parameters
    ϕ_vec = np.linspace(0.0, 0.25, num_values_per_param)
    # Investment parameters
    θ_vec = np.linspace(0.5, 0.9, num_values_per_param)
    a_vec = np.linspace(0, 1.5, num_values_per_param)
    δ_vec = np.linspace(0.05, 0.2, num_values_per_param)
    # Manager parameters
    β_vec = np.linspace(0.01, 0.1, num_values_per_param)
    α_vec = np.linspace(0, 0.01, num_values_per_param)
    s_vec = np.linspace(0, 0.02/100, num_values_per_param)
    # AR parameters
    σ_vec = np.linspace(0.1, 0.5, num_values_per_param)
    ρ_vec = np.linspace(0.5, 0.75, num_values_per_param)
    
    print("--- First column of plots: ϕ θ β---")
    with Pool() as pool:
        com_stat_cash = pool.starmap(
            run_comparative_stats, zip(repeat('ϕ'), ϕ_vec))
    with Pool() as pool2:
        com_stat_cash2 = pool2.starmap(
            run_comparative_stats, zip(repeat('θ'), θ_vec))
    with Pool() as pool3:
        com_stat_cash3 = pool3.starmap(
            run_comparative_stats, zip(repeat('β'), β_vec))

    print("--- Second column of plots: σ a α ---")
    with Pool() as pool4:
        com_stat_cash4 = pool4.starmap(
            run_comparative_stats, zip(repeat('σ'), σ_vec))
    with Pool() as pool5:
        com_stat_cash5 = pool5.starmap(
            run_comparative_stats, zip(repeat('a'), a_vec))
    with Pool() as pool6:
        com_stat_cash6 = pool6.starmap(
            run_comparative_stats, zip(repeat('α'), α_vec))

    print("--- Third column of plots: ρ δ s ---")
    with Pool() as pool7:
        com_stat_cash7 = pool7.starmap(
            run_comparative_stats, zip(repeat('ρ'), ρ_vec))
    with Pool() as pool8:
        com_stat_cash8 = pool8.starmap(run_comparative_stats, zip(
            repeat('δ'), δ_vec))  # problem: opposite slope
    with Pool() as pool9:
        com_stat_cash9 = pool9.starmap(
            run_comparative_stats, zip(repeat('s'), s_vec))
    
    F2 = fig, ((ax1, ax4, ax7), (ax2, ax5, ax8), (ax3, ax6, ax9)) = plt.subplots(3, 3, sharey=True, figsize=(15, 15))
    ax1.plot(ϕ_vec, com_stat_cash, linestyle='dashed', c='b')
    ax1.set_xlabel("ϕ")
    ax2.plot(θ_vec, com_stat_cash2, linestyle='dashed', c='b')
    ax2.set_xlabel("θ")
    ax3.plot(β_vec, com_stat_cash3, linestyle='dashed', c='b')
    ax3.set_xlabel("β")
    ax4.plot(σ_vec, com_stat_cash4, linestyle='dashed', c='b')
    ax4.set_xlabel("σ")
    ax5.plot(a_vec, com_stat_cash5, linestyle='dashed', c='b')
    ax5.set_xlabel("a")
    ax6.plot(α_vec, com_stat_cash6, linestyle='dashed', c='b')
    ax6.set_xlabel("α")
    ax7.plot(ρ_vec, com_stat_cash7, linestyle='dashed', c='b')
    ax7.set_xlabel("ρ")
    ax8.plot(δ_vec, com_stat_cash8, linestyle='dashed', c='b')
    ax8.set_xlabel("δ")
    ax9.plot(s_vec, com_stat_cash9, linestyle='dashed', c='b')
    ax9.set_xlabel("s")
    plt.show()
    fig.savefig("Figure2.png", bbox_inches='tight', dpi=600)
