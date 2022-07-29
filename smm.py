# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import numpy.typing as npt

import value_interation as vi
import simulation as sim

def solve_simulate_and_moments(param_: npt.ArrayLike)->list:
    """
    This function receives a list of model parameters and solves the model using Value iteration.
    Then, it uses the model solution (ie., optimal policies) to simulate and compute simulated moments.
    Finally, it returns the simulated moments on a list.

    Parameters
    ----------
    param_ : npt.ArrayLike
        DESCRIPTION: list with the parameters of the model.

    Returns
    -------
    list
        DESCRIPTION: list of simulated moments.

    """
    
    param_manager = param_[0:3]   # (α, β, s)
    param_inv    = param_[3:8]  # (δ, λ, a, θ, τ)
    param_fin    = param_[8:10]                 # (r, ϕ=0.043)
    param_ar     = param_[10:14]           # (μ, σ, ρ, stdbound)
    α, β, s        = param_manager     # Manager compensation
    δ, λ, a, θ, τ  = param_inv         # Investment parameters
    r, ϕ           = param_fin         # Financing Parameters 
    μ, σ, ρ, stdbound = param_ar                                      
    
    _nk = 10                             # intermediate points in the capital grid
    _nc = 10                             # intermediate points in the cash grid
    (dimc, dimk, dimz) = (11, 25, 5)
    param_dim          = (dimz, dimk, dimc, dimk*_nk, dimc*_nc)
    
    z_vec, z_prob_mat = vi.trans_matrix(param_ar, param_dim)
    k_vec, kp_vec, c_vec, cp_vec, kstar, grid_points, grid_to_interp = vi.set_vec(param_inv, param_fin, param_dim, param_manager, z_vec)
    R, D = vi.rewards_grids(param_manager, param_inv, param_fin, param_dim, z_vec, k_vec, c_vec, kp_vec, cp_vec)
   
    # value iteration functions
    Upol, Kpol, Cpol, i_kpol, i_cpol          =  vi. value_iteration(param_dim, param_fin, R, z_prob_mat, k_vec, c_vec, z_vec, kp_vec, cp_vec, grid_points, grid_to_interp)
    Vpol= vi.value_iteration_firm_value(param_dim, param_fin, D, z_prob_mat, k_vec, c_vec, z_vec, i_kpol, i_cpol, grid_points, grid_to_interp)
    
    # simulation
    Ksim, Csim, E, I_k, D, F, Op_Inc,C_ratio,TobinsQ      = sim.model_sim(param_, kstar,z_prob_mat,z_vec ,k_vec,c_vec, Kpol, Cpol,Vpol)                                                     
    
    # compute moments  
    cash_m1 = np.mean(C_ratio)
    cash_m2 = np.var(C_ratio)    
    inv_m1  = np.mean(I_k)
    inv_m2  = np.var(I_k)
    opincome_m1 = np.mean(Op_Inc)
    opincome_m2 =np.var(Op_Inc)
    tobin_m1 = np.mean(TobinsQ)
    tobin_m2 = np.var(TobinsQ)
    f_m1 = np.mean(F)
    f_m2 = np.var(F)
    d_m1 = np.mean(D)
    d_m2 = np.var(D)
   
    # df_opincome=pd.DataFrame(OperIncome,columns=['operative_inc'])
    # model_ar = AutoReg(df_opincome, lags=1)
    # model_fit = model_ar.fit()
    # print('Coefficients: %s' % model_fit.params)
    # serial correlation of investment
    # We include the mean, variance, and serial correlation of the ratio of cash to assets c/(k + c).
    # We also include the mean and variance of the ratios of distributions and security issuance to capital, d/(k + c)and f /(k + c).
    # We compute simulated Tobin’s q as V /(k + c), and we use both the mean and the variance of q.
    # Fortunately, this parameter corresponds directly to one moment from our compensation data: the average bonus.
    moments = (cash_m1,cash_m2 ,inv_m1, inv_m2, opincome_m1, opincome_m2,tobin_m1, tobin_m2,f_m1, f_m2,d_m1, d_m2)
    return moments