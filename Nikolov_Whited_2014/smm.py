# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import multi_dot
import pandas as pd
import numpy.typing as npt
from linearmodels import PanelOLS
import pyswarms as ps
from scipy.optimize import basinhopping

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
    [α, β, s, a, θ, ϕ, σ, ρ] =   param_   # if running GlobalBestPSO, use "param_.T"
    # Parameters that are not estimated
    δ:float=.13
    λ:float=0.0
    τ:float=0.2
    r:float=.011
    μ:float=0.0
    stdbound:float=4.0
    param_others=[δ,λ,τ,r,μ,stdbound]
    # Packing vectors of parameters to estimate 
    param_manager =[α, β, s]
    param_inv    = [δ, λ, a, θ, τ]   
    param_fin    = [r, ϕ]             
    param_ar     = [μ, σ, ρ, stdbound]           
                                         
    _nk = 10                             # intermediate points in the capital grid
    _nc = 10                             # intermediate points in the cash grid
    (dimc, dimk, dimz) = (11, 25, 5)
    param_dim          = (dimz, dimk, dimc, dimk*_nk, dimc*_nc)
    
    # compute economy's matrices
    z_vec, z_prob_mat = vi.trans_matrix(param_ar, param_dim)
    k_vec, kp_vec, c_vec, cp_vec, kstar, grid_points, grid_to_interp = vi.set_vec(param_inv, param_fin, param_dim, param_manager, z_vec)
    R, D = vi.rewards_grids(param_manager, param_inv, param_fin, param_dim, z_vec, k_vec, c_vec, kp_vec, cp_vec)
   
    # value iteration functions
    Upol, Kpol, Cpol, i_kpol, i_cpol          =  vi. value_iteration(param_dim, param_fin, R, z_prob_mat, k_vec, c_vec, z_vec, kp_vec, cp_vec, grid_points, grid_to_interp)
    Vpol= vi.value_iteration_firm_value(param_dim, param_fin, D, z_prob_mat, k_vec, c_vec, z_vec, i_kpol, i_cpol, grid_points, grid_to_interp)
    
    # simulation
    Ksim, Csim, E, I_k, D, F, Op_Inc,C_ratio,TobinsQ      = sim.model_sim(param_,param_others ,kstar,z_prob_mat,z_vec ,k_vec,c_vec, Kpol, Cpol,Vpol)                                                     
    
    # compute moments  
    cash_m1,cash_m2, cash_t1 = sim_mean_and_t(C_ratio)
    inv_m1, inv_m2, inv_t1  = sim_mean_and_t(I_k)
    opincome_m1, opincome_m2,opincome_t1  = sim_mean_and_t(Op_Inc)
    tobin_m1, tobin_m2, tobin_t1 = sim_mean_and_t(TobinsQ)
    f_m1, f_m2, f_t1 = sim_mean_and_t(F)
    d_m1,d_m2,d_t1 = sim_mean_and_t(D)
    bonus_m1 = α*opincome_m1*100
    
    # Serial correlation with fixed effects
    df_cash_ratio=pd.DataFrame(C_ratio)
    df_cash_ratio_lag= df_cash_ratio.shift(-1)   
    mod = PanelOLS(df_cash_ratio, df_cash_ratio_lag, entity_effects=True)
    res = mod.fit()
    cash_m3=res.params.Exog    
    df_inv=pd.DataFrame(I_k)
    df_inv_lag= df_inv.shift(-1)   
    mod2 = PanelOLS(df_inv, df_inv_lag, entity_effects=True)
    res2 = mod2.fit()
    inv_m3=res2.params.Exog
    
    df_opincome=pd.DataFrame(Op_Inc)
    df_opincome_lag= df_opincome.shift(-1)   
    mod3 = PanelOLS(df_opincome, df_opincome_lag, entity_effects=True)
    res3 = mod3.fit()
    opincome_m3=res3.params.Exog
    
    moments = np.array([cash_m1,cash_m2, cash_m3,inv_m1, inv_m2,inv_m3, opincome_m1, opincome_m2,opincome_m3,tobin_m1, tobin_m2,f_m1, f_m2,d_m1, d_m2,bonus_m1 ])
    t_stats= np.array([cash_t1,inv_t1, opincome_t1, tobin_t1, f_t1,d_t1 ])
    return moments, t_stats

def sim_mean_and_t(simulated_panel: npt.ArrayLike )->float:
    mean=np.mean(simulated_panel)
    variance = np.var(simulated_panel) 
    t_stat_1=mean/(np.std(simulated_panel)/np.sqrt(simulated_panel.size))
    return mean,variance, t_stat_1

def obj_function(param_: npt.ArrayLike )->float:
    real_moments=np.array([0.133, 0.004, 0.936, 0.131, 0.003, 0.549, 0.148, 0.003, 0.707, 1.949, 0.016, 0.024, 0.001, 0.039, 0.001, 0.111])
    variance_matrix=np.eye(16)
    sim_moments, sim_t_stats = solve_simulate_and_moments(param_)    
    A=sim_moments-real_moments
    error=multi_dot([A.T,variance_matrix,A])
    return error

def optimizer(param_: npt.ArrayLike, method: str)->list:
    """
    This function receives a list of model parameters and optimization method.
    run the optimization and return the optimization solution.
    ***pending: the parallelization of PSO generate problems since it change a param= int to a param=list.
    
    Parameters
    ----------
    param_ : npt.ArrayLike
        DESCRIPTION: list with the parameters of the model.
     method: string
        DESCRIPTION:2 options: PSO, Basinhopping
    
    Returns
    -------
    ret: list
        DESCRIPTION: list of smm moments and other optimization output depending of the method.
    """   
    if method=="PSO":
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9,'k': (11, 15),'p': 1}
        max_bound =[1/100,0.1  ,2/10000 ,0.13 ,0 ,1.5,0.8, 0.2, 0.011, 0.25, 0 , 0.4, 0.75, 4]
        min_bound =[0.0  ,0.04 ,0       ,0.13 ,0 ,0  ,0.5, 0.2, 0.011, 0,    0 , 0.1, 0   , 4]
        bounds = (min_bound, max_bound)
        optimizer =  ps.single.GlobalBestPSO(n_particles=10, dimensions=14, options=options,bounds=bounds)
        ret = optimizer.optimize(obj_function, iters=1000)            
    elif method=="Basinhopping":
        minimizer_kwargs = {"method": "BFGS"}
        ret = basinhopping(obj_function, param_, minimizer_kwargs=minimizer_kwargs, niter=1_000)
    return ret


if __name__== '__main__':
    
    param_=(0.751/100, 0.051, 0.101/1000, 1.278, 0.773,  0.043, 0.262, 0.713 )
    sim_moments, sim_t_stats =solve_simulate_and_moments(param_)

    