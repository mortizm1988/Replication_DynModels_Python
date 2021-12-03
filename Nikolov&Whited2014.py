#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 5 21:12:52 2021
@author: Marcelo Ortiz M @ UPF and BSE.
Notes: 35 sec with interp (x2) but crushes without interp (?).
get_ipython().run_line_magic('matplotlib', 'inline')  #for plots in different a window use 'qr'
"""
# In[1]: Import Packages and cleaning 
import numpy as np
import numpy.typing as npt
import quantecon as qe
from interpolation.splines import eval_linear, CGrid
import matplotlib.pyplot as plt
import math
from numba import jit,prange
from multiprocessing import Pool
from itertools import repeat

# In[2]: Method definitions
"""
List of Parameters:
     α,             # manager bonus
     β,             # manager equity share
     s,             # manager private benefit
     δ,             # capital depreciation
     λ,             # cost of issuing equity
     a,             # capital adjustment cost
     θ,             # curvature production function
     τ,             # corporate tax rate
     r,             # interest rate
     ϕ,             # cost of external finance
     dimz,          # dim of shock
     μ,             # log mean of AR process
     σ,             # s.d. deviation of innovation shocks
     ρ,             # persistence of AR process
     stdbound,      # standard deviation of AR process (set to 2)
     dimk,          # dimension of k, only works with odd numbers
     dimc,          # dimension of cash
     dimkp,         # dimension of future k
     dimcp):        # dimension of future cash
Packing:
    α, β, s                        = param_manager     # Manager compensation
    δ, λ, a, θ, τ                   = param_inv         # Investment parameters
    r, ϕ                            = param_fin         # Financing Parameters                                       
    μ, σ, ρ, stdbound               = param_ar          # AR Parameters
    dimz, dimk, dimc, dimkp, dimcp  = param_dim         # dimensional Parameters
"""   
            
def set_vec(param_inv, param_dim):   
    """
    # Compute the vector of capital stock (around the steady-state) and cash (up to k steady state).
    # Dimension: k_vec=[dimk , 1]; idem for the rest.
    """
    δ, _, a, θ, τ                = param_inv
    r, _                         = param_fin
    _, dimk, dimc, dimkp, dimcp  = param_dim
    
    kstar = ((θ*(1-τ))/(r+δ))**(1/(1-θ))
    #kstar=2083.6801320704803
    # Set up the vectors 
    k_vec  = np.reshape(np.linspace(0.01*kstar, 2*kstar,dimk),(dimk,1))
    kp_vec = np.reshape(np.linspace(0.01*kstar, 2*kstar,dimkp),(dimkp,1))
    c_vec  = np.reshape(np.linspace(0.0, kstar, dimc),(dimc,1))
    cp_vec = np.reshape(np.linspace(0.0, kstar, dimcp),(dimcp,1))
    return  [k_vec, kp_vec, c_vec, cp_vec, kstar]    

def trans_matrix(param_ar,param_dim):
    """
    # Set the State vector for the productivity shocks Z and transition matrix.
    # Dimension: z_vec =[dimZ , 1]; z_prob_mat=[dimz , dimZ] // Remember, sum(z_prob_mat[i,:])=1.
    """
    μ, σ, ρ, stdbound = param_ar
    dimz, dimk,*_     = param_dim
    
    mc          = qe.markov.approximation.tauchen(ρ,σ,μ,stdbound,dimz)
    z_vec, Pi   = mc.state_values, mc.P
    z_vec       = z_vec.reshape(dimz,1)
    z_vec       = np.e**(z_vec)
    z_prob_mat  = Pi.reshape(dimz,dimz)
    return [z_vec, z_prob_mat]

@jit(nopython=True,parallel=True)
def rewards_grids(param_manager,param_inv, param_fin, param_ar,param_dim):
    """
    Compute the manager's and shareholders' cash-flows  R and D, respectively, 
    for every (k_t, k_t+1, c_t, c_t+1) combination and every productivity shocks.
    """            
    α, β, s                         = param_manager     # Manager compensation
    δ, λ, a, θ, τ                   = param_inv         # Investment parameters
    r, ϕ                            = param_fin         # Financing Parameters                                       
    μ, σ, ρ, stdbound               = param_ar          # AR Parameters
    dimz, dimk, dimc, dimkp, dimcp  = param_dim         # dimensional Parameters                    
    
    R = np.zeros((dimk, dimkp, dimc, dimcp, dimz))
    D = np.zeros((dimk, dimkp, dimc, dimcp, dimz))
    inv:float
    d:float
    rw:float
    print("Computing reward matrix")

    for i_k in prange(dimk):
        for i_kp in range(dimkp):
            for i_c in range(dimc):
                for i_cp in range(dimcp):
                    for i_z in range(dimz):
                        inv                          = np.take(kp_vec[i_kp]-(1-δ)*k_vec[i_k],0)                          
                        d                            = np.take((1-τ)*(1-(α+s))*z_vec[i_z]*k_vec[i_k]**θ + δ*k_vec[i_k] *τ - inv - 0.5*a*((inv/k_vec[i_k] )**2)*k_vec[i_k]-cp_vec[i_cp] + c_vec[i_c]*(1+r*(1-τ))*(1-s),0)        
                        if d>=0:
                            D[i_k, i_kp, i_c, i_cp, i_z] = d
                        else:
                            D[i_k, i_kp, i_c, i_cp, i_z] = d*(1+ϕ)
                        rw                              = np.take((α+s)*z_vec[i_z]*k_vec[i_k]**θ + s*c_vec[i_c]*(1+r) + β*D[i_k, i_kp, i_c, i_cp, i_z] ,0)   
                        R[i_k, i_kp, i_c, i_cp, i_z]   = rw 
    print("Computing reward matrix - Done")                  
    return [R, D]
    
@jit(nopython=True,parallel=True)
def continuation_value(param_dim:npt.ArrayLike,U:npt.ArrayLike,z_prob_mat:npt.ArrayLike,grid:npt.ArrayLike,grid_interp:npt.ArrayLike):
    """
    Compute "Continuation Value" for every possible future state of nature (kp,cp,z).
    The "continuation value" is defined as: E[U(kp,cp,zp)]=sum{U(kp,cp,zp)*Prob(zp,p)}
    """
    dimz, dimk, dimc, dimkp, dimcp  = param_dim         # dimensional Parameters 
    cont_value=np.zeros((dimkp,dimcp,dimz))
    ztrans=np.zeros((1,dimz))    
    if (dimc==dimcp and dimk ==dimkp):
        for ind_z in prange(dimz):
            ztrans=np.transpose(z_prob_mat[ind_z, :])
            for i_k in range(dimk):
                for i_c in range(dimc):
                    cont_value[i_k,i_c,ind_z] = np.dot(ztrans, U[i_k,i_c,:])
    else:
        Uinter=eval_linear(grid,U,grid_interp)
        Uinter=Uinter.reshape((dimkp,dimcp,dimz))
        for ind_z in prange(dimz):
            ztrans=np.transpose(z_prob_mat[ind_z, :])
            for i_kpp in range(dimkp):
                for i_cpp in range(dimcp):
                    cont_value[i_kpp,i_cpp,ind_z]= np.dot(ztrans,Uinter[i_kpp,i_cpp,:])
    return cont_value


def bellman_operator(param_dim:npt.ArrayLike,param_fin:npt.ArrayLike,U:npt.ArrayLike,R:npt.ArrayLike,z_prob_mat:npt.ArrayLike,grid:npt.ArrayLike,grid_interp:npt.ArrayLike,i_kp:npt.ArrayLike, i_cp:npt.ArrayLike):
    # Second, identify max policy and save it.
    # For each current state of nature (k,c,z), find the policy {kp,cp} that maximizes RHS: U(k,c,z) + E[U(kp,cp,zp)].
    # Once found it, update the value of U in this current state of nature with the one generated with the optimal policy
    # and save the respective optimal policy (kp,cp) for each state of nature (k,c,z).
    dimz, dimk, dimc, dimkp, dimcp  = param_dim         # dimensional Parameters
    r, _                            = param_fin 
    c_value                         = continuation_value(param_dim,U,z_prob_mat,grid,grid_interp)
    RHS=np.empty((dimkp,dimcp))
    for (i_z, z) in enumerate(z_vec):
        for (i_k, k) in enumerate(k_vec):
            for (i_c, c) in enumerate(c_vec):
                RHS                 = R[i_k, :, i_c, :, i_z] + (1/(1+r))*c_value[:,:,i_z]
                i_kc                = np.unravel_index(np.argmax(RHS,axis=None),RHS.shape)           # the index of the best expected value for each kp,cp,z combination.    
                U[i_k, i_c, i_z]    = RHS[i_kc]                                       # update U with all the best expected values. 
                i_kp[i_k, i_c, i_z] = i_kc[0]
                i_cp[i_k, i_c, i_z] = i_kc[1]
    return [U,i_kp,i_cp ]

def value_iteration(param_dim:npt.ArrayLike,param_fin:npt.ArrayLike,R:npt.ArrayLike,k_vec:npt.ArrayLike,c_vec:npt.ArrayLike,z_vec:npt.ArrayLike,kp_vec:npt.ArrayLike,cp_vec:npt.ArrayLike,diff=1,tol=1e-6,imax=1400):
        """
        # Value Iteration on Eq 6.
        # am is an instance of AgencyModel class
        """
        dimz, dimk, dimc, dimkp, dimcp  = param_dim 
        U    = np.zeros((dimk, dimc, dimz))
        i_kp = np.empty((dimk, dimc, dimz),dtype=float)
        i_cp = np.empty((dimk, dimc, dimz),dtype=float)
        grid = CGrid(k_vec.reshape(dimk),c_vec.reshape(dimc),z_vec.reshape(dimz))
        grid_interp=np.zeros((0,3))
        for i_kpp in range(dimkp):
             for i_cpp in range(dimcp):
                 for i_zp in range(dimz):
                     new_point=np.array([kp_vec[i_kpp], cp_vec[i_cpp], z_vec[i_zp]]).reshape((1,3))
                     grid_interp = np.append(grid_interp,new_point,axis=0)
        #grid_interp = [np.array([kp_vec[i_kpp], cp_vec[i_cpp], z_vec[i_zp]]).reshape((1,3)) for i_kpp in range(dimkp) for i_cpp in range(dimcp) for i_zp in range(dimz)]
        print("Iteration start")                   
        for i in range(imax):
           U_old          = np.copy(U)
           [Up,i_kp,i_cp ]= bellman_operator(param_dim,param_fin,U, R,z_prob_mat,grid,grid_interp, i_kp, i_cp)         
           diff           = np.max(np.abs(Up-U_old))
           if i%100==0: 
               print(f"Error at iteration {i} is {diff}.")           
           if diff < tol:
               break
           if i == imax:
               print("Failed to converge!")        
        # Evaluating the optimal policies using the indexes obtained in the iterations.
        Kp=np.zeros((dimk, dimc, dimz))
        Cp=np.zeros((dimk, dimc, dimz))
        for index,value in  np.ndenumerate(i_kp):
            index2=int(value)
            Kp[index]=kp_vec[index2]
        for index,value in  np.ndenumerate(i_cp):
            index2=int(value)
            Cp[index]=cp_vec[index2]
            
        return [U,Kp,Cp,i_kp,i_cp]

# Plotting Policy function
def plot_policy_function(param_manager,param_inv,param_fin,param_dim,z_vec,k_vec,c_vec,Kp,Cp):
    α, β, s                         = param_manager     # Manager compensation
    δ, λ, a, θ, τ                   = param_inv         # Investment parameters
    r, ϕ                            = param_fin         # Financing Parameters                                       
    dimz, dimk, dimc, dimkp, dimcp  = param_dim         # dimensional Parameters
    
    I_p=np.empty((dimk, dimc,dimz))        
    CRatio_P=np.empty((dimk,dimc,dimz))
    CF_p=np.empty((dimk, dimc,dimz))
    F_p=np.empty((dimk, dimc,dimz))
    
    for z in range(dimz):
        for c in range(dimc):
            for k in range(dimk):            
                CF_p[k,c,z]       = ((1-τ)*z_vec[z]*k_vec[k]**θ)/k_vec[k]
                I                 = (Kp[k,c,z]-(1-δ)*k_vec[k])
                I_p[k,c,z]        = I/k_vec[k]
                CRatio_P[k,c,z]   = Cp[k,c,z]/(c_vec[c]+k_vec[k])
                d                 = (1-τ)*(1-(α+s))*z_vec[z]*k_vec[k]**θ + δ*k_vec[k]*τ - I  - 0.5*a*((I/k_vec[k])**2)*k_vec[k]  - Cp[k,c,z] +c_vec[c]*(1+r*(1-τ))*(1-s)
                F_p[k,c,z]        = d/(c_vec[c]+k_vec[k]) if d>=0 else d*(1+ϕ)/(c_vec[c]+k_vec[k])  
                 
    [i_kstar, j_kstar] = np.unravel_index(np.argmin(np.abs(kstar-k_vec),axis=None),k_vec.shape)       #the plots are with kstar, for different levels of c and z    
    logz=np.log(z_vec)
    cmed_plot=math.floor(dimc/2)
    
    fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1)
    ax1.plot(logz,CF_p[i_kstar,1,:] , label='Low Cash ratio',linestyle = 'dashed', c='b')
    ax1.plot(logz,CF_p[i_kstar,cmed_plot,:] , label='Medium Cash ratio',linestyle = 'solid', c='b')
    ax1.plot(logz,CF_p[i_kstar,dimc-1,:] , label='High Cash ratio',linestyle = 'dotted', c='b')
    ax1.set_xlabel("Log productivity shock")
    ax1.set_ylabel("Cash Flow / Capital")
    ax1.legend()
    
    ax2.plot(logz,I_p[i_kstar,1,:] , label='Low Cash ratio',linestyle = 'dashed', c='b')
    ax2.plot(logz,I_p[i_kstar,cmed_plot,:] , label='Medium Cash ratio',linestyle = 'solid', c='b')
    ax2.plot(logz,I_p[i_kstar,dimc-1,:] , label='High Cash ratio',linestyle = 'dotted', c='b')
    ax2.set_xlabel("Log productivity shock")
    ax2.set_ylabel("Investment / Capital")
    #ax2.legend()
    #
    ax3.plot(logz,CRatio_P[i_kstar,1,:] , label='Low Cash ratio',linestyle = 'dashed', c='b')
    ax3.plot(logz,CRatio_P[i_kstar,cmed_plot,:] , label='Medium Cash ratio',linestyle = 'solid', c='b')
    ax3.plot(logz,CRatio_P[i_kstar,dimc-1,:] , label='High Cash ratio',linestyle = 'dotted', c='b')
    ax3.set_xlabel("Log productivity shock")
    ax3.set_ylabel("Cash / Assets")
    #ax3.legend()
    
    ax4.plot(logz,F_p[i_kstar,1,:] , label='Low Cash ratio',linestyle = 'dashed', c='b')
    ax4.plot(logz,F_p[i_kstar,cmed_plot,:] , label='Medium Cash ratio',linestyle = 'solid', c='b')
    ax4.plot(logz,F_p[i_kstar,dimc-1,:] , label='High Cash ratio',linestyle = 'dotted', c='b')
    ax4.set_xlabel("Log productivity shock")
    ax4.set_ylabel("External FIn / Assets")
    #ax4.legend()
    
    plt.show()
    
# Simulation and Comparative Statistics
def model_sim(z_vec,z_prob_mat, kp_vec, cp_vec, Kp, Cp, kinit, cinit, N, Ttot):
        """
        # Model simulation.
        # am is an instance of AgencyModel class
        """
        mc  = qe.MarkovChain(z_prob_mat)
        E   = mc.simulate(ts_length=Ttot,num_reps=N).T
        
        Ksim = np.zeros((Ttot, N))
        Ksim[0, :] = np.take(k_vec,kinit)* np.ones((1, N))
        Csim = np.zeros((Ttot, N))
        Csim[0, :] = np.take(c_vec,cinit)* np.ones((1, N))   # Ask to Juan for a better ideas
        for n in range(N):
            for t in range(1,Ttot):
                # find previus state of nature
                closer_k=min(enumerate(k_vec), key=lambda x: abs(Ksim[t-1, 0] - x[1]))
                closer_c=min(enumerate(c_vec), key=lambda x: abs(Csim[t-1, 0] - x[1]))
                id_z = E[t-1,n]
                # identify the respective policy
                Ksim[t, n] = Kp[closer_k[0],closer_c[0], id_z] 
                Csim[t, n] = Cp[closer_k[0],closer_c[0], id_z]                  
        """        
        print()
        print("Quick simulation check: p lb, min(psim), max(psim), p ub")
        print(f"K = {k_vec[0]} {np.min(Ksim)} {np.max(Ksim)} {k_vec[-1]}")
        print(f"C = {c_vec[0]} {np.min(Csim)} {np.max(Csim)} {c_vec[-1]}")
        print()
        """
        return [Ksim, Csim,E]
    
def run_stats(variable,param_iter):
    global param_fin
    global param_inv
    global param_manager
    global param_ar
    
    if variable =='ϕ':
        param_fin                   = (0.011, param_iter)                 # (r, ϕ)
    elif variable=='θ':     
        param_inv                   = (0.130, 0, 1.278, param_iter, 0.2) # (δ, λ, a, θ, τ)
    elif variable=='β':
        param_manager               = (0.751/100,param_iter, 0.101/1000) # (α, β, s)
    elif variable=='σ':
        param_ar                    = (0, param_iter, 0.713, 4)           # (μ, σ, ρ, stdbound)
    elif variable=='a':     
        param_inv                   = (0.130, 0, param_iter, 0.773 , 0.2) # (δ, λ, a, θ, τ)
    elif variable=='α':
        param_manager               = (param_iter,0.051, 0.101/1000) # (α, β, s)
    elif variable=='ρ':
        param_ar                    = (0, 0.262, param_iter, 4)           # (μ, σ, ρ, stdbound)
    elif variable=='δ':     
        param_inv                   = (param_iter, 0, 1.278,0.773 , 0.2) # (δ, λ, a, θ, τ)
    elif variable=='s':
        param_manager               = (0.751/100,0.051, param_iter) # (α, β, s)
    else:
        print("not known variable")
        breakpoint()
    [k_vec, kp_vec, c_vec, cp_vec, kstar]= set_vec(param_inv, param_dim)
    [z_vec, z_prob_mat]                  = trans_matrix(param_ar,param_dim)
    [R, D]                               = rewards_grids(param_manager,param_inv, param_fin, param_ar,param_dim)
    [Up,Kp,Cp,i_kp,i_cp]                 = value_iteration(param_dim,param_fin,R,k_vec,c_vec,z_vec,kp_vec,cp_vec)
    [sim_K, sim_C,_]                     = model_sim(z_vec,z_prob_mat, kp_vec, cp_vec, Kp, Cp, kinit, cinit,num_reps, ts_length)
    sim_Cratio                           = sim_C/(sim_K+sim_C)       
    sim_Cratio_av                        = np.mean(sim_Cratio)      
    return sim_Cratio_av
        
# In[8]: Set params
"""
List of Parameters
    α,             # manager bonus
    β,             # manager equity share
    s,             # manager private benefit
    δ,             # capital depreciation
    λ,             # cost of issuing equity // (so far, not needed)
    a,             # capital adjustment cost
    θ,             # curvature production function
    τ,             # corporate tax rate
    r,             # interest rate
    ϕ,             # cost of external finance
    dimz,          # dim of shock
    μ,             # log mean of AR process
    σ,             # s.d. deviation of innovation shocks
    ρ,             # persistence of AR process
    stdbound,      # standard deviation of AR process (set to 2)
    dimk,          # dimension of k, only works with odd numbers
    dimc,          # dimension of cash
    dimkp,         # dimension of future k
    dimcp         # dimension of future cash
    
Packing:    
    α, β, s                         = param_manager     # Manager compensation
    δ, λ, a, θ, τ                   = param_inv         # Investment parameters
    r, ϕ                            = param_fin         # Financing Parameters                                       
    μ, σ, ρ, stdbound               = param_ar          # AR Parameters
    dimz, dimk, dimc, dimkp, dimcp  = param_dim         # dimensional Parameters 

"""
#if __name__ == '__main__':
    
param_manager = (0.751/100, 0.051, 0.101/1000) # (α, β, s) 
param_inv     = (0.130, 0, 1.278, 0.773 , 0.2) # (δ, λ, a, θ, τ)   
param_fin     = (0.011, 0.043)                 # (r, ϕ=0.043)
param_ar      = (0, 0.262, 0.713, 4)           # (μ, σ, ρ, stdbound)
param_dim     = (11,25,7,25*2,7*2)             # (dimz, dimk, dimc, dimkp, dimcp)
 
# In[9] Calculate and plot policies ("Figure 1")
# Run this cell to initialize jit methods.
qe.tic()
[k_vec, kp_vec, c_vec, cp_vec, kstar]= set_vec(param_inv, param_dim)
[z_vec, z_prob_mat]                  = trans_matrix(param_ar,param_dim)
[R, D]                               = rewards_grids(param_manager,param_inv, param_fin, param_ar,param_dim)
[Up,Kp,Cp,i_kp,i_cp]                 = value_iteration(param_dim,param_fin,R,k_vec,c_vec,z_vec,kp_vec,cp_vec)
qe.toc()

plot_policy_function(param_manager,param_inv,param_fin,param_dim,z_vec,k_vec,c_vec,Kp,Cp)
# In[9] Comparative Statistics ("Figure 2")

num_reps  = 1
ts_length = 100_000
kinit = math.floor(25/2)
cinit = math.floor(7/2)
ϕ_vec = np.linspace(0,0.25,10) #20 reps
θ_vec = np.linspace(0.5,0.9,10) #20 reps
β_vec = np.linspace(0.01,0.1,10) #20 reps

σ_vec = np.linspace(0.1,0.5,10) #20 reps
a_vec = np.linspace(0,1.5,10) #20 reps
α_vec = np.linspace(0,0.01,10) #20 reps

ρ_vec = np.linspace(0.5,0.75,10) #20 reps
δ_vec = np.linspace(0.05,0.2,10) #20 reps
s_vec = np.linspace(0,0.02/100,10) #20 reps

# First column of plots
with Pool() as pool:
    com_stat_cash = pool.starmap(run_stats, zip(repeat('ϕ'),ϕ_vec))
with Pool() as pool:
    com_stat_cash2 = pool.starmap(run_stats, zip(repeat('θ'),θ_vec))
with Pool() as pool:
    com_stat_cash3 = pool.starmap(run_stats, zip(repeat('β'),β_vec))
# Second column of plots   
with Pool() as pool:
    com_stat_cash4 = pool.starmap(run_stats, zip(repeat('σ'),σ_vec))
with Pool() as pool:
    com_stat_cash5 = pool.starmap(run_stats, zip(repeat('a'),a_vec))
with Pool() as pool:
    com_stat_cash6 = pool.starmap(run_stats, zip(repeat('α'),α_vec))
# Third column of plots   
with Pool() as pool:
    com_stat_cash7 = pool.starmap(run_stats, zip(repeat('ρ'),ρ_vec))
with Pool() as pool:
    com_stat_cash8 = pool.starmap(run_stats, zip(repeat('δ'),δ_vec))
with Pool() as pool:
    com_stat_cash9 = pool.starmap(run_stats, zip(repeat('s'),s_vec))


fig, ((ax1, ax4, ax7), (ax2, ax5, ax8), (ax3, ax6, ax9)) = plt.subplots(3,3,sharey=True,figsize=(15, 15))
ax1.plot(ϕ_vec,com_stat_cash ,linestyle = 'dashed', c='b')
ax1.set_xlabel("ϕ")
ax2.plot(θ_vec,com_stat_cash2 ,linestyle = 'dashed', c='b')
ax2.set_xlabel("θ")
ax3.plot(β_vec,com_stat_cash3 ,linestyle = 'dashed', c='b')
ax3.set_xlabel("ϕ")
ax4.plot(σ_vec,com_stat_cash4 ,linestyle = 'dashed', c='b')
ax4.set_xlabel("σ")
ax5.plot(a_vec,com_stat_cash5 ,linestyle = 'dashed', c='b')
ax5.set_xlabel("a")
ax6.plot(α_vec,com_stat_cash6 ,linestyle = 'dashed', c='b')
ax6.set_xlabel("α")
ax7.plot(ρ_vec,com_stat_cash7 ,linestyle = 'dashed', c='b')
ax7.set_xlabel("ρ")
ax8.plot(δ_vec,com_stat_cash8 ,linestyle = 'dashed', c='b')
ax8.set_xlabel("δ")
ax9.plot(s_vec,com_stat_cash9 ,linestyle = 'dashed', c='b')
ax9.set_xlabel("s")
plt.show()


