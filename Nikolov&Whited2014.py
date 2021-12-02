#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 5 21:12:52 2021
@author: Marcelo Ortiz M @ UPF and BSE.
Notes: 35 sec with interp (x2) but crushes without interp (?).
"""
# In[1]: Import Packages and cleaning 
import numpy as np
import numpy.typing as npt
import quantecon as qe
from interpolation.splines import eval_linear, CGrid
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')  #for plots in different a window use 'qr'
import math
from numba import jit,prange

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
           if i%50==0: 
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
    
# In[5]: Bellman operator and Value iteration for Eq 8
def bellman_operator_V(param_dim,param_fin, V,D, i_kp, i_cp, grid,z_vec,z_prob_mat,kp_vec,cp_vec):
        """
        RHS of Eq 8
        Pending, acelerar moving reshape en Z, primer loop.
        """
        r, _                            = param_fin
        dimz, dimk, dimc, dimkp, dimcp  = param_dim
        for (i_k, k) in enumerate(k_vec):           
            for (i_c, c) in enumerate(c_vec):
                for (i_z, z) in enumerate(z_vec):
                    EV                  = D[i_k, :, i_c, :, i_z] + (1/(1+r))*np.reshape([z_prob_mat[i_z, :] @ V[i_kpp,i_cpp,:] for i_kpp in range(dimkp) for i_cpp in range(dimcp)], (dimkp, dimcp))
                    i_kc                = np.unravel_index(np.argmax(EV,axis=None),EV.shape)           # the index of the best expected value for each k,c,z combination.    
                    V[i_k, i_c, i_z]    = EV[i_kc]                                       # update U with all the best expected values. 
        return [V]

def solution_vi_V(param_dim,diff,tol,imax, D):
        """
        # Value Iteration on Eq 8.
        """
        dimz, dimk, dimc, dimkp, dimcp  = param_dim 
        V   = np.empty((dimkp, dimcp, dimz))
        i_kp = np.empty((dimk, dimc, dimz))
        i_cp = np.empty((dimk, dimc, dimz))
        grid=CGrid(k_vec.reshape(dimk),c_vec.reshape(dimc),z_vec.reshape(dimz))
        for i in range(imax):
            V_old = np.copy(V)          
            Vp    = bellman_operator_V(param_dim,param_fin,V,D, i_kp, i_cp, grid,z_vec,z_prob_mat,kp_vec,cp_vec)
            diff = np.max(np.abs(Vp-V_old))
            if i%50==0:
               print(f"Error at iteration {i} is {diff}.")
            if diff < tol:
               break
            if i == imax:
               print("Failed to converge!")    
        return [Vp]
    

# In[6]: Plotting function
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
    
# In[7]: Simulation
def model_sim(z_vec,z_prob_mat, kp_vec, cp_vec, ikp, icp, kinit, cinit, N, Ttot):
        """
        # Model simulation.
        # am is an instance of AgencyModel class
        """
        mc  = qe.MarkovChain(z_prob_mat)
        E   = mc.simulate(ts_length=Ttot,num_reps=N).T
        
        Ksim = np.empty((Ttot, N))
        Ksim[0, :] = np.take(kp_vec,kinit)* np.ones((1, N))
        Csim = np.empty((Ttot, N))
        Csim[0, :] = np.take(cp_vec,cinit)* np.ones((1, N))   # Ask to Juan for a better ideas
       
        for n in range(N):
            for t in range(1,Ttot):
                id_k= np.argwhere(Ksim[t-1, n] == kp_vec)
                id_c= np.argwhere(Csim[t-1, n] == cp_vec)
                id_z = E[t-1,n]
                Ksim[t, n] = Kp[id_k[0,0],id_c[0,0], id_z]
                Csim[t, n] = Cp[id_k[0,0],id_c[0,0], id_z]                  
        """        
        print()
        print("Quick simulation check: p lb, min(psim), max(psim), p ub")
        print(f"K = {k_vec[0]} {np.min(Ksim)} {np.max(Ksim)} {k_vec[-1]}")
        print(f"C = {c_vec[0]} {np.min(Csim)} {np.max(Csim)} {c_vec[-1]}")
        print()
        """
        return [Ksim, Csim,E]
        
# In[8]: Set variables and grids
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
param_manager = (0.751/100, 0.051, 0.101/1000) # (α, β, s) 
param_inv     = (0.130, 0, 1.278, 0.773 , 0.2) # (δ, λ, a, θ, τ)   
param_fin     = (0.011, 0.043)                 # (r, ϕ)
param_ar      = (0, 0.262, 0.713, 4)           # (μ, σ, ρ, stdbound)
param_dim     = (11,25,7,25*2,7*2)             # (dimz, dimk, dimc, dimkp, dimcp)
 

qe.tic()
[k_vec, kp_vec, c_vec, cp_vec, kstar]= set_vec(param_inv, param_dim)
[z_vec, z_prob_mat]                  = trans_matrix(param_ar,param_dim)
[R, D]                               = rewards_grids(param_manager,param_inv, param_fin, param_ar,param_dim)
[Up,Kp,Cp,i_kp,i_cp]                 = value_iteration(param_dim,param_fin,R,k_vec,c_vec,z_vec,kp_vec,cp_vec)
qe.toc()

# In[9] Calculate and plot policies ("Figure 1")
plot_policy_function(param_manager,param_inv,param_fin,param_dim,z_vec,k_vec,c_vec,Kp,Cp)

# In[9] PLot Comparative Statistics ("Figure 2")
"""
num_reps                            = 1
ts_length                           = 100_000
kinit                               = math.floor(dimk/2)
cinit                               = math.floor(dimc/2)
sim_ϕ_vec                           = np.linspace(0,0.25,20) #20 reps
sim_ϕav_cash                        = np.ones(len(sim_ϕ_vec))
# External Financing ϕ
def run_policy():
    for ind, sim_ϕ in  enumerate(sim_ϕ_vec):
        firm2=AgencyModel(α,        # manager bonus
                     β,             # manager equity share
                     s,             # manager private benefit
                     δ,             # capital depreciation
                     λ,             # cost of issuing equity // (so far, not needed)
                     a,             # capital adjustment cost
                     θ,             # curvature production function
                     τ,             # corporate tax rate
                     r,             # interest rate
                     sim_ϕ,         # cost of external finance
                     dimz,          # dim of shock
                     μ,             # log mean of AR process
                     σ,             # s.d. deviation of innovation shocks
                     ρ,             # persistence of AR process
                     stdbound,      # standard deviation of AR process (set to 2)
                     dimk,          # dimension of k, only works with odd numbers
                     dimc,          # dimension of cash
                     dimkp,         # dimension of future k
                     dimcp)         # dimension of future cash
        [k_vec2, kp_vec2, c_vec2, cp_vec2, kstar2]= firm2.set_vec()
        [z_vec2, z_prob_mat2]                  = firm2.trans_matrix()
        [R2, _]                               = firm2.rewards_grids()
        [_,Kp2,Cp2,_,_]                         = solution_vi(1,1e-4,1_000,R2,firm2)  
        [Ksim2, Csim2,_]                       = model_sim(z_vec2,z_prob_mat2, kp_vec2, cp_vec2, Kp2, Cp2, kinit, cinit,num_reps, ts_length)
        Csim_ratio2                           = Csim2/(Ksim2+Csim2)       
        sim_ϕav_cash[ind]                    = np.mean(Csim_ratio2)
        

        plt.plot(sim_ϕ_vec,sim_ϕav_cash  ,linestyle = 'dashed', c='b')
        plt.show()
    return sim_ϕav_cash
    
cash_stats=run_policy()
"""
