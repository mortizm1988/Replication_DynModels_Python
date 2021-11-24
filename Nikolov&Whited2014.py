#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 5 21:12:52 2021
@author: Marcelo Ortiz M @ UPF and BSE.
Notes:  (1) Why Ext. financing is always positive?
"""
# In[1]: Import Packages and cleaning
from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -f')     

import numpy as np
import numpy.typing as npt
import quantecon as qe
from interpolation.splines import eval_linear, CGrid
import matplotlib.pyplot as plt
import math
from numba import jit,prange, int32, float64

# In[2]: Class definition
class AgencyModel():
    # initialization method
    def __init__(self,
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
   
        self.α, self.β, self.s  = α, β, s                              # Manager compensation
        self.δ, self.λ, self.a, self.θ, self.τ =  δ, λ, a, θ, τ        # Investment parameters
        self.r, self.ϕ = r, ϕ                                          # Financing Parameters
        self.μ, self.σ, self.ρ, self.stdbound = μ, σ, ρ, stdbound      # AR parameters
        self.dimz, self.dimk, self.dimc, self.dimkp, self.dimcp  = dimz, dimk, dimc, dimkp, dimcp                             # Grid parameters                   
        
       
    def set_vec(self):
        """
        # Compute the vector of capital stock around the steady-state  %WHERE THIS COME FROM? A: Euler E + Env C.
        # Dimension: k_vec=[dimk , 1]; idem for the rest.
        """            
        self.δ,  self.a, self.θ, self.τ =  δ, a, θ, τ        
        self.dimz, self.dimk, = dimz, dimk,
        
        kstar = ((θ*(1-τ))/(r+δ))**(1/(1-θ))
        #kstar=2083.6801320704803
        # Set up the vectors 
        k_vec  = np.reshape(np.linspace(0.01*kstar, 2*kstar,dimk),(dimk,1))
        kp_vec = np.reshape(np.linspace(0.01*kstar, 2*kstar,dimkp),(dimkp,1))
        c_vec  = np.reshape(np.linspace(0.0, kstar, dimc),(dimc,1))
        cp_vec = np.reshape(np.linspace(0.0, kstar, dimcp),(dimcp,1))
        return  [k_vec, kp_vec, c_vec, cp_vec, kstar]    
    
    def trans_matrix(self):
        """
        # Set the State vector for the productivity shocks Z and transition matrix.
        # Dimension: z_vec =[dimZ , 1]; z_prob_mat=[dimz , dimZ] // Remember, sum(z_prob_mat[i,:])=1.
        """
        self.μ, self.σ, self.ρ = μ, σ, ρ
        self.dimz, self.dimk, self.stdbound = dimz, dimk, stdbound
        
        mc          = qe.markov.approximation.tauchen(ρ,σ,μ,stdbound,dimz)
        z_vec, Pi   = mc.state_values, mc.P
        z_vec       = z_vec.reshape(dimz,1)
        z_vec       = np.e**(z_vec)
        z_prob_mat  = Pi.reshape(dimz,dimz)
        return [z_vec, z_prob_mat]
    
     
    def rewards_grids(self):
        """
        #
        # Compute the manager's and shareholders' cash-flows  R and D, respectively, 
        # for every (k_t, k_t+1, c_t, c_t+1) combination and every productivity shocks.
        """        
        
        self.α, self.β, self.s  = α, β, s                              # Manager compensation
        self.δ, self.λ, self.a, self.θ, self.τ =  δ, λ, a, θ, τ        # Investment parameters
        self.r, self.ϕ = r, ϕ                                          # Financing Parameters                                       
        self.μ, self.σ, self.ρ, self.stdbound = μ, σ, ρ, stdbound      # AR parameters
        self.dimz, self.dimk, self.dimc, self.dimkp, self.dimcp  = dimz, dimk, dimc, dimkp, dimcp                             
        
        R = np.empty((dimk, dimkp, dimc, dimcp, dimz))
        D = np.empty((dimk, dimkp, dimc, dimcp, dimz))
        print("Computing reward matrix")
        for (i_k, k) in enumerate(k_vec):
            for (i_kp, kp) in enumerate(kp_vec):
                for (i_c, c) in enumerate(c_vec):
                    for (i_cp, cp) in enumerate(cp_vec):
                        for (i_z, ϵ) in enumerate(z_vec):
                            I                            = kp-(1-δ)*k                          
                            d                            = (1-τ)*(1-(α+s))*ϵ*k**θ + δ*k*τ - I - 0.5*a*((I/k)**2)*k  - cp +c*(1+r*(1-τ))*(1-s)        
                            if d>=0:
                                D[i_k, i_kp, i_c, i_cp, i_z] = d
                            else:
                                D[i_k, i_kp, i_c, i_cp, i_z] = d*(1+ϕ)
                            R[i_k, i_kp, i_c, i_cp, i_z] = (α+s)*ϵ*k**θ + s*c*(1+r) + β*D[i_k, i_kp, i_c, i_cp, i_z]
        print("Computing reward matrix - Done")                  
        return [R, D]
    
# In[4]: Bellman operator and Value iteration for Eq 6

@jit(nopython=True,parallel=True)
def U_interp(grid,U,grid_interp,dimkp,dimcp,dimz):
    # get the interpolated Utility function
    length=dimkp*dimcp*dimz
    U2=np.empty((length,1))
    for i in prange(length):
        Up = eval_linear(grid,U,grid_interp[i])
        U2[i,0]=Up[0]
    return U2  
def bellman_operator(U,R,z_prob_mat,grid,grid_interp,i_kp, i_cp, am):
    """
    RHS of Eq 6.
    # am is an instance of AgencyModel class
    """
    dimz, dimc, dimk, dimkp, dimcp = am.dimz, am.dimc, am.dimk, am.dimkp, am.dimcp
    
    # First, compute "Continuation Value"
    cont_value=np.zeros((dimkp,dimcp,dimz))
    if dimc==dimcp and dimk ==dimkp:
        for ind_z in range(dimz):
            for i_kpp in range(dimkp):
                for i_cpp in range(dimcp):
                    cont_value[i_kpp,i_cpp,ind_z] = np.dot(z_prob_mat[ind_z, :], U[i_kpp,i_cpp,:])
    else:
        Uinter=U_interp(grid,U,grid_interp,dimkp,dimcp,dimz)
        Uinter=Uinter.reshape((dimkp,dimcp,dimz))
        for ind_z in range(dimz):
            for i_kpp in range(dimkp):
                for i_cpp in range(dimcp):
                    cont_value[i_kpp,i_cpp,ind_z]= np.dot(z_prob_mat[ind_z, :],Uinter[i_kpp,i_cpp,:]  )
    
    # Second, identify max policy and save it
    for (i_z, z) in enumerate(z_vec):
        for (i_k, k) in enumerate(k_vec):
            for (i_c, c) in enumerate(c_vec):                                           
                RHS                  = R[i_k, :, i_c, :, i_z] + (1/(1+r))*cont_value[:,:,i_z]
                i_kc                = np.unravel_index(np.argmax(RHS,axis=None),RHS.shape)           # the index of the best expected value for each kp,cp,z combination.    
                U[i_k, i_c, i_z]    = RHS[i_kc]                                       # update U with all the best expected values. 
                i_kp[i_k, i_c, i_z] = i_kc[0]
                i_cp[i_k, i_c, i_z] = i_kc[1]
    return [U,i_kp,i_cp ]


           
def solution_vi(diff,tol,imax,am, R):
        """
        # Value Iteration on Eq 6.
        # am is an instance of AgencyModel class
        """
        dimz, dimc, dimk, dimkp, dimcp = am.dimz, am.dimc, am.dimk, am.dimkp, am.dimcp 
        U    = np.zeros((dimk, dimc, dimz))
        i_kp = np.empty((dimk, dimc, dimz))
        i_cp = np.empty((dimk, dimc, dimz))
        grid = CGrid(k_vec.reshape(dimk),c_vec.reshape(dimc),z_vec.reshape(dimz))
        #grid_interp = np.asarray([np.array([kp_vec[i_kpp], cp_vec[i_cpp], z_vec[i_zp]]).reshape((1,3)) for i_kpp in range(dimkp) for i_cpp in range(dimcp) for i_zp in range(dimz)])
        grid_interp = [np.array([kp_vec[i_kpp], cp_vec[i_cpp], z_vec[i_zp]]).reshape((1,3)) for i_kpp in range(dimkp) for i_cpp in range(dimcp) for i_zp in range(dimz)]
        print("Iteration start")                   
        for i in range(imax):
           U_old          = np.copy(U)
           [Up,i_kp,i_cp ]= bellman_operator(U, R,z_prob_mat,grid,grid_interp, i_kp, i_cp,am)         
           diff           = np.max(np.abs(Up-U_old))
           if i%50==0: 
               print(f"Error at iteration {i} is {diff}.")           
           if diff < tol:
               break
           if i == imax:
               print("Failed to converge!")        
        # evaluating optimal policies
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
def bellman_operator_V(V,D, i_kp, i_cp, grid,z_vec,z_prob_mat,kp_vec,cp_vec, am):
        """
        RHS of Eq 8
        Pending, acelerar moving reshape en Z, primer loop.
        """
        for (i_k, k) in enumerate(k_vec):           
            for (i_c, c) in enumerate(c_vec):
                for (i_z, z) in enumerate(z_vec):
                    EV                  = D[i_k, :, i_c, :, i_z] + (1/(1+r))*np.reshape([z_prob_mat[i_z, :] @ V[i_kpp,i_cpp,:] for i_kpp in range(dimkp) for i_cpp in range(dimcp)], (dimkp, dimcp))
                    i_kc                = np.unravel_index(np.argmax(EV,axis=None),EV.shape)           # the index of the best expected value for each k,c,z combination.    
                    V[i_k, i_c, i_z]    = EV[i_kc]                                       # update U with all the best expected values. 
        return [V]

def solution_vi_V(diff,tol,imax,am, D):
        """
        # Value Iteration on Eq 8.
        """
        dimz, dimc, dimk, dimkp, dimcp = am.dimz, am.dimc, am.dimk, am.dimkp, am.dimcp 
        V   = np.empty((dimkp, dimcp, dimz))
        i_kp = np.empty((dimk, dimc, dimz))
        i_cp = np.empty((dimk, dimc, dimz))
        grid=CGrid(k_vec.reshape(dimk),c_vec.reshape(dimc),z_vec.reshape(dimz))
        for i in range(imax):
            V_old = np.copy(V)          
            Vp    = bellman_operator_V(V,D, i_kp, i_cp, grid,z_vec,z_prob_mat,kp_vec,cp_vec,am)
            diff = np.max(np.abs(Vp-V_old))
            if i%50==0:
               print(f"Error at iteration {i} is {diff}.")
            if diff < tol:
               break
            if i == imax:
               print("Failed to converge!")    
        return [Vp]
    
# In[6]: Simulation
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
                    
     
        print()
        print("Quick simulation check: p lb, min(psim), max(psim), p ub")
        print(f"K = {k_vec[0]} {np.min(Ksim)} {np.max(Ksim)} {k_vec[-1]}")
        print(f"C = {c_vec[0]} {np.min(Csim)} {np.max(Csim)} {c_vec[-1]}")
        print()
        
        return [Ksim, Csim,E]  
        
# In[7]: Main Code // Wrapper
θ=0.773
α=0.751/100
β=0.051   #0.005
s=0.101/1000
δ=0.130
τ=0.2     #0.3
r=0.011   #0.05
ρ=0.713
σ=0.262 
ϕ=0.043
a=1.278        #1.278
λ=0
μ=0

dimz=11      # 11
dimk=25       #25
dimc=7
dimkp=dimk*2
dimcp=dimc*2
stdbound=4
 
firm=AgencyModel(α,             # manager bonus
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
                 dimcp)         # dimension of future cash



[k_vec, kp_vec, c_vec, cp_vec, kstar]= firm.set_vec()
[z_vec, z_prob_mat]                  = firm.trans_matrix()
[R, D]                               = firm.rewards_grids()
qe.tic()
[Up,Kp,Cp,i_kp,i_cp]                 = solution_vi(1,1e-6,1100,firm,R)        #Eq 6
qe.toc()
#V                                   = solution_vi_V(1,1e-8,1000,firm,D)            #Eq 8

# In[8] Calculate and plot policies

I_p=np.empty((dimk, dimc,dimz))        
CRatio_P=np.empty((dimk,dimc,dimz))
CF_p=np.empty((dimk, dimc,dimz))
F_p=np.empty((dimk, dimc,dimz))


for z in range(dimz):
    for c in range(dimc):
        for k in range(dimk):            
            CF_p[k,c,z]       = ((1-τ)*z_vec[z]*k_vec[k]**θ)/k_vec[k]     
            I_p[k,c,z]        = (Kp[k,c,z]-(1-δ)*k_vec[k])/k_vec[k]
            CRatio_P[k,c,z]   = Cp[k,c,z]/(c_vec[c]+k_vec[k])
            d                 = (1-τ)*(1-(α+s))*z_vec[z]*k_vec[k]**θ + δ*k_vec[k]*τ - I_p[k,c,z]  - 0.5*a*((I_p[k,c,z] /k_vec[k])**2)*k_vec[k]  - Cp[k,c,z] +c_vec[c]*(1+r*(1-τ))*(1-s)
            if d>=0:
                F_p[k,c,z]    = d/(c_vec[c]+k_vec[k])
            else:
                F_p[k,c,z]    = d*(1+ϕ)/(c_vec[c]+k_vec[k])
            

[i_kstar, j_kstar] = np.unravel_index(np.argmin(np.abs(kstar-k_vec),axis=None),k_vec.shape)       #the plots are with kstar, for different levels of c and z    
logz=np.log(z_vec)
cmed_plot=math.floor(dimc/2)

fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1)
ax1.plot(logz,CF_p[i_kstar,0,:] , label='Low Cash ratio',linestyle = 'dashed', c='b')
ax1.plot(logz,CF_p[i_kstar,cmed_plot,:] , label='Medium Cash ratio',linestyle = 'solid', c='b')
ax1.plot(logz,CF_p[i_kstar,dimc-1,:] , label='High Cash ratio',linestyle = 'dotted', c='b')
ax1.set_xlabel("Log productivity shock")
ax1.set_ylabel("Cash Flow / Capital")
ax1.legend()

ax2.plot(logz,I_p[i_kstar,0,:] , label='Low Cash ratio',linestyle = 'dashed', c='b')
ax2.plot(logz,I_p[i_kstar,cmed_plot,:] , label='Medium Cash ratio',linestyle = 'solid', c='b')
ax2.plot(logz,I_p[i_kstar,dimc-1,:] , label='High Cash ratio',linestyle = 'dotted', c='b')
ax2.set_xlabel("Log productivity shock")
ax2.set_ylabel("Investment / Capital")
ax2.legend()

ax3.plot(logz,CRatio_P[i_kstar,0,:] , label='Low Cash ratio',linestyle = 'dashed', c='b')
ax3.plot(logz,CRatio_P[i_kstar,cmed_plot,:] , label='Medium Cash ratio',linestyle = 'solid', c='b')
ax3.plot(logz,CRatio_P[i_kstar, dimc-1,:] , label='High Cash ratio',linestyle = 'dotted', c='b')
ax3.set_xlabel("Log productivity shock")
ax3.set_ylabel("Cash / Assets")
ax3.legend()

ax4.plot(logz,F_p[i_kstar,0,:] , label='Low Cash ratio',linestyle = 'dashed', c='b')
ax4.plot(logz,F_p[i_kstar,cmed_plot,:] , label='Medium Cash ratio',linestyle = 'solid', c='b')
ax4.plot(logz,F_p[i_kstar, dimc-1,:] , label='High Cash ratio',linestyle = 'dotted', c='b')
ax4.set_xlabel("Log productivity shock")
ax4.set_ylabel("External FIn / Assets")
ax4.legend()

plt.show()



# In[9] Run simulation
num_reps                            = 100
ts_length                           = 1100
kinit                               = math.floor(dimk/2)
cinit                               = math.floor(dimc/2)
[Ksim, Csim,E]                      = model_sim(z_vec,z_prob_mat, kp_vec, cp_vec, Kp, Cp, kinit, cinit,num_reps, ts_length)


"""
COMPARISON (all dim=5):
    InterpC = 0:01:16.77
    Scypy   = way to long
    None    = 0:00:2.43
"""