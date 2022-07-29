# -*- coding: utf-8 -*-
from numba import jit
import numpy as np
import numpy.typing as npt
import quantecon as qe
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
import math


def set_vec(param_inv, param_fin, param_dim, param_manager, z_vec):
    """
    Compute the vector of capital stock (around the steady-state) and cash (up to k steady state).
    Dimension: k_vec=[dimk , 1]; idem for the rest.
    """
    δ, _, a, θ, τ = param_inv
    r, _ = param_fin
    _, dimk, dimc, dimkp, dimcp = param_dim
    α, β, s = param_manager
    kstar = ((θ*(1-τ))/(r*(1+a*δ)-δ*τ+δ+0.5*a*δ**2))**(1/(1-θ))
    #kstar = ((θ*(1-τ))/(r+δ))**(1/(1-θ))
    #kstar=2083.6801320704803
    k_min    = ((θ*(1-τ)*z_vec[0])/(r*(1+a*δ)-δ*τ+δ+0.5*a*δ**2))**(1/(1-θ)) # A guess for k_min?
    k_max    = ((θ*(1-τ)*(z_vec[-2]))/(r*(1+a*δ)-δ*τ+δ+0.5*a*δ**2))**(1/(1-θ)) # A guess for k_max?
    #kstar  = 0.5*np.take((k_max-k_min)+k_min,0)
    # Set up the vectors

    k_vec = np.reshape(np.linspace(k_min, k_max, dimk), (dimk, 1))
    kp_vec = np.reshape(np.linspace(k_min, k_max, dimkp), (dimkp, 1))
    c_vec = np.reshape(np.linspace(0.0, 0.5*k_max, dimc), (dimc, 1))
    cp_vec = np.reshape(np.linspace(0.0, 0.5*k_max, dimcp), (dimcp, 1))

    grid_points = (np.reshape(k_vec, k_vec.size), np.reshape(c_vec, c_vec.size), np.reshape(z_vec, z_vec.size))
    grid_to_interp = [[np.take(kp, 0), np.take(cp, 0), np.take(z, 0)] for kp in kp_vec for cp in cp_vec for z in z_vec]
    return k_vec, kp_vec, c_vec, cp_vec, kstar, grid_points, grid_to_interp


def trans_matrix(param_ar, param_dim):
    """
    Set the State vector for the productivity shocks Z and transition matrix.
    Dimension: z_vec =[dimZ , 1]; z_prob_mat=[dimz , dimZ] // Remember, sum(z_prob_mat[i,:])=1.
    *** Pending improvements: clearly last transposition is inefficient.
    """
    μ, σ, ρ, stdbound = param_ar
    dimz, dimk, *_ = param_dim

    mc = qe.markov.approximation.tauchen(ρ, σ, μ, stdbound, dimz)
    z_vec = mc.state_values
    Pi = mc.P
    z_vec = z_vec.reshape(dimz, 1)
    z_vec = np.e**(z_vec)
    z_prob_mat = Pi.reshape(dimz, dimz)
    z_prob_mat = np.transpose(z_prob_mat)
    return z_vec, z_prob_mat


@jit(nopython=True, parallel=False)
def rewards_grids(param_manager, param_inv, param_fin, param_dim, z_vec, k_vec, c_vec, kp_vec,cp_vec):
    """
    Compute the manager's and shareholders' cash-flows  R and D, respectively,
    for every (k_t, k_t+1, c_t, c_t+1) combination and every productivity shocks.
    """
    α, β, s = param_manager     # Manager compensation
    δ, λ, a, θ, τ = param_inv         # Investment parameters
    r, ϕ = param_fin         # Financing Parameters
    dimz, dimk, dimc, dimkp, dimcp = param_dim         # dimensional Parameters

    R = np.zeros((dimk, dimkp, dimc, dimcp, dimz))
    D = np.zeros((dimk, dimkp, dimc, dimcp, dimz))
    inv: float
    d: float
    rw: float
    kp: float
    k: float
    z: float
    cp: float
    c: float
    print("Computing reward matrix \n")
    for i_k in range(dimk):
        for i_kp in range(dimkp):
            for i_c in range(dimc):
                for i_cp in range(dimcp):
                    for i_z in range(dimz):
                        kp = np.take(kp_vec[i_kp], 0)
                        k = np.take(k_vec[i_k], 0)
                        z = np.take(z_vec[i_z], 0)
                        cp = np.take(cp_vec[i_cp], 0)
                        c = np.take(c_vec[i_c], 0)
                        inv = kp-(1-δ)*k
                        d = (1-τ)*(1-(α+s))*z*k**θ + δ*k*τ - inv - 0.5 *a*((inv/k)**2)*k - cp + c*(1+r*(1-τ))*(1-s)
                        if d >= 0:
                            D[i_k, i_kp, i_c, i_cp, i_z] = d
                        else:
                            D[i_k, i_kp, i_c, i_cp, i_z] = d*(1+ϕ)
                        rw = (α+s)*z*k**θ + s*c*(1+r) + β*D[i_k, i_kp, i_c, i_cp, i_z]
                        R[i_k, i_kp, i_c, i_cp, i_z] = rw
    print("Computing reward matrix - Done \n")
    return R, D

@jit(nopython=True,parallel=False)  # it does not run with jit since this package seems to not recognice scypy.interpn().
def continuation_value(param_dim: npt.ArrayLike, U: npt.ArrayLike, z_prob_mat: npt.ArrayLike, Uinter: npt.ArrayLike):
    """
    Compute "Continuation Value" for every possible future state of nature (kp,cp,z).
    The "continuation value" is defined as: E[U(kp,cp,zp)]=sum{U(kp,cp,zp)*Prob(zp,p)}
    *** Pending improvements: matrix multiplication instead of a double loop.
    """
    dimz, dimk, dimc, dimkp, dimcp = param_dim         # dimensional Parameters
    cont_value = np.zeros((dimkp, dimcp, dimz))
    for ind_z in range(dimz):
        for i_kpp in range(dimkp):
            for i_cpp in range(dimcp):
                cont_value[i_kpp, i_cpp, ind_z] = np.dot(z_prob_mat[:, ind_z], Uinter[i_kpp, i_cpp, :])
    return cont_value


def bellman_operator(param_dim: npt.ArrayLike, param_fin: npt.ArrayLike, Upol: npt.ArrayLike, R: npt.ArrayLike, z_prob_mat: npt.ArrayLike,z_vec:npt.ArrayLike,k_vec:npt.ArrayLike, c_vec:npt.ArrayLike ,grid: npt.ArrayLike, grid_interp: npt.ArrayLike, i_kpol: npt.ArrayLike, i_cpol: npt.ArrayLike):
    """
    Second, identify max policy and save it.
    For each current state of nature (k,c,z), find the policy {kp,cp} that maximizes RHS: U(k,c,z) + E[U(kp,cp,zp)].
    Once found it, update the value of U in this current state of nature with the one generated with the optimal policy
    and save the respective optimal policy (kp,cp) for each state of nature (k,c,z).
    *** Pending improvements: something faster than "enumerate"?
    """
    dimz, dimk, dimc, dimkp, dimcp = param_dim         # dimensional Parameters
    r, _ = param_fin
    Uinter = interpn(grid, Upol, grid_interp)
    Uinter = Uinter.reshape((dimkp, dimcp, dimz))
    c_value = continuation_value(param_dim, Upol, z_prob_mat, Uinter)
    RHS = np.empty((dimkp, dimcp))
    for (i_z, z) in enumerate(z_vec):
        for (i_k, k) in enumerate(k_vec):
            for (i_c, c) in enumerate(c_vec):
                RHS = R[i_k, :, i_c, :, i_z] + (1/(1+r))*c_value[:, :, i_z]
                # the index of the best expected value for each k,c,z combination.
                i_kc = np.unravel_index(np.argmax(RHS, axis=None), RHS.shape)
                # update U with all the best expected values.
                Upol[i_k, i_c, i_z] = RHS[i_kc]
                i_kpol[i_k, i_c, i_z] = i_kc[0]
                i_cpol[i_k, i_c, i_z] = i_kc[1]
    return Upol, i_kpol, i_cpol


def value_iteration(param_dim: npt.ArrayLike, param_fin: npt.ArrayLike, R: npt.ArrayLike, z_prob_mat: npt.ArrayLike, k_vec: npt.ArrayLike, c_vec: npt.ArrayLike, z_vec: npt.ArrayLike, kp_vec: npt.ArrayLike, cp_vec: npt.ArrayLike, grid_points, grid_to_interp, diff=1, tol=1e-6, imax=10_000):
    """
    Value Iteration on Eq 6.
    *** Pending improvements: why ndenumerate and not numerate?
    """
    dimz, dimk, dimc, dimkp, dimcp = param_dim
    Upol = np.zeros((dimk, dimc, dimz))
    i_kpol = np.empty((dimk, dimc, dimz), dtype=float)
    i_cpol = np.empty((dimk, dimc, dimz), dtype=float)

    print("Optimal policies: Iteration start \n")
    for i in range(imax):
        U_old = np.copy(Upol)
        Upol, i_kpol, i_cpol = bellman_operator(param_dim, param_fin, Upol, R, z_prob_mat, z_vec,k_vec, c_vec, grid_points, grid_to_interp, i_kpol, i_cpol)
        diff = np.max(np.abs(Upol-U_old))
        if i == 1:
            print(f"Error at iteration {i} is {diff:,.2f}.\n")
        if i % 300 == 0:
            print(f"Error at iteration {i} is {diff:,.2f}.\n")
        if diff < tol:
            print(f"Solution found at iteration {i}.\n")
            break
        if i == imax:
            print("Failed to converge!")
    # Evaluating the optimal policies using the indexes obtained in the iterations.
    Kpol = np.zeros((dimk, dimc, dimz))
    Cpol = np.zeros((dimk, dimc, dimz))
    for index, value in np.ndenumerate(i_kpol):
        index2 = int(value)
        # 2022-06-02 changed from kp_vec to k_vec. The same in the following loop with c_vec.
        Kpol[index] = kp_vec[index2]
    for index, value in np.ndenumerate(i_cpol):
        index2 = int(value)
        Cpol[index] = cp_vec[index2]

    return Upol, Kpol, Cpol, i_kpol, i_cpol


def value_iteration_firm_value(param_dim: npt.ArrayLike, param_fin: npt.ArrayLike, D: npt.ArrayLike, z_prob_mat: npt.ArrayLike, k_vec: npt.ArrayLike, c_vec: npt.ArrayLike, z_vec: npt.ArrayLike, i_kpol, i_cpol, grid_points, grid_to_interp, diff=1, tol=1e-6, imax=10_000):
    """
    Value Iteration on Eq 8.
    *** Pending improvements: why ndenumerate and not numerate?
    """
    dimz, dimk, dimc, dimkp, dimcp = param_dim
    Vpol = np.zeros((dimk, dimc, dimz))

    print("Firm Value: Iteration start \n")
    for i in range(imax):
        V_old = np.copy(Vpol)
        Vpol, *_ = bellman_operator(param_dim, param_fin, Vpol, D, z_prob_mat, z_vec,k_vec, c_vec, grid_points, grid_to_interp, i_kpol, i_cpol)
        diff = np.max(np.abs(Vpol-V_old))
       
        if diff < tol:
            print(f"Solution found at iteration {i}.\n")
            break
        if i == imax:
            print("Failed to converge!")
  

    return Vpol

def plot_policy_function(param_manager,param_inv,param_fin,param_dim,z_vec,k_vec,kstar,c_vec,Kp,Cp):
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
    
    fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1,figsize=(15, 15))
    ax1.plot(logz,CF_p[i_kstar,0,:] , label='Low Cash ratio',linestyle = 'dashed', c='b')
    ax1.plot(logz,CF_p[i_kstar,cmed_plot,:] , label='Medium Cash ratio',linestyle = 'solid', c='b')
    ax1.plot(logz,CF_p[i_kstar,-1,:] , label='High Cash ratio',linestyle = 'dotted', c='b')
    ax1.set_xlabel("Log productivity shock")
    ax1.set_ylabel("Cash Flow / Capital")
    ax1.legend()
    
    ax2.plot(logz,I_p[i_kstar,0,:] , label='Low Cash ratio',linestyle = 'dashed', c='b')
    ax2.plot(logz,I_p[i_kstar,cmed_plot,:] , label='Medium Cash ratio',linestyle = 'solid', c='b')
    ax2.plot(logz,I_p[i_kstar,-1,:] , label='High Cash ratio',linestyle = 'dotted', c='b')
    ax2.set_xlabel("Log productivity shock")
    ax2.set_ylabel("Investment / Capital")
    #ax2.legend()
    #
    ax3.plot(logz,CRatio_P[i_kstar,0,:] , label='Low Cash ratio',linestyle = 'dashed', c='b')
    ax3.plot(logz,CRatio_P[i_kstar,cmed_plot,:] , label='Medium Cash ratio',linestyle = 'solid', c='b')
    ax3.plot(logz,CRatio_P[i_kstar,-1,:] , label='High Cash ratio',linestyle = 'dotted', c='b')
    ax3.set_xlabel("Log productivity shock")
    ax3.set_ylabel("Cash / Assets")
    #ax3.legend()
    
    ax4.plot(logz,F_p[i_kstar,0,:] , label='Low Cash ratio',linestyle = 'dashed', c='b')
    ax4.plot(logz,F_p[i_kstar,cmed_plot,:] , label='Medium Cash ratio',linestyle = 'solid', c='b')
    ax4.plot(logz,F_p[i_kstar,-1,:] , label='High Cash ratio',linestyle = 'dotted', c='b')
    ax4.set_xlabel("Log productivity shock")
    ax4.set_ylabel("External FIn / Assets")
    #ax4.legend()
    
    plt.show()
    fig.savefig("Figure1.png", bbox_inches='tight', dpi=600)

def solve_and_figure_1():
    param_manager = (0.751/100, 0.051, 0.101/1000)  # (α, β, s)
    param_inv = (0.13, 0, 1.278, 0.773, 0.2)  # (δ, λ, a, θ, τ)
    param_fin = (0.011, 0.043)                 # (r, ϕ=0.043)
    param_ar = (0, 0.262, 0.713, 4)           # (μ, σ, ρ, stdbound)
    _nk = 10                             # intermediate points in the capital grid
    _nc = 10                             # intermediate points in the cash grid
    (dimc, dimk, dimz) = (11, 25, 5)
    param_dim = (dimz, dimk, dimc, dimk*_nk, dimc*_nc)
    
    z_vec, z_prob_mat = trans_matrix(param_ar, param_dim)
    k_vec, kp_vec, c_vec, cp_vec, kstar, grid_points, grid_to_interp = set_vec(param_inv, param_fin, param_dim, param_manager, z_vec)
    R, _ = rewards_grids(param_manager, param_inv, param_fin, param_dim, z_vec, k_vec, c_vec, kp_vec, cp_vec)
    
    Upol, Kpol, Cpol, *_ = value_iteration(param_dim, param_fin, R, z_prob_mat, k_vec, c_vec, z_vec, kp_vec, cp_vec, grid_points, grid_to_interp)
    plot_policy_function(param_manager,param_inv,param_fin,param_dim,z_vec,k_vec,kstar,c_vec,Kpol,Cpol)