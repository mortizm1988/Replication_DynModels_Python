#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# # Replication: Agency Conflicts and Cash: Estimates from a Dynamic Model. Nikolov & Whited (2014)
# Author: Marcelo Ortiz (UPF & BSE).
#
#
# ## List of Parameters:
#      α,             # manager bonus
#      β,             # manager equity share
#      s,             # manager private benefit
#      δ,             # capital depreciation
#      λ,             # cost of issuing equity
#      a,             # capital adjustment cost
#      θ,             # curvature production function
#      τ,             # corporate tax rate
#      r,             # interest rate
#      ϕ,             # cost of external finance
#      dimz,          # dim of shock
#      μ,             # log mean of AR process
#      σ,             # s.d. deviation of innovation shocks
#      ρ,             # persistence of AR process
#      stdbound,      # standard deviation of AR process (set to 2)
#      dimk,          # dimension of k, only works with odd numbers
#      dimc,          # dimension of cash
#      dimkp,         # dimension of future k
#      dimcp):        # dimension of future cash
# ## Packing:
#     α, β, s                         = param_manager     # Manager compensation
#     δ, λ, a, θ, τ                   = param_inv         # Investment parameters
#     r, ϕ                            = param_fin         # Financing Parameters
#     μ, σ, ρ, stdbound               = param_ar          # AR Parameters
#     dimz, dimk, dimc, dimkp, dimcp  = param_dim         # dimensional Parameters
"""
## Import Packages
import numpy as np
from time import time
float_formatter = "{:,.2f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})

## Import Modules
import value_interation as vi
import simulation as sim
import smm 


if __name__== '__main__':
    
    ## Figure 1
    # vi.solve_and_figure_1()  # 390 secs, 6 min
    
    ## Figure 2
    #start = time()
    #sim.parallel_simulation(5)   #7200 secs, 120 min,2 hrs.
    #end = time()    
    #print(f'It took {(end - start):,.2f} seconds!')
    
    param_=(0.751/100, 0.051, 0.101/1000,0.13, 0, 1.278, 0.773, 0.2,0.011, 0.043,0, 0.262, 0.713, 4)
    moments = smm.solve_simulate_and_moments(param_)