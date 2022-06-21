# -*- coding: utf-8 -*-
# # Replication: Agency Conflicts and Cash: Estimates from a Dynamic Model
# Author: Marcelo Ortiz (UPF & BSE).
#
# This code replicates the main results in Nikolov & Whited (2014).
#
# TO DO:
#     (1) Figure 2: why capital depreciation δ reduces cash holding? in the original paper the relation is positive.
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

## Import Packages

import numpy as np
from time import time
float_formatter = "{:,.2f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})

## Import Modules
import value_interation as vi
import simulation as sim


if __name__== '__main__':
    
    ## Figure 1
    #start = time()
    #vi.solve_and_figure_1()  # 390 secs, 6 min
    #end = time()
    # print(f'It took {(end - start):,.2f} seconds!') # 204 seconds (3 min).
    ## Figure 2
    sim.parallel_simulation(5)