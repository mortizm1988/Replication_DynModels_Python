# -*- coding: utf-8 -*-

# # Replication: Agency Conflicts and Cash: Estimates from a Dynamic Model. Nikolov & Whited (2014)
# Author: Marcelo Ortiz (UPF & BSE).
#
#
# ## List of Parameters:
#      1. α,             # manager bonus
#      2. β,             # manager equity share
#      3. s,             # manager private benefit
#      4. δ,             # capital depreciation
#      5. λ,             # cost of issuing equity
#      6. a,             # capital adjustment cost
#      7. θ,             # curvature production function
#      8. τ,             # corporate tax rate
#      9. r,             # interest rate
#      10. ϕ,             # cost of external finance
#      11. μ,             # log mean of AR process
#      12. σ,             # s.d. deviation of innovation shocks
#      13. ρ,             # persistence of AR process
#      14. stdbound,      # standard deviation of AR process (set to 2)
#      dimz,          # dim of shock
#      dimk,          # dimension of k, only works with odd numbers
#      dimc,          # dimension of cash
#      dimkp,         # dimension of future k
#      dimcp:        # dimension of future cash
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
import smm 
import pyswarms as ps
from scipy.optimize import basinhopping
import cProfile
import pstats
import snakeviz

if __name__== '__main__':
    ## ---- Figure 1 ----
    start = time()
    #with cProfile.Profile() as pr:
    vi.solve_and_figure_1()  # 305 secs using np.unravel
    # stats=pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    # stats.dump_stats(filename="needs_profiling.prof")
    end = time() 
    print(f'It took {(end - start):,.2f} seconds!')
    
    ## ---- Figure 2 ----
    # start = time()
    # sim.parallel_simulation(5)   #7200 secs, 120 min,2 hrs.
    # end = time()    
    # print(f'It took {(end - start):,.2f} seconds!')
    
    # --- SMM Table 3 ---
    # param_=(0.751/100, 0.051, 0.101/1000, 1.278, 0.773,  0.043, 0.262, 0.713 )
    # ret = smm.optimizer(param_,"Basinhopping")
