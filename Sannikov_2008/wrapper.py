# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.optimize import root

# --- MODEL PARAMETERS ---
r = 0.1      # Discount rate
sigma = 1.0  # Volatility of output process

# --- AGENT'S UTILITY AND COST FUNCTIONS ---
def u(c):
    """ Utility function: u(c) = sqrt(c) """
    return np.sqrt(c)

def h(a):
    """ Cost of effort function: h(a) = 0.5 * a^2 + 0.4 * a """
    return 0.5 * a**2 + 0.4 * a

def h_prime(a):
    """ Derivative of h(a): h'(a) = a + 0.4 """
    return a + 0.4

# --- TERMINATION VALUE FUNCTIONS ---
def F0(W):
    """ Retirement consumption function: F0(W) = -W^2 """
    return -W**2

def F0_prime(W):
    """ First derivative of F0(W) """
    return -2 * W

# --- HJB EQUATION ---
def F_ode(W, F):
    """ System of ODEs: F''(W) and F'(W) """
    F_prime = F[1]
    a_opt = np.clip((1 - F_prime) / 2, 0, 1)  # Optimal effort

    # Incentive sensitivity y(a)
    y_a = h_prime(a_opt)

    # Consumption c(W)
    with np.errstate(divide='ignore', invalid='ignore'):
        c_opt = np.where(F_prime < 0, (1 / (-F_prime))**2, 0)

    # Drift of W process
    drift_W = r * (W - u(c_opt) + h(a_opt))

    # Second-order differential equation for F(W)
    F_double_prime = (-r * (a_opt - c_opt) - F_prime * drift_W) / (0.5 * sigma**2 * y_a**2)

    return np.vstack((F_prime, F_double_prime))

# --- BOUNDARY CONDITIONS ---
def bc(ya, yb, W_gp):
    """ Boundary conditions at W = 0 and dynamically solved W_gp """
    return np.array([
        ya[0],                # F(0) = 0
        float(yb[0]) - float(F0(W_gp))  # Ensure both values are scalars
    ])

# %%

# --- SOLVE FOR W_gp DYNAMICALLY ---
def residual(W_gp_guess):
    """ Solve for W_gp such that F'(W_gp) = -2W_gp """
    sol_temp = solve_bvp(F_ode, lambda ya, yb: bc(ya, yb, W_gp_guess), W_mesh, F_init)
    return sol_temp.y[1, -1] - F0_prime(W_gp_guess)  # Ensure F'(W_gp) = -2W_gp

# Define mesh and initial conditions
W_mesh = np.linspace(0, 1.0, 500)  # High resolution
F_init = np.zeros((2, W_mesh.shape[0]))  # Initial guess

# Solve for W_gp dynamically
W_gp_solution = root(residual, 1.0)  # Initial guess: W_gp = 1.0
W_gp_final = float(W_gp_solution.x[0])  # Extract the correct W_gp

# Solve BVP with the dynamically found W_gp
sol_corrected = solve_bvp(F_ode, lambda ya, yb: bc(ya, yb, W_gp_final), W_mesh, F_init)

# --- EXTRACT RESULTS ---
W_vals, F_vals = sol_corrected.x, sol_corrected.y[0]
F_prime_vals = sol_corrected.y[1]
a_vals = np.clip((1 - F_prime_vals) / 2, 0, 1)  # Optimal effort

# --- PLOT RESULTS ---

plt.figure(figsize=(10, 5))

# Plot Principal's Value Function F(W)
plt.subplot(1, 2, 1)
plt.plot(W_vals, F_vals, label=r"$F(W)$", color='blue', linewidth=2)
plt.xlabel(r"Agent's continuation value $W$")
plt.ylabel("Principal's profit $F(W)$")
plt.title(r"Profit Function with Solved $W_{gp}$")
plt.axhline(0, color='black', linestyle="--")
plt.legend()

# Plot Effort Function a(W)
plt.subplot(1, 2, 2)
plt.plot(W_vals, a_vals, label=r"$a(W)$", color='red', linewidth=2)
plt.xlabel(r"Agent's continuation value $W$")
plt.ylabel("Effort level $a(W)$")
plt.title("Effort Function with Fully Enforced Smooth Pasting")
plt.axhline(0, color='black', linestyle="--")
plt.legend()

plt.tight_layout()
plt.show()
# %%
