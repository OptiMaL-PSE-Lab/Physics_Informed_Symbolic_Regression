"""
Model-Based Design of Experiments (MBDoE) for optimal experiment selection.

This module finds the optimal initial conditions for the next experiment that
maximizes the difference between competing models.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


def SR_model(z0, equations, t, t_eval):
    """
    Solve an ODE system defined by symbolic regression equations.
    
    Args:
        z0: Initial conditions [A0, B0]
        equations: List of equation strings for dA/dt (dB/dt is assumed to be -dA/dt)
        t: Time span [t_start, t_end]
        t_eval: Time points to evaluate
        
    Returns:
        Solution array of shape (num_species, num_timepoints)
    """
    local_eqs = equations.copy()
    
    for i, equation in enumerate(local_eqs):
        equation = str(equation)
        equation = equation.replace("A", "z[0]")
        equation = equation.replace("B", "z[1]")
        local_eqs[i] = equation

    def nest(t, z):
        try:
            dAdt = eval(str(local_eqs[0]))
            dBdt = -dAdt  # Assuming A + B = constant (isomerization)
            return [dAdt, dBdt]
        except:
            return [0, 0]

    try:
        sol = solve_ivp(nest, t, z0, t_eval=t_eval, method="RK45")
        return sol.y
    except:
        return np.zeros((2, len(t_eval)))


def MBDoE_objective(ic, time, sym_model_1, sym_model_2):
    """
    Objective function for MBDoE optimization.
    
    Finds initial conditions that MAXIMIZE the difference between two models
    (returns negative because we use minimization).
    
    Args:
        ic: Initial conditions [A0, B0]
        time: Time points for evaluation
        sym_model_1: First model (best from SR)
        sym_model_2: Second model (second-best from SR)
        
    Returns:
        Negative sum of squared differences (for minimization)
    """
    t_span = [0, np.max(time)]
    
    y1 = SR_model(ic, sym_model_1.copy(), t_span, list(time))
    y2 = SR_model(ic, sym_model_2.copy(), t_span, list(time))
    
    # Maximize difference (return negative for minimization)
    difference = -np.sum((y1 - y2) ** 2)
    return difference


def find_optimal_experiment(sym_model_1, sym_model_2, multistart=None):
    """
    Find the optimal initial conditions for the next experiment.
    
    Uses multi-start optimization to avoid local minima.
    
    Args:
        sym_model_1: Best model from SR (string)
        sym_model_2: Second-best model from SR (string)
        multistart: Number of random restarts (default from config)
        
    Returns:
        Optimal initial conditions [A0, B0]
    """
    if multistart is None:
        multistart = config.MBDOE_MULTISTART
    
    lower_bound = config.IC_LOWER_BOUND
    upper_bound = config.IC_UPPER_BOUND
    time = np.array(config.TIME_EVAL)
    
    # Convert to list format expected by SR_model
    model_1 = [sym_model_1] if isinstance(sym_model_1, str) else list(sym_model_1)
    model_2 = [sym_model_2] if isinstance(sym_model_2, str) else list(sym_model_2)
    
    bounds = [(lower_bound[i], upper_bound[i]) for i in range(len(lower_bound))]
    
    localsol = []
    localval = []
    
    for i in range(multistart):
        x0 = np.random.uniform(lower_bound, upper_bound)
        
        try:
            res = minimize(
                MBDoE_objective,
                x0,
                args=(time, model_1, model_2),
                method='L-BFGS-B',
                bounds=bounds
            )
            localsol.append(res.x)
            localval.append(res.fun)
        except Exception as e:
            print(f"  Optimization run {i+1} failed: {e}")
            continue
    
    if not localval:
        # Fallback: return random IC
        print("  All optimizations failed. Using random IC.")
        return np.random.uniform(lower_bound, upper_bound)
    
    # Find best solution (minimum value = maximum difference)
    min_idx = np.argmin(localval)
    optimal_ic = localsol[min_idx]
    
    print(f"  Optimal experiment found: A0={optimal_ic[0]:.2f}, B0={optimal_ic[1]:.2f}")
    print(f"  Expected model difference: {-localval[min_idx]:.4f}")
    
    return optimal_ic


if __name__ == "__main__":
    # Test with example models
    model_1 = "-0.15911*A**3/(0.1209*A**3 + B**2 - 0.2028)"
    model_2 = "0.1141 - 0.1391*A"
    
    print("Finding optimal experiment...")
    optimal_ic = find_optimal_experiment(model_1, model_2)
    print(f"Result: {optimal_ic}")
