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
import csv
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# Path for saving iteration results
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mbdoe_iterations.csv")


def save_iteration_results(iteration, best_model, second_best_model, best_aic, second_best_aic, optimal_ic, expected_diff):
    """
    Save the results of each MBDoE iteration to a CSV file.
    
    Args:
        iteration: Current iteration number
        best_model: Best model equation string
        second_best_model: Second best model equation string
        best_aic: AIC value of best model
        second_best_aic: AIC value of second best model
        optimal_ic: Optimal initial conditions for next experiment
        expected_diff: Expected model difference at optimal IC
    """
    file_exists = os.path.exists(RESULTS_FILE)
    
    with open(RESULTS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow([
                'Timestamp', 'Iteration', 'NumExperiments', 
                'BestModel', 'BestAIC', 
                'SecondBestModel', 'SecondBestAIC',
                'OptimalIC_A', 'OptimalIC_B', 'ExpectedModelDiff'
            ])
        
        writer.writerow([
            datetime.now().isoformat(),
            iteration,
            config.NUM_EXP,
            best_model,
            best_aic,
            second_best_model,
            second_best_aic,
            optimal_ic[0] if optimal_ic is not None else None,
            optimal_ic[1] if optimal_ic is not None else None,
            expected_diff
        ])


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
        dAdt = eval(str(local_eqs[0]))
        dBdt = -dAdt  # Assuming A + B = constant (isomerization)
        return [dAdt, dBdt]

    sol = solve_ivp(nest, t, z0, t_eval=t_eval, method="RK45")
    return sol.y


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


def find_optimal_experiment(sym_model_1, sym_model_2, multistart=None, 
                             iteration=None, best_aic=None, second_best_aic=None):
    """
    Find the optimal initial conditions for the next experiment.
    
    Uses multi-start optimization to avoid local minima.
    
    Args:
        sym_model_1: Best model from SR (string)
        sym_model_2: Second-best model from SR (string)
        multistart: Number of random restarts (default from config)
        iteration: Current iteration number (for logging)
        best_aic: AIC of best model (for logging)
        second_best_aic: AIC of second best model (for logging)
        
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
        
        res = minimize(
                MBDoE_objective,
                x0,
                args=(time, model_1, model_2),
                method='L-BFGS-B',
                bounds=bounds
                )
        localsol.append(res.x)
        localval.append(res.fun)

    # Find best solution (minimum value = maximum difference)
    min_idx = np.argmin(localval)
    optimal_ic = localsol[min_idx]
    expected_diff = -localval[min_idx]
    
    print(f"  Optimal experiment found: A0={optimal_ic[0]:.2f}, B0={optimal_ic[1]:.2f}")
    print(f"  Expected model difference: {expected_diff:.4f}")
    
    # Save iteration results to CSV
    if iteration is not None:
        save_iteration_results(
            iteration=iteration,
            best_model=sym_model_1,
            second_best_model=sym_model_2,
            best_aic=best_aic,
            second_best_aic=second_best_aic,
            optimal_ic=optimal_ic,
            expected_diff=expected_diff
        )
    
    return optimal_ic


if __name__ == "__main__":
    # Test with example models
    model_1 = "(-2.3174919959922304*A + B)/(A + B + 2.69285281841595)"
    model_2 = "(-2.4054198028450036*A + B)/(A + 1.233893904170066*B + 2.803231305980434)"
    
    print("Finding optimal experiment...")
    optimal_ic = find_optimal_experiment(model_1, model_2)
    print(f"Result: {optimal_ic}")
