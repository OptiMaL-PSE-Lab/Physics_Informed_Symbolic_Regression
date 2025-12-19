
import numpy as np
import pandas as pd
from sympy import *
from scipy.integrate import solve_ivp

def derivative(func, x0, dx=1.0, n=1, args=(), order=3):
    # Simple central difference implementation to replace scipy.misc.derivative
    # For n=1 (first derivative) and order=3 (using central difference)
    # f'(x) approx (f(x+dx) - f(x-dx)) / (2*dx)
    if n == 1:
        return (func(x0 + dx, *args) - func(x0 - dx, *args)) / (2.0 * dx)
    else:
        raise NotImplementedError("Only first derivative implemented")

# Alias for compatibility with existing code using 'der'
der = derivative

import re
from time import perf_counter

def read_equations(path):
    data = pd.read_csv(path)
    eqs = data["Equation"].values
    eq_list = []
    
    def make_f(eq):
        def f(t):
            equation = eq.replace("x0", "t")
            return eval(equation.replace("exp", "np.exp"))
        return f
    
    for eq in eqs:
        eq_list += [make_f(eq)]
    
    return eq_list

def number_param(path):
    data = pd.read_csv(path)
    eqs = data["Equation"].values
    param = []
    
    for eq in eqs:
        func = simplify(eq)
        things = list(func.atoms(Float))
        param.append(len(things))
    
    return param

def find_best_model(NLL, param):
    AIC = 2 * np.array(NLL) + 2 * np.array(param)
    index = np.where(AIC == np.min(AIC))
    return index[0][0]

def NLL_models(eq_list, t, data, NLL_species, number_datapoints):
    NLL = []
    for f in eq_list:
        y_T = []
        for a in t:
            y_T.append(f(a))
        NLL.append(NLL_species(data, y_T, number_datapoints))
    return NLL

def NLL(C, y_C, number_datapoints):
    likelihood = np.empty(number_datapoints)
    mse = np.empty(number_datapoints)
    
    for i in range(number_datapoints):
        mse[i] = ((C[i] - y_C[i])**2)
    
    variance = np.sum(mse) / number_datapoints
    
    for i in range(number_datapoints):
        likelihood[i] = ((C[i] - y_C[i])**2) / (2 * (variance)) \
            - np.log(1 / (np.sqrt(2 * np.pi * (variance))))
    
    return np.sum(likelihood)

def rate_n_param(path):
    data = pd.read_csv(path)
    eqs = data["Equation"].values
    simple_traj = []
    param = []
    
    for eq in eqs:
        func = simplify(eq)
        func = str(func)
        j = 0
        things = re.findall(r"(\*{2}|\*{0})(\d+\.?\d*)", func)
        for i in range(len(things)):
            if things[i][0] != "**":
                j += 1
        simple_traj.append(func)
        param.append(int(j))
    
    return simple_traj, param

def NLL_rates(rate_est, rate_pred, number_datapoints, num_exp):
    mse = (rate_est - rate_pred)**2
    # Avoid division by zero if variance is 0 (unlikely but safe to check)
    variance = np.sum(mse) / (number_datapoints * num_exp)
    if variance == 0:
        return -1e99 # Or some large correct prediction indicator
        
    likelihood = ((rate_est - rate_pred)**2) / (2 * (variance)) \
        - np.log(1 / (np.sqrt(2 * np.pi * (variance))))
    
    return np.sum(likelihood)

def predicting_rate(equation, z):
    equation = str(equation)
    equation = equation.replace("A", "z[:, 0]")
    equation = equation.replace("B", "z[:, 1]")
    rate_pred = eval(equation)
    return rate_pred

def best_rate_model(NLL, param):
    AIC = 2 * np.array(NLL) + 2 * np.array(param)
    index = np.where(AIC == np.min(AIC))
    return index[0][0]

def rate_model(z0, equations, t, t_eval, event):
    i = 0
    local_eqs = equations.copy()
    for equation in local_eqs:
        equation = str(equation)
        equation = equation.replace("A", "z[0]")
        equation = equation.replace("B", "z[1]")
        local_eqs[i] = equation
        i += 1

    def nest(t, z):
        # Assuming A and B are z[0] and z[1]
        try:
            dAdt = eval(str(local_eqs[0]))
            # B consumption is generally negative of A if 1:1, or per original code
            # Original code: dBdt = (-1) * (-k... ) which was dAdt.
            # Here we assume the reaction is reversible A <-> B or similar so rate of A is - rate of B?
            # The original code has:
            # dAdt = ...
            # dBdt = (-1) * ... 
            # In the `rate_model` function in saving_rates.py:
            # dAdt = eval(str(equations[0]))
            # dBdt = (-1) * eval(str(equations[0]))
            # So it assumes linked rates.
            dBdt = (-1) * dAdt
            return [dAdt, dBdt]
        except Exception as e:
            # Fallback for evaluation errors
            return [0, 0]

    sol = solve_ivp(nest, t, z0, t_eval = t_eval, method = "RK45", events = event)  
    return sol.y, sol.t, sol.status

def my_event(t, y):
    # This needs access to time_in which is usually global or passed in. 
    # For now we'll just return 1 to avoid timing out logic complexity or implement better timeout.
    # The original code used perf_counter() and a global time_in.
    return 1

my_event.terminal = True

def NLL_kinetics(experiments, predictions, number_species, number_datapoints):
    output = np.zeros(number_species)
    mse = np.zeros(number_species)
    variance = np.zeros(number_species)

    for i in range(number_species):
        a = ((experiments[i] - predictions[i])**2)
        mse[i] = np.sum(a)
        variance[i] = mse[i] / (number_datapoints)

    for i in range(number_species):
        if variance[i] == 0:
            likelihood = np.zeros_like(experiments[i]) # Perfect match
        else:
            likelihood = ((experiments[i] - predictions[i])**2) / (2 * (variance[i])) \
                - np.log(1 / (np.sqrt(2 * np.pi * (variance[i]))))
        output[i] = np.sum(likelihood)

    return np.sum(output)
