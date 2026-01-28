
import numpy as np
import pandas as pd
import os
import sys
from utils import der
import matplotlib.pyplot as plt
from scipy.differentiate import derivative as scipy_derivative

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
import utils

def process_concentration():
    print("Processing concentration profiles...")
    
    # Load in-silico data
    in_silico_data = {}
    for i in range(config.NUM_EXP):
        filename = "exp_" + str(i + 1) + ".csv"
        filepath = os.path.join(config.EXP_DATA_DIR, filename)
        # Read without header
        df = pd.read_csv(filepath, header=None)
        in_silico_data["exp_" + str(i + 1)] = df.values

    # Find best models
    equation_lists = {}
    best_models = {}
    
    time = np.array(config.TIME_EVAL)
    timesteps = config.TIMESTEPS
    
    for i in range(config.NUM_EXP):
        data = in_silico_data["exp_" + str(i + 1)]
        
        for j in range(config.NUM_SPECIES):
            species_name = config.SPECIES[j]
            filename = f"hall_of_fame_{species_name}{i + 1}.csv"
            filepath = os.path.join(config.HOF_FILES_DIR, filename)
            
            if not os.path.exists(filepath):
                print(f"Warning: {filepath} not found. Skipping.")
                continue

            a = utils.read_equations(filepath)
            # data[j] is the row for species j
            nll_a = utils.NLL_models(a, time, data[j], utils.NLL, timesteps)
            param_a = utils.number_param(filepath)
            
            name = f"{species_name}_{i + 1}"
            best_idx = utils.find_best_model(nll_a, param_a)
            best_models[name] = best_idx
            equation_lists[name] = a
            
            print(f"Best model for {name}: Index {best_idx}")

    # Derivative Calculations
    derivatives = {}
    SR_derivatives_A = np.array([])
    SR_derivatives_B = np.array([])

    for i in range(config.NUM_EXP):
        for j in range(config.NUM_SPECIES):
            species_name = config.SPECIES[j]
            name = f"{species_name}_{i + 1}"
            
            if name not in best_models:
                continue
                
            model_idx = best_models[name]
            best_model_func = equation_lists[name][model_idx]
            
            derivative = np.zeros(timesteps)
            for h in range(timesteps):
                result = scipy_derivative(best_model_func, time[h], initial_step=1e-6)
                derivative[h] = result.df
            
            derivatives[name] = derivative

    # Prepare data for rate SR
    # Concatenate derivatives
    for i in range(config.NUM_EXP):
        name_A = f"A_{i + 1}"
        name_B = f"B_{i + 1}"
        if name_A in derivatives:
            SR_derivatives_A = np.concatenate([SR_derivatives_A, derivatives[name_A]])
        if name_B in derivatives:
            SR_derivatives_B = np.concatenate([SR_derivatives_B, derivatives[name_B]])

    # Stack concentration data
    # exp_1.T, exp_2.T ...
    # in_silico_data["exp_1"] is shape (2, 15). Transpose is (15, 2)
    sr_data = in_silico_data["exp_1"].T
    for i in range(1, config.NUM_EXP):
        c = in_silico_data["exp_" + str(i + 1)].T
        sr_data = np.vstack((sr_data, c))

    # Save to const_data
    # sr_data contains [A, B] columns.
    # slice [:, 0:2] ? original code did [:, 0:3] but there are only 2 species.
    # Original: sr_data[:, 0:3].T
    # If species is 2, it should be 0:2. The original code variable `species` had length 2.
    # But `sr_data` structure:
    # exp 1 (15 rows), exp 2 (15 rows)... total 75 rows.
    # cols: A, B
    
    # Check original code line 313: save_matrix_as_csv(sr_data[:, 0:3].T, 'conc_data_for_rate_models')
    # Maybe dynamic?
    # I'll stick to config.NUM_SPECIES
    
    conc_data_to_save = sr_data[:, 0:config.NUM_SPECIES].T
    
    # Save files
    pd.DataFrame(conc_data_to_save).to_csv(os.path.join(config.CONST_DATA_DIR, 'conc_data_for_rate_models.csv'), index=False, header=False)
    
    pd.DataFrame(np.reshape(SR_derivatives_A, (1, len(SR_derivatives_A)))).to_csv(
        os.path.join(config.CONST_DATA_DIR, 'rate_data_A.csv'), index=False, header=False)
        
    pd.DataFrame(np.reshape(SR_derivatives_B, (1, len(SR_derivatives_B)))).to_csv(
        os.path.join(config.CONST_DATA_DIR, 'rate_data_B.csv'), index=False, header=False)

    print("Saved calculated derivatives to const_data.")
    
    return in_silico_data, derivatives, time


def plot_derivatives(in_silico_data, derivatives, time):
    """
    Plot the estimated rates vs actual rates for each experiment.
    Matches the plotting style from saving_rates.py (around line 281).
    """
    # Styling
    color_1 = ['salmon', 'royalblue', 'darkviolet']
    marker = ['o', 'o', 'o', 'o']
    
    # Load no-noise data for actual rates comparison
    from scipy.integrate import solve_ivp
    
    # Kinetic model for computing actual rates
    def kinetic_model(t, z):
        k_f = 7
        k_r = 3 
        k_A = 4 
        k_B = 2 
        k_C = 6 
        dAdt = (-k_f * z[0] + k_r * z[1]) / (k_A * z[0] + k_B * z[1] + k_C)
        dBdt = (-1) * (-k_f * z[0] + k_r * z[1]) / (k_A * z[0] + k_B * z[1] + k_C)
        dzdt = [dAdt, dBdt]
        return dzdt
    
    # Generate no_noise_data
    initial_conditions = {
        "ic_1": np.array([2 , 0]),
        "ic_2": np.array([10, 0]),
        "ic_3": np.array([2 , 2]),
        "ic_4": np.array([10, 2]),
        "ic_5": np.array([10, 1]),
    }
    
    no_noise_data = {}
    t_span = [0, np.max(time)]
    t_eval = list(time)
    
    for i in range(config.NUM_EXP):
        ic = initial_conditions["ic_" + str(i + 1)]
        solution = solve_ivp(kinetic_model, t_span, ic, t_eval=t_eval, method="RK45")
        no_noise_data["exp_" + str(i + 1)] = solution.y
    
    # Plotting the estimated rates and the actual rates
    for i in range(config.NUM_EXP):
        fig, ax = plt.subplots()
        ax.set_ylabel("Rate $(Mh^{-1})$", fontsize=18)
        ax.set_xlabel("Time $(h)$", fontsize=18)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        data = no_noise_data["exp_" + str(i + 1)]
        y = kinetic_model(time, data)
        ax.tick_params(axis='both', which='major', labelsize=18)
        
        for j in range(config.NUM_SPECIES):
            species_name = config.SPECIES[j]
            name = f"{species_name}_{i + 1}"
            if name in derivatives:
                yy = derivatives[name]
                ax.plot(time, y[j], marker[j], markersize=4, label=species_name, color=color_1[j])
                ax.plot(time, yy, color=color_1[j], linestyle="-")
        
        ax.grid(alpha=0.5)
        ax.legend(loc='upper right', fontsize=15)
    
    plt.show()


if __name__ == "__main__":
    in_silico_data, derivatives, time = process_concentration()
    # plot_derivatives(in_silico_data, derivatives, time)
