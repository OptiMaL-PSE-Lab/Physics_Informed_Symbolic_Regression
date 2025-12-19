
import numpy as np
import pandas as pd
import os
import sys
from utils import der
import matplotlib.pyplot as plt

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
                derivative[h] = der(best_model_func, time[h], dx = 1e-6)
            
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

if __name__ == "__main__":
    process_concentration()
