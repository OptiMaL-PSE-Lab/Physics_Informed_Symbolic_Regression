
import numpy as np
import pandas as pd
import os
import sys
import itertools as it
from time import perf_counter

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
import utils

# Global variable for timeout tracking (same pattern as saving_rates.py)
time_in = 0

def my_event(t, y):
    """Event function for solve_ivp to timeout if ODE takes too long (> 2 seconds)."""
    global time_in
    time_out = perf_counter()
    
    if (time_out - time_in) > 2:
        return 0  # Triggers termination
    else:
        return 1  # Continue solving

my_event.terminal = True

def evaluate_models():
    print("Evaluating rate models...")
    
    # Load in-silico data
    in_silico_data = {}
    for i in range(config.NUM_EXP):
        filename = "exp_" + str(i + 1) + ".csv"
        filepath = os.path.join(config.EXP_DATA_DIR, filename)
        df = pd.read_csv(filepath, header=None)
        in_silico_data["exp_" + str(i + 1)] = df.values

    # Load rate models - only A models (same as saving_rates.py)
    # The stoichiometric constraint dB/dt = -dA/dt is applied in utils.rate_model()
    GP_models = {}
    params = []
    
    # Load A and B models for reference, but only use A for evaluation
    for i in range(config.NUM_SPECIES):
        species_name = config.SPECIES[i]
        filename = f"hall_of_fame_rate_{species_name}{config.NUM_EXP}.csv"
        filepath = os.path.join(config.HOF_FILES_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"Error: {filepath} not found.")
            return

        a, b = utils.rate_n_param(filepath)
        GP_models[f"{species_name}_models", f"{species_name}_params"] = a, b
        params.append(b)

    # Only use A models (same as saving_rates.py line 509)
    # The B rate is constrained: dB/dt = -dA/dt
    all_ODEs = GP_models["A_models", "A_params"][0]
    number_models = len(all_ODEs)
    all_ODEs = [[x] for x in all_ODEs]  # Wrap each in a list for rate_model compatibility
    AIC_values = np.zeros(number_models)
    
    print(f"Evaluating {number_models} A models (with dB/dt = -dA/dt constraint)...")
    
    time = np.array(config.TIME_EVAL)
    
    for i in range(number_models):
        neg_log = 0
        if i % 10 == 0:
            print(f"Evaluating model {i}/{number_models}")

        for j in range(config.NUM_EXP):
            global time_in
            experiments = in_silico_data["exp_" + str(j + 1)]
            ics = config.INITIAL_CONDITIONS["ic_" + str(j + 1)]
            
            # Reset timer before solving ODE (same pattern as saving_rates.py)
            time_in = perf_counter()
            y, tt, status = utils.rate_model(ics, list(all_ODEs[i]), [0, np.max(time)], list(time), my_event)

            if status != 0 and status != 1: # Success is usually 0 (end of time) or 1 (event)
                 neg_log = 1e99
                 break
            
            # Sometimes solve_ivp might return fewer points if it fails or event triggers
            # We need to interpolate or check length.
            # Adaptation: if len(y[0]) != len(experiments[0]), handle it.
            if y.shape[1] != experiments.shape[1]:
                 neg_log = 1e99
                 break
            
            neg_log += utils.NLL_kinetics(experiments, y, config.NUM_SPECIES, config.TIMESTEPS)

        # Use params[0][i] for A model parameters (same as saving_rates.py line 537)
        num_parameters = np.sum(np.array(params[0][i]))
        AIC_values[i] = 2 * neg_log + 2 * num_parameters

    # Find best and second-best models
    sorted_indices = np.argsort(AIC_values)
    best_model_index = sorted_indices[0]
    second_best_index = sorted_indices[1] if len(sorted_indices) > 1 else sorted_indices[0]
    
    third_best_index = sorted_indices[2] if len(sorted_indices) > 2 else sorted_indices[0]
    
    best_model = all_ODEs[best_model_index]
    second_best_model = all_ODEs[second_best_index]
    third_best_model = all_ODEs[third_best_index]
    
    print("\n--------------------------------------------------")
    print(f"Best Model: {best_model}")
    print(f"Second Best Model: {second_best_model}")
    print(f"Third Best Model: {third_best_model}")
    print("----------------------")
    print(f"All AIC values: {AIC_values}")
    print(f"Sorted indices: {sorted_indices}")
    print("--------------------------------------------------\n")
    
    # Save results to file
    with open(os.path.join(config.BASE_DIR, "final_result.txt"), "w") as f:
        f.write(f"Best Model Index: {best_model_index}\n")
        f.write(f"Best Equations: {best_model}\n")
        f.write(f"Best AIC: {AIC_values[best_model_index]}\n")
        f.write(f"Second Best Index: {second_best_index}\n")
        f.write(f"Second Best Equations: {second_best_model}\n")
        f.write(f"Second Best AIC: {AIC_values[second_best_index]}\n")
    
    # Return the best model equation for species A (first element of the tuple)
    return {
        'best_model': best_model[0] if best_model else None,
        'second_best_model': second_best_model[0] if second_best_model else None,
        'third_best_model': third_best_model[0] if third_best_model else None,
        'best_aic': AIC_values[best_model_index],
        'second_best_aic': AIC_values[second_best_index],
        'third_best_aic': AIC_values[third_best_index],
        'all_aic': AIC_values
    }

if __name__ == "__main__":
    result = evaluate_models()
    if result:
        print(f"Returned: Best={result['best_model']}")

