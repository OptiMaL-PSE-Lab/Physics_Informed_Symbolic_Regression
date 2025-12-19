
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

def evaluate_models():
    print("Evaluating rate models...")
    
    # Load in-silico data
    in_silico_data = {}
    for i in range(config.NUM_EXP):
        filename = "exp_" + str(i + 1) + ".csv"
        filepath = os.path.join(config.EXP_DATA_DIR, filename)
        df = pd.read_csv(filepath, header=None)
        in_silico_data["exp_" + str(i + 1)] = df.values

    # Load rate models
    GP_models = {}
    
    # Assuming file naming convention from original code
    # hall_of_fame_rate_A{num_exp}.csv
    for i in range(config.NUM_SPECIES):
        species_name = config.SPECIES[i]
        filename = f"hall_of_fame_rate_{species_name}{config.NUM_EXP}.csv"
        filepath = os.path.join(config.HOF_FILES_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"Error: {filepath} not found.")
            return

        a, b = utils.rate_n_param(filepath)
        GP_models[f"{species_name}_models", f"{species_name}_params"] = a, b

    # Create all combinations of ODEs
    equations = []
    # Order: A_models, A_params, B_models, B_params
    names = []
    for s in config.SPECIES:
        names.append(f"{s}_models")
        names.append(f"{s}_params")
    
    all_models = []
    params = []

    for i in range(0, len(names), 2):
        if (names[i], names[i+1]) in GP_models:
            all_models.append(GP_models[names[i], names[i + 1]][0])
            params.append(GP_models[names[i], names[i + 1]][1])
        else:
            print(f"Missing models for {names[i]}")
            return

    all_ODEs = list(it.product(*all_models))
    param_ODEs = list(it.product(*params))
    
    number_models = len(all_ODEs)
    AIC_values = np.zeros(number_models)
    
    print(f"Evaluating {number_models} model combinations...")
    
    time = np.array(config.TIME_EVAL)
    
    for i in range(number_models):
        neg_log = 0
        if i % 10 == 0:
            print(f"Evaluating model {i}/{number_models}")

        for j in range(config.NUM_EXP):
            experiments = in_silico_data["exp_" + str(j + 1)]
            ics = config.INITIAL_CONDITIONS["ic_" + str(j + 1)]
            
            # Solve ODE
            # rate_model signature: z0, equations, t, t_eval, event
            y, tt, status = utils.rate_model(ics, list(all_ODEs[i]), [0, np.max(time)], list(time), utils.my_event)

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

        num_parameters = np.sum(np.array(param_ODEs[i]))
        AIC_values[i] = 2 * neg_log + 2 * num_parameters

    # Find best and second-best models
    sorted_indices = np.argsort(AIC_values)
    best_model_index = sorted_indices[0]
    second_best_index = sorted_indices[1] if len(sorted_indices) > 1 else sorted_indices[0]
    
    best_model = all_ODEs[best_model_index]
    second_best_model = all_ODEs[second_best_index]
    
    print("\n--------------------------------------------------")
    print(f"Best Model Index: {best_model_index}")
    print(f"Best Model Equations: {best_model}")
    print(f"AIC: {AIC_values[best_model_index]}")
    print(f"\nSecond Best Model Index: {second_best_index}")
    print(f"Second Best Model Equations: {second_best_model}")
    print(f"AIC: {AIC_values[second_best_index]}")
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
        'best_aic': AIC_values[best_model_index],
        'second_best_aic': AIC_values[second_best_index],
        'all_aic': AIC_values
    }

if __name__ == "__main__":
    result = evaluate_models()
    if result:
        print(f"Returned: Best={result['best_model']}")

