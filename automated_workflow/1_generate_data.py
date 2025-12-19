
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import os
import sys

# Add the current directory to path to import config if run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def kinetic_model(t, z):
    k_f = config.K_F
    k_r = config.K_R
    k_A = config.K_A
    k_B = config.K_B
    k_C = config.K_C

    dAdt = (-k_f * z[0] + k_r * z[1]) / (k_A * z[0] + k_B * z[1] + k_C)
    dBdt = (-1) * (-k_f * z[0] + k_r * z[1]) / (k_A * z[0] + k_B * z[1] + k_C)

    dzdt = [dAdt, dBdt]
    return dzdt

def generate_data():
    np.random.seed(config.RANDOM_SEED)
    
    t = config.TIME_SPAN
    t_eval = config.TIME_EVAL
    num_species = config.NUM_SPECIES
    timesteps = config.TIMESTEPS
    initial_conditions = config.INITIAL_CONDITIONS
    num_exp = config.NUM_EXP
    
    noise = [np.random.normal(0, config.STD_NOISE, size = (num_species, timesteps)) for i in range(num_exp)]
    in_silico_data = {}

    print("Generating synthetic data...")
    for i in range(num_exp):
        ic = initial_conditions["ic_" + str(i + 1)]
        solution = solve_ivp(kinetic_model, t, ic, t_eval = t_eval, method = "RK45")
        in_silico_data["exp_" + str(i + 1)] = np.clip(solution.y + noise[i], 0, 1e99)
        
        # Save to CSV
        matrix = in_silico_data["exp_" + str(i + 1)]
        filename = "exp_" + str(i + 1) + ".csv"
        filepath = os.path.join(config.EXP_DATA_DIR, filename)
        
        df = pd.DataFrame(matrix)
        df.to_csv(filepath, index = False, header = False)
        print(f"Saved {filepath}")

if __name__ == "__main__":
    generate_data()
