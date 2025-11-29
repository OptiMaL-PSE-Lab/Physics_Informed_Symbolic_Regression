import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os

class KineticModel:
    def __init__(self, name, num_params, species_names, initial_conditions, data_path_template):
        self.name = name
        self.num_params = num_params
        self.species_names = species_names
        self.num_species = len(species_names)
        self.initial_conditions = initial_conditions
        self.data_path_template = data_path_template
        self.observations = self.load_data()

    def load_data(self):
        observations = []
        num_datasets = len(self.initial_conditions)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        
        for i in range(num_datasets):
            path = os.path.join(base_dir, self.data_path_template.format(i + 1))
            if os.path.exists(path):
                observations.append(pd.read_csv(path, header=None).to_numpy())
            else:
                print(f"Warning: Data file not found at {path}")
                observations.append(None)
        return observations

    def ode_system(self, t, state, parameters):
        raise NotImplementedError("Subclasses must implement ode_system")

    def solve(self, parameters, time_points, initial_condition):
        t_span = [time_points[0], time_points[-1]]
        sol = solve_ivp(
            self.ode_system, 
            t_span, 
            initial_condition, 
            t_eval=time_points, 
            method="RK45", 
            args=(parameters,)
        )
        return sol.y

    def sse(self, parameters, time_points):
        total_error = 0.0
        for i, obs in enumerate(self.observations):
            if obs is None: continue
            ic = self.initial_conditions[i]
            pred = self.solve(parameters, time_points, ic)
            
            # If shapes mismatch, we might need to transpose obs
            if pred.shape != obs.shape:
                if pred.shape == obs.T.shape:
                    obs = obs.T
                else:
                    raise ValueError(f"Shape mismatch: Pred {pred.shape}, Obs {obs.shape}")
            
            total_error += np.sum((pred - obs)**2)
        return total_error

class SIModel(KineticModel):
    def __init__(self):
        initial_conditions = [
            np.array([2 , 0]),
            np.array([10, 0]),
            np.array([2 , 2]),
            np.array([10, 2]),
            np.array([10, 1]),
        ]
        super().__init__(
            name="Synthetic_Isomerisation",
            num_params=5,
            species_names=['A', 'B'],
            initial_conditions=initial_conditions,
            data_path_template="physics_informed_SR/Synthetic_Isomerisation/exp_data/exp_{}.csv"
        )
        self.true_params = np.array([7, 3, 4, 2, 6])

    def ode_system(self, t, state, parameters):
        CA, CB = state
        k_1, k_2, k_3, k_4, k_5 = parameters
        common_term = ((k_1 * CA - k_2 * CB) / (k_3 * CA + k_4 * CB + k_5))
        dAdt = -common_term
        dBdt = common_term
        return np.array([dAdt, dBdt])

class DNOModel(KineticModel):
    def __init__(self):
        # Initial conditions from Decomposition_of_Nitrous_Oxide_no_constraints/case_study.py
        initial_conditions = [
            np.array([5 , 0, 0]),
            np.array([10, 0, 0]),
            np.array([5 , 2, 0]),
            np.array([5 , 0, 3]),
            np.array([0 , 2, 3]),
        ]
        super().__init__(
            name="Decomposition_Nitrous_Oxide",
            num_params=2, # k1, k2
            species_names=['NO', 'N', 'O'],
            initial_conditions=initial_conditions,
            data_path_template="physics_informed_SR/Decomposition_Nitrous_Oxide/exp_data/exp_{}.csv"
        )
        self.true_params = np.array([2, 5])

    def ode_system(self, t, state, parameters):
        # z[0]=NO, z[1]=N, z[2]=O
        NO = state[0]
        k_1, k_2 = parameters
        
        # dNOdt = (-1) * ((k_1 * z[0]**2) / (1 + k_2 * z[0]))
        rate = (k_1 * NO**2) / (1 + k_2 * NO)
        
        dNOdt = -rate
        dNdt = rate
        dOdt = 0.5 * rate
        
        return np.array([dNOdt, dNdt, dOdt])

class HTModel(KineticModel):
    def __init__(self):
        # Initial conditions from Hydrodealkylation_of_Toluene_no_constraints/case_study.py
        initial_conditions = [
            np.array([1, 8, 2, 3]),
            np.array([5, 8, 0, 0.5]),
            np.array([5, 3, 0, 0.5]),
            np.array([1, 3, 0, 3]),
            np.array([1, 8, 2, 0.5]),
        ]
        super().__init__(
            name="Hydrodealkylation_of_Toluene",
            num_params=3, # k1, k2, k3
            species_names=['T', 'H', 'B', 'M'],
            initial_conditions=initial_conditions,
            data_path_template="physics_informed_SR/Hydrodealkylation_of_Toluene/exp_data/exp_{}.csv"
        )
        self.true_params = np.array([2, 9, 5])

    def ode_system(self, t, state, parameters):
        # z[0]=T, z[1]=H, z[2]=B, z[3]=M
        T, H, B, M = state
        k_1, k_2, k_3 = parameters
        
        # rate = (k_1 * T * H) / (1 + k_2 * B + k_3 * T)
        denom = (1 + k_2 * B + k_3 * T)
        if denom == 0: denom = 1e-8
        rate = (k_1 * T * H) / denom
        
        dTdt = -rate
        dHdt = -rate
        dBdt = rate
        dMdt = rate
        
        return np.array([dTdt, dHdt, dBdt, dMdt])
