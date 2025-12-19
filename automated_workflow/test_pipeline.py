
import unittest
import os
import shutil
import sys
import pandas as pd
import numpy as np
import importlib.util

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# Helper to load module from path
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class TestPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # We will run the pipeline stages manually
        pass

    def test_pipeline_flow(self):
        base_dir = os.path.dirname(__file__)
        
        # 1. Generate Data
        print("TEST: Running Stage 1...")
        gen_data = load_module('gen_data', os.path.join(base_dir, '1_generate_data.py'))
        gen_data.generate_data()
        
        # Verify Stage 1
        for i in range(1, config.NUM_EXP + 1):
            path = os.path.join(config.EXP_DATA_DIR, f"exp_{i}.csv")
            self.assertTrue(os.path.exists(path), f"{path} should exist")

        # 2. Mock Julia Conc SR (Stage 2)
        print("TEST: Mocking Stage 2...")
        # Create dummy HOF files that would have been created by Julia
        # We need valid equations that process_concentration can read (utils.read_equations)
        # Format: CSV with "Equation" column
        for i in range(1, config.NUM_EXP + 1):
            for species in config.SPECIES:
                filename = f"hall_of_fame_{species}{i}.csv"
                with open(os.path.join(config.HOF_FILES_DIR, filename), 'w') as f:
                    f.write("Equation,Complexity,Loss,Score\n")
                    # Provide a simple equation: exp(-0.5*t)
                    # Note: utils expects "x0" -> replaced by "t", "exp" -> "np.exp"
                    # Original code used simple replacement of x0.
                    # Let's use a constant for simplicity or a simple function.
                    f.write("2 * exp( -0.1 * x0 ), 5, 0.1, 0.9\n")
                    f.write("2 * x0, 3, 0.5, 0.5\n")

        # 3. Process Concentration (Stage 3)
        print("TEST: Running Stage 3...")
        proc_conc = load_module('proc_conc', os.path.join(base_dir, '3_process_concentration.py'))
        proc_conc.process_concentration()
        
        # Verify Stage 3
        # Should create const_data files
        self.assertTrue(os.path.exists(os.path.join(config.CONST_DATA_DIR, 'conc_data_for_rate_models.csv')))
        self.assertTrue(os.path.exists(os.path.join(config.CONST_DATA_DIR, 'rate_data_A.csv')))
        self.assertTrue(os.path.exists(os.path.join(config.CONST_DATA_DIR, 'rate_data_B.csv')))

        # 4. Mock Julia Rate SR (Stage 4)
        print("TEST: Mocking Stage 4...")
        # Create dummy HOF files for rates
        # hall_of_fame_rate_A{num_exp}.csv
        for species in config.SPECIES:
            filename = f"hall_of_fame_rate_{species}{config.NUM_EXP}.csv"
            with open(os.path.join(config.HOF_FILES_DIR, filename), 'w') as f:
                f.write("Equation,Complexity,Loss,Score\n")
                # Rate equation involving A and B. A is z0, B is z1 (?)
                # In utils.rate_n_param -> it reads equations.
                # In utils.predicting_rate -> replace "A" with "z[:, 0]", "B" with "z[:, 1]"
                # So we should put "A" and "B" in equation string.
                f.write("A * B, 3, 0.1, 0.9\n")
                f.write("A + B, 3, 0.5, 0.5\n")

        # 5. Evaluate Models (Stage 5)
        print("TEST: Running Stage 5...")
        eval_models = load_module('eval_models', os.path.join(base_dir, '5_evaluate_models.py'))
        eval_models.evaluate_models()
        
        # Verify Stage 5
        self.assertTrue(os.path.exists(os.path.join(config.BASE_DIR, "final_result.txt")))
        with open(os.path.join(config.BASE_DIR, "final_result.txt"), 'r') as f:
            content = f.read()
            self.assertIn("Best Model Index", content)

if __name__ == '__main__':
    unittest.main()
