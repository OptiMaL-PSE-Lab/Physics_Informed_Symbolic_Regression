
import os
import numpy as np

# Base directory (relative to where the script is run, expected to be project root or automated_workflow)
# We assume this config is imported by scripts running from the project root or we fix paths relative to this file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Directories
EXP_DATA_DIR = os.path.join(BASE_DIR, "exp_data")
HOF_FILES_DIR = os.path.join(BASE_DIR, "hof_files")
CONST_DATA_DIR = os.path.join(BASE_DIR, "const_data")

# Create directories if they don't exist
os.makedirs(EXP_DATA_DIR, exist_ok=True)
os.makedirs(HOF_FILES_DIR, exist_ok=True)
os.makedirs(CONST_DATA_DIR, exist_ok=True)

# Experiment Parameters
SPECIES = ["A", "B"]
NUM_SPECIES = len(SPECIES)
TIMESTEPS = 15
TOTAL_TIME = 10.0
TIME_SPAN = [0, TOTAL_TIME]
TIME_EVAL = np.linspace(0, TOTAL_TIME, TIMESTEPS).tolist()
STD_NOISE = 0.2
RANDOM_SEED = 1998

INITIAL_CONDITIONS = {
    "ic_1": np.array([2 , 0]),
    "ic_2": np.array([10, 0]),
    "ic_3": np.array([2 , 2]),
    "ic_4": np.array([10, 2]),
    "ic_5": np.array([10, 1]),

    "ic_6": np.array([2.1061, 1.0983]),
    "ic_7": np.array([7.5071, 1.4450]),
    "ic_8": np.array([8.7350, 0.7731]),
    "ic_9": np.array([7.0252, 1.4818]),
    "ic_10": np.array([9.8173, 1.6329]),
    "ic_11": np.array([7.5753, 0.3235]),
    "ic_12": np.array([4.3175, 0.2763]),
    "ic_13": np.array([9.4084, 1.2205]),
    "ic_14": np.array([6.2541, 1.1711]),
    "ic_15": np.array([3.8583, 1.1405]),
    "ic_16": np.array([3.3230, 0.4188]),
    "ic_17": np.array([2.1507, 0.1210]),
    "ic_18": np.array([6.0143, 0.6145]),
    "ic_19": np.array([3.6295, 0.0097]),
    "ic_20": np.array([6.8819, 0.2937]),
    "ic_21": np.array([8.7925, 1.6424]),
    "ic_22": np.array([8.1499, 0.2235]),
    "ic_23": np.array([8.1176, 0.5905]),
    "ic_24": np.array([4.8687, 0.8252]),}
NUM_EXP = len(INITIAL_CONDITIONS)

# Kinetic Constants (True Model)
K_F = 7
K_R = 3 
K_A = 4 
K_B = 2 
K_C = 6 

# True Model Expression (for comparison)
# dA/dt = (-k_f * A + k_r * B) / (k_A * A + k_B * B + k_C)
TRUE_MODEL = "(-7 * A + 3 * B) / (4 * A + 2 * B + 6)"

# MBDoE Configuration
MAX_ITERATIONS = 20
IC_LOWER_BOUND = np.array([2, 0])   # [A_min, B_min]
IC_UPPER_BOUND = np.array([10, 2])  # [A_max, B_max]
MBDOE_MULTISTART = 5  # Number of random restarts for optimization
