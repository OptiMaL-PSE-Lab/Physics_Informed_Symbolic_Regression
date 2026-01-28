
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

    "ic_6": np.array([10.0000, 2.0000]),
    "ic_7": np.array([10.0000, 1.4993]),
    "ic_8": np.array([10.0000, 2.0000]),
    "ic_9": np.array([2.0000, 0.0012]),
    "ic_10": np.array([10.0000, 0.0000]),
    "ic_11": np.array([3.4244, 0.3987]),
    "ic_12": np.array([5.7857, 0.0040]),
    "ic_13": np.array([10.0000, 2.0000]),
    "ic_14": np.array([10.0000, 0.0000]),
    "ic_15": np.array([5.3351, 0.0000]),
    "ic_16": np.array([7.3894, 0.0002]),
    "ic_17": np.array([2.0000, 2.0000]),
    "ic_18": np.array([6.8957, 0.0002]),
    "ic_19": np.array([2.0000, 0.0090]),
    "ic_20": np.array([10.0000, 0.0000]),
    "ic_21": np.array([2.0000, 0.0012]),
    "ic_22": np.array([6.8976, 0.0000]),
    "ic_23": np.array([10.0000, 0.3191]),
    "ic_24": np.array([2.0000, 2.0000]),
    "ic_25": np.array([5.0682, 0.0000]),
    "ic_26": np.array([4.9962, 0.0000]),
    "ic_27": np.array([2.0000, 2.0000]),
    "ic_28": np.array([2.0000, 0.0010]),
    "ic_29": np.array([10.0000, 0.0006]),
    "ic_30": np.array([10.0000, 2.0000]),
    "ic_31": np.array([10.0000, 0.0002]),
    "ic_32": np.array([5.7780, 0.0000]),
    "ic_33": np.array([7.7495, 0.0011]),
    "ic_34": np.array([10.0000, 2.0000]),
    "ic_35": np.array([5.6438, 0.0000]),
    "ic_36": np.array([2.0000, 0.0147]),
    "ic_37": np.array([2.0000, 0.0000]),
    "ic_38": np.array([9.9950, 2.0000]),
    "ic_39": np.array([9.4484, 2.0000]),
    "ic_40": np.array([10.0000, 2.0000]),
    "ic_41": np.array([10.0000, 2.0000]),
    "ic_42": np.array([5.3069, 0.0000]),
    "ic_43": np.array([10.0000, 2.0000]),
    "ic_44": np.array([4.6976, 2.0000]),
    "ic_45": np.array([9.2728, 2.0000]),
    "ic_46": np.array([10.0000, 2.0000]),
    "ic_47": np.array([6.4913, 1.6665]),
    "ic_48": np.array([2.0000, 0.0000]),
    "ic_49": np.array([7.5750, 2.0000]),
    "ic_50": np.array([10.0000, 2.0000]),
    "ic_51": np.array([2.0000, 0.0000]),
    "ic_52": np.array([2.0000, 2.0000]),
    "ic_53": np.array([5.8363, 1.6215]),
    "ic_54": np.array([2.0000, 0.0000]),
    "ic_55": np.array([3.4593, 2.0000]),
    "ic_56": np.array([2.0000, 0.0010]),}
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
MAX_ITERATIONS = 50
IC_LOWER_BOUND = np.array([2, 0])   # [A_min, B_min]
IC_UPPER_BOUND = np.array([10, 2])  # [A_max, B_max]
MBDOE_MULTISTART = 1  # Number of random restarts for optimization
