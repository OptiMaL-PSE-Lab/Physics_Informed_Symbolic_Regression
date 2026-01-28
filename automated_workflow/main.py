"""
Main Orchestrator for the Physics-Informed ADoK Workflow with MBDoE Loop.

This script runs the full workflow iteratively:
1. Generate data
2. SR on concentration profiles
3. Process concentration and calculate derivatives
4. SR on rate equations
5. Evaluate models
6. Check if correct model found
7. If not, use MBDoE to propose next experiment and repeat
"""

import subprocess
import os
import sys
import importlib.util
import importlib
import numpy as np

# Ensure current directory is in path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


def load_module(name, path):
    """Dynamically load a Python module."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_script(script_name, args=[]):
    """Run a Python or Julia script."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, script_name)
    
    if script_name.endswith('.jl'):
        julia_path = os.path.expanduser("~/.juliaup/bin/julia")
        if not os.path.exists(julia_path):
            julia_path = "julia"
        cmd = [julia_path, script_path, *args]
    else:
        cmd = [sys.executable, script_path, *args]
        
    print(f"\n[ORCHESTRATOR] Running {script_name}...")
    try:
        subprocess.run(cmd, check=True, cwd=base_dir)
    except subprocess.CalledProcessError as e:
        print(f"\n[ORCHESTRATOR] Error running {script_name}: {e}")
        return False
    return True


def run_single_iteration():
    """Run one complete iteration of the SR workflow."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Reload config to get updated NUM_EXP
    importlib.reload(config)
    
    # Stage 1: Generate Data
    if not run_script("1_generate_data.py"):
        return None
    
    # Stage 2: SR on Concentration
    if not run_script("2_sr_concentration.jl", [str(config.NUM_EXP)]):
        return None
    
    # Stage 3: Process Concentration & Calc Derivatives
    if not run_script("3_process_concentration.py"):
        return None
    
    # Stage 4: SR on Rates
    if not run_script("4_sr_rates.jl", [str(config.NUM_EXP)]):
        return None
    
    # Stage 5: Evaluate & Select Best Model (run directly to get return value)
    eval_module = load_module('evaluate', os.path.join(base_dir, '5_evaluate_models.py'))
    result = eval_module.evaluate_models()
    
    return result


def add_new_experiment(new_ic):
    """Add a new initial condition to config and regenerate data."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config.py')
    
    # Read current config
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Find the current number of ICs
    current_num = config.NUM_EXP
    new_num = current_num + 1
    
    # Add new IC to INITIAL_CONDITIONS dict
    new_ic_line = f'    "ic_{new_num}": np.array([{new_ic[0]:.4f}, {new_ic[1]:.4f}]),'
    
    # Insert before the closing brace of INITIAL_CONDITIONS
    # Find the pattern and insert
    import re
    pattern = r'(INITIAL_CONDITIONS = \{[^}]+)'
    match = re.search(pattern, content)
    if match:
        insert_pos = match.end()
        new_content = content[:insert_pos] + '\n' + new_ic_line + content[insert_pos:]
        
        with open(config_path, 'w') as f:
            f.write(new_content)
        
        print(f"\n[ORCHESTRATOR] Added new experiment IC_{new_num}: A0={new_ic[0]:.4f}, B0={new_ic[1]:.4f}")
        return True
    
    return False


def main_with_mbdoe():
    """Main function with MBDoE iterative loop."""
    print("=" * 60)
    print("   Physics-Informed ADoK Workflow with MBDoE")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load helper modules
    model_checker = load_module('model_checker', os.path.join(base_dir, 'model_checker.py'))
    mbdoe = load_module('mbdoe', os.path.join(base_dir, '6_mbdoe.py'))
    
    for iteration in range(1, config.MAX_ITERATIONS + 1):
        # Reload config at start of each iteration to get updated NUM_EXP
        importlib.reload(config)
        
        print(f"\n{'=' * 60}")
        print(f"   ITERATION {iteration} / {config.MAX_ITERATIONS}")
        print(f"   Number of experiments: {config.NUM_EXP}")
        print(f"{'=' * 60}")
        
        # Run SR workflow
        result = run_single_iteration()
        
        if result is None:
            print("\n[ORCHESTRATOR] Workflow failed. Stopping.")
            break
        
        best_model = result['best_model']
        second_best_model = result['second_best_model']
        
        if best_model is None:
            print("\n[ORCHESTRATOR] No model found. Stopping.")
            break
        
        # Check if correct model found
        print("\n[ORCHESTRATOR] Checking model correctness...")
        is_correct, similarity = model_checker.check_discovered_model(best_model)
        
        if is_correct:
            print("\n" + "=" * 60)
            print("   SUCCESS! Correct kinetic model discovered!")
            print(f"   Iterations needed: {iteration}")
            print(f"   Similarity score: {similarity:.4f}")
            print("=" * 60)
            
            # Save success result
            with open(os.path.join(base_dir, "mbdoe_result.txt"), "w") as f:
                f.write(f"SUCCESS\n")
                f.write(f"Iterations: {iteration}\n")
                f.write(f"Experiments: {config.NUM_EXP}\n")
                f.write(f"Best Model: {best_model}\n")
                f.write(f"Similarity: {similarity}\n")
            
            return True
        
        # Model not correct - use MBDoE to find next experiment
        if iteration < config.MAX_ITERATIONS:
            print("\n[ORCHESTRATOR] Model not correct. Running MBDoE...")
            
            if second_best_model and second_best_model != best_model:
                new_ic = mbdoe.find_optimal_experiment(
                    best_model, 
                    second_best_model,
                    iteration=iteration,
                    best_aic=result.get('best_aic'),
                    second_best_aic=result.get('second_best_aic')
                )
            else:
                # Fallback: random experiment in bounds
                print("  Using random IC (no competing models)")
                new_ic = np.random.uniform(config.IC_LOWER_BOUND, config.IC_UPPER_BOUND)
            
            # Add new experiment
            add_new_experiment(new_ic)
            
            # Reload config
            importlib.reload(config)
        
    print("\n" + "=" * 60)
    print("   Max iterations reached without finding correct model.")
    print("=" * 60)
    
    with open(os.path.join(base_dir, "mbdoe_result.txt"), "w") as f:
        f.write(f"INCOMPLETE\n")
        f.write(f"Iterations: {config.MAX_ITERATIONS}\n")
        f.write(f"Experiments: {config.NUM_EXP}\n")
    
    return False


def main():
    """Original main function - single iteration."""
    print("========================================")
    print("   Starting Physics-Informed ADoK Workflow")
    print("========================================")
    
    run_single_iteration()
    
    print("\n========================================")
    print("   Workflow Completed Successfully!")
    print("========================================")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Physics-Informed ADoK Workflow')
    parser.add_argument('--mbdoe', action='store_true', help='Run with MBDoE iterative loop')
    args = parser.parse_args()
    
    if args.mbdoe:
        main_with_mbdoe()
    else:
        main()
