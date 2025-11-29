import numpy as np
import argparse
from models import SIModel, DNOModel, HTModel
from inference import LaplaceApproximation, MetropolisHastings, HamiltonianMonteCarlo
import time

def run_analysis(model_name, method_name, samples):
    print(f"\n--- Running {model_name} with {method_name} ---")
    
    # Select Model
    if model_name == "SI":
        model = SIModel()
        prior_means = np.array([7.689, 1.896, 4.053, 1.608, 5.943])
        prior_covs = np.diag(np.array([2.0, 2.0, 2.0, 2.0, 2.0]))
        initial_params = np.array([7, 4, 3, 2, 6]) + np.random.normal(0, 1, 5)
        proposal_std = np.array([0.15, 0.15, 0.15, 0.15, 0.15]) * 2
    elif model_name == "DNO":
        model = DNOModel()
        prior_means = np.array([2.0, 5.0]) + np.random.normal(0, 1, 2)
        prior_covs = np.diag(np.array([1.0, 1.0]))
        initial_params = np.array([2.0, 5.0]) + np.random.normal(0, 1, 2)
        proposal_std = np.array([0.15, 0.15]) * 2
    elif model_name == "HT":
        model = HTModel()
        prior_means = np.array([2.0, 9.0, 5.0]) + np.random.normal(0, 0.1, 3)
        prior_covs = np.diag(np.array([1.0, 1.0, 1.0]))
        initial_params = np.array([2.0, 9.0, 5.0]) + np.random.normal(0, 0.1, 3)
        proposal_std = np.array([0.15, 0.15, 0.15]) * 2

    # Select Method
    start_time = time.time()
    
    if method_name == "Laplace":
        inference = LaplaceApproximation(model, prior_means, prior_covs)
        samples = inference.run(initial_params, samples)
    elif method_name == "MH":
        inference = MetropolisHastings(model, prior_means, prior_covs, proposal_std)
        samples = inference.run(initial_params, samples)
    elif method_name == "HMC":
        inference = HamiltonianMonteCarlo(model, prior_means, prior_covs)
        samples = inference.run(initial_params, samples)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time for {model_name} with {method_name}: {elapsed_time:.2f} seconds")

    # Plot Results
    output_dir = f"graphs/{model_name}_{method_name}"
    inference.plot_posterior(samples, output_dir=output_dir)
    inference.plot_uncertainty(samples, output_dir=output_dir)
    inference.plot_true_comparison(samples, output_dir=output_dir)

def main():
    parser = argparse.ArgumentParser(description="Run Uncertainty Analysis")
    parser.add_argument("--model", type=str, default="ALL", choices=["SI", "DNO", "HT", "ALL"], help="Kinetic Model (default: ALL)")
    parser.add_argument("--method", type=str, default="ALL", choices=["Laplace", "MH", "HMC", "ALL"], help="Inference Method (default: ALL)")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    args = parser.parse_args()

    models = ["SI", "DNO", "HT"] if args.model == "ALL" else [args.model]
    methods = ["Laplace", "MH", "HMC"] if args.method == "ALL" else [args.method]

    for model in models:
        for method in methods:
            try:
                run_analysis(model, method, args.samples)
            except Exception as e:
                print(f"Error running {model} with {method}: {e}")

if __name__ == "__main__":
    main()
