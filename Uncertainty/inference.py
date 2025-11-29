import numpy as np
import scipy.optimize as opt
import numdifftools as nd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import os

class BayesianInference:
    def __init__(self, model, prior_means, prior_covs):
        self.model = model
        self.prior_means = prior_means
        self.prior_covs = prior_covs
        self.time_points = np.linspace(0.0, 10.0, 15) # Standard time points

    def log_prior(self, parameters):
        return multivariate_normal.logpdf(parameters, mean=self.prior_means, cov=self.prior_covs)

    def log_likelihood(self, parameters):
        sse = self.model.sse(parameters, self.time_points)
        return -sse / 2.0

    def neg_log_posterior(self, parameters):
        if np.any(parameters < 0): return np.inf # Enforce positivity
        return -(self.log_prior(parameters) + self.log_likelihood(parameters))

    def run(self, initial_parameters, num_samples):
        raise NotImplementedError("Subclasses must implement run")

    def plot_posterior(self, samples, output_dir="graphs"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        num_params = samples.shape[1]
        colors = ['royalblue', 'salmon', 'limegreen', 'darkviolet', 'goldenrod']
        
        cols = 3
        rows = (num_params + cols - 1) // cols
        
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6*cols, 6*rows), constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        
        for i in range(num_params):
            ax = axes[i]
            ax.hist(samples[:, i], bins=50, density=True, alpha=0.7, color=colors[i % len(colors)])
            ax.set_title(f'Parameter {i + 1}', fontsize=18)
            ax.set_xlabel('Value', fontsize=14)
            if i % cols == 0:
                ax.set_ylabel('Density', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.5)
            
        for i in range(num_params, len(axes)):
            axes[i].set_visible(False)
            
        filename = f"{self.model.name}_posterior_{self.__class__.__name__}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        print(f"Posterior plot saved to {os.path.join(output_dir, filename)}")

    def plot_uncertainty(self, samples, output_dir="graphs"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Use a subset of samples for prediction to save time if too many
        n_post = min(len(samples), 1000)
        indices = np.random.choice(len(samples), n_post, replace=False)
        posterior_samples = samples[indices]
        
        color_1 = ['salmon', 'royalblue', 'darkviolet', 'limegreen']
        
        for i in range(len(self.model.initial_conditions)):
            fig, ax = plt.subplots()
            ic = self.model.initial_conditions[i]
            obs = self.model.observations[i]
            
            pred_samples = np.zeros((n_post, self.model.num_species, len(self.time_points)))
            
            for j, params in enumerate(posterior_samples):
                sol = self.model.solve(params, self.time_points, ic)
                pred_samples[j, :, :] = sol
            
            for k in range(self.model.num_species):
                species_predictions = pred_samples[:, k, :]
                ave_mean = np.mean(species_predictions, axis=0)
                ave_std = np.std(species_predictions, axis=0)
                upper_bound = ave_mean + 3 * ave_std
                lower_bound = ave_mean - 3 * ave_std
                
                ax.plot(self.time_points, ave_mean, '-', color=color_1[k % len(color_1)], linewidth=2, label=f"{self.model.species_names[k]} Pred")
                ax.fill_between(self.time_points, lower_bound, upper_bound, color=color_1[k % len(color_1)], alpha=0.3)
                
                if obs is not None:
                    # Assuming obs is (species, time)
                    ax.plot(self.time_points, obs[k, :], '.', color=color_1[k % len(color_1)], markersize=10, label=f"{self.model.species_names[k]} Obs")

            ax.set_ylabel("Concentration", fontsize=14)
            ax.set_xlabel("Time", fontsize=14)
            ax.legend(loc='upper right')
            ax.set_title(f"Experiment {i+1}")
            
            filename = f"{self.model.name}_uncertainty_exp_{i+1}_{self.__class__.__name__}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()
        print(f"Uncertainty plots saved to {output_dir}")

    def plot_true_comparison(self, samples, output_dir="graphs"):
        """
        Plots uncertainty bands against the TRUE data points (from ground truth model)
        with error bars representing the added noise (std=0.2).
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Use a subset of samples for prediction
        n_post = min(len(samples), 1000)
        indices = np.random.choice(len(samples), n_post, replace=False)
        posterior_samples = samples[indices]
        
        color_1 = ['salmon', 'royalblue', 'darkviolet', 'limegreen']
        noise_std = 0.2
        
        for i in range(len(self.model.initial_conditions)):
            fig, ax = plt.subplots()
            ic = self.model.initial_conditions[i]
            
            # 1. Generate True Data Points
            if hasattr(self.model, 'true_params'):
                true_sol = self.model.solve(self.model.true_params, self.time_points, ic)
            else:
                print(f"Warning: No true_params defined for {self.model.name}. Skipping true comparison plot.")
                plt.close()
                continue

            # 2. Generate Uncertainty Bands
            pred_samples = np.zeros((n_post, self.model.num_species, len(self.time_points)))
            for j, params in enumerate(posterior_samples):
                sol = self.model.solve(params, self.time_points, ic)
                pred_samples[j, :, :] = sol
            
            for k in range(self.model.num_species):
                # Plot Uncertainty Bands
                species_predictions = pred_samples[:, k, :]
                ave_mean = np.mean(species_predictions, axis=0)
                ave_std = np.std(species_predictions, axis=0)
                upper_bound = ave_mean + 3 * ave_std
                lower_bound = ave_mean - 3 * ave_std
                
                ax.plot(self.time_points, ave_mean, '-', color=color_1[k % len(color_1)], linewidth=2, label=f"{self.model.species_names[k]} Pred")
                ax.fill_between(self.time_points, lower_bound, upper_bound, color=color_1[k % len(color_1)], alpha=0.3)
                
                # Plot True Data Points with Error Bars
                # True data points are the ground truth model output at the time points
                true_points = true_sol[k, :]
                ax.errorbar(self.time_points, true_points, yerr=noise_std, fmt='o', color=color_1[k % len(color_1)], 
                            markersize=5, capsize=3, label=f"{self.model.species_names[k]} True")

            ax.set_ylabel("Concentration", fontsize=14)
            ax.set_xlabel("Time", fontsize=14)
            # Deduplicate legend labels
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            
            ax.set_title(f"Experiment {i+1} (True Comparison)")
            
            filename = f"{self.model.name}_true_comparison_exp_{i+1}_{self.__class__.__name__}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()
        print(f"True comparison plots saved to {output_dir}")


class LaplaceApproximation(BayesianInference):
    def run(self, initial_parameters, num_samples):
        print("Running Laplace Approximation...")
        bounds = [(0, None) for _ in initial_parameters]
        
        # 1. MAP Estimate
        res = opt.minimize(self.neg_log_posterior, initial_parameters, method='L-BFGS-B', bounds=bounds)
        mu_map = res.x
        print(f"MAP Estimate: {mu_map}")
        
        # 2. Covariance
        hessian_func = nd.Hessian(self.neg_log_posterior)
        H = hessian_func(mu_map)
        try:
            covariance = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            print("Warning: Hessian not invertible. Using identity covariance.")
            covariance = np.eye(len(initial_parameters))
            
        # 3. Sampling
        samples = np.random.multivariate_normal(mu_map, covariance, size=num_samples)
        samples = np.maximum(samples, 0) # Enforce positivity
        return samples

class MetropolisHastings(BayesianInference):
    def __init__(self, model, prior_means, prior_covs, proposal_std):
        super().__init__(model, prior_means, prior_covs)
        self.proposal_std = proposal_std

    def run(self, initial_parameters, num_samples):
        print("Running Metropolis-Hastings...")
        current_params = np.array(initial_parameters)
        samples = []
        
        current_log_post = -self.neg_log_posterior(current_params)
        
        for i in range(num_samples):
            if i % 1000 == 0: print(f"Iteration {i}/{num_samples}")
            
            proposed_params = np.random.normal(current_params, self.proposal_std)
            if np.any(proposed_params < 0):
                # Reject negative parameters immediately
                samples.append(current_params)
                continue
                
            proposed_log_post = -self.neg_log_posterior(proposed_params)
            
            # Acceptance ratio
            # log(alpha) = log(p(prop)/p(curr)) = log_post_prop - log_post_curr
            log_alpha = proposed_log_post - current_log_post
            
            if np.log(np.random.rand()) < log_alpha:
                current_params = proposed_params
                current_log_post = proposed_log_post
                
            samples.append(current_params)
            
        return np.array(samples)

class HamiltonianMonteCarlo(BayesianInference):
    def __init__(self, model, prior_means, prior_covs, step_size=0.20, num_steps=25):
        super().__init__(model, prior_means, prior_covs)
        self.step_size = step_size
        self.num_steps = num_steps

    def numerical_gradient(self, log_p_fn, theta, epsilon=1e-5):
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[i] += epsilon
            theta_minus[i] -= epsilon
            grad[i] = (log_p_fn(theta_plus) - log_p_fn(theta_minus)) / (2 * epsilon)
        return grad

    def log_posterior(self, parameters):
        # We need log_posterior, but self.neg_log_posterior returns -log_posterior
        # And it handles negative parameters by returning inf.
        # We need to handle that.
        nlp = self.neg_log_posterior(parameters)
        if nlp == np.inf:
            return -1e10 # Return a very small number for invalid parameters
        return -nlp

    def leapfrog(self, theta, momentum, log_posterior_fn, step_size, num_steps, epsilon=1e-5):
        theta = np.array(theta, dtype=float)
        momentum = np.array(momentum, dtype=float)
        
        # Half-step for momentum at the beginning
        grad_theta = self.numerical_gradient(log_posterior_fn, theta, epsilon)
        momentum = momentum + (step_size / 2.0) * grad_theta
        
        # Full steps for position and momentum
        for _ in range(num_steps):
            theta = theta + step_size * momentum
            theta = np.maximum(theta, 1e-6)  # Ensure non-negative
            
            grad_theta = self.numerical_gradient(log_posterior_fn, theta, epsilon)
            momentum = momentum + step_size * grad_theta
        
        # Half-step for momentum at the end
        grad_theta = self.numerical_gradient(log_posterior_fn, theta, epsilon)
        momentum = momentum - (step_size / 2.0) * grad_theta
        
        return theta, momentum

    def run(self, initial_parameters, num_samples):
        print("Running Hamiltonian Monte Carlo...")
        current_parameters = np.array(initial_parameters, dtype=float)
        current_parameters = np.maximum(current_parameters, 1e-6)
        
        samples = []
        num_accepts = 0
        
        for i in range(num_samples):
            if i % 100 == 0:
                print(f"Iteration: {i}, Acceptance rate: {num_accepts / max(i, 1):.3f}")
            
            # Sample random momentum
            momentum_current = np.random.normal(0, 1, size=current_parameters.shape)
            
            # Compute current Hamiltonian
            log_p_current = self.log_posterior(current_parameters)
            hamiltonian_current = -log_p_current + 0.5 * np.sum(momentum_current**2)
            
            # Propose new state using leapfrog
            proposed_parameters, proposed_momentum = self.leapfrog(
                current_parameters, 
                momentum_current, 
                self.log_posterior, 
                self.step_size, 
                self.num_steps
            )
            
            proposed_parameters = np.maximum(proposed_parameters, 1e-6)
            
            # Compute proposed Hamiltonian
            log_p_proposed = self.log_posterior(proposed_parameters)
            hamiltonian_proposed = -log_p_proposed + 0.5 * np.sum(proposed_momentum**2)
            
            # Metropolis acceptance
            log_acceptance_ratio = hamiltonian_current - hamiltonian_proposed
            
            if np.log(np.random.rand()) < log_acceptance_ratio:
                current_parameters = proposed_parameters
                num_accepts += 1
            
            samples.append(current_parameters.copy())
        
        acceptance_rate = num_accepts / num_samples
        print(f"Final acceptance rate: {acceptance_rate:.3f}")
        return np.array(samples)
