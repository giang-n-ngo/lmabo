import torch
import numpy as np
import math
from botorch.optim import optimize_acqf
from botorch.acquisition import AcquisitionFunction
from torch.distributions import StudentT, Uniform

from baselines.bo_helpers import (
    calculate_cumulative_regret,
    fit_gp
)
from baselines.gp_hedge import _get_nominated_point_and_posterior_mean

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double

class RandomFourierFeatureGP:
    """
    Approximates a GP with a Matérn(5/2) kernel using a Bayesian linear model
    with Random Fourier Features, as described in Sec 3.2 and 4.1 of the paper.
    """
    def __init__(self, model, num_features: int = 1000):
        self.model = model
        self.train_x = model.train_inputs[0]
        self.train_y = model.train_targets
        self.m = num_features
        
        # Extract GP hyperparameters
        self.outputscale = model.covar_module.outputscale.item()
        self.lengthscale = model.covar_module.base_kernel.lengthscale.detach()
        self.noise_var = model.likelihood.noise.item()
        self.mean_const = model.mean_module.constant.item()
        
        d = self.train_x.shape[-1]
        
        # Sample weights W from the spectral density of the Matérn(5/2) kernel,
        # which is a Student's-t distribution. [cite: 212]
        t_dist = StudentT(df=5)
        w_dist_inv_scale = torch.diag(self.lengthscale.squeeze() / math.sqrt(5))
        self.W = t_dist.sample(torch.Size([self.m, d])).to(DEVICE, DTYPE) @ w_dist_inv_scale
        
        # Sample biases b from Uniform(0, 2*pi)
        self.b = Uniform(0, 2 * math.pi).sample(torch.Size([self.m, 1])).to(DEVICE, DTYPE)

        # Pre-compute matrices for weight posterior calculation
        self._calculate_weight_posterior()

    def _phi(self, X: torch.Tensor) -> torch.Tensor:
        """Computes the random Fourier features for a given input X."""
        # Phi(X) = sqrt(2*alpha/m) * cos(W*X^T + b)
        proj = self.W @ X.transpose(-1, -2) + self.b
        return math.sqrt(2 * self.outputscale / self.m) * torch.cos(proj).transpose(-1, -2)

    def _calculate_weight_posterior(self):
        """
        Calculates the posterior distribution of the weights theta,
        p(theta|D) ~ N(mu_theta, Sigma_theta), as per the paper. 
        """
        Phi = self._phi(self.train_x) # Shape (n_points x m_features)
        
        # A = Phi^T * Phi + sigma^2 * I
        A = Phi.t() @ Phi + self.noise_var * torch.eye(self.m, device=DEVICE, dtype=DTYPE)
        A_inv = torch.inverse(A)
        
        # Posterior mean: mu_theta = A^-1 * Phi^T * (y - mu_0)
        self.mu_theta = A_inv @ Phi.t() @ (self.train_y - self.mean_const)
        
        # Posterior covariance: Sigma_theta = sigma^2 * A^-1
        self.Sigma_theta = self.noise_var * A_inv

    def sample_function(self) -> callable:
        """
        Draws one sample function from the approximate GP posterior.
        The function is f(x) = phi(x)^T * theta + mu_0.
        """
        # Sample weights from the posterior: theta ~ N(mu_theta, Sigma_theta)
        theta_sample = torch.distributions.MultivariateNormal(
            self.mu_theta, self.Sigma_theta
        ).sample()
        
        # Return a callable function
        def f_sample(X):
            return self._phi(X) @ theta_sample + self.mean_const
            
        return f_sample

class SampledFunctionAcquisition(AcquisitionFunction):
    """Simple wrapper to make an arbitrary callable optimizable by botorch."""
    def __init__(self, model, fn: callable):
        super().__init__(model)
        self.fn = fn
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # optimize_acqf maximizes, so we negate the function to find the minimum
        return -self.fn(X.squeeze(-2)).unsqueeze(-1)

class EntropySearchPortfolio:
    """
    Implements the Entropy Search Portfolio (ESP) meta-acquisition function.

    This class selects from a discrete set of candidate points proposed by
    other acquisition functions. The selection criterion is to choose the
    candidate that is expected to maximally reduce the entropy of the posterior
    distribution over the global minimizer's location, x*.
    
    The implementation follows Algorithm 2 from Shahriari et al. (2015) .
    """
    def __init__(
        self,
        model,
        candidates: torch.Tensor,
        num_representer_points: int = 50,
        num_hallucinated_observations: int = 10,
        num_fantasized_samples: int = 100,
    ):
        """
        Args:
            model: The fitted SingleTaskGP model.
            candidates: A (K x d) tensor of candidate points from base AFs.
            num_representer_points (G): Number of points to discretize the minimizer's posterior.
            num_hallucinated_observations (N): Number of fantasy observations to average over.
            num_fantasized_samples (S): Number of posterior samples to estimate probabilities.
        """
        self.model = model
        self.candidates = candidates
        self.G = num_representer_points
        self.N = num_hallucinated_observations
        self.S = num_fantasized_samples

    def _get_representer_points(self) -> torch.Tensor:
        """
        Generates representer points {z_i} by sampling from p(x*|D)
        using the Random Fourier Features method from the paper. 
        """
        rff_approximator = RandomFourierFeatureGP(self.model, num_features=self.num_rff_features)
        representers = []
        
        for _ in range(self.G):
            # 1. Sample a function from the posterior
            f_sample = rff_approximator.sample_function()
            
            # 2. Wrap it in a BoTorch-compatible acquisition function
            acqf_to_optimize = SampledFunctionAcquisition(self.model, f_sample)
            
            # 3. Optimize it to find the minimizer (by maximizing its negative)
            z_i, _ = optimize_acqf(
                acq_function=acqf_to_optimize,
                bounds=self.bounds,
                q=1,
                num_restarts=10,
                raw_samples=512
            )
            representers.append(z_i)
        
        return torch.cat(representers, dim=0)

    def evaluate(self, bounds: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the ESP utility for each candidate and returns the best one.

        Returns:
            The candidate point (1 x d tensor) that minimizes the expected future entropy.
        """
        
        # Algorithm 2, Line 1: Generate representer points {z_i} 
        representer_points = self._get_representer_points(bounds)
        
        candidate_utilities = []

        # Algorithm 2, Line 2: Loop over each candidate x_k 
        for i in range(self.candidates.shape[0]):
            candidate = self.candidates[i].unsqueeze(0) # Shape (1 x d)
            
            # Get the model's predictive posterior at the candidate point
            posterior = self.model.posterior(candidate)
            
            entropies_for_candidate = []
            
            # Algorithm 2, Line 3: Loop N times for hallucinations y_k^(n) 
            with torch.no_grad():
                hallucinated_outcomes = posterior.sample(torch.Size([self.N])).squeeze(-1)

            for n in range(self.N):
                y_k_n = hallucinated_outcomes[n]
                
                # Algorithm 2, Line 5: Fantasize a new model conditioned on the hallucinated data 
                fantasy_model = self.model.condition_on_observations(
                    X=candidate, Y=y_k_n
                )
                
                # Evaluate the fantasy posterior at the representer points
                fantasy_posterior_at_z = fantasy_model.posterior(representer_points)
                
                # Algorithm 2, Line 6: Draw S samples from the fantasy posterior 
                f_kn_s = fantasy_posterior_at_z.sample(torch.Size([self.S])) # Shape (S x 1 x G)
                f_kn_s = f_kn_s.squeeze(1) # Shape (S x G)
                
                # Algorithm 2, Line 7: Find the minimizer for each sample 
                # Note: The objective is minimization
                minimizers = torch.argmin(f_kn_s, dim=1) # Shape (S)
                
                # Compute the discrete probability distribution p_ikn [cite: 130]
                counts = torch.bincount(minimizers, minlength=self.G).float()
                p_ikn = counts / self.S
                
                # Calculate the entropy of this distribution
                # The paper's utility u_k is sum(p*log(p)), which is negative entropy.
                # Maximizing u_k is equivalent to minimizing entropy.
                # We add a small epsilon for numerical stability.
                entropy = -torch.sum(p_ikn * torch.log2(p_ikn + 1e-12))
                entropies_for_candidate.append(entropy)
            
            # Algorithm 2, Line 9: Average the entropies over all hallucinations 
            avg_entropy = torch.stack(entropies_for_candidate).mean()
            candidate_utilities.append(avg_entropy)

        # Algorithm 2, Line 11: Return x_k that minimizes expected future entropy 
        best_candidate_idx = torch.argmin(torch.stack(candidate_utilities))
        
        return self.candidates[best_candidate_idx].unsqueeze(0), best_candidate_idx

def esp_full_loop(
    objective_func,
    portfolio_acq_types, # List of strings, e.g., ["EI", "UCB", "PI"]
    X_init,
    Y_init,
    bounds,
    num_iterations,    
):
    # Initial data
    train_X = X_init.clone()
    train_Y = Y_init.clone()

    best_values = [train_Y.min().item()] # Simple regret values
    acq_type_list = []

    # Optimization loop
    for t in range(num_iterations):
        # Fit the GP model
        gp = fit_gp(train_X, train_Y)

        # --- Portfolio Step ---
        # Algorithm 1, Line 2: Collect candidates from base experts [cite: 69]
        # The paper uses EI, PI, and Thompson Sampling [cite: 196]
        # Here we use all acquisition functions in portfolio_acq_types
        candidates = []
        for acq_type in portfolio_acq_types:
            candidate, _ = _get_nominated_point_and_posterior_mean(
                train_X=train_X,
                train_Y=train_Y,
                acq_type=acq_type,
                bounds=bounds,
            )
            candidates.append(candidate)

        candidates = torch.cat(candidates, dim=0)

        # --- ESP Meta-Policy Step ---
        # Algorithm 1, Line 3: Select the best candidate using ESP [cite: 69]
        esp_meta_policy = EntropySearchPortfolio(
            model=gp,
            candidates=candidates,
            num_representer_points=50,
            num_hallucinated_observations=10,
            num_fantasized_samples=100
        )
        
        next_point, idx = esp_meta_policy.evaluate(bounds=bounds)
        acq_type_list.append(portfolio_acq_types[idx])

        # --- Evaluation Step ---
        # 4. Sample the objective function at the selected point
        new_Y_val = objective_func(next_point).unsqueeze(-1)

        # 5. Augment the data
        train_X = torch.cat([train_X, next_point])
        train_Y = torch.cat([train_Y, new_Y_val], dim=0)

        best_val = train_Y.min().item()
        best_values.append(best_val)
        print(f"Iteration {t+1}: Best value found = {best_val:.4f}")

    return (
        np.array(best_values) - objective_func._optimal_value, # simple regret
        calculate_cumulative_regret(
            train_Y.detach().cpu().numpy(),
            objective_func._optimal_value
        ), # cumulative regret
        np.array(train_X.detach().cpu().numpy()),
        np.array(train_Y.detach().cpu().numpy()).flatten(),
        acq_type_list
    )