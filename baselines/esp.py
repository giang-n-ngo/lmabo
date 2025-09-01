import torch
import numpy as np
from botorch.optim import optimize_acqf
from botorch.generation.sampling import MaxPosteriorSampling
from torch.quasirandom import SobolEngine

from baselines.bo_helpers import (
    calculate_cumulative_regret,
    fit_gp,
    create_minimized_acquisition_function
)
from baselines.gp_hedge import _get_nominated_point_and_posterior_mean

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double

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

    def _get_representer_points(self, bounds: torch.Tensor) -> torch.Tensor:
        """
        Generates representer points {z_i} by sampling from p(x*|D).
        This is approximated by optimizing Thompson Sampling G times.
        """
        representers = []
        ts_sampler = create_minimized_acquisition_function(MaxPosteriorSampling, self.model)
        for _ in range(self.G):
            sobol = SobolEngine(bounds.size(1), scramble=True)
            X_cand = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(20).to(dtype=DTYPE, device=DEVICE)
            representer = ts_sampler(X_cand)
            representers.append(representer)
            del sobol, X_cand
        del ts_sampler
        representers = torch.cat(representers, dim=0)
        return representers.detach()

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
                f_kn_s = f_kn_s.squeeze(-1) # Shape (S x G)
                
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
                gp=gp,
                acq_type=acq_type,
                bounds=bounds,
                best_f=best_values[-1]
            )
            candidates.append(candidate)

        candidates = torch.cat(candidates, dim=0)

        # --- ESP Meta-Policy Step ---
        # Algorithm 1, Line 3: Select the best candidate using ESP [cite: 69]
        esp_meta_policy = EntropySearchPortfolio(
            model=gp,
            candidates=candidates,
            num_representer_points=10,
            num_hallucinated_observations=10,
            num_fantasized_samples=10
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