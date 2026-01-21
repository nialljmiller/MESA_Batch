#!/usr/bin/env python
"""
Custom Likelihood Functions

This example demonstrates how to implement custom likelihood functions
for specialized observational constraints.
"""

import numpy as np
from mesa_infer import MESAInfer, InferConfig
from mesa_infer.likelihood import BaseLikelihood, ObservationalData, ModelPrediction

# -----------------------------------------------------------------------------
# 1. Custom Likelihood Class
# -----------------------------------------------------------------------------

class AsteroseismicLikelihood(BaseLikelihood):
    """
    Likelihood based on asteroseismic parameters (nu_max, delta_nu).
    
    Uses scaling relations to compare model predictions with observations.
    """
    
    # Solar reference values
    NU_MAX_SUN = 3090.0      # muHz
    DELTA_NU_SUN = 135.1     # muHz
    TEFF_SUN = 5777.0        # K
    
    def __init__(
        self,
        observations: ObservationalData,
        nu_max_weight: float = 1.0,
        delta_nu_weight: float = 1.0,
    ):
        super().__init__(observations)
        self.nu_max_weight = nu_max_weight
        self.delta_nu_weight = delta_nu_weight
        
        # Validate required observations
        if not hasattr(observations, 'nu_max') or observations.nu_max is None:
            raise ValueError("Observations must include nu_max")
        if not hasattr(observations, 'delta_nu') or observations.delta_nu is None:
            raise ValueError("Observations must include delta_nu")
    
    def compute_log_likelihood(self, model: ModelPrediction) -> float:
        """Compute log-likelihood from asteroseismic scaling relations."""
        
        if not model.success:
            return -np.inf
        
        # Get model stellar parameters
        mass = model.stellar_params.get('mass', 1.0)      # M_sun
        radius = model.stellar_params.get('radius', 1.0)  # R_sun
        teff = model.stellar_params.get('teff', 5777.0)   # K
        
        # Compute predicted nu_max and delta_nu from scaling relations
        # nu_max ~ g / sqrt(Teff) ~ M / R^2 / sqrt(Teff)
        nu_max_pred = self.NU_MAX_SUN * (mass / radius**2) * (self.TEFF_SUN / teff)**0.5
        
        # delta_nu ~ sqrt(mean density) ~ sqrt(M / R^3)
        delta_nu_pred = self.DELTA_NU_SUN * np.sqrt(mass / radius**3)
        
        # Compute chi-square
        chi2 = 0.0
        
        # nu_max contribution
        obs_nu_max = self.observations.nu_max
        obs_nu_max_err = self.observations.nu_max_err
        chi2 += self.nu_max_weight * ((obs_nu_max - nu_max_pred) / obs_nu_max_err)**2
        
        # delta_nu contribution
        obs_delta_nu = self.observations.delta_nu
        obs_delta_nu_err = self.observations.delta_nu_err
        chi2 += self.delta_nu_weight * ((obs_delta_nu - delta_nu_pred) / obs_delta_nu_err)**2
        
        return -0.5 * chi2


# -----------------------------------------------------------------------------
# 2. Custom Likelihood with Priors
# -----------------------------------------------------------------------------

class LikelihoodWithPriors(BaseLikelihood):
    """
    Combine observational likelihood with parameter priors.
    """
    
    def __init__(
        self,
        observations: ObservationalData,
        base_likelihood: BaseLikelihood,
        priors: dict = None,
    ):
        super().__init__(observations)
        self.base_likelihood = base_likelihood
        self.priors = priors or {}
    
    def compute_log_likelihood(self, model: ModelPrediction) -> float:
        """Compute log-likelihood plus log-prior."""
        
        # Base likelihood
        log_like = self.base_likelihood.compute_log_likelihood(model)
        
        if not np.isfinite(log_like):
            return log_like
        
        # Add priors
        log_prior = 0.0
        
        for param_name, prior_spec in self.priors.items():
            param_value = model.parameters.get(param_name)
            if param_value is None:
                continue
            
            prior_type = prior_spec.get('type', 'uniform')
            
            if prior_type == 'gaussian':
                mean = prior_spec['mean']
                std = prior_spec['std']
                log_prior += -0.5 * ((param_value - mean) / std)**2
                
            elif prior_type == 'log_gaussian':
                log_mean = np.log(prior_spec['mean'])
                log_std = prior_spec['log_std']
                log_prior += -0.5 * ((np.log(param_value) - log_mean) / log_std)**2
                
            elif prior_type == 'uniform':
                low = prior_spec.get('low', -np.inf)
                high = prior_spec.get('high', np.inf)
                if param_value < low or param_value > high:
                    return -np.inf
                    
            elif prior_type == 'imf':
                # Salpeter IMF: dN/dM ~ M^-2.35
                alpha = prior_spec.get('alpha', -2.35)
                log_prior += alpha * np.log(param_value)
        
        return log_like + log_prior


# -----------------------------------------------------------------------------
# 3. Using Custom Likelihoods
# -----------------------------------------------------------------------------

# Create observations with asteroseismic data
observations = ObservationalData(
    # Asteroseismic parameters
    nu_max=3050.0,
    nu_max_err=30.0,
    delta_nu=134.5,
    delta_nu_err=0.5,
    # Spectroscopic parameters
    teff=5780.0,
    teff_err=80.0,
    logg=4.44,
    logg_err=0.10,
    feh=0.0,
    feh_err=0.05,
)

# Create asteroseismic likelihood
astero_like = AsteroseismicLikelihood(observations)

# Add priors
priors = {
    'initial_mass': {
        'type': 'imf',
        'alpha': -2.35,
    },
    'mixing_length_alpha': {
        'type': 'gaussian',
        'mean': 1.9,
        'std': 0.3,
    },
    'initial_z': {
        'type': 'log_gaussian',
        'mean': 0.014,
        'log_std': 0.3,
    },
}

# Wrap with priors
likelihood_with_priors = LikelihoodWithPriors(
    observations=observations,
    base_likelihood=astero_like,
    priors=priors,
)

# -----------------------------------------------------------------------------
# 4. Run Inference with Custom Likelihood
# -----------------------------------------------------------------------------

config = InferConfig(
    mesa_config={
        'work_dir': '/path/to/mesa/work',
        'parallel_runs': 4,
    },
    sampler_config={
        'population_size': 64,
        'num_generations': 100,
    },
)

infer = MESAInfer(config)
infer.set_observations(observations)
infer.set_likelihood(likelihood_with_priors)
infer.parse_inlist()

results = infer.run()
results.summary()

# -----------------------------------------------------------------------------
# 5. Combining Multiple Likelihoods
# -----------------------------------------------------------------------------

from mesa_infer.likelihood import SpectroscopicLikelihood

class CombinedLikelihood(BaseLikelihood):
    """Combine multiple likelihood functions."""
    
    def __init__(self, observations, likelihoods: list, weights: list = None):
        super().__init__(observations)
        self.likelihoods = likelihoods
        self.weights = weights or [1.0] * len(likelihoods)
    
    def compute_log_likelihood(self, model: ModelPrediction) -> float:
        total = 0.0
        for likelihood, weight in zip(self.likelihoods, self.weights):
            log_like = likelihood.compute_log_likelihood(model)
            if not np.isfinite(log_like):
                return -np.inf
            total += weight * log_like
        return total


# Usage
spec_like = SpectroscopicLikelihood(observations)
combined = CombinedLikelihood(
    observations=observations,
    likelihoods=[astero_like, spec_like],
    weights=[1.0, 0.5],  # Weight asteroseismology higher
)
