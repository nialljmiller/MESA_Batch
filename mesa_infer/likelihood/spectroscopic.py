"""
Spectroscopic likelihood for MESA_infer.

Computes likelihood based on spectroscopic stellar parameters.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from .base import BaseLikelihood, ObservationalData, ModelPrediction


class SpectroscopicLikelihood(BaseLikelihood):
    """
    Likelihood based on spectroscopic stellar parameters.
    
    Supports:
    - Effective temperature (Teff)
    - Surface gravity (log g)
    - Metallicity ([Fe/H])
    - Individual element abundances
    
    Example:
        data = ObservationalData(
            teff=5777.0,
            teff_error=50.0,
            logg=4.44,
            logg_error=0.05,
            feh=0.0,
            feh_error=0.05,
        )
        
        likelihood = SpectroscopicLikelihood(data)
        ll = likelihood.compute(model_prediction)
    """
    
    def __init__(
        self,
        data: ObservationalData,
        teff_weight: float = 1.0,
        logg_weight: float = 1.0,
        feh_weight: float = 1.0,
        abundance_weights: Optional[Dict[str, float]] = None,
        use_systematic_errors: bool = True,
        teff_systematic: float = 50.0,
        logg_systematic: float = 0.1,
        feh_systematic: float = 0.05,
    ):
        """
        Initialize spectroscopic likelihood.
        
        Args:
            data: Observational data with spectroscopic parameters
            teff_weight: Weight for Teff constraint
            logg_weight: Weight for log g constraint
            feh_weight: Weight for [Fe/H] constraint
            abundance_weights: Weights for individual abundances
            use_systematic_errors: Add systematic error floor
            teff_systematic: Systematic Teff error (K)
            logg_systematic: Systematic log g error (dex)
            feh_systematic: Systematic [Fe/H] error (dex)
        """
        weights = {
            'teff': teff_weight,
            'logg': logg_weight,
            'feh': feh_weight,
        }
        if abundance_weights:
            weights.update(abundance_weights)
        
        super().__init__(data, weights=weights)
        
        self.teff_weight = teff_weight
        self.logg_weight = logg_weight
        self.feh_weight = feh_weight
        self.abundance_weights = abundance_weights or {}
        self.use_systematic_errors = use_systematic_errors
        self.teff_systematic = teff_systematic
        self.logg_systematic = logg_systematic
        self.feh_systematic = feh_systematic
    
    def compute(self, model: ModelPrediction) -> float:
        """
        Compute spectroscopic log-likelihood.
        
        Args:
            model: Model prediction with stellar parameters
        
        Returns:
            Log-likelihood value
        """
        if not model.success:
            return -np.inf
        
        ll = 0.0
        
        # Teff constraint
        if self.data.teff is not None and model.teff is not None:
            obs = self.data.teff
            mod = model.teff
            err = self.data.teff_error or 100.0
            
            if self.use_systematic_errors:
                err = np.sqrt(err**2 + self.teff_systematic**2)
            
            ll += self.teff_weight * self.log_likelihood_gaussian(
                np.array([obs]), np.array([mod]), np.array([err])
            )
        
        # log g constraint
        if self.data.logg is not None and model.logg is not None:
            obs = self.data.logg
            mod = model.logg
            err = self.data.logg_error or 0.2
            
            if self.use_systematic_errors:
                err = np.sqrt(err**2 + self.logg_systematic**2)
            
            ll += self.logg_weight * self.log_likelihood_gaussian(
                np.array([obs]), np.array([mod]), np.array([err])
            )
        
        # [Fe/H] constraint
        if self.data.feh is not None and model.feh is not None:
            obs = self.data.feh
            mod = model.feh
            err = self.data.feh_error or 0.1
            
            if self.use_systematic_errors:
                err = np.sqrt(err**2 + self.feh_systematic**2)
            
            ll += self.feh_weight * self.log_likelihood_gaussian(
                np.array([obs]), np.array([mod]), np.array([err])
            )
        
        # Individual abundance constraints
        if self.data.abundances and hasattr(model, 'abundances') and model.abundances:
            for element, obs_abund in self.data.abundances.items():
                if element not in model.abundances:
                    continue
                
                mod_abund = model.abundances[element]
                err = self.data.abundance_errors.get(element, 0.1) if self.data.abundance_errors else 0.1
                weight = self.abundance_weights.get(element, 1.0)
                
                ll += weight * self.log_likelihood_gaussian(
                    np.array([obs_abund]),
                    np.array([mod_abund]),
                    np.array([err])
                )
        
        return ll
    
    def chi_square_spectroscopic(self, model: ModelPrediction) -> Dict[str, float]:
        """
        Compute chi-square for each spectroscopic parameter.
        
        Returns dictionary with chi-square for each parameter.
        """
        chi2 = {}
        
        if self.data.teff is not None and model.teff is not None:
            err = self.data.teff_error or 100.0
            if self.use_systematic_errors:
                err = np.sqrt(err**2 + self.teff_systematic**2)
            chi2['teff'] = ((self.data.teff - model.teff) / err)**2
        
        if self.data.logg is not None and model.logg is not None:
            err = self.data.logg_error or 0.2
            if self.use_systematic_errors:
                err = np.sqrt(err**2 + self.logg_systematic**2)
            chi2['logg'] = ((self.data.logg - model.logg) / err)**2
        
        if self.data.feh is not None and model.feh is not None:
            err = self.data.feh_error or 0.1
            if self.use_systematic_errors:
                err = np.sqrt(err**2 + self.feh_systematic**2)
            chi2['feh'] = ((self.data.feh - model.feh) / err)**2
        
        chi2['total'] = sum(chi2.values())
        
        return chi2
