"""
Photometric likelihood for MESA_infer.

Computes likelihood based on multi-band photometry and colors.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

from .base import BaseLikelihood, ObservationalData, ModelPrediction


class PhotometricLikelihood(BaseLikelihood):
    """
    Likelihood based on photometric magnitudes and colors.
    
    Supports:
    - Individual band magnitudes (with optional distance modulus fitting)
    - Color indices (distance-independent)
    - Multiple photometric systems (Gaia, 2MASS, SDSS, etc.)
    
    Example:
        data = ObservationalData(
            magnitudes={'G': 12.34, 'BP': 12.89, 'RP': 11.67},
            magnitude_errors={'G': 0.01, 'BP': 0.02, 'RP': 0.02},
            parallax=10.0,  # mas
            parallax_error=0.1,
        )
        
        likelihood = PhotometricLikelihood(data, use_colors=True)
        ll = likelihood.compute(model_prediction)
    """
    
    # Common color definitions
    COLORS = {
        # Gaia colors
        'BP-RP': ('BP', 'RP'),
        'G-RP': ('G', 'RP'),
        'BP-G': ('BP', 'G'),
        # 2MASS colors
        'J-H': ('J', 'H'),
        'H-K': ('H', 'K'),
        'J-K': ('J', 'K'),
        # SDSS colors
        'u-g': ('u', 'g'),
        'g-r': ('g', 'r'),
        'r-i': ('r', 'i'),
        'i-z': ('i', 'z'),
        # Mixed
        'G-J': ('G', 'J'),
        'G-K': ('G', 'K'),
    }
    
    def __init__(
        self,
        data: ObservationalData,
        use_colors: bool = True,
        use_absolute_mags: bool = False,
        fit_distance: bool = False,
        fit_extinction: bool = False,
        band_weights: Optional[Dict[str, float]] = None,
        color_weights: Optional[Dict[str, float]] = None,
        extinction_law: str = "ccm89",
        rv: float = 3.1,
    ):
        """
        Initialize photometric likelihood.
        
        Args:
            data: Observational data with magnitudes
            use_colors: Use color indices (recommended)
            use_absolute_mags: Use absolute magnitudes (requires distance)
            fit_distance: Marginalize over distance
            fit_extinction: Marginalize over extinction
            band_weights: Weights for individual bands
            color_weights: Weights for colors
            extinction_law: Extinction law ('ccm89', 'fitzpatrick99')
            rv: R_V value for extinction
        """
        super().__init__(data, weights=band_weights)
        
        self.use_colors = use_colors
        self.use_absolute_mags = use_absolute_mags
        self.fit_distance = fit_distance
        self.fit_extinction = fit_extinction
        self.band_weights = band_weights or {}
        self.color_weights = color_weights or {}
        self.extinction_law = extinction_law
        self.rv = rv
        
        # Validate we have photometric data
        if not data.magnitudes:
            raise ValueError("Photometric data required for PhotometricLikelihood")
        
        # Identify available colors
        self.available_colors = []
        if use_colors:
            for color_name, (band1, band2) in self.COLORS.items():
                if band1 in data.magnitudes and band2 in data.magnitudes:
                    self.available_colors.append((color_name, band1, band2))
    
    def compute(self, model: ModelPrediction) -> float:
        """
        Compute photometric log-likelihood.
        
        Args:
            model: Model prediction with magnitudes
        
        Returns:
            Log-likelihood value
        """
        if not model.success or model.magnitudes is None:
            return -np.inf
        
        ll = 0.0
        
        if self.use_colors:
            ll += self._compute_color_likelihood(model)
        
        if self.use_absolute_mags:
            ll += self._compute_absolute_mag_likelihood(model)
        elif not self.use_colors:
            # Use apparent magnitudes with distance offset
            ll += self._compute_apparent_mag_likelihood(model)
        
        return ll
    
    def _compute_color_likelihood(self, model: ModelPrediction) -> float:
        """Compute likelihood from colors."""
        ll = 0.0
        
        for color_name, band1, band2 in self.available_colors:
            if band1 not in model.magnitudes or band2 not in model.magnitudes:
                continue
            
            # Observed color
            obs_color = self.data.magnitudes[band1] - self.data.magnitudes[band2]
            
            # Error propagation
            err1 = self.data.magnitude_errors.get(band1, 0.05)
            err2 = self.data.magnitude_errors.get(band2, 0.05)
            obs_error = np.sqrt(err1**2 + err2**2)
            
            # Model color
            mod_color = model.magnitudes[band1] - model.magnitudes[band2]
            
            # Weight
            weight = self.color_weights.get(color_name, 1.0)
            
            # Gaussian likelihood
            ll += weight * self.log_likelihood_gaussian(
                np.array([obs_color]),
                np.array([mod_color]),
                np.array([obs_error])
            )
        
        return ll
    
    def _compute_absolute_mag_likelihood(self, model: ModelPrediction) -> float:
        """Compute likelihood from absolute magnitudes."""
        if self.data.parallax is None and self.data.distance is None:
            return 0.0
        
        # Get distance
        if self.data.distance is not None:
            dist = self.data.distance
            dist_err = self.data.distance_error or dist * 0.1
        else:
            # Convert parallax to distance
            plx = self.data.parallax  # mas
            plx_err = self.data.parallax_error or plx * 0.1
            dist = 1000.0 / plx  # pc
            dist_err = dist * (plx_err / plx)
        
        # Distance modulus
        dm = 5 * np.log10(dist) - 5
        dm_err = 5 / np.log(10) * (dist_err / dist)
        
        ll = 0.0
        
        for band, obs_mag in self.data.magnitudes.items():
            if band not in model.magnitudes:
                continue
            
            obs_error = self.data.magnitude_errors.get(band, 0.05)
            
            # Observed absolute magnitude
            obs_abs = obs_mag - dm
            total_err = np.sqrt(obs_error**2 + dm_err**2)
            
            # Model absolute magnitude
            mod_abs = model.magnitudes[band]
            
            # Weight
            weight = self.band_weights.get(band, 1.0)
            
            ll += weight * self.log_likelihood_gaussian(
                np.array([obs_abs]),
                np.array([mod_abs]),
                np.array([total_err])
            )
        
        return ll
    
    def _compute_apparent_mag_likelihood(self, model: ModelPrediction) -> float:
        """
        Compute likelihood from apparent magnitudes.
        
        Fits for the optimal distance modulus offset.
        """
        obs_mags = []
        mod_mags = []
        errors = []
        
        for band, obs_mag in self.data.magnitudes.items():
            if band not in model.magnitudes:
                continue
            
            obs_mags.append(obs_mag)
            mod_mags.append(model.magnitudes[band])
            errors.append(self.data.magnitude_errors.get(band, 0.05))
        
        if len(obs_mags) == 0:
            return 0.0
        
        obs_mags = np.array(obs_mags)
        mod_mags = np.array(mod_mags)
        errors = np.array(errors)
        
        # Optimal distance modulus (weighted least squares)
        weights = 1.0 / errors**2
        dm_best = np.sum(weights * (obs_mags - mod_mags)) / np.sum(weights)
        
        # Compute likelihood with optimal offset
        residuals = obs_mags - (mod_mags + dm_best)
        
        return self.log_likelihood_gaussian(obs_mags, mod_mags + dm_best, errors)
    
    def apply_extinction(
        self,
        magnitudes: Dict[str, float],
        av: float
    ) -> Dict[str, float]:
        """
        Apply extinction to magnitudes.
        
        Args:
            magnitudes: Input magnitudes
            av: V-band extinction
        
        Returns:
            Extincted magnitudes
        """
        # Approximate extinction coefficients A_X / A_V
        extinction_coeffs = {
            # Gaia
            'G': 0.85,
            'BP': 1.08,
            'RP': 0.63,
            # 2MASS
            'J': 0.28,
            'H': 0.18,
            'K': 0.11,
            # SDSS
            'u': 1.58,
            'g': 1.23,
            'r': 0.87,
            'i': 0.68,
            'z': 0.49,
            # Johnson
            'U': 1.53,
            'B': 1.32,
            'V': 1.00,
            'R': 0.75,
            'I': 0.48,
        }
        
        extincted = {}
        for band, mag in magnitudes.items():
            coeff = extinction_coeffs.get(band, 1.0)
            extincted[band] = mag + coeff * av
        
        return extincted
