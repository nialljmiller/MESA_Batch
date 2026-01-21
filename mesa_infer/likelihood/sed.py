"""
SED (Spectral Energy Distribution) likelihood for MESA_infer.

Computes likelihood by fitting flux vs wavelength data to model SEDs.
This is the primary likelihood for the user's CSV input requirement.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Optional, Tuple, Union

from .base import BaseLikelihood, ObservationalData, ModelPrediction


class SEDLikelihood(BaseLikelihood):
    """
    Likelihood based on SED fitting (flux vs wavelength).
    
    This is designed for the user requirement of CSV input with
    flux and wavelength columns.
    
    Supports:
    - Direct flux comparison
    - Normalized flux comparison (shape fitting)
    - Scaling factor fitting (distance/radius)
    - Wavelength binning and smoothing
    - Masking of spectral regions
    
    Example:
        data = ObservationalData.from_csv("observations.csv")
        # CSV has columns: wavelength, flux, flux_error
        
        likelihood = SEDLikelihood(
            data,
            wavelength_range=(3000, 25000),
            normalize=True,
        )
        ll = likelihood.compute(model_prediction)
    """
    
    def __init__(
        self,
        data: ObservationalData,
        wavelength_range: Optional[Tuple[float, float]] = None,
        normalize: bool = True,
        fit_scaling: bool = True,
        fit_extinction: bool = False,
        smoothing_sigma: Optional[float] = None,
        bin_size: Optional[float] = None,
        mask_regions: Optional[List[Tuple[float, float]]] = None,
        systematic_error_fraction: float = 0.0,
        extinction_law: str = "ccm89",
        rv: float = 3.1,
    ):
        """
        Initialize SED likelihood.
        
        Args:
            data: Observational data with wavelength and flux
            wavelength_range: (min, max) wavelength range in Angstroms
            normalize: Normalize both observed and model flux
            fit_scaling: Fit optimal scaling factor
            fit_extinction: Fit extinction (A_V)
            smoothing_sigma: Gaussian smoothing sigma (pixels)
            bin_size: Wavelength bin size for rebinning
            mask_regions: List of (min, max) wavelength regions to mask
            systematic_error_fraction: Add fractional systematic error
            extinction_law: Extinction law to use
            rv: R_V value
        """
        super().__init__(data)
        
        # Validate flux data
        if data.wavelength is None or data.flux is None:
            raise ValueError("SEDLikelihood requires wavelength and flux data")
        
        self.wavelength_range = wavelength_range
        self.normalize = normalize
        self.fit_scaling = fit_scaling
        self.fit_extinction = fit_extinction
        self.smoothing_sigma = smoothing_sigma
        self.bin_size = bin_size
        self.mask_regions = mask_regions or []
        self.systematic_error_fraction = systematic_error_fraction
        self.extinction_law = extinction_law
        self.rv = rv
        
        # Preprocess observed data
        self._preprocess_data()
    
    def _preprocess_data(self) -> None:
        """Preprocess observed data (binning, masking, etc.)."""
        wave = self.data.wavelength.copy()
        flux = self.data.flux.copy()
        error = self.data.flux_error.copy() if self.data.flux_error is not None else np.ones_like(flux) * np.std(flux) * 0.1
        
        # Apply wavelength range
        if self.wavelength_range:
            mask = (wave >= self.wavelength_range[0]) & (wave <= self.wavelength_range[1])
            wave = wave[mask]
            flux = flux[mask]
            error = error[mask]
        
        # Mask specific regions
        for wmin, wmax in self.mask_regions:
            mask = ~((wave >= wmin) & (wave <= wmax))
            wave = wave[mask]
            flux = flux[mask]
            error = error[mask]
        
        # Bin if requested
        if self.bin_size:
            wave, flux, error = self._rebin(wave, flux, error, self.bin_size)
        
        # Smooth if requested
        if self.smoothing_sigma:
            flux = gaussian_filter1d(flux, self.smoothing_sigma)
        
        # Add systematic error
        if self.systematic_error_fraction > 0:
            error = np.sqrt(error**2 + (self.systematic_error_fraction * flux)**2)
        
        # Store preprocessed data
        self._wave = wave
        self._flux = flux
        self._error = error
    
    def _rebin(
        self,
        wave: np.ndarray,
        flux: np.ndarray,
        error: np.ndarray,
        bin_size: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rebin spectrum to uniform wavelength grid."""
        wave_min = wave.min()
        wave_max = wave.max()
        
        new_wave = np.arange(wave_min, wave_max, bin_size)
        new_flux = np.zeros(len(new_wave))
        new_error = np.zeros(len(new_wave))
        
        for i, wc in enumerate(new_wave):
            mask = (wave >= wc) & (wave < wc + bin_size)
            if mask.sum() > 0:
                # Weighted average
                w = 1.0 / error[mask]**2
                new_flux[i] = np.sum(w * flux[mask]) / np.sum(w)
                new_error[i] = 1.0 / np.sqrt(np.sum(w))
            else:
                new_flux[i] = np.nan
                new_error[i] = np.inf
        
        # Remove NaN bins
        valid = np.isfinite(new_flux)
        return new_wave[valid], new_flux[valid], new_error[valid]
    
    def compute(self, model: ModelPrediction) -> float:
        """
        Compute SED log-likelihood.
        
        Args:
            model: Model prediction with wavelength and flux
        
        Returns:
            Log-likelihood value
        """
        if not model.success:
            return -np.inf
        
        if model.wavelength is None or model.flux is None:
            return -np.inf
        
        # Interpolate model to observed wavelength grid
        try:
            interp_func = interp1d(
                model.wavelength, model.flux,
                kind='linear',
                bounds_error=False,
                fill_value=0.0
            )
            model_flux = interp_func(self._wave)
        except Exception:
            return -np.inf
        
        # Handle zeros
        model_flux = np.maximum(model_flux, 1e-30)
        
        # Apply extinction if fitting
        if self.fit_extinction and self.data.av is not None:
            model_flux = self._apply_extinction(self._wave, model_flux, self.data.av)
        
        obs_flux = self._flux.copy()
        obs_error = self._error.copy()
        
        # Normalize or fit scaling
        if self.normalize:
            # Normalize both to unit integral
            obs_norm = np.trapz(obs_flux, self._wave)
            mod_norm = np.trapz(model_flux, self._wave)
            
            if obs_norm > 0 and mod_norm > 0:
                obs_flux = obs_flux / obs_norm
                model_flux = model_flux / mod_norm
                obs_error = obs_error / obs_norm
        elif self.fit_scaling:
            # Find optimal scaling factor
            scale = self._optimal_scaling(obs_flux, model_flux, obs_error)
            model_flux = model_flux * scale
        
        # Compute likelihood
        ll = self.log_likelihood_gaussian(obs_flux, model_flux, obs_error)
        
        return ll
    
    def _optimal_scaling(
        self,
        obs: np.ndarray,
        mod: np.ndarray,
        error: np.ndarray
    ) -> float:
        """Find optimal scaling factor for model."""
        w = 1.0 / error**2
        scale = np.sum(w * obs * mod) / np.sum(w * mod**2)
        return max(scale, 1e-30)
    
    def _apply_extinction(
        self,
        wave: np.ndarray,
        flux: np.ndarray,
        av: float
    ) -> np.ndarray:
        """Apply extinction to model flux."""
        # CCM89 extinction law
        x = 1e4 / wave  # inverse microns
        
        a = np.zeros_like(x)
        b = np.zeros_like(x)
        
        # IR region
        ir = (x >= 0.3) & (x < 1.1)
        a[ir] = 0.574 * x[ir]**1.61
        b[ir] = -0.527 * x[ir]**1.61
        
        # Optical/NIR region
        opt = (x >= 1.1) & (x < 3.3)
        y = x[opt] - 1.82
        a[opt] = (1.0 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 +
                  0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7)
        b[opt] = (1.41338*y + 2.28305*y**2 + 1.07233*y**3 -
                  5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7)
        
        # UV region
        uv = (x >= 3.3) & (x < 8.0)
        if np.any(uv):
            xu = x[uv]
            fa = -0.04473 * (xu - 5.9)**2 - 0.009779 * (xu - 5.9)**3
            fb = 0.2130 * (xu - 5.9)**2 + 0.1207 * (xu - 5.9)**3
            fa[xu < 5.9] = 0
            fb[xu < 5.9] = 0
            
            a[uv] = 1.752 - 0.316*xu - 0.104 / ((xu-4.67)**2 + 0.341) + fa
            b[uv] = -3.090 + 1.825*xu + 1.206 / ((xu-4.62)**2 + 0.263) + fb
        
        # Far UV
        fuv = x >= 8.0
        if np.any(fuv):
            xf = x[fuv]
            a[fuv] = -1.073 - 0.628*(xf-8.0) + 0.137*(xf-8.0)**2 - 0.070*(xf-8.0)**3
            b[fuv] = 13.670 + 4.257*(xf-8.0) - 0.420*(xf-8.0)**2 + 0.374*(xf-8.0)**3
        
        # Total extinction
        a_lambda = av * (a + b / self.rv)
        
        # Apply extinction
        return flux * 10**(-0.4 * a_lambda)
    
    def chi_square_sed(self, model: ModelPrediction) -> Dict[str, float]:
        """
        Compute chi-square statistics for the SED fit.
        
        Returns:
            Dictionary with chi2, reduced chi2, and other statistics
        """
        if not model.success or model.wavelength is None:
            return {'chi2': np.inf, 'reduced_chi2': np.inf, 'n_points': 0}
        
        # Interpolate model
        interp_func = interp1d(
            model.wavelength, model.flux,
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )
        model_flux = interp_func(self._wave)
        
        obs_flux = self._flux.copy()
        obs_error = self._error.copy()
        
        # Apply same normalization/scaling as likelihood
        if self.normalize:
            obs_norm = np.trapz(obs_flux, self._wave)
            mod_norm = np.trapz(model_flux, self._wave)
            if obs_norm > 0 and mod_norm > 0:
                obs_flux = obs_flux / obs_norm
                model_flux = model_flux / mod_norm
                obs_error = obs_error / obs_norm
        elif self.fit_scaling:
            scale = self._optimal_scaling(obs_flux, model_flux, obs_error)
            model_flux = model_flux * scale
        
        chi2 = self.chi_square(obs_flux, model_flux, obs_error)
        n_points = len(obs_flux)
        n_params = 1 if self.fit_scaling else 0
        dof = max(1, n_points - n_params)
        
        return {
            'chi2': chi2,
            'reduced_chi2': chi2 / dof,
            'n_points': n_points,
            'dof': dof,
        }
