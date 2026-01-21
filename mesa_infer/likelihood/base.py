"""
Base likelihood class for MESA_infer.

Provides the interface and common functionality for all likelihood types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class ObservationalData:
    """Container for observational data."""
    
    # Data arrays
    wavelength: Optional[np.ndarray] = None  # Angstroms
    flux: Optional[np.ndarray] = None  # erg/s/cm^2/A or similar
    flux_error: Optional[np.ndarray] = None
    
    # Photometric data
    magnitudes: Optional[Dict[str, float]] = None
    magnitude_errors: Optional[Dict[str, float]] = None
    
    # Spectroscopic parameters
    teff: Optional[float] = None
    teff_error: Optional[float] = None
    logg: Optional[float] = None
    logg_error: Optional[float] = None
    feh: Optional[float] = None
    feh_error: Optional[float] = None
    
    # Abundances
    abundances: Optional[Dict[str, float]] = None
    abundance_errors: Optional[Dict[str, float]] = None
    
    # Astrometric data
    parallax: Optional[float] = None  # mas
    parallax_error: Optional[float] = None
    distance: Optional[float] = None  # pc
    distance_error: Optional[float] = None
    
    # Extinction
    av: Optional[float] = None
    av_error: Optional[float] = None
    
    # Metadata
    source_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_csv(cls, filepath: Union[str, Path], **kwargs) -> ObservationalData:
        """
        Load observational data from CSV file.
        
        Expected CSV format for flux data:
            wavelength,flux,flux_error
            3000.0,1.234e-15,1.0e-16
            ...
        
        Or for photometric data:
            band,magnitude,error
            G,12.34,0.01
            ...
        """
        df = pd.read_csv(filepath)
        
        data = cls(**kwargs)
        
        # Check for flux/wavelength columns
        if 'wavelength' in df.columns and 'flux' in df.columns:
            data.wavelength = df['wavelength'].values
            data.flux = df['flux'].values
            if 'flux_error' in df.columns:
                data.flux_error = df['flux_error'].values
            elif 'error' in df.columns:
                data.flux_error = df['error'].values
        
        # Check for photometric columns
        if 'band' in df.columns and 'magnitude' in df.columns:
            data.magnitudes = dict(zip(df['band'], df['magnitude']))
            if 'error' in df.columns:
                data.magnitude_errors = dict(zip(df['band'], df['error']))
        
        # Check for scalar parameters
        for col in ['teff', 'logg', 'feh', 'parallax', 'distance', 'av']:
            if col in df.columns:
                setattr(data, col, df[col].iloc[0])
            if f'{col}_error' in df.columns:
                setattr(data, f'{col}_error', df[f'{col}_error'].iloc[0])
        
        return data
    
    def validate(self) -> List[str]:
        """Validate the observational data. Returns list of errors."""
        errors = []
        
        if self.flux is not None:
            if self.wavelength is None:
                errors.append("flux provided without wavelength")
            elif len(self.flux) != len(self.wavelength):
                errors.append("flux and wavelength arrays have different lengths")
        
        return errors


@dataclass
class ModelPrediction:
    """Container for model predictions."""
    
    # SED
    wavelength: Optional[np.ndarray] = None
    flux: Optional[np.ndarray] = None
    
    # Photometry
    magnitudes: Optional[Dict[str, float]] = None
    
    # Stellar parameters
    teff: Optional[float] = None
    logg: Optional[float] = None
    feh: Optional[float] = None
    luminosity: Optional[float] = None
    radius: Optional[float] = None
    mass: Optional[float] = None
    age: Optional[float] = None
    
    # Model info
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseLikelihood(ABC):
    """
    Abstract base class for likelihood functions.
    
    All likelihood implementations should inherit from this class
    and implement the compute() method.
    """
    
    def __init__(
        self,
        data: ObservationalData,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the likelihood function.
        
        Args:
            data: Observational data to fit
            weights: Optional weights for different data components
        """
        self.data = data
        self.weights = weights or {}
        
        # Validate data
        errors = data.validate()
        if errors:
            raise ValueError(f"Invalid observational data: {errors}")
    
    @abstractmethod
    def compute(self, model: ModelPrediction) -> float:
        """
        Compute the log-likelihood.
        
        Args:
            model: Model prediction to compare against data
        
        Returns:
            Log-likelihood value (higher = better fit)
        """
        pass
    
    def loss(self, model: ModelPrediction) -> float:
        """
        Compute the loss (negative log-likelihood).
        
        Args:
            model: Model prediction to compare against data
        
        Returns:
            Loss value (lower = better fit)
        """
        return -self.compute(model)
    
    def chi_square(self, observed: np.ndarray, predicted: np.ndarray, 
                   errors: np.ndarray) -> float:
        """
        Compute chi-square statistic.
        
        Args:
            observed: Observed values
            predicted: Predicted values
            errors: Measurement uncertainties
        
        Returns:
            Chi-square value
        """
        residuals = (observed - predicted) / errors
        return np.sum(residuals ** 2)
    
    def log_likelihood_gaussian(
        self, 
        observed: np.ndarray, 
        predicted: np.ndarray, 
        errors: np.ndarray
    ) -> float:
        """
        Compute Gaussian log-likelihood.
        
        Args:
            observed: Observed values
            predicted: Predicted values  
            errors: Measurement uncertainties
        
        Returns:
            Log-likelihood value
        """
        n = len(observed)
        residuals = (observed - predicted) / errors
        
        ll = -0.5 * n * np.log(2 * np.pi)
        ll -= np.sum(np.log(errors))
        ll -= 0.5 * np.sum(residuals ** 2)
        
        return ll
    
    def reduced_chi_square(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        errors: np.ndarray,
        n_params: int = 0
    ) -> float:
        """
        Compute reduced chi-square.
        
        Args:
            observed: Observed values
            predicted: Predicted values
            errors: Measurement uncertainties
            n_params: Number of free parameters
        
        Returns:
            Reduced chi-square value
        """
        chi2 = self.chi_square(observed, predicted, errors)
        dof = len(observed) - n_params
        
        if dof <= 0:
            return chi2
        
        return chi2 / dof
