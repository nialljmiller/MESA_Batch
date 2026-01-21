"""
Likelihood functions for MESA_infer.

Provides likelihood computations for:
- Photometric data (multi-band magnitudes, colors)
- Spectroscopic data (Teff, log g, [Fe/H], abundances)
- SED fitting with flux/wavelength data
"""

from .base import BaseLikelihood, ObservationalData, ModelPrediction
from .photometric import PhotometricLikelihood
from .spectroscopic import SpectroscopicLikelihood
from .sed import SEDLikelihood

__all__ = [
    "BaseLikelihood",
    "ObservationalData",
    "ModelPrediction",
    "PhotometricLikelihood",
    "SpectroscopicLikelihood",
    "SEDLikelihood",
]
