"""
MESA_infer — Bayesian Stellar Parameter Inference with MESA

A generalized sampler framework for fitting MESA stellar evolution models
to observational data using Genetic Algorithms with SMC-DEMC refinement.

Features:
    - Full integration with MESA via environment variables and paths
    - Generalized sampler framework operating within MESA, all local
    - Support for ANY MESA inlist parameters (categorical and continuous)
    - User provides MESA-format inlist with parameter ranges/lists
    - Input data as CSV with flux and wavelength
    - GA + SMC-DEMC posterior sampling with convergence diagnostics

Example usage:
    from mesa_infer import MESAInfer, run_inference
    
    # Quick usage with convenience function
    results = run_inference(
        inlist_path="inlist_project",
        data_path="observations.csv",
        population_size=64,
        num_generations=100,
    )
    
    # Full control with MESAInfer class
    infer = MESAInfer(
        inlist_path="inlist_project",
        data_path="observations.csv",
        work_dir="mesa_work",
    )
    results = infer.run()
    
    # Access posteriors
    posteriors = results.get_posteriors()
    results.make_corner_plot()

For more information, see:
    https://github.com/nialljmiller/MESA_infer
"""

from .core import MESAInfer, run_inference
from .inlist_parser import InlistParser, ParameterSpec, ParameterType
from .samplers import GASampler, SMCDEMCSampler
from .likelihood import (
    ObservationalData,
    ModelPrediction,
    BaseLikelihood,
    SEDLikelihood,
    PhotometricLikelihood,
    SpectroscopicLikelihood,
)
from .runner import MESARunner, MESARunResult
from .results import InferenceResults
from .config import (
    InferConfig,
    SamplerConfig,
    MESAConfig,
    LikelihoodConfig,
)

__version__ = "0.1.0"
__author__ = "Niall Miller"
__email__ = "niall.j.miller@gmail.com"

__all__ = [
    # Core
    "MESAInfer",
    "run_inference",
    # Parsing
    "InlistParser",
    "ParameterSpec",
    "ParameterType",
    # Samplers
    "GASampler",
    "SMCDEMCSampler",
    # Likelihood
    "ObservationalData",
    "ModelPrediction",
    "BaseLikelihood",
    "SEDLikelihood",
    "PhotometricLikelihood",
    "SpectroscopicLikelihood",
    # Execution
    "MESARunner",
    "MESARunResult",
    # Results
    "InferenceResults",
    # Configuration
    "InferConfig",
    "SamplerConfig",
    "MESAConfig",
    "LikelihoodConfig",
]
