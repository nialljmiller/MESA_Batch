"""
MESA Batch Runner - A batch runner for MESA stellar evolution simulations.

This package provides both a Python API and CLI for running parameter sweeps
of MESA stellar evolution models.

Example usage (Python):
    from mesa_batch_runner import BatchRunner
    
    runner = BatchRunner("/path/to/mesa/work")
    runner.add_parameter("initial_mass", values=[1.0, 2.0, 5.0])
    runner.add_parameter("initial_z", min=0.001, max=0.02, steps=5)
    runner.run()

Example usage (CLI):
    mesa-batch /path/to/mesa/work --config batch_inlist
"""

from .batch import BatchRunner
from .grid import ParameterGrid
from .inlist import InlistParser, InlistModifier
from .results import ResultsCollector
from .config import BatchConfig

__version__ = "0.1.0"
__all__ = [
    "BatchRunner",
    "ParameterGrid", 
    "InlistParser",
    "InlistModifier",
    "ResultsCollector",
    "BatchConfig",
]
