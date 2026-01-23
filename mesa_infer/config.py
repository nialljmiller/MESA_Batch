"""
Configuration management for MESA_infer.

Handles environment variables, paths, and sampler hyperparameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
import json


@dataclass
class SamplerConfig:
    """Configuration for the GA+SMC-DEMC sampler."""
    
    # GA parameters
    population_size: int = 64
    num_generations: int = 100
    crossover_probability: float = 0.5
    mutation_probability: float = 0.5
    tournament_size: int = 2
    elitism_fraction: float = 0.0625  # ~6% elite preservation
    
    # Mutation parameters
    mutation_type: str = "gaussian"  # 'gaussian' or 'uniform'
    gaussian_sigma_scale: float = 0.01
    perturbation_strength: float = 0.1
    
    # DE-MC hybrid parameters
    demc_hybrid: bool = True
    demc_fraction: float = 0.5
    demc_moves_per_gen: int = 2
    demc_gamma: Optional[float] = None  # None = use ter Braak default
    
    # SMC-DEMC refinement
    run_smc_refinement: bool = True
    smc_ess_trigger: float = 0.6
    smc_moves_per_stage: int = 3
    smc_big_step_every: int = 6
    smc_nsamples: int = 50000
    smc_burn_fraction: float = 0.2
    
    # Checkpointing
    checkpoint_interval: int = 10
    output_interval: int = 10
    
    # Parallelization
    n_workers: Optional[int] = None  # None = auto-detect
    
    # Convergence
    convergence_threshold: float = 0.01
    min_generations: int = 20
    
    # Random seed
    seed: Optional[int] = None


@dataclass
class MESAConfig:
    """Configuration for MESA execution."""
    
    mesa_dir: Optional[Path] = None
    work_dir: Optional[Path] = None
    inlist_name: Optional[str] = None
    
    # Execution settings
    timeout: int = 7200  # 2 hours default
    retry_on_failure: bool = True
    max_retries: int = 3
    parallel_runs: int = 1
    
    # Output handling
    keep_logs: bool = True
    keep_photos: bool = False
    keep_models: bool = True
    
    # HPC settings
    scheduler: str = "local"  # 'local', 'slurm', 'pbs'
    slurm_partition: Optional[str] = None
    slurm_time: str = "02:00:00"
    slurm_memory: str = "4G"
    
    def __post_init__(self):
        """Resolve paths from environment if not specified."""
        if self.mesa_dir is None:
            mesa_dir_env = os.environ.get("MESA_DIR")
            if mesa_dir_env:
                self.mesa_dir = Path(mesa_dir_env)
        
        if self.work_dir is not None:
            self.work_dir = Path(self.work_dir)
        if self.mesa_dir is not None:
            self.mesa_dir = Path(self.mesa_dir)
    
    def validate(self) -> List[str]:
        """Validate MESA configuration. Returns list of errors."""
        errors = []
        
        if self.mesa_dir is None:
            errors.append("MESA_DIR not set (set environment variable or pass mesa_dir)")
        elif not self.mesa_dir.exists():
            errors.append(f"MESA_DIR does not exist: {self.mesa_dir}")
        
        if self.work_dir is None:
            errors.append("work_dir must be specified")
        elif not self.work_dir.exists():
            errors.append(f"work_dir does not exist: {self.work_dir}")
        
        return errors


@dataclass
class LikelihoodConfig:
    """Configuration for likelihood evaluation."""
    
    # Data type
    data_type: str = "photometric"  # 'photometric', 'spectroscopic', 'combined'
    
    # Photometric settings
    photometric_bands: List[str] = field(default_factory=list)
    photometric_uncertainties: bool = True
    
    # Spectroscopic settings
    teff_weight: float = 1.0
    logg_weight: float = 1.0
    feh_weight: float = 1.0
    
    # SED fitting
    use_sed_fitting: bool = True
    wavelength_range: tuple = (3000, 25000)  # Angstroms
    normalize_sed: bool = True
    fit_scaling: bool = True
    fit_extinction: bool = False
    
    # Extinction
    apply_extinction: bool = True
    extinction_prior: str = "flat"  # 'flat', 'exponential', 'green19'
    
    # Distance
    use_distance_prior: bool = True
    parallax: Optional[float] = None
    parallax_error: Optional[float] = None

    # Photometric likelihood
    use_colors: bool = True
    use_absolute_mags: bool = False

    # Spectroscopic likelihood
    use_teff: bool = True
    use_logg: bool = True
    use_feh: bool = True


@dataclass 
class InferConfig:
    """Master configuration for MESA_infer."""
    
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    mesa: MESAConfig = field(default_factory=MESAConfig)
    likelihood: LikelihoodConfig = field(default_factory=LikelihoodConfig)

    # Backwards-compatible config inputs
    sampler_config: Optional[Dict[str, Any]] = field(default=None, repr=False)
    mesa_config: Optional[Dict[str, Any]] = field(default=None, repr=False)
    likelihood_config: Optional[Dict[str, Any]] = field(default=None, repr=False)
    
    # Output settings
    output_dir: Optional[Path] = None
    run_name: str = "mesa_infer_run"
    verbose: bool = True
    
    # Plotting
    make_plots: bool = True
    plot_interval: int = 10
    
    def __post_init__(self):
        from dataclasses import asdict

        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)

        if self.sampler_config:
            sampler_data = {**asdict(self.sampler), **self.sampler_config}
            self.sampler = SamplerConfig(**sampler_data)

        if self.mesa_config:
            mesa_data = {**asdict(self.mesa), **self.mesa_config}
            self.mesa = MESAConfig(**mesa_data)

        if self.likelihood_config:
            likelihood_data = {**asdict(self.likelihood), **self.likelihood_config}
            self.likelihood = LikelihoodConfig(**likelihood_data)
    
    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> InferConfig:
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)
    
    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> InferConfig:
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> InferConfig:
        """Create config from dictionary."""
        sampler_data = data.get('sampler', {})
        mesa_data = data.get('mesa', {})
        likelihood_data = data.get('likelihood', {})

        if 'wavelength_range' not in likelihood_data:
            wavelength_min = likelihood_data.pop('wavelength_min', None)
            wavelength_max = likelihood_data.pop('wavelength_max', None)
            if wavelength_min is not None and wavelength_max is not None:
                likelihood_data['wavelength_range'] = (wavelength_min, wavelength_max)
        
        sampler = SamplerConfig(**sampler_data)
        mesa = MESAConfig(**mesa_data)
        likelihood = LikelihoodConfig(**likelihood_data)
        
        return cls(
            sampler=sampler,
            mesa=mesa,
            likelihood=likelihood,
            output_dir=data.get('output_dir'),
            run_name=data.get('run_name', 'mesa_infer_run'),
            verbose=data.get('verbose', True),
            make_plots=data.get('make_plots', True),
            plot_interval=data.get('plot_interval', 10),
        )
    
    def to_yaml(self, filepath: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        data = self._to_dict()
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        data = self._to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        
        def _convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            return obj
        
        data = {
            'sampler': asdict(self.sampler),
            'mesa': {k: _convert(v) for k, v in asdict(self.mesa).items()},
            'likelihood': asdict(self.likelihood),
            'output_dir': str(self.output_dir) if self.output_dir else None,
            'run_name': self.run_name,
            'verbose': self.verbose,
            'make_plots': self.make_plots,
            'plot_interval': self.plot_interval,
        }
        return data
    
    def validate(self) -> List[str]:
        """Validate all configuration. Returns list of errors."""
        errors = []
        errors.extend(self.mesa.validate())
        
        if self.output_dir is None:
            errors.append("output_dir must be specified")
        
        if self.sampler.population_size < 4:
            errors.append("population_size must be at least 4")
        
        if self.sampler.num_generations < 1:
            errors.append("num_generations must be at least 1")
        
        return errors
