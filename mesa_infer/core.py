"""
Core MESA_infer class - the main user-facing interface.

This module provides the MESAInfer class which orchestrates:
- Inlist parsing and parameter space definition
- MESA model execution
- Likelihood evaluation
- GA + SMC-DEMC sampling
- Results collection and analysis
"""

from __future__ import annotations

import os
import time
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import InferConfig, SamplerConfig, MESAConfig, LikelihoodConfig
from .inlist_parser import InlistParser, ParameterSpec, ParameterType
from .runner import MESARunner, MESARunResult
from .results import InferenceResults
from .likelihood.base import ObservationalData, ModelPrediction
from .likelihood.sed import SEDLikelihood
from .likelihood.photometric import PhotometricLikelihood
from .likelihood.spectroscopic import SpectroscopicLikelihood
from .samplers.ga_sampler import GASampler
from .samplers.smc_demc import Bound, run_smc_demc, SMCDEMCSampler


class MESAInfer:
    """
    Bayesian stellar parameter inference using MESA.
    
    This is the main user-facing class for MESA_infer. It handles:
    - Parsing user-provided inlists with parameter ranges
    - Running MESA models via local execution or HPC schedulers
    - Computing likelihoods against observational data
    - Running GA + SMC-DEMC sampling for Bayesian posteriors
    - Collecting and analyzing results
    
    Example:
        # Basic usage
        infer = MESAInfer(
            inlist_path="inlist_project",
            data_path="observations.csv",
            work_dir="mesa_work",
        )
        
        # Run inference
        results = infer.run()
        
        # Get posteriors
        posteriors = results.get_posteriors()
        results.make_corner_plot()
    
    Example with full configuration:
        config = InferConfig(
            sampler=SamplerConfig(
                population_size=64,
                num_generations=100,
                run_smc_refinement=True,
            ),
            mesa=MESAConfig(
                mesa_dir="/path/to/mesa",
                timeout=3600,
            ),
            likelihood=LikelihoodConfig(
                data_type="sed",
                wavelength_range=(3000, 25000),
            ),
        )
        
        infer = MESAInfer(
            inlist_path="inlist_project",
            data_path="observations.csv",
            config=config,
        )
        results = infer.run()
    """
    
    def __init__(
        self,
        inlist_path: Optional[Union[str, Path, InferConfig]] = None,
        data_path: Optional[Union[str, Path]] = None,
        work_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[InferConfig] = None,
        mesa_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize MESAInfer.
        
        Args:
            inlist_path: Path to MESA inlist with parameter ranges
            data_path: Path to observational data CSV
            work_dir: Path to MESA work directory (with src, make/, etc.)
            output_dir: Directory for inference outputs
            config: Full configuration object (optional)
            mesa_dir: Override MESA_DIR environment variable
            verbose: Print progress information
        """
        if isinstance(inlist_path, InferConfig) and config is None and data_path is None:
            config = inlist_path
            inlist_path = None

        # Configuration
        self.config = config or InferConfig()

        self.verbose = verbose
        
        # Override MESA_DIR if provided
        if mesa_dir:
            self.config.mesa.mesa_dir = mesa_dir

        resolved_work_dir = work_dir or self.config.mesa.work_dir
        if resolved_work_dir:
            self.work_dir = Path(resolved_work_dir)
        elif inlist_path:
            self.work_dir = Path(inlist_path).parent
        else:
            self.work_dir = None

        resolved_output_dir = output_dir or self.config.output_dir or "mesa_infer_output"
        self.output_dir = Path(resolved_output_dir)
        self.config.output_dir = self.output_dir

        if inlist_path is None and self.config.mesa.inlist_name and self.work_dir:
            inlist_path = self.work_dir / self.config.mesa.inlist_name

        self.inlist_path = Path(inlist_path) if inlist_path else None
        self.data_path = Path(data_path) if data_path else None
        
        self.parser: Optional[InlistParser] = None
        self.parameters: Dict[str, ParameterSpec] = {}
        self.obs_data: Optional[ObservationalData] = None

        if self.inlist_path and self.data_path:
            self.parse_inlist()
            self.load_observations(self.data_path)
        
        # Initialize components (lazy)
        self._runner: Optional[MESARunner] = None
        self._likelihood: Optional[SEDLikelihood] = None
        self._ga_sampler: Optional[GASampler] = None
        self._results: Optional[InferenceResults] = None
        
        # Tracking
        self._eval_count = 0
        self._eval_cache: Dict[str, float] = {}
        self._sample_records: List[Dict[str, Any]] = []
    
    def _log(self, msg: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(f"[mesa_infer] {msg}")
    
    def _load_data(self) -> ObservationalData:
        """Load observational data from CSV."""
        if self.data_path is None:
            raise ValueError("Data path is not set. Call load_observations first.")
        return ObservationalData.from_csv(str(self.data_path))

    def load_observations(
        self,
        data: Union[str, Path, ObservationalData],
    ) -> ObservationalData:
        """Load observational data from a CSV path or ObservationalData."""
        if isinstance(data, ObservationalData):
            self.obs_data = data
            self.data_path = None
        else:
            self.data_path = Path(data)
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            self.obs_data = ObservationalData.from_csv(str(self.data_path))
        return self.obs_data

    def set_observations(self, data: ObservationalData) -> None:
        """Set observational data directly."""
        self.load_observations(data)

    def parse_inlist(self, inlist_path: Optional[Union[str, Path]] = None) -> None:
        """Parse the inlist and detect exploration parameters."""
        if inlist_path is not None:
            self.inlist_path = Path(inlist_path)

        if self.inlist_path is None:
            raise ValueError("Inlist path is not set.")
        if not self.inlist_path.exists():
            raise FileNotFoundError(f"Inlist not found: {self.inlist_path}")

        self._log("Parsing inlist for parameter space...")
        self.parser = InlistParser(str(self.inlist_path))
        self.parameters = self.parser.get_exploration_parameters()

        if not self.parameters:
            raise ValueError("No exploration parameters found in inlist. "
                             "Use list syntax (e.g., initial_mass = 1.0, 2.0, 5.0) "
                             "or range syntax (e.g., initial_z_min = 0.001, initial_z_max = 0.02)")

        self._log(f"Found {len(self.parameters)} parameters to explore:")
        for name, spec in self.parameters.items():
            if spec.param_type == ParameterType.CONTINUOUS:
                self._log(f"  {name}: [{spec.bounds[0]:.4g}, {spec.bounds[1]:.4g}] (continuous)")
            elif spec.param_type == ParameterType.CATEGORICAL:
                self._log(f"  {name}: {spec.categories} (categorical)")
            else:
                self._log(f"  {name}: [{spec.bounds[0]:.4g}, {spec.bounds[1]:.4g}] (integer)")
    
    @property
    def runner(self) -> MESARunner:
        """Get or create MESA runner."""
        if self._runner is None:
            if self.work_dir is None:
                raise ValueError("work_dir is not set. Provide work_dir or set it in config.")
            self._runner = MESARunner(
                mesa_dir=self.config.mesa.mesa_dir,
                work_dir=str(self.work_dir),
                timeout=self.config.mesa.timeout,
                scheduler=self.config.mesa.scheduler,
            )
        return self._runner
    
    @property
    def likelihood(self) -> SEDLikelihood:
        """Get or create likelihood function."""
        if self._likelihood is None:
            if self.obs_data is None:
                raise ValueError("Observational data not loaded. Call load_observations first.")
            lc = self.config.likelihood
            
            if lc.data_type == "sed":
                self._likelihood = SEDLikelihood(
                    self.obs_data,
                    wavelength_range=lc.wavelength_range,
                    normalize=lc.normalize_sed,
                    fit_scaling=lc.fit_scaling,
                    fit_extinction=lc.fit_extinction,
                )
            elif lc.data_type == "photometric":
                self._likelihood = PhotometricLikelihood(
                    self.obs_data,
                    use_colors=lc.use_colors,
                    use_absolute_mags=lc.use_absolute_mags,
                )
            elif lc.data_type == "spectroscopic":
                self._likelihood = SpectroscopicLikelihood(
                    self.obs_data,
                    teff_weight=lc.teff_weight if lc.use_teff else 0.0,
                    logg_weight=lc.logg_weight if lc.use_logg else 0.0,
                    feh_weight=lc.feh_weight if lc.use_feh else 0.0,
                )
            else:
                # Default to SED
                self._likelihood = SEDLikelihood(
                    self.obs_data,
                    normalize=True,
                    fit_scaling=True,
                )
        
        return self._likelihood
    
    def _params_to_dict(self, theta: np.ndarray) -> Dict[str, Any]:
        """Convert parameter array to dictionary."""
        param_dict = {}
        param_names = list(self.parameters.keys())
        
        for i, name in enumerate(param_names):
            spec = self.parameters[name]
            val = theta[i]
            
            if spec.param_type == ParameterType.CATEGORICAL:
                # Index into categories
                idx = int(round(val)) % len(spec.categories)
                param_dict[name] = spec.categories[idx]
            elif spec.param_type == ParameterType.INTEGER:
                param_dict[name] = int(round(val))
            else:
                param_dict[name] = float(val)
        
        return param_dict
    
    def _dict_to_params(self, param_dict: Dict[str, Any]) -> np.ndarray:
        """Convert parameter dictionary to array."""
        param_names = list(self.parameters.keys())
        theta = np.zeros(len(param_names))
        
        for i, name in enumerate(param_names):
            spec = self.parameters[name]
            val = param_dict.get(name)
            
            if val is None:
                # Use midpoint
                if spec.param_type == ParameterType.CATEGORICAL:
                    theta[i] = len(spec.categories) // 2
                else:
                    theta[i] = (spec.bounds[0] + spec.bounds[1]) / 2
            elif spec.param_type == ParameterType.CATEGORICAL:
                if val in spec.categories:
                    theta[i] = spec.categories.index(val)
                else:
                    theta[i] = 0
            else:
                theta[i] = float(val)
        
        return theta
    
    def _get_bounds(self) -> List[Bound]:
        """Get parameter bounds for samplers."""
        bounds = []
        for name, spec in self.parameters.items():
            if spec.param_type == ParameterType.CATEGORICAL:
                bounds.append(Bound(0, len(spec.categories) - 1))
            else:
                bounds.append(Bound(spec.bounds[0], spec.bounds[1]))
        return bounds
    
    def evaluate(self, theta: np.ndarray, metadata: Any = None) -> float:
        """
        Evaluate a single parameter point.
        
        This runs MESA with the given parameters and computes the
        likelihood against the observational data.
        
        Args:
            theta: Parameter array
            metadata: Optional metadata (unused, for compatibility)
        
        Returns:
            Loss value (negative log-likelihood)
        """
        # Convert to parameter dictionary
        param_dict = self._params_to_dict(theta)
        
        # Check cache
        cache_key = json.dumps(param_dict, sort_keys=True, default=str)
        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]
        
        self._eval_count += 1
        
        # Create unique run directory
        run_id = f"run_{self._eval_count:06d}"
        run_dir = self.output_dir / "runs" / run_id
        
        # Generate inlist with these parameters
        inlist_content = self.parser.generate_inlist(param_dict)
        
        # Run MESA
        try:
            result = self.runner.run(
                param_dict,
                run_dir=str(run_dir),
                inlist_content=inlist_content,
            )
        except Exception as e:
            self._log(f"MESA run failed: {e}")
            loss = 1e10
            self._cache_and_record(cache_key, loss, param_dict, success=False)
            return loss
        
        if not result.success:
            loss = 1e10
            self._cache_and_record(cache_key, loss, param_dict, success=False)
            return loss
        
        # Create model prediction from MESA outputs
        model_pred = self._create_model_prediction(result, param_dict)
        
        # Compute likelihood
        log_like = self.likelihood.compute(model_pred)
        
        # Convert to loss (for minimization)
        loss = -log_like if np.isfinite(log_like) else 1e10
        
        self._cache_and_record(cache_key, loss, param_dict, success=True, 
                              result=result, log_like=log_like)
        
        return loss
    
    def _cache_and_record(
        self,
        cache_key: str,
        loss: float,
        param_dict: Dict[str, Any],
        success: bool,
        result: Optional[MESARunResult] = None,
        log_like: Optional[float] = None,
    ) -> None:
        """Cache result and record for later analysis."""
        self._eval_cache[cache_key] = loss
        
        record = {
            'evaluation': self._eval_count,
            'loss': loss,
            'success': success,
            **param_dict,
        }
        
        if result is not None:
            record.update({
                'final_age': result.final_age,
                'final_mass': result.final_mass,
                'final_teff': result.final_teff,
                'final_logg': result.final_logg,
                'final_luminosity': result.final_luminosity,
            })
        
        if log_like is not None:
            record['log_likelihood'] = log_like
        
        self._sample_records.append(record)
    
    def _create_model_prediction(
        self,
        result: MESARunResult,
        param_dict: Dict[str, Any],
    ) -> ModelPrediction:
        """Create ModelPrediction from MESA run result."""
        # Check for SED output
        sed_file = result.output_files.get('sed')
        
        wavelength = None
        flux = None
        
        if sed_file and os.path.exists(sed_file):
            try:
                sed_data = pd.read_csv(sed_file)
                wavelength = sed_data['wavelength'].values
                flux = sed_data['flux'].values
            except Exception:
                pass
        
        return ModelPrediction(
            wavelength=wavelength,
            flux=flux,
            teff=result.final_teff,
            logg=result.final_logg,
            feh=param_dict.get('initial_z'),  # Approximate
            mass=result.final_mass,
            radius=result.final_radius,
            luminosity=result.final_luminosity,
            age=result.final_age,
            success=result.success,
        )
    
    def run(
        self,
        population_size: Optional[int] = None,
        num_generations: Optional[int] = None,
        run_smc: Optional[bool] = None,
        checkpoint_interval: int = 10,
        resume_from: Optional[str] = None,
    ) -> InferenceResults:
        """
        Run the full inference.
        
        This executes:
        1. GA optimization to find good regions of parameter space
        2. (Optional) SMC-DEMC refinement for proper posterior sampling
        
        Args:
            population_size: Override config population size
            num_generations: Override config generations
            run_smc: Override config SMC refinement setting
            checkpoint_interval: Generations between checkpoints
            resume_from: Path to checkpoint to resume from
        
        Returns:
            InferenceResults object with posteriors and diagnostics
        """
        if not self.parameters:
            if self.inlist_path is None:
                raise ValueError("Inlist path is not set. Provide inlist_path or set config.mesa.inlist_name.")
            self.parse_inlist()

        if self.obs_data is None:
            if self.data_path is None:
                raise ValueError("Observational data not loaded. Call load_observations first.")
            self.load_observations(self.data_path)

        # Setup output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "runs").mkdir(exist_ok=True)
        
        # Get sampler config
        sc = self.config.sampler
        pop_size = population_size or sc.population_size
        n_gen = num_generations or sc.num_generations
        do_smc = run_smc if run_smc is not None else sc.run_smc_refinement
        
        self._log(f"Starting inference:")
        self._log(f"  Population: {pop_size}")
        self._log(f"  Generations: {n_gen}")
        self._log(f"  Parameters: {len(self.parameters)}")
        self._log(f"  SMC refinement: {do_smc}")
        
        # Save configuration
        self.config.to_yaml(str(self.output_dir / "config.yaml"))
        
        # Initialize GA sampler
        self._ga_sampler = GASampler(
            parameters=self.parameters,
            evaluate_fn=self.evaluate,
            output_path=str(self.output_dir),
            population_size=pop_size,
            crossover_probability=sc.crossover_probability,
            mutation_probability=sc.mutation_probability,
            tournament_size=sc.tournament_size,
            elitism_fraction=sc.elitism_fraction,
            demc_hybrid=sc.demc_hybrid,
            demc_fraction=sc.demc_fraction,
        )
        
        # Run GA
        start_time = time.time()
        self._log("Running Genetic Algorithm...")
        
        ga_results = self._ga_sampler.run(
            num_generations=n_gen,
            checkpoint_interval=checkpoint_interval,
            resume_from=resume_from,
        )
        
        ga_time = time.time() - start_time
        self._log(f"GA completed in {ga_time/60:.1f} minutes")
        self._log(f"Total evaluations: {self._eval_count}")
        
        # Run SMC-DEMC refinement if requested
        smc_results = None
        if do_smc:
            self._log("Running SMC-DEMC refinement...")
            smc_start = time.time()
            
            smc_results = self._run_smc_refinement(ga_results['final_population'])
            
            smc_time = time.time() - smc_start
            self._log(f"SMC-DEMC completed in {smc_time/60:.1f} minutes")
        
        # Collect results
        self._results = InferenceResults(
            parameters=self.parameters,
            ga_samples=self._sample_records.copy(),
            smc_samples=smc_results.get('samples') if smc_results else None,
            smc_chains=smc_results.get('chains') if smc_results else None,
            config=self.config,
            output_dir=str(self.output_dir),
        )
        
        # Save results
        self._results.save()
        self._log(f"Results saved to {self.output_dir}")
        
        return self._results
    
    def _run_smc_refinement(
        self,
        initial_population: List[Any],
    ) -> Dict[str, Any]:
        """Run SMC-DEMC refinement starting from GA population."""
        # Convert DEAP individuals to numpy array
        X0 = np.array([list(ind) for ind in initial_population])
        
        # Get bounds
        bounds = self._get_bounds()
        
        # Create SMC sampler
        smc = SMCDEMCSampler(
            output_path=str(self.output_dir),
            parameters=self.parameters,
        )
        
        # Run SMC-DEMC
        sc = self.config.sampler
        results = smc.run(
            X0=X0,
            loss_fn=self.evaluate,
            bounds=bounds,
            ess_trigger=sc.smc_ess_trigger,
            moves_per_stage=sc.smc_moves_per_stage,
            nsamples=sc.smc_nsamples,
        )
        
        return results
    
    def get_best_fit(self) -> Dict[str, Any]:
        """Get the best-fit parameters found."""
        if not self._sample_records:
            raise RuntimeError("No samples recorded. Run inference first.")
        
        # Find minimum loss
        best_record = min(self._sample_records, key=lambda r: r['loss'])
        
        # Extract parameters
        param_names = list(self.parameters.keys())
        best_params = {name: best_record[name] for name in param_names}
        
        return {
            'parameters': best_params,
            'loss': best_record['loss'],
            'log_likelihood': best_record.get('log_likelihood'),
            'evaluation': best_record['evaluation'],
        }
    
    def summary(self) -> str:
        """Get summary of inference results."""
        if self._results is None:
            return "No results available. Run inference first."
        return self._results.summary()


def run_inference(
    inlist_path: str,
    data_path: str,
    work_dir: Optional[str] = None,
    output_dir: str = "mesa_infer_output",
    population_size: int = 64,
    num_generations: int = 100,
    run_smc: bool = True,
    mesa_dir: Optional[str] = None,
    **kwargs,
) -> InferenceResults:
    """
    Convenience function to run MESA inference.
    
    This is a simple wrapper around MESAInfer for quick usage.
    
    Args:
        inlist_path: Path to MESA inlist with parameter ranges
        data_path: Path to observational data CSV
        work_dir: Path to MESA work directory
        output_dir: Directory for outputs
        population_size: GA population size
        num_generations: Number of GA generations
        run_smc: Run SMC-DEMC refinement
        mesa_dir: Override MESA_DIR
        **kwargs: Additional arguments passed to MESAInfer
    
    Returns:
        InferenceResults object
    
    Example:
        results = run_inference(
            "inlist_project",
            "observations.csv",
            population_size=32,
            num_generations=50,
        )
        print(results.summary())
    """
    config = InferConfig(
        sampler=SamplerConfig(
            population_size=population_size,
            num_generations=num_generations,
            run_smc_refinement=run_smc,
        ),
    )
    
    infer = MESAInfer(
        inlist_path=inlist_path,
        data_path=data_path,
        work_dir=work_dir,
        output_dir=output_dir,
        config=config,
        mesa_dir=mesa_dir,
        **kwargs,
    )
    
    return infer.run()
