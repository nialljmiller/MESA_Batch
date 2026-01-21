"""
Genetic Algorithm Sampler for MESA_infer.

Implements a hybrid GA+DE-MCMC sampler using DEAP, supporting both
categorical and continuous parameters with adaptive operators.
"""

from __future__ import annotations

import gc
import os
import time
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from multiprocessing import Pool, get_context
import warnings

try:
    from deap import base, creator, tools
    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False
    warnings.warn("DEAP not installed. Install with: pip install deap")

from .smc_demc import Bound, de_mh_move, run_smc_demc, SMCDEMCSampler
from ..inlist_parser import ParameterSpec, ParameterType


def log_uniform(lo: float, hi: float) -> float:
    """Sample from log-uniform distribution."""
    log_lo = np.log10(lo)
    log_hi = np.log10(hi)
    return 10 ** random.uniform(log_lo, log_hi)


def should_use_log(lo: float, hi: float) -> bool:
    """Determine if log scale should be used."""
    if lo <= 0:
        return False
    return (hi / lo) > 100


def alloc_cores() -> int:
    """Get number of available CPU cores respecting scheduler limits."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


@dataclass
class EvaluationResult:
    """Result from evaluating a single individual."""
    fitness: float
    parameters: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class GASampler:
    """
    Genetic Algorithm sampler with DE-MCMC hybrid moves.
    
    Supports:
    - Mixed categorical and continuous parameters
    - Adaptive mutation rates
    - DE-MCMC moves for faster convergence
    - Elitism to preserve best solutions
    - Checkpointing for long runs
    
    Example:
        sampler = GASampler(
            parameters=param_specs,
            evaluate_fn=my_evaluate,
            population_size=64,
            num_generations=100,
        )
        results = sampler.run()
    """
    
    def __init__(
        self,
        parameters: List[ParameterSpec],
        evaluate_fn: Callable[[Dict[str, Any]], EvaluationResult],
        population_size: int = 64,
        num_generations: int = 100,
        crossover_probability: float = 0.5,
        mutation_probability: float = 0.5,
        tournament_size: int = 2,
        elitism_fraction: float = 0.0625,
        mutation_type: str = "gaussian",
        gaussian_sigma_scale: float = 0.01,
        perturbation_strength: float = 0.1,
        demc_hybrid: bool = True,
        demc_fraction: float = 0.5,
        demc_moves_per_gen: int = 2,
        demc_gamma: Optional[float] = None,
        run_smc_refinement: bool = True,
        smc_config: Optional[Dict] = None,
        n_workers: Optional[int] = None,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        checkpoint_interval: int = 10,
        output_interval: int = 10,
        verbose: bool = True,
    ):
        """
        Initialize the GA sampler.
        
        Args:
            parameters: List of ParameterSpec objects defining the search space
            evaluate_fn: Function that evaluates a parameter dict and returns EvaluationResult
            population_size: Number of individuals in population
            num_generations: Number of generations to run
            crossover_probability: Probability of crossover
            mutation_probability: Probability of mutation
            tournament_size: Size of tournament for selection
            elitism_fraction: Fraction of population to preserve as elites
            mutation_type: 'gaussian' or 'uniform'
            gaussian_sigma_scale: Scale for Gaussian mutation
            perturbation_strength: Strength for perturbation
            demc_hybrid: Enable DE-MCMC hybrid moves
            demc_fraction: Fraction of population for DE-MCMC
            demc_moves_per_gen: DE-MCMC moves per generation
            demc_gamma: DE-MCMC gamma parameter
            run_smc_refinement: Run SMC-DEMC refinement after GA
            smc_config: Configuration for SMC-DEMC stage
            n_workers: Number of parallel workers
            seed: Random seed
            output_dir: Directory for outputs
            checkpoint_interval: Generations between checkpoints
            output_interval: Generations between result outputs
            verbose: Print progress
        """
        if not HAS_DEAP:
            raise ImportError("DEAP is required. Install with: pip install deap")
        
        self.parameters = parameters
        self.evaluate_fn = evaluate_fn
        self.population_size = population_size
        self.num_generations = num_generations
        self.cxpb = crossover_probability
        self.mutpb = mutation_probability
        self.tournament_size = tournament_size
        self.elitism_fraction = elitism_fraction
        self.mutation_type = mutation_type
        self.gaussian_sigma_scale = gaussian_sigma_scale
        self.perturbation_strength = perturbation_strength
        self.demc_hybrid = demc_hybrid
        self.demc_fraction = demc_fraction
        self.demc_moves_per_gen = demc_moves_per_gen
        self.demc_gamma = demc_gamma
        self.run_smc_refinement = run_smc_refinement
        self.smc_config = smc_config or {}
        self.n_workers = n_workers or alloc_cores()
        self.output_dir = output_dir
        self.checkpoint_interval = checkpoint_interval
        self.output_interval = output_interval
        self.verbose = verbose
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        
        # Categorize parameters
        self._setup_parameters()
        
        # DEAP setup
        self._setup_deap()
        
        # Tracking
        self.generation = 0
        self.sample_records: List[Dict] = []
        self.walker_history: Dict[int, List] = {}
        self.best_fitness = float('inf')
        self.best_individual = None
        
        # Results
        self.population = None
        self.results_df = None
        self.smc_products = None
    
    def _setup_parameters(self):
        """Setup parameter indices and bounds."""
        self.categorical_params = []
        self.continuous_params = []
        self.param_names = []
        self.param_indices = {}
        self.bounds = []
        
        idx = 0
        for param in self.parameters:
            self.param_names.append(param.name)
            self.param_indices[param.name] = idx
            
            if param.param_type == ParameterType.CATEGORICAL:
                self.categorical_params.append((idx, param))
            else:
                self.continuous_params.append((idx, param))
            
            lo, hi = param.bounds
            self.bounds.append(Bound(lo=lo, hi=hi))
            idx += 1
        
        self.n_params = len(self.parameters)
        self.categorical_indices = [i for i, _ in self.categorical_params]
        self.continuous_indices = [i for i, _ in self.continuous_params]
    
    def _setup_deap(self):
        """Setup DEAP framework."""
        # Clear any existing DEAP classes
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        # Create fitness and individual classes
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        
        # Register attribute generators for each parameter
        for idx, param in enumerate(self.parameters):
            attr_name = f"attr_{idx}"
            
            if param.param_type == ParameterType.CATEGORICAL:
                n_cats = len(param.categories)
                self.toolbox.register(attr_name, random.randint, 0, n_cats - 1)
            elif param.param_type == ParameterType.INTEGER:
                self.toolbox.register(attr_name, random.randint, 
                                     int(param.min_val), int(param.max_val))
            else:  # Continuous
                if param.log_scale:
                    self.toolbox.register(attr_name, log_uniform, 
                                         param.min_val, param.max_val)
                else:
                    self.toolbox.register(attr_name, random.uniform,
                                         param.min_val, param.max_val)
        
        # Register individual and population
        attr_tuple = tuple(getattr(self.toolbox, f"attr_{i}") for i in range(self.n_params))
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                             attr_tuple, n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", self._tournament_select)
    
    def _evaluate_individual(self, individual: list) -> Tuple[Tuple[float], EvaluationResult]:
        """Evaluate a single individual."""
        # Convert individual to parameter dictionary
        params = self._individual_to_dict(individual)
        
        # Call evaluation function
        try:
            result = self.evaluate_fn(params)
            fitness = (result.fitness,)
        except Exception as e:
            result = EvaluationResult(
                fitness=float('inf'),
                parameters=params,
                success=False,
                error=str(e)
            )
            fitness = (float('inf'),)
        
        return fitness, result
    
    def _individual_to_dict(self, individual: list) -> Dict[str, Any]:
        """Convert individual to parameter dictionary."""
        params = {}
        for idx, param in enumerate(self.parameters):
            val = individual[idx]
            
            if param.param_type == ParameterType.CATEGORICAL:
                # Convert index to category value
                val = param.categories[int(val)]
            elif param.param_type == ParameterType.INTEGER:
                val = int(round(val))
            
            params[param.name] = val
        
        return params
    
    def _dict_to_individual(self, params: Dict[str, Any]) -> list:
        """Convert parameter dictionary to individual."""
        individual = []
        for param in self.parameters:
            val = params[param.name]
            
            if param.param_type == ParameterType.CATEGORICAL:
                # Convert category value to index
                val = param.categories.index(val)
            
            individual.append(val)
        
        return creator.Individual(individual)
    
    def _crossover(self, ind1: list, ind2: list, max_bias: float = 0.75) -> Tuple[list, list]:
        """Fitness-weighted crossover favoring the better parent."""
        # Determine fitness weights
        if ind1.fitness.valid and ind2.fitness.valid:
            fit1 = ind1.fitness.values[0]
            fit2 = ind2.fitness.values[0]
            total = fit1 + fit2
            
            if total > 0:
                weight1 = fit2 / total  # Better parent gets higher weight
                weight2 = fit1 / total
                
                if weight1 > max_bias:
                    weight1 = max_bias
                    weight2 = 1 - max_bias
                elif weight2 > max_bias:
                    weight2 = max_bias
                    weight1 = 1 - max_bias
            else:
                weight1 = weight2 = 0.5
        else:
            weight1 = weight2 = 0.5
        
        # Create children
        child1 = ind1[:]
        child2 = ind2[:]
        
        # Categorical: weighted selection
        for idx in self.categorical_indices:
            child1[idx] = ind1[idx] if random.random() < weight1 else ind2[idx]
            child2[idx] = ind1[idx] if random.random() < weight1 else ind2[idx]
        
        # Continuous: weighted blend with noise
        for idx in self.continuous_indices:
            avg = weight1 * ind1[idx] + weight2 * ind2[idx]
            noise_scale = abs(ind1[idx] - ind2[idx]) * 0.05
            
            child1[idx] = avg + random.gauss(0, noise_scale)
            child2[idx] = avg + random.gauss(0, noise_scale)
            
            # Apply bounds
            lo, hi = self.bounds[idx].lo, self.bounds[idx].hi
            child1[idx] = self._reflect_at_bounds(child1[idx], lo, hi)
            child2[idx] = self._reflect_at_bounds(child2[idx], lo, hi)
        
        return creator.Individual(child1), creator.Individual(child2)
    
    def _mutate(self, individual: list) -> Tuple[list]:
        """Apply mutation to individual."""
        indpb = 1.0 / self.n_params  # Per-gene mutation probability
        
        for idx in range(len(individual)):
            if random.random() < indpb:
                param = self.parameters[idx]
                
                if param.param_type == ParameterType.CATEGORICAL:
                    if random.random() < 0.1:  # 10% chance to change category
                        individual[idx] = random.randint(0, len(param.categories) - 1)
                else:
                    lo, hi = param.bounds
                    range_size = hi - lo
                    
                    if self.mutation_type == "gaussian":
                        sigma = range_size * self.gaussian_sigma_scale
                        individual[idx] += random.gauss(0, sigma)
                    else:  # uniform
                        individual[idx] = random.uniform(lo, hi)
                    
                    individual[idx] = self._reflect_at_bounds(individual[idx], lo, hi)
        
        return (individual,)
    
    def _tournament_select(self, individuals: list) -> list:
        """Tournament selection."""
        selected = []
        
        for _ in range(len(individuals)):
            tournament = random.sample(individuals, self.tournament_size)
            winner = min(tournament, key=lambda ind: ind.fitness.values[0])
            selected.append(winner)
        
        return selected
    
    def _reflect_at_bounds(self, value: float, lo: float, hi: float) -> float:
        """Reflect value at bounds."""
        L = hi - lo
        if L <= 0:
            return lo
        
        while value < lo or value > hi:
            if value < lo:
                value = lo + (lo - value)
            if value > hi:
                value = hi - (value - hi)
        
        return value
    
    def run(self, initial_population: Optional[list] = None) -> Dict[str, Any]:
        """
        Run the genetic algorithm.
        
        Args:
            initial_population: Optional starting population
        
        Returns:
            Dictionary with results including best individual, population, and statistics
        """
        start_time = time.time()
        
        # Initialize population
        if initial_population is not None:
            self.population = initial_population
        else:
            self.population = self.toolbox.population(n=self.population_size)
        
        self.walker_history = {i: [] for i in range(len(self.population))}
        
        if self.verbose:
            self._print_config()
        
        # Run GA
        if self.n_workers > 1:
            ctx = get_context("spawn")
            with ctx.Pool(processes=self.n_workers) as pool:
                self.toolbox.register("map", pool.map)
                self._run_generations()
        else:
            self._run_generations()
        
        elapsed = time.time() - start_time
        
        if self.verbose:
            print(f"\nGA completed in {elapsed:.1f}s")
            print(f"Best fitness: {self.best_fitness:.6f}")
        
        # Export GA samples
        self._export_samples()
        
        # Run SMC-DEMC refinement if requested
        if self.run_smc_refinement:
            self.smc_products = self._run_smc_refinement()
        
        return {
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'best_params': self._individual_to_dict(self.best_individual) if self.best_individual else None,
            'population': self.population,
            'results_df': self.results_df,
            'smc_products': self.smc_products,
            'elapsed_time': elapsed,
        }
    
    def _run_generations(self):
        """Main GA loop."""
        elitism_k = max(1, int(len(self.population) * self.elitism_fraction))
        
        for gen in range(self.num_generations):
            self.generation = gen
            
            if self.verbose:
                print(f"-- Generation {gen}/{self.num_generations} --")
            
            # Evaluate invalid individuals
            invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
            if invalid_ind:
                results = list(self.toolbox.map(self.toolbox.evaluate, invalid_ind))
                for ind, (fit, result) in zip(invalid_ind, results):
                    ind.fitness.values = fit
                    self._record_evaluation(result, gen)
            
            # Update best
            current_best = tools.selBest(self.population, 1)[0]
            if current_best.fitness.values[0] < self.best_fitness:
                self.best_fitness = current_best.fitness.values[0]
                self.best_individual = self.toolbox.clone(current_best)
            
            # Select elites
            elites = [self.toolbox.clone(e) for e in tools.selBest(self.population, elitism_k)]
            
            # Select and breed
            offspring = self.toolbox.select(self.population)
            offspring = [self.toolbox.clone(o) for o in offspring]
            offspring = offspring[:len(self.population) - elitism_k]
            
            # Crossover
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    self.toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values
            
            # Mutation
            for mutant in offspring:
                if random.random() < self.mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                results = list(self.toolbox.map(self.toolbox.evaluate, invalid_ind))
                for ind, (fit, result) in zip(invalid_ind, results):
                    ind.fitness.values = fit
                    self._record_evaluation(result, gen)
            
            # Update population
            self.population[:] = elites + offspring
            
            # DE-MCMC hybrid moves
            if self.demc_hybrid:
                self._apply_demc_moves()
            
            # Record history
            for idx, ind in enumerate(self.population):
                self.walker_history[idx].append(list(ind))
            
            # Periodic output
            if self.output_interval and gen % self.output_interval == 0:
                self._save_partial_results(gen)
            
            gc.collect()
    
    def _apply_demc_moves(self):
        """Apply DE-MCMC moves to population."""
        if not self.continuous_indices:
            return
        
        n_select = max(1, int(len(self.population) * self.demc_fraction))
        
        # Extract continuous parameters
        X = np.array([[ind[i] for i in self.continuous_indices] for ind in self.population])
        cont_bounds = [self.bounds[i] for i in self.continuous_indices]
        
        # Define loss for DE-MC
        def loss_fn(theta, meta=None):
            ind = self.toolbox.clone(self.population[0])
            for i, ci in enumerate(self.continuous_indices):
                ind[ci] = theta[i]
            fit, _ = self.toolbox.evaluate(ind)
            return fit[0]
        
        # Run DE-MH moves
        X_new, accepted = de_mh_move(
            X=X,
            loglike=lambda theta, m: -loss_fn(theta, m),
            bounds=cont_bounds,
            steps=self.demc_moves_per_gen,
            gamma=self.demc_gamma,
            rng=self.rng,
        )
        
        # Update population
        for i, ind in enumerate(self.population):
            if accepted[i]:
                for j, ci in enumerate(self.continuous_indices):
                    ind[ci] = X_new[i, j]
                del ind.fitness.values
    
    def _record_evaluation(self, result: EvaluationResult, generation: int):
        """Record an evaluation result."""
        record = {
            'generation': generation,
            'fitness': result.fitness,
            'success': result.success,
            **result.parameters,
            **result.metadata,
        }
        self.sample_records.append(record)
    
    def _export_samples(self):
        """Export GA samples to DataFrame."""
        if not self.sample_records:
            return
        
        self.results_df = pd.DataFrame(self.sample_records)
        
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(self.output_dir, 'ga_samples.csv')
            self.results_df.to_csv(path, index=False)
            if self.verbose:
                print(f"[ga-sampler] Wrote {path}")
    
    def _save_partial_results(self, generation: int):
        """Save partial results during run."""
        if not self.output_dir:
            return
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save current samples
        df = pd.DataFrame(self.sample_records)
        path = os.path.join(self.output_dir, f'ga_samples_gen_{generation:04d}.csv')
        df.to_csv(path, index=False)
    
    def _run_smc_refinement(self) -> Optional[Dict]:
        """Run SMC-DEMC refinement on final GA population."""
        if not self.continuous_indices:
            if self.verbose:
                print("[smc-demc] Skipping: no continuous parameters")
            return None
        
        if self.verbose:
            print("\n[smc-demc] Starting SMC refinement...")
        
        # Extract ensemble
        X0 = np.array([[ind[i] for i in self.continuous_indices] for ind in self.population])
        cont_bounds = [self.bounds[i] for i in self.continuous_indices]
        
        def loss_fn(theta, meta=None):
            ind = self.toolbox.clone(self.population[0])
            for i, ci in enumerate(self.continuous_indices):
                ind[ci] = theta[i]
            fit, _ = self.toolbox.evaluate(ind)
            return fit[0]
        
        smc = SMCDEMCSampler(
            loss_fn=loss_fn,
            bounds=cont_bounds,
            output_dir=self.output_dir,
            **self.smc_config
        )
        
        return smc.run(X0)
    
    def _print_config(self):
        """Print configuration summary."""
        print("=" * 60)
        print("GA SAMPLER CONFIGURATION")
        print("=" * 60)
        print(f"Parameters: {self.n_params}")
        print(f"  Categorical: {len(self.categorical_params)}")
        print(f"  Continuous: {len(self.continuous_params)}")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.num_generations}")
        print(f"Workers: {self.n_workers}")
        print(f"DE-MC hybrid: {self.demc_hybrid}")
        print(f"SMC refinement: {self.run_smc_refinement}")
        print("=" * 60)
        print()
