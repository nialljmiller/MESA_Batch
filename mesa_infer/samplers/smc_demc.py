"""
Sequential Monte Carlo with Differential Evolution MCMC moves.

This module provides the SMC-DEMC refinement stage that turns the GA 
population into proper posterior samples with convergence diagnostics.

Based on ter Braak (2006) DE-MCMC and standard SMC tempering.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from typing import Callable, List, Optional, Tuple, Any
import warnings


@dataclass
class Bound:
    """Parameter bounds specification."""
    lo: float
    hi: float
    
    @property
    def range(self) -> float:
        return self.hi - self.lo


def reflect_to_bounds(x: np.ndarray, bounds: List[Bound]) -> np.ndarray:
    """
    Reflect values to stay within bounds.
    
    Uses reflection to handle boundary conditions, which preserves
    detailed balance in MCMC better than simple clipping.
    """
    y = x.copy()
    for j, b in enumerate(bounds):
        L = b.hi - b.lo
        if L <= 0:
            continue
        t = (y[j] - b.lo) % (2 * L)
        y[j] = b.lo + (t if t <= L else 2 * L - t)
    return y


def systematic_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Systematic resampling for SMC.
    
    More efficient than multinomial resampling and reduces variance.
    """
    N = len(w)
    positions = (rng.random() + np.arange(N)) / N
    cumsum = np.cumsum(w)
    idx = np.searchsorted(cumsum, positions, side='right')
    return np.clip(idx, 0, N - 1)


def effective_sample_size(w: np.ndarray) -> float:
    """
    Compute effective sample size from importance weights.
    
    ESS = (sum(w))^2 / sum(w^2)
    """
    s = w.sum()
    if s == 0:
        return 0.0
    return s * s / np.dot(w, w)


def choose_next_beta(
    loss: np.ndarray, 
    beta_prev: float, 
    target_ess_frac: float = 0.6
) -> float:
    """
    Choose next temperature beta to maintain target ESS.
    
    Uses binary search to find the beta increment that gives
    ESS(new) ≈ target_ess_frac * N.
    
    Args:
        loss: Current loss values for all particles
        beta_prev: Previous beta value
        target_ess_frac: Target ESS as fraction of N
    
    Returns:
        Next beta value (capped at 1.0)
    """
    N = len(loss)
    lo, hi = 1e-6, max(1e-6, 1.0 - beta_prev)
    target = target_ess_frac * N
    
    for _ in range(30):  # Binary search iterations
        mid = 0.5 * (lo + hi)
        w = np.exp(-mid * loss)
        w = w / w.sum() if w.sum() > 0 else np.ones(N) / N
        ess = effective_sample_size(w)
        if ess < target:
            hi = mid
        else:
            lo = mid
    
    return min(1.0, beta_prev + lo)


def de_mh_move(
    X: np.ndarray,
    loglike: Callable[[np.ndarray, Any], float],
    bounds: List[Bound],
    metadata: np.ndarray = None,
    steps: int = 2,
    gamma: float = None,
    jitter: float = 1e-9,
    rng: np.random.Generator = None,
    max_workers: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Differential Evolution Metropolis-Hastings moves on an ensemble.
    
    Implements the ter Braak DE-MC scheme where each walker proposes
    a new position using the scaled difference of two peers plus
    optional Gaussian jitter.
    
    Args:
        X: Current ensemble positions (N x d)
        loglike: Log-likelihood function(theta, metadata) -> float
        bounds: Parameter bounds
        metadata: Optional metadata carried with each particle
        steps: Number of DE-MH steps to perform
        gamma: Scale factor (None = ter Braak default 2.38/sqrt(2d))
        jitter: Small Gaussian noise for diversity
        rng: Random number generator
        max_workers: Max parallel workers (None = auto)
    
    Returns:
        (X_new, accepted): Updated positions and boolean acceptance mask
    """
    if rng is None:
        rng = np.random.default_rng()
    
    N, d = X.shape
    if gamma is None:
        gamma = 2.38 / np.sqrt(2 * d)
    
    accepted = np.zeros(N, dtype=bool)
    
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = max(1, int(max_workers))
    use_threads = max_workers > 1
    pool = ThreadPool(processes=max_workers) if use_threads else None
    batch_size = max_workers if use_threads else 1
    
    meta_array = np.asarray(metadata, dtype=object) if metadata is not None else None
    
    def _loglike(idx: int, theta: np.ndarray) -> float:
        if meta_array is None:
            return loglike(theta, None)
        return loglike(theta, meta_array[idx])
    
    def _eval_proposal(args):
        idx, theta = args
        return _loglike(idx, theta)
    
    # Initial log-likelihoods
    L = np.array([_loglike(i, X[i]) for i in range(N)], dtype=float)
    
    try:
        for _ in range(steps):
            order = rng.permutation(N)
            
            for start in range(0, N, batch_size):
                batch = order[start:start + batch_size]
                proposals = []
                eval_args = []
                
                for i in batch:
                    # Pick two distinct other indices
                    js = list(range(N))
                    js.remove(i)
                    r1, r2 = rng.choice(js, size=2, replace=False)
                    
                    # DE proposal
                    prop = X[i] + gamma * (X[r1] - X[r2]) + rng.normal(scale=jitter, size=d)
                    prop = reflect_to_bounds(prop, bounds)
                    
                    proposals.append((i, prop))
                    eval_args.append((i, prop.copy()))
                
                # Evaluate proposals
                if pool is not None:
                    L_news = pool.map(_eval_proposal, eval_args)
                else:
                    L_news = [_eval_proposal(arg) for arg in eval_args]
                
                # Metropolis accept/reject
                for (i, prop), L_new in zip(proposals, L_news):
                    if np.log(rng.random()) < (L_new - L[i]):
                        X[i] = prop
                        L[i] = L_new
                        accepted[i] = True
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    
    return X, accepted


def run_smc_demc(
    X0: np.ndarray,
    loss_fn: Callable[[np.ndarray, Any], float],
    bounds: List[Bound],
    metadata0: np.ndarray = None,
    ess_trigger: float = 0.6,
    moves_per_stage: int = 3,
    rng: np.random.Generator = None,
    gamma_schedule: Tuple[float, float] = (None, 1.0),
    big_step_every: int = 6,
    max_workers: int = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Tempered SMC with DE-MC mutation moves.
    
    This is the main SMC-DEMC refinement algorithm that transforms
    a GA population into proper posterior samples.
    
    Args:
        X0: Initial ensemble from GA (N x d)
        loss_fn: Loss function(theta, metadata) -> scalar
        bounds: Parameter bounds
        metadata0: Optional metadata for each particle
        ess_trigger: Resample when ESS/N < this
        moves_per_stage: DE-MH steps per temperature stage
        rng: Random generator
        gamma_schedule: (default_gamma, big_gamma) for DE moves
        big_step_every: Every k stages use gamma≈1
        max_workers: Parallel workers
    
    Returns:
        (ensemble, chains_df): Final ensemble and chains DataFrame
    """
    if rng is None:
        rng = np.random.default_rng()
    
    N, d = X0.shape
    
    # Handle metadata
    if metadata0 is not None:
        metadata = np.asarray(metadata0, dtype=object)
        if metadata.ndim == 1:
            metadata = metadata[:, None]
        if metadata.shape[0] != N:
            raise ValueError("metadata0 must have same length as ensemble")
    else:
        metadata = None
    
    # Work in log-likelihood; loss L -> loglike = -L
    def loglike(theta, meta=None):
        return -float(loss_fn(theta, meta))
    
    # State
    X = X0.copy()
    beta = 0.0
    stage = 0
    chains = []
    weights = np.ones(N) / N
    
    def reweight(beta_prev, beta_new):
        nonlocal weights
        delta = beta_new - beta_prev
        loss = np.array([loss_fn(X[i], None if metadata is None else metadata[i])
                         for i in range(N)], dtype=float)
        u = np.exp(-delta * loss)
        w = weights * u
        if w.sum() > 0:
            w /= w.sum()
        else:
            w = np.ones(N) / N
        return w, loss
    
    # Anneal to beta=1
    while beta < 1.0:
        # Compute losses
        loss_now = np.array([loss_fn(X[i], None if metadata is None else metadata[i])
                            for i in range(N)], dtype=float)
        
        # Choose next beta
        beta_next = choose_next_beta(loss_now, beta, target_ess_frac=ess_trigger)
        beta_next = max(beta_next, min(1.0, beta + 1e-3))
        
        # Reweight
        weights, loss_now = reweight(beta, beta_next)
        beta = beta_next
        
        # Resample if ESS too low
        ess = effective_sample_size(weights)
        if ess < ess_trigger * N:
            idx = systematic_resample(weights, rng)
            X = X[idx]
            if metadata is not None:
                metadata = metadata[idx]
            weights = np.ones(N) / N
        
        # DE-MH moves
        default_gamma, big_gamma = gamma_schedule
        gamma = big_gamma if (stage % big_step_every == 0 and stage > 0) else default_gamma
        
        # Scale loglike by current beta
        def beta_loglike(theta, meta=None):
            return beta * loglike(theta, meta)
        
        X, acc = de_mh_move(
            X, beta_loglike, bounds,
            metadata=metadata,
            steps=moves_per_stage,
            gamma=gamma,
            rng=rng,
            max_workers=max_workers
        )
        
        # Record chain state
        for i in range(N):
            record = {
                'stage': stage,
                'particle': i,
                'beta': beta,
                'accepted': acc[i],
                'loss': loss_now[i] if i < len(loss_now) else np.nan,
            }
            for j in range(d):
                record[f'param_{j}'] = X[i, j]
            chains.append(record)
        
        stage += 1
        
        if stage > 1000:  # Safety limit
            warnings.warn("SMC-DEMC reached 1000 stages without converging to beta=1")
            break
    
    chains_df = pd.DataFrame(chains)
    return X, chains_df


class SMCDEMCSampler:
    """
    High-level SMC-DEMC sampler interface for MESA_infer.
    
    Wraps the low-level SMC-DEMC functions with a cleaner interface
    and additional features like output management and convergence
    diagnostics.
    """
    
    def __init__(
        self,
        loss_fn: Callable[[np.ndarray, Any], float],
        bounds: List[Bound],
        ess_trigger: float = 0.6,
        moves_per_stage: int = 3,
        big_step_every: int = 6,
        n_samples: int = 50000,
        burn_fraction: float = 0.2,
        max_workers: int = None,
        seed: int = None,
        output_dir: str = None,
    ):
        """
        Initialize the SMC-DEMC sampler.
        
        Args:
            loss_fn: Loss function to minimize
            bounds: Parameter bounds
            ess_trigger: ESS trigger for resampling
            moves_per_stage: DE-MH steps per stage
            big_step_every: Large gamma step interval
            n_samples: Number of posterior samples to generate
            burn_fraction: Fraction of stages to discard as burn-in
            max_workers: Parallel workers
            seed: Random seed
            output_dir: Output directory for results
        """
        self.loss_fn = loss_fn
        self.bounds = bounds
        self.ess_trigger = ess_trigger
        self.moves_per_stage = moves_per_stage
        self.big_step_every = big_step_every
        self.n_samples = n_samples
        self.burn_fraction = burn_fraction
        self.max_workers = max_workers
        self.rng = np.random.default_rng(seed)
        self.output_dir = output_dir
        
        self.ensemble = None
        self.chains = None
        self.samples = None
    
    def run(
        self,
        initial_ensemble: np.ndarray,
        metadata: np.ndarray = None
    ) -> dict:
        """
        Run SMC-DEMC refinement on an initial ensemble.
        
        Args:
            initial_ensemble: Starting positions (N x d)
            metadata: Optional metadata per particle
        
        Returns:
            Dictionary with ensemble, chains, samples, and file paths
        """
        print("[smc-demc] Starting Sequential Monte Carlo refinement...")
        
        self.ensemble, self.chains = run_smc_demc(
            X0=initial_ensemble,
            loss_fn=self.loss_fn,
            bounds=self.bounds,
            metadata0=metadata,
            ess_trigger=self.ess_trigger,
            moves_per_stage=self.moves_per_stage,
            rng=self.rng,
            gamma_schedule=(None, 1.0),
            big_step_every=self.big_step_every,
            max_workers=self.max_workers,
        )
        
        # Extract posterior samples
        self.samples = self._extract_samples()
        
        # Save outputs
        outputs = {}
        if self.output_dir:
            outputs = self._save_outputs()
        
        print(f"[smc-demc] Finished. Ensemble shape: {self.ensemble.shape}")
        
        return {
            'ensemble': self.ensemble,
            'chains': self.chains,
            'samples': self.samples,
            **outputs
        }
    
    def _extract_samples(self) -> pd.DataFrame:
        """Extract posterior samples from chains with burn-in removal."""
        if self.chains is None:
            return None
        
        # Get parameter columns
        param_cols = [c for c in self.chains.columns if c.startswith('param_')]
        
        # Remove burn-in
        stages = self.chains['stage'].unique()
        n_burn = int(len(stages) * self.burn_fraction)
        post_burn = self.chains[self.chains['stage'] >= n_burn]
        
        # Sample from post-burn-in
        if len(post_burn) > self.n_samples:
            samples = post_burn.sample(n=self.n_samples, random_state=self.rng)
        else:
            samples = post_burn
        
        return samples[param_cols + ['loss']].reset_index(drop=True)
    
    def _save_outputs(self) -> dict:
        """Save chains and samples to files."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        paths = {}
        
        if self.chains is not None:
            chains_path = os.path.join(self.output_dir, 'smc_demc_chains.csv')
            self.chains.to_csv(chains_path, index=False)
            paths['chains_path'] = chains_path
            print(f"[smc-demc] Wrote {chains_path}")
        
        if self.samples is not None:
            samples_path = os.path.join(self.output_dir, 'smc_demc_samples.csv')
            self.samples.to_csv(samples_path, index=False)
            paths['samples_path'] = samples_path
            print(f"[smc-demc] Wrote {samples_path}")
        
        return paths
    
    def compute_diagnostics(self) -> dict:
        """Compute convergence diagnostics."""
        if self.chains is None:
            return {}
        
        param_cols = [c for c in self.chains.columns if c.startswith('param_')]
        
        diagnostics = {
            'n_stages': self.chains['stage'].nunique(),
            'n_particles': self.chains['particle'].nunique(),
            'acceptance_rate': self.chains['accepted'].mean(),
            'final_beta': self.chains['beta'].max(),
        }
        
        # Per-parameter statistics
        for col in param_cols:
            diagnostics[f'{col}_mean'] = self.samples[col].mean() if self.samples is not None else np.nan
            diagnostics[f'{col}_std'] = self.samples[col].std() if self.samples is not None else np.nan
        
        return diagnostics
