"""
Utility functions for MESA_infer.
"""

from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import wraps


# ==============================================================================
# Logging utilities
# ==============================================================================

class Logger:
    """Simple logging utility for MESA_infer."""
    
    LEVELS = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3}
    
    def __init__(self, name: str = "mesa_infer", level: str = "INFO"):
        self.name = name
        self.level = self.LEVELS.get(level.upper(), 1)
    
    def _log(self, level: str, msg: str) -> None:
        if self.LEVELS.get(level, 0) >= self.level:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{self.name}] [{level}] {msg}")
    
    def debug(self, msg: str) -> None:
        self._log("DEBUG", msg)
    
    def info(self, msg: str) -> None:
        self._log("INFO", msg)
    
    def warning(self, msg: str) -> None:
        self._log("WARNING", msg)
    
    def error(self, msg: str) -> None:
        self._log("ERROR", msg)


# Global logger instance
logger = Logger()


# ==============================================================================
# File I/O utilities
# ==============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if needed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_remove(path: Union[str, Path]) -> bool:
    """Safely remove file or directory."""
    path = Path(path)
    try:
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            import shutil
            shutil.rmtree(path)
        return True
    except Exception:
        return False


def hash_dict(d: Dict[str, Any]) -> str:
    """Create hash of dictionary for caching."""
    serialized = json.dumps(d, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()


# ==============================================================================
# Numerical utilities
# ==============================================================================

def log_uniform(low: float, high: float, size: int = 1) -> np.ndarray:
    """Draw samples from log-uniform distribution."""
    log_low = np.log10(low)
    log_high = np.log10(high)
    samples = 10 ** np.random.uniform(log_low, log_high, size)
    return samples if size > 1 else samples[0]


def should_use_log_scale(low: float, high: float, threshold: float = 100.0) -> bool:
    """Determine if log scale should be used based on range ratio."""
    if low <= 0:
        return False
    return (high / low) > threshold


def reflect_at_bounds(value: float, low: float, high: float) -> float:
    """Reflect value back into bounds."""
    if low >= high:
        return low
    
    span = high - low
    x = value - low
    
    # Fold back into [0, span]
    cycles = x // span
    x = x - cycles * span
    
    # Reflect if needed
    if int(cycles) % 2 == 1:
        x = span - x
    
    return low + x


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize array to sum to 1."""
    total = np.sum(arr)
    if total > 0:
        return arr / total
    return arr


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Union[float, List[float]],
) -> Union[float, np.ndarray]:
    """Compute weighted quantiles."""
    values = np.asarray(values)
    weights = np.asarray(weights)
    
    if isinstance(quantiles, (int, float)):
        quantiles = [quantiles]
    quantiles = np.asarray(quantiles)
    
    # Sort by value
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    
    # Cumulative weights
    cumsum = np.cumsum(weights)
    cumsum /= cumsum[-1]
    
    # Interpolate
    result = np.interp(quantiles, cumsum, values)
    
    return result[0] if len(result) == 1 else result


# ==============================================================================
# Parallel processing utilities
# ==============================================================================

def get_num_cores(requested: Optional[int] = None, max_fraction: float = 0.75) -> int:
    """Get number of CPU cores to use."""
    available = os.cpu_count() or 1
    max_cores = max(1, int(available * max_fraction))
    
    if requested is None:
        return max_cores
    return min(requested, max_cores)


def alloc_cores() -> int:
    """Allocate CPU cores for parallel processing."""
    # Check for SLURM allocation
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        return int(slurm_cpus)
    
    # Check for PBS allocation
    pbs_ncpus = os.environ.get('PBS_NCPUS')
    if pbs_ncpus:
        return int(pbs_ncpus)
    
    # Default to available cores
    return get_num_cores()


# ==============================================================================
# MESA-specific utilities
# ==============================================================================

def get_mesa_dir() -> Optional[str]:
    """Get MESA directory from environment."""
    return os.environ.get('MESA_DIR')


def validate_mesa_dir(mesa_dir: str) -> bool:
    """Validate MESA directory structure."""
    mesa_path = Path(mesa_dir)
    
    required = [
        'star/defaults/star_job.defaults',
        'star/defaults/controls.defaults',
        'star/make/makefile',
    ]
    
    for rel_path in required:
        if not (mesa_path / rel_path).exists():
            return False
    
    return True


def parse_mesa_history(history_file: Union[str, Path]) -> pd.DataFrame:
    """Parse MESA history.data file."""
    path = Path(history_file)
    
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")
    
    # Read header to get column names
    with open(path, 'r') as f:
        # Skip first 5 lines (header comments)
        for _ in range(5):
            f.readline()
        # Line 6 has column names
        header = f.readline().strip().split()
    
    # Read data
    df = pd.read_csv(
        path,
        delim_whitespace=True,
        skiprows=6,
        names=header,
    )
    
    return df


def parse_mesa_profile(profile_file: Union[str, Path]) -> pd.DataFrame:
    """Parse MESA profile file."""
    path = Path(profile_file)
    
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")
    
    # Similar structure to history
    with open(path, 'r') as f:
        for _ in range(5):
            f.readline()
        header = f.readline().strip().split()
    
    df = pd.read_csv(
        path,
        delim_whitespace=True,
        skiprows=6,
        names=header,
    )
    
    return df


# ==============================================================================
# Checkpoint utilities
# ==============================================================================

class CheckpointManager:
    """Manage checkpoints for long-running inference."""
    
    def __init__(self, checkpoint_dir: Union[str, Path], prefix: str = "checkpoint"):
        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.prefix = prefix
    
    def save(self, data: Dict[str, Any], generation: int) -> Path:
        """Save checkpoint."""
        filename = f"{self.prefix}_gen{generation:06d}.pkl"
        path = self.checkpoint_dir / filename
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        return path
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load most recent checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob(f"{self.prefix}_*.pkl"))
        
        if not checkpoints:
            return None
        
        latest = checkpoints[-1]
        
        with open(latest, 'rb') as f:
            return pickle.load(f)
    
    def load(self, generation: int) -> Optional[Dict[str, Any]]:
        """Load specific checkpoint."""
        filename = f"{self.prefix}_gen{generation:06d}.pkl"
        path = self.checkpoint_dir / filename
        
        if not path.exists():
            return None
        
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def clean_old(self, keep_last: int = 3) -> int:
        """Remove old checkpoints, keeping the most recent."""
        checkpoints = sorted(self.checkpoint_dir.glob(f"{self.prefix}_*.pkl"))
        
        to_remove = checkpoints[:-keep_last] if len(checkpoints) > keep_last else []
        
        for path in to_remove:
            path.unlink()
        
        return len(to_remove)


# ==============================================================================
# Timing utilities
# ==============================================================================

class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        if self.name:
            logger.info(f"{self.name}: {self.elapsed:.2f}s")


def timed(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__}: {elapsed:.2f}s")
        return result
    return wrapper


# ==============================================================================
# Statistics utilities
# ==============================================================================

def effective_sample_size(weights: np.ndarray) -> float:
    """Compute effective sample size from importance weights."""
    weights = np.asarray(weights)
    total = weights.sum()
    if total == 0:
        return 0.0
    return total**2 / np.dot(weights, weights)


def gelman_rubin(chains: np.ndarray) -> float:
    """
    Compute Gelman-Rubin R-hat statistic.
    
    Args:
        chains: Array of shape (n_chains, n_samples)
    
    Returns:
        R-hat value (should be < 1.1 for convergence)
    """
    n_chains, n_samples = chains.shape
    
    # Chain means
    chain_means = chains.mean(axis=1)
    
    # Overall mean
    overall_mean = chain_means.mean()
    
    # Between-chain variance
    B = n_samples / (n_chains - 1) * np.sum((chain_means - overall_mean)**2)
    
    # Within-chain variance
    W = np.mean(np.var(chains, axis=1, ddof=1))
    
    # Pooled variance estimate
    var_plus = (n_samples - 1) / n_samples * W + B / n_samples
    
    # R-hat
    R_hat = np.sqrt(var_plus / W)
    
    return R_hat


def autocorrelation(x: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """Compute autocorrelation function."""
    x = np.asarray(x)
    n = len(x)
    
    if max_lag is None:
        max_lag = n // 2
    
    x = x - x.mean()
    
    acf = np.correlate(x, x, mode='full')[n-1:]
    acf = acf / acf[0]
    
    return acf[:max_lag]


def integrated_autocorrelation_time(x: np.ndarray) -> float:
    """Estimate integrated autocorrelation time."""
    acf = autocorrelation(x)
    
    # Find where ACF goes negative or below threshold
    threshold = 0.05
    for i, a in enumerate(acf):
        if a < threshold:
            break
    
    # Integrate up to that point
    tau = 1 + 2 * np.sum(acf[1:i])
    
    return tau
