"""
Results handling for MESA_infer.

Provides classes for storing, analyzing, and exporting inference results.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False


@dataclass
class InferenceResults:
    """
    Container for inference results.
    
    Stores GA samples, SMC-DEMC posteriors, convergence diagnostics,
    and provides methods for analysis and export.
    """
    
    # Parameter information
    parameter_names: List[str]
    categorical_indices: List[int] = field(default_factory=list)
    continuous_indices: List[int] = field(default_factory=list)
    
    # Best fit
    best_params: Optional[Dict[str, Any]] = None
    best_fitness: Optional[float] = None
    
    # GA results
    ga_samples: Optional[pd.DataFrame] = None
    ga_population: Optional[np.ndarray] = None
    
    # SMC-DEMC results  
    smc_chains: Optional[pd.DataFrame] = None
    smc_samples: Optional[pd.DataFrame] = None
    posterior_ensemble: Optional[np.ndarray] = None
    
    # Convergence diagnostics
    acceptance_rate: Optional[float] = None
    effective_sample_size: Optional[Dict[str, float]] = None
    gelman_rubin: Optional[Dict[str, float]] = None
    
    # Timing
    elapsed_time: float = 0.0
    n_evaluations: int = 0
    
    # Configuration
    config: Optional[Dict[str, Any]] = None
    
    # Output paths
    output_dir: Optional[Path] = None
    
    def get_posteriors(
        self,
        parameters: Optional[List[str]] = None,
        burn_fraction: float = 0.2,
    ) -> pd.DataFrame:
        """
        Get posterior samples.
        
        Args:
            parameters: List of parameters to include (None = all)
            burn_fraction: Fraction of samples to discard as burn-in
        
        Returns:
            DataFrame with posterior samples
        """
        if self.smc_samples is not None:
            samples = self.smc_samples
        elif self.ga_samples is not None:
            samples = self.ga_samples
        else:
            return pd.DataFrame()
        
        # Apply burn-in
        n_burn = int(len(samples) * burn_fraction)
        samples = samples.iloc[n_burn:]
        
        # Select parameters
        if parameters:
            cols = [c for c in samples.columns if any(p in c for p in parameters)]
            samples = samples[cols]
        
        return samples
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for all parameters.
        
        Returns:
            Dictionary with mean, std, median, and quantiles for each parameter
        """
        posteriors = self.get_posteriors()
        
        stats = {}
        for col in posteriors.columns:
            if col in ['fitness', 'loss', 'generation', 'stage', 'particle']:
                continue
            
            values = posteriors[col].dropna()
            if len(values) == 0:
                continue
            
            stats[col] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'median': float(values.median()),
                'q16': float(values.quantile(0.16)),
                'q84': float(values.quantile(0.84)),
                'q05': float(values.quantile(0.05)),
                'q95': float(values.quantile(0.95)),
            }
        
        return stats
    
    def compute_credible_intervals(
        self,
        confidence: float = 0.68
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute credible intervals for all parameters.
        
        Args:
            confidence: Confidence level (default 68% = 1 sigma)
        
        Returns:
            Dictionary mapping parameter names to (lower, upper) bounds
        """
        posteriors = self.get_posteriors()
        
        alpha = (1 - confidence) / 2
        intervals = {}
        
        for col in posteriors.columns:
            if col in ['fitness', 'loss', 'generation', 'stage', 'particle']:
                continue
            
            values = posteriors[col].dropna()
            if len(values) == 0:
                continue
            
            lower = float(values.quantile(alpha))
            upper = float(values.quantile(1 - alpha))
            intervals[col] = (lower, upper)
        
        return intervals
    
    def compute_correlations(self) -> pd.DataFrame:
        """
        Compute correlation matrix for parameters.
        
        Returns:
            Correlation matrix as DataFrame
        """
        posteriors = self.get_posteriors()
        
        # Filter to numeric columns only
        numeric_cols = posteriors.select_dtypes(include=[np.number]).columns
        param_cols = [c for c in numeric_cols 
                      if c not in ['fitness', 'loss', 'generation', 'stage', 'particle']]
        
        return posteriors[param_cols].corr()
    
    def make_corner_plot(
        self,
        parameters: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Optional[Any]:
        """
        Generate corner plot of posteriors.
        
        Args:
            parameters: Parameters to include
            output_path: Path to save figure
            **kwargs: Additional arguments for corner.corner
        
        Returns:
            Figure object (or None if corner not available)
        """
        if not HAS_CORNER:
            print("Warning: corner package not installed. Install with: pip install corner")
            return None
        
        import matplotlib.pyplot as plt
        
        posteriors = self.get_posteriors(parameters=parameters)
        
        # Filter to numeric columns
        numeric_cols = posteriors.select_dtypes(include=[np.number]).columns
        param_cols = [c for c in numeric_cols 
                      if c not in ['fitness', 'loss', 'generation', 'stage', 'particle']]
        
        data = posteriors[param_cols].values
        labels = param_cols
        
        # Default corner plot settings
        default_kwargs = {
            'labels': labels,
            'show_titles': True,
            'title_fmt': '.3f',
            'quantiles': [0.16, 0.50, 0.84],
            'color': 'black',
            'plot_datapoints': False,
            'fill_contours': True,
            'hist_kwargs': {'histtype': 'stepfilled', 'alpha': 0.35, 'edgecolor': 'black'},
            'contour_kwargs': {'linewidths': 1.0},
            'contourf_kwargs': {'cmap': 'Greys'},
        }
        default_kwargs.update(kwargs)
        
        fig = corner.corner(data, **default_kwargs)
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved corner plot to {output_path}")
        
        return fig
    
    def save(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Save all results to output directory.
        
        Args:
            output_dir: Output directory (uses self.output_dir if None)
        """
        if output_dir is not None:
            output_dir = Path(output_dir)
        elif self.output_dir is not None:
            output_dir = self.output_dir
        else:
            raise ValueError("No output directory specified")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save GA samples
        if self.ga_samples is not None:
            path = output_dir / 'ga_samples.csv'
            self.ga_samples.to_csv(path, index=False)
            print(f"Saved GA samples to {path}")
        
        # Save SMC samples
        if self.smc_samples is not None:
            path = output_dir / 'smc_samples.csv'
            self.smc_samples.to_csv(path, index=False)
            print(f"Saved SMC samples to {path}")
        
        # Save SMC chains
        if self.smc_chains is not None:
            path = output_dir / 'smc_chains.csv'
            self.smc_chains.to_csv(path, index=False)
            print(f"Saved SMC chains to {path}")
        
        # Save summary statistics
        stats = self.get_summary_statistics()
        path = output_dir / 'summary_statistics.json'
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved summary statistics to {path}")
        
        # Save best fit
        if self.best_params is not None:
            path = output_dir / 'best_fit.json'
            result = {
                'parameters': self.best_params,
                'fitness': self.best_fitness,
            }
            with open(path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Saved best fit to {path}")
        
        # Generate corner plot
        if HAS_CORNER:
            try:
                self.make_corner_plot(output_path=str(output_dir / 'corner_plot.png'))
            except Exception as e:
                print(f"Warning: Could not generate corner plot: {e}")
    
    @classmethod
    def load(cls, output_dir: Union[str, Path]) -> InferenceResults:
        """
        Load results from output directory.
        
        Args:
            output_dir: Directory containing saved results
        
        Returns:
            InferenceResults object
        """
        output_dir = Path(output_dir)
        
        results = cls(parameter_names=[])
        results.output_dir = output_dir
        
        # Load GA samples
        ga_path = output_dir / 'ga_samples.csv'
        if ga_path.exists():
            results.ga_samples = pd.read_csv(ga_path)
        
        # Load SMC samples
        smc_path = output_dir / 'smc_samples.csv'
        if smc_path.exists():
            results.smc_samples = pd.read_csv(smc_path)
        
        # Load SMC chains
        chains_path = output_dir / 'smc_chains.csv'
        if chains_path.exists():
            results.smc_chains = pd.read_csv(chains_path)
        
        # Load best fit
        best_path = output_dir / 'best_fit.json'
        if best_path.exists():
            with open(best_path, 'r') as f:
                data = json.load(f)
            results.best_params = data.get('parameters')
            results.best_fitness = data.get('fitness')
        
        # Infer parameter names from samples
        samples = results.smc_samples if results.smc_samples is not None else results.ga_samples
        if samples is not None:
            param_cols = [c for c in samples.columns 
                         if c not in ['fitness', 'loss', 'generation', 'stage', 'particle', 'accepted', 'beta']]
            results.parameter_names = param_cols
        
        return results
    
    def summary(self) -> str:
        """Return a text summary of results."""
        lines = [
            "=" * 60,
            "MESA_INFER RESULTS SUMMARY",
            "=" * 60,
            "",
            f"Parameters: {len(self.parameter_names)}",
            f"Best fitness: {self.best_fitness:.6f}" if self.best_fitness else "Best fitness: N/A",
            f"Total evaluations: {self.n_evaluations}",
            f"Elapsed time: {self.elapsed_time:.1f}s",
            "",
        ]
        
        if self.best_params:
            lines.append("Best-fit parameters:")
            for name, value in self.best_params.items():
                lines.append(f"  {name}: {value}")
            lines.append("")
        
        stats = self.get_summary_statistics()
        if stats:
            lines.append("Posterior summary (median ± 1σ):")
            for name, s in stats.items():
                median = s['median']
                lower = s['q16']
                upper = s['q84']
                lines.append(f"  {name}: {median:.4f} (+{upper-median:.4f} / -{median-lower:.4f})")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
