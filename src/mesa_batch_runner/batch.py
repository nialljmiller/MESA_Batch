"""
Main BatchRunner class for MESA batch simulations.
"""

from __future__ import annotations
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from .grid import ParameterGrid, Parameter
from .runner import MESARunner, RunResult
from .results import ResultsCollector


@dataclass
class BatchRunner:
    """
    Main interface for running MESA parameter sweeps.
    
    Supports both explicit parameter values and range-based sweeps.
    
    Example:
        runner = BatchRunner("/path/to/mesa/work")
        runner.add_parameter("initial_mass", values=[1.0, 2.0, 5.0])
        runner.add_parameter("initial_z", min=0.001, max=0.02, steps=5)
        results = runner.run()
    """
    
    work_dir: str | Path
    output_dir: str | Path | None = None
    name: str | None = None
    
    # Run options
    timeout: float | None = None
    continue_on_error: bool = True
    dry_run: bool = False
    quiet: bool = False
    
    # Internal state
    _grid: ParameterGrid = field(default_factory=ParameterGrid, repr=False)
    _results: list[RunResult] = field(default_factory=list, repr=False)
    
    def __post_init__(self):
        self.work_dir = Path(self.work_dir).resolve()
        
        if not self.work_dir.exists():
            raise FileNotFoundError(f"MESA work directory not found: {self.work_dir}")
        
        # Set default output directory
        if self.output_dir is None:
            self.output_dir = self.work_dir / "batch_runs"
        else:
            self.output_dir = Path(self.output_dir).resolve()
        
        # Set default name
        if self.name is None:
            self.name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create output structure
        self._runs_dir = self.output_dir / "runs"
        self._logs_dir = self.output_dir / "logs"
    
    def add_parameter(
        self,
        name: str,
        values: Sequence[Any] | None = None,
        min: float | None = None,
        max: float | None = None,
        steps: int | None = None,
        step_size: float | None = None,
        log_scale: bool = False,
    ) -> BatchRunner:
        """
        Add a parameter to sweep over.
        
        Can specify either explicit values OR a range (min/max with steps/step_size).
        
        Args:
            name: Parameter name as it appears in MESA inlist
            values: Explicit list of values
            min: Minimum value for range
            max: Maximum value for range
            steps: Number of steps (inclusive of endpoints)
            step_size: Size of each step
            log_scale: Use logarithmic spacing
            
        Returns:
            Self for method chaining
        """
        self._grid.add(
            name=name,
            values=values,
            min_val=min,
            max_val=max,
            steps=steps,
            step_size=step_size,
            log_scale=log_scale,
        )
        return self
    
    def remove_parameter(self, name: str) -> BatchRunner:
        """Remove a parameter from the sweep."""
        self._grid.remove(name)
        return self
    
    def clear_parameters(self) -> BatchRunner:
        """Clear all parameters."""
        self._grid = ParameterGrid()
        return self
    
    @property
    def n_runs(self) -> int:
        """Total number of runs to execute."""
        return len(self._grid)
    
    @property
    def parameter_names(self) -> list[str]:
        """List of parameter names being swept."""
        return self._grid.parameter_names
    
    def summary(self) -> str:
        """Return a summary of the batch configuration."""
        lines = [
            f"BatchRunner: {self.name}",
            f"  Work directory: {self.work_dir}",
            f"  Output directory: {self.output_dir}",
            f"  Total runs: {self.n_runs}",
            "",
            self._grid.summary(),
        ]
        return "\n".join(lines)
    
    def run(
        self,
        force: bool = False,
        max_runs: int | None = None,
    ) -> list[RunResult]:
        """
        Execute all parameter combinations.
        
        Args:
            force: Skip confirmation prompt
            max_runs: Maximum number of runs (for testing)
            
        Returns:
            List of RunResult objects
        """
        if self.n_runs == 0:
            print("No parameters configured. Nothing to run.")
            return []
        
        # Confirmation
        if not force and not self.dry_run and not self.quiet:
            print(self.summary())
            print()
            response = input(f"Run {self.n_runs} MESA simulations? (yes/no): ")
            if not response.lower().startswith("y"):
                print("Batch run cancelled.")
                return []
        
        # Create output directories
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize runner
        mesa_runner = MESARunner(
            work_dir=self.work_dir,
            output_base_dir=self._runs_dir,
            quiet=self.quiet,
        )
        
        # Track results
        self._results = []
        timings = []
        
        # Progress reporting
        def on_start(name: str, params: dict):
            if not self.quiet:
                print(f"[{len(self._results) + 1}/{n_total}] Starting {name}...")
        
        def on_complete(result: RunResult):
            status = "✓" if result.success else "✗"
            if not self.quiet:
                print(f"  {status} Completed in {result.runtime_seconds:.1f}s")
        
        mesa_runner.on_run_start = on_start
        mesa_runner.on_run_complete = on_complete
        
        # Execute runs
        n_total = min(self.n_runs, max_runs) if max_runs else self.n_runs
        
        for i, params in enumerate(self._grid):
            if max_runs and i >= max_runs:
                break
            
            # Generate run name
            run_name = mesa_runner.generate_run_name(params)
            
            if self.dry_run:
                if not self.quiet:
                    print(f"[{i + 1}/{n_total}] Would run: {run_name}")
                    print(f"  Parameters: {params}")
                continue
            
            # Execute run
            try:
                result = mesa_runner.run(
                    name=run_name,
                    parameters=params,
                    timeout=self.timeout,
                )
                self._results.append(result)
                timings.append({
                    "run_name": run_name,
                    "runtime_seconds": result.runtime_seconds,
                    "status": "completed" if result.success else "failed",
                })
                
            except Exception as e:
                if not self.quiet:
                    print(f"  Error: {e}")
                if not self.continue_on_error:
                    raise
        
        # Save timing data
        if not self.dry_run:
            self._save_timings(timings)
            self._save_summary()
        
        # Final summary
        if not self.quiet and not self.dry_run:
            n_success = sum(1 for r in self._results if r.success)
            n_failed = len(self._results) - n_success
            print()
            print(f"Batch complete: {n_success} succeeded, {n_failed} failed")
            print(f"Results saved to: {self.output_dir}")
        
        return self._results
    
    def _save_timings(self, timings: list[dict]) -> None:
        """Save run timing data to CSV."""
        timing_file = self.output_dir / "run_timings.csv"
        
        with open(timing_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["run_name", "runtime_seconds", "status"])
            writer.writeheader()
            writer.writerows(timings)
    
    def _save_summary(self) -> None:
        """Save batch summary to JSON."""
        summary = {
            "name": self.name,
            "work_dir": str(self.work_dir),
            "output_dir": str(self.output_dir),
            "timestamp": datetime.now().isoformat(),
            "n_runs": len(self._results),
            "n_success": sum(1 for r in self._results if r.success),
            "parameters": self._grid.parameter_names,
            "runs": [
                {
                    "name": r.name,
                    "parameters": r.parameters,
                    "success": r.success,
                    "runtime_seconds": r.runtime_seconds,
                }
                for r in self._results
            ],
        }
        
        summary_file = self.output_dir / "batch_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
    
    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> BatchRunner:
        """
        Create a BatchRunner from a configuration dictionary.
        
        Args:
            config: Dictionary with 'work_dir', 'parameters', and optional settings
            
        Returns:
            Configured BatchRunner instance
        """
        runner = cls(
            work_dir=config["work_dir"],
            output_dir=config.get("output_dir"),
            name=config.get("name"),
            timeout=config.get("timeout"),
            continue_on_error=config.get("continue_on_error", True),
            quiet=config.get("quiet", False),
        )
        
        # Add parameters
        for param in config.get("parameters", []):
            runner.add_parameter(**param)
        
        return runner
    
    def get_results(self) -> list[RunResult]:
        """Return results from the last run."""
        return self._results
    
    def collect_results(self) -> ResultsCollector:
        """Create a ResultsCollector for analyzing outputs."""
        return ResultsCollector(self._runs_dir)
