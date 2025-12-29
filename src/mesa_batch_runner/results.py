"""
Results collection and analysis for MESA batch runs.
"""

from __future__ import annotations
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from mesa_reader import MesaData
    HAS_MESA_READER = True
except ImportError:
    HAS_MESA_READER = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class RunData:
    """Data from a single MESA run."""
    
    name: str
    directory: Path
    parameters: dict[str, Any]
    has_history: bool = False
    has_profiles: bool = False
    completed: bool = False
    
    @property
    def history_path(self) -> Path:
        return self.directory / "LOGS" / "history.data"
    
    @property
    def logs_dir(self) -> Path:
        return self.directory / "LOGS"
    
    def load_history(self) -> Any:
        """Load history data using mesa_reader."""
        if not HAS_MESA_READER:
            raise ImportError("mesa_reader is required for loading history data")
        
        if not self.has_history:
            raise FileNotFoundError(f"No history.data found in {self.logs_dir}")
        
        return MesaData(str(self.history_path))
    
    def get_profile_numbers(self) -> list[int]:
        """Get list of available profile numbers."""
        profiles = []
        for path in self.logs_dir.glob("profile*.data"):
            match = re.search(r"profile(\d+)\.data", path.name)
            if match:
                profiles.append(int(match.group(1)))
        return sorted(profiles)
    
    def load_profile(self, number: int) -> Any:
        """Load a specific profile."""
        if not HAS_MESA_READER:
            raise ImportError("mesa_reader is required for loading profile data")
        
        profile_path = self.logs_dir / f"profile{number}.data"
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile {number} not found")
        
        return MesaData(str(profile_path))


class ResultsCollector:
    """
    Collects and analyzes results from MESA batch runs.
    
    Provides methods to extract data at specific evolutionary stages
    (e.g., ZAMS, TAMS) and compile summary statistics.
    """
    
    def __init__(self, runs_dir: str | Path):
        """
        Initialize collector with the runs directory.
        
        Args:
            runs_dir: Path to directory containing run subdirectories
        """
        self.runs_dir = Path(runs_dir)
        if not self.runs_dir.exists():
            raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
        
        self._runs: list[RunData] = []
        self._scan_runs()
    
    def _scan_runs(self) -> None:
        """Scan for completed runs."""
        for path in sorted(self.runs_dir.iterdir()):
            if not path.is_dir():
                continue
            
            # Extract parameters from run name or inlist
            parameters = self._extract_parameters(path)
            
            # Check for outputs
            history_exists = (path / "LOGS" / "history.data").exists()
            profiles_exist = any((path / "LOGS").glob("profile*.data"))
            
            # Check completion status
            completed = self._check_completion(path)
            
            run_data = RunData(
                name=path.name,
                directory=path,
                parameters=parameters,
                has_history=history_exists,
                has_profiles=profiles_exist,
                completed=completed,
            )
            self._runs.append(run_data)
    
    def _extract_parameters(self, run_dir: Path) -> dict[str, Any]:
        """Extract parameters from inlist file."""
        from .inlist import InlistParser
        
        inlist_path = run_dir / "inlist_project"
        if not inlist_path.exists():
            return {}
        
        try:
            parser = InlistParser(inlist_path)
            return parser.to_dict()
        except Exception:
            return {}
    
    def _check_completion(self, run_dir: Path) -> bool:
        """Check if a run completed successfully."""
        log_path = run_dir / "run.log"
        if not log_path.exists():
            return False
        
        try:
            content = log_path.read_text()
            # Common MESA termination indicators
            indicators = [
                "termination code:",
                "LOGS saved",
                "model saved",
            ]
            return any(ind in content for ind in indicators)
        except Exception:
            return False
    
    @property
    def runs(self) -> list[RunData]:
        """List of all runs."""
        return self._runs
    
    @property
    def completed_runs(self) -> list[RunData]:
        """List of completed runs with history data."""
        return [r for r in self._runs if r.completed and r.has_history]
    
    def __len__(self) -> int:
        return len(self._runs)
    
    def find_tams_index(self, history: Any, h1_limit: float = 0.001) -> int:
        """
        Find the model index at TAMS (Terminal Age Main Sequence).
        
        Args:
            history: MesaData history object
            h1_limit: Central H1 threshold for TAMS
            
        Returns:
            Index of TAMS model (-1 if using last model)
        """
        if not HAS_NUMPY:
            raise ImportError("numpy is required for TAMS detection")
        
        if not hasattr(history, "center_h1"):
            return -1
        
        tams_indices = np.where(history.center_h1 <= h1_limit)[0]
        if len(tams_indices) > 0:
            return tams_indices[0]
        return -1
    
    def extract_at_tams(
        self,
        columns: list[str] | None = None,
        h1_limit: float = 0.001,
    ) -> list[dict[str, Any]]:
        """
        Extract values at TAMS for all completed runs.
        
        Args:
            columns: History columns to extract (defaults to common set)
            h1_limit: Central H1 threshold for TAMS
            
        Returns:
            List of dictionaries with parameter and output values
        """
        if columns is None:
            columns = [
                "star_age",
                "log_Teff",
                "log_L",
                "log_R",
                "he_core_mass",
                "center_h1",
            ]
        
        results = []
        
        for run in self.completed_runs:
            try:
                history = run.load_history()
                tams_idx = self.find_tams_index(history, h1_limit)
                
                row = {"run_name": run.name}
                row.update(run.parameters)
                
                for col in columns:
                    if hasattr(history, col):
                        data = getattr(history, col)
                        row[col] = data[tams_idx]
                    else:
                        row[col] = None
                
                # Convert age to Myr
                if "star_age" in row and row["star_age"] is not None:
                    row["age_myr"] = row["star_age"] / 1e6
                
                results.append(row)
                
            except Exception as e:
                print(f"Warning: Could not process {run.name}: {e}")
        
        return results
    
    def to_csv(
        self,
        output_path: str | Path,
        columns: list[str] | None = None,
        stage: str = "tams",
    ) -> None:
        """
        Export results to CSV.
        
        Args:
            output_path: Path for output CSV
            columns: Columns to include
            stage: Evolutionary stage ('tams' supported)
        """
        if stage == "tams":
            data = self.extract_at_tams(columns)
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        if not data:
            print("No data to export")
            return
        
        # Get all column names
        all_columns = set()
        for row in data:
            all_columns.update(row.keys())
        
        # Order columns sensibly
        ordered_columns = ["run_name"]
        param_cols = [c for c in all_columns if c.startswith("initial_") or c.startswith("overshoot_")]
        output_cols = [c for c in all_columns if c not in ordered_columns and c not in param_cols]
        ordered_columns.extend(sorted(param_cols))
        ordered_columns.extend(sorted(output_cols))
        
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ordered_columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)
        
        print(f"Results saved to: {output_path}")
    
    def summary(self) -> str:
        """Return a summary of collected results."""
        n_total = len(self._runs)
        n_completed = len([r for r in self._runs if r.completed])
        n_with_history = len([r for r in self._runs if r.has_history])
        
        lines = [
            f"ResultsCollector: {self.runs_dir}",
            f"  Total runs: {n_total}",
            f"  Completed: {n_completed}",
            f"  With history: {n_with_history}",
        ]
        
        return "\n".join(lines)
