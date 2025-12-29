"""
MESA execution and run management.
"""

from __future__ import annotations
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .inlist import InlistModifier, InlistParser, DEFAULT_INLIST_TEMPLATE


@dataclass
class RunResult:
    """Result of a single MESA run."""
    
    name: str
    parameters: dict[str, Any]
    output_dir: Path
    runtime_seconds: float
    return_code: int
    success: bool
    error_message: str | None = None
    
    @property
    def runtime_minutes(self) -> float:
        return self.runtime_seconds / 60.0
    
    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "LOGS"
    
    @property
    def photos_dir(self) -> Path:
        return self.output_dir / "photos"
    
    def has_history(self) -> bool:
        return (self.logs_dir / "history.data").exists()


@dataclass
class MESARunner:
    """
    Handles execution of individual MESA runs.
    
    Manages the work directory, inlist modification, execution, and output collection.
    """
    
    work_dir: Path
    output_base_dir: Path
    inlist_template: str | None = None
    run_script: str = "rn"
    quiet: bool = False
    
    # Callbacks for progress reporting
    on_run_start: Callable[[str, dict], None] | None = None
    on_run_complete: Callable[[RunResult], None] | None = None
    
    def __post_init__(self):
        self.work_dir = Path(self.work_dir).resolve()
        self.output_base_dir = Path(self.output_base_dir).resolve()
        
        # Validate work directory
        if not self.work_dir.exists():
            raise FileNotFoundError(f"Work directory not found: {self.work_dir}")
        
        # Find the run script
        self._run_script_path = self._find_run_script()
        
        # Find or create inlist template
        self._inlist_content = self._get_inlist_template()
    
    def _find_run_script(self) -> Path:
        """Find the MESA run script in the work directory."""
        # Check for common run script names
        for name in [self.run_script, "rn", "star", "./rn", "./star"]:
            path = self.work_dir / name
            if path.exists() and os.access(path, os.X_OK):
                return path
        
        # Check if 'star' executable exists
        star_path = self.work_dir / "star"
        if star_path.exists():
            return star_path
        
        raise FileNotFoundError(
            f"No run script found in {self.work_dir}. "
            f"Looking for: {self.run_script}, rn, star"
        )
    
    def _get_inlist_template(self) -> str:
        """Get the inlist template content."""
        if self.inlist_template:
            return self.inlist_template
        
        # Look for existing inlist files
        for name in ["inlist_project", "inlist"]:
            path = self.work_dir / name
            if path.exists():
                return path.read_text()
        
        # Use default template
        return DEFAULT_INLIST_TEMPLATE
    
    def run(
        self,
        name: str,
        parameters: dict[str, Any],
        timeout: float | None = None,
    ) -> RunResult:
        """
        Execute a single MESA run with the given parameters.
        
        Args:
            name: Unique name for this run (used for output directory)
            parameters: Dictionary of parameter names to values
            timeout: Optional timeout in seconds
            
        Returns:
            RunResult with execution details
        """
        output_dir = self.output_base_dir / name
        
        # Report start
        if self.on_run_start:
            self.on_run_start(name, parameters)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "LOGS").mkdir(exist_ok=True)
        (output_dir / "photos").mkdir(exist_ok=True)
        
        # Create modified inlist
        modifier = InlistModifier(content=self._inlist_content)
        
        # Apply parameters
        for param_name, value in parameters.items():
            # Handle array parameters (e.g., "overshoot_f(1)")
            if "(" in param_name:
                base_name = param_name.split("(")[0]
                index = int(param_name.split("(")[1].rstrip(")"))
                modifier.set_array_value(base_name, index, value)
            else:
                modifier.set_value(param_name, value)
        
        # Update save model filename
        modifier.set_value("save_model_filename", f"{name}.mod", section="star_job")
        
        # Write inlist to work directory
        inlist_path = self.work_dir / "inlist_project"
        modifier.save(inlist_path)
        
        # Also save a copy to output directory
        shutil.copy(inlist_path, output_dir / "inlist_project")
        
        # Run MESA
        start_time = time.time()
        log_path = output_dir / "run.log"
        
        try:
            with open(log_path, "w") as log_file:
                result = subprocess.run(
                    [str(self._run_script_path)],
                    cwd=str(self.work_dir),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            return_code = result.returncode
            success = return_code == 0
            error_message = None
        except subprocess.TimeoutExpired:
            return_code = -1
            success = False
            error_message = f"Run timed out after {timeout} seconds"
        except Exception as e:
            return_code = -1
            success = False
            error_message = str(e)
        
        runtime = time.time() - start_time
        
        # Copy output files
        self._collect_outputs(output_dir, name)
        
        # Create result
        run_result = RunResult(
            name=name,
            parameters=parameters,
            output_dir=output_dir,
            runtime_seconds=runtime,
            return_code=return_code,
            success=success,
            error_message=error_message,
        )
        
        # Report completion
        if self.on_run_complete:
            self.on_run_complete(run_result)
        
        return run_result
    
    def _collect_outputs(self, output_dir: Path, run_name: str) -> None:
        """Copy MESA output files to the run's output directory."""
        # Copy LOGS
        logs_src = self.work_dir / "LOGS"
        logs_dst = output_dir / "LOGS"
        if logs_src.exists():
            for item in logs_src.iterdir():
                if item.is_file():
                    shutil.copy2(item, logs_dst / item.name)
        
        # Copy photos
        photos_src = self.work_dir / "photos"
        photos_dst = output_dir / "photos"
        if photos_src.exists():
            for item in photos_src.iterdir():
                if item.is_file():
                    shutil.copy2(item, photos_dst / item.name)
        
        # Copy model file if it exists
        model_file = self.work_dir / f"{run_name}.mod"
        if model_file.exists():
            shutil.copy2(model_file, output_dir / model_file.name)
        
        # Copy any .mod file (fallback)
        for mod_file in self.work_dir.glob("*.mod"):
            if not (output_dir / mod_file.name).exists():
                shutil.copy2(mod_file, output_dir / mod_file.name)
    
    def generate_run_name(self, parameters: dict[str, Any]) -> str:
        """
        Generate a unique run name from parameters.
        
        Args:
            parameters: Dictionary of parameters
            
        Returns:
            A descriptive name string
        """
        parts = []
        for key, value in sorted(parameters.items()):
            # Shorten common parameter names
            short_key = key.replace("initial_", "").replace("overshoot_", "ov_")
            
            # Format value
            if isinstance(value, float):
                if abs(value) < 0.01 or abs(value) >= 1000:
                    val_str = f"{value:.2e}"
                else:
                    val_str = f"{value:.4g}"
            else:
                val_str = str(value)
            
            parts.append(f"{short_key}{val_str}")
        
        return "_".join(parts)
