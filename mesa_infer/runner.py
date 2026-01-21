"""
MESA Runner for MESA_infer.

Handles MESA execution including:
- Inlist generation from parameter values
- Subprocess execution with timeout
- Output parsing and failure detection
- HPC job submission (SLURM, PBS)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

from .inlist_parser import InlistParser, ParameterSpec
from .config import MESAConfig

logger = logging.getLogger(__name__)


@dataclass
class MESARunResult:
    """Result from a MESA run."""
    success: bool
    run_dir: Path
    params: Dict[str, Any]
    
    # Output files
    logs_dir: Optional[Path] = None
    history_file: Optional[Path] = None
    profile_files: List[Path] = field(default_factory=list)
    final_model: Optional[Path] = None
    
    # Extracted data
    final_age: Optional[float] = None
    final_mass: Optional[float] = None
    final_radius: Optional[float] = None
    final_teff: Optional[float] = None
    final_logg: Optional[float] = None
    final_luminosity: Optional[float] = None
    
    # Execution info
    elapsed_time: float = 0.0
    termination_code: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'run_dir': str(self.run_dir),
            'params': self.params,
            'final_age': self.final_age,
            'final_mass': self.final_mass,
            'final_radius': self.final_radius,
            'final_teff': self.final_teff,
            'final_logg': self.final_logg,
            'final_luminosity': self.final_luminosity,
            'elapsed_time': self.elapsed_time,
            'termination_code': self.termination_code,
            'error_message': self.error_message,
        }


class MESARunner:
    """
    Execute MESA stellar evolution models.
    
    Handles the full workflow of:
    1. Setting up a work directory
    2. Generating inlists with parameter values
    3. Running MESA (locally or via HPC scheduler)
    4. Parsing outputs and detecting failures
    5. Extracting relevant physical quantities
    
    Example:
        runner = MESARunner(
            mesa_dir="/path/to/mesa",
            work_dir="/path/to/work",
            inlist_template="inlist_project"
        )
        
        result = runner.run({"initial_mass": 1.0, "initial_z": 0.02})
    """
    
    # Common MESA termination codes
    TERMINATION_CODES = {
        'xa_central_lower_limit': 'Central abundance limit reached',
        'max_age': 'Maximum age reached',
        'max_model_number': 'Maximum model number reached',
        'Lnuc_div_L_zams_limit': 'Nuclear luminosity limit',
        'log_L_upper_limit': 'Luminosity upper limit',
        'log_Teff_lower_limit': 'Teff lower limit',
        'gamma_center_limit': 'Central gamma limit',
        'eta_center_limit': 'Central eta limit',
        'min_timestep_limit': 'Minimum timestep limit',
        'fe_core_infall_limit': 'Iron core infall limit',
    }
    
    def __init__(
        self,
        mesa_dir: Optional[Union[str, Path]] = None,
        work_dir: Optional[Union[str, Path]] = None,
        inlist_template: str = "inlist_project",
        config: Optional[MESAConfig] = None,
        output_dir: Optional[Union[str, Path]] = None,
        keep_runs: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize the MESA runner.
        
        Args:
            mesa_dir: Path to MESA installation (or use MESA_DIR env var)
            work_dir: Path to MESA work directory
            inlist_template: Name of inlist template file
            config: MESAConfig object
            output_dir: Directory to store run outputs
            keep_runs: Keep individual run directories
            verbose: Print progress
        """
        # Use config if provided
        if config:
            mesa_dir = mesa_dir or config.mesa_dir
            work_dir = work_dir or config.work_dir
            self.timeout = config.timeout
            self.retry_on_failure = config.retry_on_failure
            self.max_retries = config.max_retries
            self.keep_logs = config.keep_logs
            self.keep_photos = config.keep_photos
            self.keep_models = config.keep_models
            self.scheduler = config.scheduler
            self.slurm_partition = config.slurm_partition
            self.slurm_time = config.slurm_time
            self.slurm_memory = config.slurm_memory
        else:
            self.timeout = 7200
            self.retry_on_failure = True
            self.max_retries = 3
            self.keep_logs = True
            self.keep_photos = False
            self.keep_models = True
            self.scheduler = "local"
            self.slurm_partition = None
            self.slurm_time = "02:00:00"
            self.slurm_memory = "4G"
        
        # Resolve MESA_DIR
        if mesa_dir is None:
            mesa_dir = os.environ.get("MESA_DIR")
            if mesa_dir is None:
                raise ValueError(
                    "MESA_DIR must be specified or set as environment variable"
                )
        
        self.mesa_dir = Path(mesa_dir)
        if not self.mesa_dir.exists():
            raise ValueError(f"MESA_DIR does not exist: {self.mesa_dir}")
        
        # Resolve work directory
        if work_dir is None:
            raise ValueError("work_dir must be specified")
        
        self.work_dir = Path(work_dir)
        if not self.work_dir.exists():
            raise ValueError(f"work_dir does not exist: {self.work_dir}")
        
        self.inlist_template = inlist_template
        self.output_dir = Path(output_dir) if output_dir else self.work_dir / "runs"
        self.keep_runs = keep_runs
        self.verbose = verbose
        
        # Load and parse template inlist
        self.template_path = self.work_dir / self.inlist_template
        if not self.template_path.exists():
            self.template_path = self.work_dir / f"{self.inlist_template}.txt"
        
        if self.template_path.exists():
            self.inlist_parser = InlistParser(self.template_path)
            self.template_content = self.template_path.read_text()
        else:
            logger.warning(f"Inlist template not found: {self.template_path}")
            self.inlist_parser = None
            self.template_content = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track runs
        self.run_count = 0
        self.successful_runs = 0
        self.failed_runs = 0
    
    def run(
        self,
        params: Dict[str, Any],
        run_id: Optional[str] = None,
    ) -> MESARunResult:
        """
        Run MESA with the given parameters.
        
        Args:
            params: Dictionary of parameter values
            run_id: Optional identifier for this run
        
        Returns:
            MESARunResult with success status and extracted data
        """
        self.run_count += 1
        
        if run_id is None:
            run_id = f"run_{self.run_count:06d}"
        
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        try:
            # Setup run directory
            self._setup_run_dir(run_dir, params)
            
            # Execute MESA
            success, term_code, error = self._execute_mesa(run_dir)
            
            # Parse outputs
            result = self._parse_outputs(run_dir, params, success, term_code, error)
            result.elapsed_time = time.time() - start_time
            
            if result.success:
                self.successful_runs += 1
            else:
                self.failed_runs += 1
            
            # Cleanup if not keeping runs
            if not self.keep_runs and result.success:
                self._cleanup_run(run_dir)
            
            return result
            
        except Exception as e:
            logger.error(f"Run {run_id} failed with exception: {e}")
            self.failed_runs += 1
            return MESARunResult(
                success=False,
                run_dir=run_dir,
                params=params,
                error_message=str(e),
                elapsed_time=time.time() - start_time,
            )
    
    def _setup_run_dir(self, run_dir: Path, params: Dict[str, Any]) -> None:
        """Setup the run directory with inlist and necessary files."""
        # Copy essential files from work directory
        for item in ['make', 'src', 'mk']:
            src = self.work_dir / item
            dst = run_dir / item
            if src.exists() and not dst.exists():
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
        
        # Generate modified inlist
        if self.template_content and self.inlist_parser:
            inlist_content = self._generate_inlist(params)
        else:
            inlist_content = self._generate_inlist_basic(params)
        
        # Write inlist
        inlist_path = run_dir / self.inlist_template
        inlist_path.write_text(inlist_content)
        
        # Also write as inlist (MESA default)
        (run_dir / "inlist").write_text(inlist_content)
        
        # Create LOGS directory
        (run_dir / "LOGS").mkdir(exist_ok=True)
    
    def _generate_inlist(self, params: Dict[str, Any]) -> str:
        """Generate inlist content with parameter values."""
        content = self.template_content
        
        for name, value in params.items():
            formatted = self._format_fortran_value(value)
            
            # Try to replace existing parameter
            # Handle array notation like param(1)
            pattern = rf'^(\s*)({re.escape(name)})\s*=\s*[^\n]+'
            replacement = rf'\g<1>\g<2> = {formatted}'
            
            new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            
            if new_content == content:
                # Parameter not found - might need to add it
                # Try to find the appropriate section
                section = self._find_parameter_section(name)
                if section:
                    # Add parameter to section
                    section_pattern = rf'(&{section}.*?)(/ *!.*?end)'
                    match = re.search(section_pattern, content, re.DOTALL | re.IGNORECASE)
                    if match:
                        insert_pos = match.end(1)
                        param_line = f"\n    {name} = {formatted}"
                        content = content[:insert_pos] + param_line + content[insert_pos:]
                        continue
            else:
                content = new_content
        
        return content
    
    def _generate_inlist_basic(self, params: Dict[str, Any]) -> str:
        """Generate a basic inlist when no template exists."""
        lines = [
            "! MESA inlist generated by MESA_infer",
            "",
            "&star_job",
            "    save_model_when_terminate = .true.",
            "    save_model_filename = 'final.mod'",
            "    pgstar_flag = .false.",
            "/ ! end of star_job namelist",
            "",
            "&eos",
            "/ ! end of eos namelist",
            "",
            "&kap",
            "    Zbase = 0.02d0",
            "/ ! end of kap namelist",
            "",
            "&controls",
        ]
        
        # Add parameters
        for name, value in params.items():
            formatted = self._format_fortran_value(value)
            lines.append(f"    {name} = {formatted}")
        
        lines.extend([
            "",
            "/ ! end of controls namelist",
            "",
            "&pgstar",
            "/ ! end of pgstar namelist",
        ])
        
        return "\n".join(lines)
    
    def _format_fortran_value(self, value: Any) -> str:
        """Format Python value for Fortran."""
        if isinstance(value, bool):
            return ".true." if value else ".false."
        elif isinstance(value, str):
            return f"'{value}'"
        elif isinstance(value, float):
            formatted = f"{value:.10g}"
            if "e" in formatted.lower():
                formatted = formatted.replace("e", "d").replace("E", "d")
            elif "." not in formatted:
                formatted += "d0"
            return formatted
        elif isinstance(value, int):
            return str(value)
        else:
            return str(value)
    
    def _find_parameter_section(self, param_name: str) -> Optional[str]:
        """Determine which inlist section a parameter belongs to."""
        # Common parameter prefixes/patterns and their sections
        section_patterns = {
            'star_job': ['save_model', 'load_model', 'pgstar', 'history_', 'profile_'],
            'eos': ['eos_'],
            'kap': ['kap_', 'opacity_', 'Zbase'],
            'controls': ['initial_', 'mixing_', 'overshoot', 'mass_', 'xa_', 'max_', 'min_'],
            'pgstar': ['pgstar_', 'Grid_', 'HR_', 'TRho_', 'History_'],
        }
        
        for section, patterns in section_patterns.items():
            for pattern in patterns:
                if param_name.startswith(pattern) or param_name.lower().startswith(pattern.lower()):
                    return section
        
        # Default to controls
        return 'controls'
    
    def _execute_mesa(self, run_dir: Path) -> Tuple[bool, Optional[str], Optional[str]]:
        """Execute MESA in the run directory."""
        if self.scheduler == "local":
            return self._execute_local(run_dir)
        elif self.scheduler == "slurm":
            return self._execute_slurm(run_dir)
        elif self.scheduler == "pbs":
            return self._execute_pbs(run_dir)
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")
    
    def _execute_local(self, run_dir: Path) -> Tuple[bool, Optional[str], Optional[str]]:
        """Execute MESA locally."""
        # Set environment
        env = os.environ.copy()
        env['MESA_DIR'] = str(self.mesa_dir)
        
        # Run command
        star_exe = run_dir / "star"
        if not star_exe.exists():
            # Try to compile
            compile_result = subprocess.run(
                ["./mk"],
                cwd=run_dir,
                env=env,
                capture_output=True,
                timeout=300,
            )
            if compile_result.returncode != 0:
                return False, None, f"Compilation failed: {compile_result.stderr.decode()}"
        
        # Execute
        try:
            result = subprocess.run(
                ["./rn"],
                cwd=run_dir,
                env=env,
                capture_output=True,
                timeout=self.timeout,
            )
            
            stdout = result.stdout.decode()
            stderr = result.stderr.decode()
            
            # Check for success
            success = result.returncode == 0
            
            # Try to extract termination code
            term_code = None
            for code in self.TERMINATION_CODES:
                if code in stdout or code in stderr:
                    term_code = code
                    break
            
            # Check for common failure patterns
            if not success:
                error = stderr or stdout[-1000:] if stdout else "Unknown error"
            else:
                error = None
            
            return success, term_code, error
            
        except subprocess.TimeoutExpired:
            return False, None, f"Timeout after {self.timeout}s"
        except Exception as e:
            return False, None, str(e)
    
    def _execute_slurm(self, run_dir: Path) -> Tuple[bool, Optional[str], Optional[str]]:
        """Submit and wait for SLURM job."""
        # Write job script
        job_script = run_dir / "run.slurm"
        script_content = f"""#!/bin/bash
#SBATCH --job-name=mesa_infer
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time={self.slurm_time}
#SBATCH --mem={self.slurm_memory}
"""
        if self.slurm_partition:
            script_content += f"#SBATCH --partition={self.slurm_partition}\n"
        
        script_content += f"""
export MESA_DIR={self.mesa_dir}
cd {run_dir}
./rn
"""
        job_script.write_text(script_content)
        
        # Submit job
        result = subprocess.run(
            ["sbatch", str(job_script)],
            capture_output=True,
            cwd=run_dir,
        )
        
        if result.returncode != 0:
            return False, None, f"SLURM submit failed: {result.stderr.decode()}"
        
        # Extract job ID
        job_id = result.stdout.decode().strip().split()[-1]
        
        # Wait for job to complete
        while True:
            check = subprocess.run(
                ["squeue", "-j", job_id, "-h"],
                capture_output=True,
            )
            if not check.stdout.strip():
                break
            time.sleep(10)
        
        # Check result
        log_files = list(run_dir.glob("slurm_*.out"))
        if log_files:
            output = log_files[0].read_text()
            success = "termination" in output.lower() or "finished" in output.lower()
            return success, None, None if success else output[-500:]
        
        return False, None, "No SLURM output found"
    
    def _execute_pbs(self, run_dir: Path) -> Tuple[bool, Optional[str], Optional[str]]:
        """Submit and wait for PBS job."""
        # Similar to SLURM but with PBS syntax
        raise NotImplementedError("PBS support not yet implemented")
    
    def _parse_outputs(
        self,
        run_dir: Path,
        params: Dict[str, Any],
        success: bool,
        term_code: Optional[str],
        error: Optional[str],
    ) -> MESARunResult:
        """Parse MESA outputs and extract physical quantities."""
        result = MESARunResult(
            success=success,
            run_dir=run_dir,
            params=params,
            termination_code=term_code,
            error_message=error,
        )
        
        logs_dir = run_dir / "LOGS"
        if logs_dir.exists():
            result.logs_dir = logs_dir
            
            # Find history file
            history_file = logs_dir / "history.data"
            if history_file.exists():
                result.history_file = history_file
                self._parse_history(result)
            
            # Find profile files
            profile_files = sorted(logs_dir.glob("profile*.data"))
            result.profile_files = profile_files
        
        # Find final model
        for model_name in ["final.mod", "final_model.mod"]:
            model_path = run_dir / model_name
            if model_path.exists():
                result.final_model = model_path
                break
        
        return result
    
    def _parse_history(self, result: MESARunResult) -> None:
        """Parse history file to extract final stellar parameters."""
        try:
            # Read history file
            with open(result.history_file, 'r') as f:
                lines = f.readlines()
            
            # Find header line
            header_idx = None
            for i, line in enumerate(lines):
                if 'model_number' in line or 'star_age' in line:
                    header_idx = i
                    break
            
            if header_idx is None:
                return
            
            # Parse header
            header = lines[header_idx].split()
            
            # Get last data line
            data_line = lines[-1].split()
            
            # Create mapping
            data = {header[i]: float(data_line[i]) for i in range(min(len(header), len(data_line)))}
            
            # Extract quantities
            result.final_age = data.get('star_age')
            result.final_mass = data.get('star_mass')
            result.final_radius = data.get('radius') or (10 ** data.get('log_R', 0) if 'log_R' in data else None)
            result.final_teff = data.get('Teff') or (10 ** data.get('log_Teff', 0) if 'log_Teff' in data else None)
            
            # Calculate log g if we have mass and radius
            if result.final_mass and result.final_radius:
                # log g = log(GM/R^2) in cgs
                G = 6.674e-8  # cgs
                Msun = 1.989e33  # g
                Rsun = 6.96e10  # cm
                M_cgs = result.final_mass * Msun
                R_cgs = result.final_radius * Rsun
                result.final_logg = np.log10(G * M_cgs / R_cgs**2)
            
            result.final_luminosity = data.get('luminosity') or (10 ** data.get('log_L', 0) if 'log_L' in data else None)
            
        except Exception as e:
            logger.warning(f"Failed to parse history file: {e}")
    
    def _cleanup_run(self, run_dir: Path) -> None:
        """Clean up run directory, keeping only essential outputs."""
        if not self.keep_logs:
            logs_dir = run_dir / "LOGS"
            if logs_dir.exists():
                shutil.rmtree(logs_dir)
        
        # Remove source and build files
        for item in ['src', 'make', 'mk', 'build', '*.o', '*.mod']:
            for path in run_dir.glob(item):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()


# Import numpy for log10 calculation in _parse_history
import numpy as np
