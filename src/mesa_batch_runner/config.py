"""
Configuration file parsing for MESA batch runner.

Supports a Fortran-namelist-like configuration format for CLI usage.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ParameterSpec:
    """Specification for a parameter sweep."""
    
    name: str
    values: list[Any] | None = None
    min_val: float | None = None
    max_val: float | None = None
    steps: int | None = None
    step_size: float | None = None
    log_scale: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for BatchRunner.add_parameter()."""
        result = {"name": self.name}
        
        if self.values is not None:
            result["values"] = self.values
        else:
            result["min"] = self.min_val
            result["max"] = self.max_val
            if self.steps is not None:
                result["steps"] = self.steps
            if self.step_size is not None:
                result["step_size"] = self.step_size
            if self.log_scale:
                result["log_scale"] = True
        
        return result


@dataclass
class BatchConfig:
    """
    Configuration for a batch run.
    
    Can be loaded from a batch_inlist file or created programmatically.
    
    Example batch_inlist format:
    
        &batch_control
            work_dir = '/path/to/mesa/work'
            output_dir = '/path/to/output'
            name = 'my_batch'
            timeout = 3600
            continue_on_error = .true.
        /
        
        &parameters
            ! Explicit values
            initial_mass = 1.0, 2.0, 5.0, 10.0
            
            ! Range specification
            initial_z_min = 0.001
            initial_z_max = 0.02
            initial_z_steps = 5
            
            ! Log-spaced range
            mixing_length_alpha_min = 1.5
            mixing_length_alpha_max = 2.5
            mixing_length_alpha_step = 0.1
        /
    """
    
    work_dir: str | Path
    output_dir: str | Path | None = None
    name: str | None = None
    timeout: float | None = None
    continue_on_error: bool = True
    quiet: bool = False
    parameters: list[ParameterSpec] = field(default_factory=list)
    
    @classmethod
    def from_file(cls, filepath: str | Path) -> BatchConfig:
        """
        Load configuration from a batch_inlist file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            BatchConfig instance
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        content = filepath.read_text()
        return cls._parse(content)
    
    @classmethod
    def from_string(cls, content: str) -> BatchConfig:
        """Parse configuration from a string."""
        return cls._parse(content)
    
    @classmethod
    def _parse(cls, content: str) -> BatchConfig:
        """Parse the batch_inlist format."""
        # Extract sections
        batch_control = cls._extract_section(content, "batch_control")
        parameters_section = cls._extract_section(content, "parameters")
        
        # Parse batch_control
        work_dir = cls._get_string(batch_control, "work_dir")
        if work_dir is None:
            raise ValueError("work_dir is required in &batch_control")
        
        config = cls(
            work_dir=work_dir,
            output_dir=cls._get_string(batch_control, "output_dir"),
            name=cls._get_string(batch_control, "name"),
            timeout=cls._get_number(batch_control, "timeout"),
            continue_on_error=cls._get_logical(batch_control, "continue_on_error", True),
            quiet=cls._get_logical(batch_control, "quiet", False),
        )
        
        # Parse parameters
        if parameters_section:
            config.parameters = cls._parse_parameters(parameters_section)
        
        return config
    
    @classmethod
    def _extract_section(cls, content: str, section_name: str) -> str:
        """Extract content of a namelist section."""
        # Match section start, then content, then closing / at start of line
        pattern = rf"&{section_name}\s*(.*?)\n\s*/"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        return match.group(1) if match else ""
    
    @classmethod
    def _get_string(cls, content: str, key: str) -> str | None:
        """Extract a string value."""
        pattern = rf"^\s*{key}\s*=\s*'([^']*)'"
        match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
        return match.group(1) if match else None
    
    @classmethod
    def _get_number(cls, content: str, key: str) -> float | None:
        """Extract a numeric value."""
        pattern = rf"^\s*{key}\s*=\s*([+-]?\d*\.?\d+(?:[dDeE][+-]?\d+)?)"
        match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
        if match:
            value = match.group(1).lower().replace("d", "e")
            return float(value)
        return None
    
    @classmethod
    def _get_logical(cls, content: str, key: str, default: bool = False) -> bool:
        """Extract a logical value."""
        pattern = rf"^\s*{key}\s*=\s*\.(true|false)\."
        match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"
        return default
    
    @classmethod
    def _get_list(cls, content: str, key: str) -> list[Any] | None:
        """Extract a comma-separated list of values."""
        pattern = rf"^\s*{key}\s*=\s*(.+?)(?:\s*!.*)?$"
        match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
        if not match:
            return None
        
        values_str = match.group(1).strip()
        values = []
        
        for item in values_str.split(","):
            item = item.strip()
            if not item:
                continue
            
            # Try to parse as number
            try:
                item_clean = item.lower().replace("d", "e")
                if "." in item_clean or "e" in item_clean:
                    values.append(float(item_clean))
                else:
                    values.append(int(item))
            except ValueError:
                # Keep as string
                values.append(item.strip("'\""))
        
        return values if values else None
    
    @classmethod
    def _parse_parameters(cls, content: str) -> list[ParameterSpec]:
        """Parse the parameters section."""
        params: dict[str, ParameterSpec] = {}
        
        # Find all parameter assignments
        lines = content.split("\n")
        
        for line in lines:
            # Skip comments and empty lines
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            
            # Parse assignment
            if "=" not in line:
                continue
            
            key, value_part = line.split("=", 1)
            key = key.strip()
            value_part = value_part.split("!")[0].strip()  # Remove inline comments
            
            # Check for range specifiers
            if key.endswith("_min"):
                base_name = key[:-4]
                if base_name not in params:
                    params[base_name] = ParameterSpec(name=base_name)
                params[base_name].min_val = cls._parse_value(value_part)
                
            elif key.endswith("_max"):
                base_name = key[:-4]
                if base_name not in params:
                    params[base_name] = ParameterSpec(name=base_name)
                params[base_name].max_val = cls._parse_value(value_part)
                
            elif key.endswith("_steps"):
                base_name = key[:-6]
                if base_name not in params:
                    params[base_name] = ParameterSpec(name=base_name)
                params[base_name].steps = int(cls._parse_value(value_part))
                
            elif key.endswith("_step"):
                base_name = key[:-5]
                if base_name not in params:
                    params[base_name] = ParameterSpec(name=base_name)
                params[base_name].step_size = cls._parse_value(value_part)
                
            elif key.endswith("_log"):
                base_name = key[:-4]
                if base_name not in params:
                    params[base_name] = ParameterSpec(name=base_name)
                params[base_name].log_scale = value_part.lower() in (".true.", "true", "1")
                
            else:
                # Check if it's a list of values
                if "," in value_part:
                    values = []
                    for item in value_part.split(","):
                        item = item.strip()
                        if item:
                            values.append(cls._parse_value(item))
                    params[key] = ParameterSpec(name=key, values=values)
                else:
                    # Single value - treat as list of one
                    params[key] = ParameterSpec(name=key, values=[cls._parse_value(value_part)])
        
        return list(params.values())
    
    @staticmethod
    def _parse_value(value_str: str) -> int | float | str:
        """Parse a single value string."""
        value_str = value_str.strip().strip("'\"")
        
        try:
            value_clean = value_str.lower().replace("d", "e")
            if "." in value_clean or "e" in value_clean:
                return float(value_clean)
            return int(value_str)
        except ValueError:
            return value_str
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for BatchRunner.from_dict()."""
        return {
            "work_dir": str(self.work_dir),
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "name": self.name,
            "timeout": self.timeout,
            "continue_on_error": self.continue_on_error,
            "quiet": self.quiet,
            "parameters": [p.to_dict() for p in self.parameters],
        }
    
    def validate(self) -> list[str]:
        """
        Validate the configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check work_dir exists
        work_path = Path(self.work_dir)
        if not work_path.exists():
            errors.append(f"work_dir does not exist: {self.work_dir}")
        
        # Check parameters have valid specifications
        for param in self.parameters:
            if param.values is None and (param.min_val is None or param.max_val is None):
                errors.append(
                    f"Parameter '{param.name}' needs either 'values' or 'min'/'max' specification"
                )
            
            if param.min_val is not None and param.max_val is not None:
                if param.steps is None and param.step_size is None:
                    errors.append(
                        f"Parameter '{param.name}' range needs 'steps' or 'step_size'"
                    )
        
        return errors


# Example batch_inlist template
EXAMPLE_BATCH_INLIST = """\
! MESA Batch Runner Configuration
! This file configures a parameter sweep for MESA models

&batch_control
    ! Path to MESA work directory (required)
    work_dir = '/path/to/mesa/work'
    
    ! Output directory (optional, defaults to work_dir/batch_runs)
    ! output_dir = '/path/to/output'
    
    ! Batch name (optional)
    ! name = 'my_batch_run'
    
    ! Timeout per run in seconds (optional)
    ! timeout = 7200
    
    ! Continue if a run fails (default: .true.)
    continue_on_error = .true.
    
    ! Suppress progress output (default: .false.)
    quiet = .false.
/

&parameters
    ! Explicit list of values
    initial_mass = 1.0, 2.0, 5.0, 10.0
    
    ! Range with number of steps
    initial_z_min = 0.001
    initial_z_max = 0.02
    initial_z_steps = 5
    
    ! Range with step size
    ! mixing_length_alpha_min = 1.5
    ! mixing_length_alpha_max = 2.5
    ! mixing_length_alpha_step = 0.1
    
    ! Logarithmic spacing
    ! some_param_min = 1e-4
    ! some_param_max = 1e-2
    ! some_param_steps = 10
    ! some_param_log = .true.
/
"""
