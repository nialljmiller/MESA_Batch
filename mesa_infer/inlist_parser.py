"""
MESA Inlist Parser for MESA_infer.

Parses MESA inlists and intelligently detects parameters specified as:
- Lists: initial_mass = 1.0, 2.0, 5.0
- Ranges: initial_mass_min = 0.8, initial_mass_max = 2.0, initial_mass_steps = 10
- Categorical: some_option = 'option1', 'option2', 'option3'

The user provides an inlist in EXACT MESA format but with parameters
expressed as ranges or lists where they want exploration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np


class ParameterType(Enum):
    """Type of parameter for sampling."""
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    INTEGER = "integer"
    FIXED = "fixed"


@dataclass
class ParameterSpec:
    """Specification for a parameter to be explored."""
    
    name: str
    param_type: ParameterType
    section: str  # MESA inlist section: 'star_job', 'controls', 'eos', 'kap', 'pgstar'
    
    # For continuous/integer parameters
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    log_scale: bool = False
    
    # For categorical parameters
    categories: Optional[List[Any]] = None
    
    # For fixed parameters
    fixed_value: Optional[Any] = None
    
    # Prior specification
    prior: str = "uniform"  # 'uniform', 'log_uniform', 'normal', 'truncnormal'
    prior_params: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.param_type == ParameterType.CONTINUOUS:
            if self.min_val is None or self.max_val is None:
                raise ValueError(f"Continuous parameter {self.name} requires min_val and max_val")
        elif self.param_type == ParameterType.CATEGORICAL:
            if self.categories is None or len(self.categories) == 0:
                raise ValueError(f"Categorical parameter {self.name} requires categories")
        elif self.param_type == ParameterType.FIXED:
            if self.fixed_value is None:
                raise ValueError(f"Fixed parameter {self.name} requires fixed_value")
    
    @property
    def n_categories(self) -> int:
        """Number of categories for categorical parameters."""
        if self.categories:
            return len(self.categories)
        return 0
    
    @property
    def bounds(self) -> Tuple[float, float]:
        """Return (min, max) bounds for continuous parameters."""
        if self.param_type in (ParameterType.CONTINUOUS, ParameterType.INTEGER):
            return (self.min_val, self.max_val)
        elif self.param_type == ParameterType.CATEGORICAL:
            return (0, len(self.categories) - 1)
        else:
            return (self.fixed_value, self.fixed_value)
    
    def sample_prior(self, rng: np.random.Generator = None) -> Union[float, int, Any]:
        """Sample a value from the prior distribution."""
        if rng is None:
            rng = np.random.default_rng()
        
        if self.param_type == ParameterType.FIXED:
            return self.fixed_value
        
        if self.param_type == ParameterType.CATEGORICAL:
            idx = rng.integers(0, len(self.categories))
            return self.categories[idx]
        
        # Continuous or integer
        if self.prior == "uniform" or self.prior == "flat":
            val = rng.uniform(self.min_val, self.max_val)
        elif self.prior == "log_uniform" or self.log_scale:
            log_min = np.log10(self.min_val)
            log_max = np.log10(self.max_val)
            val = 10 ** rng.uniform(log_min, log_max)
        elif self.prior == "normal":
            mu = self.prior_params.get('mu', (self.min_val + self.max_val) / 2)
            sigma = self.prior_params.get('sigma', (self.max_val - self.min_val) / 4)
            val = rng.normal(mu, sigma)
            val = np.clip(val, self.min_val, self.max_val)
        else:
            val = rng.uniform(self.min_val, self.max_val)
        
        if self.param_type == ParameterType.INTEGER:
            val = int(round(val))
        
        return val
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'type': self.param_type.value,
            'section': self.section,
            'min': self.min_val,
            'max': self.max_val,
            'log_scale': self.log_scale,
            'categories': self.categories,
            'prior': self.prior,
            'prior_params': self.prior_params,
        }


class InlistParser:
    """
    Parse MESA inlists and detect parameters for exploration.
    
    The parser recognizes several patterns for specifying parameter ranges:
    
    1. List syntax: `initial_mass = 1.0, 2.0, 5.0, 10.0`
    2. Range syntax: `initial_z_min = 0.001` / `initial_z_max = 0.02` / `initial_z_steps = 5`
    3. Categorical: `some_flag = .true., .false.` or `yield_table = 'table1', 'table2'`
    """
    
    # MESA inlist sections
    SECTIONS = ['star_job', 'eos', 'kap', 'controls', 'pgstar']
    
    # Patterns for Fortran values
    FORTRAN_TRUE = re.compile(r'^\.true\.$', re.IGNORECASE)
    FORTRAN_FALSE = re.compile(r'^\.false\.$', re.IGNORECASE)
    FORTRAN_DOUBLE = re.compile(r'^[+-]?\d*\.?\d+[dDeE][+-]?\d+$')
    FORTRAN_FLOAT = re.compile(r'^[+-]?\d*\.\d+$')
    FORTRAN_INT = re.compile(r'^[+-]?\d+$')
    FORTRAN_STRING = re.compile(r"^['\"].*['\"]$")
    
    def __init__(self, filepath: Optional[Union[str, Path]] = None, content: Optional[str] = None):
        """
        Initialize parser with inlist file or content.
        
        Args:
            filepath: Path to inlist file
            content: Raw inlist content string
        """
        self.filepath = Path(filepath) if filepath else None
        self.content = content
        
        if filepath and content is None:
            with open(filepath, 'r') as f:
                self.content = f.read()
        
        self.parameters: Dict[str, ParameterSpec] = {}
        self.fixed_parameters: Dict[str, Tuple[str, Any]] = {}  # section -> (name, value)
        self._sections: Dict[str, Dict[str, Any]] = {}
        
        if self.content:
            self._parse()
    
    def _parse(self) -> None:
        """Parse the inlist content."""
        current_section = None
        pending_ranges: Dict[str, Dict[str, Any]] = {}  # Track _min/_max/_steps patterns
        
        for line in self.content.split('\n'):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('!'):
                continue
            
            # Remove inline comments
            if '!' in line:
                # Be careful with strings containing !
                if "'" in line or '"' in line:
                    # More careful parsing needed
                    pass
                else:
                    line = line.split('!')[0].strip()
            
            # Check for section markers
            section_match = re.match(r'&(\w+)', line)
            if section_match:
                current_section = section_match.group(1).lower()
                if current_section not in self._sections:
                    self._sections[current_section] = {}
                continue
            
            # Check for section end
            if line.startswith('/'):
                current_section = None
                continue
            
            if current_section is None:
                continue
            
            # Parse parameter assignment
            param_match = re.match(r'(\w+(?:\([^)]+\))?)\s*=\s*(.+)', line)
            if not param_match:
                continue
            
            param_name = param_match.group(1).strip()
            value_str = param_match.group(2).strip()
            
            # Remove trailing comma if present
            if value_str.endswith(','):
                value_str = value_str[:-1].strip()
            
            # Check if this is a range specification (_min, _max, _steps, _log)
            range_match = re.match(r'(.+)_(min|max|steps?|step_size|log)$', param_name, re.IGNORECASE)
            if range_match:
                base_name = range_match.group(1)
                range_type = range_match.group(2).lower()
                
                if base_name not in pending_ranges:
                    pending_ranges[base_name] = {'section': current_section}
                
                value = self._parse_value(value_str)
                pending_ranges[base_name][range_type] = value
                continue
            
            # Check if value is a list (comma-separated)
            values = self._split_value_list(value_str)
            
            if len(values) > 1:
                # This is a list specification - create a parameter
                parsed_values = [self._parse_value(v) for v in values]
                self._create_parameter_from_list(param_name, parsed_values, current_section)
            else:
                # Single value - store as fixed parameter
                value = self._parse_value(value_str)
                self._sections[current_section][param_name] = value
                self.fixed_parameters[param_name] = (current_section, value)
        
        # Process pending range specifications
        for base_name, spec in pending_ranges.items():
            self._create_parameter_from_range(base_name, spec)
    
    def _split_value_list(self, value_str: str) -> List[str]:
        """Split a comma-separated value list, respecting strings."""
        values = []
        current = []
        in_string = False
        string_char = None
        
        for char in value_str:
            if char in ('"', "'") and not in_string:
                in_string = True
                string_char = char
                current.append(char)
            elif char == string_char and in_string:
                in_string = False
                string_char = None
                current.append(char)
            elif char == ',' and not in_string:
                val = ''.join(current).strip()
                if val:
                    values.append(val)
                current = []
            else:
                current.append(char)
        
        val = ''.join(current).strip()
        if val:
            values.append(val)
        
        return values
    
    def _parse_value(self, value_str: str) -> Any:
        """Parse a Fortran value string to Python type."""
        value_str = value_str.strip()
        
        # Boolean
        if self.FORTRAN_TRUE.match(value_str):
            return True
        if self.FORTRAN_FALSE.match(value_str):
            return False
        
        # String
        if self.FORTRAN_STRING.match(value_str):
            return value_str.strip("'\"")
        
        # Numeric
        if self.FORTRAN_DOUBLE.match(value_str):
            return float(value_str.lower().replace('d', 'e'))
        
        if self.FORTRAN_FLOAT.match(value_str):
            return float(value_str)
        
        if self.FORTRAN_INT.match(value_str):
            return int(value_str)
        
        # Fallback to string
        return value_str
    
    def _create_parameter_from_list(self, name: str, values: List[Any], section: str) -> None:
        """Create a ParameterSpec from a list of values."""
        # Determine parameter type from values
        all_bool = all(isinstance(v, bool) for v in values)
        all_str = all(isinstance(v, str) for v in values)
        all_numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
        all_int = all(isinstance(v, int) and not isinstance(v, bool) for v in values)
        
        if all_bool or all_str:
            # Categorical parameter
            param = ParameterSpec(
                name=name,
                param_type=ParameterType.CATEGORICAL,
                section=section,
                categories=values,
            )
        elif all_numeric:
            # Could be categorical (discrete set) or range
            if len(values) <= 5 and all_int:
                # Treat as categorical if few integer values
                param = ParameterSpec(
                    name=name,
                    param_type=ParameterType.CATEGORICAL,
                    section=section,
                    categories=values,
                )
            else:
                # Treat as continuous range
                param = ParameterSpec(
                    name=name,
                    param_type=ParameterType.CONTINUOUS,
                    section=section,
                    min_val=min(values),
                    max_val=max(values),
                    log_scale=self._should_use_log(min(values), max(values)),
                )
        else:
            # Mixed types - treat as categorical
            param = ParameterSpec(
                name=name,
                param_type=ParameterType.CATEGORICAL,
                section=section,
                categories=values,
            )
        
        self.parameters[name] = param
    
    def _create_parameter_from_range(self, name: str, spec: Dict[str, Any]) -> None:
        """Create a ParameterSpec from range specification."""
        section = spec.get('section', 'controls')
        min_val = spec.get('min')
        max_val = spec.get('max')
        steps = spec.get('steps') or spec.get('step')
        step_size = spec.get('step_size')
        log_scale = spec.get('log', False)
        
        if min_val is not None and max_val is not None:
            # Determine if integer or continuous
            is_int = isinstance(min_val, int) and isinstance(max_val, int)
            
            param = ParameterSpec(
                name=name,
                param_type=ParameterType.INTEGER if is_int else ParameterType.CONTINUOUS,
                section=section,
                min_val=float(min_val),
                max_val=float(max_val),
                log_scale=log_scale if log_scale else self._should_use_log(min_val, max_val),
            )
            self.parameters[name] = param
    
    def _should_use_log(self, min_val: float, max_val: float) -> bool:
        """Determine if log scale should be used based on range."""
        if min_val <= 0:
            return False
        ratio = max_val / min_val
        return ratio > 100
    
    def add_parameter(
        self,
        name: str,
        param_type: Union[str, ParameterType] = "continuous",
        section: str = "controls",
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        categories: Optional[List[Any]] = None,
        log_scale: bool = False,
        prior: str = "uniform",
        **prior_params
    ) -> InlistParser:
        """
        Manually add a parameter to explore.
        
        Args:
            name: Parameter name as in MESA inlist
            param_type: 'continuous', 'categorical', 'integer', or ParameterType enum
            section: MESA inlist section
            min_val: Minimum value for continuous/integer
            max_val: Maximum value for continuous/integer
            categories: List of values for categorical
            log_scale: Use logarithmic scale
            prior: Prior distribution type
            **prior_params: Additional prior parameters
        
        Returns:
            Self for method chaining
        """
        if isinstance(param_type, str):
            param_type = ParameterType(param_type)
        
        param = ParameterSpec(
            name=name,
            param_type=param_type,
            section=section,
            min_val=min_val,
            max_val=max_val,
            categories=categories,
            log_scale=log_scale,
            prior=prior,
            prior_params=prior_params,
        )
        self.parameters[name] = param
        return self
    
    def get_parameter(self, name: str) -> Optional[ParameterSpec]:
        """Get a parameter specification by name."""
        return self.parameters.get(name)

    def get_exploration_parameters(self) -> Dict[str, ParameterSpec]:
        """Return parameters marked for exploration."""
        return self.parameters
    
    def get_continuous_parameters(self) -> List[ParameterSpec]:
        """Get all continuous parameters."""
        return [p for p in self.parameters.values() 
                if p.param_type in (ParameterType.CONTINUOUS, ParameterType.INTEGER)]
    
    def get_categorical_parameters(self) -> List[ParameterSpec]:
        """Get all categorical parameters."""
        return [p for p in self.parameters.values() 
                if p.param_type == ParameterType.CATEGORICAL]
    
    def get_fixed_value(self, name: str) -> Optional[Any]:
        """Get a fixed parameter value."""
        if name in self.fixed_parameters:
            return self.fixed_parameters[name][1]
        return None
    
    @property
    def n_parameters(self) -> int:
        """Total number of free parameters."""
        return len(self.parameters)
    
    @property
    def n_continuous(self) -> int:
        """Number of continuous parameters."""
        return len(self.get_continuous_parameters())
    
    @property
    def n_categorical(self) -> int:
        """Number of categorical parameters."""
        return len(self.get_categorical_parameters())
    
    @property
    def parameter_names(self) -> List[str]:
        """List of parameter names."""
        return list(self.parameters.keys())
    
    def generate_inlist(self, values: Dict[str, Any], template: Optional[str] = None) -> str:
        """
        Generate a MESA inlist with specific parameter values.
        
        Args:
            values: Dictionary mapping parameter names to values
            template: Template inlist content (uses original if None)
        
        Returns:
            Generated inlist content
        """
        if template is None:
            template = self.content
        
        content = template
        
        for name, value in values.items():
            param = self.parameters.get(name)
            if param is None:
                continue
            
            # Format value for Fortran
            formatted = self._format_fortran_value(value)
            
            # Replace in content
            # Handle both simple assignment and list patterns
            patterns = [
                rf'^(\s*){re.escape(name)}\s*=\s*[^\n]+',  # Standard assignment
                rf'^(\s*){re.escape(name)}_min\s*=.*?{re.escape(name)}_max\s*=.*?({re.escape(name)}_steps?\s*=.*?)?',  # Range pattern
            ]
            
            for pattern in patterns:
                content = re.sub(
                    pattern,
                    rf'\g<1>{name} = {formatted}',
                    content,
                    flags=re.MULTILINE | re.DOTALL
                )
        
        return content
    
    def _format_fortran_value(self, value: Any) -> str:
        """Format a Python value for Fortran."""
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
    
    def summary(self) -> str:
        """Return a summary of parsed parameters."""
        lines = [
            f"MESA Inlist Parser Summary",
            f"{'=' * 40}",
            f"Total free parameters: {self.n_parameters}",
            f"  Continuous: {self.n_continuous}",
            f"  Categorical: {self.n_categorical}",
            f"Fixed parameters: {len(self.fixed_parameters)}",
            "",
            "Free Parameters:",
        ]
        
        for name, param in self.parameters.items():
            if param.param_type == ParameterType.CATEGORICAL:
                lines.append(f"  {name} ({param.section}): {param.categories}")
            else:
                scale = " [log]" if param.log_scale else ""
                lines.append(f"  {name} ({param.section}): [{param.min_val}, {param.max_val}]{scale}")
        
        return "\n".join(lines)
