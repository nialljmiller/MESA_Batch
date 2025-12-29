"""
Parameter grid generation for MESA batch runs.
"""

from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence
import numpy as np


@dataclass
class Parameter:
    """A single parameter to sweep over."""
    
    name: str
    values: list[Any] = field(default_factory=list)
    
    @classmethod
    def from_values(cls, name: str, values: Sequence[Any]) -> Parameter:
        """Create a parameter from explicit values."""
        return cls(name=name, values=list(values))
    
    @classmethod
    def from_range(
        cls, 
        name: str, 
        min_val: float, 
        max_val: float, 
        steps: int | None = None,
        step_size: float | None = None,
        log_scale: bool = False,
    ) -> Parameter:
        """
        Create a parameter from a range specification.
        
        Args:
            name: Parameter name (as it appears in MESA inlist)
            min_val: Minimum value
            max_val: Maximum value  
            steps: Number of steps (inclusive of endpoints)
            step_size: Size of each step (alternative to steps)
            log_scale: If True, use logarithmic spacing
            
        Returns:
            Parameter instance with computed values
        """
        if steps is None and step_size is None:
            raise ValueError("Must specify either 'steps' or 'step_size'")
        
        if steps is not None and step_size is not None:
            raise ValueError("Cannot specify both 'steps' and 'step_size'")
        
        if log_scale:
            if min_val <= 0 or max_val <= 0:
                raise ValueError("Log scale requires positive min and max values")
            if steps is not None:
                values = np.logspace(np.log10(min_val), np.log10(max_val), steps)
            else:
                # Approximate steps for log scale with step_size
                log_range = np.log10(max_val) - np.log10(min_val)
                n_steps = max(2, int(np.ceil(log_range / np.log10(1 + step_size / min_val))))
                values = np.logspace(np.log10(min_val), np.log10(max_val), n_steps)
        else:
            if steps is not None:
                values = np.linspace(min_val, max_val, steps)
            else:
                values = np.arange(min_val, max_val + step_size / 2, step_size)
        
        return cls(name=name, values=values.tolist())
    
    def __len__(self) -> int:
        return len(self.values)
    
    def __iter__(self) -> Iterator[Any]:
        return iter(self.values)


class ParameterGrid:
    """
    A grid of parameters for batch MESA runs.
    
    The grid generates all combinations of parameter values (Cartesian product).
    """
    
    def __init__(self):
        self._parameters: list[Parameter] = []
    
    def add(
        self,
        name: str,
        values: Sequence[Any] | None = None,
        min_val: float | None = None,
        max_val: float | None = None,
        steps: int | None = None,
        step_size: float | None = None,
        log_scale: bool = False,
    ) -> ParameterGrid:
        """
        Add a parameter to the grid.
        
        Can specify either explicit values OR a range (min/max with steps/step_size).
        
        Args:
            name: Parameter name as it appears in MESA inlist
            values: Explicit list of values to use
            min_val: Minimum value for range
            max_val: Maximum value for range
            steps: Number of steps for range (inclusive)
            step_size: Step size for range
            log_scale: Use logarithmic spacing for range
            
        Returns:
            Self for method chaining
        """
        if values is not None:
            param = Parameter.from_values(name, values)
        elif min_val is not None and max_val is not None:
            param = Parameter.from_range(
                name, min_val, max_val, steps, step_size, log_scale
            )
        else:
            raise ValueError(
                f"Parameter '{name}': must specify either 'values' or 'min_val'/'max_val'"
            )
        
        # Check for duplicate parameter names
        if any(p.name == name for p in self._parameters):
            raise ValueError(f"Parameter '{name}' already added to grid")
        
        self._parameters.append(param)
        return self
    
    def remove(self, name: str) -> ParameterGrid:
        """Remove a parameter from the grid."""
        self._parameters = [p for p in self._parameters if p.name != name]
        return self
    
    @property
    def parameter_names(self) -> list[str]:
        """List of parameter names in the grid."""
        return [p.name for p in self._parameters]
    
    @property
    def n_parameters(self) -> int:
        """Number of parameters in the grid."""
        return len(self._parameters)
    
    @property
    def n_combinations(self) -> int:
        """Total number of parameter combinations."""
        if not self._parameters:
            return 0
        result = 1
        for p in self._parameters:
            result *= len(p)
        return result
    
    def __len__(self) -> int:
        return self.n_combinations
    
    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all parameter combinations as dictionaries."""
        if not self._parameters:
            return
        
        names = [p.name for p in self._parameters]
        value_lists = [p.values for p in self._parameters]
        
        for combo in itertools.product(*value_lists):
            yield dict(zip(names, combo))
    
    def to_list(self) -> list[dict[str, Any]]:
        """Return all combinations as a list of dictionaries."""
        return list(self)
    
    def summary(self) -> str:
        """Return a summary of the grid."""
        lines = [f"ParameterGrid with {self.n_parameters} parameters, {self.n_combinations} combinations:"]
        for p in self._parameters:
            if len(p.values) <= 5:
                vals_str = str(p.values)
            else:
                vals_str = f"[{p.values[0]}, {p.values[1]}, ..., {p.values[-1]}] ({len(p)} values)"
            lines.append(f"  {p.name}: {vals_str}")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"ParameterGrid(n_params={self.n_parameters}, n_combos={self.n_combinations})"
