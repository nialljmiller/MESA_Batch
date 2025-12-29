"""
MESA inlist parsing and modification.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Any


class InlistParser:
    """
    Parser for MESA inlist files.
    
    Extracts parameter values from Fortran namelist format.
    """
    
    # Patterns for different value types
    PATTERNS = {
        "string": r"'([^']*)'",
        "logical": r"\.(true|false)\.",
        "integer": r"([+-]?\d+)(?![dDeE.])",
        "real": r"([+-]?\d*\.?\d+(?:[dDeE][+-]?\d+)?)",
        "array_string": r"\(\s*(\d+)\s*\)\s*=\s*'([^']*)'",
        "array_numeric": r"\(\s*(\d+)\s*\)\s*=\s*([+-]?\d*\.?\d+(?:[dDeE][+-]?\d+)?)",
    }
    
    def __init__(self, filepath: str | Path | None = None, content: str | None = None):
        """
        Initialize parser with either a file path or content string.
        
        Args:
            filepath: Path to inlist file
            content: Inlist content as string
        """
        if filepath is not None:
            self.filepath = Path(filepath)
            self.content = self.filepath.read_text()
        elif content is not None:
            self.filepath = None
            self.content = content
        else:
            raise ValueError("Must provide either 'filepath' or 'content'")
    
    def get_sections(self) -> list[str]:
        """Return list of namelist sections found in the inlist."""
        pattern = r"&(\w+)"
        return re.findall(pattern, self.content)
    
    def has_section(self, section: str) -> bool:
        """Check if a namelist section exists."""
        return f"&{section}" in self.content
    
    def get_value(self, param: str, section: str | None = None) -> Any | None:
        """
        Get the value of a parameter.
        
        Args:
            param: Parameter name
            section: Optional section to limit search to
            
        Returns:
            Parsed value or None if not found
        """
        content = self._get_section_content(section) if section else self.content
        
        # Check if parameter is commented out
        if self._is_commented(param, content):
            return None
        
        # Try different patterns
        # String value
        pattern = rf"^\s*{re.escape(param)}\s*=\s*'([^']*)'"
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            return match.group(1)
        
        # Logical value
        pattern = rf"^\s*{re.escape(param)}\s*=\s*\.(true|false)\."
        match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"
        
        # Numeric value
        pattern = rf"^\s*{re.escape(param)}\s*=\s*([+-]?\d*\.?\d+(?:[dDeE][+-]?\d+)?)"
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            return self._parse_number(match.group(1))
        
        return None
    
    def get_array_value(self, param: str, index: int, section: str | None = None) -> Any | None:
        """
        Get the value of an array parameter at a specific index.
        
        Args:
            param: Parameter name (without index)
            index: Array index (1-based, as in Fortran)
            section: Optional section to limit search to
            
        Returns:
            Parsed value or None if not found
        """
        content = self._get_section_content(section) if section else self.content
        
        # String array
        pattern = rf"^\s*{re.escape(param)}\s*\(\s*{index}\s*\)\s*=\s*'([^']*)'"
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            return match.group(1)
        
        # Numeric array
        pattern = rf"^\s*{re.escape(param)}\s*\(\s*{index}\s*\)\s*=\s*([+-]?\d*\.?\d+(?:[dDeE][+-]?\d+)?)"
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            return self._parse_number(match.group(1))
        
        return None
    
    def _get_section_content(self, section: str) -> str:
        """Extract content of a specific namelist section."""
        pattern = rf"&{section}\s*(.*?)\s*/"
        match = re.search(pattern, self.content, re.DOTALL | re.IGNORECASE)
        return match.group(1) if match else ""
    
    def _is_commented(self, param: str, content: str) -> bool:
        """Check if a parameter line is commented out."""
        pattern = rf"^\s*!\s*{re.escape(param)}\s*="
        return re.search(pattern, content, re.MULTILINE) is not None
    
    def _parse_number(self, value: str) -> int | float:
        """Parse a Fortran numeric value."""
        # Replace Fortran 'd' exponent with 'e'
        value = value.lower().replace("d", "e")
        if "." in value or "e" in value:
            return float(value)
        return int(value)
    
    def to_dict(self) -> dict[str, Any]:
        """
        Extract all parameters as a dictionary.
        
        Returns:
            Dictionary mapping parameter names to values
        """
        result = {}
        
        # Find all parameter assignments
        pattern = r"^\s*(\w+(?:\(\d+\))?)\s*=\s*(.+?)(?:\s*!.*)?$"
        for match in re.finditer(pattern, self.content, re.MULTILINE):
            param_name = match.group(1)
            value_str = match.group(2).strip()
            
            # Skip commented lines
            line_start = self.content.rfind("\n", 0, match.start()) + 1
            if self.content[line_start:match.start()].strip().startswith("!"):
                continue
            
            # Parse value
            if value_str.startswith("'"):
                # String
                value = value_str.strip("'")
            elif value_str.lower() in (".true.", ".false."):
                # Logical
                value = value_str.lower() == ".true."
            else:
                # Numeric
                try:
                    value = self._parse_number(value_str.split()[0])
                except ValueError:
                    value = value_str
            
            result[param_name] = value
        
        return result


class InlistModifier:
    """
    Modifier for MESA inlist files.
    
    Modifies existing parameters or adds new ones while preserving formatting.
    """
    
    def __init__(self, filepath: str | Path | None = None, content: str | None = None):
        """
        Initialize modifier with either a file path or content string.
        
        Args:
            filepath: Path to inlist file
            content: Inlist content as string
        """
        if filepath is not None:
            self.filepath = Path(filepath)
            self.content = self.filepath.read_text()
        elif content is not None:
            self.filepath = None
            self.content = content
        else:
            raise ValueError("Must provide either 'filepath' or 'content'")
        
        self._modified = False
    
    def set_value(
        self,
        param: str,
        value: Any,
        section: str = "controls",
        comment: str | None = None,
    ) -> InlistModifier:
        """
        Set a parameter value.
        
        If the parameter exists, it will be updated. If not, it will be added
        to the specified section.
        
        Args:
            param: Parameter name
            value: Value to set
            section: Namelist section (default: 'controls')
            comment: Optional inline comment
            
        Returns:
            Self for method chaining
        """
        value_str = self._format_value(value)
        comment_str = f" ! {comment}" if comment else ""
        new_line = f"    {param} = {value_str}{comment_str}"
        
        # Try to find and replace existing parameter
        # Match both uncommented and commented versions
        patterns = [
            # Uncommented
            rf"^(\s*){re.escape(param)}\s*=\s*[^\n]+",
            # Commented
            rf"^(\s*)!\s*{re.escape(param)}\s*=\s*[^\n]+",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.content, re.MULTILINE)
            if match:
                indent = match.group(1)
                self.content = re.sub(
                    pattern,
                    f"{indent}{param} = {value_str}{comment_str}",
                    self.content,
                    count=1,
                    flags=re.MULTILINE,
                )
                self._modified = True
                return self
        
        # Parameter not found - add it to the section
        section_pattern = rf"(&{section}.*?)(^\s*/)"
        match = re.search(section_pattern, self.content, re.DOTALL | re.MULTILINE | re.IGNORECASE)
        
        if match:
            # Insert before the closing '/'
            section_content = match.group(1)
            self.content = self.content.replace(
                match.group(0),
                f"{section_content}\n{new_line}\n/",
            )
        else:
            # Section doesn't exist - add it
            self.content += f"\n&{section}\n{new_line}\n/ ! end of {section} namelist\n"
        
        self._modified = True
        return self
    
    def set_array_value(
        self,
        param: str,
        index: int,
        value: Any,
        section: str = "controls",
    ) -> InlistModifier:
        """
        Set an array parameter value.
        
        Args:
            param: Parameter name (without index)
            index: Array index (1-based)
            value: Value to set
            section: Namelist section
            
        Returns:
            Self for method chaining
        """
        full_param = f"{param}({index})"
        return self.set_value(full_param, value, section)
    
    def comment_out(self, param: str) -> InlistModifier:
        """
        Comment out a parameter.
        
        Args:
            param: Parameter name to comment out
            
        Returns:
            Self for method chaining
        """
        pattern = rf"^(\s*)({re.escape(param)}\s*=\s*[^\n]+)"
        
        def replacer(m):
            return f"{m.group(1)}! {m.group(2)}"
        
        self.content = re.sub(pattern, replacer, self.content, flags=re.MULTILINE)
        self._modified = True
        return self
    
    def uncomment(self, param: str) -> InlistModifier:
        """
        Uncomment a parameter.
        
        Args:
            param: Parameter name to uncomment
            
        Returns:
            Self for method chaining
        """
        pattern = rf"^(\s*)!\s*({re.escape(param)}\s*=\s*[^\n]+)"
        
        def replacer(m):
            return f"{m.group(1)}{m.group(2)}"
        
        self.content = re.sub(pattern, replacer, self.content, flags=re.MULTILINE)
        self._modified = True
        return self
    
    def apply_parameters(self, params: dict[str, Any], section: str = "controls") -> InlistModifier:
        """
        Apply multiple parameters at once.
        
        Args:
            params: Dictionary of parameter names to values
            section: Default section for new parameters
            
        Returns:
            Self for method chaining
        """
        for name, value in params.items():
            self.set_value(name, value, section)
        return self
    
    def _format_value(self, value: Any) -> str:
        """Format a Python value for Fortran."""
        if isinstance(value, bool):
            return ".true." if value else ".false."
        elif isinstance(value, str):
            return f"'{value}'"
        elif isinstance(value, float):
            # Use Fortran 'd' notation for doubles
            formatted = f"{value:.10g}"
            if "e" in formatted.lower():
                formatted = formatted.replace("e", "d").replace("E", "d")
            elif "." not in formatted:
                formatted += "d0"
            return formatted
        else:
            return str(value)
    
    def save(self, filepath: str | Path | None = None) -> None:
        """
        Save the modified inlist to a file.
        
        Args:
            filepath: Path to save to (defaults to original filepath)
        """
        if filepath is None:
            if self.filepath is None:
                raise ValueError("No filepath specified")
            filepath = self.filepath
        
        Path(filepath).write_text(self.content)
    
    def get_content(self) -> str:
        """Return the current content."""
        return self.content
    
    @property
    def modified(self) -> bool:
        """Check if any modifications have been made."""
        return self._modified


# Default inlist template for when no inlist exists
DEFAULT_INLIST_TEMPLATE = """\
! MESA inlist generated by mesa-batch-runner

&star_job
    save_model_when_terminate = .true.
    save_model_filename = 'final.mod'
    pgstar_flag = .false.
/ ! end of star_job namelist


&eos
/ ! end of eos namelist


&kap
    Zbase = 0.02d0
/ ! end of kap namelist


&controls
    ! starting specifications
    initial_mass = 1.0d0
    initial_z = 0.02d0
    initial_y = 0.28d0
    
    ! when to stop
    xa_central_lower_limit_species(1) = 'h1'
    xa_central_lower_limit(1) = 1d-3
    
/ ! end of controls namelist


&pgstar
/ ! end of pgstar namelist
"""
