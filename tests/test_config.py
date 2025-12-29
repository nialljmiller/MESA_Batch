"""Tests for the config module."""

import pytest
from mesa_batch_runner.config import BatchConfig, ParameterSpec


SAMPLE_CONFIG = """\
&batch_control
    work_dir = '/path/to/mesa/work'
    output_dir = '/path/to/output'
    name = 'test_batch'
    timeout = 3600
    continue_on_error = .true.
    quiet = .false.
/

&parameters
    initial_mass = 1.0, 2.0, 5.0
    
    initial_z_min = 0.001
    initial_z_max = 0.02
    initial_z_steps = 5
    
    alpha_min = 1.5
    alpha_max = 2.5
    alpha_step = 0.5
/
"""


class TestParameterSpec:
    """Tests for ParameterSpec class."""
    
    def test_to_dict_values(self):
        spec = ParameterSpec(name="mass", values=[1.0, 2.0])
        d = spec.to_dict()
        
        assert d["name"] == "mass"
        assert d["values"] == [1.0, 2.0]
    
    def test_to_dict_range(self):
        spec = ParameterSpec(name="z", min_val=0.001, max_val=0.02, steps=5)
        d = spec.to_dict()
        
        assert d["name"] == "z"
        assert d["min"] == 0.001
        assert d["max"] == 0.02
        assert d["steps"] == 5


class TestBatchConfig:
    """Tests for BatchConfig class."""
    
    def test_parse_batch_control(self):
        config = BatchConfig.from_string(SAMPLE_CONFIG)
        
        assert config.work_dir == "/path/to/mesa/work"
        assert config.output_dir == "/path/to/output"
        assert config.name == "test_batch"
        assert config.timeout == 3600
        assert config.continue_on_error is True
        assert config.quiet is False
    
    def test_parse_explicit_values(self):
        config = BatchConfig.from_string(SAMPLE_CONFIG)
        
        # Find the initial_mass parameter
        mass_param = next((p for p in config.parameters if p.name == "initial_mass"), None)
        assert mass_param is not None
        assert mass_param.values == [1.0, 2.0, 5.0]
    
    def test_parse_range_with_steps(self):
        config = BatchConfig.from_string(SAMPLE_CONFIG)
        
        z_param = next((p for p in config.parameters if p.name == "initial_z"), None)
        assert z_param is not None
        assert z_param.min_val == 0.001
        assert z_param.max_val == 0.02
        assert z_param.steps == 5
    
    def test_parse_range_with_step_size(self):
        config = BatchConfig.from_string(SAMPLE_CONFIG)
        
        alpha_param = next((p for p in config.parameters if p.name == "alpha"), None)
        assert alpha_param is not None
        assert alpha_param.min_val == 1.5
        assert alpha_param.max_val == 2.5
        assert alpha_param.step_size == 0.5
    
    def test_to_dict(self):
        config = BatchConfig.from_string(SAMPLE_CONFIG)
        d = config.to_dict()
        
        assert d["work_dir"] == "/path/to/mesa/work"
        assert d["name"] == "test_batch"
        assert len(d["parameters"]) == 3
    
    def test_missing_work_dir_raises(self):
        bad_config = """\
&batch_control
    output_dir = '/path/to/output'
/
"""
        with pytest.raises(ValueError):
            BatchConfig.from_string(bad_config)
    
    def test_validate_missing_work_dir(self):
        config = BatchConfig(work_dir="/nonexistent/path")
        errors = config.validate()
        
        assert len(errors) > 0
        assert any("work_dir" in e for e in errors)
    
    def test_validate_incomplete_range(self):
        config = BatchConfig(
            work_dir="/tmp",
            parameters=[
                ParameterSpec(name="z", min_val=0.001, max_val=0.02)
                # Missing steps or step_size
            ]
        )
        errors = config.validate()
        
        assert any("steps" in e or "step_size" in e for e in errors)
