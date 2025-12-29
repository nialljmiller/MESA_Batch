"""Tests for the grid module."""

import pytest
import numpy as np
from mesa_batch_runner.grid import Parameter, ParameterGrid


class TestParameter:
    """Tests for Parameter class."""
    
    def test_from_values(self):
        param = Parameter.from_values("initial_mass", [1.0, 2.0, 5.0])
        assert param.name == "initial_mass"
        assert param.values == [1.0, 2.0, 5.0]
        assert len(param) == 3
    
    def test_from_range_steps(self):
        param = Parameter.from_range("initial_z", 0.001, 0.02, steps=5)
        assert param.name == "initial_z"
        assert len(param) == 5
        assert np.isclose(param.values[0], 0.001)
        assert np.isclose(param.values[-1], 0.02)
    
    def test_from_range_step_size(self):
        param = Parameter.from_range("alpha", 1.5, 2.5, step_size=0.5)
        assert len(param) == 3
        assert np.isclose(param.values[0], 1.5)
        assert np.isclose(param.values[1], 2.0)
        assert np.isclose(param.values[2], 2.5)
    
    def test_from_range_log_scale(self):
        param = Parameter.from_range("param", 1e-4, 1e-2, steps=3, log_scale=True)
        assert len(param) == 3
        assert np.isclose(param.values[0], 1e-4)
        assert np.isclose(param.values[1], 1e-3)
        assert np.isclose(param.values[2], 1e-2)
    
    def test_from_range_requires_steps_or_step_size(self):
        with pytest.raises(ValueError):
            Parameter.from_range("param", 0, 1)
    
    def test_from_range_rejects_both_steps_and_step_size(self):
        with pytest.raises(ValueError):
            Parameter.from_range("param", 0, 1, steps=5, step_size=0.1)


class TestParameterGrid:
    """Tests for ParameterGrid class."""
    
    def test_empty_grid(self):
        grid = ParameterGrid()
        assert grid.n_parameters == 0
        assert grid.n_combinations == 0
        assert len(grid) == 0
    
    def test_single_parameter(self):
        grid = ParameterGrid()
        grid.add("mass", values=[1.0, 2.0, 5.0])
        
        assert grid.n_parameters == 1
        assert grid.n_combinations == 3
        assert grid.parameter_names == ["mass"]
    
    def test_multiple_parameters(self):
        grid = ParameterGrid()
        grid.add("mass", values=[1.0, 2.0])
        grid.add("z", values=[0.01, 0.02, 0.03])
        
        assert grid.n_parameters == 2
        assert grid.n_combinations == 6  # 2 * 3
    
    def test_iteration(self):
        grid = ParameterGrid()
        grid.add("a", values=[1, 2])
        grid.add("b", values=["x", "y"])
        
        combos = list(grid)
        assert len(combos) == 4
        assert {"a": 1, "b": "x"} in combos
        assert {"a": 1, "b": "y"} in combos
        assert {"a": 2, "b": "x"} in combos
        assert {"a": 2, "b": "y"} in combos
    
    def test_method_chaining(self):
        grid = ParameterGrid()
        result = grid.add("a", values=[1]).add("b", values=[2])
        
        assert result is grid
        assert grid.n_parameters == 2
    
    def test_duplicate_parameter_raises(self):
        grid = ParameterGrid()
        grid.add("mass", values=[1.0])
        
        with pytest.raises(ValueError):
            grid.add("mass", values=[2.0])
    
    def test_remove_parameter(self):
        grid = ParameterGrid()
        grid.add("a", values=[1])
        grid.add("b", values=[2])
        grid.remove("a")
        
        assert grid.n_parameters == 1
        assert grid.parameter_names == ["b"]
    
    def test_add_with_range(self):
        grid = ParameterGrid()
        grid.add("z", min_val=0.001, max_val=0.02, steps=3)
        
        assert grid.n_parameters == 1
        assert grid.n_combinations == 3
    
    def test_add_requires_values_or_range(self):
        grid = ParameterGrid()
        
        with pytest.raises(ValueError):
            grid.add("param")
