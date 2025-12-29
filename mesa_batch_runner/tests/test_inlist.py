"""Tests for the inlist module."""

import pytest
from mesa_batch_runner.inlist import InlistParser, InlistModifier


SAMPLE_INLIST = """\
! Sample MESA inlist

&star_job
    save_model_when_terminate = .true.
    save_model_filename = 'final.mod'
    pgstar_flag = .false.
/ ! end of star_job namelist

&controls
    ! starting specifications
    initial_mass = 1.5d0
    initial_z = 0.02d0
    
    ! mixing
    overshoot_scheme(1) = 'exponential'
    overshoot_f(1) = 0.015d0
    overshoot_f0(1) = 0.005d0
    
    ! stop condition
    xa_central_lower_limit_species(1) = 'h1'
    xa_central_lower_limit(1) = 1d-3
/ ! end of controls namelist
"""


class TestInlistParser:
    """Tests for InlistParser class."""
    
    def test_parse_sections(self):
        parser = InlistParser(content=SAMPLE_INLIST)
        sections = parser.get_sections()
        
        assert "star_job" in sections
        assert "controls" in sections
    
    def test_has_section(self):
        parser = InlistParser(content=SAMPLE_INLIST)
        
        assert parser.has_section("star_job")
        assert parser.has_section("controls")
        assert not parser.has_section("pgstar")
    
    def test_get_string_value(self):
        parser = InlistParser(content=SAMPLE_INLIST)
        
        assert parser.get_value("save_model_filename") == "final.mod"
        assert parser.get_value("overshoot_scheme(1)") == "exponential"
    
    def test_get_logical_value(self):
        parser = InlistParser(content=SAMPLE_INLIST)
        
        assert parser.get_value("save_model_when_terminate") is True
        assert parser.get_value("pgstar_flag") is False
    
    def test_get_numeric_value(self):
        parser = InlistParser(content=SAMPLE_INLIST)
        
        assert parser.get_value("initial_mass") == 1.5
        assert parser.get_value("initial_z") == 0.02
    
    def test_get_array_value(self):
        parser = InlistParser(content=SAMPLE_INLIST)
        
        assert parser.get_array_value("overshoot_f", 1) == 0.015
        assert parser.get_array_value("overshoot_f0", 1) == 0.005
    
    def test_get_missing_value(self):
        parser = InlistParser(content=SAMPLE_INLIST)
        
        assert parser.get_value("nonexistent_param") is None
    
    def test_to_dict(self):
        parser = InlistParser(content=SAMPLE_INLIST)
        params = parser.to_dict()
        
        assert "initial_mass" in params
        assert params["initial_mass"] == 1.5


class TestInlistModifier:
    """Tests for InlistModifier class."""
    
    def test_set_existing_value(self):
        modifier = InlistModifier(content=SAMPLE_INLIST)
        modifier.set_value("initial_mass", 2.5)
        
        # Parse the modified content
        parser = InlistParser(content=modifier.get_content())
        assert parser.get_value("initial_mass") == 2.5
    
    def test_set_new_value(self):
        modifier = InlistModifier(content=SAMPLE_INLIST)
        modifier.set_value("new_param", 123, section="controls")
        
        assert "new_param = 123" in modifier.get_content()
    
    def test_set_string_value(self):
        modifier = InlistModifier(content=SAMPLE_INLIST)
        modifier.set_value("save_model_filename", "test.mod", section="star_job")
        
        assert "save_model_filename = 'test.mod'" in modifier.get_content()
    
    def test_set_logical_value(self):
        modifier = InlistModifier(content=SAMPLE_INLIST)
        modifier.set_value("pgstar_flag", True, section="star_job")
        
        assert "pgstar_flag = .true." in modifier.get_content()
    
    def test_comment_out(self):
        modifier = InlistModifier(content=SAMPLE_INLIST)
        modifier.comment_out("overshoot_scheme(1)")
        
        assert "! overshoot_scheme(1)" in modifier.get_content()
    
    def test_uncomment(self):
        content = SAMPLE_INLIST.replace(
            "overshoot_scheme(1)", "! overshoot_scheme(1)"
        )
        modifier = InlistModifier(content=content)
        modifier.uncomment("overshoot_scheme(1)")
        
        # Should have uncommented version
        lines = modifier.get_content().split("\n")
        found = False
        for line in lines:
            if "overshoot_scheme(1)" in line and not line.strip().startswith("!"):
                found = True
                break
        assert found
    
    def test_apply_parameters(self):
        modifier = InlistModifier(content=SAMPLE_INLIST)
        modifier.apply_parameters({
            "initial_mass": 3.0,
            "initial_z": 0.01,
        })
        
        parser = InlistParser(content=modifier.get_content())
        assert parser.get_value("initial_mass") == 3.0
        assert parser.get_value("initial_z") == 0.01
    
    def test_method_chaining(self):
        modifier = InlistModifier(content=SAMPLE_INLIST)
        result = modifier.set_value("initial_mass", 2.0).set_value("initial_z", 0.01)
        
        assert result is modifier
    
    def test_modified_flag(self):
        modifier = InlistModifier(content=SAMPLE_INLIST)
        assert not modifier.modified
        
        modifier.set_value("initial_mass", 2.0)
        assert modifier.modified
