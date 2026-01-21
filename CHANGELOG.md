# Changelog

All notable changes to MESA_infer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-21

### Added

- Initial release
- Intelligent MESA inlist parser with automatic parameter range detection
  - List syntax: `param = val1, val2, val3`
  - Range syntax: `param_min = X`, `param_max = Y`
  - Support for continuous, categorical, integer, and boolean parameters
- Hybrid GA+SMC-DEMC sampler
  - Genetic algorithm with DEAP backend
  - Differential evolution MCMC hybrid moves
  - SMC tempering for posterior refinement
  - Adaptive temperature scheduling
- Multiple likelihood functions
  - `SEDLikelihood`: Spectral energy distribution fitting with CCM89 extinction
  - `PhotometricLikelihood`: Multi-band photometry and colors
  - `SpectroscopicLikelihood`: Teff, logg, [Fe/H], and abundances
- MESA runner with HPC support
  - Local execution
  - SLURM scheduler integration
  - PBS scheduler support (basic)
- Results management
  - Posterior sample storage and access
  - Summary statistics and credible intervals
  - Corner plot generation
  - Full checkpoint/restart capability
- Configuration system
  - YAML/JSON configuration files
  - Programmatic configuration
  - Validation and defaults

### Dependencies

- Required: numpy, scipy, pandas, deap, pyyaml
- Optional: matplotlib, corner, seaborn (for plotting)
- Optional: astropy, mesa_reader (for extended functionality)

[Unreleased]: https://github.com/nialljmiller/MESA_infer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nialljmiller/MESA_infer/releases/tag/v0.1.0
