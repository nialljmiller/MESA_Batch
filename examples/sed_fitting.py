#!/usr/bin/env python
"""
SED Fitting Example with MESA_infer

This example demonstrates fitting a spectral energy distribution (SED)
from a CSV file containing wavelength, flux, and flux error columns.
"""

import os
from mesa_infer import MESAInfer, InferConfig
from mesa_infer.likelihood import ObservationalData, SEDLikelihood

# -----------------------------------------------------------------------------
# 1. Setup Configuration
# -----------------------------------------------------------------------------

config = InferConfig(
    mesa_config={
        'work_dir': '/path/to/your/mesa/work',
        'timeout': 3600,
        'scheduler': 'local',
        'parallel_runs': 4,
    },
    
    sampler_config={
        'population_size': 64,
        'num_generations': 100,
        'run_smc_refinement': True,
        'smc_nsamples': 50000,
    },
    
    likelihood_config={
        'data_type': 'sed',
        'wavelength_min': 3000.0,   # Angstroms
        'wavelength_max': 10000.0,  # Angstroms
        'extinction_av': 0.1,       # V-band extinction
        'extinction_rv': 3.1,       # R_V (standard MW)
        'normalize': True,          # Normalize SED (distance-independent)
    }
)

# -----------------------------------------------------------------------------
# 2. Load Observational Data from CSV
# -----------------------------------------------------------------------------

# CSV format expected:
#   wavelength,flux,flux_error
#   3500.0,1.234e-15,1.0e-16
#   3600.0,1.456e-15,1.1e-16
#   ...

observations = ObservationalData.from_csv(
    'data/observed_sed.csv',
    data_type='sed',
    wavelength_column='wavelength',  # Column name for wavelength
    flux_column='flux',              # Column name for flux
    flux_err_column='flux_error',    # Column name for flux error
    wavelength_unit='angstrom',      # or 'nm', 'micron'
    flux_unit='flam',                # F_lambda units
)

# -----------------------------------------------------------------------------
# 3. Optional: Add Additional Constraints
# -----------------------------------------------------------------------------

# You can add spectroscopic constraints to help break degeneracies
observations.teff = 5500.0
observations.teff_err = 200.0  # Loose constraint

# Or parallax for absolute flux scaling
observations.parallax = 10.5      # mas
observations.parallax_err = 0.3   # mas

# -----------------------------------------------------------------------------
# 4. Configure SED Likelihood (Advanced Options)
# -----------------------------------------------------------------------------

# For fine-grained control over the likelihood:
sed_likelihood = SEDLikelihood(
    observations=observations,
    wavelength_range=(3000, 10000),
    extinction_av=0.1,
    extinction_rv=3.1,
    normalize=True,
    smoothing_sigma=5.0,       # Gaussian smoothing in Angstroms
    bin_size=50.0,             # Bin the SED to this resolution
    mask_regions=[             # Mask problematic regions
        (6555, 6575),          # H-alpha
        (4855, 4870),          # H-beta
    ],
    systematic_error=0.02,     # Add 2% systematic error floor
)

# -----------------------------------------------------------------------------
# 5. Example Inlist with Parameter Ranges
# -----------------------------------------------------------------------------

# Your inlist_project should specify ranges:
#
# &star_job
#     ! Initial abundances
#     initial_z_min = 0.005
#     initial_z_max = 0.030
# /
#
# &controls
#     ! Mass range
#     initial_mass_min = 0.7
#     initial_mass_max = 1.5
#     
#     ! Age constraint (if fitting evolved stars)
#     max_age_min = 1e9
#     max_age_max = 10e9
#     
#     ! Mixing length
#     mixing_length_alpha_min = 1.5
#     mixing_length_alpha_max = 2.5
# /

# -----------------------------------------------------------------------------
# 6. Run Inference
# -----------------------------------------------------------------------------

infer = MESAInfer(config)
infer.set_observations(observations)

# Optionally use custom likelihood
# infer.set_likelihood(sed_likelihood)

infer.parse_inlist()
results = infer.run()

# -----------------------------------------------------------------------------
# 7. Results
# -----------------------------------------------------------------------------

results.summary()
results.save('output/sed_fitting')

# Get the best-fit SED for comparison
best_params = results.best_fit_parameters
print(f"\nBest-fit parameters: {best_params}")

# Compute chi-square diagnostics
diagnostics = sed_likelihood.chi_square_sed(
    model_wavelength=results.best_model_wavelength,
    model_flux=results.best_model_flux,
)
print(f"Reduced chi-square: {diagnostics['reduced_chi2']:.2f}")
