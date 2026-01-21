#!/usr/bin/env python
"""
Basic Stellar Parameter Inference with MESA_infer

This example demonstrates a simple workflow for inferring stellar parameters
from spectroscopic observations (Teff, logg, [Fe/H]).
"""

import os
from mesa_infer import MESAInfer, InferConfig
from mesa_infer.likelihood import ObservationalData

# -----------------------------------------------------------------------------
# 1. Setup Configuration
# -----------------------------------------------------------------------------

config = InferConfig(
    # MESA configuration
    mesa_config={
        'work_dir': '/path/to/your/mesa/work',  # Your MESA work directory
        'mesa_dir': os.environ.get('MESA_DIR'),  # Or specify path directly
        'inlist_name': 'inlist_project',
        'timeout': 3600,  # 1 hour timeout per model
        'scheduler': 'local',
        'parallel_runs': 4,  # Run 4 models in parallel
    },
    
    # Sampler configuration
    sampler_config={
        'population_size': 32,        # GA population size
        'num_generations': 50,        # Number of GA generations
        'crossover_probability': 0.5,
        'mutation_probability': 0.5,
        'elitism_fraction': 0.0625,   # Keep top 6.25%
        'demc_hybrid': True,          # Use DE-MCMC moves
        'run_smc_refinement': True,   # Refine with SMC-DEMC
        'smc_nsamples': 10000,        # Posterior samples
    },
    
    # Likelihood configuration
    likelihood_config={
        'data_type': 'spectroscopic',
    }
)

# -----------------------------------------------------------------------------
# 2. Define Observational Constraints
# -----------------------------------------------------------------------------

# Example: Solar-like star
observations = ObservationalData(
    teff=5780.0,      teff_err=80.0,    # Effective temperature [K]
    logg=4.44,        logg_err=0.10,    # Surface gravity [dex]
    feh=0.0,          feh_err=0.08,     # Metallicity [dex]
)

# -----------------------------------------------------------------------------
# 3. Prepare MESA Inlist with Parameter Ranges
# -----------------------------------------------------------------------------

# Your inlist_project should contain parameter ranges like:
#
# &controls
#     ! Mass range to explore
#     initial_mass_min = 0.8
#     initial_mass_max = 1.2
#     
#     ! Metallicity range
#     initial_z_min = 0.01
#     initial_z_max = 0.025
#     
#     ! Mixing length
#     mixing_length_alpha_min = 1.6
#     mixing_length_alpha_max = 2.2
# /

# -----------------------------------------------------------------------------
# 4. Run Inference
# -----------------------------------------------------------------------------

# Initialize
infer = MESAInfer(config)

# Load observations
infer.set_observations(observations)

# Parse the inlist to detect parameters
infer.parse_inlist()

# Print detected parameters
print("Detected parameters:")
for name, spec in infer.parameters.items():
    print(f"  {name}: {spec.param_type}, bounds={spec.bounds}")

# Run the inference
print("\nStarting inference...")
results = infer.run()

# -----------------------------------------------------------------------------
# 5. Analyze Results
# -----------------------------------------------------------------------------

# Print summary
results.summary()

# Get posterior statistics
stats = results.get_summary_statistics()
print("\nParameter Estimates:")
for param, values in stats.items():
    print(f"  {param}: {values['median']:.4f} +{values['84th']-values['median']:.4f} -{values['median']-values['16th']:.4f}")

# Save results
results.save('output/basic_inference')

# Generate corner plot (requires corner and matplotlib)
try:
    results.make_corner_plot(output_path='output/basic_inference/corner.png')
    print("\nCorner plot saved to output/basic_inference/corner.png")
except ImportError:
    print("\nInstall corner and matplotlib for corner plots: pip install corner matplotlib")

print("\nDone! Results saved to output/basic_inference/")
