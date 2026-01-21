#!/usr/bin/env python
"""
HPC Execution with SLURM

This example demonstrates running MESA_infer on a SLURM-managed cluster.
Each MESA model is submitted as a separate SLURM job.
"""

import os
from mesa_infer import MESAInfer, InferConfig

# -----------------------------------------------------------------------------
# 1. SLURM Configuration
# -----------------------------------------------------------------------------

config = InferConfig(
    mesa_config={
        'work_dir': '/scratch/user/mesa_work',  # Use scratch filesystem
        'mesa_dir': '/software/mesa/mesa-r24.03.1',
        'inlist_name': 'inlist_project',
        'timeout': 14400,  # 4 hours max per model
        
        # Scheduler settings
        'scheduler': 'slurm',
        
        # SLURM-specific options
        'slurm_partition': 'compute',      # Partition/queue name
        'slurm_time': '04:00:00',          # Wall time limit
        'slurm_nodes': 1,                   # Nodes per job
        'slurm_ntasks': 1,                  # Tasks per job
        'slurm_cpus_per_task': 4,          # CPUs per task
        'slurm_mem': '8G',                  # Memory per job
        'slurm_account': 'myproject',       # Account/allocation
        
        # Optional SLURM settings
        'slurm_constraint': None,           # Node constraints
        'slurm_exclude': None,              # Nodes to exclude
        'slurm_qos': None,                  # Quality of service
        
        # Job management
        'parallel_runs': 32,                # Max concurrent jobs
        'poll_interval': 30,                # Check job status every 30s
        'resubmit_on_failure': True,        # Retry failed jobs
        'max_resubmits': 2,                 # Max retry attempts
    },
    
    sampler_config={
        'population_size': 128,             # Larger population for HPC
        'num_generations': 200,
        'checkpoint_interval': 10,          # Save progress every 10 gens
        'run_smc_refinement': True,
        'smc_nsamples': 100000,
    },
    
    likelihood_config={
        'data_type': 'combined',
        'use_sed': True,
        'use_spectroscopic': True,
    }
)

# -----------------------------------------------------------------------------
# 2. Additional SLURM Environment Setup
# -----------------------------------------------------------------------------

# You may need to set up modules or environment in your job script
# This can be done via slurm_setup_commands

config.mesa_config['slurm_setup_commands'] = [
    'module purge',
    'module load mesa/r24.03.1',
    'module load mesasdk/24.3.1',
    'export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK',
]

# -----------------------------------------------------------------------------
# 3. Output Management
# -----------------------------------------------------------------------------

# Configure where outputs go
config.mesa_config['output_dir'] = '/scratch/user/mesa_infer_output'
config.mesa_config['keep_all_runs'] = False  # Only keep best models
config.mesa_config['keep_logs'] = True
config.mesa_config['compress_outputs'] = True

# -----------------------------------------------------------------------------
# 4. Run
# -----------------------------------------------------------------------------

from mesa_infer.likelihood import ObservationalData

observations = ObservationalData.from_csv('data/target_star.csv', data_type='sed')
observations.teff = 5800.0
observations.teff_err = 100.0
observations.logg = 4.4
observations.logg_err = 0.1
observations.feh = -0.1
observations.feh_err = 0.1

infer = MESAInfer(config)
infer.set_observations(observations)
infer.parse_inlist()

# Monitor progress
infer.enable_progress_logging('logs/inference.log')

# Run (this will submit SLURM jobs)
print("Submitting SLURM jobs...")
results = infer.run()

# -----------------------------------------------------------------------------
# 5. Job Monitoring (Alternative: Run in Stages)
# -----------------------------------------------------------------------------

# For very long runs, you might want to run in stages:

# Stage 1: GA exploration
# ga_results = infer.run_ga_only()
# ga_results.save('output/ga_checkpoint')

# Stage 2: SMC refinement (can be restarted later)
# from mesa_infer import InferenceResults
# ga_results = InferenceResults.load('output/ga_checkpoint')
# final_results = infer.run_smc_refinement(ga_results)

# -----------------------------------------------------------------------------
# 6. Results
# -----------------------------------------------------------------------------

results.summary()
results.save('/scratch/user/mesa_infer_output/final_results')

print("\nJob statistics:")
print(f"  Total MESA runs: {results.diagnostics['total_runs']}")
print(f"  Successful runs: {results.diagnostics['successful_runs']}")
print(f"  Failed runs: {results.diagnostics['failed_runs']}")
print(f"  Total compute time: {results.diagnostics['total_time_hours']:.1f} hours")
