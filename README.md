# MESA_infer

**Bayesian Stellar Parameter Inference with MESA**

A professional framework for inferring stellar parameters using MESA stellar evolution models and modern Bayesian sampling techniques. Implements a hybrid Genetic Algorithm + Sequential Monte Carlo with Differential Evolution Markov Chain (GA+SMC-DEMC) sampler for efficient exploration of high-dimensional parameter spaces.

---

## Features

- **Intelligent Inlist Parsing**: Specify parameter ranges directly in standard MESA inlist format
- **Flexible Parameter Types**: Continuous, categorical, and integer parameters with automatic detection
- **Hybrid Sampling**: Genetic Algorithm for global exploration + SMC-DEMC for posterior refinement
- **Multiple Likelihood Functions**: SED fitting, photometric, and spectroscopic constraints
- **HPC Support**: Local execution, SLURM, and PBS schedulers
- **Comprehensive Results**: Posterior distributions, corner plots, convergence diagnostics

---

## Installation

### Requirements

- Python ≥ 3.8
- MESA stellar evolution code (with `MESA_DIR` environment variable set)
- NumPy, SciPy, Pandas, PyYAML, DEAP

### Install from source

```bash
git clone https://github.com/yourusername/mesa_infer.git
cd mesa_infer
pip install -e .
```

### Optional dependencies

```bash
pip install corner matplotlib  # For corner plots and visualization
```

---

## Quick Start

### 1. Prepare your MESA work directory

Your work directory should contain a standard MESA setup with `inlist_project` (or similar). Modify the inlist to specify parameter ranges using one of these formats:

**List syntax** (categorical or discrete values):
```fortran
&controls
    initial_mass = 0.8, 1.0, 1.2, 1.5, 2.0
    initial_z = 0.001, 0.01, 0.02
/
```

**Range syntax** (continuous parameters):
```fortran
&controls
    initial_mass_min = 0.8
    initial_mass_max = 2.0
    
    mixing_length_alpha_min = 1.5
    mixing_length_alpha_max = 2.5
/
```

### 2. Prepare observational data

Create a CSV file with your observational constraints. For SED fitting:

```csv
wavelength,flux,flux_error
3500.0,1.234e-15,1.0e-16
3600.0,1.456e-15,1.1e-16
3700.0,1.589e-15,1.2e-16
```

### 3. Run inference

```python
from mesa_infer import MESAInfer, InferConfig

# Load configuration
config = InferConfig.from_yaml('my_config.yaml')

# Or create programmatically
config = InferConfig(
    mesa_config={'work_dir': '/path/to/mesa/work'},
    sampler_config={'population_size': 64, 'num_generations': 100},
    likelihood_config={'data_type': 'sed'}
)

# Initialize and run
infer = MESAInfer(config)
infer.load_observations('observations.csv')
results = infer.run()

# Analyze results
results.summary()
results.make_corner_plot()
results.save('output_directory')
```

---

## Configuration

### YAML Configuration File

```yaml
# config.yaml
mesa:
  work_dir: "/path/to/mesa/work"
  mesa_dir: null  # Uses MESA_DIR environment variable if null
  inlist_name: "inlist_project"
  timeout: 7200
  scheduler: "local"  # or "slurm", "pbs"
  parallel_runs: 4

sampler:
  population_size: 64
  num_generations: 100
  crossover_probability: 0.5
  mutation_probability: 0.5
  tournament_size: 2
  elitism_fraction: 0.0625
  demc_hybrid: true
  demc_fraction: 0.5
  run_smc_refinement: true
  smc_nsamples: 50000

likelihood:
  data_type: "sed"  # or "photometric", "spectroscopic", "combined"
  wavelength_min: 3000.0
  wavelength_max: 10000.0
  extinction_av: 0.0
  extinction_rv: 3.1
```

### Parameter Specification in Inlist

The parser automatically detects exploration parameters:

| Pattern | Type | Example |
|---------|------|---------|
| `param = a, b, c` | Categorical | `initial_z = 0.001, 0.01, 0.02` |
| `param_min`, `param_max` | Continuous | `initial_mass_min = 0.8` / `initial_mass_max = 2.0` |
| Boolean list | Categorical | `use_ledoux = .true., .false.` |

Parameters not matching these patterns are treated as fixed values.

---

## Detailed Usage

### Using the Inlist Parser

```python
from mesa_infer import InlistParser

# Parse an inlist with parameter ranges
parser = InlistParser('/path/to/inlist_project')
parser.parse()

# View detected parameters
for name, spec in parser.parameters.items():
    print(f"{name}: {spec.param_type}, bounds={spec.bounds}")

# Manually add a parameter
parser.add_parameter(
    name='overshoot_f',
    param_type='continuous',
    bounds=[0.01, 0.03],
    section='controls'
)

# Generate a modified inlist with specific values
parser.generate_inlist(
    output_path='inlist_run1',
    parameter_values={'initial_mass': 1.2, 'initial_z': 0.015}
)
```

### Custom Likelihood Functions

```python
from mesa_infer.likelihood import SEDLikelihood, SpectroscopicLikelihood, ObservationalData

# Load observational data
obs = ObservationalData.from_csv('observations.csv', data_type='sed')

# SED likelihood with extinction
sed_like = SEDLikelihood(
    observations=obs,
    extinction_av=0.5,
    extinction_rv=3.1,
    wavelength_range=(3000, 10000),
    normalize=True
)

# Spectroscopic constraints
spec_obs = ObservationalData(
    teff=5777.0, teff_err=50.0,
    logg=4.44, logg_err=0.1,
    feh=0.0, feh_err=0.05
)
spec_like = SpectroscopicLikelihood(observations=spec_obs)
```

### Running the Samplers Directly

```python
from mesa_infer.samplers import GASampler, SMCDEMCSampler

# Define parameter specifications
parameters = {
    'initial_mass': {'type': 'continuous', 'bounds': [0.8, 2.0]},
    'initial_z': {'type': 'continuous', 'bounds': [0.001, 0.03], 'log_scale': True},
    'mixing_length_alpha': {'type': 'continuous', 'bounds': [1.5, 2.5]}
}

# Define your log-likelihood function
def log_likelihood(params):
    # params is a dict: {'initial_mass': 1.2, 'initial_z': 0.015, ...}
    # Run MESA, compute likelihood, return log(L)
    ...

# GA sampling
ga = GASampler(parameters, log_likelihood, population_size=64)
ga_results = ga.run(num_generations=100)

# SMC-DEMC refinement
smc = SMCDEMCSampler(parameters, log_likelihood)
posterior_samples = smc.run(
    initial_samples=ga_results['final_population'],
    n_samples=50000
)
```

### HPC Execution with SLURM

```python
config = InferConfig(
    mesa_config={
        'work_dir': '/path/to/work',
        'scheduler': 'slurm',
        'slurm_partition': 'compute',
        'slurm_time': '04:00:00',
        'slurm_nodes': 1,
        'slurm_ntasks': 1,
        'slurm_cpus_per_task': 4,
        'slurm_mem': '8G'
    }
)
```

---

## Output and Results

### Results Directory Structure

```
output/
├── config.yaml              # Configuration used
├── ga_samples.csv           # GA population history
├── posterior_samples.csv    # Final posterior samples
├── summary.json             # Parameter estimates and uncertainties
├── corner.png               # Corner plot (if matplotlib/corner installed)
├── diagnostics/
│   ├── convergence.csv      # Convergence metrics
│   └── acceptance.csv       # Acceptance rates
└── best_model/
    └── LOGS/                # MESA output for best-fit model
```

### Accessing Results

```python
from mesa_infer import InferenceResults

# Load saved results
results = InferenceResults.load('output/')

# Get posterior samples
samples = results.get_posteriors(burn_in=0.2, thin=10)

# Summary statistics
stats = results.get_summary_statistics()
print(f"Mass: {stats['initial_mass']['median']:.3f} +/- {stats['initial_mass']['std']:.3f}")

# Credible intervals
intervals = results.compute_credible_intervals(confidence=0.95)

# Correlation matrix
corr = results.compute_correlations()
```

---

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `MESAInfer` | Main interface for running inference |
| `InferConfig` | Configuration container |
| `InlistParser` | MESA inlist parsing and generation |
| `MESARunner` | MESA execution management |
| `InferenceResults` | Results storage and analysis |

### Samplers

| Class | Description |
|-------|-------------|
| `GASampler` | Genetic Algorithm with DE-MCMC hybrid |
| `SMCDEMCSampler` | SMC with DE-MH moves |

### Likelihood Functions

| Class | Description |
|-------|-------------|
| `SEDLikelihood` | Spectral energy distribution fitting |
| `PhotometricLikelihood` | Multi-band photometry |
| `SpectroscopicLikelihood` | Teff, logg, [Fe/H] constraints |

---

## Examples

See the `examples/` directory for complete worked examples:

- `basic_inference.py` - Simple single-star inference
- `sed_fitting.py` - SED fitting with extinction
- `hpc_slurm.py` - Running on a SLURM cluster
- `custom_likelihood.py` - Implementing custom constraints

---

## Citation

If you use this package in your research, please cite:

```bibtex
@software{mesa_infer,
  author = {Miller, Niall},
  title = {MESA_infer: Bayesian Stellar Parameter Inference with MESA},
  year = {2025},
  url = {https://github.com/yourusername/mesa_infer}
}
```

And the underlying methods:

- MESA: Paxton et al. (2011, 2013, 2015, 2018, 2019)
- SMC-DEMC: ter Braak & Vrugt (2008)

---

## License

MIT License. See `LICENSE` for details.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
