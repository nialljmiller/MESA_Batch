# MESA Batch Runner

A Python package for running parameter sweeps of [MESA](https://docs.mesastar.org/) stellar evolution simulations.

## Features

- **Dual interface**: Use as a Python library or command-line tool
- **Flexible parameter specification**: Define sweeps with explicit values or ranges (linear/logarithmic)
- **Automatic output organization**: Each run's outputs are stored in separate directories
- **Results collection**: Extract data at specific evolutionary stages (e.g., TAMS)
- **Progress tracking**: Runtime logging and status reporting

## Installation

```bash
pip install mesa-batch-runner
```

Or install from source:

```bash
git clone https://github.com/yourusername/mesa-batch-runner.git
cd mesa-batch-runner
pip install -e .
```

## Quick Start

### Python API

```python
from mesa_batch_runner import BatchRunner

# Initialize with path to MESA work directory
runner = BatchRunner("/path/to/mesa/work")

# Add parameters to sweep
runner.add_parameter("initial_mass", values=[1.0, 2.0, 5.0, 10.0])
runner.add_parameter("initial_z", min=0.001, max=0.02, steps=5)

# Run all combinations
results = runner.run()

# Collect and export results
collector = runner.collect_results()
collector.to_csv("results.csv", stage="tams")
```

### Command Line

```bash
# Create a template configuration file
mesa-batch init

# Edit batch_inlist to configure your sweep, then run
mesa-batch run /path/to/mesa/work --config batch_inlist

# Check status of runs
mesa-batch status batch_runs/runs

# Collect results
mesa-batch collect batch_runs/runs -o results.csv
```

## Configuration File Format

The `batch_inlist` configuration file uses a Fortran-namelist-like format:

```fortran
&batch_control
    work_dir = '/path/to/mesa/work'
    output_dir = '/path/to/output'
    timeout = 7200
    continue_on_error = .true.
/

&parameters
    ! Explicit list of values
    initial_mass = 1.0, 2.0, 5.0, 10.0
    
    ! Range with number of steps
    initial_z_min = 0.001
    initial_z_max = 0.02
    initial_z_steps = 5
    
    ! Range with step size
    mixing_length_alpha_min = 1.5
    mixing_length_alpha_max = 2.5
    mixing_length_alpha_step = 0.1
    
    ! Logarithmic spacing
    some_param_min = 1e-4
    some_param_max = 1e-2
    some_param_steps = 10
    some_param_log = .true.
/
```

## How It Works

1. **Grid Generation**: The package generates all combinations of specified parameters
2. **Inlist Modification**: For each combination, it modifies the template inlist with the parameter values
3. **Execution**: Runs MESA and captures output to a log file
4. **Output Collection**: Copies LOGS, photos, and model files to a dedicated run directory
5. **Results Extraction**: Provides tools to extract data at specific evolutionary stages

## Output Structure

```
batch_runs/
├── runs/
│   ├── mass1.0_z0.001/
│   │   ├── LOGS/
│   │   │   ├── history.data
│   │   │   └── profile*.data
│   │   ├── photos/
│   │   ├── inlist_project
│   │   └── run.log
│   ├── mass1.0_z0.005/
│   └── ...
├── run_timings.csv
└── batch_summary.json
```

## API Reference

### BatchRunner

The main class for configuring and executing batch runs.

```python
runner = BatchRunner(
    work_dir="/path/to/mesa/work",
    output_dir="/path/to/output",  # Optional
    name="my_batch",               # Optional
    timeout=3600,                  # Optional, seconds
    continue_on_error=True,        # Continue if a run fails
)

# Add parameters
runner.add_parameter("initial_mass", values=[1.0, 2.0])
runner.add_parameter("initial_z", min=0.001, max=0.02, steps=5, log_scale=False)

# Execute
results = runner.run(force=False, max_runs=None)
```

### ParameterGrid

For programmatic grid construction:

```python
from mesa_batch_runner import ParameterGrid

grid = ParameterGrid()
grid.add("initial_mass", values=[1.0, 2.0, 5.0])
grid.add("initial_z", min_val=0.001, max_val=0.02, steps=5)

print(grid.summary())
print(f"Total combinations: {len(grid)}")

for params in grid:
    print(params)
```

### ResultsCollector

For analyzing completed runs:

```python
from mesa_batch_runner import ResultsCollector

collector = ResultsCollector("batch_runs/runs")

# Get summary
print(collector.summary())

# Extract TAMS values
tams_data = collector.extract_at_tams(columns=["log_Teff", "log_L", "star_age"])

# Export to CSV
collector.to_csv("results.csv", stage="tams")
```

## Requirements

- Python 3.8+
- MESA installed and configured
- numpy
- pandas
- mesa_reader

## License

MIT License
