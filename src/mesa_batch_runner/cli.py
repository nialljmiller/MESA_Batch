"""
Command-line interface for MESA batch runner.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

from . import __version__
from .batch import BatchRunner
from .config import BatchConfig, EXAMPLE_BATCH_INLIST
from .results import ResultsCollector


def main(args: list[str] | None = None) -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    if parsed_args.command is None:
        parser.print_help()
        return 0
    
    try:
        if parsed_args.command == "run":
            return cmd_run(parsed_args)
        elif parsed_args.command == "init":
            return cmd_init(parsed_args)
        elif parsed_args.command == "collect":
            return cmd_collect(parsed_args)
        elif parsed_args.command == "status":
            return cmd_status(parsed_args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if parsed_args.debug:
            raise
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="mesa-batch",
        description="MESA Batch Runner - Run parameter sweeps for MESA stellar evolution models",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Execute a batch run",
    )
    run_parser.add_argument(
        "work_dir",
        nargs="?",
        help="Path to MESA work directory",
    )
    run_parser.add_argument(
        "-c", "--config",
        dest="config_file",
        help="Path to batch_inlist configuration file",
    )
    run_parser.add_argument(
        "-o", "--output",
        dest="output_dir",
        help="Output directory for results",
    )
    run_parser.add_argument(
        "-n", "--name",
        help="Name for this batch run",
    )
    run_parser.add_argument(
        "--timeout",
        type=float,
        help="Timeout per run in seconds",
    )
    run_parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    run_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    run_parser.add_argument(
        "--max-runs",
        type=int,
        help="Maximum number of runs (for testing)",
    )
    
    # Init command
    init_parser = subparsers.add_parser(
        "init",
        help="Create a template batch_inlist file",
    )
    init_parser.add_argument(
        "output",
        nargs="?",
        default="batch_inlist",
        help="Output filename (default: batch_inlist)",
    )
    init_parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Overwrite existing file",
    )
    
    # Collect command
    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect results from completed runs",
    )
    collect_parser.add_argument(
        "runs_dir",
        help="Path to runs directory",
    )
    collect_parser.add_argument(
        "-o", "--output",
        default="results.csv",
        help="Output CSV filename (default: results.csv)",
    )
    collect_parser.add_argument(
        "--stage",
        default="tams",
        choices=["tams"],
        help="Evolutionary stage to extract (default: tams)",
    )
    
    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show status of batch runs",
    )
    status_parser.add_argument(
        "runs_dir",
        help="Path to runs directory",
    )
    
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the run command."""
    # Load configuration
    if args.config_file:
        config_path = Path(args.config_file)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_path}", file=sys.stderr)
            return 1
        
        config = BatchConfig.from_file(config_path)
        
        # Override with command line arguments
        if args.work_dir:
            config.work_dir = args.work_dir
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.name:
            config.name = args.name
        if args.timeout:
            config.timeout = args.timeout
        if args.quiet:
            config.quiet = True
        
        # Validate
        errors = config.validate()
        if errors:
            for error in errors:
                print(f"Configuration error: {error}", file=sys.stderr)
            return 1
        
        # Create runner from config
        runner = BatchRunner.from_dict(config.to_dict())
        
    else:
        # No config file - need work_dir at minimum
        if not args.work_dir:
            print("Error: Must specify work_dir or --config", file=sys.stderr)
            return 1
        
        runner = BatchRunner(
            work_dir=args.work_dir,
            output_dir=args.output_dir,
            name=args.name,
            timeout=args.timeout,
            quiet=args.quiet,
        )
        
        # Check for batch_inlist in work directory
        batch_inlist = Path(args.work_dir) / "batch_inlist"
        if batch_inlist.exists():
            print(f"Found batch_inlist in work directory, loading configuration...")
            config = BatchConfig.from_file(batch_inlist)
            for param in config.parameters:
                runner.add_parameter(**param.to_dict())
    
    # Set dry run mode
    runner.dry_run = args.dry_run
    
    # Check we have parameters
    if runner.n_runs == 0:
        print("No parameters configured. Use --config or create a batch_inlist file.")
        print("Run 'mesa-batch init' to create a template.")
        return 1
    
    # Execute
    results = runner.run(force=args.force, max_runs=args.max_runs)
    
    # Report
    n_success = sum(1 for r in results if r.success)
    n_failed = len(results) - n_success
    
    if n_failed > 0:
        return 1
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Create a template batch_inlist file."""
    output_path = Path(args.output)
    
    if output_path.exists() and not args.force:
        print(f"Error: {output_path} already exists. Use --force to overwrite.")
        return 1
    
    output_path.write_text(EXAMPLE_BATCH_INLIST)
    print(f"Created template configuration: {output_path}")
    print("Edit this file to configure your parameter sweep, then run:")
    print(f"  mesa-batch run --config {output_path}")
    
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    """Collect results from completed runs."""
    runs_dir = Path(args.runs_dir)
    
    if not runs_dir.exists():
        print(f"Error: Runs directory not found: {runs_dir}", file=sys.stderr)
        return 1
    
    collector = ResultsCollector(runs_dir)
    print(collector.summary())
    print()
    
    collector.to_csv(args.output, stage=args.stage)
    
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show status of batch runs."""
    runs_dir = Path(args.runs_dir)
    
    if not runs_dir.exists():
        print(f"Error: Runs directory not found: {runs_dir}", file=sys.stderr)
        return 1
    
    collector = ResultsCollector(runs_dir)
    print(collector.summary())
    print()
    
    # Show individual run status
    print("Run status:")
    for run in collector.runs:
        status = "✓" if run.completed else "✗"
        history = "H" if run.has_history else "-"
        profiles = "P" if run.has_profiles else "-"
        print(f"  {status} [{history}{profiles}] {run.name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
