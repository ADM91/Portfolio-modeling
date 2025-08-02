#!/usr/bin/env python3
"""
Test runner script for the Portfolio Modeling database layer.
Provides convenient commands for running different test suites.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle the output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run database tests")
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "all", "coverage", "fast", "slow", "financial"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--fail-fast", "-x",
        action="store_true",
        help="Stop on first failure"
    )
    
    args = parser.parse_args()
    
    # Base pytest command - use virtual environment python if available
    import os
    venv_python = Path(".venv/Scripts/python.exe")
    if venv_python.exists():
        base_cmd = [str(venv_python), "-m", "pytest"]
    else:
        base_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        base_cmd.append("-v")
    
    if args.parallel:
        base_cmd.extend(["-n", "auto"])
    
    if args.fail_fast:
        base_cmd.append("-x")
    
    # Test type specific commands
    commands = []
    
    if args.test_type == "unit":
        commands.append((
            base_cmd + ["-m", "unit", "tests/test_database_access.py"],
            "Unit Tests"
        ))
    
    elif args.test_type == "integration":
        commands.append((
            base_cmd + ["-m", "integration", "tests/integration/"],
            "Integration Tests"
        ))
    
    elif args.test_type == "fast":
        commands.append((
            base_cmd + ["-m", "not slow", "tests/"],
            "Fast Tests (excluding slow tests)"
        ))
    
    elif args.test_type == "slow":
        commands.append((
            base_cmd + ["-m", "slow", "tests/"],
            "Slow Tests"
        ))
    
    elif args.test_type == "financial":
        commands.append((
            base_cmd + ["-m", "financial", "tests/"],
            "Financial Calculation Tests"
        ))
    
    elif args.test_type == "coverage":
        commands.append((
            base_cmd + [
                "--cov=database",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--cov-fail-under=80",
                "tests/"
            ],
            "All Tests with Coverage Report"
        ))
    
    elif args.test_type == "all":
        commands.extend([
            (base_cmd + ["-m", "unit", "tests/test_database_access.py"], "Unit Tests"),
            (base_cmd + ["-m", "integration", "tests/integration/"], "Integration Tests"),
            (base_cmd + [
                "--cov=database",
                "--cov-report=html",
                "--cov-report=term-missing",
                "tests/"
            ], "Coverage Report")
        ])
    
    # Run commands
    success_count = 0
    total_count = len(commands)
    
    for cmd, description in commands:
        if run_command(cmd, description):
            success_count += 1
        else:
            if args.fail_fast:
                break
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
