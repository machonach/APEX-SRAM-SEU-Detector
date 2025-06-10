#!/usr/bin/env python3
"""
Run tests and generate coverage reports for SEU Detector
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Define project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

def install_test_dependencies():
    """Install test dependencies if needed"""
    try:
        import pytest
        import pytest_cov
    except ImportError:
        print("Installing test dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "pytest>=6.0.0", "pytest-cov>=2.0.0"
        ])

def run_tests(args):
    """Run tests with specified options"""
    # Create command with appropriate arguments
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add specific test module/class/method if provided
    if args.test_path:
        cmd.append(args.test_path)
    else:
        cmd.append("test_seu_detector.py")
    
    # Add coverage options
    if args.coverage:
        cmd.extend([
            f"--cov={args.coverage_source}", 
            "--cov-report=term", 
            "--cov-report=xml"
        ])
    
    # Add verbose flag if specified
    if args.verbose:
        cmd.append("-v")
    
    # Run the tests
    env = os.environ.copy()
    env["SIMULATION_MODE"] = "1"  # Force simulation mode for tests
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    
    return result.returncode

def main():
    """Main function to parse arguments and run tests"""
    parser = argparse.ArgumentParser(description="Run SEU Detector tests")
    parser.add_argument(
        "--test-path", 
        help="Specific test module, class, or method to run"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Generate coverage report"
    )
    parser.add_argument(
        "--coverage-source", 
        default=".", 
        help="Source directory for coverage measurement"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Ensure we have the needed test dependencies
    install_test_dependencies()
    
    # Run the tests
    return run_tests(args)

if __name__ == "__main__":
    sys.exit(main())
