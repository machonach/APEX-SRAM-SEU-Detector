# SEU Detector CI/CD Implementation

This document describes the CI/CD pipeline implemented for the SEU Detector project.

## Overview

The Continuous Integration/Continuous Deployment (CI/CD) pipeline automates testing, code quality verification, and container builds for the SEU Detector project. The pipeline is implemented using GitHub Actions and consists of multiple stages that run when code is pushed to the repository or pull requests are opened.

## Files Added

1. `.github/workflows/ci_cd.yml` - Main CI/CD workflow for the project
2. `.github/workflows/pr_checks.yml` - Quick checks for pull requests
3. `.flake8` - Configuration file for the Flake8 linter
4. `pytest.ini` - Configuration file for pytest and coverage reports
5. `Dockerfile.api` - Docker container definition for the API server
6. `Dockerfile.dashboard` - Docker container definition for the dashboard
7. `run_tests.py` - Helper script for running tests with coverage reports

## CI/CD Pipeline Stages

### 1. Build and Test

This stage runs on every push to main/master branches and pull requests:

- Sets up Python environments (3.8, 3.9, 3.10)
- Installs project dependencies
- Runs code quality checks using flake8, black, and isort
- Generates synthetic test data
- Runs unit tests with pytest
- Generates code coverage reports

### 2. Docker Build

This stage runs only on pushes to main/master branches:

- Builds Docker containers for API server and dashboard
- Tags containers with commit SHA and "latest"
- Pushes containers to DockerHub (requires setting up secrets)

## Pull Request Checks

For pull requests, a simplified workflow runs to quickly verify code changes:

- Runs basic code formatting checks
- Runs a subset of tests for quick feedback
- Shows code coverage change information

## Local Development Integration

You can run the same checks locally before pushing code:

1. Install development tools:
   ```
   pip install flake8 pytest pytest-cov black isort
   ```

2. Run the test suite:
   ```
   python run_tests.py --coverage
   ```

3. Check code formatting:
   ```
   black --check .
   isort --check .
   flake8 .
   ```

## Using the Start Script

The start.py script has been updated to include test running functionality:

- Run all tests: `python start.py --test`
- Use the interactive menu: `python start.py` → Select "Run test suite"

## Next Steps

To complete the CI/CD implementation:

1. Set up DockerHub credentials as GitHub repository secrets:
   - DOCKER_USERNAME
   - DOCKER_PASSWORD

2. Consider implementing automatic deployment to cloud services

3. Set up notification system for build results

4. Implement pre-commit hooks for local development

## Benefits

This CI/CD pipeline provides several benefits:

- Ensures code quality through automated testing
- Catches issues early through pull request checks
- Provides consistent builds across environments
- Automates repetitive tasks in the development workflow
- Makes deployment more reliable and reproducible
