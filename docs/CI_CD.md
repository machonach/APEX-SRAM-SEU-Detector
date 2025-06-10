## CI/CD Pipeline

This project includes a comprehensive continuous integration and continuous deployment pipeline implemented with GitHub Actions. The pipeline automates testing, code quality checks, and deployment processes.

### Pipeline Features

- **Automated Testing**: Runs the test suite to verify system functionality
- **Code Quality Checks**: Enforces coding standards with linters and formatters
- **Multi-environment Testing**: Tests across different Python versions
- **Code Coverage**: Tracks test coverage to ensure comprehensive testing
- **Containerization**: Builds Docker containers for easy deployment
- **Automated Deployment**: Publishes containers to registry on successful builds

### Workflow Components

1. **Build and Test**:
   - Runs on multiple Python versions (3.8, 3.9, 3.10)
   - Installs all project dependencies
   - Runs linters (flake8, black, isort)
   - Generates synthetic data for testing
   - Executes test suite with coverage reporting

2. **Docker Build**:
   - Triggered after successful tests on main/master branch
   - Builds separate containers for API server and dashboard
   - Tags images with both latest and commit SHA
   - Publishes images to container registry

### How to Use

The CI/CD pipeline runs automatically when code is pushed to the main/master branch or when a pull request is created. You can also manually trigger the workflow from the GitHub Actions tab.

### Local Development Integration

To ensure your code will pass CI checks locally before pushing:

1. Install development dependencies:
   ```
   pip install flake8 pytest pytest-cov black isort mypy pre-commit
   ```

2. Set up pre-commit hooks (recommended):
   ```
   pre-commit install
   ```
   
   This will automatically run checks before each commit.

3. Run lint checks manually:
   ```
   flake8 .
   black --check .
   isort --check .
   ```

4. Run tests with coverage:
   ```
   pytest test_seu_detector.py --cov=.
   ```
   
   Or use the provided script:
   ```
   python run_tests.py --coverage
   ```

### Configuration

The CI/CD pipeline can be configured by editing the `.github/workflows/ci_cd.yml` file.
