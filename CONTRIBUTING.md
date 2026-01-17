# Contributing to Lie Algebra Smoothness

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/soorajkcphd/lie-algebra-smoothness.git`
3. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
4. Install development dependencies: `pip install -e ".[dev]"`

## Development Workflow

### Code Style

We use the following tools for code quality:

```bash
# Format code
black src/ tests/ experiments/
isort src/ tests/ experiments/

# Check style
flake8 src/ tests/ experiments/

# Type checking
mypy src/lie_smoothness/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/lie_smoothness --cov-report=html

# Run specific test file
pytest tests/test_algebras.py -v
```

### Making Changes

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass: `pytest`
5. Format your code: `black . && isort .`
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Pull Request Guidelines

- Include a clear description of the changes
- Reference any related issues
- Add tests for new functionality
- Update documentation if needed
- Ensure CI passes

## Reporting Issues

When reporting issues, please include:

1. Python version (`python --version`)
2. Package versions (`pip freeze`)
3. Minimal code to reproduce the issue
4. Expected vs actual behavior
5. Full error traceback

## Code of Conduct

Please be respectful and constructive in all interactions.

## Questions?

Open an issue or contact the maintainers:
- sooraj.kc@alliance.edu.in
