# Contributing to MESA_infer

Thank you for your interest in contributing to MESA_infer! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/mesa_infer.git
   cd mesa_infer
   ```
3. Create a virtual environment and install development dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```

## Development Workflow

### Branching

- Create a feature branch from `main`:
  ```bash
  git checkout -b feature/your-feature-name
  ```
- Use descriptive branch names:
  - `feature/` for new features
  - `fix/` for bug fixes
  - `docs/` for documentation changes

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for function arguments and return values
- Write docstrings for all public functions, classes, and modules
- Keep lines under 100 characters

### Testing

Run tests before submitting:
```bash
pytest tests/
```

Add tests for new functionality:
- Unit tests go in `tests/unit/`
- Integration tests go in `tests/integration/`

### Documentation

- Update docstrings for any modified code
- Update README.md if adding new features
- Add examples for new functionality

## Pull Request Process

1. Update your branch with the latest `main`:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. Ensure all tests pass:
   ```bash
   pytest tests/
   ```

3. Push your branch:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request on GitHub with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots/examples if applicable

5. Address review feedback

## Reporting Issues

When reporting issues, please include:

- Python version
- MESA version
- Operating system
- Complete error traceback
- Minimal code example to reproduce

## Feature Requests

Feature requests are welcome! Please:

- Check existing issues first
- Describe the use case
- Explain why this would be useful

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on what's best for the community

## Questions?

Open an issue with the "question" label or reach out to the maintainers.

Thank you for contributing!
