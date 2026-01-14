# Contributing to Deepfake Detection Project

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Getting Started

1. **Fork the repository** (if contributing externally)
2. **Clone your fork**:
   ```bash
   git clone <your-fork-url>
   cd dfprojectv2
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv dfenv
   source dfenv/bin/activate  # On Windows: dfenv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

1. **Make your changes** following the project structure
2. **Test your changes**:
   ```bash
   python -m pytest tests/
   ```

3. **Check code style**:
   ```bash
   flake8 .
   black --check .
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

## Code Style Guidelines

- Follow PEP 8 for Python code
- Use type hints where possible
- Write docstrings for all functions and classes
- Keep functions focused and small
- Add comments for complex logic

## Commit Message Guidelines

Use clear, descriptive commit messages:

- `feat: Add new feature`
- `fix: Fix bug in data loader`
- `docs: Update README`
- `refactor: Refactor model architecture`
- `test: Add unit tests for EDA module`
- `chore: Update dependencies`

## Project Structure

```
dfprojectv2/
├── notebooks/      # Jupyter notebooks for EDA
├── eda/            # EDA modules
├── data/           # Data pipeline
├── models/         # Model architectures
├── training/       # Training utilities
├── inference/      # Inference API
├── configs/        # Configuration files
└── tests/          # Unit tests
```

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for good test coverage

## Documentation

- Update README.md if adding major features
- Add docstrings to new functions/classes
- Update relevant notebooks if EDA changes

## Questions?

Feel free to open an issue for questions or discussions.

Thank you for contributing! 🎉

