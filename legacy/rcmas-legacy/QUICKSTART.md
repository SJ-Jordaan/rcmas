# Quick Start Guide

## Initial Setup

Run the automated setup script:

```bash
./setup.sh
```

Or manually:

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate rcmas

# Install package with dev dependencies
pip install -e ".[dev]"
```

## Daily Workflow

1. **Activate environment:**

   ```bash
   conda activate rcmas
   ```

2. **Run your code:**

   ```bash
   python -m rcmas.example
   ```

3. **Run tests:**

   ```bash
   make test
   # or
   pytest
   ```

4. **Format code before committing:**
   ```bash
   make format
   make lint
   make type-check
   ```

## Development Commands

| Command           | Description                      |
| ----------------- | -------------------------------- |
| `make help`       | Show all available commands      |
| `make test`       | Run tests                        |
| `make coverage`   | Run tests with coverage report   |
| `make format`     | Format code with black and isort |
| `make lint`       | Run flake8 linter                |
| `make type-check` | Run mypy type checker            |
| `make clean`      | Clean build artifacts            |

## Project Structure Explained

```
rcmas/
├── src/rcmas/           # Source code package
│   ├── __init__.py      # Package initialization
│   └── example.py       # Example Z3 usage
├── tests/               # Test suite
│   ├── __init__.py
│   └── test_example.py  # Example tests
├── docs/                # Documentation
├── environment.yml      # Conda environment specification
├── requirements.txt     # Production dependencies
├── requirements-dev.txt # Development dependencies
├── setup.py            # Package setup (setuptools)
├── pyproject.toml      # Modern Python project config
├── .flake8             # Flake8 linter configuration
├── Makefile            # Development commands
├── setup.sh            # Automated setup script
└── README.md           # Project documentation
```

## Adding New Dependencies

1. **For conda packages:**

   ```bash
   conda install package-name
   # Then update environment.yml
   ```

2. **For pip packages:**

   ```bash
   pip install package-name
   # Then add to requirements.txt
   ```

3. **Update environment file:**
   ```bash
   conda env export --from-history > environment.yml
   ```

## Z3 Solver Examples

The project includes a simple example in `src/rcmas/example.py`. Run it:

```bash
python -m rcmas.example
```

Or use it programmatically:

```python
from rcmas import solve_simple_constraint

result = solve_simple_constraint()
print(result)  # {'x': 6, 'y': 4}
```

## Tips

- Always activate the conda environment before working: `conda activate rcmas`
- Run tests frequently to catch issues early: `pytest`
- Format your code before committing: `make format`
- Keep dependencies up to date: `conda update --all`
- Deactivate environment when done: `conda deactivate`
