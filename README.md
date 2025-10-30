# RCMAS

A Python project using the Z3 solver.

## Setup

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/SJ-Jordaan/rcmas.git
cd rcmas
```

2. Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate rcmas
```

3. Install the package in development mode:

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

```python
from rcmas import example

# Your code here
```

## Development

### Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=src --cov-report=html
```

### Code Formatting

```bash
# Format code with black
black src tests

# Sort imports
isort src tests

# Lint with flake8
flake8 src tests

# Type checking with mypy
mypy src
```

### Project Structure

```
rcmas/
├── src/
│   └── rcmas/
│       ├── __init__.py
│       └── example.py
├── tests/
│   ├── __init__.py
│   └── test_example.py
├── docs/
├── .gitignore
├── environment.yml
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── pyproject.toml
├── README.md
└── LICENSE
```

## License

MIT License - see LICENSE file for details.
