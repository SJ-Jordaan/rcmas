# Contributing to RCMAS

We welcome contributions to the RCMAS project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (Python version, OS, etc.)

### Suggesting Features

Feature requests are welcome! Please:
- Describe the feature and its use case
- Explain how it fits with RCMAS goals
- Provide examples if possible

### Pull Requests

1. **Fork and Clone**
   ```bash
   git fork <repository>
   git clone <your-fork>
   cd rcmas
   ```

2. **Create Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow code style guidelines (see DEVELOPMENT.md)
   - Add tests for new functionality
   - Update documentation as needed

4. **Test**
   ```bash
   pytest
   black src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

5. **Commit**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

6. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Review Process

- Maintainers will review your PR
- Address feedback and comments
- Once approved, your PR will be merged

## Code of Conduct

- Be respectful and constructive
- Focus on the technical merits
- Help others learn and grow

## Questions?

Contact the maintainers:
- Steven Jordaan: u18074848@tuks.co.za
- Nils Timm: ntimm@cs.up.ac.za

Thank you for contributing!
