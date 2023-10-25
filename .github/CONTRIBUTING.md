# Contributing to PyTorch Frame

If you are interested in contributing to PyTorch Frame, your contributions will likely fall into one of the following two categories:

1. You want to implement a new feature:
   - In general, we accept any features as long as they fit the scope of this package. If you are unsure about this or need help on the design/implementation of your feature, post about it in an issue.
2. You want to fix a bug:
   - Feel free to send a Pull Request (PR) any time you encounter a bug. Please provide a clear and concise description of what the bug was. If you are unsure about if this is a bug at all or how to fix, post about it in an issue.

Once you finish implementing a feature or bug-fix, please send a PR to https://github.com/pyg-team/pytorch-frame.

Your PR will be merged after one or more rounds of reviews by the [pyg-team](https://github.com/pyg-team).


## Developing PyTorch Frame

To develop PyTorch Frame on your machine, here are some tips:

1. Ensure that you are running on one of the supported PyTorch versions (*e.g.*, `2.1.0`):

   ```python
   import torch
   print(torch.__version__)
   ```

2. Uninstall all existing PyTorch Frame installations.
   It is advised to run this command repeatedly to confirm that installations across all locations are properly removed.

   ```bash
   pip uninstall pytorch_frame
   ```

3. Fork and clone the PyTorch Frame repository:

   ```bash
   git clone https://github.com/<your_username>/pytorch-frame.git
   cd pytorch-frame

5. If you already cloned PyTorch Frame from source, update it:

   ```bash
   git pull
   ```

6. Install PyTorch Frame in editable mode:

   ```bash
   pip install -e ".[dev,full]"
   ```

   This mode will symlink the Python files from the current local source tree into the Python install.
   Hence, if you modify a Python file, you do not need to re-install PyTorch Frame again.

7. Ensure that you have a working PyTorch Frame installation by running the entire test suite with

   ```bash
   pytest
   ```

8. Install pre-commit hooks:

   ```bash
    pre-commit install
   ```

## Unit Testing

The PyTorch Frame testing suite is located under `test/`.
Run the test suite with

```bash
# all test cases
pytest

# individual test cases
pytest test/utils/test_split.py
```

## Continuous Integration

PyTorch Frame uses [GitHub Actions](https://github.com/pyg-team/pytorch-frame/actions) in combination with [CodeCov](https://codecov.io/github/pyg-team/pytorch-frame?branch=master) for continuous integration.

Everytime you send a Pull Request, your commit will be built and checked against the PyTorch Frame guidelines:

1. Ensure that your code is formatted correctly by testing against the styleguide of [`flake8`](https://github.com/PyCQA/flake8).
   We use the [`Flake8-pyproject`](https://pypi.org/project/Flake8-pyproject/) plugin for configuration:

   ```bash
   flake8
   ```

   If you do not want to format your code manually, we recommend to use [`yapf`](https://github.com/google/yapf).

2. Ensure that the entire test suite passes and that code coverage roughly stays the same.
   Please feel encouraged to provide a test with your submitted code.
   To test, either run

   ```bash
   pytest --cov
   ```

   (which runs a set of additional but time-consuming tests) dependening on your needs.

3. Add your feature/bugfix to the [`CHANGELOG.md`](https://github.com/pyg-team/pyotrch-frame/blob/master/CHANGELOG.md?plain=1).
   If multiple PRs move towards integrating a single feature, it is advised to group them together into one bullet point.

## Building Documentation

To build the documentation:

1. [Build and install](#developing-pytorch-frame) PyTorch Frame from source.
2. Install [Sphinx](https://www.sphinx-doc.org/en/master/) theme via
   ```bash
   pip install git+https://github.com/pyg-team/pyg_sphinx_theme.git
   ```
3. Generate the documentation via:
   ```bash
   cd docs
   make html
   ```
