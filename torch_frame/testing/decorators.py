from importlib import import_module
from importlib.util import find_spec
from typing import Callable

import torch
from packaging.requirements import Requirement
from packaging.version import Version


def has_package(package: str) -> bool:
    r"""Returns :obj:`True` in case :obj:`package` is installed."""
    if '|' in package:
        return any(has_package(p) for p in package.split('|'))

    req = Requirement(package)
    if find_spec(req.name) is None:
        return False
    module = import_module(req.name)
    if not hasattr(module, '__version__'):
        return True

    version = Version(module.__version__).base_version
    return version in req.specifier


def withPackage(*args) -> Callable:
    r"""A decorator to skip tests if certain packages are not installed.
    Also supports version specification.
    """
    na_packages = {package for package in args if not has_package(package)}

    def decorator(func: Callable) -> Callable:
        import pytest
        return pytest.mark.skipif(
            len(na_packages) > 0,
            reason=f"Package(s) {na_packages} are not installed",
        )(func)

    return decorator


def withCUDA(func: Callable):
    r"""A decorator to test both on CPU and CUDA (if available)."""
    import pytest

    devices = [pytest.param(torch.device('cpu'), id='cpu')]
    if torch.cuda.is_available():
        devices.append(pytest.param(torch.device('cuda:0'), id='cuda:0'))

    return pytest.mark.parametrize('device', devices)(func)
