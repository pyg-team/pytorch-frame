import importlib
import sys
import types

import pytest

import torch_frame
from torch_frame._compat import pandas as pandas_compat


def test_cudf_pandas_status_is_reexported():
    assert torch_frame.is_cudf_pandas_enabled() == (
        pandas_compat.is_cudf_pandas_enabled())


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_cudf_pandas_env_flag_enabled(monkeypatch, value):
    monkeypatch.setenv(pandas_compat.CUDF_PANDAS_ENV, value)
    assert pandas_compat._env_flag_enabled(pandas_compat.CUDF_PANDAS_ENV)


@pytest.mark.parametrize("value", ["0", "false", "off", ""])
def test_cudf_pandas_env_flag_disabled(monkeypatch, value):
    monkeypatch.setenv(pandas_compat.CUDF_PANDAS_ENV, value)
    assert not pandas_compat._env_flag_enabled(pandas_compat.CUDF_PANDAS_ENV)


def test_cudf_pandas_env_import_installs_fake_module(monkeypatch):
    calls = []

    cudf = types.ModuleType("cudf")
    cudf.__path__ = []
    cudf_pandas = types.ModuleType("cudf.pandas")

    def install():
        calls.append("install")

    cudf_pandas.install = install
    cudf.pandas = cudf_pandas
    monkeypatch.setitem(sys.modules, "cudf", cudf)
    monkeypatch.setitem(sys.modules, "cudf.pandas", cudf_pandas)
    monkeypatch.setenv(pandas_compat.CUDF_PANDAS_ENV, "1")

    reloaded = importlib.reload(pandas_compat)
    try:
        assert reloaded.is_cudf_pandas_enabled()
        assert calls == ["install"]
    finally:
        monkeypatch.delenv(pandas_compat.CUDF_PANDAS_ENV, raising=False)
        importlib.reload(pandas_compat)


def test_cudf_pandas_detects_external_accelerator(monkeypatch):
    class ModuleAccelerator:
        pass

    monkeypatch.setattr(pandas_compat, "CUDF_PANDAS_ENABLED", False)
    monkeypatch.setattr(pandas_compat, "pd", ModuleAccelerator())

    assert pandas_compat.is_cudf_pandas_enabled()


def test_missing_cudf_pandas_has_helpful_error(monkeypatch):
    def raise_import_error(name):
        raise ImportError(name)

    monkeypatch.setattr(pandas_compat, "import_module", raise_import_error)

    with pytest.raises(ImportError, match="Install RAPIDS cuDF"):
        pandas_compat._install_cudf_pandas()
