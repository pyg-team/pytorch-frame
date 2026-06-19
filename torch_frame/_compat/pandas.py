from __future__ import annotations

import os
from importlib import import_module
from typing import Final

CUDF_PANDAS_ENV: Final = "PYTORCH_FRAME_USE_CUDF_PANDAS"
_TRUE_VALUES: Final = {"1", "true", "yes", "on"}


def _env_flag_enabled(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip().lower() in _TRUE_VALUES


def _install_cudf_pandas() -> bool:
    try:
        cudf_pandas = import_module("cudf.pandas")
    except ImportError as exc:
        raise ImportError(
            f"{CUDF_PANDAS_ENV}=1 was set, but 'cudf.pandas' could not be "
            "imported. Install RAPIDS cuDF in this environment or unset "
            f"{CUDF_PANDAS_ENV}.") from exc

    cudf_pandas.install()
    return True


CUDF_PANDAS_ENABLED: Final = (_install_cudf_pandas()
                              if _env_flag_enabled(CUDF_PANDAS_ENV) else False)

import pandas as pd  # noqa: E402
import pandas.api.types as ptypes  # noqa: E402


def is_cudf_pandas_enabled() -> bool:
    r"""Returns whether pandas imports are backed by ``cudf.pandas``."""
    return (CUDF_PANDAS_ENABLED
            or pd.__class__.__name__ == "ModuleAccelerator")
