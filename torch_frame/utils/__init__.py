r"""Utility package."""
from .io import save, load
from .concat import cat
from .split import generate_random_split
from .infer_stype import infer_series_stype, infer_df_stype
from .compare import eq

__all__ = functions = [
    "save",
    "load",
    "cat",
    "eq",
    "generate_random_split",
    "infer_series_stype",
    "infer_df_stype",
]
