from __future__ import annotations

import copy
from inspect import signature
from typing import Any

import torch


class Module(torch.nn.Module):
    r"""A base class for defining modules in which attributes may be defined
    in a later stage. As such, users only need to define the dynamic
    hyperparameters of a module, and do not need to care about connecting the
    module to the underlying data, e.g., specifying the number of input or
    output channels.

    This is achieved by postponing submodule creation
    (via :meth:`init_modules`) until all attributes in :obj:`LAZY_ATTRS` are
    fully-specified.
    """
    LAZY_ATTRS: set[str] = set()

    def init_modules(self):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__()

        self._in_init = True
        self._missing_attrs = copy.copy(self.LAZY_ATTRS)

        for key, value in zip(signature(self.__init__).parameters, args):
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

        self._in_init = False

        if self.is_fully_specified:
            self._init_modules()

    def __setattr__(self, key: str, value: Any):
        super().__setattr__(key, value)
        if value is not None and key in getattr(self, '_missing_attrs', {}):
            self._missing_attrs.remove(key)
            if not self._in_init and self.is_fully_specified:
                self._init_modules()

    @property
    def is_fully_specified(self) -> bool:
        return len(self._missing_attrs) == 0

    def validate(self):
        if len(self._missing_attrs) > 0:
            raise ValueError(
                f"The '{self.__class__.__name__}' module is not fully-"
                f"specified yet. It is missing the following attribute(s): "
                f"{self._missing_attrs}. Please specify them before using "
                f"this module in a deep learning pipeline.")

    def _init_modules(self):
        self.validate()
        self.init_modules()

    def _apply(self, *args, **kwargs) -> Any:
        self.validate()
        return super()._apply(*args, **kwargs)

    def named_parameters(self, *args, **kwargs) -> Any:
        self.validate()
        return super().named_parameters(*args, **kwargs)

    def named_children(self) -> Any:
        self.validate()
        return super().named_children()

    def named_modules(self, *args, **kwargs) -> Any:
        self.validate()
        return super().named_modules(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        self.validate()
        return super().__call__(*args, **kwargs)
