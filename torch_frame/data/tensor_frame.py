import copy
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor

import torch_frame
from torch_frame import stype
from torch_frame.data.multi_nested_tensor import MultiNestedTensor
from torch_frame.typing import IndexSelectType, TensorData


class TensorFrame:
    r"""A tensor frame holds a :pytorch:`PyTorch` tensor for each table column.
    Table columns are organized into their semantic types
    :class:`~torch_frame.stype` (*e.g.*, categorical, numerical) and mapped to
    a compact tensor representation (*e.g.*, strings in a categorical column
    are mapped to indices from :obj:`{0, ..., num_categories - 1}`), and can be
    accessed through :obj:`feat_dict`.
    For instance, :obj:`feat_dict[stype.numerical]` stores a concatenated
    :pytorch:`PyTorch` tensor for all numerical features, where the first and
    second dimension represents the row and column in the original data frame,
    respectively.

    :class:`TensorFrame` handles missing values via :obj:`float('NaN')` for
    floating-point tensors, and :obj:`-1` otherwise.

    :obj:`col_names_dict` maps each column in :obj:`feat_dict` to their
    original column name.
    For example, :obj:`col_names_dict[stype.numerical][i]` stores the column
    name of :obj:`feat_dict[stype.numerical][:, i]`.

    Additionally, :class:`TensorFrame` can store any target values in :obj:`y`.

    .. code-block:: python

        import torch_frame

        tf = torch_frame.TensorFrame({
            feat_dict = {
                # Two numerical columns:
                torch_frame.numerical: torch.randn(10, 2),
                # Three categorical columns:
                torch_frame.categorical: torch.randint(0, 5, (10, 3)),
            },
            col_names_dict = {
                torch_frame.numerical: ['num_1', 'num_2'],
                torch_frame.categorical: ['cat_1', 'cat_2', 'cat_3'],

            },
        })

        print(len(tf))
        >>> 10

        # Row-wise filtering:
        tf = tf[torch.tensor([0, 2, 4, 6, 8])]
        print(len(tf))
        >>> 5

        # Transfer tensor frame to the GPU:
        tf = tf.to('cuda')
    """
    def __init__(
        self,
        feat_dict: Dict[torch_frame.stype, TensorData],
        col_names_dict: Dict[torch_frame.stype, List[str]],
        y: Optional[Tensor] = None,
    ):
        self.feat_dict = feat_dict
        self.col_names_dict = col_names_dict
        self.y = y
        self.validate()

    def validate(self):
        r"""Validates the :class:`TensorFrame` object."""
        if self.feat_dict.keys() != self.col_names_dict.keys():
            raise RuntimeError(
                f"The keys of feat_dict and col_names_dict must be the same, "
                f"but got {self.feat_dict.keys()} for feat_dict and "
                f"{self.col_names_dict.keys()} for col_names_dict.")

        num_rows = self.num_rows
        empty_stypes: List[stype] = []
        for stype_name, feats in self.feat_dict.items():
            num_cols = len(self.col_names_dict[stype_name])
            if num_cols == 0:
                empty_stypes.append(stype_name)

            if not isinstance(feats, dict):
                feats = [feats]
            else:
                feats = [feat for feat in feats.values()]

            for feat in feats:
                if feat.dim() < 2:
                    raise ValueError(f"feat_dict['{stype_name}'] must be at "
                                     f"least 2-dimensional")
                if num_cols != feat.size(1):
                    raise ValueError(
                        f"The expected number of columns for {stype_name} "
                        f"feature is {num_cols}, which does not align with "
                        f"the column dimensionality of "
                        f"feat_dict[{stype_name}] (got {feat.size(1)})")
                if feat.size(0) != num_rows:
                    raise ValueError(
                        f"The length of elements in feat_dict are "
                        f"not aligned, got {feat.size(0)} but "
                        f"expected {num_rows}.")

        if len(empty_stypes) > 0:
            raise RuntimeError(
                f"Empty columns for the following stypes: {empty_stypes}."
                f"Please manually delete the above stypes.")

        if self.y is not None:
            if len(self.y) != num_rows:
                raise ValueError(
                    f"The length of y is {len(self.y)}, which is not aligned "
                    f"with the number of rows ({num_rows}).")

    @property
    def stypes(self) -> List[stype]:
        r"""Returns a canonical ordering of stypes in :obj:`feat_dict`."""
        return list(
            filter(lambda x: x in self.feat_dict, list(torch_frame.stype)))

    @property
    def num_cols(self) -> int:
        r"""The number of columns in the :class:`TensorFrame`."""
        return sum(
            len(col_names) for col_names in self.col_names_dict.values())

    @property
    def num_rows(self) -> int:
        r"""The number of rows in the :class:`TensorFrame`."""
        feat = next(iter(self.feat_dict.values()))
        if isinstance(feat, dict):
            return len(next(iter(feat.values())))
        return len(feat)

    @property
    def device(self) -> torch.device:
        feat = next(iter(self.feat_dict.values()))
        if isinstance(feat, dict):
            return next(iter(feat.values())).device
        return feat.device

    # Python Built-ins ########################################################

    def __len__(self) -> int:
        return self.num_rows

    def __eq__(self, other: Any) -> bool:
        # Match instance type
        if not isinstance(other, TensorFrame):
            return False
        # Match length
        if len(self) != len(other):
            return False
        # Match target
        if self.y is not None:
            if other.y is None:
                return False
            elif not torch.allclose(other.y, self.y):
                return False
        else:
            if other.y is not None:
                return False
        # Match col_names_dict
        if self.col_names_dict != other.col_names_dict:
            return False
        # Match feat_dict
        for stype_name, self_feat in self.feat_dict.items():
            other_feat = other.feat_dict[stype_name]
            if isinstance(self_feat, Tensor):
                if not isinstance(other_feat, Tensor):
                    return False
                if self_feat.shape != other_feat.shape:
                    return False
                if not torch.allclose(self_feat, other_feat):
                    return False
            elif isinstance(self_feat, MultiNestedTensor):
                if not isinstance(other_feat, MultiNestedTensor):
                    return False
                if not MultiNestedTensor.allclose(self_feat, other_feat,
                                                  equal_nan=True):
                    return False
            elif isinstance(self_feat, dict):
                if not isinstance(other_feat, dict):
                    return False
                if self_feat.keys() != other_feat.keys():
                    return False
                for feat_name in self_feat.keys():
                    if not MultiNestedTensor.allclose(
                            self_feat[feat_name],
                            other_feat[feat_name],
                            equal_nan=True,
                    ):
                        return False
        return True

    def __neq__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        stype_repr = "\n".join([
            f"  {stype.value} ({len(col_names)}): {col_names},"
            for stype, col_names in self.col_names_dict.items()
        ])

        return (f"{self.__class__.__name__}(\n"
                f"  num_cols={self.num_cols},\n"
                f"  num_rows={self.num_rows},\n"
                f"{stype_repr}\n"
                f"  has_target={self.y is not None},\n"
                f"  device='{self.device}',\n"
                f")")

    def __getitem__(self, index: IndexSelectType) -> "TensorFrame":
        if isinstance(index, int):
            index = [index]

        def fn(x):
            if isinstance(x, dict):
                y = {}
                for key in x:
                    y[key] = x[key][index]
            else:
                return x[index]
            return y

        return self._apply(fn)

    def __copy__(self) -> "TensorFrame":
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value

        out.feat_dict = copy.copy(out.feat_dict)
        out.col_names_dict = copy.copy(out.col_names_dict)

        return out

    # Device Transfer #########################################################

    def to(self, *args, **kwargs):
        def fn(x):
            if isinstance(x, dict):
                for key in x:
                    x[key] = x[key].to(*args, **kwargs)
            else:
                x = x.to(*args, **kwargs)
            return x

        return self._apply(fn)

    def cpu(self, *args, **kwargs):
        def fn(x):
            if isinstance(x, dict):
                for key in x:
                    x[key] = x[key].cpu(*args, **kwargs)
            else:
                x = x.cpu(*args, **kwargs)
            return x

        return self._apply(fn)

    def cuda(self, *args, **kwargs):
        def fn(x):
            if isinstance(x, dict):
                for key in x:
                    x[key] = x[key].cuda(*args, **kwargs)
            else:
                x = x.cuda(*args, **kwargs)
            return x

        return self._apply(fn)

    # Helper Functions ########################################################

    def _apply(self, fn: Callable[[Tensor], Tensor]) -> "TensorFrame":
        out = copy.copy(self)
        out.feat_dict = {stype: fn(x) for stype, x in out.feat_dict.items()}
        if out.y is not None:
            out.y = fn(out.y)

        return out
