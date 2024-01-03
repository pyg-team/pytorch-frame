from __future__ import annotations

from collections import Counter, defaultdict
# typing.Dict is necessary for TypeVar definition in Python 3.8
from typing import Dict, TypeVar

import torch
from torch import Tensor

import torch_frame
from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor
from torch_frame.data.multi_nested_tensor import MultiNestedTensor
from torch_frame.data.tensor_frame import TensorFrame
from torch_frame.typing import TensorData

T = TypeVar(
    "T",
    TensorFrame,
    Tensor,
    MultiEmbeddingTensor,
    MultiNestedTensor,
    Dict[str, MultiNestedTensor],
)
TD = TypeVar(
    "TD",
    Tensor,
    MultiEmbeddingTensor,
    MultiNestedTensor,
    Dict[str, MultiNestedTensor],
)


def cat(
    lst: list[T],
    dim: int,
) -> T:
    r"""Concatenates a list of :class:`TensorFrame` or :class:`TensorData`.

    Args:
        lst (list[TensorFrame] | list[TensorData]): A list of tensor frames to
            or tensor data to concatenate.
        dim (int): Dimension to concatenate tensor data.

    Returns:
        TensorFrame | TensorData: Concatenated output.
    """
    if isinstance(lst[0], TensorFrame):
        return _cat_tensor_frame(lst, dim=dim)
    else:
        return _cat_tensor_data(lst, dim=dim)


def _cat_tensor_data(td_list: list[TD], dim: int) -> TD:
    r"""Concatenates a list of :class:`TensorData` objects along a specified
    dimension (:obj:`0` or :obj:`1`).

    Args:
        td_list (List[TensorData]): A list of tensor data to concatenate.
        dim (int): Dimension to concatenate tensor data.

    Returns:
        td (TensorData): Concatenated tensor data.
    """
    if len(td_list) == 0:
        raise ValueError("Cannot concatenate an empty list.")
    elif len(td_list) == 1:
        return td_list[0]
    else:
        for td in td_list:
            if not isinstance(td, td_list[0].__class__):
                raise RuntimeError(
                    "The TensorData to be concatenated "
                    f"must be of the same type, got {td.__class__} "
                    f" and {td_list[0].__class__}.")
        td = td_list[0]
        if isinstance(td, Tensor):
            return torch.cat(td_list, dim=dim)
        elif isinstance(td, MultiEmbeddingTensor):
            return MultiEmbeddingTensor.cat(td_list, dim=dim)
        elif isinstance(td, MultiNestedTensor):
            return MultiNestedTensor.cat(td_list, dim=dim)
        elif isinstance(td, dict):
            result = {}
            for name in td.keys():
                result[name] = MultiNestedTensor.cat(
                    [td_dict[name] for td_dict in td_list], dim=dim)
            return result
        else:
            raise NotImplementedError(
                f"Concatenation not implemented for {type(td)}")


def _cat_tensor_frame(tf_list: list[TensorFrame], dim: int) -> TensorFrame:
    r"""Concatenates a list of :class:`TensorFrame` objects along a specified
    dimension (:obj:`0` or :obj:`1`). If :obj:`dim` is set to :obj:`0`, this
    will concatenate the tensor frames along the rows, keeping columns
    unchanged. If :obj:`dim` is set to :obj:`1`, this will concatenate the
    tensor frames along the columns, which increases the columns while keeping
    the rows unchanged.

    Args:
        tf_list (List[TensorFrame]): A list of tensor frames to concatenate.
        dim (int): Dimension to concatenate tensor frames.

    Returns:
        tf (TensorFrame): Concatenated tensor frame.
    """
    if len(tf_list) == 0:
        raise RuntimeError(
            "Cannot concatenate an empty list of tensor frames.")
    if dim == 0:
        return _cat_row(tf_list)
    elif dim == 1:
        return _cat_col(tf_list)
    else:
        raise ValueError(f"Unsupported concatenation dim={dim}.")


def _cat_helper(
    tf_list: list[TensorFrame],
    dim: int,
) -> dict[torch_frame.stype, TensorData]:
    r"""Helper function that takes a list of :class:`TensorFrame` objects and
    returns :obj:`feat_dict` of the concatenated :class:`TensorFrame` object.
    """
    feat_list_dict: dict[torch_frame.stype,
                         list[TensorData]] = defaultdict(list)
    for tf in tf_list:
        for stype, feat in tf.feat_dict.items():
            feat_list_dict[stype].append(feat)

    feat_dict: dict[torch_frame.stype, TensorData] = {}
    for stype, feat_list in feat_list_dict.items():
        feat_dict[stype] = _cat_tensor_data(
            feat_list,
            dim=dim,
        )  # type: ignore [type-var]
    return feat_dict


def _cat_row(tf_list: list[TensorFrame]) -> TensorFrame:
    col_names_dict = tf_list[0].col_names_dict
    for tf in tf_list[1:]:
        if tf.col_names_dict != col_names_dict:
            raise RuntimeError(
                f"Cannot perform cat(..., dim=0) since col_names_dict's "
                f"of given tensor frames do not match (expect all "
                f"{col_names_dict}).")
    if tf_list[0].y is None:
        if not all([tf.y is None for tf in tf_list]):
            raise RuntimeError(
                "Cannot perform cat(..., dim=0) since 'y' attribute "
                "types of given tensor frames do not match (expect all "
                " `None`).")
    else:
        if not all([tf.y is not None for tf in tf_list]):
            raise RuntimeError(
                "Cannot perform cat(..., dim=0) since 'y' attribute "
                "types of given tensor frames do not match (expect all "
                "`Tensor`).")

    y = None
    if tf_list[0].y is not None:
        ys: list[Tensor] = []
        for tf in tf_list:
            assert tf.y is not None
            ys.append(tf.y)
        y = torch.cat(ys, dim=0)
    return TensorFrame(feat_dict=_cat_helper(tf_list, dim=0),
                       col_names_dict=tf_list[0].col_names_dict, y=y)


def _get_duplicates(lst: list[str]) -> list[str]:
    count = Counter(lst)
    return [item for item, count in count.items() if count > 1]


def _cat_col(tf_list: list[TensorFrame]) -> TensorFrame:
    idx_with_non_nan_y = []
    for i, tf in enumerate(tf_list):
        if tf.y is not None:
            idx_with_non_nan_y.append(i)
    if len(idx_with_non_nan_y) == 0:
        y = None
    elif len(idx_with_non_nan_y) == 1:
        y = tf_list[idx_with_non_nan_y[0]].y
    else:
        raise RuntimeError(
            "Cannot perform cat(..., dim=1) since given tensor frames "
            "contain more than one tensor frame with non-None y attribute.")

    col_names_dict: dict[torch_frame.stype, list[str]] = defaultdict(list)
    for tf in tf_list:
        for stype in tf.col_names_dict.keys():
            col_names_dict[stype].extend(tf.col_names_dict[stype])

    # Check duplicates in col_names_dict
    for stype, col_names in col_names_dict.items():
        duplicates = _get_duplicates(col_names)
        if len(duplicates) > 0:
            raise RuntimeError(
                f"Cannot perform cat(..., dim=1) since {stype} contains "
                f"duplicated column names: {duplicates}.")

    return TensorFrame(feat_dict=_cat_helper(tf_list, dim=1),
                       col_names_dict=col_names_dict, y=y)
