from typing import Dict, List

import torch
from torch import Tensor

import torch_frame
from torch_frame import TensorFrame


def cat_tf(tf_list: List[TensorFrame], along: str) -> TensorFrame:
    r"""Concatenates a list of :class:`TensorFrame` objects along a specified
    axis (:obj:`row` or :obj:`col`). If set to :obj:`row`, this will
    concatenate the tensor frames along the rows, keeping columns unchanced.
    If set to :obj:`col`, this will concatenate the tensor frames along the
    columns, which increases the columns while keeping the rows unchanced.

    Args:
        tf_list (List[TensorFrame]): A list of tensor frames to concatenate.
        along (str): Specifies whether to concatenate along :obj:`row` or
            :obj:`col`.

    Returns:
        tf (TensorFrame): Concatenated tensor frame.
    """
    if len(tf_list) == 0:
        raise RuntimeError(
            "Cannot concatenate an empty list of tensor frames.")
    if along == 'row':
        return _cat_tf_row(tf_list)
    elif along == 'col':
        return _cat_tf_col(tf_list)
    else:
        raise ValueError(
            f"'along' must be either 'row' or 'col' (got {along}).")


def _cat_tf_row(tf_list: List[TensorFrame]) -> TensorFrame:
    feat_dict: Dict[torch_frame.stype, List[Tensor]] = {}
    for stype in tf_list[0].feat_dict.keys():
        feat_dict[stype] = torch.cat(
            [tf.feat_dict[stype] for tf in tf_list],
            dim=0,
        )
    y = None
    if tf_list[0].y is not None:
        y = torch.cat([tf.y for tf in tf_list], dim=0)
    return TensorFrame(feat_dict=feat_dict,
                       col_names_dict=tf_list[0].col_names_dict, y=y)


def _raise_on_non_matching_y(tf_list: List[TensorFrame]):
    msg = "torch.cat_tf(along = 'col') requires y's in tensor frames to match."
    y = tf_list[0].y
    for tf in tf_list[1:]:
        if y is None:
            if tf.y is not None:
                raise RuntimeError(msg)
        else:
            if tf.y is None:
                raise RuntimeError(msg)
            elif not torch.allclose(y, tf.y):
                raise RuntimeError(msg)


def _cat_tf_col(tf_list: List[TensorFrame]) -> TensorFrame:
    _raise_on_non_matching_y(tf_list)
    y = tf_list[0].y
    # Gather all stypes
    stypes = set().union(*[tf.col_names_dict.keys() for tf in tf_list])
    feat_list_dict: Dict[torch_frame.stype, List[Tensor]] = {
        stype: []
        for stype in stypes
    }
    col_names_dict: Dict[torch_frame.stype, List[str]] = {
        stype: []
        for stype in stypes
    }
    for tf in tf_list:
        for stype in tf.col_names_dict.keys():
            feat_list_dict[stype].append(tf.feat_dict[stype])
            col_names_dict[stype].extend(tf.col_names_dict[stype])
    feat_dict = {
        stype: torch.cat(feat_list, dim=1)
        for stype, feat_list in feat_list_dict.items()
    }
    return TensorFrame(feat_dict=feat_dict, col_names_dict=col_names_dict, y=y)
