from __future__ import annotations

import json
from typing import Any, TypeAlias

import torch.utils._pytree as _pytree
from torch.utils._pytree import SequenceKey

from torch_frame import stype
from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor
from torch_frame.data.multi_nested_tensor import MultiNestedTensor
from torch_frame.data.tensor_frame import TensorFrame

_TFCtx: TypeAlias = tuple[list[stype], dict[stype, list[str]], int | None]
_MTCtx: TypeAlias = list[int]
_Keyed: TypeAlias = list[tuple[SequenceKey, Any]]


def _tf_flatten(tf: TensorFrame) -> tuple[list[Any], _TFCtx]:
    keys = sorted(tf.feat_dict.keys(), key=lambda s: s.value)
    flat: list[Any] = [tf.feat_dict[k] for k in keys]
    flat.append(tf.y)
    return flat, (keys, tf.col_names_dict, tf._num_rows)


def _tf_flatten_with_keys(tf: TensorFrame) -> tuple[_Keyed, _TFCtx]:
    flat, ctx = _tf_flatten(tf)
    return [(SequenceKey(i), v) for i, v in enumerate(flat)], ctx


def _tf_unflatten(flat: list[Any], ctx: _TFCtx) -> TensorFrame:
    keys, col_names_dict, num_rows = ctx
    feat_dict = {k: flat[i] for i, k in enumerate(keys)}
    return TensorFrame(
        feat_dict,
        col_names_dict,
        y=flat[-1],
        num_rows=num_rows,
    )


def _tf_to_dumpable_context(ctx: _TFCtx) -> str:
    keys, col_names_dict, num_rows = ctx
    return json.dumps({
        'keys': [k.value for k in keys],
        'col_names_dict': {
            k.value: v
            for k, v in col_names_dict.items()
        },
        'num_rows': num_rows,
    })


def _tf_from_dumpable_context(dumpable: str) -> _TFCtx:
    d = json.loads(dumpable)
    keys = [stype(v) for v in d['keys']]
    col_names_dict = {stype(k): v for k, v in d['col_names_dict'].items()}
    return (keys, col_names_dict, d['num_rows'])


def _met_flatten(met: MultiEmbeddingTensor) -> tuple[list[Any], _MTCtx]:
    return [met.values, met.offset], [met.num_rows, met.num_cols]


def _met_flatten_with_keys(met: MultiEmbeddingTensor) -> tuple[_Keyed, _MTCtx]:
    flat, ctx = _met_flatten(met)
    return [(SequenceKey(i), v) for i, v in enumerate(flat)], ctx


def _met_unflatten(
    flat: list[Any],
    ctx: _MTCtx,
) -> MultiEmbeddingTensor:
    num_rows, num_cols = ctx
    # Bypass __init__ to avoid validate()
    met = MultiEmbeddingTensor.__new__(MultiEmbeddingTensor)
    met.num_rows = num_rows
    met.num_cols = num_cols
    met.values = flat[0]
    met.offset = flat[1]
    return met


def _mnt_flatten(mnt: MultiNestedTensor) -> tuple[list[Any], _MTCtx]:
    return [mnt.values, mnt.offset], [mnt.num_rows, mnt.num_cols]


def _mnt_flatten_with_keys(mnt: MultiNestedTensor) -> tuple[_Keyed, _MTCtx]:
    flat, ctx = _mnt_flatten(mnt)
    return [(SequenceKey(i), v) for i, v in enumerate(flat)], ctx


def _mnt_unflatten(
    flat: list[Any],
    ctx: _MTCtx,
) -> MultiNestedTensor:
    num_rows, num_cols = ctx
    # Bypass __init__ to avoid validate()
    mnt = MultiNestedTensor.__new__(MultiNestedTensor)
    mnt.num_rows = num_rows
    mnt.num_cols = num_cols
    mnt.values = flat[0]
    mnt.offset = flat[1]
    return mnt


_pytree.register_pytree_node(
    TensorFrame,
    _tf_flatten,
    _tf_unflatten,  # type: ignore[arg-type]
    flatten_with_keys_fn=_tf_flatten_with_keys,  # type: ignore[arg-type]
    serialized_type_name='torch_frame.TensorFrame',
    to_dumpable_context=_tf_to_dumpable_context,
    from_dumpable_context=_tf_from_dumpable_context,
)
_pytree.register_pytree_node(
    MultiEmbeddingTensor,
    _met_flatten,
    _met_unflatten,  # type: ignore[arg-type]
    flatten_with_keys_fn=_met_flatten_with_keys,  # type: ignore[arg-type]
    serialized_type_name='torch_frame.MultiEmbeddingTensor',
)
_pytree.register_pytree_node(
    MultiNestedTensor,
    _mnt_flatten,
    _mnt_unflatten,  # type: ignore[arg-type]
    flatten_with_keys_fn=_mnt_flatten_with_keys,  # type: ignore[arg-type]
    serialized_type_name='torch_frame.MultiNestedTensor',
)
