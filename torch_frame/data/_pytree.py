from __future__ import annotations

import importlib
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
    # Bypass ``__init__`` (which calls ``validate``) so that the data-dependent
    # ``tensor.size(0) != num_rows`` checks do not specialize symbolic shapes
    # under ``torch.export``.
    out = TensorFrame.__new__(TensorFrame)
    out.feat_dict = feat_dict
    out.col_names_dict = col_names_dict
    out.y = flat[-1]
    out._num_rows = num_rows
    out._col_to_stype_idx = {}
    for stype_name, cols in col_names_dict.items():
        for idx, col in enumerate(cols):
            out._col_to_stype_idx[col] = (stype_name, idx)
    return out


def _tf_to_dumpable_context(ctx: _TFCtx) -> str:
    keys, col_names_dict, num_rows = ctx
    # Record the enum class used as feat_dict keys so deserialization can
    # reconstruct the same class.  Downstream projects sometimes substitute
    # their own enum (e.g. ``kumoapi.typing.Stype``) for ``torch_frame.stype``;
    # without this, ``_tf_from_dumpable_context`` would fail on values that
    # only exist in the downstream enum.
    cls = type(keys[0]) if keys else stype
    return json.dumps({
        'keys': [k.value for k in keys],
        'col_names_dict': {
            k.value: v
            for k, v in col_names_dict.items()
        },
        'num_rows': num_rows,
        'stype_cls': f'{cls.__module__}:{cls.__qualname__}',
    })


def _tf_from_dumpable_context(dumpable: str) -> _TFCtx:
    d = json.loads(dumpable)
    cls_name = d.get('stype_cls')
    if cls_name:
        module_name, _, attr = cls_name.partition(':')
        cls = getattr(importlib.import_module(module_name), attr)
    else:
        cls = stype
    keys = [cls(v) for v in d['keys']]
    col_names_dict = {cls(k): v for k, v in d['col_names_dict'].items()}
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
    met.num_cols = num_cols
    met.values = flat[0]
    met.offset = flat[1]
    # Derive ``num_rows`` from the values tensor so it stays symbolic under
    # ``torch.export``; ``values`` has shape ``(num_rows, total_embed_dim)``.
    met.num_rows = (met.values.size(0) if met.values.numel() > 0 else num_rows)
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
    mnt.num_cols = num_cols
    mnt.values = flat[0]
    mnt.offset = flat[1]
    # Derive ``num_rows`` from the offset tensor so it stays symbolic under
    # ``torch.export``; ``offset`` has shape ``(num_rows * num_cols + 1,)``.
    mnt.num_rows = ((mnt.offset.size(0) - 1) //
                    num_cols if num_cols > 0 else num_rows)
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
