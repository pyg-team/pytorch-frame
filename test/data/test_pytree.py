from collections.abc import Callable

import torch
import torch.utils._pytree as pytree

import torch_frame
from torch_frame import TensorFrame
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor


def _make_met(num_rows: int = 3, num_cols: int = 2) -> MultiEmbeddingTensor:
    tensor_list = [torch.randn(num_rows, dim) for dim in [4, 2][:num_cols]]
    return MultiEmbeddingTensor.from_tensor_list(tensor_list)


def _make_mnt(num_rows: int = 3, num_cols: int = 2) -> MultiNestedTensor:
    tensor_mat = [[torch.randn(i + j + 1) for j in range(num_cols)]
                  for i in range(num_rows)]
    return MultiNestedTensor.from_tensor_mat(tensor_mat)


def _make_tensor_frame(num_rows: int = 4) -> TensorFrame:
    feat_dict = {
        torch_frame.numerical: torch.randn(num_rows, 3),
        torch_frame.categorical: torch.randint(0, 5, (num_rows, 2)),
    }
    col_names_dict = {
        torch_frame.numerical: ['n1', 'n2', 'n3'],
        torch_frame.categorical: ['c1', 'c2'],
    }
    return TensorFrame(feat_dict, col_names_dict, y=torch.randn(num_rows))


def test_met_roundtrip() -> None:
    met = _make_met()
    flat, spec = pytree.tree_flatten(met)
    assert len(flat) == 2  # values, offset
    assert all(isinstance(t, torch.Tensor) for t in flat)

    met2 = pytree.tree_unflatten(flat, spec)
    assert isinstance(met2, MultiEmbeddingTensor)
    assert met2.num_rows == met.num_rows
    assert met2.num_cols == met.num_cols
    assert torch.equal(met2.values, met.values)
    assert torch.equal(met2.offset, met.offset)


def test_mnt_roundtrip() -> None:
    mnt = _make_mnt()
    flat, spec = pytree.tree_flatten(mnt)
    assert len(flat) == 2  # values, offset
    assert all(isinstance(t, torch.Tensor) for t in flat)

    mnt2 = pytree.tree_unflatten(flat, spec)
    assert isinstance(mnt2, MultiNestedTensor)
    assert mnt2.num_rows == mnt.num_rows
    assert mnt2.num_cols == mnt.num_cols
    assert torch.equal(mnt2.values, mnt.values)
    assert torch.equal(mnt2.offset, mnt.offset)


def test_tf_roundtrip() -> None:
    tf1 = _make_tensor_frame()
    flat, spec = pytree.tree_flatten(tf1)
    # numerical tensor + categorical tensor + y
    assert len(flat) == 3
    assert all(isinstance(t, torch.Tensor) for t in flat)

    tf2 = pytree.tree_unflatten(flat, spec)
    assert tf1 == tf2


def test_tf_roundtrip_no_y() -> None:
    feat_dict = {torch_frame.numerical: torch.randn(4, 2)}
    col_names_dict = {torch_frame.numerical: ['a', 'b']}
    tf1 = TensorFrame(feat_dict, col_names_dict, y=None)

    flat, spec = pytree.tree_flatten(tf1)
    tf2 = pytree.tree_unflatten(flat, spec)
    assert tf1 == tf2


def test_tf_roundtrip_with_multi_tensors() -> None:
    """TensorFrame with MET and MNT feat_dict values."""
    num_rows = 4
    feat_dict = {
        torch_frame.numerical: torch.randn(num_rows, 2),
        torch_frame.embedding: _make_met(num_rows, num_cols=2),
        torch_frame.multicategorical: _make_mnt(num_rows, num_cols=2),
    }
    col_names_dict = {
        torch_frame.numerical: ['n1', 'n2'],
        torch_frame.embedding: ['e1', 'e2'],
        torch_frame.multicategorical: ['mc1', 'mc2'],
    }
    tf1 = TensorFrame(feat_dict, col_names_dict, y=torch.randn(num_rows))

    flat, spec = pytree.tree_flatten(tf1)
    # All leaves should be plain tensors (MET/MNT are recursively flattened).
    assert all(isinstance(t, torch.Tensor) for t in flat)

    tf2 = pytree.tree_unflatten(flat, spec)
    assert tf1 == tf2


def test_tf_roundtrip_with_dict_feat(get_fake_tensor_frame: Callable) -> None:
    """Full TensorFrame including dict-valued feat (text_tokenized)."""
    tf1 = get_fake_tensor_frame(num_rows=5)

    flat, spec = pytree.tree_flatten(tf1)
    assert all(isinstance(t, torch.Tensor) for t in flat)

    tf2 = pytree.tree_unflatten(flat, spec)
    assert tf1 == tf2


def test_tf_col_to_stype_idx() -> None:
    """Unflatten must rebuild _col_to_stype_idx correctly."""
    tf1 = _make_tensor_frame()
    flat, spec = pytree.tree_flatten(tf1)
    tf2 = pytree.tree_unflatten(flat, spec)
    assert tf1._col_to_stype_idx == tf2._col_to_stype_idx


def test_met_treespec_serialization() -> None:
    met = _make_met()
    _, spec = pytree.tree_flatten(met)
    serialized = pytree.treespec_dumps(spec)
    roundtrip_spec = pytree.treespec_loads(serialized)
    assert roundtrip_spec == spec


def test_mnt_treespec_serialization() -> None:
    mnt = _make_mnt()
    _, spec = pytree.tree_flatten(mnt)
    serialized = pytree.treespec_dumps(spec)
    roundtrip_spec = pytree.treespec_loads(serialized)
    assert roundtrip_spec == spec


def test_tf_treespec_serialization() -> None:
    tf = _make_tensor_frame()
    _, spec = pytree.tree_flatten(tf)
    serialized = pytree.treespec_dumps(spec)
    roundtrip_spec = pytree.treespec_loads(serialized)
    assert roundtrip_spec == spec


def test_compile_met() -> None:
    met = _make_met()

    @torch.compile(fullgraph=True)
    def double_values(x: MultiEmbeddingTensor) -> MultiEmbeddingTensor:
        return pytree.tree_map(lambda t: t * 2, x)

    met2 = double_values(met)
    assert torch.equal(met2.values, met.values * 2)


def test_compile_mnt() -> None:
    mnt = _make_mnt()

    @torch.compile(fullgraph=True)
    def double_values(x: MultiNestedTensor) -> MultiNestedTensor:
        return pytree.tree_map(lambda t: t * 2, x)

    mnt2 = double_values(mnt)
    assert torch.equal(mnt2.values, mnt.values * 2)


def test_compile_tf() -> None:
    tf = _make_tensor_frame()

    @torch.compile(fullgraph=True)
    def double_feats(x: TensorFrame) -> TensorFrame:
        return pytree.tree_map(lambda t: t * 2, x)

    tf2 = double_feats(tf)
    for stype in tf.feat_dict:
        assert torch.equal(tf2.feat_dict[stype], tf.feat_dict[stype] * 2)


def test_export_construct_met() -> None:
    class Model(torch.nn.Module):
        def forward(self, v: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
            met = MultiEmbeddingTensor(
                num_rows=4,
                num_cols=1,
                values=v,
                offset=o,
            )
            return met.values.sum()

    model = Model()
    values = torch.randn(4, 3)
    offset = torch.tensor([0, 3])

    exported = torch.export.export(model, args=(values, offset), strict=False)
    result = exported.module()(values, offset)
    assert torch.allclose(result, values.sum())


def test_export_construct_mnt() -> None:
    class Model(torch.nn.Module):
        def forward(self, v: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
            mnt = MultiNestedTensor(num_rows=1, num_cols=2, values=v, offset=o)
            return mnt.values.sum()

    model = Model()
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    offset = torch.tensor([0, 2, 5])

    exported = torch.export.export(model, args=(values, offset), strict=False)
    result = exported.module()(values, offset)
    assert torch.allclose(result, values.sum())
