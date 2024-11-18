import copy

import pytest
import torch

import torch_frame
from torch_frame import TensorFrame
from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor
from torch_frame.data.multi_nested_tensor import MultiNestedTensor


def test_tensor_frame_basics(get_fake_tensor_frame):
    tf = get_fake_tensor_frame(num_rows=10)
    assert tf.is_empty is False
    assert tf.device == torch.device("cpu")
    assert tf.num_rows == len(tf) == 10
    assert tf.num_cols == 16
    # The order is guaranteed to be the same as that of stype definitions in
    # the stype enum even though `feat_dict` has a different order.
    assert tf.stypes == [
        torch_frame.numerical,
        torch_frame.categorical,
        torch_frame.text_embedded,
        torch_frame.text_tokenized,
        torch_frame.multicategorical,
        torch_frame.sequence_numerical,
        torch_frame.embedding,
    ]
    assert str(tf) == (
        "TensorFrame(\n"
        "  num_cols=16,\n"
        "  num_rows=10,\n"
        "  categorical (3): ['cat_1', 'cat_2', 'cat_3'],\n"
        "  numerical (2): ['num_1', 'num_2'],\n"
        "  multicategorical (2): ['multicat_1', 'multicat_2'],\n"
        "  text_embedded (3): ['text_embedded_1', 'text_embedded_2',"
        " 'text_embedded_3'],\n"
        "  text_tokenized (2): ['text_tokenized_1', 'text_tokenized_2'],\n"
        "  sequence_numerical (2): ['seq_num_1', 'seq_num_2'],\n"
        "  embedding (2): ['emb_1', 'emb_2'],\n"
        "  has_target=True,\n"
        "  device='cpu',\n"
        ")")

    tf = TensorFrame({}, {})
    assert tf.is_empty is True
    assert tf.device is None
    assert tf.num_rows == len(tf) == 0
    assert tf.num_cols == 0
    assert tf.stypes == []
    assert str(tf) == ("TensorFrame(\n"
                       "  num_cols=0,\n"
                       "  num_rows=0,\n"
                       "  has_target=False,\n"
                       "  device=None,\n"
                       ")")


def test_tensor_frame_error():
    feat_dict = {
        torch_frame.categorical: torch.randint(0, 3, size=(10, 3)),
        torch_frame.numerical: torch.randn(size=(10, 2)),
    }
    col_names_dict = {
        torch_frame.categorical: ['cat_1', 'cat_2', 'cat_3'],
        torch_frame.numerical: ['num_1', 'num_2'],
    }
    y = torch.randn(10)

    # Wrong number of channels
    feat_dict[torch_frame.categorical] = torch.randint(0, 3, size=(10, ))
    with pytest.raises(ValueError, match='at least 2-dimensional'):
        TensorFrame(feat_dict=feat_dict, col_names_dict=col_names_dict, y=y)
    feat_dict[torch_frame.categorical] = torch.randint(0, 3, size=(10, 3))

    # Mis-alignment of the col_names and the number of columns in feat_dict
    feat_dict[torch_frame.categorical] = torch.randint(0, 3, size=(10, 4))
    with pytest.raises(ValueError, match='not align with'):
        TensorFrame(feat_dict=feat_dict, col_names_dict=col_names_dict, y=y)
    feat_dict[torch_frame.categorical] = torch.randint(0, 3, size=(10, 3))

    # Mis-alignment of the lengths within feat_dict
    feat_dict[torch_frame.categorical] = torch.randint(0, 3, size=(11, 3))
    with pytest.raises(ValueError, match='not aligned'):
        TensorFrame(feat_dict=feat_dict, col_names_dict=col_names_dict, y=y)
    feat_dict[torch_frame.categorical] = torch.randint(0, 3, size=(10, 3))

    # Mis-alignment between the lengths of feat_dict and y
    y = torch.randn(11)
    with pytest.raises(ValueError, match='not aligned'):
        TensorFrame(feat_dict=feat_dict, col_names_dict=col_names_dict, y=y)


@pytest.mark.parametrize('index', [
    4,
    [4, 8],
    range(2, 6),
    torch.tensor([4, 8]),
])
def test_tensor_frame_index_select(get_fake_tensor_frame, index):
    tf = get_fake_tensor_frame(num_rows=10)

    out = tf[index]

    if isinstance(index, int):
        assert out.num_rows == 1
    else:
        assert out.num_rows == len(index)

    assert out.col_names_dict == tf.col_names_dict


def test_empty_tensor_frame():
    # Categorical feature is empty
    feat_dict = {
        torch_frame.categorical: torch.randint(0, 3, size=(10, 0)),
        torch_frame.numerical: torch.randn(size=(10, 2)),
    }
    col_names_dict = {
        torch_frame.categorical: [],
        torch_frame.numerical: ['num_1', 'num_2'],
    }
    with pytest.raises(RuntimeError, match='Empty columns'):
        TensorFrame(feat_dict=feat_dict, col_names_dict=col_names_dict)

    col_names_dict = {
        torch_frame.numerical: ['num_1', 'num_2'],
    }
    with pytest.raises(ValueError, match='The keys of feat_dict'):
        TensorFrame(feat_dict=feat_dict, col_names_dict=col_names_dict)


def test_equal_tensor_frame(get_fake_tensor_frame):
    tf1 = get_fake_tensor_frame(num_rows=10)

    # Test equal
    tf2 = copy.copy(tf1)
    assert tf1 == tf2
    assert tf2 == tf1

    # Test difference in col_names_dict
    tf2.col_names_dict[torch_frame.numerical] = [
        name + '_' for name in tf1.col_names_dict[torch_frame.numerical]
    ]
    assert tf1 != tf2
    assert tf2 != tf1

    # Test difference in y (Tensor versus None)
    tf2 = copy.copy(tf1)
    tf2.y = None
    assert tf1 != tf2
    assert tf2 != tf1

    # Test difference in y (Tensor versus Tensor)
    tf2.y = torch.randn(tf1.y.shape)
    assert tf1 != tf2
    assert tf2 != tf1

    # Test difference in feat_dict
    tf2 = copy.copy(tf1)
    tf2.feat_dict[torch_frame.numerical] = torch.randn(
        tf2.feat_dict[torch_frame.numerical].shape)
    assert tf1 != tf2
    assert tf2 != tf1

    # Test difference in length
    tf2 = get_fake_tensor_frame(num_rows=11)
    assert tf1 != tf2
    assert tf2 != tf1

    # Test empty TensorFrames
    tf1 = TensorFrame({}, {})
    tf2 = TensorFrame({}, {})
    assert tf1 == tf2
    assert tf2 == tf1


def test_get_col_feat(get_fake_tensor_frame):
    num_rows = 10
    tf = get_fake_tensor_frame(num_rows=num_rows)
    for stype, cols in tf.col_names_dict.items():
        feat_list = []
        for col in cols:
            feat = tf.get_col_feat(col)
            feat_list.append(feat)
            # Check that shapes are all (num_rows, 1, *)
            if stype.use_dict_multi_nested_tensor:
                assert all(value.shape[:2] == (num_rows, 1)
                           for value in feat.values())
            else:
                assert feat.shape[:2] == (num_rows, 1)
        # Check that concatenation of feat_list reproduces the original
        # feat_dict[stype]
        if stype.use_multi_nested_tensor:
            assert MultiNestedTensor.allclose(
                MultiNestedTensor.cat(feat_list, dim=1), tf.feat_dict[stype])
        elif stype.use_multi_embedding_tensor:
            assert MultiEmbeddingTensor.allclose(
                MultiEmbeddingTensor.cat(feat_list, dim=1),
                tf.feat_dict[stype])
        elif stype.use_dict_multi_nested_tensor:
            for key in tf.feat_dict[stype].keys():
                assert MultiNestedTensor.allclose(
                    MultiNestedTensor.cat([feat[key] for feat in feat_list],
                                          dim=1), tf.feat_dict[stype][key])
        else:
            assert torch.allclose(torch.cat(feat_list, dim=1),
                                  tf.feat_dict[stype])


def test_empty_tensor_fraome():
    tf = TensorFrame({}, {})
    assert tf.num_rows == 0
    assert tf.device is None

    tf = TensorFrame({}, {}, num_rows=4)
    assert tf.num_rows == 4
    assert tf.device is None
    assert tf[0].num_rows == 1
    assert tf[[0, 1]].num_rows == 2
    assert tf[0:2].num_rows == 2
    assert tf[torch.tensor([0, 1])].num_rows == 2
    assert tf[torch.tensor([True, True, False, False])].num_rows == 2
