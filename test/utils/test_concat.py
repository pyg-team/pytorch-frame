import torch
from torch import Tensor

import torch_frame
from torch_frame import TensorFrame
from torch_frame.utils import cat_tensor_data


def test_cat_tensor_data_along_row(get_fake_tensor_frame):
    num_rows = 10
    num_repeats = 5
    tf = get_fake_tensor_frame(num_rows=num_rows)
    stypes = list(tf.col_names_dict.keys())
    for stype in stypes:
        td_cat = cat_tensor_data(
            [tf.feat_dict[stype] for _ in range(num_repeats)], dim=0)
        for i in range(num_repeats):
            if stype.use_dict_multi_nested_tensor:
                assert len(td_cat) == len(tf.feat_dict[stype])
                for key in td_cat:
                    td_mini = td_cat[key][num_rows * i:num_rows * (i + 1)]
                    assert key in tf.feat_dict[stype]
                    assert torch.allclose(td_mini.values,
                                          tf.feat_dict[stype][key].values)
                    assert torch.allclose(td_mini.offset,
                                          tf.feat_dict[stype][key].offset)
                continue
            td_mini = td_cat[num_rows * i:num_rows * (i + 1)]
            assert len(td_cat) == num_rows * num_repeats
            if isinstance(td_mini, Tensor):
                assert torch.allclose(td_mini, tf.feat_dict[stype])
            elif stype.use_multi_tensor:
                assert torch.allclose(td_mini.values,
                                      tf.feat_dict[stype].values)
                assert torch.allclose(td_mini.offset,
                                      tf.feat_dict[stype].offset)


def test_cat_tensor_data_along_col(get_fake_tensor_frame):
    num_rows = 10
    tf = get_fake_tensor_frame(num_rows=num_rows)
    stypes = list(tf.col_names_dict.keys())
    td1 = {}
    td2 = {}
    for stype in stypes:
        if stype.use_dict_multi_nested_tensor:
            td1 = {
                name: tf.feat_dict[stype][name][:, :1]
                for name in tf.feat_dict[stype].keys()
            }
            td2 = {
                name: tf.feat_dict[stype][name][:, 1:]
                for name in tf.feat_dict[stype].keys()
            }
        else:
            td1 = tf.feat_dict[stype][:, :1]
            td2 = tf.feat_dict[stype][:, 1:]
        td_cat = cat_tensor_data([td1, td2], dim=1)
        if isinstance(td_cat, Tensor):
            assert torch.allclose(td_cat, tf.feat_dict[stype])
        elif stype.use_multi_tensor:
            assert torch.allclose(td_cat.values, tf.feat_dict[stype].values)
            assert torch.allclose(td_cat.offset, tf.feat_dict[stype].offset)
        elif stype.use_dict_multi_nested_tensor:
            assert len(td_cat) == len(tf.feat_dict[stype])
            for key in td_cat:
                assert key in tf.feat_dict[stype]
                assert torch.allclose(td_cat[key].values,
                                      tf.feat_dict[stype][key].values)
                assert torch.allclose(td_cat[key].offset,
                                      tf.feat_dict[stype][key].offset)


def test_cat_tensor_frames_along_row(get_fake_tensor_frame):
    num_rows = 10
    num_repeats = 5
    tf = get_fake_tensor_frame(num_rows=num_rows)
    tf_cat = torch_frame.cat([tf for _ in range(num_repeats)], along='row')
    assert len(tf_cat) == num_rows * num_repeats
    for i in range(num_repeats):
        tf_mini = tf_cat[num_rows * i:num_rows * (i + 1)]
        assert tf_mini == tf


def test_cat_tensor_frames_along_col(get_fake_tensor_frame):
    num_rows = 10
    tf = get_fake_tensor_frame(num_rows=num_rows)
    stypes = list(tf.col_names_dict.keys())
    feat_dict1 = {}
    feat_dict2 = {}
    col_names_dict1 = {}
    col_names_dict2 = {}
    for stype in stypes:
        if stype.use_dict_multi_nested_tensor:
            feat_dict1[stype] = {
                name: tf.feat_dict[stype][name][:, :1]
                for name in tf.feat_dict[stype].keys()
            }
            col_names_dict1[stype] = tf.col_names_dict[stype][:1]
            feat_dict2[stype] = {
                name: tf.feat_dict[stype][name][:, 1:]
                for name in tf.feat_dict[stype].keys()
            }
            col_names_dict2[stype] = tf.col_names_dict[stype][1:]
        else:
            feat_dict1[stype] = tf.feat_dict[stype][:, :1]
            col_names_dict1[stype] = tf.col_names_dict[stype][:1]
            feat_dict2[stype] = tf.feat_dict[stype][:, 1:]
            col_names_dict2[stype] = tf.col_names_dict[stype][1:]

    tf1 = TensorFrame(feat_dict1, col_names_dict1, tf.y)
    tf2 = TensorFrame(feat_dict2, col_names_dict2, None)
    tf_cat = torch_frame.cat([tf1, tf2], along='col')
    assert tf_cat == tf
