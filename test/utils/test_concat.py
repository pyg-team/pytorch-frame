import torch_frame
from torch_frame import TensorFrame


def test_cat_df_along_row(get_fake_tensor_frame):
    num_rows = 10
    num_repeats = 5
    tf = get_fake_tensor_frame(num_rows=num_rows)
    tf_cat = torch_frame.cat_tf([tf for _ in range(num_repeats)], along='row')
    assert len(tf_cat) == num_rows * num_repeats
    for i in range(num_repeats):
        tf_mini = tf_cat[num_rows * i:num_rows * (i + 1)]
        assert tf_mini == tf


def test_cat_df_along_col(get_fake_tensor_frame):
    num_rows = 10
    tf = get_fake_tensor_frame(num_rows=num_rows)
    stypes = list(tf.col_names_dict.keys())
    feat_dict1 = {}
    feat_dict2 = {}
    col_names_dict1 = {}
    col_names_dict2 = {}
    for stype in stypes:
        feat_dict1[stype] = tf.feat_dict[stype][:, :1]
        col_names_dict1[stype] = tf.col_names_dict[stype][:1]
        feat_dict2[stype] = tf.feat_dict[stype][:, 1:]
        col_names_dict2[stype] = tf.col_names_dict[stype][1:]

    tf1 = TensorFrame(feat_dict1, col_names_dict1, tf.y)
    tf2 = TensorFrame(feat_dict2, col_names_dict2, tf.y)
    tf_cat = torch_frame.cat_tf([tf1, tf2], along='col')
    return tf_cat == tf
