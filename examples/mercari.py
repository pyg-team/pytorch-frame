import os.path as osp

from torch_frame.datasets import Mercari

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
dataset = Mercari(root=path)
print(dataset.df)
missing_ratio = dataset.df.isna().mean()
print(missing_ratio)
