import os.path as osp

from torch_frame.datasets import Jobs

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', "jobs")
dataset = Jobs(root=path)
