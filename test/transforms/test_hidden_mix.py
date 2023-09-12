from torch_frame import TensorFrame, stype
from torch_frame.data import Dataset
from torch_frame.datasets.fake import FakeDataset
from torch_frame.nn import LinearEncoder
from torch_frame.transforms import HiddenMix


def test_hidden_mix():
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False,
                                   stypes=[stype.numerical])
    dataset.materialize()
    tensor_frame: TensorFrame = dataset.tensor_frame
    x = tensor_frame.x_dict[stype.numerical]
    stats_list = [
        dataset.col_stats[col_name]
        for col_name in tensor_frame.col_names_dict[stype.numerical]
    ]
    encoder = LinearEncoder(out_channels=7, stats_list=stats_list)
    hidden_mix = HiddenMix()
    out = encoder(x)
    out = hidden_mix(out)
    print(out)


test_hidden_mix()
