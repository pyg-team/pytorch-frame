import csv
import tempfile

from torch_frame import stype
from torch_frame.datasets import CustomDataset


def test_custom_dataset():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False,
                                     suffix='.csv') as temp:

        writer = csv.writer(temp)

        writer.writerow(['Name', 'Age', 'Occupation'])
        writer.writerow(['Alice', '30', 'Software Developer'])
        writer.writerow(['Bob', '25', 'Data Scientist'])

    dataset = CustomDataset(
        path=temp.name, col_to_stype={
            "Name": stype.categorical,
            "Age": stype.numerical,
            "Occupation": stype.categorical
        })

    assert str(dataset) == 'CustomDataset()'
    assert dataset.feat_cols == ['Name', 'Age', 'Occupation']
