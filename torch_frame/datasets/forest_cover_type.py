import os.path as osp
import zipfile

import pandas as pd

import torch_frame


class ForestCoverType(torch_frame.data.Dataset):
    r"""The `Forest Cover Type
    <https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset>`_
    dataset from Kaggle. It's a task of forest cover type classification
    based on attributes such as elevation, slop and soil type etc.

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 20 10
        :header-rows: 1

        * - #rows
          - #cols (numerical)
          - #cols (categorical)
          - #classes
          - Task
          - Missing value ratio
        * - 581,012
          - 10
          - 44
          - 7
          - multiclass_classification
          - 0.0%
    """

    url = 'http://archive.ics.uci.edu/static/public/31/covertype.zip'

    def __init__(self, root: str):
        path = self.download_url(self.url, root)
        folder_path = osp.dirname(path)

        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)

        data_path = osp.join(folder_path, 'covtype.data.gz')

        names = [
            'Elevation',
            'Aspect',
            'Slope',
            'Horizontal_Distance_To_Hydrology',
            'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways',
            'Hillshade_9am',
            'Hillshade_Noon',
            'Hillshade_3pm',
            'Horizontal_Distance_To_Fire_Points',
            'Wilderness_Area1',
            'Wilderness_Area2',
            'Wilderness_Area3',
            'Wilderness_Area4',
            'Soil_Type1',
            'Soil_Type2',
            'Soil_Type3',
            'Soil_Type4',
            'Soil_Type5',
            'Soil_Type6',
            'Soil_Type7',
            'Soil_Type8',
            'Soil_Type9',
            'Soil_Type10',
            'Soil_Type11',
            'Soil_Type12',
            'Soil_Type13',
            'Soil_Type14',
            'Soil_Type15',
            'Soil_Type16',
            'Soil_Type17',
            'Soil_Type18',
            'Soil_Type19',
            'Soil_Type20',
            'Soil_Type21',
            'Soil_Type22',
            'Soil_Type23',
            'Soil_Type24',
            'Soil_Type25',
            'Soil_Type26',
            'Soil_Type27',
            'Soil_Type28',
            'Soil_Type29',
            'Soil_Type30',
            'Soil_Type31',
            'Soil_Type32',
            'Soil_Type33',
            'Soil_Type34',
            'Soil_Type35',
            'Soil_Type36',
            'Soil_Type37',
            'Soil_Type38',
            'Soil_Type39',
            'Soil_Type40',
            'Cover_Type',
        ]
        df = pd.read_csv(data_path, names=names)

        col_to_stype = {
            'Elevation': torch_frame.numerical,
            'Aspect': torch_frame.numerical,
            'Slope': torch_frame.numerical,
            'Horizontal_Distance_To_Hydrology': torch_frame.numerical,
            'Vertical_Distance_To_Hydrology': torch_frame.numerical,
            'Horizontal_Distance_To_Roadways': torch_frame.numerical,
            'Hillshade_9am': torch_frame.numerical,
            'Hillshade_Noon': torch_frame.numerical,
            'Hillshade_3pm': torch_frame.numerical,
            'Horizontal_Distance_To_Fire_Points': torch_frame.numerical,
            'Wilderness_Area1': torch_frame.categorical,
            'Wilderness_Area2': torch_frame.categorical,
            'Wilderness_Area3': torch_frame.categorical,
            'Wilderness_Area4': torch_frame.categorical,
            'Soil_Type1': torch_frame.categorical,
            'Soil_Type2': torch_frame.categorical,
            'Soil_Type3': torch_frame.categorical,
            'Soil_Type4': torch_frame.categorical,
            'Soil_Type5': torch_frame.categorical,
            'Soil_Type6': torch_frame.categorical,
            'Soil_Type7': torch_frame.categorical,
            'Soil_Type8': torch_frame.categorical,
            'Soil_Type9': torch_frame.categorical,
            'Soil_Type10': torch_frame.categorical,
            'Soil_Type11': torch_frame.categorical,
            'Soil_Type12': torch_frame.categorical,
            'Soil_Type13': torch_frame.categorical,
            'Soil_Type14': torch_frame.categorical,
            'Soil_Type15': torch_frame.categorical,
            'Soil_Type16': torch_frame.categorical,
            'Soil_Type17': torch_frame.categorical,
            'Soil_Type18': torch_frame.categorical,
            'Soil_Type19': torch_frame.categorical,
            'Soil_Type20': torch_frame.categorical,
            'Soil_Type21': torch_frame.categorical,
            'Soil_Type22': torch_frame.categorical,
            'Soil_Type23': torch_frame.categorical,
            'Soil_Type24': torch_frame.categorical,
            'Soil_Type25': torch_frame.categorical,
            'Soil_Type26': torch_frame.categorical,
            'Soil_Type27': torch_frame.categorical,
            'Soil_Type28': torch_frame.categorical,
            'Soil_Type29': torch_frame.categorical,
            'Soil_Type30': torch_frame.categorical,
            'Soil_Type31': torch_frame.categorical,
            'Soil_Type32': torch_frame.categorical,
            'Soil_Type33': torch_frame.categorical,
            'Soil_Type34': torch_frame.categorical,
            'Soil_Type35': torch_frame.categorical,
            'Soil_Type36': torch_frame.categorical,
            'Soil_Type37': torch_frame.categorical,
            'Soil_Type38': torch_frame.categorical,
            'Soil_Type39': torch_frame.categorical,
            'Soil_Type40': torch_frame.categorical,
            'Cover_Type': torch_frame.categorical,
        }

        super().__init__(df, col_to_stype, target_col='Cover_Type')
