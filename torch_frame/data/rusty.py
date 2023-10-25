from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


class Rusty:
    r"""3-D sparse matrix to store multi-categorical data
     [num_rows + 1, num_cols]
    """
    def __init__(self, categories: Dict[str, List[str]], sep: str):
        self.columns = list(categories.keys())
        self.categories_to_id = {
            col: {
                categories[col][i]: i
                for i in range(len(categories[col]))
            }
            for col in self.columns
        }
        self.id_to_categories = categories
        self.sep = sep

    def forward(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        values = []
        boundaries = [0]
        df = df[self.columns]
        for col in self.columns:
            df[col] = df[col].apply(lambda x: [
                self.categories_to_id[col][s] for s in x.split(self.sep)
                if s in self.categories_to_id[col]
            ] if x is not None else [])
        values = np.array(sum(df.apply(lambda row: sum(row, []), axis=1), []))
        boundaries = np.array(
            sum(df.apply(lambda row: [len(item) for item in row], axis=1), []))
        boundaries = np.cumsum(boundaries)

        return values, boundaries

    def backward(self, values, boundaries):
        values = values.tolist()
        data = {col: [] for col in self.columns}

        for i in range(len(boundaries)):
            col = self.columns[i % len(self.columns)]
            index = list(
                filter(lambda x: x is not None, [
                    self.id_to_categories[col][item] for item in
                    values[boundaries[i - 1] if i != 0 else 0:boundaries[i]]
                ]))
            data[col].append(self.sep.join(index) if index else None)
        return pd.DataFrame(data)
