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
        self.categories = {
            col: {
                categories[col][i]: i
                for i in range(len(categories[col]))
            }
            for col in self.columns
        }
        self.id_to_col = categories
        self.sep = sep

    def forward(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        values = []
        boundaries = [0]
        df = df[self.columns]
        for col in self.columns:
            df[col] = df[col].apply(lambda x: [
                self.categories[col][s] for s in x.split(self.sep)
                if s in self.categories[col]
            ] if x is not None else [])
        values = np.array(sum(df.apply(lambda row: sum(row, []), axis=1), []))
        boundaries = np.array(
            sum(df.apply(lambda row: [len(item) for item in row], axis=1), []))
        boundaries = np.cumsum(boundaries)
        print(boundaries)
        return values, boundaries

    def backward(self, values, boundaries):
        values = values.tolist()
        data = {col: [] for col in self.columns}
        values_ptr = 0

        for i in range(1, len(boundaries)):
            col = self.columns[(i - 1) % len(self.columns)]
            index = list(
                filter(lambda x: x is not None, [
                    self.id_to_col[col][item] if item != -1 else None
                    for item in values[values_ptr:values_ptr + boundaries[i] -
                                       boundaries[i - 1]]
                ]))
            categories = self.delimiter.join(
                index) if index is not None else None
            data[col].append(categories)
            values_ptr += boundaries[i] - boundaries[i - 1]
        return pd.DataFrame(data)
