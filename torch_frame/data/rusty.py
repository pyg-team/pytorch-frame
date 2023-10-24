from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


class Rusty:
    r"""3-D sparse matrix to store multi-categorical data
     [num_rows + 1, num_cols]
    """
    def __init__(self, categories: Dict[str, List[str]], delimiter: str):
        self.columns = list(categories.keys())
        self.categories = {
            col: {
                categories[col][i]: i
                for i in range(len(categories[col]))
            }
            for col in self.columns
        }
        self.id_to_col = categories
        self.delimiter = delimiter

    def forward(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        values = []
        boundaries = [0]
        df = df[self.columns]
        for _, row in df.iterrows():
            for col in self.columns:
                if row[col] is None:
                    boundaries.append(boundaries[-1])
                    continue
                items = list(
                    filter(lambda x: x in self.categories[col],
                           row[col].split(self.delimiter)))
                if not items:
                    boundaries.append(boundaries[-1])
                    continue
                boundaries.append(boundaries[-1] + len(items))
                values += [self.categories[col][item] for item in items]
        values = torch.from_numpy(np.array(values))
        boundaries = torch.from_numpy(np.array(boundaries))
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
            print(
                "values ", values[values_ptr:values_ptr + boundaries[i] -
                                  boundaries[i - 1]], values_ptr)
            categories = self.delimiter.join(
                index) if index is not None else None
            data[col].append(categories)
            values_ptr += boundaries[i] - boundaries[i - 1]
        return pd.DataFrame(data)
