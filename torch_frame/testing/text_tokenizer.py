from typing import List, Optional

import torch
from torch.nested import nested_tensor


class WhiteSpaceTokenizer:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device
        self.num_hash_bins = 10

    def __call__(self, sentences: List[str]) -> nested_tensor:
        res = []
        for s in sentences:
            tokens = s.split(' ')
            idx = torch.LongTensor(
                [hash(t) % self.num_hash_bins for t in tokens])
            res.append(idx)
        return torch.nested.nested_tensor(res, device=self.device)
