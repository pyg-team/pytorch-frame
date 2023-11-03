from typing import List, Optional

import torch

from torch_frame.typing import TextTokenizationOutput


class WhiteSpaceHashTokenizer:
    r"""A simple white space tokenizer for testing purposes.
    It split sentence to tokens via white space and hashes each
    token into an index modulo :obj:`num_hash`.

    Args:
        num_hash_bins (int): Number of hash bins to use.
            (default: :obj:`64`)
        device (torch.device, optional): The device to put tokens.
            (default: :obj:`None`)
    """
    def __init__(self, num_hash_bins: int = 64,
                 device: Optional[torch.device] = None):
        self.device = device
        self.num_hash_bins = num_hash_bins

    def __call__(self, sentences: List[str]) -> TextTokenizationOutput:
        res = []
        for s in sentences:
            tokens = s.split(' ')
            idx = torch.LongTensor(
                [hash(t) % self.num_hash_bins for t in tokens])
            res.append({'input_ids': idx})
        return res
