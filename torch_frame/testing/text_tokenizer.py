from typing import List, Optional

import torch

from torch_frame.typing import TensorData, TextTokenizationOutputs


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

    def __call__(self, sentences: List[str]) -> TextTokenizationOutputs:
        res = []
        for s in sentences:
            tokens = s.split(' ')
            idx = torch.LongTensor(
                [hash(t) % self.num_hash_bins for t in tokens])
            mask = torch.ones(idx.shape, dtype=torch.long)
            res.append({'input_ids': idx, 'attention_mask': mask})
        return res


class RandomTextModel(torch.nn.Module):
    r"""A text embedding model that takes the tokenized input from
    :class:`WhiteSpaceHashTokenizer` and outputs random embeddings. Should be
    used only for testing purposes."""
    def __init__(self, text_emb_channels: int, num_cols: int):
        self.text_emb_channels = text_emb_channels
        self.num_cols = num_cols
        super().__init__()

    def forward(self, feat: TensorData):
        input_ids = feat['input_ids'].to_dense(fill_value=0)
        _ = feat['attention_mask'].to_dense(fill_value=0)
        return torch.rand(size=(input_ids.shape[0], self.num_cols,
                                self.text_emb_channels))
