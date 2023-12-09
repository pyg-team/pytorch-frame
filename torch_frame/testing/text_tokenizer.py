from __future__ import annotations

import torch
import torch.nn.functional as F

from torch_frame.data import MultiNestedTensor
from torch_frame.typing import TextTokenizationOutputs


class WhiteSpaceHashTokenizer:
    r"""A simple white space tokenizer for testing purposes.
    It split sentence to tokens via white space and hashes each
    token into an index modulo :obj:`num_hash`.

    Args:
        num_hash_bins (int): Number of hash bins to use.
            (default: :obj:`64`)
        device (torch.device, optional): The device to put tokens.
            (default: :obj:`None`)
        batched (bool): Whether to tokenize in a batched format.
            If :obj:`True`, tokenizer returns Mapping[str, 2dim-Tensor],
            else List[Mapping[str, 1dim-Tensor]]. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_hash_bins: int = 64,
        device: torch.device | None = None,
        batched: bool = False,
    ):
        self.device = device
        self.num_hash_bins = num_hash_bins
        self.batched = batched

    def __call__(self, sentences: list[str]) -> TextTokenizationOutputs:
        input_ids = []
        attention_mask = []
        for s in sentences:
            tokens = s.split(' ')
            idx = torch.tensor([hash(t) % self.num_hash_bins for t in tokens])
            input_ids.append(idx)
            attention_mask.append(torch.ones(idx.shape, dtype=torch.bool))

        if self.batched:
            max_length = max(t.size(0) for t in input_ids)
            padded_input_ids = [
                F.pad(t, (0, max_length - t.size(0)), value=-1)
                for t in input_ids
            ]
            input_ids = torch.stack(padded_input_ids)
            padded_attention_mask = [
                F.pad(t, (0, max_length - t.size(0)), value=False)
                for t in attention_mask
            ]
            attention_mask = torch.stack(padded_attention_mask)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        else:
            return [{
                'input_ids': input_ids[i],
                'attention_mask': attention_mask[i]
            } for i in range(len(sentences))]


class RandomTextModel(torch.nn.Module):
    r"""A text embedding model that takes the tokenized input from
    :class:`WhiteSpaceHashTokenizer` and outputs random embeddings. Should be
    used only for testing purposes.
    """
    def __init__(self, text_emb_channels: int):
        self.text_emb_channels = text_emb_channels
        super().__init__()

    def forward(self, feat: dict[str, MultiNestedTensor]):
        input_ids = feat['input_ids'].to_dense(fill_value=0)
        _ = feat['attention_mask'].to_dense(fill_value=0)
        return torch.rand(size=(input_ids.shape[0], 1, self.text_emb_channels))
