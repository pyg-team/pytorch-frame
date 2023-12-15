Handling Text Columns
=====================

:pyf:`PyTorch Frame` handles text columns by utilizing text embedding models, which can be pre-trained language models.
We support two major options of utilizing text embedding models:

(1) To pe-encode texts into embeddings at the materialization stage (so that the model parameters are frozen during training stage)

(2) To generate text embeddings during the training stage and finetune their model parameters.

These options have trade-off. The option (1) allows faster training, while option (2)
allows more accurate prediction but with more costly training due to fine-tuning into the text models.
In :pyf:`PyTorch Frame`, one can specify which option to use for each text column by simply
specifying :class:`~torch_frame.stype`: :class:`stype.text_embedded<torch_frame.stype>`
for option (1) and :class:`stype.text_tokenized<torch_frame.stype>` for option (2).
Let's use a real-world dataset to learn how to achieve this.

.. contents::
    :local:

Handling Text Columns in a Real-World Dataset
---------------------------------------------

:pyf:`PyTorch Frame` provides a collection of tabular benchmark datasets
with text columns, such as :obj:`~torch_frame.datasets.MultimodalTextBenchmark`.

As we briefly discussed, :pyf:`PyTorch Frame` provides two semantic types for
text columns:

1. :class:`stype.text_embedded<torch_frame.stype>` will pre-encode texts using user-specified
text embedding models at the dataset materialization stage.

2. :class:`stype.text_tokenized<torch_frame.stype>` will tokenize texts using user-specified
text tokenizers at the dataset materialization stage. The tokenized texts (sequences of integers)
are fed into text models at training stage, and the parameters of the text models are fine-tuned.

The processes of initializing and materializing datasets are similar
to :doc:`/get_started/introduction`.
Below we highlight the difference for each semantic type.

Pre-encode texts into embeddings
--------------------------------

For :class:`stype.text_embedded<torch_frame.stype>`, first you need to specify the text embedding models.
Here, we use the `SentenceTransformer <https://www.sbert.net/>`_ package.

.. code-block:: bash

    pip install -U sentence-transformers

Specifying Text Embedders
~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we create a text encoder class that encodes a list of strings into text embeddings in a mini-batch manner.

.. code-block:: python

    from typing import List
    import torch
    from torch import Tensor
    from sentence_transformers import SentenceTransformer

    class TextToEmbedding:
        def __init__(self, device: torch.device):
            self.model = SentenceTransformer('all-distilroberta-v1', device=device)

        def __call__(self, sentences: List[str]) -> Tensor:
            # Encode a list of batch_size sentences into a PyTorch Tensor of
            # size [batch_size, emb_dim]
            embeddings = self.model.encode(
                sentences,
                convert_to_numpy=False,
                convert_to_tensor=True,
            )
            return embeddings.cpu()

Then we instantiate :obj:`~torch_frame.config.TextEmbedderConfig` that specifies
the :obj:`text_embedder` and :obj:`batch_size` we use to pre-encode
the texts using the :obj:`text_embedder`.

.. code-block:: python

    from torch_frame.config.text_embedder import TextEmbedderConfig

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    col_to_text_embedder_cfg = TextEmbedderConfig(
        text_embedder=TextToEmbedding(device),
        batch_size=8,
    )

Note that Transformer-based text embedding models are often GPU memory intensive,
so it is important to specify a reasonable :obj:`batch_size` (e.g., :obj:`8`).
Also, note that we will use the same :obj:`~torch_frame.config.TextEmbedderConfig`
across all text columns by default.
If we want to use different :obj:`text_embedder` for different text columns
(let's say :obj:`"text_col0"` and :obj:`"text_col1"`), we can
use a dictionary as follows:

.. code-block:: python

    # Prepare text_embedder0 and text_embedder1 for text_col0 and text_col1, respectively.
    col_to_text_embedder_cfg = {
        "text_col0":
        TextEmbedderConfig(text_embedder=text_embedder0, batch_size=4),
        "text_col1":
        TextEmbedderConfig(text_embedder=text_embedder1, batch_size=8),
    }

Embedding Text Columns for a Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once :obj:`col_to_text_embedder_cfg` is specified, we can pass it to
:obj:`Dataset<torch_frame.data.Dataset>` object as follows.

.. code-block:: python

    import torch_frame
    from torch_frame.datasets import MultimodalTextBenchmark

    dataset = MultimodalTextBenchmark(
        root='/tmp/multimodal_text_benchmark/wine_reviews',
        name='wine_reviews',
        col_to_text_embedder_cfg=col_to_text_embedder_cfg,
    )

    dataset.feat_cols  # This dataset contains one text column `description`
    >>> ['description', 'country', 'province', 'points', 'price']

    dataset.col_to_stype['description']
    >>> <stype.text_embedded: 'text_embedded'>

We then call :obj:`dataset.materialize(path=...)`, which will use text embedding models
to pre-encode :obj:`text_embedded` columns based on the given :obj:`col_to_text_embedder_cfg`.

.. code-block:: python

    # Pre-encode text columns based on col_to_text_embedder_cfg. This may take a while.
    dataset.materialize(path='/tmp/multimodal_text_benchmark/wine_reviews/data.pt')

    len(dataset)
    >>> 105154

    # Text embeddings are stored as MultiNestedTensor
    dataset.tensor_frame.feat_dict[torch_frame.embedding]
    >>> MultiNestedTensor(num_rows=105154, num_cols=1, device='cpu')

It is strongly recommended to specify the :obj:`path` during :meth:`~torch_frame.data.Dataset.materialize`.
It will cache generated :class:`~torch_frame.TensorFrame`, therefore, avoiding embedding texts in
every materialization run, which can be quite time-consuming.
Once cached, :class:`~torch_frame.TensorFrame` can be reused for
subsequent :meth:`~torch_frame.data.Dataset.materialize` calls.

.. note::
    Internally, :class:`~torch_frame.stype.text_embedded` is grouped together
    in the parent stype :class:`~torch_frame.stype.embedding` within :obj:`TensorFrame`.

Fusing Text Embeddings into Tabular Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:pyf:`PyTorch Frame` offers :class:`~torch_frame.nn.encoder.LinearEmbeddingEncoder` designed
for encoding :class:`~torch_frame.stype.embedding` within :class:`TensorFrame`.
This module applies linear function over the pre-computed embeddings.

.. code-block:: python

    from torch_frame.nn.encoder import (
        EmbeddingEncoder,
        LinearEmbeddingEncoder,
        LinearEncoder,
    )

    stype_encoder_dict = {
        stype.categorical: EmbeddingEncoder(),
        stype.numerical: LinearEncoder(),
        stype.embedding: LinearEmbeddingEncoder()
    }

Then, :obj:`stype_encoder_dict` can be directly fed into
:class:`~torch_frame.nn.encoder.StypeWiseFeatureEncoder`.

Fine-tuning Text Models
-----------------------

In contrast to :class:`stype.text_embedded<torch_frame.stype>`,
:class:`stype.text_tokenized<torch_frame.stype>` does minimal processing at the dataset materialization stage
by only tokenizing raw texts, i.e., transforming strings into sequences of integers.
Then, during the training stage, the fully-fledged text models take the tokenized sentences as input
and output text embeddings, which allows the text models to be trained in an end-to-end manner.

Here, we use the
`Transformers <https://huggingface.co/docs/transformers>`_ package.

.. code-block:: bash

    pip install transformers

Specifying Text Tokenization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In :class:`stype.text_tokenized<torch_frame.stype>`, text columns will be tokenized
during the dataset materialization stage.
Let's first create a tokenization class that tokenizes a list of strings to a dictionary of :class:`torch.Tensor`.

.. code-block:: python

    from typing import List
    from transformers import AutoTokenizer
    from torch_frame.typing import TextTokenizationOutputs

    class TextToEmbeddingTokenization:
        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        def __call__(self, sentences: List[str]) -> TextTokenizationOutputs:
            # Tokenize batches of sentences
            return self.tokenizer(
                sentences,
                truncation=True,
                padding=True,
                return_tensors='pt',
            )

Here, the output :class:`TextTokenizationOutputs` is a dictionary,
where the keys include :obj:`input_ids` and :obj:`attention_mask`, and the values
contain :pytorch:`PyTorch` tensors of tokens and attention masks.

Then we instantiate :class:`~torch_frame.config.TextTokenizerConfig` for our text embedding model as follows.

.. code-block:: python

    from torch_frame.config.text_tokenizer import TextTokenizerConfig

    col_to_text_tokenizer_cfg = TextTokenizerConfig(
        text_tokenizer=TextToEmbeddingTokenization(),
        batch_size=10_000,
    )

Here :obj:`text_tokenizer` maps a list of sentences into a dictionary of :class:`torch.Tensor`,
which are input to text models at training time.
Tokenization is processed in mini-batch, where :obj:`batch_size` represents the batch size.
Because text tokenizer runs fast on CPU, we can specify relatively large :obj:`batch_size` here.
Also, note that we allow to specify a dictionary of :obj:`text_tokenizer` for different
text columns with :class:`stype.text_tokenized<torch_frame.stype>`.

.. code-block:: python

    # Prepare text_tokenizer0 and text_tokenizer1 for text_col0 and text_col1, respectively.
    col_to_text_tokenizer_cfg = {
        "text_col0":
        TextTokenizerConfig(text_tokenizer=text_tokenizer0, batch_size=10000),
        "text_col1":
        TextTokenizerConfig(text_tokenizer=text_tokenizer1, batch_size=20000),
    }


Tokenizing Text Columns for a Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once :obj:`col_to_text_tokenizer_cfg` is specified, we can pass it to
:obj:`Dataset<torch_frame.data.Dataset>` object as follows.

.. code-block:: python

    import torch_frame
    from torch_frame.datasets import MultimodalTextBenchmark

    dataset = MultimodalTextBenchmark(
        root='/tmp/multimodal_text_benchmark/wine_reviews',
        name='wine_reviews',
        text_stype=torch_frame.text_tokenized,
        col_to_text_tokenizer_cfg=col_to_text_tokenizer_cfg,
    )

    dataset.col_to_stype['description']
    >>> <stype.text_tokenized: 'text_tokenized'>


We then call :obj:`dataset.materialize()`, which will use the text tokenizers
to pre-tokenize :obj:`text_tokenized` columns based on the given :obj:`col_to_text_tokenizer_cfg`.

.. code-block:: python

    # Pre-encode text columns based on col_to_text_tokenizer_cfg.
    dataset.materialize()

    # A dictionary of text tokenization results
    dataset.tensor_frame.feat_dict[torch_frame.text_tokenized]
    >>> {'input_ids': MultiNestedTensor(num_rows=105154, num_cols=1, device='cpu'), 'attention_mask': MultiNestedTensor(num_rows=105154, num_cols=1, device='cpu')}


Notice that we use a dictionary of :obj:`~torch_frame.data.MultiNestedTensor` to store the tokenized results.
The reason we use dictionary is that common text tokenizers usually return multiple text model inputs such as
:obj:`input_ids` and :obj:`attention_mask` as shown before.

Finetuning Text Models with Tabular Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we use `PEFT <https://huggingface.co/docs/peft>`_ package and the
`LoRA <https://arxiv.org/abs/2106.09685>`_ strategy to finetune the underlying text model.

.. code-block:: bash

    pip install peft

Next we need to specify the text model embedding with `LoRA <https://arxiv.org/abs/2106.09685>`_ finetuning.

.. code-block:: python

    import torch
    from torch import Tensor
    from transformers import AutoModel
    from torch_frame.data import MultiNestedTensor
    from peft import LoraConfig, TaskType, get_peft_model

    class TextToEmbeddingFinetune(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = AutoModel.from_pretrained('distilbert-base-uncased')
            # Set LoRA config
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=32,
                lora_alpha=32,
                inference_mode=False,
                lora_dropout=0.1,
                bias="none",
                target_modules=["ffn.lin1"],
            )
            # Update the model with LoRA config
            self.model = get_peft_model(self.model, peft_config)

        def forward(self, feat: dict[str, MultiNestedTensor]) -> Tensor:
            # [batch_size, batch_max_seq_len]
            input_ids = feat["input_ids"].to_dense(fill_value=0).squeeze(dim=1)
            mask = feat["attention_mask"].to_dense(fill_value=0).squeeze(dim=1)

            # Get text embeddings for each text tokenized column
            # `out.last_hidden_state` has the shape:
            # [batch_size, batch_max_seq_len, text_model_out_channels]
            out = self.model(input_ids=input_ids, attention_mask=mask)

            # Use the CLS embedding to represent the sentence embedding
            # Return value has the shape [batch_size, 1, text_model_out_channels]
            return out.last_hidden_state[:, 0, :].unsqueeze(1)


As mentioned above, we store text model inputs in the format of dictionary of
:obj:`~torch_frame.data.MultiNestedTensor`.
During the :meth:`forward`, we first transform each
:obj:`~torch_frame.data.MultiNestedTensor` back to padded :class:`torch.Tensor` by using
:meth:`~torch_frame.data.MultiNestedTensor.to_dense` with the padding value
specified by :obj:`fill_value`.

:pyf:`PyTorch Frame` offers :class:`~torch_frame.nn.encoder.LinearModelEncoder` designed
to flexibly apply any :pytorch:`PyTorch` module in per-column manner. We first specify :class:`ModelConfig`
object that declares the module to apply to each column.

.. note::
    :class:`ModelConfig` has two arguments to specify:
    First, :obj:`model` is a learnable :pytorch:`PyTorch` module that takes per-column
    tensors in :class:`TensorFrame` as input
    and outputs per-column embeddings. Formally, :obj:`model` takes a :obj:`TensorData` object of
    shape :obj:`[batch_size, 1, \*]` as input and outputs embeddings of shape
    :obj:`[batch_size, 1, out_channels]`. Then, :obj:`out_channels` specifies the output
    embedding dimensionality of :obj:`model`.

.. code-block:: python

    from torch_frame.config import ModelConfig
    model_cfg = ModelConfig(model=TextToEmbeddingFinetune(), out_channels=768)
    col_to_model_cfg = {"description": model_cfg}


Once :obj:`col_to_model_cfg` is specified, we pass it to :class:`LinearModelEncoder`
so that it applies the specified :obj:`model` to the desired column.
In this case, we apply the model :class:`TextToEmbeddingFinetune` to :obj:`text_tokenized`
column of :class:`TensorFrame`.

.. code-block:: python

    from torch_frame.nn import (
        EmbeddingEncoder,
        LinearEncoder,
        LinearModelEncoder,
    )

    stype_encoder_dict = {
        stype.categorical: EmbeddingEncoder(),
        stype.numerical: LinearEncoder(),
        stype.text_tokenized: LinearModelEncoder(col_to_model_cfg=col_to_model_cfg),
    }

The resulting :obj:`stype_encoder_dict` can be directly fed into
:class:`~torch_frame.nn.encoder.StypeWiseFeatureEncoder`.

Please refer to the
`pytorch-frame/examples/transformers_text.py <https://github.com/pyg-team/pytorch-frame/blob/master/examples/transformers_text.py>`_
for more text embedding and finetuning information with `Transformers <https://huggingface.co/docs/transformers>`_ package.

Also, please refer to the
`pytorch-frame/examples/llm_embedding.py <https://github.com/pyg-team/pytorch-frame/blob/master/examples/llm_embedding.py>`_
for more text embedding information with large language models such as
`OpenAI embeddings <https://platform.openai.com/docs/guides/embeddings>`_ and
`Cohere embed <https://docs.cohere.com/reference/embed>`_.
