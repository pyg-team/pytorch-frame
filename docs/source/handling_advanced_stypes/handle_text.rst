Handling Text Columns
=====================

:pyf:`PyTorch Frame` supports text columns by utilizing or finetuning external pre-trained
text embedding models, such as language models. Currently, we support
:class:`stype.text_embedded<torch_frame.stype>` and :class:`stype.text_tokenized<torch_frame.stype>`
for text columns.
:class:`stype.text_embedded<torch_frame.stype>` uses text embedding model(s) to pre-encode text columns into embeddings
(those models are frozen during model training).
:class:`stype.text_tokenized<torch_frame.stype>` finetunes underlying text embedding models by using the
gradients backpropagated from the tabular learning. Now :pyf:`PyTorch Frame` pipeline supports supervised finetuning
on text models with other libraries such as the `PEFT <https://huggingface.co/docs/peft/>`_.


.. contents::
    :local:

Handling Text Columns in a Benchmark Dataset
--------------------------------------------

:pyf:`PyTorch Frame` provides a collection of tabular benchmark datasets
with text columns, such as :obj:`~torch_frame.datasets.MultimodalTextBenchmark`.

In :pyf:`PyTorch Frame`, you can specify text columns as
:class:`stype.text_embedded<torch_frame.stype>`. This will
encode text columns using a user-specified text embedding model(s) during the
dataset materialization stage.
You can also specify text columns as
:class:`stype.text_tokenized<torch_frame.stype>`, which will finetune
user-specified text model(s) during the training stage.

The processes of initializing and materializing datasets are similar
to :doc:`/get_started/introduction`. Below we highlight the difference.


Using Pre-trained Text Embeddings
---------------------------------

For text columns with :class:`stype.text_embedded<torch_frame.stype>` that
utilizes pre-trained text embeddings,
first you need to specify your text embedding model. Here, we use the
`SentenceTransformer <https://www.sbert.net/>`_ package.

.. code-block:: bash

    pip install -U sentence-transformers


Specifying Text Embedders
~~~~~~~~~~~~~~~~~~~~~~~~~

Next we create a text encoder class that encodes a list of strings into text embeddings.

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
            embeddings = self.model.encode(sentences, convert_to_numpy=False,
                                            convert_to_tensor=True)
            return embeddings.cpu()

Then we instantiate :obj:`~torch_frame.config.TextEmbedderConfig` for our text embedding model as follows.

.. code-block:: python

    from torch_frame.config.text_embedder import TextEmbedderConfig

    device = (torch.device('cuda')
          if torch.cuda.is_available() else torch.device('cpu'))
    text_embedder = TextToEmbedding(device)
    col_to_text_embedder_cfg = TextEmbedderConfig(text_embedder=text_embedder, batch_size=5)

Here :obj:`text_embedder` maps a list of sentences into PyTorch Tensor embeddings
in mini-batch, where :obj:`batch_size` represents the batch size.
Also, notice that we allow user to specify a dictionary of :obj:`text_embedder`s for different
text columns with :class:`stype.text_embedded<torch_frame.stype>`. This also allows using
:obj:`text_embedder`s with different embedding size for different text columns.

Embedding Text Columns for a Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch_frame
    from torch_frame.datasets import MultimodalTextBenchmark

    dataset = MultimodalTextBenchmark(
        root='/tmp/multimodal_text_benchmark/wine_reviews',
        name='wine_reviews',
        col_to_text_embedder_cfg=col_to_text_embedder_cfg,
    )

    len(dataset)
    >>> 105154

    dataset.feat_cols  # This dataset contains one text column `description`
    >>> ['description', 'country', 'province', 'points', 'price']

    dataset.col_to_stype['description']
    >>> <stype.text_embedded: 'text_embedded'>

    # Materialize will call pre-defined encoding for text columns
    dataset.materialize(path='/tmp/multimodal_text_benchmark/wine_reviews/data.pt')

    # Text embedding of shape [num_rows, num_text_cols, -1]
    dataset.tensor_frame.feat_dict[torch_frame.text_embedded].shape
    >>> (105154, 1, -1)

    # Use `MultiEmbeddingTensor` to allow different text columns to
    # be encoded to different dimensions
    type(dataset.tensor_frame.feat_dict[torch_frame.text_embedded])
    >>> "<class 'torch_frame.data.multi_embedding_tensor.MultiEmbeddingTensor'>"

.. note::
    Internally, :class:`~torch_frame.stype.text_embedded` is grouped together in the parent stype :class:`~torch_frame.stype.embedding` within :obj:`TensorFrame`.

It is strongly recommended to cache :class:`~torch_frame.TensorFrame`
by specifying the :obj:`path` during :meth:`~torch_frame.data.Dataset.materialize`,
as embedding texts in every materialization run can be quite time-consuming.
Once cached, :class:`~torch_frame.TensorFrame` can be reused for
subsequent :meth:`~torch_frame.data.Dataset.materialize` calls.

Fusing Text Embeddings into Tabular Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:pyf:`PyTorch Frame` offers :class:`~torch_frame.nn.encoder.LinearEmbeddingEncoder` designed
to encode pre-computed embeddings. This encoder applies linear function over the
pre-computed embeddings, which can easily handle :obj:`~torch_frame.stype.text_embedded`.

As mentioned earlier, :class:`~torch_frame.stype.text_embedded` is stored together with other :obj:`embeddings` in :obj:`TensorFrame`, so we only need to specify the encoder for parent :obj:`~torch_frame.stype`, i.e. :class:`~torch_frame.stype.embedding`, in the :obj:`stype_encoder_dict`.

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
:class:`~torch_frame.nn.encoder.StypeWiseFeatureEncoder` to handle text columns.


Finetuning Underlying Text Models
---------------------------------

For text columns with :class:`stype.text_tokenized<torch_frame.stype>`
that finetunes underlying text models during the tabular learning, you need to specify both
of the tokenization and encoding.
Here, we use the
`Transformers <https://huggingface.co/docs/transformers>`_ package.

.. code-block:: bash

    pip install transformers


Specifying Text Tokenization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different from text columns with :class:`stype.text_embedded<torch_frame.stype>`, text columns with
:class:`stype.text_tokenized<torch_frame.stype>` will be tokenized at first during the materialization
stage. Let's create a tokenization class that tokenizes a list of strings to a dictionary of PyTorch Tensors,
where the keys include :obj:`input_ids` and :obj:`attention_mask`, and values are tokens and masks tensors.


.. code-block:: python

    from typing import List
    from transformers import AutoTokenizer
    from torch_frame.typing import TextTokenizationOutputs

    class TextToEmbeddingTokenization:
        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        def __call__(self, sentences: List[str]) -> TextTokenizationOutputs:
            # Tokenize batches of sentences
            return self.tokenizer(sentences, truncation=True, padding=True,
                                  return_tensors='pt')

Then we instantiate :obj:`~torch_frame.config.TextTokenizerConfig` for our text embedding model as follows.

.. code-block:: python

    from torch_frame.config.text_tokenizer import TextTokenizerConfig

    text_tokenizer = TextToEmbeddingTokenization()
    col_to_text_tokenizer_cfg = TextTokenizerConfig(text_tokenizer=text_tokenizer, batch_size=10000)


Here :obj:`text_tokenizer` maps a list of sentences into a dictionary of PyTorch Tensors,
in mini-batch, where :obj:`batch_size` represents the batch size.
Also, notice that we allow user to specify a dictionary of :obj:`text_tokenizer`s for different
text columns with :class:`stype.text_tokenized<torch_frame.stype>`.


Tokenizing Text Columns for a Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch_frame
    from torch_frame.datasets import MultimodalTextBenchmark

    dataset = MultimodalTextBenchmark(
        root='/tmp/multimodal_text_benchmark/wine_reviews',
        name='wine_reviews',
        text_stype=torch_frame.text_tokenized,
        col_to_text_tokenizer_cfg=col_to_text_tokenizer_cfg,
    )

    len(dataset)
    >>> 105154

    dataset.feat_cols  # This dataset contains one text column `description`
    >>> ['description', 'country', 'province', 'points', 'price']

    dataset.col_to_stype['description']
    >>> <stype.text_tokenized: 'text_tokenized'>

    # Materialize will call tokenizer for text columns
    dataset.materialize()

    # A dictionary of text tokenization results
    dataset.tensor_frame.feat_dict[torch_frame.text_tokenized]
    >>> {'input_ids': MultiNestedTensor(num_rows=105154, num_cols=1, device='cpu'), 'attention_mask': MultiNestedTensor(num_rows=105154, num_cols=1, device='cpu')}


Finetuning Text Models with Tabular Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To finetune the underlying text models together with tabular learning,
we need at first specify how to embed tokenization results to text embeddings and how to finetune the text model.
Here we use `PEFT <https://huggingface.co/docs/peft>`_ package to use
`LoRA <https://arxiv.org/abs/2106.09685>`_ finetune the text model.

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


Notice that we use a dictionary of :obj:`~torch_frame.data.MultiNestedTensor` to store the tokenized results.
The reason we use dictionary is that tokenization returns multiple text model inputs such as
:obj:`input_ids` and :obj:`attention_mask` as shown before.
And the reason we use a :obj:`~torch_frame.data.MultiNestedTensor` for each text input is that for each row or sentence,
model input (such as :obj:`input_ids`) has different length. During the :meth:`forward`, you can
transform each :obj:`~torch_frame.data.MultiNestedTensor` back to a two-dimensional PyTorch Tensor by using
:meth:`~torch_frame.data.MultiNestedTensor.to_dense` with a specific padding value by specifying the :obj:`fill_value`.

Similar to the one for :obj:`~torch_frame.stype.text_embedded`, :pyf:`PyTorch Frame` offers
:class:`~torch_frame.nn.encoder.LinearModelEncoder` designed
to encode columns embeddings with enabling gradients backpropagated to underlying embedding models.
This encoder applies different linear function over different
column embeddings, which can easily handle :obj:`~torch_frame.stype.text_tokenized`.

.. code-block:: python

    from torch_frame.config import ModelConfig
    from torch_frame.nn.encoder import (
        EmbeddingEncoder,
        LinearEncoder,
        LinearModelEncoder,
    )

    model_cfg = ModelConfig(model=TextToEmbeddingFinetune(), out_channels=768)
    col_to_model_cfg = {
        col_name: model_cfg
        for col_name in dataset.tensor_frame.col_names_dict[
            torch_frame.text_tokenized]
    }

    stype_encoder_dict = {
        stype.categorical: EmbeddingEncoder(),
        stype.numerical: LinearEncoder(),
        stype.text_tokenized: LinearModelEncoder(col_to_model_cfg=col_to_model_cfg)
    }

We provides :class:`~torch_frame.config.ModelConfig` to specify the text model to finetune and its output size.
Then you can specify the model config for each :obj:`~torch_frame.stype.text_tokenized` columns
in a dictionary and pass it to the :class:`~torch_frame.nn.encoder.LinearModelEncoder`.

Then, :obj:`stype_encoder_dict` can be directly fed into
:class:`~torch_frame.nn.encoder.StypeWiseFeatureEncoder` to handle text columns.


Please refer to the
`pytorch-frame/examples/transformers_text.py <https://github.com/pyg-team/pytorch-frame/blob/master/examples/transformers_text.py>`_
for more text embedding and finetuning information with `Transformers <https://huggingface.co/docs/transformers>`_ package.

Also, please refer to the
`pytorch-frame/examples/llm_embedding.py <https://github.com/pyg-team/pytorch-frame/blob/master/examples/llm_embedding.py>`_
for more text embedding information with large language models such as
`OpenAI embeddings <https://platform.openai.com/docs/guides/embeddings>`_ and
`Cohere embed <https://docs.cohere.com/reference/embed>`_.
