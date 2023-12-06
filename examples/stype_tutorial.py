import random

import numpy as np
import pandas as pd

from torch_frame import stype
from torch_frame.data import Dataset
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEmbeddingEncoder,
    LinearEncoder,
    MultiCategoricalEmbeddingEncoder,
    TimestampEncoder,
)
from torch_frame.nn.models.ft_transformer import FTTransformer
from torch_frame.typing import NAStrategy

# Numerical column
numerical = np.random.randint(0, 100, size=100)

# Categorical column
simple_categories = ['Type 1', 'Type 2', 'Type 3']
categorical = np.random.choice(simple_categories, size=100)

# Time column
time = pd.date_range(start='2023-01-01', periods=100, freq='D')

# Multicategorical column
categories = ['Category A', 'Category B', 'Category C', 'Category D']
multicategorical = [
    random.sample(categories, k=random.randint(0, len(categories)))
    for _ in range(100)
]

# Embedding column (assuming an embedding size of 5 for simplicity)
embedding_size = 5
embedding = np.random.rand(100, embedding_size)

# Create the DataFrame
df = pd.DataFrame({
    'Numerical': numerical,
    'Categorical': categorical,
    'Time': time,
    'Multicategorical': multicategorical,
    'Embedding': list(embedding)
})

# Displaying the first few rows of the DataFrame
print(df.head())

dataset = Dataset(
    df, col_to_stype={
        'Numerical': stype.numerical,
        'Categorical': stype.categorical,
        'Time': stype.timestamp,
        'Multicategorical': stype.multicategorical,
        'Embedding': stype.embedding
    }, target_col='Numerical')
dataset.materialize()
print(dataset.tensor_frame)
train_dataset, val_dataset, test_dataset = dataset[:0.8], dataset[
    0.8:0.9], dataset[0.9:]
# Set up data loaders
train_tensor_frame = train_dataset.tensor_frame

val_tensor_frame = val_dataset.tensor_frame
test_tensor_frame = test_dataset.tensor_frame

stype_encoder_dict = {
    stype.categorical: EmbeddingEncoder(),
    stype.numerical: LinearEncoder(),
    stype.embedding: LinearEmbeddingEncoder(),
    stype.multicategorical: MultiCategoricalEmbeddingEncoder(),
    stype.timestamp: TimestampEncoder(na_strategy=NAStrategy.MEDIAN_TIMESTAMP)
}

model = FTTransformer(
    channels=16,
    out_channels=1,
    num_layers=2,
    col_stats=dataset.col_stats,
    col_names_dict=train_tensor_frame.col_names_dict,
    stype_encoder_dict=stype_encoder_dict,
)

print(model(test_dataset.tensor_frame))
