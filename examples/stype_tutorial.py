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
from torch_frame.utils.infer_stype import infer_df_stype

# Numerical column
numerical = np.random.randint(0, 10, size=100)

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
multicategorical2 = [
    ','.join(random.sample(categories, k=random.randint(0, len(categories))))
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
print(infer_df_stype(df))
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
    col_names_dict=dataset.tensor_frame.col_names_dict,
    stype_encoder_dict=stype_encoder_dict,
)

print(model(dataset.tensor_frame[:10]))

# Multicategorical column
categories = ['Category A', 'Category B', 'Category C', 'Category D']
multicategorical1 = [
    random.sample(categories, k=random.randint(0, len(categories)))
    for _ in range(100)
]
multicategorical2 = [
    ','.join(random.sample(categories, k=random.randint(0, len(categories))))
    for _ in range(100)
]
multicategorical3 = [
    '/'.join(random.sample(categories, k=random.randint(0, len(categories))))
    for _ in range(100)
]
# Create the DataFrame
df = pd.DataFrame({
    'Multicategorical1': multicategorical1,
    'Multicategorical2': multicategorical2,
    'Multicategorical3': multicategorical3,
})

# Displaying the first few rows of the DataFrame
print(df.head())
dataset = Dataset(
    df, col_to_stype={
        'Multicategorical1': stype.multicategorical,
        'Multicategorical2': stype.multicategorical,
        'Multicategorical3': stype.multicategorical,
    }, col_to_sep={
        'Multicategorical2': ',',
        'Multicategorical3': '/'
    })
dataset.materialize()
print(dataset.col_stats)

# Multicategorical column
categories = ['Category A', 'Category B', 'Category C', 'Category D']
multicategorical1 = [
    random.sample(categories, k=random.randint(0, len(categories)))
    for _ in range(100)
]
multicategorical2 = [
    ','.join(random.sample(categories, k=random.randint(0, len(categories))))
    for _ in range(100)
]
multicategorical3 = [
    '/'.join(random.sample(categories, k=random.randint(0, len(categories))))
    for _ in range(100)
]
# Create the DataFrame
df = pd.DataFrame({
    'Multicategorical1': multicategorical1,
    'Multicategorical2': multicategorical2,
})

# Displaying the first few rows of the DataFrame
print(df.head())
dataset = Dataset(
    df, col_to_stype={
        'Multicategorical1': stype.multicategorical,
        'Multicategorical2': stype.multicategorical,
    }, col_to_sep=',')
dataset.materialize()
print(dataset.col_stats)

dates = pd.date_range(start="2023-01-01", periods=5, freq='D')
# Different timestamp formats
df = pd.DataFrame({
    'Time1': dates,  # ISO 8601 format (default)
    'Time2': dates.strftime('%Y-%m-%d %H:%M:%S'),
})
print(df.head())
dataset = Dataset(
    df, col_to_stype={
        'Time1': stype.timestamp,
        'Time2': stype.timestamp,
    }, col_to_time_format='%Y-%m-%d %H:%M:%S')

dataset.materialize()
print(dataset.col_stats)
