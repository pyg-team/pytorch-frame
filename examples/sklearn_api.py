from typing import Any

import torch.nn as nn
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from torch_frame import stype
from torch_frame.data.stats import StatType
from torch_frame.nn import Trompt
from torch_frame.nn.models.trompt import Trompt
from torch_frame.utils.skorch import NeuralNetPytorchFrame

# load the diabetes dataset
X, y = load_diabetes(return_X_y=True, as_frame=True)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)


# define the function to get the module
def get_module(col_stats: dict[str, dict[StatType, Any]],
               col_names_dict: dict[stype, list[str]]) -> Trompt:
    channels = 8
    out_channels = 1
    num_prompts = 2
    num_layers = 3
    return Trompt(channels=channels, out_channels=out_channels,
                  num_prompts=num_prompts, num_layers=num_layers,
                  col_stats=col_stats, col_names_dict=col_names_dict,
                  stype_encoder_dicts=None)


# wrap the function in a NeuralNetPytorchFrame
# NeuralNetClassifierPytorchFrame and NeuralNetBinaryClassifierPytorchFrame
# are also available
net = NeuralNetPytorchFrame(
    module=get_module,
    criterion=nn.MSELoss(),
    max_epochs=10,
    verbose=1,
    lr=0.0001,
    batch_size=30,
)

# fit the model
net.fit(X_train, y_train)

# predict on the test set
y_pred = net.predict(X_test)

# calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(mse)
