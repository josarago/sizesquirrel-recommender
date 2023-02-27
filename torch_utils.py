import pandas as pd
from torch import nn


def count_parameters(model: nn.Module) -> pd.DataFrame:
    modules = []
    parameters = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            modules.append(name)
            parameters.append(parameter.numel())
    return pd.DataFrame(data={"module": modules, "parameters": parameters})
