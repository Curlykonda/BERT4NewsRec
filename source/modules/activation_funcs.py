import torch.nn as nn
import torch
import math


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def get_activation_func(act_func: str) -> nn.Module:
    act_func = act_func.lower()

    if "relu" == act_func:
        return nn.ReLU()
    elif "gelu" == act_func:
        return GELU()
    elif "tanh" == act_func:
        return nn.Tanh()
    else:
        raise ValueError("{} is invalid activation function".format(act_func))


ACT_FUNC ={}