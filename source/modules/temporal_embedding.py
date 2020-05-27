import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import math

class NormalLinear(nn.Linear):
    """
    Linear layer (weight matrix + bias vector) initialised with 0-mean Gaussian
    """

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)

class LinearProject(nn.Module):
    """
    Use to create Temporal Embedding from 0D input (e.g. time stamp)
    (B x L_hist) -> (B x L_hist x D_e)
    """
    def __init__(self, d_model=768):
        super(LinearProject, self).__init__()

        #self.lin = NormalLinear(1, d_model)
        self.lin = nn.Linear(1, d_model)
        self.d_model = d_model
        #self.lin.reset_parameters()

    @staticmethod
    def code():
        # linear projected temporal embedding
        return 'lte'

    def forward(self, t):
        # project single-value time stamps to temporal embedding
        # (B x L_hist) -> (B x L_hist x D_e)
        t = t.float()
        t_out = self.lin(t.unsqueeze(2))
        return t_out / math.sqrt(self.d_model)

class NeuralFunc(nn.Module):
    def __init__(self, embed_dim, hidden_units=[256, 768], act_func="relu"):
        super(NeuralFunc, self).__init__()

        assert embed_dim == hidden_units[-1], "Last layer with {} units must match embedding dimension {}".format(hidden_units[-1], embed_dim)

        self.lin_layers = nn.ModuleList()

        for i, hidden in enumerate(hidden_units):
            if 0 == i:
                self.lin_layers.append(nn.Linear(1, hidden))
            else:
                self.lin_layers.append(nn.Linear(hidden_units[i-1], hidden))

        func = act_func.lower()

        if "relu" == func:
            self.activation = nn.ReLU()
        elif "gelu" == func:
            raise NotImplementedError()
        elif "tanh" == func:
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError()

    @staticmethod
    def code():
        # neural temporal embedding (func approx)
        return 'nte'

    def forward(self, x_in):
        x = x_in.unsqueeze(2).float()
        # input x: time stamp in UNIX format, i.e. single int value
        for i, lin in enumerate(self.lin_layers):
            if i < len(self.lin_layers)-1:
                x = self.activation(lin(x))
            else:
                x_out = lin(x)

        return x_out

TEMP_EMBS = {
    LinearProject.code(): LinearProject,
    NeuralFunc.code(): NeuralFunc
}