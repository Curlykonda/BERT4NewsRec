import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import math

class NormalLinear(nn.Linear):
    """
    Linear layer (weight matrix + bias vector) initialised with 0-mean Gaussian

    Use to create Temporal Embedding from 1D input (e.g. time stamp)
    self.temp_embedding = NormalLinear(1, self.embedding_dim)
    """
    def forward(self, t):
        return self.forward(t)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)

class NormalLinearTimeDiff(NormalLinear):

    def forward(self, t1, t2):
        return self.forward(t2-t1)