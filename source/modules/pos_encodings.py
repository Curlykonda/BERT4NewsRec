import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import math, copy, time
import matplotlib.pyplot as plt

class SinePositionalEncoding(nn.Module):
    '''
    Implement the sinoid PE function.
    c.f. "Attention is all you need", Vaswani et al., 2017
    '''

    def __init__(self, d_model, dropout, max_len=5000):
        super(SinePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

    def plot_sine_waves(self):
        '''
        Below the positional encoding will add in a sine wave based on position.
        The frequency and offset of the wave is different for each dimension.

        '''
        plt.figure(figsize=(15, 5))
        pe = SinePositionalEncoding(20, 0)
        y = pe.forward(Variable(torch.zeros(1, 100, 20)))
        plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
        plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])