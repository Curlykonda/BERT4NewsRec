import torch.nn as nn
import torch
from torch.autograd import Variable

import math


class LearnablePositionEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

class TrigonometricPositionEmbedding(nn.Module):
    '''
    Fixed positional embedding with sinusoid functions
    c.f. "Attention is all you need", Vaswani et al., 2017
    '''
    def __init__(self, d_model, dropout, max_len=5000):
        super(TrigonometricPositionEmbedding, self).__init__()

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
        #TODO: check if the forward pass add this to the WordEmb or if we should rather return only the PosEmb and sum them later
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
