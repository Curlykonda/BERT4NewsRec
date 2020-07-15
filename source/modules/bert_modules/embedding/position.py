
import torch.nn as nn
import torch
from torch.autograd import Variable

import math


class LearnablePositionEmbedding(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    @staticmethod
    def code():
        return 'lpe'

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

class GaussNoiseEmb(nn.Module):
    """ Experimental module to add random noise instead of positional emb """

    def __init__(self, d_emb):
        super().__init__()

        self.d_emb = d_emb

    @staticmethod
    def code():
        return 'gnoise'

    def forward(self, x):
        b, l, d = x.shape
        if d == self.d_emb:
            return torch.rand_like(x).to(x.device)
        else:
            return torch.rand([b, l, self.d_emb], requires_grad=False, device=x.device)


class TrigonometricPositionEmbedding(nn.Module):
    '''
    Fixed positional embedding with sinusoid functions
    c.f. "Attention is all you need", Vaswani et al., 2017
    '''
    def __init__(self, d_model, max_len=5000):
        super(TrigonometricPositionEmbedding, self).__init__()

        #self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
        #self.register_buffer('pe', pe)

    @staticmethod
    def code():
        # trigonometric positional embedding
        return 'tpe'

    def forward(self, x):

        if self.pe.device != x.device:
            self.pe = self.pe.to(x.device)

        pe = Variable(self.pe[:, :x.size(1)].repeat(x.size(0), 1, 1), requires_grad=False)
        return pe


POS_EMBS = {
    LearnablePositionEmbedding.code(): LearnablePositionEmbedding,
    TrigonometricPositionEmbedding.code(): TrigonometricPositionEmbedding,
    GaussNoiseEmb.code(): GaussNoiseEmb
}