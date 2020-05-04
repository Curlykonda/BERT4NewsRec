import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import math, copy, time

'''
Code from OpenNMT: http://nlp.seas.harvard.edu/2018/04/03/attention.html

https://doi.org/10.18653/v1/P17-4012
'''

class PersonalisedAttentionWu(nn.Module):
    def __init__(self, dim_pref_q, dim_news_rep):
        super(PersonalisedAttentionWu, self).__init__()

        self.dim_pref_q = dim_pref_q
        self.dim_news_rep = dim_news_rep

        self.proj_pref_q = nn.Sequential(
            nn.Linear(dim_pref_q, dim_news_rep),
            nn.Tanh()
        )

        self.attn_weights = None

    def forward(self, enc_input, pref_q):

        # enc_input.shape = (batch_size, d_news, news_len)

        # (B x d_pref_q) -> (B x d_news)
        pref_q = self.proj_pref_q(pref_q) # transform pref query

        attn_a = torch.bmm(torch.transpose(enc_input, 1, 2), pref_q.unsqueeze(2)).squeeze(-1) # dot product over batch http://pytorch.org/docs/0.2.0/torch.html#torch.bmm
        attn_weights = F.softmax(attn_a, dim=-1)
        self.attn_weights = attn_weights
        #assert torch.sum(attn_weights, dim=1) == torch.ones(attn_weights.shape[0], dtype=float) # (normalised) attn weights should sum to 1

        attn_w_rep = torch.matmul(enc_input, attn_weights.unsqueeze(2)).squeeze(-1) # attn-weighted representation r of i-th news

        return attn_w_rep

def clones(module, N):
    "Produce N identical modules."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)