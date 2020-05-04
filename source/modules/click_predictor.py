import numpy as np

import torch
import torch.nn as nn
import torch.functional as F

class SimpleDot(nn.Module):
    def __init__(self, dim_user_rep, dim_cand_rep):
        super(SimpleDot, self).__init__()

        self.dim_user_rep = dim_user_rep
        self.dim_cand_rep = dim_cand_rep

        assert dim_user_rep == dim_cand_rep


    def forward(self, user_rep, candidate_reps):
        # (B x user_rep_dim) * (B x news_rep_dim x n_candidates)
        # => (B x n_candidates)
        return torch.bmm(user_rep.unsqueeze(1), candidate_reps).squeeze()

class MLP(nn.Module):
    def __init__(self, dim_user_rep, dim_cand_rep, n_outputs=1, hidden_units=[200, 100], act_func='tanh'):

        super(MLP, self).__init__()

        self.dim_user_rep = dim_user_rep
        self.dim_cand_rep = dim_cand_rep
        self.n_outputs = n_outputs

        self.mlp = nn.ModuleList()

        self.act_funcs = {'tanh': nn.Tanh(), 'relu': nn.ReLU()}

        if act_func not in self.act_funcs.keys():
            raise KeyError()
        else:
            self.act_func = self.act_funcs[act_func]

        for i_layer, hidden in enumerate(hidden_units):
            if i_layer == 0:
                self.mlp.add(nn.Linear(in_features=(self.dim_cand_rep + self.dim_user_rep), out_features=hidden))
            else:
                self.mlp.add(nn.Linear(in_features=hidden_units[i_layer - 1], out_features=hidden))

            if i_layer < len(hidden_units)-1:
                self.mlp.add(self.act_func)

    def forward(self, user_rep, cand_rep):
        x = torch.cat((user_rep, cand_rep), dim=1)
        raw_scores = self.mlp(x)

        return raw_scores