import torch
import torch.nn as nn

class PrefQueryWu(nn.Module):
    '''
    Given an embedded user id, create a preference query vector (that is used in personalised attention)

    '''
    def __init__(self, dim_emb_u_id=50, dim_pref_query=200, activation='relu', device='cpu'):
        super(PrefQueryWu, self).__init__()

        self.dim_pref_query = dim_pref_query
        self.dim_u_id = dim_emb_u_id

        self.lin_proj = nn.Linear(self.dim_u_id, self.dim_pref_query)

        assert activation in ['relu', 'tanh']

        if activation == 'relu':
            self.activation = nn.Tanh()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise KeyError()

    def forward(self, u_id):
        #print(u_id.shape)
        pref_query = self.lin_proj(u_id) # batch_size X u_id_emb_dim

        return self.activation(pref_query)