import torch
import torch.nn as nn
import torch.functional as F


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, device):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        self.output = None
        self.h_n = None

        self.gru = nn.GRUCell(input_size, hidden_size)

    def forward(self, input, h0=None, **kwargs):
        if h0 is None:
            b = input.shape[0]
            h0 = torch.zeros(b, self.hidden_size, device=self.device)

        output = []
        h_x = h0
        # input (B x d_news x seq_len) => (seq_len x B x d_news)
        input = input.permute(2, 0, 1)

        for i in range(self.n_layers):
            h_x = self.gru(input[i], h_x) # GRU input should be of shape (B, input_size):
            output.append(h_x) # (B, d_hidden)

        self.output = output
        self.h_n = h_x

        return output