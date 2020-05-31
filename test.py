import torch
import torch.nn as nn
import time

if __name__ == "__main__":
    b = 10
    n_art = 100
    l_art = 30
    d_we = 128
    d_art = 64

    k1 = 3

    inp = torch.randn((b, n_art, l_art, d_we))
    print("Input {}".format(inp.shape))
    print("(B x N_art x L_art x D_we)")
    conv1 = nn.Conv1d(1, d_art, kernel_size=(k1, d_we), padding=(k1 - 2, 0))

    out = []
    t0 = time.time()
    for i in range(n_art):
        conv_out = conv1(inp[:, i, :, :].unsqueeze(1)) # contextualise words
        out.append(torch.sum(conv_out, dim=2))

    out = torch.stack(out, dim=2).squeeze().transpose(1, 2)  # (B x N_art x D_art)
    t1 = time.time()
    print("Iterative 1D convs: {}".format(out.shape))
    print("(B x N_art x D_art)")
    print("{:.3f}s".format(t1-t0))

    conv1a = nn.Conv1d(n_art, d_art, kernel_size=(k1, d_we), padding=(k1 - 2, 0))
    conv_out = conv1a(inp)
    print(conv_out.shape)
