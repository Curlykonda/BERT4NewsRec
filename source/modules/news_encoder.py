import torch
import torch.nn as nn
import torch.functional as F

from source.modules.attention import PersonalisedAttentionWu

class NewsEncoderWuCNN(nn.Module):

    def __init__(self, n_filters=400, dim_pref_q=200, word_emb_dim=300, kernel_size=3, dropout_p=0.2):
        super(NewsEncoderWuCNN, self).__init__()

        self.n_filters = n_filters # output dimension
        self.dim_pref_q = dim_pref_q
        self.word_emb_dim = word_emb_dim

        self.dropout_in = nn.Dropout(p=dropout_p)

        self.cnn_encoder = nn.Sequential(nn.Conv1d(1, n_filters, kernel_size=(kernel_size, word_emb_dim), padding=(kernel_size - 2, 0)),
                        nn.ReLU(),
                        nn.Dropout(p=dropout_p)
        )

        self.pers_attn_word = PersonalisedAttentionWu(dim_pref_q, n_filters) # word level attn

    def forward(self, embedded_news, pref_query):
        contextual_rep = []
        # embedded_news.shape = batch_size X max_hist_len X max_title_len X word_emb_dim
        embedded_news = self.dropout_in(embedded_news)
        # encode each browsed news article and concatenate
        for n_news in range(embedded_news.shape[1]):

            # concatenate words
            article_one = embedded_news[:, n_news, :, :].squeeze(1) # shape = (batch_size, title_len, emb_dim)

            encoded_news = self.cnn_encoder(article_one.unsqueeze(1))
            # encoded_news.shape = batch_size X n_cnn_filters X max_title_len

            #pers attn
            contextual_rep.append(self.pers_attn_word(encoded_news.squeeze(-1), pref_query))
            assert contextual_rep[-1].shape[1] == self.n_filters # batch_size X n_cnn_filters

        return torch.stack(contextual_rep, axis=2) # batch_s X dim_news_rep X history_len

# Utility function to calc output size
def output_size(in_size, kernel_size, stride, padding):
  output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
  return output//2

def mot_pooling(x):
  # Max-over-time pooling
  # X is shape n,c,w
  x = F.max_pool1d(x, kernel_size=x.shape[2])
  return x


class KimCNN(torch.nn.Module):
  # Shape after conv is (batch, x, y)
  def __init__(self, n_filters, word_emb_dim, kernels=[3, 4, 5]):
    super(KimCNN, self).__init__()

    self.n_filters = n_filters
    self.word_emb_dim = word_emb_dim
    self.kernels = kernels
    self.convs = nn.ModuleList([
      nn.Conv1d(1, out_channels=n_filters, kernel_size=(kernel, word_emb_dim), padding=(kernel-2, 0))
      for kernel in kernels
    ])

  def forward(self, x):
    # Pass through each conv layer
    outs = [conv(x) for conv in self.convs]

    # Max over time pooling
    outs_pooled = [mot_pooling(out) for out in outs]
    # Concatenate over channel dim
    out = torch.cat(outs_pooled, 1)
    # Flatten
    out = out.view(out.size(0), -1)

    return out