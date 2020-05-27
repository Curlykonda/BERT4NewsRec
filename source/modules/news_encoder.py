import torch
import torch.nn as nn
import transformers

from source.modules.attention import PersonalisedAttentionWu


class BERTje(transformers.BertModel):

    @classmethod
    def code(cls):
        return "bertje"

class RandomEmbedding(nn.Module):

    def __init__(self, dim_art_emb):
        super(RandomEmbedding, self).__init__()

        self.emb_dim = dim_art_emb

    @classmethod
    def code(cls):
        return "rnd"

    def forward(self, tokens):
        return torch.randn(self.emb_dim)

    def _get_emb(self):
        return torch.randn(self.emb_dim)

class PrecomputedFixedEmbeddings(nn.Module):
    """
    precomputed_art_embs : embedding matrix containing the pre-computed article embedding
    """
    def __init__(self, precomputed_art_embs, freeze=True):

        super(PrecomputedFixedEmbeddings, self).__init__()
        self.freeze = freeze
        if precomputed_art_embs is not None:
            self.article_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(precomputed_art_embs), freeze=freeze)
        else:
            raise ValueError("Need to pass valid article embeddings!")

    def forward(self, article_indices):
        # lookup embedding
        return self.article_embeddings(article_indices)


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

    @classmethod
    def code(cls):
        return "npa_cnn"

# Utility function to calc output size
def output_size(in_size, kernel_size, stride, padding):
  output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
  return output//2

def mot_pooling(x):
  # Max-over-time pooling
  # X is conv output
  # (B x n_filters x L_out)
  # Note that L_out depends on kernel and L_in
  #print(x.shape)
  return nn.MaxPool1d(kernel_size=x.shape[2])(x)



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

    self.proj_out = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.ReLU(),
        nn.Linear(n_filters * len(kernels), n_filters))


    #self.mot_pooling = torch.nn.MaxPool1d(kernel_size=n_filters)

  def forward(self, x):
    # Pass through each conv layer
    outs = [conv(x) for conv in self.convs]

    # Max over time pooling
    outs_pooled = [mot_pooling(out.squeeze()) for out in outs]
    # Concatenate over channel dim
    out = torch.cat(outs_pooled, 1)
    # Flatten
    # (B x (n_kernels * n_filters))
    out = out.view(out.size(0), -1)

    # Dropout & Project
    out = self.proj_out(out)
    # (B x n_filters)
    assert out.shape[1] == self.n_filters

    return out

  @classmethod
  def code(cls):
      return "kim_cnn"