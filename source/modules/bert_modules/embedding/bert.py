import torch.nn as nn
from .token import *
from .position import *

POS_EMBS = [
    TrigonometricPositionEmbedding.code(),
    LearnablePositionEmbedding.code(),
    None
]

TOKEN_EMBS = [TokenEmbedding.code(), 'pt', None]

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, token_code, pos_code, vocab_size, embed_size, max_len, dropout=0.1, pretrained_tokens=None):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_len
        self.embed_size = embed_size

        self.token_code = token_code
        self.pos_code = pos_code

        self.token_emb = self._get_token_emb()
        self.position_emb = self._get_pos_emb()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, seq):
        tkn = self.token_emb(seq) if self.token_emb is not None else seq
        pos = self.position_emb(seq) if self.position_emb is not None else seq

        return self.dropout(tkn + pos)

    def _get_token_emb(self):
        if self.token_code not in TOKEN_EMBS:
            raise KeyError("Unknown Token Embedding")

        if 'new_emb' == self.token_code:
            return TokenEmbedding(vocab_size=self.vocab_size, embed_size=self.embed_size)
        elif 'pt_emb' == self.token_code:
            # TODO: load pretrained embeddings
            return None
        else:
            return None

    def _get_pos_emb(self):
        if self.pos_code not in POS_EMBS:
            raise KeyError("Unknown Positional Embedding")

        if 'tpe' == self.pos_code:
            return TrigonometricPositionEmbedding(d_model=self.embed_size, max_len=self.max_seq_len)
        elif 'lpe' == self.pos_code:
            return LearnablePositionEmbedding(self.embed_size, self.max_seq_len)
        else:
            return None