import torch.nn as nn
from .token import TokenEmbedding
from .position import LearnablePositionEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_len
        self.embed_size = embed_size

        self.token_emb = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position_emb = LearnablePositionEmbedding(max_len=max_len, d_model=embed_size)
        # self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence):
        x = self.token_emb(sequence) + self.position_emb(sequence)  # + self.segment(segment_label)
        return self.dropout(x)
