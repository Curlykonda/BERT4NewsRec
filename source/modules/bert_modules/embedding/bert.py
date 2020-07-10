import torch.nn as nn
from .token import *
from .position import *
from source.modules.temporal_embedding import *


POS_EMBS = [
    TrigonometricPositionEmbedding.code(),
    LearnablePositionEmbedding.code(),
    None
]

#TOKEN_EMBS = [TokenEmbedding.code(), 'pt', None]

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, args, token_code, pos_code, vocab_size, tkn_emb_size, max_len, dropout=0.1, pos_emb_size=None, pretrained_tokens=None):
        """
        :param vocab_size: total vocab size
        :param tkn_emb_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.max_seq_len = max_len
        self.tkn_emb_size = tkn_emb_size
        self.pos_emb_size = pos_emb_size if pos_emb_size is not None else tkn_emb_size

        self.comb_func = args.add_embs_func # how to combine pos & token emb
        if 'add' == self.comb_func:
            assert self.pos_emb_size == self.tkn_emb_size
            self.output_size = self.tkn_emb_size
        elif 'concat' == self.comb_func:
            self.output_size = self.tkn_emb_size + self.pos_emb_size
        else:
            raise ValueError(args.add_embs_func)

        self.temp_embs_hidden_units = args.temp_embs_hidden_units
        self.temp_embs_act_func = args.temp_embs_act_func
        self.len_time_vec = args.len_time_vec

        self.token_code = token_code
        self.pos_code = pos_code

        self.token_emb = self._get_token_emb()
        self.position_emb = self._get_pos_emb()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, to_emb):
        # sum token & positional embedding
        if isinstance(to_emb, list):
            seq, ts = to_emb
            # seq: (B x L x D_a)
            # ts: (B x L)
        else:
            seq = ts = to_emb

        tkn = self.token_emb(seq) * math.sqrt(self.tkn_emb_size) if self.token_emb is not None else seq
        pos = self.position_emb(ts) if self.position_emb is not None else ts

        if 'add' == self.comb_func:
            out = tkn + pos
        elif 'concat' == self.comb_func:
            out = torch.cat([tkn, pos], dim=2)
        else:
            raise NotImplementedError()

        return self.dropout(out)

    def _get_token_emb(self):
        # if self.token_code not in TOKEN_EMBS:
        #     raise KeyError("Unknown Token Embedding")

        if 'new' == self.token_code:
            return TokenEmbedding(vocab_size=self.vocab_size, token_embed_size=self.tkn_emb_size)
        elif 'pt' == self.token_code:
            return get_token_embeddings(self.args)
        else:
            return None

    def _get_pos_emb(self):
        # valid code?
        if self.pos_code is not None \
            and self.pos_code not in POS_EMBS and self.pos_code not in TEMP_EMBS:
                raise KeyError("{} is unknown Positional/Temporal Embedding".format(self.pos_code))

        if self.pos_code is not None:
            self.pos_code = self.pos_code.lower()

        # pos embs
        if 'tpe' == self.pos_code:
            return TrigonometricPositionEmbedding(d_model=self.pos_emb_size, max_len=self.max_seq_len)
        elif 'lpe' == self.pos_code:
            return LearnablePositionEmbedding(self.pos_emb_size, self.max_seq_len)
        # temp embs
        elif 'lte' == self.pos_code:
            temp_emb = TEMP_EMBS[self.pos_code]
            return temp_emb(self.len_time_vec, self.pos_emb_size)
        elif 'nte' == self.pos_code:
            temp_emb = TEMP_EMBS[self.pos_code]
            return temp_emb(self.len_time_vec, self.pos_emb_size, self.temp_embs_hidden_units, self.temp_embs_act_func)
        else:
            return None



class BERTEmbeddingOrg(nn.Module):
    """
    Original BERT Embedding from BERT4Rec
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos

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