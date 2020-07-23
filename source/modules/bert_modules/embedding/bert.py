import torch.nn as nn
from .token import *
from .position import *
from source.modules.temporal_embedding import *
from .position import POS_EMBS

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
        self.pos_emb_size = pos_emb_size if pos_emb_size is not None else 0
        self.comb_func = args.add_embs_func # how to combine pos & token emb

        self.output_size = self.tkn_emb_size

        if 'add' == self.comb_func:
            if self.pos_emb_size > 0:
                assert self.pos_emb_size == self.tkn_emb_size
        elif 'concat' == self.comb_func:
            self.output_size = self.tkn_emb_size + self.pos_emb_size
        elif self.comb_func is not None:
            raise ValueError(self.comb_func)

        self.temp_embs_hidden_units = args.temp_embs_hidden_units
        self.temp_embs_act_func = args.temp_embs_act_func
        self.len_time_vec = args.len_time_vec

        self.token_code = token_code
        self.pos_code = pos_code

        self.token_emb = self._get_token_emb()
        self.position_emb = self._get_pos_emb()

        if args.norm_art_pos_embs:
            self.layer_norm = nn.LayerNorm(self.tkn_emb_size + self.pos_emb_size)
        else:
            self.layer_norm = None

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, to_emb):

        if isinstance(to_emb, list):
            seq, ts = to_emb
            # seq: (B x L x D_a)
            # ts: (B x L)
        else:
            seq = ts = to_emb

        # * math.sqrt(self.tkn_emb_size)
        tkn = self.token_emb(seq) if self.token_emb is not None else seq
        if self.position_emb is not None:
            pos = self.position_emb(ts)
        else:
            pos = None

        # normalise tkn & pos embs
        if pos is not None:
            tkn_pos = torch.cat([tkn, pos], dim=2)
        else:
            tkn_pos = tkn

        if self.layer_norm is not None:
            tkn_pos = self.layer_norm(tkn_pos)

        if 'add' == self.comb_func:
            if pos is not None:
                out = tkn_pos[:, :, :self.tkn_emb_size] + tkn_pos[:, :, self.tkn_emb_size:]
            else:
                out = tkn_pos
            assert out.shape[-1] == self.tkn_emb_size

        else: #'concat' == self.comb_func:
            # 'concat' & None case
            out = tkn_pos
            assert out.shape[-1] == self.output_size


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
        elif 'gnoise' == self.pos_code:
            return GaussNoiseEmb(self.pos_emb_size)
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