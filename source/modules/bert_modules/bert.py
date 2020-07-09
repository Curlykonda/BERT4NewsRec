from torch import nn as nn

from source.modules.bert_modules.embedding import BERTEmbedding
from source.modules.bert_modules.transformer import TransformerBlock
from utils import fix_random_seed_as


class BERT(nn.Module):
    def __init__(self, args, max_len=None, n_items=None, n_layers=2, n_heads=4, n_hidden=300, dropout=0.1, req_mask=True, token_emb='new', pos_emb='lpe'):
        super().__init__()

        #fix_random_seed_as(args.model_init_seed)
        # self.init_weights()

        self.max_len = max_len if max_len is not None else args.max_hist_len
        num_items = n_items if n_items is not None else args.num_items
        self.n_layers = n_layers # args.bert_num_blocks
        self.n_heads = n_heads # args.bert_num_heads
        vocab_size = num_items + 2
        n_hidden = n_hidden if n_hidden is not None else args.bert_hidden_units
        self.dropout = dropout # args.bert_dropout
        self.req_mask = req_mask

        # embedding for BERT, sum of positional & token embeddings
        self.token_embedding = token_emb
        self.pos_embedding = pos_emb

        self.embedding = BERTEmbedding(args, token_emb, pos_emb, vocab_size=vocab_size,
                                       tkn_emb_size=n_hidden, pos_emb_size=args.add_emb_size,
                                       max_len=self.max_len, dropout=self.dropout)
        self.n_hidden = self.embedding.output_size

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.n_hidden, self.n_heads, self.n_hidden * 4, self.dropout) for _ in range(n_layers)])

    def forward(self, x, mask=None):

        if mask is None and self.req_mask:
            if isinstance(x, list):
                mask = (x[0] > 0).unsqueeze(1).repeat(1, x[0].size(1), 1).unsqueeze(1)
            else:
                mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        elif mask is not None:
            # check dimensions of mask to match requirements for self-attention
            # in case of News, input mask is of shape (B x L_hist)
            # -> (B x 1 x L_hist x D_model)
            mask = mask.unsqueeze(1).repeat(1, mask.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        # combine token & positional embeddings
        x = self.embedding(x)
        # (B x L x D_model)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        # (B x seq_len x d_bert)
        #print(x.norm(2))
        return x