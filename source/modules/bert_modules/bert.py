from torch import nn as nn

from source.modules.bert_modules.embedding import BERTEmbedding
from source.modules.bert_modules.transformer import TransformerBlock
from utils import fix_random_seed_as


class BERT(nn.Module):
    def __init__(self, args, token_emb='new', pos_emb='lpe'):
        super().__init__()

        #fix_random_seed_as(args.model_init_seed)
        # self.init_weights()

        max_len = args.max_hist_len
        num_items = args.num_items
        n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
        vocab_size = num_items + 2
        self.hidden = args.bert_hidden_units
        dropout = args.bert_dropout

        # embedding for BERT, sum of positional & token embeddings
        self.token_embedding = token_emb
        self.pos_embedding = pos_emb

        self.embedding = BERTEmbedding(args, token_emb, pos_emb, vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden, heads, self.hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, mask=None):

        if mask is None:
            if isinstance(x, list):
                mask = (x[0] > 0).unsqueeze(1).repeat(1, x[0].size(1), 1).unsqueeze(1)
            else:
                mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        else:
            # check dimesions of mask to match requirements for self-attention
            # in case of News, input mask is of shape (B x L_hist)
            # -> (B x 1 x L_hist x D_model)
            mask = mask.unsqueeze(1).repeat(1, mask.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        # combine token & positional embeddings
        #print(x[0].norm(2))
        x = self.embedding(x)
        #print(x.norm(2))
        #x = x_emb

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        # (B x seq_len x d_bert)
        #print(x.norm(2))
        return x