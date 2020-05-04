from torch import nn as nn

from source.modules.bert_modules.embedding import BERTEmbedding
from source.modules.bert_modules.transformer import TransformerBlock
from utils import fix_random_seed_as


class BERT(nn.Module):
    def __init__(self, args, token_emb='new', pos_emb='lpe'):
        super().__init__()

        fix_random_seed_as(args.model_init_seed)
        # self.init_weights()

        max_len = args.bert_max_len
        num_items = args.num_items
        n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
        vocab_size = num_items + 2
        self.hidden = args.bert_hidden_units
        dropout = args.bert_dropout

        # embedding for BERT, sum of positional & token embeddings
        self.token_embedding = token_emb
        self.pos_embedding = pos_emb

        self.embedding = BERTEmbedding(token_emb, pos_emb, vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden, heads, self.hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        # (B x seq_len x d_bert)
        return x

    def init_weights(self):
        pass
