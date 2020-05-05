from .base import *
from source.modules.bert_modules.bert import BERT

import torch.nn as nn

from ..modules.click_predictor import LinLayer
from ..modules.news_encoder import *
from ..modules.bert_modules.embedding.token import *


class BERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.bert = BERT(args)
        self.out = nn.Linear(self.bert.hidden, args.num_items + 1) # + 1 for the mask token

    @classmethod
    def code(cls):
        return 'bert4rec'

    def forward(self, x):
        x = self.bert(x)
        logits = self.out(x) # compute raw scores for all possible items for each position (masked or not)
        return logits


class BERT4NewsRecModel(NewsRecBaseModel):
    def __init__(self, vocab, art_id2word_id, args, pretrained_emb=None):

        #Later: news_encoder = BERT(args, token_emb='new')
        token_embedding = TokenEmbeddingWithMask(len(vocab), args.dim_word_emb, args.bert_hidden_units, pretrained_emb)
        news_encoder = KimCNN(args.bert_hidden_units, args.dim_word_emb)
        user_encoder = BERT(args, token_emb=None)
        interest_extractor = None
        click_predictor = LinLayer(args.bert_hidden_units, args.num_items)

        super().__init__(token_embedding, news_encoder, user_encoder, interest_extractor, click_predictor, args)

    @classmethod
    def code(cls):
        return 'bert4news'


    def encode_news(self, user_id, article_seq_as_art_idx):

        encoded_articles = []

        # build mask: perhaps by adding up the word ids? -> make efficient for batch
        mask = (article_seq_as_art_idx > 0).unsqueeze(1).repeat(1, article_seq_as_art_idx.size(1), 1).unsqueeze(1)

        # encode each browsed news article and concatenate
        for art_pos in range(article_seq_as_art_idx.shape[1]):

            # concatenate word IDs
            article_one = article_seq_as_art_idx[:, art_pos, :, :].squeeze(1)  # shape = (batch_size, title_len, emb_dim)
            # embed word IDs
            embedded_articles = self.art_id2word_embedding(article_one)

            # encode
            encoded_articles.append(self.news_encoder(user_id, embedded_articles))
            assert encoded_articles[-1].shape[1] == self.n_filters  # batch_size X n_cnn_filters

        encoded_articles = torch.stack(encoded_articles, axis=2) # batch_s X dim_news_rep X history_len

        if mask is not None:
            # replace mask positions with mask embedding
            encoded_articles = encoded_articles.masked_fill(mask == 0, self.token_embedding._mask_embedding)

        return encoded_articles

    def art_id2word_embedding(self, art_id):
        # lookup art id -> [word ids]

        # exception: for art id -> [0] * max_len

        embedded_articles = None
        raise NotImplementedError()
        return embedded_articles