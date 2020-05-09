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
    def __init__(self, args):
        # load pretrained embeddings

        vocab_size = args.max_vocab_size # account for all items including PAD token
        #Later: news_encoder = BERT(args, token_emb='new')
        token_embedding = TokenEmbeddingWithMask(vocab_size, args.dim_word_emb, args.bert_hidden_units, args.pretrained_emb_path)
        news_encoder = KimCNN(args.bert_hidden_units, args.dim_word_emb)
        user_encoder = BERT(args, token_emb=None) # new module: BERTasUserEncoder
        interest_extractor = None
        click_predictor = LinLayer(args.bert_hidden_units, args.num_items)

        super().__init__(token_embedding, news_encoder, user_encoder, interest_extractor, click_predictor, args)

        self.encoded_art = None
        self.mask_token = args.bert_mask_token
        self.mask_val_we = 0.0

    @classmethod
    def code(cls):
        return 'bert4news'

    def forward(self, brows_hist_as_word_ids, mask):

        encoded_arts = self.encode_news(brows_hist_as_word_ids)
        self.encoded_art = encoded_arts

        interest_reps = self.create_hidden_interest_representations(encoded_arts, mask)

        click_scores = self.click_predictor(interest_reps) # compute raw click score
        self.click_scores = click_scores

        return click_scores

    def encode_news(self, article_seq_as_word_ids):

        encoded_arts = []

        # embedding the indexed sequence to sequence of vectors
        # (B x L_hist x L_article) => (B x L_hist x L_article x D_word_emb)
        embedded_arts = self.token_embedding(article_seq_as_word_ids)

        # encode word embeddings into article embedding
        # (B x L_hist x L_article x D_word_emb) => (B x L_hist x D_article)
        for n_news in range(embedded_arts.shape[1]):

            # concatenate words
            article_one = embedded_arts[:, n_news, :, :]

            context_art = self.news_encoder(article_one.unsqueeze(1))
            encoded_arts.append(context_art)

        # -> (B x D_article x L_hist)
        encoded_arts = torch.stack(encoded_arts, axis=2)

        #encoded_arts = self.news_encoder(embedded_arts)


        # (B x L_hist x D_article)
        return encoded_arts

    def create_hidden_interest_representations(self, encoded_articles, mask):
        # build mask: perhaps by adding up the word ids? -> make efficient for batch
        # mask = (article_seq_as_word_ids != self.mask_token).unsqueeze(1).repeat(1, article_seq_as_word_ids.size(1), 1).unsqueeze(1)
        # mask = (article_seq_as_word_ids[:, :, 0] == self.mask_token) # B x L_hist

        # (B x D_article x L_hist) -> (B x L_hist x D_article)
        art_emb = encoded_articles.transpose(1,2)
        if mask is not None:
            # replace mask positions with mask embedding
            art_emb[mask] = self.token_embedding._mask_embedding
            #encoded_articles = encoded_articles.masked_fill(mask==True, self.token_embedding._mask_embedding)
        else:
            raise ValueError("Should apply mask before using BERT ;)")

        interest_reps = self.user_encoder(art_emb, mask)

        return interest_reps
