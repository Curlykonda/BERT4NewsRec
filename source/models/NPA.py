import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from source.utils_npa import *
from source.metrics import *

from source.modules.click_predictor import SimpleDot
from source.modules.news_encoder import NewsEncoderWuCNN
from source.modules.attention import PersonalisedAttentionWu
from source.modules.interest_extractor import GRU as GRU_interest


class BaseModelNPA(nn.Module):

    def __init__(self, n_users, vocab_len, pretrained_emb, emb_dim_user_id=50, emb_dim_pref_query=200,
                 emb_dim_words=300, max_news_len=30, max_hist_len=50, n_filters_cnn=400, dropout_p=0.2, device='cpu',
                 news_encoder=None, user_encoder=None, interest_extractor=None, click_predictor=None):
        super(BaseModelNPA, self).__init__()

        self.device = device
        self.vocab_len = vocab_len
        self.max_title_len = max_news_len
        self.max_hist_len = max_hist_len

        self.dim_news_rep = n_filters_cnn
        self.dim_user_rep = n_filters_cnn
        self.dim_emb_user_id = emb_dim_user_id
        self.dim_pref_q = emb_dim_pref_query

        #representations
        self.user_rep = None
        self.brows_hist_reps = None
        self.candidate_reps = None
        self.click_scores = None

        if pretrained_emb is not None:
            #assert pretrained_emb.shape == [vocab_len, emb_dim_words]
            #print("Emb shape is {} and should {}".format(pretrained_emb.shape, (vocab_len, emb_dim_words)))
            self.word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_emb), freeze=False, padding_idx=0)      # word embeddings
        else:
            self.word_embeddings = nn.Embedding(vocab_len, emb_dim_words, padding_idx=0)

        self.user_id_embeddings = nn.Embedding(n_users, self.dim_emb_user_id)

        self.news_encoder = (NewsEncoderWuCNN(n_filters=n_filters_cnn, word_emb_dim=emb_dim_words, dim_pref_q=emb_dim_pref_query, dropout_p=dropout_p)
                             if news_encoder is None else news_encoder)

        # preference queries
        self.pref_q_word = PrefQueryWu(self.dim_pref_q, self.dim_emb_user_id)
        self.pref_q_article = PrefQueryWu(self.dim_pref_q, self.dim_emb_user_id)

        self.interest_extractor = (None if interest_extractor is None
                                   else GRU_interest(self.dim_news_rep, self.dim_user_rep, self.max_hist_len, self.device))

        self.user_encoder = (PersonalisedAttentionWu(emb_dim_pref_query, self.dim_news_rep)
                             if user_encoder is None else user_encoder)

        self.click_predictor = (SimpleDot(self.dim_user_rep, self.dim_news_rep)
                                if click_predictor is None else click_predictor)

    def forward(self, user_id, brows_hist_as_ids, candidates_as_ids):

        brows_hist_reps = self.encode_news(user_id, brows_hist_as_ids) # encode browsing history
        self.brows_hist_reps = brows_hist_reps

        candidate_reps = self.encode_news(user_id, candidates_as_ids) # encode candidate articles
        self.candidate_reps = candidate_reps

        user_rep = self.create_user_rep(user_id, brows_hist_reps) # create user representation

        click_scores = self.click_predictor(user_rep, candidate_reps) # compute raw click score
        self.click_scores = click_scores

        #self.get_representation_shapes()

        return click_scores


    def encode_news(self, user_id, news_articles_as_ids):

        # (B x hist_len x art_len) -> (vocab_len x emb_dim_word)
        # => (B x hist_len x art_len x emb_dim_word)
        emb_news = self.word_embeddings(news_articles_as_ids) # assert dtype == 'long'

        pref_q_word = self.pref_q_word(self.user_id_embeddings(user_id))

        encoded_articles = self.news_encoder(emb_news, pref_q_word)

        return encoded_articles

    def create_user_rep(self, user_id, encoded_brows_hist):

        pref_q_article = self.pref_q_article(self.user_id_embeddings(user_id))

        if self.interest_extractor is not None:
            in_shape = encoded_brows_hist.shape
            encoded_brows_hist = torch.stack(self.interest_extractor(encoded_brows_hist), dim=2)

        self.user_rep = self.user_encoder(encoded_brows_hist, pref_q_article)

        return self.user_rep

    def get_representation_shapes(self):
        shapes = {}
        shapes['user_rep'] = self.user_rep.shape if self.user_rep != None else None
        shapes['brow_hist'] = self.brows_hist_reps.shape
        shapes['cands'] = self.candidate_reps.shape
        shapes['scores'] = self.click_scores.shape

        for key, shape in shapes.items():
            print("{} shape {}".format(key, shape))

        return

class VanillaNPA(BaseModelNPA):

    def __init__(self, n_users, vocab_len, pretrained_emb, emb_dim_user_id=50, emb_dim_pref_query=200,
                 emb_dim_words=300, max_news_len=30, max_hist_len=50, n_filters_cnn=400, dropout_p=0.2, device='cpu'):
        super(VanillaNPA, self).__init__(n_users, vocab_len, pretrained_emb)

        self.device = device
        self.vocab_len = vocab_len
        self.max_title_len = max_news_len
        self.max_hist_len = max_hist_len

        self.dim_news_rep = n_filters_cnn
        self.dim_user_rep = n_filters_cnn
        self.dim_emb_user_id = emb_dim_user_id
        self.dim_pref_q = emb_dim_pref_query

        #representations
        self.user_rep = None
        self.brows_hist_reps = None
        self.candidate_reps = None
        self.click_scores = None

        if pretrained_emb is not None:
            #assert pretrained_emb.shape == [vocab_len, emb_dim_words]
            #print("Emb shape is {} and should {}".format(pretrained_emb.shape, (vocab_len, emb_dim_words)))
            self.word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_emb), freeze=False, padding_idx=0)      # word embeddings
        else:
            self.word_embeddings = nn.Embedding(vocab_len, emb_dim_words)

        self.user_id_embeddings = nn.Embedding(n_users, self.dim_emb_user_id, padding_idx=0)

        self.news_encoder = NewsEncoderWuCNN(n_filters=n_filters_cnn, word_emb_dim=emb_dim_words, dim_pref_q=emb_dim_pref_query, dropout_p=dropout_p)

        # preference queries
        self.pref_q_word = PrefQueryWu(self.dim_pref_q, self.dim_emb_user_id)
        self.pref_q_article = PrefQueryWu(self.dim_pref_q, self.dim_emb_user_id)

        self.interest_extractor = None

        self.user_encoder = PersonalisedAttentionWu(emb_dim_pref_query, self.dim_news_rep)

        self.click_predictor = SimpleDot(self.dim_user_rep, self.dim_news_rep)


    def create_user_rep(self, user_id, encoded_brows_hist):

        pref_q_article = self.pref_q_article(self.user_id_embeddings(user_id))

        self.user_rep = self.user_encoder(encoded_brows_hist, pref_q_article)

        return self.user_rep

    def get_representation_shapes(self):
        shapes = {}
        shapes['user_rep'] = self.user_rep.shape if self.user_rep != None else None
        shapes['brow_hist'] = self.brows_hist_reps.shape
        shapes['cands'] = self.candidate_reps.shape
        shapes['scores'] = self.click_scores.shape

        for key, shape in shapes.items():
            print("{} shape {}".format(key, shape))

        return

class PrefQueryWu(nn.Module):
    '''
    Given an embedded user id, create a preference query vector (that is used in personalised attention)

    '''
    def __init__(self, dim_pref_query=200, dim_emb_u_id=50, activation='relu', device='cpu'):
        super(PrefQueryWu, self).__init__()

        self.dim_pref_query = dim_pref_query
        self.dim_u_id = dim_emb_u_id

        self.lin_proj = nn.Linear(self.dim_u_id, self.dim_pref_query)

        assert activation in ['relu', 'tanh']

        if activation == 'relu':
            self.activation = nn.Tanh()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise KeyError()

    def forward(self, u_id):
        #print(u_id.shape)
        pref_query = self.lin_proj(u_id) # batch_size X u_id_emb_dim

        return self.activation(pref_query)