import torch
import torch.nn as nn

from abc import *

from source.modules.attention import PersonalisedAttentionWu
from source.modules.click_predictor import SimpleDot
from source.modules.news_encoder import NpaCNN
from source.modules.preference_query import PrefQueryWu


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, args):
        super().__init__()
        self.args = args

    @classmethod
    @abstractmethod
    def code(cls):
        pass

class NewsRecBaseModel(BaseModel):
    def __init__(self, token_embedding, news_encoder, user_encoder, prediction_layer, args):
        super(NewsRecBaseModel, self).__init__(args)

        #representations
        self.user_rep = None
        self.brows_hist_reps = None
        self.candidate_reps = None
        self.click_scores = None

        self.token_embedding = token_embedding
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder
        self.prediction_layer = prediction_layer

    def forward(self, brows_hist_as_ids, candidates_as_ids):

        brows_hist_reps = self.encode_news(brows_hist_as_ids) # encode browsing history
        self.brows_hist_reps = brows_hist_reps

        candidate_reps = self.encode_news(candidates_as_ids) # encode candidate articles
        self.candidate_reps = candidate_reps

        user_rep = self.create_user_rep(brows_hist_reps) # create user representation

        scores = self.prediction_layer(user_rep, candidate_reps) # compute raw click score
        self.click_scores = scores

        return scores


    def encode_news(self, news_articles_as_ids):

        # (B x hist_len x art_len) -> (vocab_len x emb_dim_word)
        # => (B x hist_len x art_len x emb_dim_word)
        emb_news = self.token_embedding(news_articles_as_ids) # assert dtype == 'long'
        encoded_articles = self.news_encoder(emb_news)

        return encoded_articles

    def create_user_rep(self, user_id, encoded_brows_hist):
        self.user_rep = self.user_encoder(encoded_brows_hist)
        return self.user_rep

    def get_representation_shapes(self):
        shapes = {}
        shapes['user_rep'] = self.user_rep.shape if self.user_rep is not None else None
        shapes['brow_hist'] = self.brows_hist_reps.shape
        shapes['cands'] = self.candidate_reps.shape
        shapes['scores'] = self.click_scores.shape

        for key, shape in shapes.items():
            print("{} shape {}".format(key, shape))

        return

    def _get_pos_emb(self):
        return self.code()