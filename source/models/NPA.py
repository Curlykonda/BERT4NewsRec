import torch
import torch.nn as nn
import pickle
import copy

from pathlib import Path

from source.modules.bert_modules.embedding.token import get_token_embeddings
from source.models.base import NewsRecBaseModel
from source.modules.click_predictor import SimpleDot
from source.modules.news_encoder import NpaCNN
from source.modules.attention import PersonalisedAttentionWu
from source.modules.preference_query import PrefQueryWu


class NpaBaseModel(NewsRecBaseModel):

    def __init__(self, args):

        self.args = args
        self.n_users = args.n_users
        self.vocab_len = args.max_vocab_size

        self.d_u_id_emb = args.dim_u_id_emb
        self.d_pref_q = args.dim_pref_query

        self.d_art_emb = args.dim_art_emb
        self.d_user_emb = args.dim_art_emb

        self.dropout_p = args.npa_dropout

        # if token_embedding is not None:
        #     # embedding matrix is passed
        #     # assert corect shape
        #     token_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(token_embedding), freeze=False, padding_idx=0)
        # else:
        # if available, load pre-trained token embeddings
        token_embedding = get_token_embeddings(args)

        news_encoder = NpaCNN(n_filters=args.dim_art_emb, word_emb_dim=args.dim_word_emb,
                                    dim_pref_q=args.dim_pref_query, dropout_p=args.npa_dropout)

        user_encoder = PersonalisedAttentionWu(self.d_pref_q, args.dim_art_emb)

        prediction_layer = SimpleDot(args.dim_art_emb, args.dim_art_emb)

        super(NpaBaseModel, self).__init__(token_embedding, news_encoder, user_encoder, prediction_layer, args)

        self.user_id_embeddings = nn.Embedding(args.n_users, args.dim_u_id_emb)

        # preference queries
        self.pref_q_word = PrefQueryWu(args.dim_u_id_emb, self.d_pref_q, )
        self.pref_q_article = PrefQueryWu(args.dim_u_id_emb, self.d_pref_q, )

        #representations
        self.user_rep = None
        self.brows_hist_reps = None
        self.candidate_reps = None
        self.click_scores = None

    @classmethod
    def code(cls):
        return 'npa'

    def forward(self, **kwargs):
        """
        Descr:
            Encode articles from reading history. Articles are usually represented as sequenes of word IDs
            Encode candidates articles
            Build user representation from encoded reading history
            Compute (raw) similarity scores between user representation and candidates

        Input:
         - user_index: (B x 1)
         - brows_hist_as_ids: (B x L_hist x L_art)
         - candidates_as_ids: (B x N_c x L_art)

        Output:
         raw_scores (B x N_c): unnormalised similarity scores for corresponding user-candidate pairs

        """

        u_idx = kwargs['u_idx'][:, 0] # (B x 1)
        read_hist = kwargs['hist'] # (B x L_hist x L_art)
        candidates = kwargs['cands'] # (B x N_c x L_art)

        brows_hist_reps = self.encode_news(u_idx, read_hist) # encode browsing history
        self.brows_hist_reps = brows_hist_reps

        candidate_reps = self.encode_news(u_idx, candidates) # encode candidate articles
        self.candidate_reps = candidate_reps

        user_rep = self.create_user_rep(u_idx, brows_hist_reps) # create user representation

        raw_scores = self.prediction_layer(user_rep, candidate_reps) # compute raw click score
        self.click_scores = raw_scores

        # (B x N_c)
        return raw_scores

    def encode_hist(self, u_idx, hist):
        pass

    def encode_candidates(self, u_idx, cands):
        pass

    def encode_news(self, u_idx, articles):

        # (B x hist_len x art_len) -> (vocab_len x emb_dim_word)
        # => (B x L_hist x L_art x D_we)
        emb_news = self.token_embedding(articles) # assert dtype == 'long'

        # (B x 1) -> (B x D_u) -> (B x D_q)
        pref_query = self.pref_q_word(self.user_id_embeddings(u_idx))

        # -> (B x D_article x L_hist)
        encoded_arts = []
        for x_i in torch.unbind(emb_news, dim=1):
            encoded_arts.append(self.news_encoder(x_i, pref_query))

        encoded_arts = torch.stack(encoded_arts, dim=2)

        # -> (B x D_art x L_hist)
        return encoded_arts

    def create_user_rep(self, user_id, encoded_brows_hist):

        pref_query = self.pref_q_article(self.user_id_embeddings(user_id))

        # (B x D_art x L_hist) & (B x D_q) -> (B x D_art)
        self.user_rep = self.user_encoder(encoded_brows_hist, pref_query)

        return self.user_rep

class VanillaNPA(NpaBaseModel):

    def __init__(self, args):

        vanilla_args = copy.deepcopy(args)

        vanilla_args.dim_u_id_emb = 50
        vanilla_args.dim_pref_query = 200

        vanilla_args.dim_art_emb = 400
        vanilla_args.dim_word_emb = 300

        vanilla_args.max_hist_len = 50
        vanilla_args.max_art_len = 30

        vanilla_args.npa_dropout = 0.2

        super(VanillaNPA, self).__init__(vanilla_args)


    @classmethod
    def code(cls):
        return 'vanilla_npa'