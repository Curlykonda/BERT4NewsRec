import torch
import torch.nn as nn
import pickle
import copy

from pathlib import Path

from source.modules.bert_modules.embedding.token import get_token_embeddings
from source.models.base import NewsRecBaseModel
from source.modules.click_predictor import SimpleDot
from source.modules.news_encoder import NpaCNN, PrecomputedFixedEmbeddings
from source.modules.attention import PersonalisedAttentionWu
from source.modules.preference_query import PrefQueryWu


def make_npa_model(args):
    token_embedding = get_token_embeddings(args) # return 'None' when using fixed BERTje embs

    # article embeddings
    if args.rel_pc_art_emb_path is not None:
        # load pre-computed article embs
        with Path(args.rel_pc_art_emb_path).open('rb') as fin:
            precomputed_art_embs = pickle.load(fin)

        assert precomputed_art_embs.shape[0] == args.num_items, "Mismatch in number of items!"
        assert precomputed_art_embs.shape[1] == args.dim_art_emb, "Mismatch in dimension of article embeddings!"

        news_encoder = PrecomputedFixedEmbeddings(precomputed_art_embs)
    else:
        # get news_encoder from code
        if "wucnn" == args.news_encoder:
            #news_encoder = NpaNewsEncoder(args.n_users, args.dim_art_emb)
            news_encoder = NpaCNN(n_filters=args.dim_art_emb, word_emb_dim=args.dim_word_emb,
                                  dim_pref_q=args.dim_pref_query, dropout_p=args.npa_dropout)
        else:
            raise NotImplementedError(args.news_encoder)


    user_encoder = PersonalisedAttentionWu(args.dim_pref_query, args.dim_art_emb)

    prediction_layer = SimpleDot(args.dim_art_emb, args.dim_art_emb)

    return token_embedding, news_encoder, user_encoder, prediction_layer


class NpaBaseModel(NewsRecBaseModel):

    def __init__(self, args):

        self.args = args
        self.n_users = args.n_users
        self.vocab_len = args.max_vocab_size

        if 'vanilla' == args.npa_variant:
            print('Vanilla NPA setting')
            args.dim_u_id_emb = 50
            args.dim_pref_query = 200

            if args.news_encoder is None:
                args.news_encoder = "wucnn"

            args.dim_art_emb = 400
            args.dim_word_emb = 300

            args.max_hist_len = 50
            args.max_article_len = 30

            args.npa_dropout = 0.2

        elif 'bertje' == args.npa_variant:
            if args.dim_art_emb != 768:
                args.dim_art_emb = 768


        token_embedding, news_encoder, user_encoder, prediction_layer = make_npa_model(args)

        super(NpaBaseModel, self).__init__(token_embedding, news_encoder, user_encoder, prediction_layer, args)

        self.d_u_id_emb = args.dim_u_id_emb
        self.d_pref_q = args.dim_pref_query

        self.d_art_emb = args.dim_art_emb
        self.d_user_emb = args.dim_art_emb

        self.dropout_p = args.npa_dropout

        self.hist_len = args.max_hist_len
        self.art_len = args.max_article_len

        self.user_id_embeddings = nn.Embedding(args.n_users, args.dim_u_id_emb)

        # preference queries
        if isinstance(news_encoder, PrecomputedFixedEmbeddings):
            self.pref_q_word = None
        else:
            self.pref_q_word = PrefQueryWu(args.dim_u_id_emb, self.d_pref_q)

        self.pref_q_article = PrefQueryWu(args.dim_u_id_emb, self.d_pref_q)

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

        brows_hist_reps = self.encode_hist(u_idx, read_hist)  # encode browsing history

        candidate_reps = self.encode_candidates(u_idx, candidates) # encode candidate articles

        user_rep = self.create_user_rep(u_idx, brows_hist_reps) # create user representation

        logits = self.compute_scores(user_rep, candidate_reps)

        # (B x N_c)
        return logits

    def encode_hist(self, u_idx, hist):
        brows_hist_reps = self.encode_news(u_idx, hist) # encode browsing history
        self.brows_hist_reps = brows_hist_reps

        return brows_hist_reps

    def encode_candidates(self, u_idx, cands):
        candidate_reps = self.encode_news(u_idx, cands) # encode candidate articles
        self.candidate_reps = candidate_reps

        return candidate_reps

    def compute_scores(self, user_rep, cand_rep):
        raw_scores = self.prediction_layer(user_rep, cand_rep) # compute raw click score
        self.click_scores = raw_scores

        return raw_scores


    def encode_news(self, u_idx, articles):

        if self.token_embedding is not None:
            # (B x hist_len x L_art) & (vocab_len x emb_dim_word)
            # => (B x L_hist x L_art x D_we)
            emb_news = self.token_embedding(articles) # assert dtype == 'long'

        else:
            emb_news = articles

        if self.pref_q_word is not None:
            # (B x 1) -> (B x D_u) -> (B x D_q)
            pref_query = self.pref_q_word(self.user_id_embeddings(u_idx))

        # -> (B x D_article x L_hist)
        encoded_arts = []
        for x_i in torch.unbind(emb_news, dim=1):
            if self.pref_q_word is not None:
                encoded_arts.append(self.news_encoder(x_i, pref_query))
            else:
                encoded_arts.append(self.news_encoder(x_i))

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


class NpaModModel(NpaBaseModel):

    def __init__(self, args):
        super(NpaModModel, self).__init__(args)

    @classmethod
    def code(cls):
        return 'npa_mod'

    def forward(self, **kwargs):
        u_idx = kwargs['u_idx'][:, 0] # (B x 1)
        read_hist = kwargs['hist'] # (B x L_hist x L_art)
        candidates = kwargs['cands'] # (B x N_c x L_art)
        cand_mask = kwargs['cand_mask']

        brows_hist_reps = self.encode_hist(u_idx, read_hist) # encode browsing history
        # (B x D_A x L_hist)

        candidate_reps = self.encode_candidates(u_idx, candidates, cand_mask) # encode candidate articles
        # (N_T x N_C x D_A)

        user_rep = self.create_user_rep(u_idx, brows_hist_reps) # create user representation

        logits = self.compute_scores(user_rep, candidate_reps, cand_mask)

        # (N_T x N_C)
        return logits


    def encode_candidates(self, u_idx, cands, cand_mask=None):
        # multiple predictions for single history (rather than target-specific hist in VanillaNPA)

        if len(cands.shape) > 3:
            # train case
            B, n_targets, n_cands, art_len = cands.shape #  (B x x N_T x N_c x L_art)

            # filter out relevant candidates (only in train case)
            # select masking positions with provided mask (N_T := number of targets in batch)
            if u_idx is not None:
                rel_u_idx = u_idx.unsqueeze(1).repeat(1, cand_mask.shape[1])[cand_mask != -1]
            else:
                rel_u_idx = None

            try:
                # select candidate subset (N_T x N_c)
                rel_cands = cands[cand_mask != -1]
            except:
                print(cands.shape)
                print(cand_mask.shape)
                print(cands.device)
                print(cand_mask.device)

        elif isinstance(self.news_encoder, PrecomputedFixedEmbeddings) and len(cands.shape) > 2:
            rel_cands = cands[cand_mask != -1]
            rel_u_idx = None
        else:
            # test case
            # (B x N_c x L_art)
            rel_cands = cands
            rel_u_idx = u_idx


        # create article embeddings
        rel_enc_cands = self.encode_news(rel_u_idx, rel_cands)
        # rel_enc_cands = torch.stack([self.encode_news(rel_u_idx[i], x_i) for i, x_i
        #                              in enumerate(torch.unbind(rel_cands, dim=0))], dim=2)
        # (N_T x D_art x N_C)
        return rel_enc_cands


        # return := (B x N_T x N_C x D_A)

    def compute_scores(self, user_rep, cand_rep, cand_mask=None):
        """
        Given the user and candidate representation, compute unnormalised similarity scores
        Since each history could have multiple predictions, we first repeat the required user rep
        to form a matrix

        Input:
        cand_rep := (N_T x D_art x N_C)
        user_rep := (B x D_user)
        cand_mask := (B x L_hist)

        """
        if cand_mask is not None:
            # select and repeat relevant user representations
            # for each user-specific target we need the corresponding user rep to compute scores
            rel_user_rep = user_rep.unsqueeze(1).repeat(1, cand_mask.shape[1], 1)[cand_mask != -1]
            # (N_T x D_user)
        else:
            rel_user_rep = user_rep

        raw_scores = self.prediction_layer(rel_user_rep, cand_rep)  # compute raw click score

        # (N_T x N_C)
        return raw_scores