import pickle
import torch.nn as nn
from pathlib import Path


from .base import *
from source.modules.bert_modules.bert import BERT
from source.preprocessing.utils_news_prep import get_word_embs_from_pretrained_ft



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


def make_bert4news_model(args):
    token_embedding, news_encoder, user_encoder, prediction_layer, nie_layer = None, None, None, None, None

    vocab_size = args.max_vocab_size  # account for all items including PAD token
    # load pretrained embeddings

    if args.fix_pt_art_emb:
        # fix pre-computed article embs - no need for Word Embs or vocab
        token_embedding = None
    else:
        # compute article embs end-to-end using vocab + Word Embs + News Encoder
        # load vocab
        with Path(args.vocab_path).open('rb') as fin:
            data = pickle.load(fin)
            vocab = data['vocab']

        # load pre-trained Word Embs, if exists
        pt_word_emb = get_word_embs_from_pretrained_ft(vocab, args.pt_word_emb_path, args.dim_word_emb)
        # intialise Token (Word) Embs either with pre-trained or random
        token_embedding = TokenEmbedding(vocab_size, args.dim_word_emb, pt_word_emb)

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
            news_encoder = NpaNewsEncoder(args.n_users, args.dim_art_emb)
        else:
            raise NotImplementedError("Selected invalid News Encoder!")

    # select positional or temporal emb
    pos_emb = None
    if args.pos_embs is not None:
        pos_emb = args.pos_embs
    elif args.temp_embs is not None:
        pos_emb = args.temp_embs
    elif args.pos_embs is not None and args.temp_embs is not None:
        raise ValueError("Can't use positional and temporal embedding! Use one or none")

    # initialise BERT user encoder
    user_encoder = BERT(args, token_emb=None, pos_emb=pos_emb)

    # output layer: return the predicted and candidate embeddings
    if args.pred_layer is None:
        prediction_layer = None
    elif 'l2' == args.pred_layer:
        # L2 distance
        # compute similarities between prediction & candidates
        prediction_layer = SimpleDot(args.dim_art_emb, args.dim_art_emb)
    elif 'cos' == args.pred_layer:
        # cosine distance
        raise NotImplementedError()

    if args.nie_layer is not None:
        # project hidden interest representation to next-item embedding
        nie_layer = nn.Linear(args.bert_hidden_units, args.dim_art_emb)

    return token_embedding, news_encoder, user_encoder, prediction_layer, nie_layer

class BERT4NewsRecModel(NewsRecBaseModel):
    def __init__(self, args):

        token_embedding, news_encoder, user_encoder, prediction_layer, nie_layer = make_bert4news_model(args)

        super().__init__(token_embedding, news_encoder, user_encoder, prediction_layer, args)

        self.nie_layer = nie_layer

        # trainable mask embedding
        self.mask_embedding = torch.randn(args.dim_art_emb, requires_grad=True, device=args.device)
        self.mask_token = args.bert_mask_token
        self.encoded_art = None


    @classmethod
    def code(cls):
        return 'bert4news'

    def forward(self, **kwargs):
        history = kwargs['hist']
        mask = kwargs['mask']
        candidates = kwargs['cands']
        u_ids = kwargs['u_id'][:, 0] if 'u_id' in kwargs else None
        time_stamps = kwargs['ts'] if 'ts' in kwargs else None

        # article encoding
        # (B x L_hist) => (B x L_hist x D_art)
        encoded_hist = self.encode_hist(history, u_ids)
        # encode candidates
        # (B x L_hist x n_candidates) -> (B x L_hist x n_candidates x D_art)
        encoded_cands = self.encode_candidates(candidates, u_ids, mask)


        # interest modeling
        interest_reps = self.create_hidden_interest_representations(encoded_hist, time_stamps, mask)
        # (B x L_hist x D_bert)
        rel_interests = interest_reps[mask == 0]

        # embedding projection
        if self.nie_layer is not None:
            # create next-item embeddings from interest representations
            interest_reps = self.nie_layer(interest_reps)  # (B x L_hist x D_article)

        # score prediction
        if self.prediction_layer is not None:

            # print(rel_interests.shape)
            # print(encoded_cands.shape)
            logits = self.prediction_layer(rel_interests, encoded_cands) # .reshape(-1, encoded_cands.shape[-1], encoded_cands.shape[1])
            # (B x N_c)
            return logits
        else:
            # item embedding case
            return interest_reps, encoded_cands

    def encode_hist(self, article_seq, u_idx=None):

        if self.token_embedding is not None:
            # embedding the indexed sequence to sequence of vectors
            # (B x L_hist x L_article) => (B x L_hist x L_article x D_word_emb)
            embedded_arts = self.token_embedding(article_seq)

            # -> (B x D_article x L_hist)
            encoded_arts = []
            for x_i in torch.unbind(embedded_arts, dim=1):
                encoded_arts.append(self.encode_news(x_i, u_idx))

            encoded_arts = torch.stack(encoded_arts, dim=2)
            # encoded_arts = torch.stack([self.encode_news(x_i, u_idx) for x_i
            #                                 in torch.unbind(embedded_arts, dim=1)], dim=2)

            # (B x L_hist x D_article)
            return encoded_arts.transpose(1, 2)

        else:
            # BERTje case
            # (B x L_hist) -> (B x L_hist x D_article)
            encoded_arts = self.encode_news(article_seq, u_idx)
            return encoded_arts

        # encode word embeddings into article embedding
        # (B x L_hist x L_art x D_word_emb) => (B x L_hist x D_art)
        # for n_news in range(embedded_arts.shape[1]):
        #     #
        #     article_pos = embedded_arts[:, n_news, :, :]
        #
        #     # create article embeddings
        #     encoded_arts.append(self.encode_news(article_pos, u_idx))



    def encode_news(self, article_seq, u_idx=None):

        # (B x L_art) => (B x D_art)
        if u_idx is not None:
            encoded_arts = self.news_encoder(article_seq, u_idx)
        else:
            encoded_arts = self.news_encoder(article_seq)

        # (B x D_article)
        return encoded_arts.squeeze(1)

    def encode_candidates(self, cands, u_idx=None, mask=None):
        # print(cands.shape)

        if self.token_embedding is not None:
            if len(cands.shape) > 3:

                # filter out relevant candidates (only in train case)
                # select masking positions with provided mask (L_M := number of all mask positions in batch)
                if u_idx is not None:
                    rel_u_idx = u_idx.unsqueeze(1).repeat(1, mask.shape[1])[mask == 0]
                else:
                    rel_u_idx = None
                # select candidate subset  (L_M x N_c)
                rel_cands = cands[mask == 0]
            else:
                # test case
                # (B x N_c x L_art)
                rel_cands = cands
                rel_u_idx = u_idx

            # (L x N_c x L_art) => (L x N_c x L_article x D_word_emb)
            emb_cands = self.token_embedding(rel_cands)

            # create article embeddings
            rel_enc_cands = torch.stack([self.encode_news(x_i, rel_u_idx) for x_i
                                in torch.unbind(emb_cands, dim=1)], dim=2)

            return rel_enc_cands

        else:
            # using pre-computed embeddings -> news encoder as a lookup
            if len(cands.shape) > 2:
                # filter out relevant candidates (only in train case)
                # select masking positions with provided mask (L_M := number of all mask positions in batch)
                rel_cands = cands[mask == 0]
            else:
                rel_cands = cands

            encoded_arts = self.news_encoder(rel_cands)
            return encoded_arts.transpose(1, 2)

    def create_hidden_interest_representations(self, encoded_articles, time_stamps, mask):
        # build mask: perhaps by adding up the word ids? -> make efficient for batch
        # mask = (article_seq_as_word_ids != self.mask_token).unsqueeze(1).repeat(1, article_seq_as_word_ids.size(1), 1).unsqueeze(1)
        # mask = (article_seq_as_word_ids[:, :, 0] == self.mask_token) # B x L_hist

        # (B x D_article x L_hist) -> (B x L_hist x D_article)
        art_emb = encoded_articles
        if mask is not None:
            # replace mask positions with mask embedding
            if self.mask_embedding.device != art_emb.device:
                self.mask_embedding = self.mask_embedding.to(art_emb.device)
            art_emb[mask] = self.mask_embedding
            # encoded_articles = encoded_articles.masked_fill(mask==True, self.token_embedding._mask_embedding)
        else:
            raise ValueError("Should apply masking before using BERT ;)")

        # (B x L_hist x D_model) -> (B x L_hist x D_model)
        if time_stamps is not None:
            interest_reps = self.user_encoder([art_emb, time_stamps], mask)
        else:
            interest_reps = self.user_encoder(art_emb, mask)

        return interest_reps