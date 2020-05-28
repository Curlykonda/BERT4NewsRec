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


class Bert4NextItemEmbedPrediction(NewsRecBaseModel):
    def __init__(self, args):
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

            # load pre-trained Word Embs, if exist
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
                news_encoder = NewsEncoderWuCNN(n_filters=args.dim_art_emb)
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

        super().__init__(token_embedding, news_encoder, user_encoder, prediction_layer, args)

        if args.nie_layer is not None:
            # project hidden interest representation to next-item embedding
            self.nie_layer = nn.Linear(args.bert_hidden_units, args.dim_art_emb)
        else:
            self.nie_layer = None

        # trainable mask embedding
        self.mask_embedding = torch.randn(args.dim_art_emb, requires_grad=True, device=args.device)
        self.mask_token = args.bert_mask_token
        self.encoded_art = None


    @classmethod
    def code(cls):
        return 'bert4nie'

    def forward(self, history, mask, candidates):

        # hist as article indices, candidates_as_article_indices
        if isinstance(history, list):
            history, time_stamps = history
        else:
            time_stamps = None

        # article encoding
        if self.token_embedding is not None:
            encoded_arts = self.encode_news_w_token(history)
            # encode candidates
            # (B x L_hist x n_candidates) -> (B x L_hist x n_candidates x D_article)
            encoded_cands = self.encode_news_w_token(candidates)
        else:
            encoded_arts = self.encode_news(history) # (B x L_hist x D_article)
            encoded_cands = self.encode_news(candidates)

        # interest modeling
        interest_reps = self.create_hidden_interest_representations(encoded_arts, time_stamps, mask)
        # (B x L_hist x D_bert)

        # embedding projection
        if self.nie_layer is not None:
            # create next-item embeddings from interest representations
            interest_reps = self.nie_layer(interest_reps)  # (B x L_hist x D_article)

        # score prediction
        if self.prediction_layer is not None:
            if len(interest_reps.shape) < len(encoded_cands.shape):
                # (B x L x D_a) x (B x L x N_c x D_a) -> (B x L x N_c)
                # flatten inputs to compute scores
                logits = self.prediction_layer(interest_reps.view(-1, interest_reps.shape[-1]),
                                               encoded_cands.view(-1, encoded_cands.shape[-1], encoded_cands.shape[2]))
                # (B x L x D_a) x (B x L x N_c x D_a) -> ((B*L) x N_c)
            elif len(interest_reps.shape) == len(encoded_cands.shape):
                # test case where we only have candidates for the last position
                # hence, we only need the relevant embedding at the last position

                pred_embs = interest_reps[:, -1, :] # (B x D_a)
                cands = encoded_cands.transpose(1, 2)
                logits = self.prediction_layer(pred_embs, cands) # (B x N_c)
            else:
                raise NotImplementedError()
            return logits
        else:
            return interest_reps, encoded_cands

    def encode_news_w_token(self, article_seq):
        encoded_arts = []
        # embedding the indexed sequence to sequence of vectors
        # (B x L_hist x L_article) => (B x L_hist x L_article x D_word_emb)
        embedded_arts = self.token_embedding(article_seq)

        # encode word embeddings into article embedding
        # (B x L_hist x L_article x D_word_emb) => (B x L_hist x D_article)
        for n_news in range(embedded_arts.shape[1]):
            #
            article_one = embedded_arts[:, n_news, :, :]

            # create article embeddings
            context_art = self.news_encoder(article_one.unsqueeze(1))
            encoded_arts.append(context_art)

        # -> (B x D_article x L_hist)
        encoded_arts = torch.stack(encoded_arts, axis=2)

        # encoded_arts = self.news_encoder(embedded_arts)

        # (B x L_hist x D_article)
        return encoded_arts.transpose(1, 2)

    def encode_news(self, article_seq):
        # retrieve article embedding
        # (B x L_hist) => (B x L_hist x D_article)
        encoded_arts = self.news_encoder(article_seq)

        # (B x L_hist x D_article)
        return encoded_arts

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


class BERT4NewsRecModel(NewsRecBaseModel):
    """

    returns: un-normalised scores over candidate items
    """
    def __init__(self, args):
        # load pretrained embeddings

        vocab_size = args.max_vocab_size # account for all items including PAD token
        #Later: news_encoder = BERT(args, token_emb='new')
        token_embedding = TokenEmbedding(vocab_size, args.dim_word_emb, args.bert_hidden_units, args.pretrained_emb_path)
        #news_encoder = KimCNN(args.bert_hidden_units, args.dim_word_emb)

        # TODO: load precomputed article embeddings from directory and verfiy matrix format (n_articles x dim_art_emb)
        if args.pt_news_encoder is not None:
            precomputed_art_embs = None
            news_encoder = PrecomputedFixedEmbeddings(precomputed_art_embs)
        else:
            # get news_encoder from code
            raise NotImplementedError("Selected invalid News Encoder!")

        user_encoder = BERT(args, token_emb=None) # new module: BERTasUserEncoder

        # output layer: return the predicted and candidate embeddings
        # Trainer will compute similarities between prediction & candidates
        prediction_layer = SimpleDot()


        super().__init__(token_embedding, news_encoder, user_encoder, prediction_layer, args)

        self.encoded_art = None
        self.mask_token = args.bert_mask_token

    @classmethod
    def code(cls):
        return 'bert4news'

    def forward(self, brows_hist_as_article_indices, mask, candidates_as_article_indices):

        encoded_arts = self.encode_news(brows_hist_as_article_indices)
        self.encoded_art = encoded_arts # (B x L_hist x D_article)

        interest_reps = self.create_hidden_interest_representations(encoded_arts, mask)
        # (B x L_hist x D_bert)

        # create next-item embeddings from interest representations
        self.predicted_embs = self.nie_layer(interest_reps) # (B x L_hist x D_article)

        # encode candidates
        # (B x L_hist x n_candidates) -> (B x L_hist x n_candidates x D_article)
        encoded_cands = self.encode_news(candidates_as_article_indices)

        scores = self.prediction_layer(self.predicted_embs, encoded_cands) # (B x L_hist x n_candidates)

        return scores

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
            art_emb[mask] = self.token_embedding._mask_embedding.to(art_emb.device)
            #encoded_articles = encoded_articles.masked_fill(mask==True, self.token_embedding._mask_embedding)
        else:
            raise ValueError("Should apply mask before using BERT ;)")
        # (B x L_hist x D_model) -> (B x L_hist x D_model)
        interest_reps = self.user_encoder(art_emb, mask)

        return interest_reps
