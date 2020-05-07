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

        #Later: news_encoder = BERT(args, token_emb='new')
        token_embedding = TokenEmbeddingWithMask(args.max_vocab_size, args.dim_word_emb, args.bert_hidden_units, args.pretrained_emb_path)
        news_encoder = KimCNN(args.bert_hidden_units, args.dim_word_emb)
        user_encoder = BERT(args, token_emb=None)
        interest_extractor = None
        click_predictor = LinLayer(args.bert_hidden_units, args.num_items)

        super().__init__(token_embedding, news_encoder, user_encoder, interest_extractor, click_predictor, args)

        self.encoded_art = None
        self.mask_token = args.bert_mask_token
        self.mask_val_we = 0.0

    @classmethod
    def code(cls):
        return 'bert4news'

    def forward(self, brows_hist_as_word_ids):

        encoded_arts = self.encode_news(brows_hist_as_word_ids)
        self.encoded_art = encoded_arts

        bert_rep = self.user_encoder(encoded_arts) # create user representation

        click_scores = self.click_predictor(bert_rep) # compute raw click score
        self.click_scores = click_scores

        return click_scores

    def encode_news(self, article_seq_as_word_ids):

        encoded_arts = []

        # build mask: perhaps by adding up the word ids? -> make efficient for batch
        mask = (article_seq_as_word_ids != self.mask_token).unsqueeze(1).repeat(1, article_seq_as_word_ids.size(1), 1).unsqueeze(1)
        """
        mask = (article_seq_as_word_ids.len == 1) 
        -> at masked position we only use the Mask token id 
        while all other position are a seq of word ids
        
        given this mask, we then either apply the News Encoder or directly put the Mask Embedding
        
        """
        # encode each browsed news article and concatenate
        for art_pos in range(article_seq_as_word_ids.shape[1]):

            # concatenate word IDs
            article_one = article_seq_as_word_ids[:, art_pos, :, :].squeeze(1)  # shape = (batch_size, title_len, emb_dim)
            # embed word IDs
            embedded_arts = self.art_id2word_embedding(article_one)

            if mask is not None:
                embedded_arts = embedded_arts.masked_fill(mask == 0, self.mask_val_we)

            # encode
            encoded_arts.append(self.news_encoder(embedded_arts))
            assert encoded_arts[-1].shape[1] == self.n_filters  # batch_size X n_cnn_filters

        encoded_arts = torch.stack(encoded_arts, axis=2) # batch_s X dim_news_rep X history_len

        if mask is not None:
            # replace mask positions with mask embedding
            encoded_arts = encoded_arts.masked_fill(mask == 0, self.token_embedding._mask_embedding)

        return encoded_arts

    def art_id2word_embedding(self, art_id):
        # lookup art id -> [word ids]

        # exception: for art id -> [0] * max_len

        embedded_articles = None
        raise NotImplementedError()
        return embedded_articles