from .base import *
from source.modules.bert_modules.bert import BERT

import torch.nn as nn

from ..modules.click_predictor import LinLayer
from ..modules.news_encoder import *


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

        #End Goal: news_encoder = BERT(args, token_emb='new')
        news_encoder = KimCNN(args.bert_hidden_units, args.dim_word_emb)
        user_encoder = BERT(args, token_emb=None)
        interest_extractor = None
        click_predictor = LinLayer(args.bert_hidden_units, args.num_items)

        super().__init__(news_encoder, user_encoder, interest_extractor, click_predictor, args)

    @classmethod
    def code(cls):
        return 'bert4news'