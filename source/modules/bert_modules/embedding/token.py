import pickle
from pathlib import Path

import fasttext
import numpy as np
import torch
import torch.nn as nn


# class TokenEmbedding(nn.Embedding):
#     def __init__(self, vocab_size, embed_size=512):
#         super().__init__(vocab_size, embed_size, padding_idx=0)
#
#     @staticmethod
#     def code():
#         return 'new'

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, token_embed_size=300, pretrained=None):
        super(TokenEmbedding, self).__init__()

        if pretrained is None:
            self.token_embedding = nn.Embedding(vocab_size, token_embed_size, padding_idx=0)
            self.from_pt = False
        else:
            # Add position for mask token to pretrained
            self.token_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained), freeze=False, padding_idx=0)
            self.from_pt = True

    def code(self):
        if self.from_pt:
            return 'pt_emb'
        else:
            return 'new_emb'

    def forward(self, token_ids):
        #map the token ids to embedding vectors
        return self.token_embedding(token_ids)


def get_token_embeddings(args):

    vocab_size = args.max_vocab_size  # account for all items including PAD token
    #vocab_size = 17023

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

    return token_embedding


def get_word_embs_from_pretrained_ft(vocab, emb_path, emb_dim=300):
    if emb_path is None:
        print("No path to pretrained word embeddings given")
        return None

    try:
        ft = fasttext.load_model(emb_path) # load pretrained vectors

        # check & adjust dimensionality
        if ft.get_dimension() != emb_dim:
            fasttext.util.reduce_model(ft, emb_dim)

        embedding_matrix = [0] * len(vocab)
        embedding_matrix[0] = np.zeros(emb_dim, dtype='float32')  # placeholder with zero values for 'PAD'

        for word, idx in vocab.items():
            embedding_matrix[idx] = ft[word] # how to deal with unknown words?

        return np.array(embedding_matrix, dtype='float32')

    except:
        print("Could not load word embeddings")
        return None