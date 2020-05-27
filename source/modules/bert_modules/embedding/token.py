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
