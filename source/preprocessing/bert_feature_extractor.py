
from collections import defaultdict
from nltk.tokenize import sent_tokenize

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel


def get_dummy_bert_output(self, batch_size, dim_bert=64, seq_len=20, n_layers=12, hidden_outs=True, attn_weights=True):
    bert_outs = []
    # last_hidden, pooled_out, hidden_outs, attention_weights = bert_output
    # 1. last_hidden
    last_hidden = torch.randn([batch_size, seq_len, dim_bert])
    bert_outs.append(last_hidden)  # batch_size x seq_len x dim_bert

    # 2. pooled_out
    bert_outs.append(torch.randn([batch_size, dim_bert]))

    # 3. hidden_outs
    if hidden_outs:
        h_outs = [torch.randn([batch_size, seq_len, dim_bert]) for n in range(n_layers)]
        h_outs.append(last_hidden)
        bert_outs.append(h_outs)

    # 4. attn_weights
    if attn_weights:
        a_weights = [torch.randn([batch_size, seq_len, dim_bert]) for n in range(n_layers)]
        bert_outs.append(a_weights)

    return bert_outs


class BertFeatureExtractor():

    def __init__(self, bert_dir_path, **kwargs):
        super().__init__(**kwargs)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.batch_size = batch_size
        # self.max_seq_len = max_seq_len

        self.dev_mode = False
        self.bert_emd_dim = 768

        self.sent_tokenizer = sent_tokenize
        self.bert_tokenizer = BertTokenizer._from_pretrained(bert_dir_path)
        self.bert_tokenizer.add_special_tokens({"unk_token": '[UNK]', 'cls_token': '[CLS]',
                                                'pad_token': '[PAD]', 'sep_token': '[SEP]'})

        self.bert_model = BertModel.from_pretrained(bert_dir_path, output_hidden_states=True)
        self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))
        self.bert_model.to(self.device)

        self._feat_methods = ['pooled_out', 'last_cls', 'pool_all_last', 'pool_cls_n', 'pool_last_n', 'sum_last_n',
                              'sum_all']
        self._naive_feat_methods = ['naive_mean', 'naive_sum']

    @property
    def _get_feat_methods(self):
        return self._feat_methods

    @property
    def _get_naive_feat_methods(self):
        return self._naive_feat_methods

    def __call__(self, *args, **kwargs):
        return self.bert_model(*args, **kwargs)

    def get_naive_seq_emb(self, tokens, method='naive_mean'):

        if method not in self._get_naive_feat_methods():
            raise KeyError("'{}' is not a valid method!".format(method))

        word_embeddings = self.bert_model.get_input_embeddings()
        word_embeddings.to(self.device)
        tokens_emb = word_embeddings(torch.tensor(tokens, device=self.device)).to(self.device)
        if method == 'naive_mean':
            emb_out = torch.mean(tokens_emb, dim=1)
        elif method == 'naive_sum':
            emb_out = torch.sum(tokens_emb, dim=1)
        else:
            raise NotImplementedError()

        return emb_out

    def get_bert_word_embeddings(self):
        return self.bert_model.get_input_embeddings().to(self.device)

    def extract_bert_features(self, bert_output, method, n_layers):
        """

        :param bert_output:
        :param method:
        :param n_layers:
        :return: torch tensor containing the extracted features of shape batch_size X emb_dim (768)
        """

        assert len(bert_output) == 3

        if method not in self._get_feat_methods:
            raise KeyError("'{}' is not a valid method! \n Choose from {}".format(method, self._get_feat_methods))

        last_hidden, pooled_out, hidden_outs = bert_output

        if 'pooled_out' == method:
            x_out = pooled_out  # batch_size x dim_emb
        elif 'last_cls' == method:
            # take the embedding of CLS token of last hidden layer
            x_out = last_hidden[:, 0, :]  # batch_size x dim_emb
        elif 'pool_all_last' == method:
            # average embeddings of last hidden layer of all tokens
            x_out = torch.mean(last_hidden, dim=1)
        elif 'pool_cls_n' == method and n_layers:
            x_out = torch.mean(torch.cat([hidden[:, 0, :].unsqueeze(1) for hidden in hidden_outs[-n_layers:]], dim=1),
                               dim=1)
        elif 'pool_last_n' == method and n_layers:
            # average embeddings of last N hidden modules of all tokens
            x_out = torch.mean(torch.cat(hidden_outs[-n_layers:], dim=1), dim=1)
        elif 'sum_last_n' == method and n_layers:
            # sum embeddings of last N hidden modules of all tokens
            x_out = torch.sum(torch.cat(hidden_outs[-n_layers:], dim=1), dim=1)
            # sum last four hidden => 95.9 F1 on dev set for NER
        elif 'sum_all' == method:
            x_out = torch.sum(torch.cat(hidden_outs, dim=1), dim=1)
        else:
            raise NotImplementedError()

        # print(method)
        # print(x_out.shape)
        return x_out

    def test_feature_extraction(self, n_items, methods, batch_size, emb_dim, seq_len, **kwargs):

        self.dev_mode = True
        #save default properties



        dummy_content = [0] * n_items
        self.embedding_dim_cur = emb_dim
        self.batch_size_cur = batch_size

        encoded_text, n_words = self.encode_text_to_features_batches(dummy_content, methods, max_seq_len=seq_len)

        #restore default properties
        self.dev_mode = False

        return encoded_text

    def encode_text_to_features_batches(self, content: dict, methods: list, batch_size: int, max_seq_len: int,
                                        add_special_tokens=True, lower_case=True):
        self.max_seq_len = max_seq_len
        seq_embeddings = defaultdict(dict) # {method: {article ID -> embedding vector}}

        start_idx = 0
        stop_idx = batch_size

        n_items = len(content)
        #idx2key = {i: key for i, key in enumerate(content.keys())}
        item_keys = list(content.keys())
        slice_idx = list(range(0, batch_size))

        #
        while (start_idx < n_items):

            # handling edge cases
            if stop_idx > n_items:
                # indices and slice range
                slice_idx = list(range(0, (n_items - start_idx)))
                stop_idx = n_items

            # divide item_inds into batches
            item_indices = list(range(start_idx, stop_idx))

            # assert len(item_keys) == batch_size
            if len(item_indices) != batch_size:
                print(start_idx)
                print(stop_idx)
                print(len(slice_idx))

            # create item batch


            #tokenise text into token IDs
            tokens = []
            for i in item_indices:
                tokens.append(self.tokenize_text_to_ids(content[item_keys[i]]['text'], add_special_tokens, lower_case))


            # if naive_method:
            #     tokens_raw, _ = zip(*[self.tokenize_text_to_ids(text, add_special_tokens=False, lower_case=True)
            #                           for text in content[item_keys]])

            #create tensor from tokens
            x_in = torch.tensor(tokens, requires_grad=False, device=self.device).long()  # batch_size x max_len

            # create naive sequence features
            # if naive_method:
            #     naive_emb = self.get_naive_seq_emb(tokens_raw, method=naive_method) #batch_size x emb_dim

            # generate BERT output
            bert_outputs = self.encode_input_ids(x_in)

            # else:
            #     # create naive sequence features
            #     naive_emb = torch.randn([self.batch_size, self.embedding_dim_cur])
            #
            #     # generate dummy BERT output
            #     bert_outputs = get_dummy_bert_output(len(slice_idx), self.embedding_dim_cur, self.max_seq_len_cur)

            # extract BERT features & add features to dictionary "seq_embeddings"
            # {method: {art_ID: [feature_tensor]}}
            for i, (method, n) in enumerate(methods):
                bert_features = self.extract_bert_features(bert_outputs, method, n)  # batch_size x emb_dim

                assert bert_features.shape[0] == len(slice_idx)

                # add features to dictionary
                for idx in slice_idx:
                    seq_embeddings[i][item_keys[item_indices[idx]]] = bert_features[idx, :]

            start_idx += batch_size
            stop_idx += batch_size

        return seq_embeddings

    def truncate_seq(self, tokens, max_seq_len=None):
        if max_seq_len is None:
            max_len = self.max_seq_len
        else: 
            max_len = max_seq_len

        if len(tokens) < max_len:
            tokens = tokens[:-1]
            n = max_len - len(tokens) - 1
            tokens += n * [self.bert_tokenizer.pad_token]
        elif len(tokens) > max_len:
            tokens = tokens[:max_len - 1]
        else:
            return tokens

        tokens.append(self.bert_tokenizer.sep_token)

        return tokens

    def tokenize_text_to_ids(self, text, add_special_tokens=True, lower_case=False):
        """
        With tokenizer, separate text first into tokens
        and then convert these to corresponding IDs of the vocabulary

        Return:
            tokens: list of token IDs
            n_words: number of words in full sequence (before truncating)
        """
        sents = self.sent_tokenizer(text)
        tokens = []

        added_tokens = 0

        if add_special_tokens:
            tokens.append(self.bert_tokenizer.cls_token)
            added_tokens += 1

        # split each sentence of the text into tokens
        for s in sents:
            if lower_case:
                tokens.extend([word.lower() for word in self.bert_tokenizer.tokenize(s) if word.isalpha()])
            else:
                tokens.extend([word for word in self.bert_tokenizer.tokenize(s) if word.isalpha()])

            if add_special_tokens:
                tokens.append(self.bert_tokenizer.sep_token)
                added_tokens += 1

        tokens = self.truncate_seq(tokens)

        #assert len(tokens) == max_len

        return self.bert_tokenizer.convert_tokens_to_ids(tokens)

    def encode_input_ids(self, x_in):
        """
        Forward pass through BertModel with input tokens to produce Bert Outputs (last_hidden, pooled_out, etc.)

        :param x_in: torch tensor of shape batch_size x max_seq_len
        :return: list of torch tensor (bert_output): last_hidden, pooled_out, hidden_outs, attention_weights
        """
        with torch.no_grad():
            return self.bert_model(x_in)