import torch
import torch.nn as nn

from dataloaders.bert import BertTrainDataset, BertEvalDataset, art_idx2word_ids


class BertTrainDatasetNews(BertTrainDataset):
    def __init__(self, u2seq, art2words, max_hist_len, max_article_len, mask_prob, mask_token, num_items, rng):
        super(BertTrainDatasetNews, self).__init__(u2seq, max_hist_len, mask_prob, mask_token, num_items, rng)

        self.art2words = art2words
        self.max_article_len = max_article_len

    def gen_train_instance(self, seq):
        hist = []
        labels = []
        mask = []

        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    # put original token but note mask position
                    # tokens.append([self.mask_token] * self.max_article_len)
                    tkn = s
                    m_val = 0
                elif prob < 0.9:
                    # put random item
                    tkn = self.rng.randint(1, self.num_items)
                    # tokens.append(art_idx2word_ids(self.rng.randint(1, self.num_items), self.art2words))
                    m_val = 1
                else:
                    # put original token but no masking
                    tkn = s
                    m_val = 1

                hist.append(art_idx2word_ids(tkn, self.art2words))
                labels.append(s)
                mask.append(m_val)
            else:
                hist.append(art_idx2word_ids(s, self.art2words))
                labels.append(0)
                mask.append(1)

        # truncate sequence & apply left-side padding
        ##############################################
        # if art2word mapping is applied, hist is shape (max_article_len x max_hist_len), i.e. sequences of word IDs
        # else, hist is shape (max_hist_len), i.e. sequence of article indices
        hist = self.pad_seq(hist, max_article_len=(self.max_article_len if self.art2words is not None else None))
        labels = self.pad_seq(labels)
        mask = self.pad_seq(mask, pad_token=1)

        assert len(hist) == self.max_hist_len

        return torch.LongTensor(hist), torch.LongTensor(mask), torch.LongTensor(labels)

    def pad_seq(self, seq, pad_token=None, max_article_len=None):
        seq = seq[-self.max_hist_len:]
        pad_len = self.max_hist_len - len(seq)

        if pad_token is None:
            pad_token = self.pad_token

        if pad_len > 0:
            if max_article_len is not None:
                return [[pad_token] * max_article_len] * pad_len + seq
            else:
                return [pad_token] * pad_len + seq
        else:
            return seq


class BertEvalDatasetNews(BertEvalDataset):

    def __init__(self, u2seq, u2answer, art2words, max_hist_len, max_article_len, mask_token, negative_samples, multiple_eval_items=True):
        super(BertEvalDatasetNews, self).__init__(u2seq, u2answer, max_hist_len, mask_token, negative_samples, multiple_eval_items=multiple_eval_items)

        self.art2words = art2words
        self.max_article_len = max_article_len # len(next(iter(art2words.values())))
        self.eval_mask = [1] * (max_hist_len-1) + [0]

    def gen_eval_instance(self, hist, target, negs):
        candidates = target + negs # candidates as article indices
        #candidates = [art_idx2word_ids(cand, self.art2words) for cand in candidates]
        labels = [1] * len(target) + [0] * len(negs)

        hist = [art_idx2word_ids(art, self.art2words) for art in hist[-(self.max_hist_len- 1):]]
        # append a target to history which will be masked off
        # alternatively we could put a random item. does not really matter because it's gonna be masked off anyways
        hist = hist + [art_idx2word_ids(*target, self.art2words)]  # predict only the next/last token in seq

        ## apply padding
        padding_len = self.max_hist_len - len(hist)
        hist = [[self.pad_token] * self.max_article_len] * padding_len + hist # Padding token := 0
        #
        assert len(hist) == self.max_hist_len

        return torch.LongTensor(hist), torch.LongTensor(self.eval_mask), torch.LongTensor(candidates), torch.LongTensor(labels)

class BertTrainDatasetPrecomputedNews(BertTrainDatasetNews):
    # change: no mapping from article index to word IDs
    # simply return sequence of article indices to lookup embedding in model
    def __init__(self, u2seq, max_hist_len, max_article_len, mask_prob, mask_token, num_items, rng):

        self.art2words = None
        super(BertTrainDatasetPrecomputedNews, self).__init__(u2seq, self.art2words, max_hist_len, max_article_len, mask_prob, mask_token, num_items, rng)

