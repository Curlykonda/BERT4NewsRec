from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory

import torch
import torch.utils.data as data_utils

class BertDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        args.num_items = len(self.smap)
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.item_count + 1

        self.split_method = args.split
        self.multiple_eval_items = args.split == "time_threshold"

        # self.vocab = dataset['vocab']
        # self.art_idx2word_ids = dataset['art2words']

        ####################
        # Negative Sampling
        code = args.train_negative_sampler_code
        train_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                          self.user_count, self.item_count,
                                                          args.train_negative_sample_size,
                                                          args.train_negative_sampling_seed,
                                                          self.save_folder)
        code = args.test_negative_sampler_code
        test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         self.save_folder)

        self.train_negative_samples = train_negative_sampler.get_negative_samples()
        self.test_negative_samples = test_negative_sampler.get_negative_samples()

    @classmethod
    def code(cls):
        return 'bert'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = BertTrainDataset(self.train, self.art_idx2word_ids, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        targets = self.val if mode == 'val' else self.test
        dataset = BertEvalDataset(self.train, targets, self.art_idx2word_ids, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples, self.multiple_eval_items)
        return dataset


class BertDataloaderNews(BertDataloader):
    def __init__(self, args, dataset):
        super(BertDataloaderNews, self).__init__(args, dataset)

        dataset = dataset.load_dataset()
        self.vocab = dataset['vocab']
        self.art_index2word_ids = dataset['art2words'] # art ID -> [word IDs]
        #smap:

        # create direct mapping art_id -> word_ids
        self.art_id2word_ids = {art_idx: self.art_index2word_ids[art_id] for art_id, art_idx in self.smap.items()}
        del self.art_index2word_ids

        # map negative samples to word_ids

        # ####################
        # # Negative Sampling
        # code = args.train_negative_sampler_code
        # train_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
        #                                                   self.user_count, self.item_count,
        #                                                   args.train_negative_sample_size,
        #                                                   args.train_negative_sampling_seed,
        #                                                   self.save_folder)
        # code = args.test_negative_sampler_code
        # test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
        #                                                  self.user_count, self.item_count,
        #                                                  args.test_negative_sample_size,
        #                                                  args.test_negative_sampling_seed,
        #                                                  self.save_folder)
        #
        # self.train_negative_samples = train_negative_sampler.get_negative_samples()
        # self.test_negative_samples = test_negative_sampler.get_negative_samples()

    @classmethod
    def code(cls):
        return 'bert_news'

    def _get_train_dataset(self):
        dataset = BertTrainDatasetNews(self.train, self.art_id2word_ids, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng)
        return dataset

    def _get_eval_dataset(self, mode):
        targets = self.val if mode == 'val' else self.test
        dataset = BertEvalDatasetNews(self.train, targets, self.art_id2word_ids, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples, self.multiple_eval_items)
        return dataset

def art_idx2word_ids(art_idx, mapping):
    if mapping is not None:
        return mapping[art_idx]
    else:
        return art_idx

class BertTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):

        # generate masked item sequence on-the-fly
        user = self.users[index]
        seq = self._getseq(user)

        tokens, labels = self.gen_masked_seq(seq)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        # mask token are also append to the left if sequence needs padding
        # Why not uses separate padding token? how does model know when to predict for an actual masked off item?
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]

    def gen_masked_seq(self, seq):
        tokens = []
        labels = []

        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        return tokens, labels


class BertTrainDatasetNews(BertTrainDataset):
    def __init__(self, u2seq, art2words, max_len, mask_prob, mask_token, num_items, rng):
        super(BertTrainDatasetNews, self).__init__(u2seq, max_len, mask_prob, mask_token, num_items, rng)

        self.art2words = art2words

    def gen_masked_seq(self, seq):
        tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(art_idx2word_ids(self.rng.randint(1, self.num_items), self.art2words))
                else:
                    tokens.append(art_idx2word_ids(s, self.art2words))

                labels.append(s)
            else:
                tokens.append(art_idx2word_ids(s, self.art2words))
                labels.append(0)

        return tokens, labels


class BertEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples, multiple_eval_items=True):
        self.u2hist = u2seq
        self.u_sample_ids = sorted(self.u2hist.keys())
        self.u2targets = u2answer
        #self.art2words = art2words
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

        self.mul_eval_items = multiple_eval_items

        if self.mul_eval_items:
            u2hist_ext = {}
            u2targets_ext = {}
            for u in self.u_sample_ids:
                hist = self.u2hist[u]
                for i, item in enumerate(self.u2targets[u]):
                    u_ext = self.concat_ints(u, i) # extend user id with item enumerator
                    target = item
                    u2hist_ext[u_ext] = hist
                    u2targets_ext[u_ext] = target
                    hist.append(item)

            self.u_sample_ids = list(u2hist_ext.keys())
            self.u2hist = u2hist_ext
            self.u2targets = u2targets_ext

    def __len__(self):
        return len(self.u_sample_ids)

    def __getitem__(self, index):
        user = self.u_sample_ids[index]
        seq = self.u2hist[user]
        target = self.u2targets[user]

        if target == []:
            return
        else:
            if isinstance(user, str):
                negs = self.negative_samples[int(user[:-1])]
            else:
                negs = self.negative_samples[user] # get negative samples
            return self.gen_eval_instance(seq, target, negs)

        # negs = self.negative_samples[user]
        #
        # candidates = target + negs
        # labels = [1] * len(target) + [0] * len(negs)
        #
        # seq = seq + [self.mask_token] # model can only predict the next
        # seq = seq[-self.max_len:]
        # padding_len = self.max_len - len(seq)
        # seq = [0] * padding_len + seq
        #
        # return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)

    def gen_eval_instance(self, hist, target, negs):
        candidates = target + negs
        #candidates = [art_idx2word_ids(cand, self.art2words) for cand in candidates]
        labels = [1] * len(target) + [0] * len(negs)

        hist = hist + [self.mask_token]  # predict only the next/last token in seq
        hist = hist[-self.max_len:]
        padding_len = self.max_len - len(hist)
        hist = [0] * padding_len + hist

        return torch.LongTensor(hist), torch.LongTensor(candidates), torch.LongTensor(labels)

    def concat_ints(self, a, b):
        return str(f"{a}{b}")


class BertEvalDatasetNews(BertEvalDataset):

    def __init__(self, u2seq, u2answer, art2words, max_len, mask_token, negative_samples, multiple_eval_items=True):
        super(BertEvalDatasetNews, self).__init__(u2seq, u2answer, max_len, mask_token, negative_samples, multiple_eval_items)

        self.art2words = art2words

    def gen_eval_instance(self, hist, target, negs):
        candidates = target + negs
        candidates = [art_idx2word_ids(cand, self.art2words) for cand in candidates]
        labels = [1] * len(target) + [0] * len(negs)

        hist = hist + [self.mask_token]  # predict only the next/last token in seq
        hist = hist[-self.max_len:]
        padding_len = self.max_len - len(hist) # apply left-side padding
        hist = [0] * padding_len + hist # Padding token := 0

        return torch.LongTensor(hist), torch.LongTensor(candidates), torch.LongTensor(labels)