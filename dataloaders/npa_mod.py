import itertools
from collections import defaultdict

import torch
import torch.utils.data as data_utils

from dataloaders.base import AbstractDataloader
from dataloaders.bert import art_idx2word_ids, pad_seq
from dataloaders.negative_samplers import negative_sampler_factory


class NpaModDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        args.num_items = len(self.smap)
        self.max_hist_len = args.max_hist_len
        self.target_prob = args.bert_mask_prob
        self.max_n_targets = int(self.max_hist_len * self.target_prob)

        self.split_method = args.split
        self.multiple_eval_items = args.split == "time_threshold"

        self.w_time_stamps = args.incl_time_stamp

        data = dataset.load_dataset()
        self.vocab = data['vocab']
        if self.vocab is not None:
            args.vocab_path = dataset._get_preprocessed_dataset_path()

        self.art_id2word_ids = data['art2words']  # art ID -> [word IDs]
        self.max_article_len = args.max_article_len

        if args.fix_pt_art_emb:
            args.rel_pc_art_emb_path = dataset._get_precomputed_art_emb_path()
            self.art_index2word_ids = None
        else:
            # create direct mapping art_id -> word_ids
            self.art_index2word_ids = {art_idx: self.art_id2word_ids[art_id] for art_id, art_idx in self.smap.items()}
        del self.art_id2word_ids

        # retrieve valid items for negative sampling
        if 'valid_items' in data:
            self.valid_items = data['valid_items']
        else:
            self.valid_items = self.get_valid_items()


        ####################
        # Negative Sampling
        train_neg_sampler = self.get_negative_sampler("train",
                                                           args.train_negative_sampler_code,
                                                           args.train_negative_sample_size,
                                                           args.train_negative_sampling_seed,
                                                           self.valid_items['train'],
                                                           self.get_seq_lengths(self.train))

        self.train_negative_samples = train_neg_sampler.get_negative_samples()
        # (dict): {u_idx: [neg_samples] * train_items[u_idx]}

        test_neg_sampler = self.get_negative_sampler("test",
                                                           args.test_negative_sampler_code,
                                                           args.test_negative_sample_size,
                                                           args.test_negative_sampling_seed,
                                                           self.valid_items['test'],
                                                           self.get_seq_lengths(self.test, mode='eval'))

        self.test_negative_samples = test_neg_sampler.get_negative_samples()
        # (dict): {u_idx: [neg_samples] * test_items[u_idx]}

        val_neg_sampler = self.get_negative_sampler("val",
                                                          args.test_negative_sampler_code,
                                                          args.test_negative_sample_size,
                                                          args.test_negative_sampling_seed,
                                                          self.valid_items['test'],
                                                          self.get_seq_lengths(self.val, mode='eval'))

        self.val_neg_samples = val_neg_sampler.get_negative_samples()


    @classmethod
    def code(cls):
        return 'npa_mod'

    def get_valid_items(self):
        """
        collect art_ids that fall in train & test period for negative sampling

        """

        all_items = set(self.smap.values()) # train + test + val
        raise NotImplementedError()
        #
        # if self.split_method != "time_threshold":
        #     test = train = all_items
        # else:
        #     train = set(itertools.chain.from_iterable(self.train.values()))
        #     test = all_items
        #
        # return {'train': list(train), 'test': list(test)}

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = NpaTrainDataset(self.train, self.train_negative_samples, self.art_index2word_ids, self.max_hist_len, self.max_article_len, self.target_prob, self.rnd) # , self.art_idx2word_ids
        return dataset

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        if 'val' == mode:
            idx2val_items = self.val
            neg_samples = self.val_neg_samples
        else:
            idx2val_items = self.test
            neg_samples = self.test_negative_samples

        # idx2val_items, neg_samples, art2words, max_hist_len, max_article_len, rnd, pad_token=0, w_time_stamp=False):
        dataset = NpaEvalDataset(idx2val_items, neg_samples, self.art_index2word_ids,
                                    self.max_hist_len, self.max_article_len, self.rnd)
        return dataset

    def get_negative_sampler(self, mode, code, neg_sample_size, seed, item_set, seq_lengths):
        # sample negative instances for each user

        # use item set for simple neg sampling
        negative_sampler = negative_sampler_factory(self.args.train_method, mode, code, self.train, self.val, self.test,
                                                    self.user_count, item_set, neg_sample_size,
                                                    seed, seq_lengths, self.save_folder)
        return negative_sampler

    def get_seq_lengths(self, data: dict, mode='train'):
        # determine sequence length for each user entry
        if 'train' == mode:
            seq_lengths = {u_idx: len(instances) for u_idx, instances in data.items()}
        elif 'eval' == mode or 'test' == mode:
            # {u_idx: [hist, targets]
            seq_lengths = {u_idx: len(val_items) for u_idx, val_items in data.items()}
        else:
            raise ValueError("{} is not a valid mode for this function! Try 'train', 'test', or 'eval'.".format(mode))

        return seq_lengths



class NpaTrainDataset(data_utils.Dataset):
    def __init__(self, idx2hist, neg_samples, art2words, max_hist_len, max_article_len, target_prob, rnd, pad_token=0, w_time_stamp=False):

        self.idx2hist = idx2hist
        self.neg_samples = neg_samples
        self.art2words = art2words

        self.max_hist_len = max_hist_len
        self.max_article_len = max_article_len
        self.target_prob = target_prob # probability to select item as target
        self.max_n_targets = int(max_hist_len * target_prob)

        self.pad_token = pad_token
        self.rnd = rnd

        self.w_time_stamp = w_time_stamp
        self.n_cands = None

    def __len__(self):
        return len(self.idx2hist)

    def __getitem__(self, u_idx):
        # retrieve prepped data
        hist = self.idx2hist[u_idx]
        neg_samples = self.neg_samples[u_idx]

        return self.gen_train_instance(u_idx, hist, neg_samples)

    def gen_train_instance(self, u_idx, org_hist, neg_samples):

        hist = []
        labels = []
        candidates = []
        time_stamps = []

        pos_irrelevant_lbl = -1  # label to indicate irrelevant position to avoid confusion with other categorical labels

        for idx, entry in enumerate(org_hist):
            if self.w_time_stamp:
                art_id, ts = entry
                time_stamps.append(ts)
            else:
                art_id = entry

            prob = self.rnd.random()
            if prob < self.target_prob and len(candidates) < self.max_n_targets:

                cands = neg_samples[idx] + [art_id]
                self.rnd.shuffle(cands)  # shuffle candidates so model cannot trivially guess target position
                candidates.append([art_idx2word_ids(art, self.art2words) for art in cands])

                if self.n_cands is None:
                    self.n_cands = len(cands)

                # add categorical label [0, N_C]
                labels.append(cands.index(art_id))

            else:
                hist.append(art_id)

        # randomly subsample history from remaining items + map to words
        hist = [art_idx2word_ids(art_id, self.art2words) for art_id
                    in self.rnd.sample(hist, min(len(hist), self.max_hist_len))]

        # padding
        hist = pad_seq(hist, self.pad_token, self.max_hist_len,
                       max_article_len=(self.max_article_len if self.art2words is not None else None))

        candidates = pad_seq(candidates, self.pad_token, self.max_hist_len, n=self.n_cands,
                             max_article_len=(self.max_article_len if self.art2words is not None else None))

        labels = pad_seq(labels, pos_irrelevant_lbl, self.max_hist_len)

        assert len(hist) == self.max_hist_len
        assert len(labels) == self.max_hist_len
        assert len(candidates) == self.max_hist_len

        ### Output ####

        inp = {'hist': torch.LongTensor(hist), 'cands': torch.LongTensor(candidates)}

        if self.w_time_stamp:
            # len_time_vec = len(time_stamps[0])
            # time_stamps = pad_seq(time_stamps, pad_token=0,
            #                       max_hist_len=self.max_hist_len, n=len_time_vec)
            # inp['ts'] = torch.LongTensor(time_stamps)
            raise NotImplementedError()

        if u_idx is not None:
            inp['u_idx'] = torch.LongTensor([u_idx] * self.max_hist_len)
            # need tensors of equal length for collate function

        return {'input': inp, 'lbls': torch.LongTensor(labels)}

class NpaEvalDataset(data_utils.Dataset):
    def __init__(self, idx2val_items, neg_samples, art2words, max_hist_len, max_article_len, rnd, pad_token=0, w_time_stamp=False):
        self.idx2hist = idx2val_items # contains hist + val_items

        self.neg_samples = neg_samples
        self.art2words = art2words

        self.max_hist_len = max_hist_len
        self.max_article_len = max_article_len
        self.pad_token = pad_token
        self.rnd = rnd

        self.w_time_stamp = w_time_stamp
        self.n_cands = None

    def __len__(self):
        return len(self.idx2hist)

    def __getitem__(self, u_idx):
        # generate model input
        return self.gen_eval_instance(u_idx, self.idx2hist[u_idx], self.neg_samples[u_idx])

    def gen_eval_instance(self, u_idx, hist, neg_samples):
        # apply leave-one-out strategy
        target = hist.pop(-1)
        # shuffle candidates
        candidates = [target] + neg_samples[-1]
        self.rnd.shuffle(candidates)

        self.n_cands = len(candidates) if self.n_cands is None else self.n_cands

        # construct labels
        labels = [0] * len(candidates)
        labels[candidates.index(target)] = 1

        candidates = [art_idx2word_ids(art, self.art2words) for art in candidates]  # map to words

        # sub sample history
        hist = [art_idx2word_ids(art, self.art2words) for art
                    in self.rnd.sample(hist, min(len(hist), self.max_hist_len))] # subsample & map to words

        # padding
        hist = pad_seq(hist, self.pad_token, self.max_hist_len,
                       max_article_len=(self.max_article_len if self.art2words is not None else None))

        assert len(hist) == self.max_hist_len

        ### Output ####

        inp = {'hist': torch.LongTensor(hist), 'cands': torch.LongTensor(candidates)}

        if self.w_time_stamp:
            # len_time_vec = len(time_stamps[0])
            # time_stamps = pad_seq(time_stamps, pad_token=0,
            #                       max_hist_len=self.max_hist_len, n=len_time_vec)
            # inp['ts'] = torch.LongTensor(time_stamps)
            raise NotImplementedError()

        if u_idx is not None:
            inp['u_idx'] = torch.LongTensor(
                [u_idx] * self.max_hist_len)  # need tensors of equal lenght for collate function

        return {'input': inp, 'lbls': torch.LongTensor(labels)}
