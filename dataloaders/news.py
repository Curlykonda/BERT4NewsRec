import itertools

import torch
import torch.utils.data as data_utils

from dataloaders.base import AbstractDataloader
from dataloaders.bert import art_idx2word_ids, pad_seq
from dataloaders.negative_samplers import negative_sampler_factory


class NpaDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        args.num_items = len(self.smap)
        self.max_hist_len = args.max_hist_len

        self.split_method = args.split
        self.multiple_eval_items = args.split == "time_threshold"
        self.valid_items = self.get_valid_items()

        self.w_time_stamps = args.incl_time_stamp

        data = dataset.load_dataset()
        self.vocab = data['vocab']
        if self.vocab is not None:
            args.vocab_path = dataset._get_preprocessed_dataset_path()

        self.art_index2word_ids = data['art2words'] # art ID -> [word IDs]
        self.max_article_len = args.max_article_len

        if args.fix_pt_art_emb:
            args.rel_pc_art_emb_path = dataset._get_precomputed_art_emb_path()

        self.mask_token = args.max_vocab_size + 1
        args.bert_mask_token = self.mask_token

        if self.args.fix_pt_art_emb:
            self.art_id2word_ids = None
        else:
            # create direct mapping art_id -> word_ids
            self.art_id2word_ids = {art_idx: self.art_index2word_ids[art_id] for art_id, art_idx in self.smap.items()}
        del self.art_index2word_ids


        ####################
        # Negative Sampling

        self.train_neg_sampler = self.get_negative_sampler("train",
                                                           args.train_negative_sampler_code,
                                                           args.train_negative_sample_size,
                                                           args.train_negative_sampling_seed,
                                                           self.valid_items['train'],
                                                           self.get_seq_lengths(self.train))

        self.train_negative_samples = self.train_neg_sampler.get_negative_samples()

        self.test_neg_sampler = self.get_negative_sampler("test",
                                                           args.test_negative_sampler_code,
                                                           args.test_negative_sample_size,
                                                           args.test_negative_sampling_seed,
                                                           self.valid_items['test'],
                                                           None)

        self.test_negative_samples = self.test_neg_sampler.get_negative_samples()

    @classmethod
    def code(cls):
        return 'npa'

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = NpaTrainDataset(self.train, self.max_hist_len, self.mask_prob, self.mask_token, self.item_count, self.rnd) # , self.art_idx2word_ids
        return dataset

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        targets = self.val if mode == 'val' else self.test
        dataset = NpaEvalDataset(self.train, targets, self.max_hist_len, self.mask_token, self.test_negative_samples,
                                  self.rnd, multiple_eval_items=self.multiple_eval_items)
        return dataset

    def get_negative_sampler(self, mode, code, neg_sample_size, seed, item_set, seq_lengths):
        # sample negative instances for each user

        # use item set for simple neg sampling
        negative_sampler = negative_sampler_factory(mode, code, self.train, self.val, self.test,
                                                    self.user_count, item_set,
                                                    neg_sample_size,
                                                    seed,
                                                    seq_lengths,
                                                    self.save_folder)
        return negative_sampler

    def get_valid_items(self):
        all_items = set(self.smap.values()) # train + test + val

        if self.split_method != "time_threshold":
            test = train = all_items
        else:
            train = set(itertools.chain.from_iterable(self.train.values()))
            test = all_items

        return {'train': list(train), 'test': list(test)}



class NpaTrainDataset(data_utils.Dataset):
    def __init__(self, idx2instance, art2words, train_neg_samples, max_hist_len, rnd, pad_token=0):
        self.idx2instance = idx2instance
        self.users = sorted(self.u2seq.keys())
        self.art2words = art2words
        self.train_neg_samples = train_neg_samples

        self.max_hist_len = max_hist_len
        self.pad_token = pad_token
        self.rnd = rnd

    def __len__(self):
        return len(self.idx2instance)

    def __getitem__(self, index):
        # generate model input
        u_idx, hist, target = self.idx2instance[index]

        # retrieve neg samples
        neg_samples = self.train_neg_samples[index]

        return self.gen_train_instance(u_idx, hist, target, neg_samples)

    def gen_train_instance(self, u_idx, hist, target, neg_samples):

        # shuffle candidates
        candidates = target + neg_samples
        self.rnd.shuffle(candidates)
        # construct labels
        labels = [0] * len(candidates)
        labels[candidates.index(target)] = 1

        # map articles to words
        hist = [art_idx2word_ids(art, self.art2words) for art in hist]
        candidates = [art_idx2word_ids(art, self.art2words) for art in candidates]

        # padding
        hist = pad_seq(hist, self.pad_token, self.max_hist_len,
                       max_article_len=(self.max_article_len if self.art2words is not None else None))

        labels = pad_seq(labels, pad_token=0, max_hist_len=self.max_hist_len)

        assert len(hist) == self.max_hist_len

        ### Output ####

        inp = {'hist': torch.LongTensor(hist), 'cands': torch.LongTensor(candidates)}

        if self.w_time_stamps:
            # len_time_vec = len(time_stamps[0])
            # time_stamps = pad_seq(time_stamps, pad_token=0,
            #                       max_hist_len=self.max_hist_len, n=len_time_vec)
            # inp['ts'] = torch.LongTensor(time_stamps)
            raise NotImplementedError()

        if u_idx is not None:
            inp['u_id'] = torch.LongTensor([u_idx] * self.max_hist_len)  # need tensors of equal lenght for collate function

        return {'input': inp, 'lbls': torch.LongTensor(labels)}

class NpaEvalDataset(data_utils.Dataset):
    def __init__(self, idx2instance, art2words, train_neg_samples, max_hist_len, rnd, pad_token=0):
        self.idx2instance = idx2instance
        self.users = sorted(self.u2seq.keys())
        self.art2words = art2words
        self.train_neg_samples = train_neg_samples

        self.max_hist_len = max_hist_len
        self.pad_token = pad_token
        self.rnd = rnd

    def __len__(self):
        return len(self.idx2instance)

    def __getitem__(self, index):
        # generate model input
        u_idx, hist, target = self.idx2instance[index]

        # retrieve neg samples
        neg_samples = self.train_neg_samples[index]

        return self.gen_eval_instance(u_idx, hist, target, neg_samples)

    def gen_eval_instance(self, u_idx, hist, target, neg_samples):

        # shuffle candidates
        candidates = target + neg_samples
        self.rnd.shuffle(candidates)
        # construct labels
        labels = [0] * len(candidates)
        labels[candidates.index(target)] = 1

        # map articles to words
        hist = [art_idx2word_ids(art, self.art2words) for art in hist]
        candidates = [art_idx2word_ids(art, self.art2words) for art in candidates]

        # padding
        hist = pad_seq(hist, self.pad_token, self.max_hist_len,
                       max_article_len=(self.max_article_len if self.art2words is not None else None))

        labels = pad_seq(labels, pad_token=0, max_hist_len=self.max_hist_len)

        assert len(hist) == self.max_hist_len

        ### Output ####

        inp = {'hist': torch.LongTensor(hist), 'cands': torch.LongTensor(candidates)}

        if self.w_time_stamps:
            # len_time_vec = len(time_stamps[0])
            # time_stamps = pad_seq(time_stamps, pad_token=0,
            #                       max_hist_len=self.max_hist_len, n=len_time_vec)
            # inp['ts'] = torch.LongTensor(time_stamps)
            raise NotImplementedError()

        if u_idx is not None:
            inp['u_id'] = torch.LongTensor(
                [u_idx] * self.max_hist_len)  # need tensors of equal lenght for collate function

        return {'input': inp, 'lbls': torch.LongTensor(labels)}
