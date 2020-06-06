import itertools
from collections import defaultdict

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

        '''
        Currently: 
        train (dict): {idx: [u_idx, hist, target]}
        test (dic):  "                      "
        
        Req.:        
        How can we get proper negative samples? 
        For each user, we must know her complete (train) history and the test items
        So that we can exclude them from sampling
        Furthermore, we want to know all test items to exclude them from neg sampling for train data
        
        Idea: 
        
        From pre-processing:
        train (dict(list)): {u_idx: [([hist1], target1), .. ([histN], targetN])}
        test (dict(list)): {u_idx: ([hist], [targets])}
        
        In Dataloader: 
         - get negative samples: for each target (train & test) 
            -> train_neg_samples (dict(list)): {idx: [samples]}
         - re-format train & test -> (dict(list)): {idx:[u_idx, [hist], target]}
         - convert dict with u_idx as keys to dict with indices for each train/test instance
        
        '''

        ####################
        # Negative Sampling

        self.train_neg_sampler = self.get_negative_sampler("train",
                                                           args.train_negative_sampler_code,
                                                           args.train_negative_sample_size,
                                                           args.train_negative_sampling_seed,
                                                           self.valid_items['train'],
                                                           self.get_seq_lengths(self.train))

        train_negative_samples = self.train_neg_sampler.get_negative_samples()
        # (dict): {u_idx: [neg_samples] * seq_length[u_idx]}

        self.test_neg_sampler = self.get_negative_sampler("test",
                                                           args.test_negative_sampler_code,
                                                           args.test_negative_sample_size,
                                                           args.test_negative_sampling_seed,
                                                           self.valid_items['test'],
                                                           self.get_seq_lengths(self.test, mode='eval'))

        test_negative_samples = self.test_neg_sampler.get_negative_samples()

        self.val_neg_sampler = self.get_negative_sampler("val",
                                                          args.test_negative_sampler_code,
                                                          args.test_negative_sample_size,
                                                          args.test_negative_sampling_seed,
                                                          self.valid_items['test'],
                                                          self.get_seq_lengths(self.val, mode='eval'))

        # create indexed train, test and val instances
        # -> train (dict): {idx: [u_idx, hist, target, neg_samples]}
        train_instances = defaultdict(list)
        for u_idx, vals in self.train.items():
            neg_samples = train_negative_samples[u_idx]
            for i, (hist, target) in enumerate(vals):
                train_instances[len(train_instances)] = [u_idx, hist, target, neg_samples[i]]

        self.train = train_instances

        # test
        test_instances = defaultdict(list)
        for u_idx, (hist, targets) in self.test.items():
            neg_samples = test_negative_samples[u_idx]
            for i, target in enumerate(targets):
                test_instances[len(test_instances)] = [u_idx, hist, target, neg_samples[i]]

        # val
        val_instances = defaultdict(list)
        for u_idx, (hist, targets) in self.val.items():
            if len(targets) > 1:
                neg_samples = test_negative_samples[u_idx]
                for i, target in enumerate(targets):
                    val_instances[len(val_instances)] = [u_idx, hist, target, self.rnd.choice(neg_samples)]

        # replace train, test and val with the new instances
        self.train = train_instances
        self.test = test_instances
        self.val = val_instances

        print("Train instances: {} \n Test: {} \n Val: {}".format(*list(map(len, [self.train, self.test, self.val]))))


    @classmethod
    def code(cls):
        return 'npa'

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
        dataset = NpaTrainDataset(self.train, self.art_index2word_ids, self.max_hist_len, self.max_article_len, self.rnd) # , self.art_idx2word_ids
        return dataset

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        idx2data = self.val if mode == 'val' else self.test
        dataset = NpaEvalDataset(idx2data, self.art_index2word_ids, self.max_hist_len, self.max_article_len, self.rnd)
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
            seq_lengths = {u_idx: len(targets) for u_idx, (hist, targets) in data.items()}
        else:
            raise ValueError("{} is not a valid mode for this function! Try 'train', 'test', or 'eval'.".format(mode))

        return seq_lengths



class NpaTrainDataset(data_utils.Dataset):
    def __init__(self, idx2instance, art2words, max_hist_len, max_article_len, rnd, pad_token=0, w_time_stamp=False):
        self.idx2instance = idx2instance
        self.art2words = art2words

        self.max_hist_len = max_hist_len
        self.max_article_len = max_article_len
        self.pad_token = pad_token
        self.rnd = rnd

        self.w_time_stamp = w_time_stamp

    def __len__(self):
        return len(self.idx2instance)

    def __getitem__(self, index):
        # retrieve prepped data
        u_idx, hist, target, neg_samples = self.idx2instance[index]

        return self.gen_train_instance(u_idx, hist, target, neg_samples)

    def gen_train_instance(self, u_idx, hist, target, neg_samples):

        # shuffle candidates
        candidates = [target] + neg_samples
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

        # labels = pad_seq(labels, pad_token=0, max_hist_len=self.max_hist_len)

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
            inp['u_idx'] = torch.LongTensor([u_idx] * self.max_hist_len)
            # need tensors of equal length for collate function

        return {'input': inp, 'lbls': torch.LongTensor(labels)}

class NpaEvalDataset(data_utils.Dataset):
    def __init__(self, idx2data, art2words, max_hist_len, max_article_len, rnd, pad_token=0, w_time_stamp=False):
        self.idx2instance = idx2data
        self.art2words = art2words

        self.max_hist_len = max_hist_len
        self.max_article_len = max_article_len
        self.pad_token = pad_token
        self.rnd = rnd

        self.w_time_stamp = w_time_stamp

    def __len__(self):
        return len(self.idx2instance)

    def __getitem__(self, index):
        # generate model input
        u_idx, hist, target, neg_samples = self.idx2instance[index]

        return self.gen_eval_instance(u_idx, hist, target, neg_samples)

    def gen_eval_instance(self, u_idx, hist, target, neg_samples):

        # shuffle candidates
        candidates = [target] + neg_samples
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

        #labels = pad_seq(labels, pad_token=0, max_hist_len=self.max_hist_len)

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
