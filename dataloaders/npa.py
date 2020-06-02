import itertools

from dataloaders.base import AbstractDataloader
from dataloaders.negative_samplers import negative_sampler_factory
from source.utils import check_all_equal, map_time_stamp_to_vector


import torch
import torch.utils.data as data_utils

class NpaDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        args.num_items = len(self.smap)
        self.max_hist_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob

        self.split_method = args.split
        self.multiple_eval_items = args.split == "time_threshold"
        self.valid_items = self.get_valid_items()

        # self.vocab = dataset['vocab']
        # self.art_idx2word_ids = dataset['art2words']

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