import datetime
import arrow
import pickle
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from abc import *

from .base import AbstractDataset
from source.preprocessing.utils_news_prep import precompute_dpg_art_emb, preprocess_dpg_news_file, prep_dpg_user_file
from source.preprocessing.get_dpg_data_sample import get_data_n_rnd_users



class AbstractDatasetDPG(AbstractDataset):

    def __init__(self, args):
        super(AbstractDatasetDPG, self).__init__(args)
        self.args = args
        self.min_hist_len = self.args.min_hist_len
        self.use_content = args.use_article_content
        self.pt_news_encoder = args.pt_news_encoder

        seed = args.dataloader_random_seed
        self.rnd = random.Random(seed)

        self.vocab = None
        self.art_idx2word_ids = None
        self.art_embs = None

    @property
    def sample_method(self):
        return self.args.data_sample_method

    @classmethod
    def all_raw_file_names(cls):
        pass

    def _get_all_precomputed_art_emb_path(self):
        return Path(self.args.pt_art_emb_path)

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}-min_len{}-split{}-news_enc_{}' \
            .format(self.code(), self.min_hist_len, self.split, self.pt_news_encoder)
        return preprocessed_root.joinpath(folder_name)

    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def load_pc_art_embs(self):
        pt_art_emb_path = self._get_precomputed_art_emb_path()
        with pt_art_emb_path.open('rb') as fin:
            art_embs = pickle.load(fin) # matrix containing article embeddings

        return art_embs

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)

        pt_art_emb_path = self._get_precomputed_art_emb_path()
        if pt_art_emb_path.is_file():
            print("Found pre-computed article embeddings for this setting. Skip re-computing")
            with pt_art_emb_path.open('rb') as fin:
                self.art_embs = pickle.load(fin) # matrix containing article embeddings

        ###################
        # Preprocess data
        news_data, art_id2idx = self.prep_dpg_news_data()

        train, val, test, u_id2idx = self.prep_dpg_user_data(news_data, art_id2idx)

        #train, val, test = self.split_data(user_data, len(u_id2idx))

        # train (dict): {user_idx: [art_idx_1, ..., art_idx_L_u] } ## w/o time
        # train (dict): {user_idx: [(art_idx_1, time_stamp_1), ..., (art_idx_L_u, time_stamp_L_u)] } ## w/ time

        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': u_id2idx,
                   'smap': art_id2idx,
                   'vocab': self.vocab,
                   'art2words': self.art_idx2word_ids,
                   'art_emb': self.art_embs, # article emb matrix
                   'rnd': self.rnd}

        # save
        with dataset_path.open('wb') as fout:
            pickle.dump(dataset, fout)

        with pt_art_emb_path.open('wb') as fout:
            pickle.dump(self.art_embs, fout)


    def split_data(self, user_data, user_count):

        if self.args.split == 'leave_one_out':
            # preserve the last two items for validation and testing respectively. remainder for training
            print('Splitting')

            train, val, test = {}, {}, {}
            for user in range(user_count):
                items = [*list(zip(*user_data[user]))[0]] # only take article ids, discard the timestamps
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]

            return train, val, test
        else:
            raise NotImplementedError()

    def load_raw_read_histories(self):
        # this assumes that we already have sampled a subset of users
        folder_path = self._get_sampledata_folder_path()
        file_path = folder_path.joinpath('user_data.pkl')
        user_data = pickle.load(file_path.open('rb')) # dict

        return user_data

    def _get_sampledata_folder_path(self):
        return Path(self.sampledata_folder_path)

    def sample_dpg_data(self):
        m = self.sample_method
        print(m)
        if 'n_rnd_users' == m:
            news_data, user_data, logging_dates = get_data_n_rnd_users(config.data_dir, n_users,
                                                                       news_len=config.news_len,
                                                                       min_hist_len=config.min_hist_len,
                                                                       max_hist_len=config.max_hist_len,
                                                                       min_test_len=config.min_test_len,
                                                                       save_path=config.save_path,
                                                                       sample_name=sample_name,
                                                                       test_time_thresh=threshold_time,
                                                                       overwrite_existing=config.overwrite_existing)
        else:
            raise NotImplementedError()

    def load_raw_news_data(self):
        folder_path = self._get_sampledata_folder_path()
        file_path = folder_path.joinpath('news_data.pkl')
        news_data = pickle.load(file_path.open('rb'))  # dict

        return news_data

    def _get_precomputed_art_emb_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('pt_art_embs.pkl')

    @abstractmethod
    def prep_dpg_user_data(self):
        raise NotImplementedError("Please specify preprocessing of user data for this dataset!")

    @abstractmethod
    def prep_dpg_news_data(self):
        news_data = self.load_raw_news_data()
        raise NotImplementedError("Please specify preprocessing of user data for this dataset!")

    def get_rnd_obj(self):
        return self.rnd


class DPG_Nov19Dataset(AbstractDatasetDPG):
    def __init__(self, args):
        super(DPG_Nov19Dataset, self).__init__(args)

        self.sampledata_folder_path = "./Data/DPG_nov19/medium_time_split_n_rnd_users"

    @classmethod
    def code(cls):
        return 'DPG_nov19'

    @classmethod
    def all_raw_file_names(cls):
        return ['items',
                'users']

    def sample_dpg_data(self):
        m = self.sample_method
        print(m)
        if 'wu' == m:
            pass
        elif 'm_common' == m:
            pass
        else:
            raise NotImplementedError()

    def prep_dpg_user_data(self, news_data, art_id2idx):

        user_data = self.load_raw_read_histories()
        u_id2idx = {}

        # for user in range(user_count):
        #     items = [*list(zip(*user_data[user]))[0]]  # only take article ids, discard the timestamps
        #     train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
        #
        #     return train, val, test

        train, val, test = defaultdict(list), defaultdict(list), defaultdict(list)

        for u_id in user_data.keys():

            if 'masked_item' == self.args.train_method:

                if 'time_threshold' == self.args.split:
                    # split user interactions according to certain threshold timestamp
                    # e.g. first 3 weeks for training, last week for testing
                    try:
                        threshold_date = arrow.get(self.args.time_threshold, "DD-MM-YYYY HH:mm:ss").timestamp
                        #1574639999
                    except:
                        raise ValueError("Threshold date must string of format: 'DD-MM-YYYY HH:mm:ss'")

                    # check if data has already been separated into train & test
                    if 'articles_train' in user_data[u_id].keys():
                        train_items = [(art_id2idx[art_id], time_stamp) for art_id, time_stamp
                                       in sorted(user_data[u_id]['articles_train'], key=lambda tup: tup[1])
                                       if art_id in art_id2idx]

                        test_items = [(art_id2idx[art_id], time_stamp) for art_id, time_stamp
                                      in sorted(user_data[u_id]['articles_test'], key=lambda tup: tup[1])
                                      if art_id in art_id2idx]


                        if len(test_items) < 1 or len(train_items) < 2:
                            continue

                        # confirm time intervals
                        if train_items[-1][1] <= threshold_date and test_items[0][1] >= threshold_date:
                            pass
                        else:
                            raise ValueError("Split into time intervals incorrect. check preprocessing!")

                        u_id2idx[u_id] = u_idx = len(u_id2idx)  # create mapping from u_id to index

                        if self.args.incl_time_stamp:
                            train[u_idx] = train_items
                            test[u_idx] = test_items
                        else:
                            train[u_idx] = [*list(zip(*train_items))[0]]
                            test[u_idx] = [*list(zip(*test_items))[0]]

                        # add to validation sample
                        val[u_idx] = self.select_rnd_item_for_validation(test[u_idx])

                    else:
                        # split user history according to time stamps. usually this is done in sampling process earlier
                        full_hist = [(art_id2idx[art_id], time_stamp) for art_id, time_stamp
                                      in sorted(user_data[u_id]['articles_read'], key=lambda tup: tup[1])]
                        tmp_test = []
                        for i, (item, time_stamp) in enumerate(full_hist):
                            if time_stamp < threshold_date:
                                train[u_idx].append((item, time_stamp) if self.args.incl_time_stamp else item)
                            else:
                                if self.args.incl_time_stamp:
                                    tmp_test = full_hist[i:]
                                else:
                                    tmp_test = [*list(zip(*full_hist[i:]))[0]]
                                break

                        test[u_idx] = tmp_test

                        # add validation sample
                        val[u_idx] = self.select_rnd_item_for_validation(test[u_idx])
                else:
                    raise NotImplementedError()

            elif 'wu' == self.args.train_method:
                # create instance for each train impression
                # create instance for each test impression
                #


                raise NotImplementedError()
            elif 'pos_cut_off' == self.args.train_method:
                raise NotImplementedError()
            else:
                raise NotImplementedError()

        return train, val, test, u_id2idx

    def select_rnd_item_for_validation(self, test_items):
        # select random portion of test items as subset for validation
        # exclude possibility to use last test interaction for validation because it's reserved for testing
        val_pos = self.rnd.choice(range(len(test_items) - 1)) if len(test_items) > 1 else None
        coin_flip = self.rnd.random()
        if val_pos is not None and coin_flip > (1 - self.args.validation_portion):
            return test_items[:val_pos]
        else:
            return []

    def prep_dpg_news_data(self):
        news_data = self.load_raw_news_data()
        if self.use_content:
            # use article content to create contextualised representations

            # check for existing
            if self.args.pt_art_emb_path is not None:
                raise NotImplementedError()

            if self.args.pt_news_encoder is None:
                vocab, news_as_word_ids, art_id2idx = preprocess_dpg_news_file(news_file=news_data,
                                                                               tokenizer=word_tokenize,
                                                                               min_counts_for_vocab=self.args.min_counts_for_vocab,
                                                                               max_article_len=self.args.max_article_len,
                                                                               max_vocab_size=self.args.max_vocab_size)
                self.art_idx2word_ids = news_as_word_ids
                self.vocab = vocab

            else:
                art_id2idx, art_embs = precompute_dpg_art_emb(news_data=news_data,
                                                                news_encoder_code=self.args.pt_news_encoder,
                                                                max_article_len=self.args.max_article_len,
                                                                art_emb_dim=self.args.dim_art_emb,
                                                                lower_case=self.args.lower_case,
                                                                pd_vocab=self.args.pd_vocab,
                                                                path_to_pt_model=self.args.path_pt_news_enc,
                                                                feature_method=self.args.bert_feature_method)

                self.art_embs = art_embs

        else:
            art_id2idx = {}  # {'0': 0} dictionary news indices
            for art_id in news_data['all'].keys():
                art_id2idx[art_id] = len(art_id2idx)  # map article ID -> index

        return news_data, art_id2idx


# class DPG_Dec19Dataset(AbstractDatasetDPG):
#     def __init__(self, args):
#         super(DPG_Dec19Dataset, self).__init__(args)
#
#         self.sampledata_folder_path = "./Data/DPG_dec19/dev_time_split_wu"
#
#
#     @classmethod
#     def code(cls):
#         return 'DPG_dec19'
#
#     @classmethod
#     def sample_method(self):
#         return self.args.data_sample_method
#
#     @classmethod
#     def all_raw_file_names(cls):
#         return ['items',
#                 'users']
#
#     def sample_dpg_data(self):
#         m = self.sample_method()
#         print(m)
#         if 'wu' == m:
#             pass
#         elif 'm_common' == m:
#             pass
#         else:
#             raise NotImplementedError()
#
#     def prep_dpg_user_data(self, news_data, art_id2idx):
#
#         user_data = self.load_raw_read_histories()
#         u_id2idx = {}
#         u_data_prepped = {}
#
#         for u_id in user_data.keys():
#
#             u_id2idx[u_id] = len(u_id2idx)  # create mapping from u_id to index
#
#             # pos_impre, time_stamps = zip(*[(art_id2idx[art_id], time_stamp) for _, art_id, time_stamp in user_data[u_id]["articles_read"]])
#
#             if 'masked_interest' == self.args.train_method:
#
#                 hist = [(art_id2idx[art_id], time_stamp) for _, art_id, time_stamp
#                             in sorted(user_data[u_id]["articles_read"], key=lambda tup: tup[2])]
#                 u_data_prepped[u_id2idx[u_id]] = hist
#
#             elif 'wu' == self.args.train_method:
#                 #cand_article_ids = (set(news_data['all'].keys()) - news_data['test']).union(set(news_data['train']))
#                 pass
#             elif 'pos_cut_off' == self.args.train_method:
#                 pass
#             else:
#                 raise NotImplementedError()
#
#         return u_data_prepped, u_id2idx
#
#     def prep_dpg_news_data(self):
#         news_data = self.load_raw_news_data()
#         if self.args.use_article_content:
#             # use article content to create contextualised representations
#             # vocab, news_as_word_ids, art_id2idx = preprocess_dpg_news_file(news_file=path_article_data,
#             #                                                                tokenizer=word_tokenize,
#             #                                                                min_counts_for_vocab=min_counts_for_vocab,
#             #                                                                max_article_len=max_article_len)
#             art_id2idx = None
#         else:
#             art_id2idx = {'0': 0}  # dictionary news indices
#             for art_id in news_data['all'].keys():
#                 art_id2idx[art_id] = len(art_id2idx)  # map article id to index
#
#         return news_data, art_id2idx
