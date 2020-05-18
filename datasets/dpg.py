import datetime
import pickle
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from abc import *

from .base import AbstractDataset
from source.preprocessing.utils_news_prep import *



class AbstractDatasetDPG(AbstractDataset):

    def __init__(self, args):
        super(AbstractDatasetDPG, self).__init__(args)
        self.args = args
        self.min_hist_len = self.args.min_hist_len
        self.use_content = args.use_article_content
        self.pt_news_encoder = args.pt_news_encoder

        self.vocab = None
        self.art_idx2word_ids = None
        self.art_embs = None

    @classmethod
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

        user_data, u_id2idx = self.prep_dpg_user_data(news_data, art_id2idx)

        train, val, test = self.split_data(user_data, len(u_id2idx))

        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': u_id2idx,
                   'smap': art_id2idx,
                   'vocab': self.vocab,
                   'art2words': self.art_idx2word_ids,
                   'art_emb': self.art_embs} # article emb matrix

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

        elif self.args.split == 'time_threshold':
            #assert self.args.incl_time_stamp, "No time stamps given, cannot divide by time"
            # split user interactions according to certain threshold timestamp
            # e.g. first 3 weeks for training, last week for testing

            try:
                threshold_date = int(datetime.datetime.strptime(self.args.time_threshold, '%d-%m-%Y-%H-%M-%S').strftime("%s"))  # 1577228399
            except:
                raise ValueError("Threshold date must string of format: 'dd-mm-yyyy-HH-MM-SS'")

            print('Splitting')
            train, val, test = defaultdict(list), {}, {}
            for user in range(user_count):
                tmp_test = []
                for i, (item, time_stamp) in enumerate(user_data[user]):
                    if time_stamp < threshold_date:
                        train[user].append((item, time_stamp) if self.args.incl_time_stamp else item)
                    else:
                        if self.args.incl_time_stamp:
                            tmp_test = user_data[user][i:]
                        else:
                            tmp_test = [*list(zip(*user_data[user][i:]))[0]]
                        break

                # sample 10% from test set as validation
                k = len(tmp_test) / 10
                if k > 0.5:
                    k = int(np.ceil(k))
                    val[user] = tmp_test[:k]
                    test[user] = tmp_test[k:]
                elif k >= 0.1:
                    # add one item to val set, rest to test
                    val[user] = [tmp_test[0]]
                    test[user] = tmp_test[0:]
                else:
                    # no items in testing phase
                    val[user] = []
                    test[user] = []

            return train, val, test

        else:
            raise NotImplementedError

    def load_raw_read_histories(self):
        # this assumes that we already have sampled a subset of users
        folder_path = self._get_sampledata_folder_path()
        file_path = folder_path.joinpath('user_data.pkl')
        user_data = pickle.load(file_path.open('rb')) # dict

        return user_data

    def _get_sampledata_folder_path(self):
        return Path(self.sampledata_folder_path)

    def sample_dpg_data(self):
        m = self.sample_method()
        print(m)
        if 'wu' == m:
            pass
        elif 'm_common' == m:
            pass
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


class DPG_Dec19Dataset(AbstractDatasetDPG):
    def __init__(self, args):
        super(DPG_Dec19Dataset, self).__init__(args)

        self.sampledata_folder_path = "./Data/DPG_dec19/dev_time_split_wu"


    @classmethod
    def code(cls):
        return 'DPG_dec19'

    @classmethod
    def sample_method(self):
        return self.args.data_sample_method

    @classmethod
    def all_raw_file_names(cls):
        return ['items',
                'users']

    def sample_dpg_data(self):
        m = self.sample_method()
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
        u_data_prepped = {}

        for u_id in user_data.keys():

            u_id2idx[u_id] = len(u_id2idx)  # create mapping from u_id to index

            # pos_impre, time_stamps = zip(*[(art_id2idx[art_id], time_stamp) for _, art_id, time_stamp in user_data[u_id]["articles_read"]])

            if 'masked_interest' == self.args.train_method:

                hist = [(art_id2idx[art_id], time_stamp) for _, art_id, time_stamp
                            in sorted(user_data[u_id]["articles_read"], key=lambda tup: tup[2])]
                u_data_prepped[u_id2idx[u_id]] = hist

            elif 'wu' == self.args.train_method:
                #cand_article_ids = (set(news_data['all'].keys()) - news_data['test']).union(set(news_data['train']))
                pass
            elif 'pos_cut_off' == self.args.train_method:
                pass
            else:
                raise NotImplementedError()

        return u_data_prepped, u_id2idx

    def prep_dpg_news_data(self):
        news_data = self.load_raw_news_data()
        if self.args.use_article_content:
            # use article content to create contextualised representations
            # vocab, news_as_word_ids, art_id2idx = preprocess_dpg_news_file(news_file=path_article_data,
            #                                                                tokenizer=word_tokenize,
            #                                                                min_counts_for_vocab=min_counts_for_vocab,
            #                                                                max_article_len=max_article_len)
            art_id2idx = None
        else:
            art_id2idx = {'0': 0}  # dictionary news indices
            for art_id in news_data['all'].keys():
                art_id2idx[art_id] = len(art_id2idx)  # map article id to index

        return news_data, art_id2idx


class DPG_Nov19Dataset(AbstractDatasetDPG):
    def __init__(self, args):
        super(DPG_Nov19Dataset, self).__init__(args)

        self.sampledata_folder_path = "./Data/DPG_nov19/medium_time_split_wu"

    @classmethod
    def code(cls):
        return 'DPG_nov19'

    @classmethod
    def sample_method(self):
        return self.args.data_sample_method

    @classmethod
    def all_raw_file_names(cls):
        return ['items',
                'users']

    def sample_dpg_data(self):
        m = self.sample_method()
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
        u_data_prepped = {}

        for u_id in user_data.keys():

            u_id2idx[u_id] = len(u_id2idx)  # create mapping from u_id to index

            # pos_impre, time_stamps = zip(*[(art_id2idx[art_id], time_stamp) for _, art_id, time_stamp in user_data[u_id]["articles_read"]])

            if 'masked_interest' == self.args.train_method:

                hist = [(art_id2idx[art_id], time_stamp) for art_id, time_stamp
                            in sorted(user_data[u_id]["articles_read"], key=lambda tup: tup[1])]
                if np.max(hist) >= len(art_id2idx):
                    pass
                u_data_prepped[u_id2idx[u_id]] = hist

            elif 'wu' == self.args.train_method:
                #cand_article_ids = (set(news_data['all'].keys()) - news_data['test']).union(set(news_data['train']))
                pass
            elif 'pos_cut_off' == self.args.train_method:
                pass
            else:
                raise NotImplementedError()

        return u_data_prepped, u_id2idx

    def prep_dpg_news_data(self):
        news_data = self.load_raw_news_data()
        if self.use_content:
            # use article content to create contextualised representations

            # TODO: given code for News Encoder, load or compute article embeddings
            # create dataloader to feed batches of content into Encoder
            # no shuffle, in order

            # 
            #
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
                                                                pd_vocab=self.args.pd_vocab)

                self.art_embs = art_embs

        else:
            art_id2idx = {}  # {'0': 0} dictionary news indices
            for art_id in news_data['all'].keys():
                art_id2idx[art_id] = len(art_id2idx)  # map article ID -> index

        return news_data, art_id2idx


#class DatasetDPG_Content(AbstractDatasetDPG):
