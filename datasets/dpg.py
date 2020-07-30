import arrow
import pickle
import random
import numpy as np
from tqdm import tqdm

from pathlib import Path
from collections import defaultdict
from abc import *
from sklearn.preprocessing import StandardScaler

from .base import AbstractDataset
from source.preprocessing.utils_news_prep import precompute_dpg_art_emb, preprocess_dpg_news_file, prep_dpg_user_file
from source.preprocessing.get_dpg_data_sample import get_data_n_rnd_users
from source.utils import map_time_stamp_to_vector



class AbstractDatasetDPG(AbstractDataset):

    def __init__(self, args):
        super(AbstractDatasetDPG, self).__init__(args)
        self.args = args
        self.min_hist_len = self.args.min_hist_len

        self.use_content = args.use_article_content
        self.pt_news_encoder = args.pt_news_encoder

        seed = args.dataloader_random_seed
        self.rnd = random.Random(seed)

        ## time stamps ##
        self.w_time_stamp = args.incl_time_stamp
        self.parts_time_vec = ['WD', 'HH', 'mm'] if args.parts_time_vec is None else args.parts_time_vec
        args.len_time_vec = len(self.parts_time_vec)
        self.ts_scaler = None

        ## article content ##
        self.vocab = None
        self.art_id2idx = None # mapping article ID -> article index
        self.art_idx2word_ids = None
        self.art_embs = None
        self.valid_items = {'train': None, 'test': None}

        if args.dataset_path is not None:
            self.data_dir_path = args.dataset_path #"./Data/DPG_nov19/medium_time_split_n_rnd_users"
        else:
            raise ValueError("Need path to dataset folder! None was given")

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
        folder_name = '{}-{}-artl{}-histl{}-nenc_{}-time{}' \
            .format(self.code(), self.args.dataloader_code, self.args.max_article_len, self.args.max_hist_len, self.pt_news_encoder, int(self.w_time_stamp))
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
        news_data = self.prep_dpg_news_data()

        train, val, test, u_id2idx = self.prep_dpg_user_data(news_data)

        #train, val, test = self.split_data(user_data, len(u_id2idx))

        # train (dict): {user_idx: [art_idx_1, ..., art_idx_L_u] } ## w/o time
        # train (dict): {user_idx: [(art_idx_1, time_stamp_1), ..., (art_idx_L_u, time_stamp_L_u)] } ## w/ time

        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': u_id2idx,
                   'smap': self.art_id2idx,
                   'vocab': self.vocab,
                   'art2words': self.art_idx2word_ids,
                   'art_emb': self.art_embs, # article emb matrix
                   'valid_items': self.valid_items,
                   'rnd': self.rnd,
                   'ts_scaler': self.ts_scaler}

        # save
        with dataset_path.open('wb') as fout:
            pickle.dump(dataset, fout)

        if self.art_embs is not None:
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
        return Path(self.data_dir_path)

    def sample_dpg_data(self):
        m = self.sample_method
        print(m)
        if 'n_rnd_users' == m:
            raise NotImplementedError()
            # news_data, user_data, logging_dates = get_data_n_rnd_users(config.data_dir, n_users,
            #                                                            news_len=config.news_len,
            #                                                            min_hist_len=config.min_hist_len,
            #                                                            max_hist_len=config.max_hist_len,
            #                                                            min_test_len=config.min_test_len,
            #                                                            save_path=config.save_path,
            #                                                            sample_name=sample_name,
            #                                                            test_time_thresh=threshold_time,
            #                                                            overwrite_existing=config.overwrite_existing)
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

        if args.dataset_path is not None:
            self.data_dir_path = args.dataset_path #"./Data/DPG_nov19/medium_time_split_n_rnd_users"
        else:
            raise ValueError("Need path to dataset folder! None was given")

        self.n_users = args.n_users

    @classmethod
    def code(cls):
        return 'DPG_nov19'

    @classmethod
    def all_raw_file_names(cls):
        return ['items',
                'users']

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()

        if self.n_users > 1e3:
            size = "u" + str(int(self.n_users / 1e3)) + "k"
        else:
            size = "u" + str(self.n_users)

        folder_name = '{}_{}-{}-artl{}-histl{}-nenc_{}-time{}' \
            .format(self.code(), size, self.args.dataloader_code, self.args.max_article_len, self.args.max_hist_len, self.pt_news_encoder, int(self.w_time_stamp))
        return preprocessed_root.joinpath(folder_name)

    def sample_dpg_data(self):
        m = self.sample_method
        print(m)
        if 'wu' == m:
            pass
        elif 'm_common' == m:
            pass
        else:
            raise NotImplementedError()

    def prep_dpg_user_data(self, news_data):

        user_data = self.load_raw_read_histories()
        u_id2idx = {}

        # for user in range(user_count):
        #     items = [*list(zip(*user_data[user]))[0]]  # only take article ids, discard the timestamps
        #     train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
        #
        #     return train, val, test

        train, val, test = defaultdict(list), defaultdict(list), defaultdict(list)
        all_ts = defaultdict(list)

        self.valid_items = dict()
        if 'train' in news_data:
            self.valid_items['train'] = set([self.art_id2idx[a] for a in set(news_data['train'])])
            self.valid_items['test'] = set([self.art_id2idx[a] for a in set(news_data['test'])])
            needs_update = False
        else:
            self.valid_items['train'] = set()
            self.valid_items['test'] = set()
            needs_update = True

        print("> Prepping user data ..")
        u_keys = list(user_data.keys())

        for u_id in tqdm(u_keys):

            # get train & test dat
            if 'time_threshold' == self.args.split:
                train_items, test_items = self.get_train_test_items_from_data(user_data[u_id])
                # train_items (list): [(art_idx_1, ts_1), ..., (art_idx_N, ts_N)]
                # list of tuples consisting of article indices (already mapped) and (converted) time_stamps

                if train_items is None:
                    print("Reading history of user {} was too short and is excluded".format(u_id))
                    continue
            else:
                raise NotImplementedError()

            if 'cloze' == self.args.train_method:

                u_id2idx[u_id] = u_idx = len(u_id2idx)  # create mapping from u_id to index

                if self.w_time_stamp:
                    train[u_idx] = train_items
                    test[u_idx] = test_items
                    all_ts['train'].extend([*list(zip(*train_items))[1]]) # ts: [DD, HH, mm, ss]
                    all_ts['test'].extend([*list(zip(*test_items))[1]])
                else:
                    train[u_idx] = [*list(zip(*train_items))[0]]
                    test[u_idx] = [*list(zip(*test_items))[0]]

                # add to validation sample
                val_items, test_items = self.select_rnd_item_for_validation(test[u_idx])

                val[u_idx] = train[u_idx] + val_items
                test[u_idx] = val[u_idx] + test_items

            elif 'npa' == self.args.train_method:
                # each instance consists of [u_id, hist, target]
                #train(dict(list)): {u_idx: [[[hist1], target1],..[[histN], targetN]]}
                #test(dict(list)): {u_idx: [[hist], [targets]]}
                # candidates will be sampled in Dataloader and stored
                # retrieve candidates based on user & target

                u_id2idx[u_id] = u_idx = len(u_id2idx)  # create mapping from u_id to index

                train_arts, train_ts = zip(*train_items)

                if self.w_time_stamp:
                    all_ts['train'].extend(train_ts)  # ts: [DD, HH, mm, ss]

                # create instance for each train impression
                for art_id, ts in train_items:
                    # construct target-specific history
                    target = art_id
                    # TODO: extend method to use incorporate time stamps
                    # subsample unordered, random history for this target
                    item_set = [a for a in train_arts if a != target]
                    hist = random.sample(item_set, min(self.args.max_hist_len, len(item_set)))[:self.args.max_hist_len]

                    # add train instance
                    train[u_idx].append((hist, target))

                # same history for all test cases
                test_hist = random.sample(train_arts, min(self.args.max_hist_len, len(train_arts)))[:self.args.max_hist_len]

                val_targets, test_targets = [], []
                for art_id, ts in test_items:
                    # randomly sample validation instances
                    coin_flip = self.rnd.random()
                    if coin_flip > (1 - self.args.validation_portion):
                        val_targets.append(art_id)
                    else:
                        test_targets.append(art_id)

                if len(test_targets) < 1 or len(val_targets) > len(test_targets):
                    test_targets.append(val_targets.pop())

                test[u_idx] = [test_hist, test_targets]
                val[u_idx] = [test_hist, val_targets]

            else:
                raise NotImplementedError()

            if needs_update:
                self.update_valid_items(train_items, key='train')
                self.update_valid_items(test_items, key='test')

        if self.w_time_stamp:
            ## map & transform all time stamps

            if self.args.normalise_time_stamps is not None:
                # fit scaler
                if 'standard' == self.args.normalise_time_stamps:

                    self.ts_scaler = StandardScaler()
                    print("> Fitting scaler ..")
                    self.ts_scaler.fit(np.array(all_ts['train']))
                else:
                    raise NotImplementedError(self.args.normalise_time_stamps)

                # transform data
                print("> Scaling time stamps ..")
                for u_id, seq in train.items():
                    articles, ts = zip(*seq)
                    train[u_id] = list(zip(articles, self.ts_scaler.transform(np.array(ts)).tolist()))

                for u_id, seq in test.items():
                    articles, ts = zip(*seq)
                    test[u_id] = list(zip(articles, self.ts_scaler.transform(np.array(ts)).tolist()))

                for u_id, seq in val.items():
                    if len(seq) > 0:
                        articles, ts = zip(*seq)
                        val[u_id] = list(zip(articles, self.ts_scaler.transform(np.array(ts)).tolist()))

        return train, val, test, u_id2idx

    def select_rnd_item_for_validation(self, test_items):
        # select random portion of test items as subset for validation
        # exclude possibility to use last test interaction for validation because it's reserved for testing
        val_pos = self.rnd.choice(range(len(test_items) - 1)) if len(test_items) > 1 else None
        #coin_flip = self.rnd.random()
        if val_pos is not None:
            return test_items[:val_pos], test_items[val_pos:]
        else:
            raise ValueError("Number of test items insufficient to creat validation instance")

        # is not None and coin_flip > (1 - self.args.validation_portion):
        # else:
        #     return [], test_items

    def get_train_test_items_from_data(self, u_data):
        # split user interactions according to certain threshold timestamp
        # e.g. first 3 weeks for training, last week for testing
        try:
            threshold_date = arrow.get(self.args.time_threshold, "DD-MM-YYYY HH:mm:ss").timestamp
            # 1574639999
        except:
            raise ValueError("Threshold date must string of format: 'DD-MM-YYYY HH:mm:ss'")

        # check if data has already been separated into train & test
        if 'articles_train' in u_data.keys():
            train_items = [(self.art_id2idx[art_id], map_time_stamp_to_vector(ts, rel_parts=self.parts_time_vec)) for art_id, ts
                           in sorted(u_data['articles_train'], key=lambda tup: tup[1])
                           if art_id in self.art_id2idx]

            test_items = [(self.art_id2idx[art_id], map_time_stamp_to_vector(ts, rel_parts=self.parts_time_vec)) for art_id, ts
                          in sorted(u_data['articles_test'], key=lambda tup: tup[1])
                          if art_id in self.art_id2idx]

        else:
            # split user history according to time stamps. usually this is done in sampling process earlier
            full_hist = [(self.art_id2idx[art_id], time_stamp) for art_id, time_stamp
                         in sorted(u_data['articles_read'], key=lambda tup: tup[1])]
            train_items = []

            for i, (item, ts) in enumerate(full_hist):
                if ts < threshold_date:
                    train_items.append((item, map_time_stamp_to_vector(ts, rel_parts=self.parts_time_vec)) if self.w_time_stamp else item)
                else:
                    test_items = [(item, map_time_stamp_to_vector(ts, rel_parts=self.parts_time_vec)) for item, ts in full_hist[i:]]
                    break

        if len(test_items) < 1 or len(train_items) < 2:
            return None, None
        else:
            return train_items, test_items

    def update_valid_items(self, items, key='train'):
        if key not in self.valid_items:
            raise ValueError("Invalid key passed: {}".format(key))

        self.valid_items[key].update([*list(zip(*items))[0]])

    def prep_dpg_news_data(self):
        news_data = self.load_raw_news_data()
        if self.use_content:
            # use article content to create contextualised representations

            # check for existing
            if self.args.pt_art_emb_path is not None:
                raise NotImplementedError()

            if self.args.pt_news_encoder is None:
                # prepare news articles to create article embeddings end-to-end
                vocab, news_as_word_ids, art_id2idx = preprocess_dpg_news_file(news_file=news_data,
                                                                               language=self.args.language,
                                                                               min_counts_for_vocab=self.args.min_counts_for_vocab,
                                                                               max_article_len=self.args.max_article_len,
                                                                               max_vocab_size=self.args.max_vocab_size,
                                                                               lower_case=self.args.lower_case)
                self.art_idx2word_ids = news_as_word_ids
                self.vocab = vocab

            else:
                # use pre-trained news encoder to get article embeddings
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

        self.art_id2idx = art_id2idx

        return news_data


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
