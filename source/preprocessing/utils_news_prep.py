import argparse
import json
import csv
import itertools
import random
import os
import nltk
from nltk.tokenize import word_tokenize
#nltk.download('punkt')
from collections import defaultdict, Counter, OrderedDict
from pathlib import Path
import numpy as np
import pickle

import torch
import torch.nn as nn

import fasttext

from source.preprocessing.bert_feature_extractor import BertFeatureExtractor
from source.utils import get_art_id_from_dpg_history, build_vocab_from_word_counts, pad_sequence, reverse_mapping_dict
from source.modules import NEWS_ENCODER
from sklearn.model_selection import train_test_split
import arrow


DATA_TYPES = ["NPA", "DPG", "Adressa"]

FILE_NAMES_NPA = {"click": "ClickData_sample.tsv", "news": "DocMeta_sample.tsv"}



def sample_n_from_elements(elements, ratio):
    '''

    :param elements: collection of elements from which to sample, e.g. article id for neg impressions
    :param ratio: total number of samples to be returned
    :return: random sample of size 'ratio' from the given collection 'nnn'
    '''
    if ratio > len(elements):
        return random.sample(elements * (ratio // len(elements) + 1), ratio) # expand sequence with duplicates so that we can sample enough elems
    else:
        return random.sample(elements, ratio)

def determine_n_samples(hist_len, n_max=20, max_hist_len=50, scale=0.08):

    #determine number of (training) instance depending on the history length
    if scale is None:
        scale = np.log(n_max) / max_hist_len
    n_samples = min(n_max, round(np.exp(hist_len * scale)))

    return int(n_samples)

def generate_target_hist_instance_pos_cutoff(u_id, history, test_impressions, candidate_art_ids, art_id2idx, max_hist_len, min_hist_len=5, candidate_generation='neg_sampling', neg_sample_ratio=4, mode="train"):
    '''
    Generate samples with target-history tuples at different positions

    :param u_id: user id mapped to user index
    :param history: sequence of article indices (article ids already mapped to indices)
    :param candidate_art_ids: set of candidate article ids
    :param art_id2idx: dictionary mapping article ids to indices
    :param max_hist_len:
    :param min_hist_len:
    :param candidate_generation:
    :param neg_sample_ratio:
    :return:
    '''
    samples = []

    if "train" == mode:
        n_samples = determine_n_samples(len(history))

        for i in range(n_samples):
            if i == 0:
                target_idx = len(history)-1
            else:
                target_idx = random.choice(range(min_hist_len, len(history)))
            # generate target history instance
            target = history[target_idx]
            hist = pad_sequence(history[:target_idx], max_hist_len, pad_value=0)
            # candidate generation
            cands, lbls = generate_candidates_train(target, candidate_art_ids, art_id2idx, neg_sample_ratio)
            samples.append((u_id, hist, cands, lbls))

    elif "test" == mode:

        for test_art in test_impressions:
            target = test_art
            hist = pad_sequence(history, max_hist_len)
            # candidate generation
            cands, lbls = generate_candidates_train(target, candidate_art_ids, art_id2idx, neg_sample_ratio)
            samples.append((u_id, hist, cands, lbls))

            history.append(target) # history is extended by the new impression to predict the next one
    else:
        raise KeyError()

    return samples

def generate_candidates_train(target, cand_article_ids, art_id2idx, neg_sample_ratio, candidate_generation='neg_sampling', constrained_target_time=False):
    '''

    :param target: target article as index
    :param cand_article_ids: set of article ids as valid candidates
    :param art_id2idx: dictionary mapping article id to index
    :param neg_sample_ratio:
    :param candidate_generation: indicate method of candidate generation

    :return:
    '''

    if candidate_generation == 'neg_sampling':
        candidates = [art_id2idx[art_id] for art_id in sample_n_from_elements(cand_article_ids, neg_sample_ratio)]
        candidates.append(target)
        lbls = [0] * neg_sample_ratio + [1]  # create temp labels
        candidates = list(zip(candidates, lbls))  # zip art_id and label
        random.shuffle(candidates)  # shuffle article ids with corresponding label

        candidates, lbls = zip(*candidates)
    else:
        raise NotImplementedError()

    return candidates, lbls

def add_instance_to_data(data, u_id, hist, cands, lbls):
    data.append({'input': (np.array(u_id, dtype='int32'), np.array(hist, dtype='int32'), np.array(cands, dtype='int32')),
                 'labels': np.array(lbls, dtype='int32')})

def get_hyper_model_params(config):

    hyper_params = {'random_seed': config.random_seed,
                    'lr': config.lr, 'neg_sample_ratio': config.neg_sample_ratio,
                    'batch_size': config.batch_size, 'lambda_l2': config.lambda_l2,
                    'train_act_func': config.train_act_func, 'test_act_func': config.test_act_func,
                    'n_epochs': config.n_epochs, 'data_type': config.data_type,
                    'data_path': config.data_path
                    }

    if "vanilla" == config.npa_variant:
        model_params = {'dim_user_id': 50, 'dim_pref_query': 200, 'dim_words': 300,
                        'max_news_len': 30, 'max_hist_len': 50}
    else:
        model_params = {'dim_user_id': 50, 'dim_pref_query': 200, 'dim_words': 300,
                        'max_title_len': config.max_hist_len, 'max_news_len': config.max_news_len}

    return hyper_params, model_params


def get_art_ids_from_read_hist(raw_articles_read, art_id2idx):
    if len(raw_articles_read):

        if len(raw_articles_read[0]) == 3:
            art_ids, time_stamps = zip(*[(art_id2idx[art_id], time_stamp)
                                           for _, art_id, time_stamp in raw_articles_read])
        elif len(raw_articles_read[0]) == 2:
            art_ids, time_stamps = zip(*[(art_id2idx[art_id], time_stamp)
                                           for art_id, time_stamp in raw_articles_read])
        else:
            raise NotImplementedError()

        return list(art_ids), list(time_stamps)
    else:
        return [], []

def prep_dpg_user_file(user_file, news_file, art_id2idx, train_method, test_interval_days : int, neg_sample_ratio=4, max_hist_len=50, preserve_seq=False):
    '''
    Given a subsample of users and the valid articles, we truncate and encode users' reading history with article indices.
    Construct the data input for the Wu NPA as follows:
     'input': (u_id, history, candidates), 'labels': lbls

    Apply negative sampling to produce 'candidate' list of positive and negative elements

    :param user_file: (str) path pointing to pickled subsample of user data
    :param article_ids: set containing all valid article ids
    :param art_id2idx: (dict) mapping from article id to index
    :param neg_sample_ratio: int ratio for number of negative samples
    :param max_hist_len: (int) maximum length of a user's history, for truncating
    :param test_interval: (int) number of days specifying the time interval for the testing set
    :return:
    '''

    with open(user_file, "rb") as fin:
        user_data = pickle.load(fin)

    with open(news_file, "rb") as fin:
        news_data = pickle.load(fin)

    log_file = "/".join(user_file.split("/")[:-1]) + '/logging_dates.json'
    with open(log_file, 'r') as fin:
        logging_dates = json.load(fin)

    # determine start of the test interval in UNIX time
    if "threshold" in logging_dates.keys():
        start_test_interval = logging_dates['threshold']
    elif test_interval_days is not None:
        start_test_interval = logging_dates['last'] - ((60*60*24) * test_interval_days)
    else:
        raise NotImplementedError()

    u_id2idx = {}
    data = defaultdict(list)

    '''
    - For each user, we produce N training samples (N := # pos impressions) => all users have at least 5 articles
    - test samples is constrained by time (last week)
    
    Notes from Wu: 
    - testing: for each user, subsample a user history (same for all test candidates)
    - take each positive impression from the test week and create a test instance with history and user_id
    - take each 'neg' impression as well
    - during inference, the forward pass processes 1 candidate at a time
    - the evaluation metrics are computed per user with the individual results for each testing instances following a session-index
    
    '''

    if len(news_data) > 1:
        articles_train = news_data['train'] # articles appearing in the training interval
        articles_test = news_data['test'] # collect all articles appearing in the test interval
    else:
        articles_train = set()
        articles_test = set()

    for u_id in user_data.keys():

        u_id2idx[u_id] = len(u_id2idx) # create mapping from u_id to index

        # aggregate article ids from user data
        # divide into training & test samples
        # constraint train set by time

        w_time_stamp = False
        pos_impre, time_stamps = get_art_ids_from_read_hist(user_data[u_id]['articles_train'], art_id2idx)

        if "articles_train" in user_data[u_id].keys():
            #[f(x) if condition else g(x) for x in sequence]

            art_ids, time_stamps = get_art_ids_from_read_hist(user_data[u_id]['articles_train'], art_id2idx)
            train_impres = list(zip(art_ids, time_stamps)) if w_time_stamp else art_ids

            art_ids, time_stamps = get_art_ids_from_read_hist(user_data[u_id]['articles_test'], art_id2idx)
            test_impres = list(zip(art_ids, time_stamps)) if w_time_stamp else art_ids

            # train_impres = [(art_id2idx[art_id], time_stamp) if w_time_stamp else art_id2idx[art_id]
            #                 for _, art_id, time_stamp in user_data[u_id]['articles_train']] # (id, time)
            # test_impres = [(art_id2idx[art_id], time_stamp) if w_time_stamp else art_id2idx[art_id]
            #                 for _, art_id, time_stamp in user_data[u_id]['articles_test']]
        else:
            train_impres, test_impres = [], []

            for impression, time_stamp in zip(pos_impre, time_stamps):
                if time_stamp < start_test_interval:
                    train_impres.append((impression, time_stamp) if w_time_stamp else impression)
                else:
                    test_impres.append((impression, time_stamp) if w_time_stamp else impression)

        # assumption: for testing use the entire reading history of that user but evaluate on unseen articles

        cand_article_ids = (set(news_data['all'].keys()) - news_data['test']).union(set(news_data['train']))

        if 'wu' == train_method:
            #########################################
            # Wu Sub-Sampling of random histories
            #########################################
            for pos_sample in train_impres:
                #
                # (u_id, hist, cands, lbls)
                #
                # Candidate Generation: generate negative samples
                candidate_articles = [art_id2idx[art_id] for art_id in
                                      sample_n_from_elements(cand_article_ids, neg_sample_ratio)]
                candidate_articles.append(pos_sample)
                lbls = [0] * neg_sample_ratio + [1]  # create temp labels
                candidate_articles = list(zip(candidate_articles, lbls))  # zip art_id and label
                random.shuffle(candidate_articles)  # shuffle article ids with corresponding label
                candidate_articles = np.array(candidate_articles)

                # sample RANDOM elems from set of pos impressions -> Note that this is the orig. NPA approach => Sequence Order is lost
                pos_set = list(set(pos_impre) - set([pos_sample]))  # remove positive sample from user history
                hist = [int(p) for p in random.sample(pos_set, min(max_hist_len, len(pos_set)))[:max_hist_len]]
                hist += [0] * (max_hist_len - len(hist))
                add_instance_to_data(data['train'], u_id2idx[u_id], hist, candidate_articles[:, 0], candidate_articles[:, 1])

            # create test instances
            if len(test_impres) != 0:
                # subsample history: RANDOM elems from set of pos impressions
                pos_set = pos_impre
                hist_test = [int(p) for p in random.sample(pos_set, min(max_hist_len, len(pos_set)))[:max_hist_len]]
                hist_test += [0] * (max_hist_len - len(hist_test))

                if w_time_stamp:
                    test_ids = set(list(zip(*test_impres))[0])
                else:
                    test_ids = set(test_impres)

                # Assumption: for testing, use all articles (train + test interval) as potential candidates
                # but exclude user-specific test impressions
                cand_article_ids = set(news_data['all'].keys()) - set(test_impres)

                for pos_test_sample in test_impres:
                    '''              
                    - Remove pos impressions from pool of cand articles
                    - Sample negative examples
                    - Add labels
                    - Shuffle zip(cands, lbls)            
                    - add samples to data dict            
                    '''
                    # generate test candidates
                    ## sample candidates from articles in test interval
                    candidate_articles = [art_id2idx[art_id] for art_id in
                                          sample_n_from_elements(cand_article_ids, neg_sample_ratio)]
                    candidate_articles.append(pos_test_sample)
                    lbls = [0] * neg_sample_ratio + [1]  # create temp labels
                    candidate_articles = list(zip(candidate_articles, lbls))  # zip art_id and label
                    # random.shuffle(candidate_articles)  # shuffle article ids with corresponding label
                    candidate_articles = np.array(candidate_articles)

                    # add data instance
                    add_instance_to_data(data['test'], u_id2idx[u_id], hist_test, candidate_articles[:, 0], candidate_articles[:, 1])

        elif 'pos_cut_off' == train_method:

            # candidate articles for training: exlcude those from testing interval
            #cand_article_ids = set(news_data['all'].keys()) - set(news_data['test']) + set(news_data['train'])

            u_id = u_id2idx[u_id]

            train_samples = generate_target_hist_instance_pos_cutoff(u_id, train_impres, None,
                                                                        cand_article_ids,
                                                                        art_id2idx, max_hist_len,
                                                                        min_hist_len=5, mode="train")
            # add train instances to data
            for (u_id, hist, cands, lbls) in train_samples:
                add_instance_to_data(data['train'], u_id, hist, cands, lbls)

            if len(test_impres) != 0:
                cand_article_ids = set(news_data['all'].keys()) - set(test_impres)

                test_samples = generate_target_hist_instance_pos_cutoff(u_id, train_impres, test_impres,
                                                                            cand_article_ids,
                                                                            art_id2idx, max_hist_len,
                                                                            min_hist_len=5, mode="test")
                # add test instance to data
                for (u_id, hist, cands, lbls) in test_samples:
                    add_instance_to_data(data['test'], u_id, hist, cands, lbls)

        elif 'masked_interests' == train_method:
            raise NotImplementedError()
        else:
            raise KeyError()

    #
    #reformat to np int arrays
    # candidates['train'] = np.array(candidates['train'], dtype='int32')
    # labels['train'] = np.array(labels['train'], dtype='int32')
    # user_ids['train'] = np.array(user_ids['train'], dtype='int32')
    # user_hist_pos['train'] = np.array(user_hist_pos['train'], dtype='int32')
    #
    # candidates['test'] = np.array(candidates['test'], dtype='int32')
    # labels['test'] = np.array(labels['test'], dtype='int32')
    # user_ids['test'] = np.array(user_ids['test'], dtype='int32')
    # user_hist_pos['test'] = np.array(user_hist_pos['test'], dtype='int32')
    #
    #
    # data['train'] = [{'input': (u_id, hist, cands), 'labels': np.array(lbls)} for u_id, hist, cands, lbls
    #                 in zip(user_ids['train'], user_hist_pos['train'], candidates['train'], labels['train'])]
    #
    # data['test'] = [{'input': (u_id, hist, cands), 'labels': np.array(lbls)} for u_id, hist, cands, lbls
    #                 in zip(user_ids['test'], user_hist_pos['test'], candidates['test'], labels['test'])]

    print("Train samples: {} \t Test: {}".format(len(data['train']), len(data['test'])))

    return u_id2idx, data

def preprocess_dpg_news_file(news_file, tokenizer, min_counts_for_vocab=2, max_article_len=30, max_vocab_size=30000):

    if isinstance(news_file, str):
        with open(news_file, 'rb') as f:
            news_data = pickle.load(f)
    elif isinstance(news_file, dict):
        news_data = news_file
    else:
        raise NotImplementedError()

    article_ids = news_data['all']

    vocab = defaultdict(int)
    news_as_word_ids = []
    art_id2idx = {}

    # 1. construct raw vocab
    print("construct raw vocabulary ...")
    vocab_raw = Counter({'PAD': 999999})

    for art_id in article_ids:
        tokens = tokenizer(article_ids[art_id]["snippet"].lower(), language='dutch')
        vocab_raw.update(tokens)
        article_ids[art_id]['tokens'] = tokens

        if len(vocab_raw) % 1e4 == 0:
            print(len(vocab_raw))

    # 2. construct working vocab
    print("construct working vocabulary ...")
    vocab = build_vocab_from_word_counts(vocab_raw, max_vocab_size, min_counts_for_vocab)
    print("Vocab: {}  Raw: {}".format(len(vocab), len(vocab_raw)))
    #del(vocab_raw)

    # 3. encode news as sequence of word_ids
    print("encode news as word_ids ...")
    news_as_word_ids = {} # {'0': [0] * max_article_len}
    art_id2idx = {}  # {'0': 0}

    for art_id in article_ids:
        word_ids = []

        art_id2idx[art_id] = len(art_id2idx) # map article id to index

        # get word_ids from news title
        for word in article_ids[art_id]['tokens']:
            # if word occurs in vocabulary, add the id
            # unknown words are omitted
            if word in vocab:
                word_ids.append(vocab[word])

        # pad & truncate sequence
        news_as_word_ids[art_id] = pad_sequence(word_ids, max_article_len)
        #news_as_word_ids.append(pad_sequence(word_ids, max_article_len))

    # reformat as array
    # news_as_word_ids = np.array(news_as_word_ids, dtype='int32')

    return vocab, news_as_word_ids, art_id2idx

def get_word_embs_from_pretrained_ft(vocab, emb_path, emb_dim=300):
    try:
        ft = fasttext.load_model(emb_path) # load pretrained vectors

        # check & adjust dimensionality
        if ft.get_dimension() != emb_dim:
            fasttext.util.reduce_model(ft, emb_dim)

        embedding_matrix = [0] * len(vocab)
        embedding_matrix[0] = np.zeros(emb_dim, dtype='float32')  # placeholder with zero values for 'PAD'

        for word, idx in vocab.items():
            embedding_matrix[idx] = ft[word] # how to deal with unknown words?

        return np.array(embedding_matrix, dtype='float32')

    except:
        print("Could not load word embeddings")
        return None

def generate_batch_data_test(all_test_pn, all_label, all_test_id, batch_size, all_test_user_pos, news_words):
    inputid = np.arange(len(all_label))
    y = all_label
    batches = [inputid[range(batch_size * i, min(len(y), batch_size * (i + 1)))] for i in
               range(len(y) // batch_size + 1)]

    while (True):
        for i in batches:
            candidate = news_words[all_test_pn[i]]
            browsed_news = news_words[all_test_user_pos[i]]
            browsed_news_split = [browsed_news[:, k, :] for k in range(browsed_news.shape[1])]
            userid = np.expand_dims(all_test_id[i], axis=1)
            label = all_label[i]

            yield ([candidate] + browsed_news_split + [userid], label)

def gen_batch_data_test(data, news_as_word_ids, batch_size=100, candidate_pos=0):
    n_batches = range(len(data) // batch_size + 1)

    batches = [(batch_size * i, min(len(data), batch_size * (i + 1))) for i in
               n_batches]

    for start, stop in batches:
        # get data for this batch
        users, hist, cands, labels = zip(*[(data_p['input'][0], news_as_word_ids[data_p['input'][1]],
                                            news_as_word_ids[data_p['input'][2]], data_p['labels'])
                                                for data_p in data[start:stop]]) #return multiple lists from list comprehension

        # get candidates
        candidates = np.array(cands)  # shape: batch_size X n_candidates X title_len
        candidate = candidates[:, candidate_pos, :]  # candidate.shape := (batch_size, max_title_len)

        # get history
        hist = np.array(hist)  # shape: batch_size X max_hist_len X max_title_len
        history_split = [hist[:, k, :] for k in range(hist.shape[1])]  # shape := (batch_size, max_title_len)

        # get user ids
        user_ids = np.expand_dims(np.array(users), axis=1)

        # get labels
        labels = np.array(labels)
        labels = labels[:, candidate_pos]

    yield ([candidate] + history_split + [user_ids], labels)

def get_labels_from_data(data):
    labels = {}
    for entry_dict in data: #columns = ["u_id", "history", "candidates", "labels"]
        labels[entry_dict['u_id']] = entry_dict['labels']
    return labels

def get_dpg_data_processed(data_path, train_method, neg_sample_ratio=4, max_hist_len=50, max_article_len=30, min_counts_for_vocab=2, load_prepped=False):

    news_path = data_path + "news_prepped_" + train_method + ".pkl"
    prepped_path = data_path + "data_prepped_" + train_method + ".pkl"

    if load_prepped:
        try:
            with open(news_path, 'rb') as fin:
                (vocab, news_as_word_ids, art_id2idx) = pickle.load(fin)

            with open(prepped_path, 'rb') as fin:
                u_id2idx, data = pickle.load(fin)

            return data, vocab, news_as_word_ids, art_id2idx, u_id2idx
        except:
            print("Could not load preprocessed files! Continuing preprocessing now..")

    path_article_data = data_path + "news_data.pkl"

    vocab, news_as_word_ids, art_id2idx = preprocess_dpg_news_file(news_file=path_article_data,
                                                                   tokenizer=word_tokenize,
                                                                   min_counts_for_vocab=min_counts_for_vocab,
                                                                   max_article_len=max_article_len)

    with open(news_path, 'wb') as fout:
        pickle.dump((vocab, news_as_word_ids, art_id2idx), fout)

    path_user_data = data_path + "user_data.pkl"

    u_id2idx, data = prep_dpg_user_file(path_user_data, path_article_data, art_id2idx, train_method,
                                        test_interval_days=7, neg_sample_ratio=neg_sample_ratio, max_hist_len=max_hist_len)

    with open(prepped_path, 'wb') as fout:
        pickle.dump((u_id2idx, data), fout)

    return data, vocab, news_as_word_ids, art_id2idx, u_id2idx

def precompute_dpg_art_emb(news_data: dict, news_encoder_code: str, max_article_len: int, art_emb_dim: int,
                               lower_case=False, pd_vocab=False, path_to_pt_model=None, feature_method=None):

    if isinstance(news_data, str):
        with open(news_data, 'rb') as f:
            news_data = pickle.load(f)
    elif isinstance(news_data, dict):
        news_data = news_data
    else:
        raise NotImplementedError()

    all_articles = news_data['all']


    # initialise News Encoder
    # if news_encoder_code not in NEWS_ENCODER:
    #     raise ValueError()
    #
    # news_encoder = NEWS_ENCODER[news_encoder_code]
    if 'rnd' == news_encoder_code:
        news_encoder = NEWS_ENCODER[news_encoder_code](art_emb_dim)
    elif 'BERTje' == news_encoder_code:

        bert_export_path = Path("/".join(['.', 'pc_article_embeddings', news_encoder_code]))
        if not bert_export_path.is_dir():
            os.makedirs(bert_export_path)

        if feature_method is None:
            print('No BERTje method specified, using default "last_cls"')
            methods = [('last_cls', 0), ('sum_last_n', 4)]
            feature_method = methods[1]
        else:
            if not isinstance(feature_method, tuple):
                raise ValueError("Feature method should be tuple (method, N). If 'N' does not apply, pass 0")
            methods = [feature_method]
            #raise ValueError('specify method to extract BERTje features!')

        # if feature_method not in methods:
        #     methods.append(feature_method)

        ## check for existing
        print(feature_method)
        m, n = feature_method
        file_name = get_emb_file_name(m, n, max_article_len, lower_case)
        if bert_export_path.joinpath(file_name + ".pkl").is_file():
            with bert_export_path.joinpath(file_name + ".pkl").open('rb') as fin:
                bert_embeddings = pickle.load(fin)

            print("Found pre-computed embeddings and will use these!")
            art_id2idx, art_emb_matrix = select_rel_embs_and_stack(all_articles, bert_embeddings)

            return art_id2idx, art_emb_matrix

        bert_feat_extractor = BertFeatureExtractor(path_to_pt_model, lower_case)
        print("encode news articles ...")
        #subset = {k: all_articles[k] for k in list(all_articles.keys())[:100]}
        bert_embeddings = bert_feat_extractor.encode_text_to_features_batches(
                                            all_articles, methods,
                                            10, max_article_len)

        # save pre computed bert embeddings
        for i, (m, n) in enumerate(methods):
            file_name = get_emb_file_name(m, n, max_article_len, lower_case)

            with bert_export_path.joinpath(file_name + ".pkl").open('wb') as fout:
                pickle.dump(bert_embeddings[i], fout)

        art_id2idx, art_emb_matrix = select_rel_embs_and_stack(all_articles, bert_embeddings[methods.index(feature_method)])

        return art_id2idx, art_emb_matrix

    else:
        raise NotImplementedError()

    # 2. construct working vocab
    # check for existing, pre-defined vocab
    if pd_vocab:
        # vocab = news_encoder.vocabulary
        pass
    else:
        print("construct working vocabulary ...")
        raise NotImplementedError()
        # vocab = build_vocab_from_word_counts(vocab_raw, max_vocab_size, min_counts_for_vocab)
        # print("Vocab: {}  Raw: {}".format(len(vocab), len(vocab_raw)))
        # del(vocab_raw)

    # 3.
    print("encode news articles ...")
    art_id2idx = OrderedDict()
    art_embs = []
    for art_id in all_articles:
        art_id2idx[art_id] = len(art_id2idx)  # map article id to index
        tokens = None

        art_embs.append(news_encoder(tokens))

    # reformat as matrix
    # (n_items x dim_art_emb)
    art_emb_matrix = torch.stack(art_embs, dim=0)

    return art_id2idx, art_emb_matrix


def select_rel_embs_and_stack(articles, embedding_dict):
    # select only relevant ones for embedding matrix
    art_embs = []
    art_id2idx = OrderedDict()
    for art_id in articles:
        art_id2idx[art_id] = len(art_id2idx)
        art_embs.append(embedding_dict[art_id])

    # reformat as matrix
    # (n_items x dim_art_emb)
    art_emb_matrix = torch.stack(art_embs, dim=0)
    return art_id2idx, art_emb_matrix


def get_emb_file_name(method: str, n, max_len: int, lower_case: bool):
    n = 0 if n is None else n
    return "_".join([method, "n%i" % n, "max-len%i" % max_len,
                            'lower%i' % int(lower_case)])
