import itertools
from typing import List, Dict, Tuple, Any

import numpy as np

from dataloaders.base import AbstractDataloader
# from dataloaders.news import BertTrainDatasetNews, BertEvalDatasetNews
from dataloaders.negative_samplers import negative_sampler_factory
from source.utils import check_all_equal, map_time_stamp_to_vector

import torch
import torch.utils.data as data_utils


class BertDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        self.dataset = dataset

        args.num_items = len(self.item_id2idx)
        self.max_hist_len = args.max_hist_len
        self.mask_prob = args.bert_mask_prob

        self.mask_token = self.item_count + 1
        args.bert_mask_token = self.mask_token

        self.split_method = args.split
        self.multiple_eval_items = args.split == "time_threshold"

        data = dataset.load_dataset()
        if 'valid_items' in data:
            self.valid_items = data['valid_items']
            # list(data['valid_items'])
        else:
            self.valid_items = self.get_valid_items()

        ####################
        # Negative Sampling

        self.train_neg_sampler = self.get_negative_sampler("train",
                                                           args.train_negative_sampler_code,
                                                           args.train_negative_sample_size,
                                                           args.train_negative_sampling_seed,
                                                           self.valid_items['train'],
                                                           self.get_seq_lengths(self.train))

        self.train_negative_samples = self.train_neg_sampler.get_negative_samples()

        self.val_neg_sampler = self.get_negative_sampler("val",
                                                          args.test_negative_sampler_code,
                                                          args.test_negative_sample_size,
                                                          args.test_negative_sampling_seed,
                                                          self.valid_items['test'],
                                                          None)

        self.val_negative_samples = self.val_neg_sampler.get_negative_samples()



        self.test_neg_sampler = self.get_negative_sampler("test",
                                                          args.test_negative_sampler_code,
                                                          args.test_negative_sample_size,
                                                          args.test_negative_sampling_seed,
                                                          self.valid_items['test'],
                                                          None)

        self.test_negative_samples = self.test_neg_sampler.get_negative_samples()

    @classmethod
    def code(cls):
        return 'bert'

    def _get_data(self) -> Dict[str, Any]:
        data = self.dataset.load_dataset()
        return data

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = BertTrainDataset(self.train, self.max_hist_len, self.mask_prob, self.mask_token, self.item_count,
                                   self.rnd)  # , self.art_idx2word_ids
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
        dataset = BertEvalDataset(self.train, targets, self.max_hist_len, self.mask_token, self.test_negative_samples,
                                  self.rnd, multiple_eval_items=self.multiple_eval_items)
        return dataset

    # def get_negative_sampler(self, mode, code, neg_sample_size, seed, item_set, seq_lengths):
    #     # sample negative instances for each user
    #     """
    #     param: seq_lengths (dict): how many separate neg samples do we need for this user?
    #         E.g. if seq_length[u_id] = 20, we generate 'neg_sample_size' samples for each of the 20 sequence positions
    #     """
    #     if False:
    #         # use time-sensitive set for neg sampling
    #         raise NotImplementedError()
    #     else:
    #         # use item set for simple neg sampling
    #         negative_sampler = negative_sampler_factory(code=code, train_method=self.args.train_method,
    #                                                     mode=mode, train=self.train, val=self.val, test=self.test,
    #                                                     n_users=self.user_count, valid_items=item_set,
    #                                                     sample_size=neg_sample_size, seq_lengths=seq_lengths, seed=seed,
    #                                                     save_folder=self.save_folder, id2idx=self.item_id2idx,
    #                                                     id2info=self.item_id2info
    #                                                     )
    #     return negative_sampler

    def get_valid_items(self):
        all_items = set(self.item_id2idx.values())  # train + test + val

        if self.split_method != "time_threshold":
            test = train = all_items
        else:
            train = set(itertools.chain.from_iterable(self.train.values()))
            test = all_items

        return {'train': list(train), 'test': list(test)}

    def get_seq_lengths(self, data_dict):
        # determine sequence length for each entry
        seq_lengths = list(map(len, data_dict.values()))
        if check_all_equal(seq_lengths):
            return None
        else:
            return seq_lengths


class BertDataloaderNews(BertDataloader):
    def __init__(self, args, dataset):
        self.w_time_stamps = args.incl_time_stamp
        self.w_u_id = args.incl_u_id

        super(BertDataloaderNews, self).__init__(args, dataset)

        data = dataset.load_dataset()
        self.vocab = data['vocab']
        if self.vocab is not None:
            args.vocab_path = dataset._get_preprocessed_dataset_path()

        self.art_index2word_ids = data['art2words']  # art ID -> [word IDs]
        self.max_article_len = args.max_article_len

        if args.fix_pt_art_emb:
            args.rel_pc_art_emb_path = dataset._get_precomputed_art_emb_path()

        self.mask_token = args.max_vocab_size + 1
        args.bert_mask_token = self.mask_token

        if self.args.fix_pt_art_emb:
            self.art_id2word_ids = None
        else:
            # create direct mapping art_id -> word_ids
            self.art_id2word_ids = {art_idx: self.art_index2word_ids[art_id] for art_id, art_idx in
                                    self.item_id2idx.items()}
        del self.art_index2word_ids

        if 'ts_scaler' in data:
            self.ts_scaler = data['ts_scaler']
        else:
            self.ts_scaler = None


    @classmethod
    def code(cls):
        return 'bert_news'

    def _get_train_dataset(self):
        # u2seq, art2words, neg_samples, max_hist_len, max_article_len, mask_prob, mask_token, num_items, rng):
        dataset = BertTrainDatasetNews(self.train, self.art_id2word_ids, self.train_negative_samples, self.max_hist_len,
                                       self.max_article_len, self.mask_prob, self.mask_token, self.item_count, self.rnd,
                                       self.w_time_stamps, self.w_u_id,
                                       seq_order=self.args.train_seq_order)
        return dataset

    def _get_eval_dataset(self, mode):
        if 'val' == mode:
            u2hist = self.val
            neg_samples = self.val_negative_samples
        else:
            u2hist = self.test
            neg_samples = self.test_negative_samples

        # for now, we just assume to always use 'last_as_target'
        dataset = BertEvalDatasetNews(u2hist, self.art_id2word_ids, neg_samples, self.max_hist_len,
                                      self.max_article_len,
                                      self.mask_token, self.rnd, self.w_time_stamps, self.w_u_id,
                                      seq_order=self.args.eval_seq_order)
        return dataset

    def get_valid_items(self):
        all_items = set(self.item_id2idx.values())  # train + test + val

        if self.split_method != "time_threshold":
            test = train = all_items
        else:
            if self.w_time_stamps:
                ids, ts = zip(*itertools.chain.from_iterable(self.train.values()))
            else:
                ids = list(itertools.chain.from_iterable(self.train.values()))
            train = set(ids)
            test = all_items

        return {'train': list(train), 'test': list(test)}

    def create_eval_dataset_from_hist_negs(self, u2hist, neg_samples):
        return BertEvalDatasetNews(u2hist, self.art_id2word_ids, neg_samples, self.max_hist_len,
                                      self.max_article_len,
                                      self.mask_token, self.rnd, self.w_time_stamps, self.w_u_id,
                                      seq_order=self.args.eval_seq_order)

    def get_working_data_match_time_criteria(self, match_crit, mode='val'):

        if 'val' == mode:
            u2hist = self.val
            neg_samples = self.val_negative_samples
        else:
            u2hist = self.test
            neg_samples = self.test_negative_samples

        # select users based on matching criteria
        valid_u_idx = filter_user_query_time(u2hist, match_criteria=match_crit, ts_scaler=self.ts_scaler,
                                             time_vec_format=self.args.parts_time_vec, max_users=match_crit['n_max'])

        # filter u2hist & neg_samples
        work_u2hist, work_negs, work_idx2info = {}, {}, {}

        for i, u_idx in enumerate(valid_u_idx):
            u_id = self.idx2u_id[u_idx]

            new_u_idx = len(work_idx2info)

            # add orginal data of selected users to working data
            work_idx2info[new_u_idx] = {'u_id': u_id}
            work_negs[new_u_idx] = neg_samples[u_idx]
            work_u2hist[new_u_idx] = u2hist[u_idx]

            # get and store org query ts
            seq, ts = zip(*u2hist[u_idx])
            org_query_ts = self.ts_scaler.inverse_transform(ts[-1])  # hard coding postion = -1
            # query_ts[new_u_idx] = org_query_ts

            work_idx2info[new_u_idx]['qt'] = org_query_ts
            work_idx2info[new_u_idx]['mod'] = 'org'

        return work_u2hist, work_negs, work_idx2info


    def create_eval_dataset_modify_timestamps(self, u2hist, neg_samples, idx2u_id, mod_crit=[{'mode': 'wd_single', 'pos': -1, 'func': 'to_val', 'val': 5}]):
        """
        Builds a working EvalDataset featuring original user histories and the ones with modified query times

        Output:
            dataloader : Dataloader of working dataset (with original and modified sequences)
            work_idx2u_id : mapping user idx info
                { u_idx :
                    { u_id: 'AbC',
                    'qt': [WD, HH, mm],
                    'mod': 'mod_key' }
                }

        """

        # modify sequence based on modification criteria
        # and compile working data
        work_idx2info = {}
        work_u2hist = {}

        for i, u_idx in enumerate(u2hist):
            u_id = idx2u_id[u_idx] # map from working u_idx to ID

            # add orginal data of selected users to working data
            work_idx2info[u_idx] = {'u_id': u_id}

            mod_key = "_".join([mod_crit['mode'], mod_crit['func']])

            # create modified sequence & add to working data
            work_u2hist[u_idx], mod_query_ts = modify_query_time_of_seq(u2hist[u_idx], mod_criteria=mod_crit, ts_scaler=self.ts_scaler, time_vec_format=self.args.parts_time_vec)

            work_idx2info[u_idx]['qt'] = mod_query_ts
            work_idx2info[u_idx]['mod'] = mod_key

        dataset = self.create_eval_dataset_from_hist_negs(work_u2hist, neg_samples)

        return dataset, work_idx2info

def filter_user_query_time(u_data: Dict[int, List[Tuple[int, int]]], match_criteria: Dict[str, Any], ts_scaler,
                           time_vec_format, max_users=None) -> List[int]:

    max_users = max_users if max_users is not None else -1

    mode = match_criteria['mode'] # type of match
    pos = match_criteria['position'] # which article position
    match_val = match_criteria['match_val']  # value to match, e.g. weekday == 7 to filter for Sundays

    valid_u_idx = []
    for u_idx, hist in u_data.items():
        seq, time_stamps = zip(*hist)

        if len(seq) < match_criteria['min_seq_len']:
            continue

        if 'rnd' == match_criteria['mode']:
            valid_u_idx.append(u_idx)

        else:

            # rescale & transform normalised time stamps
            ts_vec = [ts_scaler.inverse_transform(ts) for ts in time_stamps]

            if 'wd_match' == match_criteria['mode']:
                # weekday match: map index of ts_vec[pos]
                ts_vec_idx = time_vec_format.index('WD')

            elif 'dt_match' == match_criteria['mode']:
                # daytime match
                ts_vec_idx = time_vec_format.index('HH')
            else:
                raise NotImplementedError()

            # compare to matching values
            if int(ts_vec[pos][ts_vec_idx]) in match_val and not match_criteria['neg']:
                valid_u_idx.append(u_idx) # add to valid list of matching
            elif int(ts_vec[pos][ts_vec_idx]) not in match_val and match_criteria['neg']:
                valid_u_idx.append(u_idx)

        # check break criteria
        if len(valid_u_idx) == max_users:
            break

    return valid_u_idx

def modify_query_time_of_seq(hist: List[Tuple[int, int]], mod_criteria: Dict[str, Any], ts_scaler,
                           time_vec_format) -> List[Tuple[int, int]]:

    valid_ranges = {
        'WD': [0, 6],
        'HH': [0, 23],
        'mm': [0, 59]
    }

    seq, time_stamps = zip(*hist)

    ts_org = time_stamps[mod_criteria['pos']]

    # if isinstance(mod_criteria['val'], list):
    #     # choose random value
    #     mod_val = mod_criteria['val'][0]
    # else:
    #     mod_val = mod_criteria['val']

    # rescale & transform normalised time stamps
    ts_vec = [ts_scaler.inverse_transform(ts) for ts in time_stamps]


    if 'wd_single' == mod_criteria['mode']:
        # weekday, modify single value
        ts_vec_idx = time_vec_format.index('WD')
    elif 'dt_single' == mod_criteria['mode']:
        ts_vec_idx = time_vec_format.index('HH') # daytime
    elif 'mm_single' == mod_criteria['mode']:
        ts_vec_idx = time_vec_format.index('mm')  # minutes
    else:
        raise NotImplementedError(mod_criteria['mode'])

    # modify query time stamp
    new_time = ts_vec[mod_criteria['pos']]
    if 'by_val' == mod_criteria['func']:
        new_time[ts_vec_idx] += mod_criteria['val']
    elif 'to_val' == mod_criteria['func']:
        new_time[ts_vec_idx] = mod_criteria['val']
    else:
        raise NotImplementedError(mod_criteria['func'])

    # assert valid range of new time and account for overrun
    #if valid_range[0] <= new_time <= valid_range[1]:

    rev_time_format = list(reversed(time_vec_format))

    for i in range(len(time_vec_format)):
        time_comp = rev_time_format[i]
        if new_time[time_vec_format.index(time_comp)] > valid_ranges[time_comp][1]: # exceeding max
            new_time[time_vec_format.index(time_comp)] -= (valid_ranges[time_comp][1] + 1) # substract max

            if time_comp != time_vec_format[0]:
                # increment next higher value
                new_time[time_vec_format.index(rev_time_format[i + 1])] += 1

        # elif new_time[time_vec_format.index(time_comp)] < valid_ranges[time_comp][0]: # under min
        # # TODO: substracting 'val' is lower than 'min'
        #     ts_vec[mod_criteria['pos']][ts_vec_idx] = valid_range[0]  # set to min

    query_ts_vec = ts_vec[mod_criteria['pos']] = new_time
    org_ts_vec = ts_scaler.inverse_transform(ts_org)

    time_stamps = ts_scaler.transform(np.array(ts_vec)).tolist()  # transform to normalised values

    if time_stamps[mod_criteria['pos']] == ts_org:
        print(query_ts_vec)
        print(org_ts_vec)


    return list(zip(seq, time_stamps)), query_ts_vec

def art_idx2word_ids(art_idx, mapping):
    if mapping is not None:
        return mapping[art_idx]
    else:
        return art_idx


def pad_seq(seq, pad_token, max_hist_len, max_article_len=None, n=None):
    """
    seq (list): sequence(s) of token (int)
    pad_token (int): padding token to fill gaps
    max_article_len (int): article length; padding has to fit this length, e.g.
        if an article is mapped to sequence of L words, then padded items are also sequences of L * pad_token
    n (int): number of candidates; for each position we could have N candidates

    """
    seq = list(seq[-max_hist_len:])
    pad_len = max_hist_len - len(seq)

    if pad_len > 0:
        if max_article_len is not None and n is None:
            return [[pad_token] * max_article_len] * pad_len + seq
        elif max_article_len is not None and n is not None:
            return [[[pad_token] * max_article_len] * n] * pad_len + seq
        elif max_article_len is None and n is not None:
            return [[pad_token] * n] * pad_len + seq
        else:
            return [pad_token] * pad_len + seq
    else:
        return seq


class BertTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_hist_len, mask_prob, mask_token, num_items, rng, pad_token=0):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_hist_len = max_hist_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):

        # generate masked item sequence on-the-fly
        user = self.users[index]
        seq = self._getseq(user)

        return self.gen_train_instance(seq)

    def _getseq(self, user):
        return self.u2seq[user]

    def gen_train_instance(self, seq):
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

        return torch.LongTensor(pad_seq(tokens, self.max_hist_len, pad_token=self.pad_token)), \
               torch.LongTensor(pad_seq(labels, self.max_hist_len, pad_token=self.pad_token))


class BertEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_hist_len, mask_token, neg_samples, rnd, pad_token=0, u_idx=False):
        self.u2hist = u2seq
        self.u_sample_ids = sorted(self.u2hist.keys())
        self.u2targets = u2answer
        self.neg_samples = neg_samples

        self.max_hist_len = max_hist_len
        self.mask_token = mask_token
        self.pad_token = pad_token

        self.rnd = rnd

        self.w_u_id = u_idx  # indicate usage of user id

    def __len__(self):
        return len(self.u_sample_ids)

    def __getitem__(self, index):
        u_idx = self.u_sample_ids[index]
        hist = self.u2hist[u_idx]
        test_items = self.u2targets[u_idx]

        if test_items == []:
            pass
        else:
            if isinstance(u_idx, str):
                negs = self.neg_samples[int(u_idx[:-1])]
            else:
                negs = self.neg_samples[u_idx]  # get negative samples

            if not self.w_u_id:
                u_idx = None
            return self.gen_eval_instance(hist, test_items, negs, u_idx)

    def gen_eval_instance(self, hist, target, negs, user=None):
        candidates = target + negs
        # candidates = [art_idx2word_ids(cand, self.art2words) for cand in candidates]
        labels = [1] * len(target) + [0] * len(negs)

        hist = hist + [self.mask_token]  # predict only the next/last token in seq
        hist = pad_seq(hist, self.pad_token, self.max_hist_len)

        return torch.LongTensor(hist), torch.LongTensor(candidates), torch.LongTensor(labels)

    def concat_ints(self, a, b):
        return str(f"{a}{b}")


### NEWS ###
class BertTrainDatasetNews(BertTrainDataset):
    def __init__(self, u2seq, art2words, neg_samples, max_hist_len, max_article_len, mask_prob, mask_token, num_items,
                 rng,
                 w_time_stamps=False, w_u_id=False, seq_order=None):
        super(BertTrainDatasetNews, self).__init__(u2seq, max_hist_len, mask_prob, mask_token, num_items, rng)

        self.art2words = art2words
        self.max_article_len = max_article_len
        self.train_neg_samples = neg_samples

        self.w_time_stamps = w_time_stamps
        self.w_u_idx = w_u_id
        self.shuffle_seq_order = seq_order

    def __getitem__(self, index):

        # generate masked item sequence on-the-fly
        u_idx = self.users[index]
        seq = self._getseq(u_idx)
        neg_samples = self._get_neg_samples(u_idx)

        if not self.w_u_idx:
            u_idx = None

        return self.gen_train_instance(seq, neg_samples, u_idx=u_idx)

    def gen_train_instance(self, seq, neg_samples, u_idx=None):
        hist = []
        labels = []
        mask = []
        candidates = []
        time_stamps = []
        n_cands = len(neg_samples[0])
        n_predictions = 0

        pos_irrelevant_lbl = -1  # label to indicate irrelevant position to avoid confusion with other categorical labels

        ## create masked input sequence ##

        for idx, entry in enumerate(seq):
            if self.w_time_stamps:
                art_id, ts = entry
                time_stamps.append(ts)
            else:
                art_id = entry

            if 0 not in mask and idx + 1 == len(seq):
                # force at least 1 mask position if no seq element has been masked off
                prob = 0
            else:
                prob = self.rng.random()

            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    # Mask this position
                    # Note: we put original token but record the Mask position
                    # After the News Encoder this position will be replaced with Mask Embedding
                    tkn = art_id
                    m_val = 0
                    n_predictions += 1
                elif prob < 0.9:
                    # put random item
                    tkn = self.rng.choice(range(self.num_items))
                    # tokens.append(art_idx2word_ids(self.rng.randint(1, self.num_items), self.art2words))
                    m_val = 1
                else:
                    # put original token but no masking
                    tkn = art_id
                    m_val = 1

                hist.append(art_idx2word_ids(tkn, self.art2words))

                mask.append(m_val)

                cands = neg_samples[idx] + [art_id]
                self.rng.shuffle(cands)  # shuffle candidates so model cannot trivially guess target position

                labels.append(cands.index(art_id))

                candidates.append([art_idx2word_ids(art, self.art2words) for art in cands])

            else:
                hist.append(art_idx2word_ids(art_id, self.art2words))
                labels.append(pos_irrelevant_lbl)
                mask.append(1)

                if self.art2words is None:
                    cands = [0] * (len(neg_samples[idx]) + 1)
                else:
                    cands = [[0] * self.max_article_len] * (len(neg_samples[idx]) + 1)

                candidates.append(cands)

        #################################################
        ## truncate sequence & apply left-side padding ##

        # if art2word mapping is applied, hist is shape (max_article_len x max_hist_len), i.e. sequences of word IDs
        # else, hist is shape (max_hist_len), i.e. sequence of article indices
        hist = pad_seq(hist, self.pad_token, self.max_hist_len,
                       max_article_len=(self.max_article_len if self.art2words is not None else None))
        candidates = pad_seq(candidates, self.pad_token, self.max_hist_len,
                             max_article_len=(self.max_article_len if self.art2words is not None else None),
                             n=n_cands + 1)
        labels = pad_seq(labels, pad_token=pos_irrelevant_lbl, max_hist_len=self.max_hist_len, )
        mask = pad_seq(mask, pad_token=1, max_hist_len=self.max_hist_len, )
        assert len(hist) == self.max_hist_len

        ### Output ####

        inp = {'hist': torch.LongTensor(hist), 'mask': torch.LongTensor(mask), \
               'cands': torch.LongTensor(candidates)}

        if self.w_time_stamps:
            len_time_vec = len(time_stamps[0])
            time_stamps = pad_seq(time_stamps, pad_token=0,
                                  max_hist_len=self.max_hist_len, n=len_time_vec)
            inp['ts'] = torch.LongTensor(time_stamps)

        if u_idx is not None:
            inp['u_id'] = torch.LongTensor(
                [u_idx] * self.max_hist_len)  # need tensors of equal lenght for collate function

        return {'input': inp, 'lbls': torch.LongTensor(labels)}

    def _get_neg_samples(self, user):
        return self.train_neg_samples[user]


def reorder_sequence(seq, rnd, seq_order='shuffle_exc_t'):
    """ Reorder sequence according to passed argument """

    if 'shuffle_all' == seq_order:
        rnd.shuffle(list(seq))
        return seq
    elif 'shuffle_exc_t' == seq_order:
        t = seq[-1]  # preserve target
        seq_to_shuffle = list(seq[:-1])
        rnd.shuffle(seq_to_shuffle)
        return seq_to_shuffle + [t]
    elif 'shuffle_exc_t_1' == seq_order:
        t = list(seq[-2:])
        seq_to_shuffle = list(seq[:-2])
        rnd.shuffle(seq_to_shuffle)
        return seq_to_shuffle + t
    elif 'ordered' == seq_order:
        return seq
    else:
        raise ValueError(seq_order)


class BertEvalDatasetNews(BertEvalDataset):

    def __init__(self, u2seq, art2words, neg_samples, max_hist_len, max_article_len, mask_token,
                 rnd, w_time_stamps=False, u_idx=False, seq_order=None):
        super(BertEvalDatasetNews, self).__init__(u2seq, None, max_hist_len, mask_token, neg_samples, rnd, u_idx=u_idx)

        self.art2words = art2words
        self.max_article_len = max_article_len  # len(next(iter(art2words.values())))
        self.eval_mask = [1] * (max_hist_len - 1) + [0]  # insert mask token at the end
        self.w_time_stamps = w_time_stamps
        self.seq_order = seq_order

    def __getitem__(self, u_idx, map_art2words=True):
        hist = self.u2hist[u_idx]
        negs = self.neg_samples[u_idx]  # get negative samples

        # if not self.w_u_id:
        #     u_idx = None
        return self.gen_eval_instance(hist, negs, u_idx, map_art2words)

    def gen_eval_instance(self, hist, negs, u_idx=None, map_art2words=True):

        if self.w_time_stamps:
            hist, time_stamps = zip(*hist)
            # test_items, test_time_stamps = zip(*test_items)
            # time_stamps = list(time_stamps + test_time_stamps)[-self.max_hist_len:]
            # time vectors
            # time_stamps = list(map(map_time_stamp_to_vector, time_stamps))
            # args.len_time_vec
            len_time_vec = len(time_stamps[0])
            # padding
            time_stamps = pad_seq(time_stamps, pad_token=0,
                                  max_hist_len=self.max_hist_len, n=len_time_vec)
        else:
            time_stamps = None

        # adjust sequence if applicable
        if self.seq_order is not None:
            hist = reorder_sequence(hist, self.rnd, self.seq_order)

        target = hist[-1]  # predict only the next/last token in seq
        candidates = [target] + negs  # candidates as article indices
        # shuffle to avoid trivial guessing
        self.rnd.shuffle(candidates)
        labels = [0] * len(candidates)
        labels[candidates.index(target)] = 1

        if map_art2words:
            # map art indices to words if applicable
            candidates = [art_idx2word_ids(cand, self.art2words) for cand in candidates]
            hist = [art_idx2word_ids(art, self.art2words) for art in hist[-self.max_hist_len:]]
            # note: target will be masked off by model

            ## apply padding
            hist = pad_seq(hist, self.pad_token, self.max_hist_len,
                           max_article_len=(self.max_article_len
                                            if self.art2words is not None else None))
        else:
            hist = pad_seq(hist[-self.max_hist_len:], self.pad_token, self.max_hist_len,
                           max_article_len=None)
        #
        assert len(hist) == self.max_hist_len

        inp = {'hist': torch.LongTensor(hist), 'mask': torch.LongTensor(self.eval_mask), \
               'cands': torch.LongTensor(candidates)}
        if self.w_time_stamps:
            inp['ts'] = torch.LongTensor(time_stamps)

        # if u_idx is not None:
        # add u_id
        inp['u_id'] = torch.LongTensor([u_idx] * self.max_hist_len)

        return {'input': inp, 'lbls': torch.LongTensor(labels)}

    def filter_users(self, valid_u_idx):
        """ Reduce dataset to users indicated by passed indices """
        # for u_idx in valid_u_idx:
        # self.u2hist = {u_idx: self.u2hist[u_idx] for u_idx in self.u2hist if u_idx in valid_u_idx}
        pass