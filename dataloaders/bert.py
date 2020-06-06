import itertools

from dataloaders.base import AbstractDataloader
#from dataloaders.news import BertTrainDatasetNews, BertEvalDatasetNews
from dataloaders.negative_samplers import negative_sampler_factory
from source.utils import check_all_equal, map_time_stamp_to_vector


import torch
import torch.utils.data as data_utils

class BertDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        args.num_items = len(self.smap)
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

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = BertTrainDataset(self.train, self.max_hist_len, self.mask_prob, self.mask_token, self.item_count, self.rnd) # , self.art_idx2word_ids
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

    def get_negative_sampler(self, mode, code, neg_sample_size, seed, item_set, seq_lengths):
        # sample negative instances for each user
        """
        param: seq_lengths (dict): how many separate neg samples do we need for this user?
            E.g. if seq_length[u_id] = 20, we generate 'neg_sample_size' samples for each of the 20 sequence positions
        """
        if False:
            # use time-sensitive set for neg sampling
            raise NotImplementedError()
        else:
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

    @classmethod
    def code(cls):
        return 'bert_news'

    def _get_train_dataset(self):
        # u2seq, art2words, neg_samples, max_hist_len, max_article_len, mask_prob, mask_token, num_items, rng):
        dataset = BertTrainDatasetNews(self.train, self.art_id2word_ids, self.train_negative_samples, self.max_hist_len,
                                       self.max_article_len, self.mask_prob, self.mask_token, self.item_count, self.rnd,
                                       self.w_time_stamps, self.w_u_id)
        return dataset

    def _get_eval_dataset(self, mode):
        if 'val' == mode:
            test_items = {}
            u2hist = {}
            # filter out user without validation instance
            for u_idx, items in self.val.items():
                if len(items) >= 1:
                    test_items[u_idx] = items
                    u2hist[u_idx] = self.train[u_idx]
        else:
            test_items = self.test
            u2hist = self.train

        # for now, we just assume to always use 'last_as_target'
        dataset = BertEvalDatasetNews(u2hist, test_items, self.art_id2word_ids, self.test_negative_samples, self.max_hist_len, self.max_article_len,
                                      self.mask_token, self.rnd, self.w_time_stamps, self.w_u_id)
        return dataset

    def get_valid_items(self):
        all_items = set(self.smap.values()) # train + test + val

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
    seq = seq[-max_hist_len:]
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

        self.w_u_id = u_idx # indicate usage of user id

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
                negs = self.neg_samples[u_idx] # get negative samples

            if not self.w_u_id:
                u_idx = None
            return self.gen_eval_instance(hist, test_items, negs, u_idx)


    def gen_eval_instance(self, hist, target, negs, user=None):
        candidates = target + negs
        #candidates = [art_idx2word_ids(cand, self.art2words) for cand in candidates]
        labels = [1] * len(target) + [0] * len(negs)

        hist = hist + [self.mask_token]  # predict only the next/last token in seq
        hist = pad_seq(hist, self.pad_token, self.max_hist_len)

        return torch.LongTensor(hist), torch.LongTensor(candidates), torch.LongTensor(labels)

    def concat_ints(self, a, b):
        return str(f"{a}{b}")



class BertTrainDatasetNews(BertTrainDataset):
    def __init__(self, u2seq, art2words, neg_samples, max_hist_len, max_article_len, mask_prob, mask_token, num_items, rng,
                 w_time_stamps=False, w_u_id=False):
        super(BertTrainDatasetNews, self).__init__(u2seq, max_hist_len, mask_prob, mask_token, num_items, rng)

        self.art2words = art2words
        self.max_article_len = max_article_len
        self.train_neg_samples = neg_samples

        self.w_time_stamps = w_time_stamps
        self.w_u_idx = w_u_id

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

        pos_irrelevant_lbl = -1 # label to indicate irrelevant position to avoid confusion with other categorical labels

        for idx, entry in enumerate(seq):
            if self.w_time_stamps:
                art_id, ts = entry
                time_stamps.append(ts)
            else:
                art_id = entry

            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    # Mask this position
                    # Note: we put original token but record the Mask position
                    # After the News Encoder this position will be replaced with Mask Embedding
                    tkn = art_id
                    m_val = 0
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
                self.rng.shuffle(cands) # shuffle candidates so model cannot trivially guess target position

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

                # cands = neg_samples[idx] + [art_id]
                #candidates.append([art_idx2word_ids(art, self.art2words) for art in cands])



        # truncate sequence & apply left-side padding
        ##############################################
        # if art2word mapping is applied, hist is shape (max_article_len x max_hist_len), i.e. sequences of word IDs
        # else, hist is shape (max_hist_len), i.e. sequence of article indices
        hist = pad_seq(hist, self.pad_token, self.max_hist_len, max_article_len=(self.max_article_len if self.art2words is not None else None))
        candidates = pad_seq(candidates, self.pad_token, self.max_hist_len,
                             max_article_len=(self.max_article_len if self.art2words is not None else None),
                             n=n_cands+1)
        labels = pad_seq(labels, pad_token=pos_irrelevant_lbl, max_hist_len=self.max_hist_len,)
        mask = pad_seq(mask, pad_token=1, max_hist_len=self.max_hist_len,)

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
            inp['u_id'] = torch.LongTensor([u_idx] * self.max_hist_len) # need tensors of equal lenght for collate function

        return {'input': inp, 'lbls': torch.LongTensor(labels)}


#         if self.w_time_stamps:
# #            time_stamps = list(map(map_time_stamp_to_vector, time_stamps))
#             #args.len_time_vec
#
#
#             return torch.LongTensor(hist), torch.LongTensor(mask), \
#                    torch.LongTensor(candidates), torch.LongTensor(labels), \
#                    torch.LongTensor(time_stamps)
#         else:
#             return torch.LongTensor(hist), torch.LongTensor(mask), \
#                    torch.LongTensor(candidates), torch.LongTensor(labels)

    def _get_neg_samples(self, user):
        return self.train_neg_samples[user]


class BertEvalDatasetNews(BertEvalDataset):

    def __init__(self, u2seq, u2answer, art2words, neg_samples, max_hist_len, max_article_len, mask_token,
                 rnd, w_time_stamps=False, u_idx=False):
        super(BertEvalDatasetNews, self).__init__(u2seq, u2answer, max_hist_len, mask_token, neg_samples, rnd, u_idx=u_idx)

        self.art2words = art2words
        self.max_article_len = max_article_len # len(next(iter(art2words.values())))
        self.eval_mask = [1] * (max_hist_len-1) + [0]  # insert mask token at the end
        self.w_time_stamps = w_time_stamps

    def gen_eval_instance(self, hist, test_items, negs, u_idx=None):
        # hist = train + test[:-1]
        if self.w_time_stamps:
            hist, time_stamps = zip(*hist)
            test_items, test_time_stamps = zip(*test_items)
            time_stamps = list(time_stamps + test_time_stamps)[-self.max_hist_len:]
            # time vectors
            #time_stamps = list(map(map_time_stamp_to_vector, time_stamps))
            #args.len_time_vec
            len_time_vec = len(time_stamps[0])
            # padding
            time_stamps = pad_seq(time_stamps, pad_token=0,
                                  max_hist_len=self.max_hist_len, n=len_time_vec)
        else:
            time_stamps = None
            test_time_stamps = None

        target = [test_items[-1]]
        candidates = target + negs # candidates as article indices
        # shuffle to avoid trivial guessing
        self.rnd.shuffle(candidates)
        labels = [0] * len(candidates)
        labels[candidates.index(*target)] = 1

        candidates = [art_idx2word_ids(cand, self.art2words) for cand in candidates]

        # extend train history with new test interactions
        hist = hist + test_items[:-1]
        hist = [art_idx2word_ids(art, self.art2words) for art in hist[-(self.max_hist_len- 1):]]
        # append a target to history which will be masked off
        # alternatively we could put a random item. does not really matter because it's gonna be masked off anyways
        hist = hist + [art_idx2word_ids(*target, self.art2words)]  # predict only the next/last token in seq

        ## apply padding
        hist = pad_seq(hist, self.pad_token, self.max_hist_len,
                       max_article_len=(self.max_article_len
                                        if self.art2words is not None else None))
        #
        assert len(hist) == self.max_hist_len

        #print(target)
        # return dictionary
        # return {'input': [torch.LongTensor(hist), torch.LongTensor(self.eval_mask)],
        #         'cands': torch.LongTensor(candidates),
        #         'lbls': torch.LongTensor(labels)}

        inp = {'hist': torch.LongTensor(hist), 'mask': torch.LongTensor(self.eval_mask), \
               'cands': torch.LongTensor(candidates)}
        if self.w_time_stamps:
            inp['ts'] = torch.LongTensor(time_stamps)

        if u_idx is not None:
            inp['u_id'] = torch.LongTensor([u_idx] * self.max_hist_len)

        return {'input': inp, 'lbls': torch.LongTensor(labels)}

        # if self.w_time_stamps:
        #     return torch.LongTensor(hist), torch.LongTensor(self.eval_mask), \
        #            torch.LongTensor(candidates), torch.LongTensor(labels), \
        #            torch.LongTensor(time_stamps)
        # else:
        #     return torch.LongTensor(hist), torch.LongTensor(self.eval_mask), \
        #            torch.LongTensor(candidates), torch.LongTensor(labels)
