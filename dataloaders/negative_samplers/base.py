import itertools
from abc import *
from collections import Counter
from pathlib import Path
import pickle
import random


class AbstractNegativeSampler(metaclass=ABCMeta):
    def __init__(self, train_method, mode, train, val, test, user_count, item_set, sample_size, seed, seq_lengths, save_folder):
        self.train = train
        self.val = val
        self.test = test
        self.user_count = user_count
        self.valid_items = list(item_set)
        self.sample_size = sample_size
        self.seq_lengths = seq_lengths # indicates the sequence length for each user
        self.seed = seed
        self.mode = mode
        assert self.seed is not None, 'Specify seed for random sampling'
        random.seed(self.seed)
        self.rnd = random.random()

        self.save_folder = save_folder
        self.train_method = train_method

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def generate_negative_samples(self):
        pass

    def get_negative_samples(self):
        savefile_path = self._get_save_path()
        if savefile_path.is_file():
            print('Negatives samples exist. Loading.')
            negative_samples = pickle.load(savefile_path.open('rb'))
            return negative_samples
        print("Negative samples don't exist. Generating.")
        negative_samples = self.generate_negative_samples()
        with savefile_path.open('wb') as f:
            pickle.dump(negative_samples, f)
        return negative_samples

    def _get_save_path(self):
        folder = Path(self.save_folder)
        filename = '{}-sample_size{}-{}-seed{}.pkl'.format(self.code(), self.sample_size, self.mode, self.seed)
        return folder.joinpath(filename)

    @abstractmethod
    def get_naive_random_samples(self, sample_size, item_set):
        pass

    def determine_seen_items(self, user):
        # determine the items already seen by the user
        popularity = Counter()
        seen = set()
        if "npa" == self.train_method and isinstance(self.train[user][0], tuple):
            # train (dict(list)): {u_idx: [([hist1], target1), .. ([histN], targetN])}
            # test (dict(list)): {u_idx: ([hist], [targets])}
            chained_items = list(itertools.chain(*[hist for (hist, tgt) in self.train[user]]))
            seen.update(chained_items)
            popularity.update(chained_items)

            for items in (self.val[user][0], self.val[user][1], self.test[user][0], self.test[user][1]):
                chained_items = [x for x in items]
                seen.update(chained_items)
                popularity.update(chained_items)

            # chained_items = [x for x in self.val[user][0]]
            #
            # seen.update(x for x in self.val[user][1])
            # seen.update(x for x in self.test[user][0])
            # seen.update(x for x in self.test[user][1])

        elif isinstance(self.train[user][0], tuple):
            # TE case (art_id, [time_vector])
            for items in (self.train[user], self.val[user], self.test[user]):
                chained_items = [x[0] for x in items]
                seen.update(chained_items)
                popularity.update(chained_items)

            # seen = set(x[0] for x in self.train[user])
            # seen.update(x[0] for x in self.val[user])
            # seen.update(x[0] for x in self.test[user])

        else:
            for items in (self.train[user], self.val[user], self.test[user]):
                seen.update(items)
                popularity.update(items)
            #
            # seen = set(self.train[user])
            # seen.update(self.val[user])
            # seen.update(self.test[user])

        return seen, popularity
