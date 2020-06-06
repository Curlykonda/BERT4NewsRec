from abc import *
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
