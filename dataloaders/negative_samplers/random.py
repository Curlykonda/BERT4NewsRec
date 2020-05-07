from .base import AbstractNegativeSampler

from tqdm import trange

import numpy as np
import random


class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        #random.seed(self.seed)
        negative_samples = {}
        print('Sampling negative items')
        for user in trange(self.user_count):
            # determine the items already seen by the user
            if isinstance(self.train[user][1], tuple):
                seen = set(x[0] for x in self.train[user])
                seen.update(x[0] for x in self.val[user])
                seen.update(x[0] for x in self.test[user])
            else:
                seen = set(self.train[user])
                seen.update(self.val[user])
                seen.update(self.test[user])

            # sample random unseen items from the full set
            # note: for 'time_split' need to separate into train and test intervals
            samples = []
            for _ in range(self.sample_size):
                item = random.choice(self.item_set)
                while item in seen or item in samples:
                    item = random.choice(self.item_set)
                samples.append(item)

            negative_samples[user] = samples

        return negative_samples
