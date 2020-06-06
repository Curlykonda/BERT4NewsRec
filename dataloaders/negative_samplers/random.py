import itertools

from .base import AbstractNegativeSampler

from tqdm import trange

import numpy as np
import random


class RandomNegativeSamplerPerUser(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self):
        # seed is set in AbstractClass
        #random.seed(self.seed)
        negative_samples = {}
        print('Sampling negative items')
        for user in trange(self.user_count):
            # determine the items already seen by the user
            if "npa" == self.train_method and isinstance(self.train[user][0], tuple):
                #train (dict(list)): {u_idx: [([hist1], target1), .. ([histN], targetN])}
                #test (dict(list)): {u_idx: ([hist], [targets])}
                seen = set(list(itertools.chain(*[hist for (hist, tgt) in self.train[user]])))
                seen.update(x for x in self.val[user][0])
                seen.update(x for x in self.val[user][1])
                seen.update(x for x in self.test[user][0])
                seen.update(x for x in self.test[user][1])

            elif isinstance(self.train[user][0], tuple):
                # TE case (art_id, [time_vector])
                seen = set(x[0] for x in self.train[user])
                seen.update(x[0] for x in self.val[user])
                seen.update(x[0] for x in self.test[user])

            else:
                seen = set(self.train[user])
                seen.update(self.val[user])
                seen.update(self.test[user])

            # sample random unseen items from the full set
            # note: for 'time_split' need to separate into train and test intervals
            if self.seq_lengths is None:
                # one set of neg samples for each user
                samples = self.get_rnd_samples_for_position(seen)
            else:
                # neg samples for each position in each user sequence
                samples = []
                for _ in range(self.seq_lengths[user]):
                    neg_samples = self.get_rnd_samples_for_position(seen)
                    samples.append(neg_samples)

                assert len(samples) == self.seq_lengths[user]

            negative_samples[user] = samples

        return negative_samples

    def get_rnd_samples_for_position(self, seen):
        samples = []
        for _ in range(self.sample_size):
            item = random.choice(self.valid_items)
            while item in seen or item in samples:
                item = random.choice(self.valid_items)
            samples.append(item)

        return samples

    def get_naive_random_samples(self, sample_size, item_set):
        if sample_size is None:
            sample_size = self.sample_size

        naive_sample = random.sample(item_set, sample_size)

        return naive_sample

# class RandomNegativeSamplerPerPosition(AbstractNegativeSampler):
#     @classmethod
#     def code(cls):
#         return 'random'
#
#     def generate_negative_samples(self):
#         negative_samples = {}
#         print('Sampling negative items')
#         for user in trange(self.user_count):
#             # determine the items already seen by the user
#             if isinstance(self.train[user][1], tuple):
#                 seen = set(x[0] for x in self.train[user])
#                 seen.update(x[0] for x in self.val[user])
#                 seen.update(x[0] for x in self.test[user])
#             else:
#                 seen = set(self.train[user])
#                 seen.update(self.val[user])
#                 seen.update(self.test[user])
#
#             # sample random unseen items from the full set for each position in user seq
#             # note: for 'time_split' need to separate into train and test intervals
#             samples = []
#             for i in range(self.seq_lengths[user]):
#                 neg_samples = []
#                 for _ in range(self.sample_size):
#                     item = random.choice(self.item_set)
#                     while item in seen or item in samples:
#                         item = random.choice(self.item_set)
#                     neg_samples.append(item)
#                 samples.append(neg_samples)
#
#             negative_samples[user] = samples
#
#         return negative_samples


