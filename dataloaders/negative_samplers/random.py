import copy
import itertools
from collections import Counter

from .base import AbstractNegativeSampler
from tqdm import trange


class RandomNegativeSamplerPerUser(AbstractNegativeSampler):
    def __init__(self, **kwargs):
        super(RandomNegativeSamplerPerUser, self).__init__(**kwargs)

    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self):
        # seed is set in AbstractClass
        #random.seed(self.seed)
        negative_samples = {}
        print('Sampling negative items')
        for user in trange(self.n_users):

            seen, _ = self.determine_seen_items(user)

            # sample uniform random unseen items from the full set
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
            item = self.rnd.choice(self.valid_items)
            while item in seen or item in samples:
                item = self.rnd.choice(self.valid_items)
            samples.append(item)

        return samples

    def get_naive_random_samples(self, sample_size, item_set):
        if sample_size is None:
            sample_size = self.sample_size

        naive_sample = self.rnd.sample(item_set, sample_size)

        return naive_sample


class RandomBrandSensitiveNegativeSampler(AbstractNegativeSampler):
    def __init__(self, **kwargs):
        super(RandomBrandSensitiveNegativeSampler, self).__init__(**kwargs)

        if 'id2idx' not in kwargs.keys():
            raise ValueError('Article index to ID mapping required for brand-sensitive sampling')
        else:
            # build art2brand
            id2idx = kwargs['id2idx']
            id2info = kwargs['id2info']

            self.art_idx2brand = {indx: list(id2info[id]['brands']) for id, indx in id2idx.items()}

        self.targets = self.get_prediction_targets_for_position()

    @classmethod
    def code(cls):
        return 'rnd_brand_sens'

    def generate_negative_samples(self):

        negative_samples = {}
        print('Sampling negative items')
        for user in trange(self.n_users):

            seen, _ = self.determine_seen_items(user)

            # sample uniform random unseen items from the full set
            # note: for 'time_split' need to separate into train and test intervals

            if isinstance(self.targets[user], int):
                samples = self.get_brandsens_sample_for_target(self.targets[user], seen)
            else:
                # neg samples for each position in each user sequence
                samples = []
                for i in range(len(self.targets[user])):
                    neg_samples = self.get_brandsens_sample_for_target(self.targets[user][i], seen)
                    samples.append(neg_samples)

                assert len(samples) == len(self.targets[user])

            negative_samples[user] = samples

        return negative_samples

    def get_brandsens_sample_for_target(self, target, seen):
        samples = []
        target_brands = self.art_idx2brand[target]
        while len(samples) < self.sample_size:
            item = self.rnd.choice(self.valid_items)

            if item not in seen and item not in samples \
                and self.check_brand_match(target_brands, self.art_idx2brand[item]):
                samples.append(item)

        return samples

    def check_brand_match(self, tgt_brands, itm_brands):
        for b in itm_brands:
            if b in tgt_brands:
                return True

        return False

    def get_prediction_targets_for_position(self):
        """
        Returns the target article for each position of the user history depending on the mode
        """
        if "train" == self.mode:
            return self.train
        elif 'val' == self.mode:
            return {u_idx: hist[-1] for u_idx, hist in self.val.items()}
        elif 'test' == self.mode:
            return {u_idx: hist[-1] for u_idx, hist in self.test.items()}
        else:
            raise ValueError()

class RandomFromCommonNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random_common'

    def generate_negative_samples(self):
        popular_items, seens = self.items_by_popularity()

        self.pop_items = sorted(popular_items, key=popular_items.get, reverse=True)

        negative_samples = {}
        print('Sampling negative items')
        for user in trange(self.n_users):
            seen = seens[user]

            # sample uniform random from most common items
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

    def get_rnd_samples_for_position(self, seen, m_common=10000):
        samples = []
        # remove seen items from counter
        #pop_items = [x for x in self.pop_items if x not in seen][:m_common]
        pop_items = self.pop_items[:m_common]

        while len(samples) < self.sample_size:
            item = self.rnd.choice(pop_items)
            if item not in samples and item not in seen and item in self.valid_items:
                samples.append(item)
        #samples = self.rnd.sample(pop_items, self.sample_size)

        return samples

    def removeall_inplace(self, l, val):
        for _ in range(l.count(val)):
            l.remove(val)

    def items_by_popularity(self):
        popularity = Counter()
        seens = {}
        for user in range(self.n_users):
            seen, pop = self.determine_seen_items(user)
            seens[user] = seen
            popularity.update(pop)
        #popular_items = sorted(popularity, key=popularity.get, reverse=True)
        for art_id, cnt in dict(popularity).items():
            if art_id not in self.valid_items:
                del popularity[art_id]

        return popularity, seens

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


