import copy
from random import random

from .base import AbstractNegativeSampler

from tqdm import trange

from collections import Counter


class PopularNaiveNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'popular_naive'

    def generate_negative_samples(self):
        popular_items = self.items_by_popularity()

        negative_samples = {}
        print('Sampling negative items')
        for user in trange(self.user_count):
            seen = set(self.train[user])
            seen.update(self.val[user])
            seen.update(self.test[user])

            samples = []
            for item in popular_items:
                if len(samples) == self.sample_size:
                    break
                if item in seen:
                    continue
                samples.append(item)

            negative_samples[user] = samples

        return negative_samples

    def items_by_popularity(self):
        popularity = Counter()
        for user in range(self.user_count):
            popularity.update(self.train[user])
            popularity.update(self.val[user])
            popularity.update(self.test[user])
        popular_items = sorted(popularity, key=popularity.get, reverse=True)
        return popular_items




class PopularLikelihoodNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'popular_rnd'

    def generate_negative_samples(self):
        popular_items, seens = self.items_by_popularity()

        # for x in self.valid_items:
        #     del popular_items[x]

        self.pop_items = popular_items

        negative_samples = {}
        print('Sampling negative items')
        for user in trange(self.user_count):
            seen = seens[user]

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
        pop_items = copy.deepcopy(self.pop_items)
        # remove seen items from counter to reduce computational effort
        for x in seen:
            del pop_items[x]

        pop_items = list(pop_items.elements())

        while len(samples) < self.sample_size:
            item = self.rnd.choice(pop_items)
            if item not in samples and item in self.valid_items:
                samples.append(item)
                pop_items = [x for x in pop_items if x != item]
                #self.removeall_inplace(pop_items, item)

        return samples

    def removeall_inplace(self, l, val):
        for _ in range(l.count(val)):
            l.remove(val)

    def items_by_popularity(self):
        popularity = Counter()
        seens = {}
        for user in range(self.user_count):
            seen, pop = self.determine_seen_items(user)
            seens[user] = seen
            popularity.update(pop)
        #popular_items = sorted(popularity, key=popularity.get, reverse=True)
        for art_id, cnt in dict(popularity).items():
            if art_id not in self.valid_items:
                del popularity[art_id]

        return popularity, seens
