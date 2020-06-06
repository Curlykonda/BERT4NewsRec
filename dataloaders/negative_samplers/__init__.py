from .popular import PopularNegativeSampler
from .random import RandomNegativeSamplerPerUser


NEGATIVE_SAMPLERS = {
    PopularNegativeSampler.code(): PopularNegativeSampler,
    RandomNegativeSamplerPerUser.code(): RandomNegativeSamplerPerUser,
}

def negative_sampler_factory(train_method, mode, code, train, val, test, user_count, item_count, sample_size, seq_lengths, seed, save_folder):
    negative_sampler = NEGATIVE_SAMPLERS[code]
    return negative_sampler(train_method, mode, train, val, test, user_count, item_count, sample_size, seq_lengths, seed, save_folder)
