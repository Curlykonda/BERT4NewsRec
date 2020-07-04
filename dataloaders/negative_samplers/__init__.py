from .popular import *
from .random import *


NEGATIVE_SAMPLERS = {
    RandomNegativeSamplerPerUser.code(): RandomNegativeSamplerPerUser,
    RandomFromCommonNegativeSampler.code(): RandomFromCommonNegativeSampler,
    PopularLikelihoodNegativeSampler.code(): PopularLikelihoodNegativeSampler,
    PopularNaiveNegativeSampler.code(): PopularNaiveNegativeSampler
}

def negative_sampler_factory(train_method, mode, code, train, val, test, user_count, item_count, sample_size, seq_lengths, seed, save_folder, m_common=None):
    negative_sampler = NEGATIVE_SAMPLERS[code]
    return negative_sampler(train_method, mode, train, val, test, user_count, item_count, sample_size, seq_lengths, seed, save_folder)
