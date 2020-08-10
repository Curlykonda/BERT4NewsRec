from .popular import *
from .random import *


NEGATIVE_SAMPLERS = {
    RandomNegativeSamplerPerUser.code(): RandomNegativeSamplerPerUser,
    RandomFromCommonNegativeSampler.code(): RandomFromCommonNegativeSampler,
    PopularLikelihoodNegativeSampler.code(): PopularLikelihoodNegativeSampler,
    PopularNaiveNegativeSampler.code(): PopularNaiveNegativeSampler,
    RandomBrandSensitiveNegativeSampler.code(): RandomBrandSensitiveNegativeSampler
}

def negative_sampler_factory(code, **kwargs):
    negative_sampler = NEGATIVE_SAMPLERS[code]
    return negative_sampler(**kwargs)
