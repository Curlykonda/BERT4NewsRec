from .bert import BERTModel, BERT4NewsRecModel
from .NPA import VanillaNPA
# from .dae import DAEModel
# from .vae import VAEModel

MODELS = {
    BERTModel.code(): BERTModel,
    VanillaNPA.code(): VanillaNPA,
    BERT4NewsRecModel.code(): BERT4NewsRecModel
    # DAEModel.code(): DAEModel,
    # VAEModel.code(): VAEModel
}


def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)
