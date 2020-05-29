from .bert import BERTModel, BERT4NewsRecModel, Bert4NextItemEmbedPrediction
from .NPA import VanillaNPA
from source.utils import init_weights

# from .dae import DAEModel
# from .vae import VAEModel

MODELS = {
    BERTModel.code(): BERTModel,
    VanillaNPA.code(): VanillaNPA,
    BERT4NewsRecModel.code(): BERT4NewsRecModel,
    Bert4NextItemEmbedPrediction.code(): Bert4NextItemEmbedPrediction
}

def model_factory(args):
    model = MODELS[args.model_code]
    model = model(args)
    n_params = 0
    for m in model.parameters():
        if m.dim() > 1:
            init_weights(m)
        if m.requires_grad:
            n_params += m.numel()
    print("Number of trainable parameters: {}".format(n_params))
    return model
