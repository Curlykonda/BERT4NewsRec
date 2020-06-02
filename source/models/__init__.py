from .bert import BERT4RecModel, BERT4NewsRecModel, BERT4NewsRecModel
from .NPA import VanillaNPA
from source.utils import init_weights

# from .dae import DAEModel
# from .vae import VAEModel

MODELS = {
    BERT4RecModel.code(): BERT4RecModel,
    VanillaNPA.code(): VanillaNPA,
    BERT4NewsRecModel.code(): BERT4NewsRecModel,
    BERT4NewsRecModel.code(): BERT4NewsRecModel
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
    # format params:
    if n_params > 1e6:
        n_params = "{:.3f} M".format(n_params / 1e6)
    elif n_params > 1e3:
        n_params = "{:.2f} k".format(n_params / 1e3)

    print("Number of trainable parameters: {}".format(n_params))
    return model
