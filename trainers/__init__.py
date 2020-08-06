from .bert import *
from .npa import *
# from .dae import DAETrainer
# from .vae import VAETrainer

TRAINERS = {
    BERTTrainer.code(): BERTTrainer,
    BERT4NewsCategoricalTrainer.code(): BERT4NewsCategoricalTrainer,
    Bert4NewsDistanceTrainer.code(): Bert4NewsDistanceTrainer,
    NpaTrainer.code(): NpaTrainer,
    NpaModTrainer.code(): NpaModTrainer
}

def trainer_factory(args, model, dataloader, export_root):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, dataloader, export_root)
