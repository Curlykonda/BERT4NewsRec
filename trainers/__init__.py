from .bert import *
# from .dae import DAETrainer
# from .vae import VAETrainer

TRAINERS = {
    BERTTrainer.code(): BERTTrainer,
    BERT4NewsCategoricalTrainer.code(): BERT4NewsCategoricalTrainer,
    Bert4NewsDistanceTrainer.code(): Bert4NewsDistanceTrainer
}

def trainer_factory(args, model, train_loader, val_loader, test_loader, export_root):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader, export_root)
