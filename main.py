from options import args
from source.models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *

def train():
    export_root = setup_train(args)
    fix_random_seed_as(args.model_init_seed)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)

    trainer.train()

    # if test_model:


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
