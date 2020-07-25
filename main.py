from options import args
from source.models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *

def train():

    trainer.train()

    #test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
    #if test_model:
    trainer.test()


def setup_trainer():
    export_root = setup_train(args)
    fix_random_seed_as(args.model_init_seed)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)

    return trainer

if __name__ == '__main__':

    trainer = setup_trainer()

    if 'train' == args.mode:
        trainer.train()
        trainer.test()
    elif 'test' == args.mode:
        trainer.test()
    else:
        raise ValueError('Invalid mode')
