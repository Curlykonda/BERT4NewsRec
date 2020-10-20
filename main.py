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
    dataloader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, dataloader, export_root)

    return trainer

if __name__ == '__main__':

    trainer = setup_trainer()

    if 'train' == args.mode:
        trainer.train()
        trainer.test()
    elif 'test' == args.mode:
        trainer.test()
    elif 'eval_users' == args.mode:
        trainer.detail_eval_users()
    elif 'mod_query_time' == args.mode:
        trainer.eval_mod_query_time()
    elif "eval_embs" == args.mode:
        trainer.eval_order_embs()
    else:
        raise ValueError('Invalid mode')
