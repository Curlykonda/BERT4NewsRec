from abc import *
import random


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args

        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.umap = dataset['umap'] # mapping from u_id to index
        self.smap = dataset['smap'] # mapping from item_id to index

        if dataset['rnd'] is not None:
            # re-use Random obj
            self.rnd = dataset['rnd']
        else:
            # instantiate new Random obj
            seed = args.dataloader_random_seed
            self.rnd = random.Random(seed)

        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

        if args.n_users is not None:
            if args.n_users != self.user_count:
                print("some users did not meet all criteria and were excluded during pre-procressing.")
                print("User count: {}".format(self.user_count))
        else:
            args.n_users = self.user_count

        if args.n_articles is not None:
            assert args.n_articles == self.item_count
        else:
            args.n_articles = self.item_count



    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass
