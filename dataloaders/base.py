from abc import *
import random

from source.utils import reverse_mapping_dict

class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args

        self.save_folder = dataset._get_preprocessed_folder_path()

        self.dataset_instance = dataset
        data = dataset.load_dataset()
        self.train = data['train']
        self.val = data['val']
        self.test = data['test']
        self.u_id2idx = data['umap'] # mapping from u_id to index
        self.idx2u_id = reverse_mapping_dict(data['umap'])

        self.item_id2idx = data['smap'] # mapping from item_id to index
        self.idx2item_id = reverse_mapping_dict(data['smap'])

        if 'art_id2info' in data:
            self.item_id2info = data['art_id2info']


        if data['rnd'] is not None:
            # re-use Random obj
            self.rnd = data['rnd']
        else:
            # instantiate new Random obj
            seed = args.dataloader_random_seed
            self.rnd = random.Random(seed)

        self.user_count = len(self.u_id2idx)
        self.item_count = len(self.item_id2idx)

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
            #args.num_items = self.item_count


    @classmethod
    @abstractmethod
    def code(cls):
        pass

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def get_data(self):
        return self.dataset_instance.load_dataset()

    @abstractmethod
    def _get_train_loader(self):
        pass

    @abstractmethod
    def _get_train_dataset(self):
        pass

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    @abstractmethod
    def _get_eval_loader(self, mode):
        pass

    @abstractmethod
    def _get_eval_dataset(self, mode):
        pass
