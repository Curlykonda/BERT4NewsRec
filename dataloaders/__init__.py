from datasets import dataset_factory
from .bert import BertDataloader, BertDataloaderNews
from .npa import NpaDataloader
from .npa_mod import NpaModDataloader


DATALOADERS = {
    BertDataloader.code(): BertDataloader,
    BertDataloaderNews.code(): BertDataloaderNews,
    NpaDataloader.code(): NpaDataloader,
    NpaModDataloader.code(): NpaModDataloader
}


def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)

    return dataloader
    # train, val, test = dataloader.get_pytorch_dataloaders()
    # return train, val, test
