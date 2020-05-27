from .ml_1m import ML1MDatasetML
from .ml_20m import ML20MDatasetML
from .dpg import *

DATASETS = {
    ML1MDatasetML.code(): ML1MDatasetML,
    ML20MDatasetML.code(): ML20MDatasetML,
    DPG_Nov19Dataset.code(): DPG_Nov19Dataset

}

#DPG_Feb20_med


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
