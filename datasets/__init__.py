from datasets.seq_cifar100_online import SequentialCIFAR100Online
from datasets.seq_miniimagenet_online import SequentialMiniImagenetOnline
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace

NAMES = {
    SequentialCIFAR100Online.NAME: SequentialCIFAR100Online,
    SequentialMiniImagenetOnline.NAME: SequentialMiniImagenetOnline
}

GCL_NAMES = {}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)


def get_gcl_dataset(args: Namespace):
    """
    Creates and returns a GCL dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in GCL_NAMES.keys()
    return GCL_NAMES[args.dataset](args)
