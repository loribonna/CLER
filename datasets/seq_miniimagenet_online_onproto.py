import numpy as np
from argparse import Namespace
import torchvision.transforms as transforms
from backbone.ResNet18 import lopeznet
import torch.nn.functional as F
from utils.conf import base_path
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from PIL import Image
import os
from torch.utils.data import Dataset

class MiniImagenet(Dataset):
    """
    Defines Mini Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from onedrivedownloader import download
                ln = 'https://unimore365-my.sharepoint.com/:u:/g/personal/215580_unimore_it/Ecbo1YCVHCFBhQhcuHAufeoBGPJ3jUfOv7BUdC3C88E8Iw?e=FYLgHN'
                download(ln, filename=os.path.join(root, 'miniimagenet.zip'), unzip=True, unzip_path=root, clean=True)
                

        self.data = np.load(os.path.join(
                root, '%s_x.npy' %
                      ('train' if self.train else 'test')))
        self.targets = np.load(os.path.join(
                root, '%s_y.npy' %
                      ('train' if self.train else 'test')))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target


class MyMiniImagenet(MiniImagenet):
    """
    Defines Mini Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        super(MyMiniImagenet, self).__init__(
            root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
          return img, target, not_aug_img, self.logits[index]

        return img, target,  not_aug_img

class SequentialMiniImagenetOnlineOnproto(ContinualDataset):

    NAME = 'seq-miniimg-online-onproto'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 5
    N_TASKS = 20
    MEAN = (0.5, 0.5, 0.5)
    STD = (0.5, 0.5, 0.5)
    TRANSFORM = transforms.Compose(
            [transforms.Resize((84,84)),
             transforms.ToTensor(),
             transforms.Normalize(MEAN,
                                  STD)])

    TEST_TRANSFORM = transforms.Compose(
        [transforms.Resize((84,84)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN,  STD)])

    def get_data_loaders(self):
        transform = self.TRANSFORM
        test_transform = self.TEST_TRANSFORM

        train_dataset = MyMiniImagenet(base_path() + 'MINIIMG',
                                 train=True, download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = MiniImagenet(base_path() + 'MINIIMG',
                        train=False, download=True, transform=test_transform)

        class_order = None

        train, test = store_masked_loaders(train_dataset, test_dataset, self, class_order=class_order)
        return train, test


    @staticmethod
    def get_backbone():
        return lopeznet(SequentialMiniImagenetOnlineOnproto.N_CLASSES_PER_TASK
                        * SequentialMiniImagenetOnlineOnproto.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialMiniImagenetOnlineOnproto.MEAN,
                                         SequentialMiniImagenetOnlineOnproto.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialMiniImagenetOnlineOnproto.MEAN,
                                SequentialMiniImagenetOnlineOnproto.STD)
        return transform

    @staticmethod
    def get_setting():
        return Namespace(**{
            "batch_size":10,
            "minibatch_size":10,
            "n_epochs":1})
