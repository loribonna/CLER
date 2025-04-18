from argparse import Namespace
import torch
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from utils.conf import base_path
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize

from torchvision.datasets import CIFAR100
from PIL import Image

class TCIFAR100(CIFAR100):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR100, self).__init__(root, train, transform, target_transform, download)

class MyCIFAR100(CIFAR100):
    N_CLASSES = 100    
    
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        
        super(MyCIFAR100, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int):
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img

class SequentialCIFAR100OnlineOnproto(ContinualDataset):

    NAME = 'seq-cifar100-online-onproto'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    TRANSFORM = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.49137255, 0.48235294, 0.44666667),
                                  (0.24705882, 0.24352941, 0.26156863))])

    def get_examples_number(self):
        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                  download=True)
        return len(train_dataset.data)

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                  download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = TCIFAR100(base_path() + 'CIFAR100',train=False,
                                   download=True, transform=test_transform)

        class_order = None

        train, test = store_masked_loaders(train_dataset, test_dataset, self, class_order)
        
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR100OnlineOnproto.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR100OnlineOnproto.N_CLASSES_PER_TASK
                        * SequentialCIFAR100OnlineOnproto.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_setting():
        return Namespace(**{
            "batch_size":10,
            "minibatch_size":10,
            "n_epochs":1})