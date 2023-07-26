import torch.nn as nn
import torch
import torchvision
from argparse import Namespace

from datasets import get_dataset
from utils.conf import get_device
from torchvision import transforms

class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def reset_classifier(self, inherit=False, bias=True):
        if not inherit or self.net.num_classes < self.net.classifier.out_features:
            self.net.classifier = torch.nn.Linear(
                    self.net.classifier.in_features, self.net.num_classes,bias=bias).to(self.device)
        else:
            # inherit weights
            old_weights, old_bias = self.net.classifier.weight.data, self.net.classifier.bias.data if bias else None
            self.net.classifier = torch.nn.Linear(
                    self.net.classifier.in_features, self.net.num_classes, bias=bias).to(self.device)
            self.net.classifier.weight.data[:old_weights.shape[0]] = old_weights
            if bias:
                self.net.classifier.bias.data[:old_weights.shape[0]] = old_bias
            
        self.reset_opt()

    def get_scheduler(self):
        sched = None
        if self.args.scheduler == 'simple':
            assert self.args.scheduler_rate is not None
            self.reset_opt()

        return sched

    def to(self, device):
        super().to(device)
        self.device = device
        for d in [x for x in self.__dir__() if hasattr(getattr(self, x), 'device')]:
            getattr(self, d).to(device)

        return self

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                 args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()
        
        ds = get_dataset(args)
        self.cpt = ds.N_CLASSES_PER_TASK
        self.n_tasks = ds.N_TASKS
        self.num_classes = self.n_tasks * self.cpt

        self.train_transform = ds.TRANSFORM
        self.test_transform = ds.TEST_TRANSFORM if hasattr(ds, 'TEST_TRANSFORM') else transforms.Compose(
            [transforms.ToTensor(), ds.get_normalization_transform()])

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.reset_opt()
        self.device = get_device()
        self._partial_train_transform = transforms.Compose(self.train_transform.transforms[-1:])
        self.orig_classes = self.net.num_classes

    def reset_opt(self):
        self.opt = get_dataset(self.args).get_optimizer(self.net.parameters(), self.args)
        return self.opt

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x, **kwargs)

    @torch.no_grad()
    def apply_transform(self, inputs, transform, device=None, add_pil_transforms=True):
        tr = transforms.Compose([transforms.ToPILImage()] + transform.transforms) if add_pil_transforms else transform
        device = self.device if device is None else device
        if len(inputs.shape) == 3:
            return tr(inputs)
        return torch.stack([tr(inp) for inp in inputs.cpu()], dim=0).to(device)

    @torch.no_grad()
    def aug_batch(self, not_aug_inputs, device=None):
        """
        Full train transform 
        """
        return self.apply_transform(not_aug_inputs, self.train_transform, device=device)

    @torch.no_grad()
    def base_data_aug(self, inputs, device=None):
        """
        Base transform: totensor + normalization
        """
        return self.apply_transform(inputs, self._partial_train_transform, device=device, add_pil_transforms=False)

    @torch.no_grad()
    def test_data_aug(self, inputs, device=None):
        """
        Test transform
        """
        return self.apply_transform(inputs, self.test_transform, device=device)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass