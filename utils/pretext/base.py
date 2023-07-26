"""Data collators for self-supervised pretext tasks in vision."""
import argparse
from typing import Tuple
from abc import abstractmethod

import torch
from torch import nn

class BasePretexter(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def to(self, device):
        super().to(device)
        self.device=device
        return self

    @abstractmethod
    @torch.no_grad()
    def __call__(self, examples, force_labels=None) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        :param examples: 
        :param force_labels: if not None, use the pretext labels for the examples in batch
        """
        pass

    @abstractmethod
    def get_num_pretext_classes(self) -> int:
        """
        Returns the number of classes for the pretext task.
        """
        pass
    
    @staticmethod
    def update_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Updates the parser with the arguments required for the pretext task.
        """

        return parser


    def get_task_heads_mask(self, labels):
        mask = torch.ones((len(labels), self.get_num_pretext_classes()))

        return mask.to(labels.device).bool()
