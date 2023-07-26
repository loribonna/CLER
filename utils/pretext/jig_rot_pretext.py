"""Data collators for self-supervised pretext tasks in vision."""
import numpy as np
import argparse
from copy import deepcopy
from typing import Tuple
from abc import abstractmethod

import torch
from utils.pretext.base import BasePretexter
from utils.pretext.jigsaw import JigsawPretext

from utils.pretext.rotation import RotationPretext

class JigRotPretext(BasePretexter):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        self.rotator = RotationPretext(args)
        self.jigsaw = JigsawPretext(args)

        self.n_rot_classes = self.rotator.get_num_pretext_classes()
        self.n_jig_classes = self.jigsaw.get_num_pretext_classes()

    def to(self, device):
        super().to(device)
        self.rotator.to(device)
        self.jigsaw.to(device)
        return self
    
    @torch.no_grad()
    def __call__(self, examples, force_labels=None) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Apply random aug from all pretext tasks available.

        :param examples: list of examples to be augmented
        :param force_labels: if not None, use the pretext labels for the examples in batch
        """
        if force_labels is None:
            y_labs_rot = torch.tensor(np.random.choice(self.n_rot_classes, size=examples.shape[0]), dtype=torch.int64, device=examples.device)
            y_labs_jig = torch.tensor(np.random.choice(self.n_jig_classes, size=examples.shape[0]), dtype=torch.int64, device=examples.device)
        else:
            y_labs_jig, y_labs_rot = force_labels.split(len(force_labels)//2, dim=0)
        
        x_rot = examples.clone()
        x_jig = examples.clone()
        if len(examples.shape) == 3:
            x_rot = x_rot.unsqueeze(0)
            x_jig = x_jig.unsqueeze(0)

        x_rot = self.rotator(x_rot, force_labels=y_labs_rot)[0]
        x_jig = self.jigsaw(x_jig, force_labels=y_labs_jig)[0]

        x = torch.cat((x_jig, x_rot), dim=0)
        y_labs = torch.cat((y_labs_jig, y_labs_rot), dim=0)

        return x, y_labs

    def get_num_pretext_classes(self) -> int:
        """
        Returns the number of classes for the pretext task.
        """
        return self.n_rot_classes + self.n_jig_classes
    
    @staticmethod
    def update_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Updates the parser with the arguments required for the pretext task.
        """
        parser = RotationPretext.update_parser(parser)
        parser = JigsawPretext.update_parser(parser)

        return parser