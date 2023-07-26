"""Data collators for self-supervised pretext tasks in vision."""
from typing import Tuple
import numpy as np

import torch
import torch.nn.functional as F

from utils.pretext.base import BasePretexter

class VFlipPretext(BasePretexter):
    """
    Data collator used for vertical flip classification task.
    """
    def __init__(self, args):
        super().__init__(args)
        self.n_flip = 2

    @torch.no_grad()
    def __call__(self, examples, force_labels=None) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Apply random vertical flip to each example in batch.

        :param examples: list of examples to be flipped
        :param force_labels: if not None, use the rotation angles for the examples in batch
        """
        x = examples.clone() 
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        if force_labels is None:
            flip_y = torch.tensor(np.random.choice(self.n_flip, size=x.shape[0]), dtype=torch.int64, device=x.device)
        else:
            flip_y = force_labels
        # degrees = self.rotation_degrees[rot_y]
        # rotation_matrix = self.get_rotation_matrix(theta=degrees).type(examples.dtype)
        # grid = F.affine_grid(rotation_matrix, x.shape, align_corners=True).type(examples.dtype).to(x.device)
        # x = F.grid_sample(x, grid, align_corners=True)

        # Vertical flip the image only where flip_y is 1
        x[flip_y.bool()] = torch.flip(x[flip_y.bool()], dims=[2])

        return x, flip_y

    def get_num_pretext_classes(self) -> int:
        """
        Returns the number of classes for the pretext task.
        """
        return self.n_flip

    @staticmethod
    def update_parser(parser):
        """
        Updates the parser with the arguments required for the pretext task.
        """
        return parser