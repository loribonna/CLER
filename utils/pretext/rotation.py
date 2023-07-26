"""Data collators for self-supervised pretext tasks in vision."""
from typing import Tuple
import numpy as np

import torch
import torch.nn.functional as F

from utils.pretext.base import BasePretexter

class RotationPretext(BasePretexter):
    """
    Data collator used for rotation classification task.
    """
    def __init__(self, args):
        super().__init__(args)
        self.n_rotations = args.n_rotations
        rotation_degrees = torch.from_numpy(np.linspace(0, 360, args.n_rotations + 1)[:-1])
        self.register_buffer('rotation_degrees', rotation_degrees)

    @torch.no_grad()
    def __call__(self, examples, force_labels=None) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Apply random rotation to each example in batch.

        :param examples: list of examples to be rotated
        :param force_labels: if not None, use the rotation angles for the examples in batch
        """
        x = examples.clone() 
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        if force_labels is None:
            rot_y = torch.tensor(np.random.choice(self.n_rotations, size=x.shape[0]), dtype=torch.int64, device=x.device)
        else:
            rot_y = force_labels
        degrees = self.rotation_degrees[rot_y]
        rotation_matrix = self.get_rotation_matrix(theta=degrees).type(examples.dtype)
        grid = F.affine_grid(rotation_matrix, x.shape, align_corners=True).type(examples.dtype).to(x.device)
        x = F.grid_sample(x, grid, align_corners=True)

        return x, rot_y

    @staticmethod
    def get_rotation_matrix(theta, mode='degrees'):
        """
        Computes and returns the rotation matrix.
        :param (int or float) theta: integer angle value for `mode`='degrees' and float angle value
               for `mode`='radians'
        :param (str) mode: set to 'degrees' or 'radians'
        :return: rotation matrix
        """
        assert mode in ['degrees', 'radians']

        if mode == 'degrees':
            theta *= np.pi / 180

        zeros = torch.zeros_like(theta)
        cos, sin = torch.cos(theta), torch.sin(theta)
        mx = torch.stack([torch.stack([cos, -sin, zeros],dim=0), torch.stack([sin, cos, zeros],dim=0)],dim=0).permute(2,0,1)
        return mx

    def get_num_pretext_classes(self) -> int:
        """
        Returns the number of classes for the pretext task.
        """
        return self.n_rotations

    @staticmethod
    def update_parser(parser):
        """
        Updates the parser with the arguments required for the pretext task.
        """
        parser.add_argument('--n_rotations', type=int, default=4, choices=[2,4,8])
        return parser