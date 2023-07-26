
"""Data collators for self-supervised pretext tasks in vision."""
from typing import Tuple
import numpy as np

import torch
import torch.nn.functional as F

from utils.pretext.base import BasePretexter

class JigsawPretext(BasePretexter):
    """
    Data collator used for jigsaw classification task.
    """
    def __init__(self, args):
        super().__init__(args)
        self.n_patches = args.n_patches
        self.h_patches = int(np.sqrt(self.n_patches))
        self.w_patches = self.h_patches

        # generate set of permutation indices with greatest hamming distance
        self.permutations = self.get_permutations(self.n_patches)

    def get_permutations(self, n_patches):
        """
        Generates a set of permutation indices with greatest hamming distance.
        """
        if n_patches == 4:
            start = 0
            if self.args.pretext != 'jigsaw':
                start = 1
            # all permutations of 4 numbers
            all_perms = torch.tensor([[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1], [0, 3, 1, 2], [0, 3, 2, 1],
                    [1, 0, 2, 3], [1, 0, 3, 2], [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 0, 2], [1, 3, 2, 0],
                    [2, 0, 1, 3], [2, 0, 3, 1], [2, 1, 0, 3], [2, 1, 3, 0], [2, 3, 0, 1], [2, 3, 1, 0],
                    [3, 0, 1, 2], [3, 0, 2, 1], [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 0, 1], [3, 2, 1, 0]])[start:]
            idxes = torch.randperm(len(all_perms))[:self.args.max_permutations]
            return all_perms[idxes]

        elif n_patches == 9:
            # choose max_permutations permutations with greatest hamming distance
            return torch.stack([torch.randperm(n_patches) for _ in range(self.args.max_permutations)],dim=0)

    @torch.no_grad()
    def __call__(self, examples, force_labels=None) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Apply random jigsaw to each example in batch.

        :param examples: list of examples to be mixed
        :param force_labels: if not None, use the given permutation for the examples in batch
        """
        x = examples.clone()
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        if force_labels is None:
            y_labs = torch.tensor(np.random.choice(len(self.permutations), size=examples.shape[0]), dtype=torch.int64, device=examples.device)
        else:
            y_labs = force_labels

        assert len(x.shape) == 4, 'Input must be a batch of images! (got {}, expected 4 dims)'.format(x.shape)
        assert x.shape[-1] == x.shape[-2], 'Input must be a batch of square images! (got {}, expected square)'.format(x.shape)

        # unfold examples into n_patches patches
        # B, C, H, w -> B, C, pixel_per_patch, pixel_per_patch, w_patches, h_patches
        if x.shape[-1]%self.w_patches!=0:
            pad = self.w_patches - x.shape[-1]%self.w_patches
            x = F.pad(x, (pad,0,pad,0))
        else:
            pad = 0
 
        pixel_per_patch = x.shape[-1] // self.w_patches
        x = x.unfold(2, pixel_per_patch, pixel_per_patch).unfold(3, pixel_per_patch, pixel_per_patch)
        # B, C, h_patches, w_patches, pixel_per_patch, pixel_per_patch -> B, C, n_patches, pixel_per_patch, pixel_per_patch
        x = x.reshape(x.shape[0], x.shape[1], -1, pixel_per_patch, pixel_per_patch)

        for i in range(x.shape[0]):
            y = y_labs[i] if x.shape[0]>1 else y_labs
            x[i] = x[i, :, self.permutations[y]]
        
        # B, C, n_patches, pixel_per_patch, pixel_per_patch -> B, C, h_patches, w_patches, pixel_per_patch, pixel_per_patch
        x = x.reshape(x.shape[0], x.shape[1], self.w_patches, self.h_patches, pixel_per_patch, pixel_per_patch)

        # B, C, h_patches, w_patches, pixel_per_patch, pixel_per_patch -> B, C, h_patches, pixel_per_patch, w_patches, pixel_per_patch
        x = x.permute(0, 1, 2, 4, 3, 5)

        # B, C, H, W
        x = x.reshape(x.shape[0],x.shape[1],pixel_per_patch*self.w_patches, pixel_per_patch*self.h_patches)

        # Remove pad
        x = x[:,:,pad:,pad:]
        
        return x, y_labs

    @staticmethod
    def update_parser(parser):
        parser.add_argument('--n_patches', type=int, default=4, help='number of patches to use for jigsaw pretext task',
                            choices=[4, 9, 16])
        parser.add_argument('--max_permutations', type=int, default=100, help='number of permutations to use for jigsaw pretext task')
        return parser

    def get_num_pretext_classes(self) -> int:
        """
        Returns the number of classes for the pretext task.
        """
        return len(self.permutations)