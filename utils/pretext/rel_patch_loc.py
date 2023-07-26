"""Data collators for self-supervised pretext tasks in vision."""
from typing import Tuple
import numpy as np

import torch
import torch.nn.functional as F

from utils.pretext.base import BasePretexter


class RelPatchLocPretext(BasePretexter):
    """
    Data collator used for jigsaw classification task.
    """
    def __init__(self, args):
        super().__init__(args)
        self.n_patches = args.n_patches
        self.perc_patch_to_switch = args.perc_patch_to_switch
        self.n_patches_to_switch = int(self.n_patches*self.perc_patch_to_switch)
        assert self.n_patches_to_switch > 0, "Number of patches to switch must be greater than 0"
        self.h_patches = int(np.sqrt(self.n_patches))
        self.w_patches = self.h_patches

        assert not (self.args.cross == 1 and self.args.binary == 1)

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
            orig_labs = torch.arange(self.n_patches, dtype=torch.int64).expand(x.shape[0], self.n_patches)
            patches_to_switch_from = np.apply_along_axis(lambda x:x[np.random.permutation(x.shape[0])][:self.n_patches_to_switch], 1, orig_labs)

            if self.args.close_only==0:
                patches_to_switch_to = np.apply_along_axis(lambda x:x[np.random.permutation(x.shape[0])][:self.n_patches_to_switch], 1, patches_to_switch_from)
            else:
                w_directions = np.random.choice([-1,1], size=(x.shape[0], self.n_patches))
                h_directions = np.random.choice([-1,1], size=(x.shape[0], self.n_patches))
                has_sx = orig_labs%self.w_patches==0
                has_dx = orig_labs%self.w_patches==self.w_patches-1
                has_up = orig_labs//self.w_patches==0
                has_dw = orig_labs//self.w_patches==self.h_patches-1

                w_directions[has_sx] = np.abs(w_directions[has_sx])
                w_directions[has_dx] = -np.abs(w_directions[has_dx])
                h_directions[has_up] = np.abs(h_directions[has_up])
                h_directions[has_dw] = -np.abs(h_directions[has_dw])

                w_or_h = np.random.choice([0,1], size=(x.shape[0], self.n_patches))

                offset = w_directions*w_or_h + (h_directions*int(np.sqrt(self.n_patches)))*(1-w_or_h)

                patches_to_switch_to = patches_to_switch_from + np.take(offset, patches_to_switch_from)

            patches_to_switch_from = torch.from_numpy(patches_to_switch_from).long() #.to(examples.device)
            patches_to_switch_to = torch.from_numpy(patches_to_switch_to).long() #.to(examples.device)

            # y_labs = F.one_hot(patches_to_switch, num_classes=self.n_patches).sum(dim=1)
            tmp_labs = orig_labs.clone() # to(x.device)
            tmp_labs.scatter_(1, patches_to_switch_from, patches_to_switch_to)

            if self.args.drop_others:
                diff_mask = (tmp_labs != orig_labs)
                n_patch_row = int(np.sqrt(tmp_labs.shape[1]))

                diff_mask[(orig_labs%n_patch_row==tmp_labs%n_patch_row)] = 0 # same col
                diff_mask[(orig_labs//n_patch_row==tmp_labs//n_patch_row)] = 0 # same row
                
                tmp_labs[diff_mask] = orig_labs[diff_mask]

            y_labs = tmp_labs.to(x.device)

            # y_labs = F.one_hot(orig_labs, num_classes=self.n_patches).sum(dim=1)
        else:
            y_labs = force_labels
            orig_labs = torch.arange(self.n_patches)
            label_cng = orig_labs!=y_labs

            patches_to_switch_from = orig_labs[label_cng]
            patches_to_switch_to = y_labs[label_cng]

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

        x[patches_to_switch_to.unsqueeze(1)] = x[patches_to_switch_from.unsqueeze(1)]
        
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
        parser.add_argument('--binary', type=int, default=0, help='classify only patch change?',
                            choices=[0,1])
        parser.add_argument('--cross', type=int, default=0, help='classify easier relative positin?',
                            choices=[0,1])
        parser.add_argument('--drop_others', type=int, default=0, help='drop others outside other row/col?',
                            choices=[0,1])
        parser.add_argument('--close_only', type=int, default=0, help='change position between close patches only',
                            choices=[0,1])
        parser.add_argument('--perc_patch_to_switch', type=float, default=0.5, help='percentage of patches to switch')
        return parser

    def get_num_pretext_classes(self) -> int:
        """
        Returns the number of classes for the pretext task.
        """
        return self.n_patches