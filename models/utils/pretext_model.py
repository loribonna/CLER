from copy import deepcopy
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms

from datasets import get_dataset
from models.utils.continual_model import ContinualModel
from utils.no_bn import bn_track_stats
from utils.pretext import get_pretext_task
from utils.pretext.jigsaw import JigsawPretext
from utils.pretext.rotation import RotationPretext

split_height_resnet = {
    "l4": [6,7],
    "cl": [7,7]
}

def features_hook(module, input, output):
    """
    Hook to save the output of the features
    """
    module.partial_features = output

class PretextModel(ContinualModel):

    def __init__(self, backbone, loss, args, transform):
        super(PretextModel, self).__init__(backbone, loss, args, transform)
        assert args.pretext is not None, "Pretext model needs pretext task"

        self.pretexter = get_pretext_task(args).to(self.device)
        self._partial_train_transform = transforms.Compose(self.train_transform.transforms[-1:])

        self.current_task = 0
        self.eye = torch.eye(self.pretexter.get_num_pretext_classes())
        
    def init_ptx_heads(self, args, backbone) -> nn.Module:
        bname = type(backbone).__name__.lower()
        assert "resnet" in bname or "dualnetnet" in bname, "Ptx heads only implemented for resnet, got {}".format(type(backbone).__name__)

        dset = get_dataset(args)
        x = dset.get_data_loaders()[0].dataset[0][0]
        with torch.no_grad():
            features_shape = backbone(x.unsqueeze(0), returnt="full")[-1][-2].shape

        self.child_index_start, self.child_index_end = split_height_resnet[args.split_height] if hasattr(args, "split_height") else (6, 7)
        
        if args.pretext == 'jig_rot':
            backbone.ptx_net_rot = nn.Sequential(
                *deepcopy(list(backbone.children())[self.child_index_start:self.child_index_end]), 
                nn.AvgPool2d(features_shape[-1]),
                nn.Flatten(),
                nn.Linear(backbone.nf * 8 * backbone.block.expansion, get_pretext_task(args).n_rot_classes)
                )
            backbone.ptx_net_jig = nn.Sequential(
                *deepcopy(list(backbone.children())[self.child_index_start:self.child_index_end]), 
                nn.AvgPool2d(features_shape[-1]),
                nn.Flatten(),
                nn.Linear(backbone.nf * 8 * backbone.block.expansion, get_pretext_task(args).n_jig_classes)
                )
        elif args.pretext == "rel_patch_loc":
            backbone.ptx_net = nn.Sequential(
                *deepcopy(list(backbone.children())[self.child_index_start:self.child_index_end]), 
                nn.AdaptiveAvgPool2d(int(np.sqrt(args.n_patches))),
                # nn.Flatten(),
                nn.Conv2d(in_channels=backbone.nf * 8 * backbone.block.expansion, 
                          out_channels=get_pretext_task(args).get_num_pretext_classes(),
                          kernel_size=1,stride=1)
                )
        else:
            backbone.ptx_net = nn.Sequential(
                *deepcopy(list(backbone.children())[self.child_index_start:self.child_index_end]), 
                nn.AvgPool2d(features_shape[-1]),
                nn.Flatten(),
                nn.Linear(backbone.nf * 8 * backbone.block.expansion, get_pretext_task(args).get_num_pretext_classes())
                )


        self.hook = list(backbone.children())[self.child_index_start-1].register_forward_hook(features_hook)

        return backbone

    def _compute_ptx_outputs(self, inputs, net, ptx_net=None):
        with bn_track_stats(net, False):
            _ = net(inputs)
        c = list(net.children())
        stream_partial_features = c[self.child_index_start-1].partial_features

        return ptx_net(stream_partial_features)

    def get_ptx_outputs(self, ptx_inputs, net=None):
        net = self.net if net is None else net
        if self.args.pretext == 'jig_rot':
            inputs_jig, inputs_rot = ptx_inputs.split(len(ptx_inputs)//2,dim=0)
            outs_jig = self._compute_ptx_outputs(inputs_jig, net, self.net.ptx_net_jig)
            outs_rot = self._compute_ptx_outputs(inputs_rot, net, self.net.ptx_net_rot)

            return outs_jig, outs_rot
        else:
            return self._compute_ptx_outputs(ptx_inputs, net, self.net.ptx_net)

    def get_ce_pret_loss(self, logits, labels):
        if self.args.pretext == "jig_rot":
            jig_logits, rot_logits = logits[0], logits[1]
            jig_labels, rot_labels = labels.split(len(labels)//2, dim=0)
            return (F.cross_entropy(jig_logits, jig_labels) + F.cross_entropy(rot_logits, rot_labels))/2
        
        if self.args.pretext == "rel_patch_loc":
            # logits = logits.reshape(-1, self.args.n_patches, self.args.n_patches)
            logits=logits.reshape(-1, self.args.n_patches, self.args.n_patches)
            if self.args.binary==1:
                labs = (labels != torch.arange(labels.shape[1]).expand_as(labels).to(labels.device)).long()
                return F.cross_entropy(logits, labs)     
            elif self.args.cross==1 or self.args.close_only==1:
                orig_labs = torch.arange(labels.shape[1]).expand_as(labels).to(labels.device)
                labels = labels.long()
                diff_mask = (labels != orig_labs)
                labs = torch.zeros_like(labels).to(labels.device)
                n_patch_row = int(np.sqrt(labels.shape[1]))

                labs[diff_mask] = 3 # default
                labs[(orig_labs%n_patch_row==labels%n_patch_row)&diff_mask] = 1 # same col
                labs[(orig_labs//n_patch_row==labels//n_patch_row)&diff_mask] = 2 # same row
                return F.cross_entropy(logits, labs.long())
            
        return F.cross_entropy(logits, labels)