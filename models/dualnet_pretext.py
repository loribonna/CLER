from copy import deepcopy
import torch
from torch import nn
from torch.optim import SGD
from backbone.DualnetNetNet import DualnetNetNet18
from models.utils.pretext_model import PretextModel, features_hook
from utils.buffer import Buffer
from utils.args import *
from datasets import get_dataset
import torch.nn.functional as F
from models.dualnet import add_dualnet_args
from utils.pretext import get_pretext_task

def batch_iterate(size: int, batch_size: int):
    n_chunks = size // batch_size
    for i in range(n_chunks):
        yield torch.LongTensor(list(range(i * batch_size, (i + 1) * batch_size)))

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='DualNet + Equivariant pretext task')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--ptx_alpha', type=float, required=True)

    add_dualnet_args(parser)

    return parser

def init_ptx_heads_dualnet(module, args, backbone):
    bname = type(backbone).__name__.lower()
    assert "resnet" in bname or "dualnetnet" in bname, "Ptx heads only implemented for resnet, got {}".format(type(backbone).__name__)

    dset = get_dataset(args)
    x = dset.get_data_loaders()[0].dataset[0][0]

    with torch.no_grad():
        features_shape = backbone(x.unsqueeze(0), returnt="full")[-1][-2].shape

    module.child_index_start, module.child_index_end = 6, 7
    
    module.child_index_start, module.child_index_end = module.child_index_start + 8, module.child_index_end + 8

    backbone.ptx_net = nn.Sequential(
        *deepcopy(list(backbone.children())[module.child_index_start:module.child_index_end]), 
        nn.AvgPool2d(features_shape[-1]),
        nn.Flatten(),
        nn.Linear(backbone.nf * 8 * backbone.block.expansion, get_pretext_task(args).get_num_pretext_classes())
        )

    module.hook = list(backbone.children())[module.child_index_start-1].register_forward_hook(features_hook)
        
    return backbone

class DualNetPretext(PretextModel):
    NAME = 'dualnet_pretext'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        ds = get_dataset(args)
        num_classes = ds.N_CLASSES_PER_TASK * ds.N_TASKS

        backbone = DualnetNetNet18(num_classes, nf=20)
        backbone.num_classes = num_classes

        backbone = init_ptx_heads_dualnet(self, args, backbone)

        super().__init__(
            backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.slowiter_opt = SGD(self.net.parameters(), lr=self.args.lr_teacher, weight_decay=args.optim_wd, momentum=args.optim_mom)

        self.task = 0

        self.selfsup_transform = ds.TRANSFORM

        print("\n")

    @torch.no_grad()
    def end_task(self, dataset):
        self.task += 1

    def slownet_train_iter(self, not_aug_inputs):
        was_training = self.net.training
        weights_before = deepcopy(self.net.state_dict())
        
        self.net.train() # always update BN even if disable_train

        for _ in range(self.args.n_inner):
            if self.task > 0:
                buf_choices = self.buffer.get_data(32, transform=None, return_index=True, filter_c_task=self.task)[0]
                buf_not_aug_inputs = self.buffer.examples[buf_choices]
                inputs = buf_not_aug_inputs
            else:
                inputs = not_aug_inputs

            x1 = self.apply_transform(inputs, self.selfsup_transform, device=self.device, add_pil_transforms=True)
            x2 = self.apply_transform(inputs, self.selfsup_transform, device=self.device, add_pil_transforms=True)
            x1, x2 = x1.to(self.device), x2.to(self.device)
            
            loss = self.net.BarlowTwins(x1, x2)

            self.slowiter_opt.zero_grad()
            loss.backward()
            self.slowiter_opt.step()

        weights_after = self.net.state_dict()

        new_params_net = {name : weights_before[name] + ((weights_after[name] - weights_before[name]) * self.args.slownet_beta) for name in weights_before.keys()}
        self.net.load_state_dict(new_params_net)

        self.net.train(was_training)

        return loss

    def begin_task(self, dataset):
        if self.task > 0 and ("start_from" not in self.args or self.args.start_from is None or self.task != self.args.start_from):
            self.net.eval()
            with torch.no_grad():
                for chunk in batch_iterate(self.buffer.examples.__len__(), self.args.batch_size):
                    inputs = self.test_data_aug(self.buffer.examples[chunk], device=self.device)
                    self.buffer.logits[chunk] = self.net(inputs).detach().data

        self.reset_opt()
        
        self.net.train()
    
    def fast_net_iter(self, not_aug_inputs, labels):
        self.net.train()

        inputs = self.test_data_aug(not_aug_inputs) # train with test aug

        stream_outputs = self.net(inputs)
        loss_stream = self.loss(stream_outputs, labels)

        all_inputs = not_aug_inputs
        if self.task>0:
            # sample from buffer
            with torch.no_grad():
                buf_not_aug_inputs, buf_labels, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=None, filter_c_task=self.task)[:3]
                all_inputs = torch.cat([all_inputs, buf_not_aug_inputs], dim=0)
                buf_inputs = self.test_data_aug(buf_not_aug_inputs)

            buf_pred = self.net(buf_inputs)
            loss_ce = self.loss(buf_pred, buf_labels)

            loss_reg = F.kl_div(
                F.log_softmax(buf_pred / self.args.temp_reg, dim=-1), F.softmax(buf_logits / self.args.temp_reg, dim=-1)
                , reduce=True) * buf_pred.size(0)

            loss = loss_stream + loss_ce + self.args.alpha_reg * loss_reg
        else:
            loss = loss_stream

        ptx_inputs, ptx_labels = self.pretexter(self.base_data_aug(all_inputs))
        ptx_outputs = self.get_ptx_outputs(ptx_inputs, self.net)
        loss_ptx = self.get_ce_pret_loss(ptx_outputs, ptx_labels)

        loss += self.args.ptx_alpha * loss_ptx

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss, stream_outputs

    def forward(self, inputs):
        self.net.eval()
        
        return self.net(inputs)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        labels = labels.long()
        B = len(not_aug_inputs)

        for _ in range(self.args.n_outer):
            loss_slownet = self.slownet_train_iter(not_aug_inputs)
            

        for _ in range(self.args.n_inner):
            loss_fastnet, stream_logits = self.fast_net_iter(not_aug_inputs, labels)

        

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=stream_logits.detach().data,
                             task_labels=(self.task - 1) * torch.ones(B).long().to(self.device))


        return loss_fastnet.item(), 0, 0, 0, 0
