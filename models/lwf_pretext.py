# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from datasets import get_dataset
from torch.optim import SGD
from utils.args import *
from models.utils.pretext_model import PretextModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='LwF + Equivariant Pretext task')
    add_management_args(parser)
    add_experiment_args(parser)

    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--softmax_temp', type=float, default=2,
                        help='Temperature of the softmax function.')
    parser.add_argument('--wd_reg', type=float, required=True,
                        help='Coefficient of the weight decay regularizer.')

    parser.add_argument('--ptx_alpha', type=float, required=True)
    return parser


def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)


def modified_kl_div(old, new):
    return -torch.mean(torch.sum(old * torch.log(new), 1))


class LwfPretext(PretextModel):
    NAME = 'lwf_pretext'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        # Add additional heads at the specified height
        backbone = self.init_ptx_heads(args, backbone)
        super().__init__(backbone, loss, args, transform)
        self.old_net = None
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.dataset = get_dataset(args)
        self.current_task = 0
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK
        nc = get_dataset(args).N_TASKS * self.cpt
        self.eye = torch.tril(torch.ones((nc, nc))).bool().to(self.device)

        self.num_classes = nc

    def begin_task(self, dataset):
        self.net.eval()
        if self.current_task > 0:
            # warm-up
            opt = SGD(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
            for epoch in range(self.args.n_epochs):
                for i, data in enumerate(dataset.train_loader):
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                    opt.zero_grad()
                    with torch.no_grad():
                        feats = self.net.features(inputs)
                    mask = self.eye[(self.current_task + 1) * self.cpt - 1] ^ self.eye[self.current_task * self.cpt - 1]
                    outputs = self.net.classifier(feats)[:, mask]
                    loss = self.loss(outputs, labels - self.current_task * self.cpt)
                    loss.backward()
                    opt.step()

            logits = []
            with torch.no_grad():
                for i in range(0, dataset.train_loader.dataset.data.shape[0], self.args.batch_size):
                    inputs = torch.stack([dataset.train_loader.dataset.__getitem__(j)[2]
                                          for j in range(i, min(i + self.args.batch_size,
                                                         len(dataset.train_loader.dataset)))])
                    log = self.net(inputs.to(self.device)).cpu()
                    logits.append(log)
            setattr(dataset.train_loader.dataset, 'logits', torch.cat(logits))
        self.net.train()

        self.current_task += 1

    def observe(self, inputs, labels, not_aug_inputs, logits=None, epoch=None):
        labels = labels.long()
        self.opt.zero_grad()
        outputs = self.net(inputs)

        mask = self.eye[self.current_task * self.cpt - 1]
        loss = self.loss(outputs[:, mask], labels)
        if logits is not None:
            mask = self.eye[(self.current_task - 1) * self.cpt - 1]
            loss += self.args.alpha * modified_kl_div(smooth(self.soft(logits[:, mask]).to(self.device), self.args.softmax_temp, 1),
                                                      smooth(self.soft(outputs[:, mask]), self.args.softmax_temp, 1))

        loss += self.args.wd_reg * torch.sum(self.net.get_params() ** 2)
        
        ptx_inputs, ptx_labels = self.pretexter(self.base_data_aug(not_aug_inputs))
        ptx_outputs = self.get_ptx_outputs(ptx_inputs, self.net)
        loss_ptx = self.get_ce_pret_loss(ptx_outputs, ptx_labels)

        loss += self.args.ptx_alpha * loss_ptx
        loss.backward()
        self.opt.step()

        return loss.item(), 0, 0, 0, 0
