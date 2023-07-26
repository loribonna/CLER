# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.args import *
from models.utils.pretext_model import PretextModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')

    parser.add_argument('--ptx_alpha', type=float, required=True)

    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class SgdPretext(PretextModel):
    NAME = 'sgd_pretext'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        # Add additional heads at the specified height
        backbone = self.init_ptx_heads(args, backbone)

        super(SgdPretext, self).__init__(backbone, loss, args, transform)
        self.current_task = 0

    def end_task(self, dataset):
        self.current_task += 1

    def observe(self, inputs, labels, not_aug_inputs, epoch=0):
        labels = labels.long()
        self.opt.zero_grad()
        outputs = self.net(inputs)

        ptx_inputs, ptx_labels = self.pretexter(self.base_data_aug(not_aug_inputs))
        ptx_outputs = self.get_ptx_outputs(ptx_inputs, self.net)
        loss_ptx = self.get_ce_pret_loss(ptx_outputs, ptx_labels)

        loss = self.loss(outputs, labels)
        loss += self.args.ptx_alpha * loss_ptx
        loss.backward()
        self.opt.step()

        return loss.item(),0,0,0,0