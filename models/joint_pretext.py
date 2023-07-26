# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

from tqdm import tqdm
from utils.training import evaluate

from utils.args import *
from models.utils.pretext_model import PretextModel
import torch
import numpy as np

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training + Equivariant pretext task')
    add_management_args(parser)
    add_experiment_args(parser)

    parser.add_argument('--ptx_alpha', type=float, required=True)

    return parser


class JointPretext(PretextModel):
    NAME = 'joint_pretext'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        # Add additional heads at the specified height
        backbone = self.init_ptx_heads(args, backbone)
        super().__init__(backbone, loss, args, transform)
        self.old_data = []
        self.old_labels = []
        self.old_test_data = []
        self.old_test_labels = []
        self.current_task = 0

    def end_task(self, dataset):
        assert len(dataset.train_loader.dataset.data)
        self.old_data.append(dataset.train_loader.dataset.data)
        self.old_labels.append(torch.tensor(dataset.train_loader.dataset.targets))

        self.old_test_data.append(dataset.test_loaders[-1].dataset.data)
        self.old_test_labels.append(torch.tensor(dataset.test_loaders[-1].dataset.targets))

        self.current_task += 1

        if self.current_task != dataset.N_TASKS: 
            return

        self.net.train()

        # prepare dataloader
        all_data, all_labels = None, None
        all_test_data, all_test_labels = None, None
        for i in range(len(self.old_data)):
            if all_data is None:
                all_data = self.old_data[i]
                all_labels = self.old_labels[i]
                all_test_data = self.old_test_data[i]
                all_test_labels = self.old_test_labels[i]
            else:
                all_data = np.concatenate([all_data, self.old_data[i]])
                all_labels = np.concatenate([all_labels, self.old_labels[i]])
                all_test_data = np.concatenate([all_test_data, self.old_test_data[i]])
                all_test_labels = np.concatenate([all_test_labels, self.old_test_labels[i]])


        dataset.train_loader.dataset.data = all_data
        dataset.train_loader.dataset.targets = all_labels

        dataset.test_loaders[0].dataset.data = all_test_data
        dataset.test_loaders[0].dataset.targets = all_test_labels
        
        train_loader = torch.utils.data.DataLoader(dataset.train_loader.dataset, batch_size=self.args.batch_size, shuffle=True)

        scheduler = self.get_scheduler()
        
        acc = np.mean(evaluate(self, dataset)[0])
        print(f'\nmean acc BEFORE:',acc, '\n')

        self.opt.zero_grad()
        for e in range(self.args.n_epochs):
            for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {e + 1}/{self.args.n_epochs}', disable=self.args.non_verbose)):
                if i>2 and self.args.debug_mode:
                    break
                inputs, labels, not_aug_inputs = batch[0], batch[1], batch[2]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)

                ptx_inputs, ptx_labels = self.pretexter(self.base_data_aug(not_aug_inputs))
                ptx_outputs = self.get_ptx_outputs(ptx_inputs, self.net)
                loss_ptx = self.get_ce_pret_loss(ptx_outputs, ptx_labels)

                loss = self.loss(outputs, labels.long()) + self.args.ptx_alpha * loss_ptx
                self.opt.zero_grad()
                loss.backward()
                # if i % 3 == 0:
                self.opt.step()

            if scheduler is not None:
                scheduler.step()
            if e < 5 or e % 5 == 0:
                acc = np.mean(evaluate(self, dataset)[0])
                print(f'\nmean acc AFTER {e}:',acc, '\n')

        acc = np.mean(evaluate(self, dataset)[0])
        print(f'\nmean acc AFTER ALL:',acc, '\n')


    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        return 0,0,0,0,0
