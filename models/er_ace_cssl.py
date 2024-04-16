import torch
import torch.nn.functional as F
from models.utils.continual_model import ContinualModel
from utils.buffer import Buffer
from utils.args import *
from utils.selfsup import get_self_func, init_model, add_self_args, post_update_ssl, begin_task_ssl


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='ER-ACE with future not fixed (as made by authors) + CSSL')
    parser.add_argument('--inv_alpha', type=float, required=True)

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_self_args(parser)
    return parser

def features_hook(module, input, output):
    module.partial_features = output

class ErACECssl(ContinualModel):
    NAME = 'er_ace_cssl'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        init_model(backbone, args)

        super(ErACECssl, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.task = 0
        self.selffunc = get_self_func(args)
    
    def begin_task(self, dataset):
        begin_task_ssl(self.net, dataset, self.args)

    def end_task(self, dataset):
        self.task += 1

    def to(self, device):
        self.device = device
        super().to(device)
        self.seen_so_far = self.seen_so_far.to(device)
        
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        B = inputs.shape[0]
        labels = labels.long()
        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        if self.task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)
        loss_re = torch.tensor(0.)

        if self.task > 0:
            # sample from buffer
            buf_indexes, buf_inputs, buf_labels = self.buffer.get_data(
                self.args.batch_size, transform=self.transform, return_index=True)
            buf_not_aug_inputs = self.buffer.examples[buf_indexes]
            buf_logits = self.net(buf_inputs)

            loss_re = self.loss(buf_logits, buf_labels)

            not_aug_inputs = torch.cat([not_aug_inputs, buf_not_aug_inputs], dim=0)
            inputs = torch.cat([inputs, buf_inputs], dim=0)
        
        inputs_b = self.aug_batch(not_aug_inputs, device=inputs.device)
        loss_inv = self.selffunc(self.net, inputs, inputs_b)

        loss += loss_re + self.args.inv_alpha * loss_inv

        loss.backward()
        self.opt.step()
        post_update_ssl(self.net, self.args)

        self.buffer.add_data(examples=not_aug_inputs[:B],
                             labels=labels)

        return loss.item(), 0, 0, 0, 0