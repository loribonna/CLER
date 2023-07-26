import torch
from models.utils.pretext_model import PretextModel
from utils.buffer import Buffer
from utils.args import *


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='ER-ACE with future not fixed (as made by authors) + Equivariant pretext task')
    parser.add_argument('--ptx_alpha', type=float, required=True)

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser

def features_hook(module, input, output):
    module.partial_features = output

class ErACEPretext(PretextModel):
    NAME = 'er_ace_pretext'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        # Add additional heads at the specified height
        backbone = self.init_ptx_heads(args, backbone)

        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.task = 0

    def end_task(self, dataset):
        self.task += 1

    def to(self, device):
        super().to(device)
        self.seen_so_far = self.seen_so_far.to(device)
        
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
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
        all_inputs = not_aug_inputs

        if self.task > 0:
            # sample from buffer
            buf_indexes, buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=True)
            buf_not_aug_inputs = self.buffer.examples[buf_indexes]
            loss_re = self.loss(self.net(buf_inputs), buf_labels)

            all_inputs = torch.cat([not_aug_inputs, buf_not_aug_inputs])

        ptx_inputs, ptx_labels = self.pretexter(self.base_data_aug(all_inputs))
        ptx_outputs = self.get_ptx_outputs(ptx_inputs, self.net)
        loss_ptx = self.get_ce_pret_loss(ptx_outputs, ptx_labels)

        loss += loss_re + self.args.ptx_alpha * loss_ptx

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels)

        return loss.item(), 0, 0, 0, 0