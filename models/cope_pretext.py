import torch
import torch.nn.functional as F
from utils.buffer import Buffer
from utils.args import *
from models.utils.pretext_model import PretextModel
from datasets import get_dataset
from models.cope import PPPloss

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Prototype Evolution + Equivariant pretext task')
    parser.add_argument('--ptx_alpha', type=float, required=True)
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Dimensionality of prototypes')
    parser.add_argument('--num_batches', type=int, default=5,
                        help='Number of inner cycles for online CL')
    parser.add_argument('--loss_T', type=float, default=0.05,
                        help='loss temperature')
    parser.add_argument('--p_momentum', type=float, default=0.9,
                        help='momentum for prototype updates')

    return parser

class CoPEPretext(PretextModel):
    NAME = 'cope_pretext'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        # Add additional heads at the specified height
        backbone = self.init_ptx_heads(args, backbone)

        super().__init__(backbone, loss, args, transform)
        
        self.net.num_classes = self.args.hidden_dim
        self.reset_classifier()

        self.buffer = Buffer(self.args.buffer_size, self.device, mode='balancoir')
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK
        self.task = 0

        self.loss = PPPloss(T=self.args.loss_T)
        self.eye = torch.eye(self.num_classes).to(self.device)

        self.proto_shape = self.args.hidden_dim
        self.protos_x = torch.empty(0, self.proto_shape).to(self.device)
        self.protos_y = torch.empty(0).long().to(self.device)
        self.tmp_protx = torch.empty(0, self.proto_shape).to(self.device)
        self.tmp_protcnt = torch.empty(0).long().to(self.device)

    def end_task(self, dataset):
        self.task += 1

    def to(self, device):
        super().to(device)
        self.seen_so_far = self.seen_so_far.to(device)
        
    def init_protos(self, new_classes):
        for c in new_classes:
            print('Initializing prototype for class {}'.format(c))
            p = torch.nn.functional.normalize(torch.empty((1, self.proto_shape)).uniform_(0, 1), p=2, dim=1).detach().to(self.device)
            self.protos_x = torch.cat([self.protos_x, p], dim=0)
            self.protos_y = torch.cat([self.protos_y, torch.tensor([c]).long().to(self.device)])
            self.tmp_protx = torch.cat([self.tmp_protx, torch.zeros_like(p)], dim=0)
            self.tmp_protcnt = torch.cat([self.tmp_protcnt, torch.zeros(1).long().to(self.device)])
    
    def accumulate_protos(self, f, y):
        ''' Accumulate prototype values for each item in a batch '''
        for c in torch.unique(y):
            p_tmp_batch = f[c == y].sum(dim=0)
            index = (self.protos_y == c).nonzero().squeeze(1)[0]
            self.tmp_protx[index] += p_tmp_batch.detach()
            self.tmp_protcnt[index] += len(p_tmp_batch)

    def update_protos(self):
        for c in self.protos_y:
            proto_ind = (self.protos_y == c).nonzero().squeeze(1)[0]
            if self.tmp_protcnt[proto_ind] > 0:
                # Momentum Update
                incr_p = self.tmp_protx[proto_ind] / self.tmp_protcnt[proto_ind]
                old_p = self.protos_x[proto_ind].clone()
                new_p = self.args.p_momentum * old_p + (1 - self.args.p_momentum) * incr_p
                new_p = torch.nn.functional.normalize(new_p, p=2,dim=0)
                # Update
                self.protos_x[proto_ind] = new_p.detach()
                assert not torch.isnan(self.protos_x).any()
                # Reset counters
                self.tmp_protx[proto_ind] *= 0
                self.tmp_protcnt[proto_ind] = 0
        

    def forward(self, x):
        """ Deployment forward. Find closest prototype for each sample. """
        # nearest neighbor
        nd = self.proto_shape
        ns = x.size(0)

        if not len(self.seen_so_far):
            # no exemplar in memory yet, output uniform distr. over all classes
            out = torch.Tensor(ns, self.n_classes).fill_(1.0 / self.n_classes)
            if self.gpu:
                out = out.cuda()
            return out
            
        means = torch.ones(len(self.seen_so_far), nd).to(x.device) * -float('inf')
        means[self.protos_y] = self.protos_x # Class idx gets allocated its prototype

        preds = self.net(x)
        preds = F.normalize(preds, p=2, dim=1)  # L2-embedding normalization
        # Predict to nearest
        sims = []
        for sample_idx in range(ns):  # Per class
            simlr = torch.mm(means, preds[sample_idx].view(-1, preds[sample_idx].shape[-1]).t())  # Dot product
            sims.append(simlr.T)

        sims = torch.cat(sims, dim=0)
        if sims.shape[1] < self.num_classes:
            sims = torch.cat([sims, torch.ones(sims.shape[0], self.num_classes - sims.shape[1]).to(sims.device) * -float('inf')], dim=1)


        return sims  

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        labels = labels.long()
        present = labels.unique()
        new_classes = present[~torch.isin(present, self.seen_so_far)]
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        self.opt.zero_grad()
        overall_loss = 0
        for i in range(self.args.num_batches):
            update = i == self.args.num_batches - 1
            all_inputs, all_labels = inputs, labels
            all_not_aug_inputs = not_aug_inputs
            if not self.buffer.is_empty():
                buf_indexes, buf_inputs, buf_labels = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, return_index=True)
                buf_not_aug_inputs = self.buffer.examples[buf_indexes]
                all_inputs = torch.cat([all_inputs, buf_inputs], dim=0)
                all_labels = torch.cat([all_labels, buf_labels], dim=0)
                all_not_aug_inputs = torch.cat([all_not_aug_inputs, buf_not_aug_inputs], dim=0)
            all_latents = self.net(all_inputs)
            all_latents = F.normalize(all_latents, p=2)
            if i == 0 and len(new_classes):
                self.init_protos(new_classes)
            
            # Accumulate prototypes update at each inner iteration
            if not self.buffer.is_empty():
                self.accumulate_protos(all_latents[len(labels):], buf_labels)
            
            if len(self.protos_x) > 1:
                loss, loss_pos, loss_neg = self.loss(all_latents, all_labels, self.protos_x, self.protos_y)
            else:
                loss, loss_pos, loss_neg = torch.tensor(0.).to(self.device), torch.tensor(0.).to(self.device), torch.tensor(0.).to(self.device)

            ptx_inputs, ptx_labels = self.pretexter(self.base_data_aug(all_not_aug_inputs))
            ptx_outputs = self.get_ptx_outputs(ptx_inputs, self.net)
            loss_ptx = self.get_ce_pret_loss(ptx_outputs, ptx_labels)
            loss += loss_ptx * self.args.ptx_alpha

            if loss.requires_grad:
                loss.backward()
                self.opt.step()

            
            overall_loss += loss.item()

            if update:
                # Accumulate prototypes for stream examples
                self.accumulate_protos(all_latents[:len(labels)], labels)
                # And Update
                self.update_protos()
                self.buffer.add_data(examples=not_aug_inputs,
                                labels=labels)
            
            
        return overall_loss / self.args.num_batches, 0,0,0,0