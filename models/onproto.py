import torch
from torch import nn
import torch.nn.functional as F
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset
from utils.kornia_utils import to_kornia_transform
from utils.selfsup import init_model
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from models.onproto_utils.ope import OPELoss
from models.onproto_utils.apf import AdaptivePrototypicalFeedback

def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--enable_rotation', default=0, choices=[0, 1], type=int)
    parser.add_argument('--oop', type=int, default=16, help='No mention of this in the original paper')
    parser.add_argument('--proto_t', type=float, default=0.5, help='(default=%(default)s)')
    parser.add_argument('--ins_t', type=float, default=0.07, help='(default=%(default)s)')

    # AFP
    parser.add_argument('--mixup_base_rate', type=float, default=0.75, help='(default=%(default)s)')
    parser.add_argument('--mixup_alpha', type=float, default=0.4, help='(default=%(default)s)')
    parser.add_argument('--mixup_p', type=float, default=0.6, help='(default=%(default)s)')
    parser.add_argument('--mixup_lower', type=float, default=0, help='(default=%(default)s)')
    parser.add_argument('--mixup_upper', type=float, default=0.6, help='(default=%(default)s)')

    parser.add_argument('--sim_lambda', type=float, default=1, help='(default=%(default)s)')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


def rot_inner_all(x):
    num = x.shape[0]
    c, h, w = x.shape[1], x.shape[2], x.shape[3]

    R = x.repeat(4, 1, 1, 1)
    a = x.permute(0, 1, 3, 2)
    a = a.view(num, c, 2, h//2, w)

    a = a.permute(2, 0, 1, 3, 4)

    s1 = a[0]
    s2 = a[1]
    s1_1 = torch.rot90(s1, 2, (2, 3))
    s2_2 = torch.rot90(s2, 2, (2, 3))

    R[num:2 * num] = torch.cat((s1_1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num, c, h, w).permute(0, 1, 3, 2)
    R[3 * num:] = torch.cat((s1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num, c, h, w).permute(0, 1, 3, 2)
    R[2 * num:3 * num] = torch.cat((s1_1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num, c, h, w).permute(0, 1, 3, 2)
    return R

def Rotation(x):
    # rotation augmentation in OCM
    X = rot_inner_all(x)
    return torch.cat((X, torch.rot90(X, 2, (2, 3)), torch.rot90(X, 1, (2, 3)), torch.rot90(X, 3, (2, 3))), dim=0)

def Supervised_NT_xent_uni(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8):
    """
        Code from OCM: https://github.com/gydpku/OCM
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    """

    device = sim_matrix.device
    labels1 = labels.repeat(2)

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)

    sim_matrix = sim_matrix - logits_max.detach()
    B = sim_matrix.size(0) // chunk

    sim_matrix = torch.exp(sim_matrix / temperature)
    denom = torch.sum(sim_matrix, dim=1, keepdim=True)

    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)
    labels1 = labels1.contiguous().view(-1, 1)

    Mask1 = torch.eq(labels1, labels1.t()).float().to(device)
    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)

    return torch.sum(Mask1 * sim_matrix) / (2 * B)

def Supervised_NT_xent_n(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8):
    """
        Code from OCM : https://github.com/gydpku/OCM
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    """
    device = sim_matrix.device
    labels1 = labels.repeat(2)

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)

    sim_matrix = sim_matrix - logits_max.detach()
    B = sim_matrix.size(0) // chunk

    eye = torch.zeros((B * chunk, B * chunk), dtype=torch.bool, device=device)
    eye[:, :].fill_diagonal_(True)
    sim_matrix = torch.exp(sim_matrix / temperature) * (~eye)

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)

    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)
    labels1 = labels1.contiguous().view(-1, 1)

    Mask1 = torch.eq(labels1, labels1.t()).float().to(device)
    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)

    loss2 = 2 * torch.sum(Mask1 * sim_matrix) / (2 * B)
    loss1 = torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)

    return loss1 + loss2


class OnProto(ContinualModel):
    NAME = 'onproto'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        args.selfsup = 'onproto'
        init_model(backbone, args)
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        dset = get_dataset(args)
        self.cpt = dset.N_CLASSES_PER_TASK
        self.num_classes = dset.N_TASKS * dset.N_CLASSES_PER_TASK
        self.task = 0
        self.scaler = GradScaler()
        self.buffer_per_class = 7
        self.classes_mean = torch.zeros((self.num_classes, 128), requires_grad=False).cuda()
        self.OPELoss = OPELoss(self.cpt, temperature=self.args.proto_t)
        self.APF = AdaptivePrototypicalFeedback(self.buffer, args.mixup_base_rate, args.mixup_p, args.mixup_lower, args.mixup_upper,
                                  args.mixup_alpha, self.cpt)
        self.weak_transform = to_kornia_transform(transform)
                                  
        if "cifar100" in args.dataset:
            self.sim_lambda = 1.0
        elif "cifar10" in args.dataset:
            self.sim_lambda = 0.5
        elif args.dataset == "tiny_imagenet":
            self.sim_lambda = 1.0
        else:
            self.sim_lambda = self.args.sim_lambda


    def end_task(self, dataset):
        self.task += 1

    def to(self, device):
        super().to(device)
        self.seen_so_far = self.seen_so_far.to(device)

    
    def cal_buffer_proto_loss(self, buffer_x, buffer_y, buffer_x_pair, task_id):
        buffer_fea = self.net(buffer_x_pair, returnt='features')
        buffer_z = self.net.projector(buffer_fea)
        buffer_z_norm = F.normalize(buffer_z)
        buffer_z1 = buffer_z_norm[:buffer_x.shape[0]]
        buffer_z2 = buffer_z_norm[buffer_x.shape[0]:]

        buffer_proto_loss, buffer_z1_proto, buffer_z2_proto = self.OPELoss(buffer_z1, buffer_z2, buffer_y, task_id)
        self.classes_mean = (buffer_z1_proto + buffer_z2_proto) / 2

        return buffer_proto_loss, buffer_z1_proto, buffer_z2_proto, buffer_z_norm


    def observe_task0(self, inputs, labels, not_aug_inputs, epoch=None):
        labels = labels.long()
        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        with autocast():
            # x = not_aug_inputs.requires_grad_()
            x = not_aug_inputs

            if self.args.enable_rotation:
                rot_x = Rotation(x)
                rot_x_aug = self.weak_transform(rot_x)

                rot_sim_labels = torch.cat([labels + self.num_classes * i for i in range(self.args.oop)], dim=0)
            else:
                rot_sim_labels = labels
                x_aug = self.weak_transform(x)
                rot_x = x
                rot_x_aug = x_aug # as written in the paper
            inputs = torch.cat([rot_x, rot_x_aug], dim=0)

            features = self.net(inputs, returnt='features')
            projections = self.net.projector(features)
            projections = F.normalize(projections)

            # instance-wise contrastive loss in OCM
            features = F.normalize(features)
            dim_diff = features.shape[1] - projections.shape[1]  # 512 - 128
            dim_begin = torch.randperm(dim_diff)[0]
            dim_len = projections.shape[1]

            sim_matrix = torch.matmul(projections, features[:, dim_begin:dim_begin + dim_len].t())
            sim_matrix += torch.mm(projections, projections.t())

            ins_loss = Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=self.args.ins_t)
            
            if not self.buffer.is_empty():
                buffer_x, buffer_y = self.buffer.get_data(self.args.batch_size)
                # buffer_x.requires_grad = True
                buffer_x, buffer_y = buffer_x.cuda(), buffer_y.cuda()
                buffer_x_pair = torch.cat([buffer_x, self.weak_transform(buffer_x)], dim=0)

                proto_seen_loss, _, _, _ = self.cal_buffer_proto_loss(buffer_x, buffer_y, buffer_x_pair, self.task)
            else:
                proto_seen_loss = 0

            z = projections[:rot_x.shape[0]]
            zt = projections[rot_x.shape[0]:]
            proto_new_loss, cur_new_proto_z, cur_new_proto_zt = self.OPELoss(z[:x.shape[0]], zt[:x.shape[0]], labels, self.task, True)

            OPE_loss = proto_new_loss + proto_seen_loss

            y_pred = self.net(self.weak_transform(x))
            ce = F.cross_entropy(y_pred, labels)

            loss = ce + ins_loss + OPE_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad()
        self.buffer.add_data(examples=not_aug_inputs.detach(), labels=labels.detach())

        return loss.item(), 0, 0, 0, 0
    
    def observe_other_tasks(self, inputs, labels, not_aug_inputs, epoch=None):
        with autocast():
            # x = x.requires_grad_()
            buffer_batch_size = min(self.args.minibatch_size, self.buffer_per_class * len(self.seen_so_far))

            ori_mem_x, ori_mem_y = self.buffer.get_data(buffer_batch_size)
            if not self.buffer.is_empty():
                mem_x, mem_y, mem_y_mix = self.APF(ori_mem_x, ori_mem_y, buffer_batch_size, self.classes_mean, self.task)
                rot_sim_labels = torch.cat([labels + self.num_classes * i for i in range(self.args.oop)], dim=0)
                rot_sim_labels_r = torch.cat([mem_y + self.num_classes * i for i in range(self.args.oop)], dim=0)
                rot_mem_y_mix = torch.zeros(rot_sim_labels_r.shape[0], 3).cuda()
                rot_mem_y_mix[:, 0] = torch.cat([mem_y_mix[:, 0] + self.num_classes * i for i in range(self.args.oop)], dim=0)
                rot_mem_y_mix[:, 1] = torch.cat([mem_y_mix[:, 1] + self.num_classes * i for i in range(self.args.oop)], dim=0)
                rot_mem_y_mix[:, 2] = mem_y_mix[:, 2].repeat(self.args.oop)
            else:
                mem_x = ori_mem_x
                mem_y = ori_mem_y

                rot_sim_labels = torch.cat([labels + self.num_classes * i for i in range(self.args.oop)], dim=0)
                rot_sim_labels_r = torch.cat([mem_y + self.num_classes * i for i in range(self.args.oop)], dim=0)

            # mem_x = mem_x.requires_grad_()

            rot_x = Rotation(inputs)
            rot_x_r = Rotation(mem_x)
            rot_x_aug = self.weak_transform(rot_x)
            rot_x_r_aug = self.weak_transform(rot_x_r)
            images_pair = torch.cat([rot_x, rot_x_aug], dim=0)
            images_pair_r = torch.cat([rot_x_r, rot_x_r_aug], dim=0)

            all_images = torch.cat((images_pair, images_pair_r), dim=0)

            features = self.net(all_images, returnt='features')
            projections = self.net.projector(features)

            projections_x = projections[:images_pair.shape[0]]
            projections_x_r = projections[images_pair.shape[0]:]

            projections_x = F.normalize(projections_x)
            projections_x_r = F.normalize(projections_x_r)

            # instance-wise contrastive loss in OCM
            features_x = F.normalize(features[:images_pair.shape[0]])
            features_x_r = F.normalize(features[images_pair.shape[0]:])

            dim_diff = features_x.shape[1] - projections_x.shape[1]
            dim_begin = torch.randperm(dim_diff)[0]
            dim_begin_r = torch.randperm(dim_diff)[0]
            dim_len = projections_x.shape[1]

            sim_matrix = self.sim_lambda * torch.matmul(projections_x, features_x[:, dim_begin:dim_begin + dim_len].t())
            sim_matrix_r = self.sim_lambda * torch.matmul(projections_x_r, features_x_r[:, dim_begin_r:dim_begin_r + dim_len].t())

            sim_matrix += self.sim_lambda * torch.mm(projections_x, projections_x.t())
            sim_matrix_r += self.sim_lambda * torch.mm(projections_x_r, projections_x_r.t())

            loss_sim_r = Supervised_NT_xent_uni(sim_matrix_r, labels=rot_sim_labels_r, temperature=self.args.ins_t)
            loss_sim = Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=self.args.ins_t)
            
            ins_loss = loss_sim_r + loss_sim

            y_pred = self.net(self.weak_transform(mem_x))

            buffer_x = ori_mem_x
            buffer_y = ori_mem_y
            buffer_x_pair = torch.cat([buffer_x, self.weak_transform(buffer_x)], dim=0)
            proto_seen_loss, cur_buffer_z1_proto, cur_buffer_z2_proto, cur_buffer_z = self.cal_buffer_proto_loss(buffer_x, buffer_y, buffer_x_pair, self.task)

            z = projections_x[:rot_x.shape[0]]
            zt = projections_x[rot_x.shape[0]:]
            proto_new_loss, cur_new_proto_z, cur_new_proto_zt = self.OPELoss(z[:inputs.shape[0]], zt[:inputs.shape[0]], labels, self.task, True)

            OPE_loss = proto_new_loss + proto_seen_loss

            if not self.buffer.is_empty():
                ce = self.loss_mixup(y_pred, mem_y_mix)
            else:
                ce = F.cross_entropy(y_pred, mem_y)

            loss = ce + ins_loss + OPE_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad()
        self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        return loss.item(), 0, 0, 0, 0

    def loss_mixup(self, logits, y):
        loss_a = F.cross_entropy(logits, y[:, 0].long(), reduction='none')
        loss_b = F.cross_entropy(logits, y[:, 1].long(), reduction='none')
        return ((1 - y[:, 2]) * loss_a + y[:, 2] * loss_b).mean()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if self.task==0:
            return self.observe_task0(inputs, labels, not_aug_inputs, epoch)
        else:
            return self.observe_other_tasks(inputs, labels, not_aug_inputs, epoch)
            