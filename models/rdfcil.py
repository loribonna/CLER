from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset
from utils.inversion import GenerativeInversion
import math
from tqdm import trange


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Relation-Guided Representation Learning for Data-Free Continual Learning')
    add_management_args(parser)
    add_experiment_args(parser)
    
    parser.add_argument('--lambda_lce', type=float, required=False, default=0.5, help='lambda for LCE loss')
    parser.add_argument('--lambda_hkd', type=float, required=False, default=0.15, help='lambda for HKD loss')
    parser.add_argument('--lambda_rkd', type=float, required=False, default=0.5, help='lambda for RKD loss')

    parser.add_argument('--epochs_fitting', type=int, required=False, default=40, help='proportion of training data used for fitting')
    parser.add_argument('--lr_fitting', type=float, required=False, default=5e-3, help='learning rate for fitting')
    
    parser.add_argument('--inv_steps', type=int, required=False, default=5000, help='inversion steps')
    parser.add_argument('--inv_lr', type=float, required=False, default=0.001, help='inversion learning rate')
    parser.add_argument('--inv_tau', type=float, required=False, default=1000, help='inversion temperature')
    parser.add_argument('--inv_alpha_pr', type=float, required=False, default=0.001, help='inversion alpha_pr')
    parser.add_argument('--inv_alpha_rf', type=float, required=False, default=50.0, help='inversion alpha_rf')

    parser.add_argument('--z_dim', type=int, required=False, default=1000, help='inversion source noise dimension')
    return parser

class RKDAngleLoss(nn.Module):

    def forward(self, student, teacher, net):
        # N x C
        # N x N x C
        student, teacher = net.rkd_embed1(student), net.rkd_embed2(teacher)

        td = teacher.unsqueeze(0) - teacher.unsqueeze(1)
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = student.unsqueeze(0) - student.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.l1_loss(s_angle, t_angle)
        return loss

class RDFCIL(ContinualModel):
    NAME = 'rdfcil'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK
        self.num_classes = get_dataset(args).N_TASKS * self.cpt
        self.old_net = None
        self.task = 0
        self.backbone_size = self.net.classifier.in_features
        self.rkd = RKDAngleLoss()   

    def train_inversion(self):
        max_iters = self.args.inv_steps
        miniters = max(max_iters // 100, 1)
        for it in trange(max_iters, miniters=miniters, desc="Inversion"):
            if self.args.debug_mode==1 and it>3:
                break
            self.inversion()

    def fit_buffer(self, dataset):
        training = self.training
        self.train()
        bfopt = torch.optim.SGD(self.net.parameters(), lr=self.args.lr_fitting, weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)

        cls_weights = torch.ones(self.net.num_classes).to(self.device)
        cls_weights[:-self.cpt] *= self.task
        
        for _ in trange(self.args.epochs_fitting, desc="Buffer fitting"):
            for it, data in enumerate(dataset.train_loader):
                if self.args.debug_mode==1 and it>3:
                    break
                inputs, labels = data[0], data[1]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # train classifier
                synt_inputs, synt_labels = self.inversion.sample(len(labels))
                outputs      = self.net(inputs,      returnt='features')
                logits = self.net.classifier(outputs)
                synt_outputs = self.net(synt_inputs, returnt='features')
                synt_logits  = self.net.classifier(synt_outputs)
                with torch.no_grad():
                    old_resp = self.old_net(inputs)
                
                # HKD LOSS
                loss_hkd = F.l1_loss(old_resp, self.net(synt_inputs)[:, :self.old_net.num_classes]) / self.old_net.num_classes
                # FITTING LOSS
                loss_fit = F.cross_entropy(torch.cat([logits, synt_logits]),
                                             torch.cat([labels, synt_labels]),
                                                weight=cls_weights)
                loss = self.lambda_lce * loss_fit + self.lambda_hkd * loss_hkd
                bfopt.zero_grad()
                loss.backward()
                bfopt.step()
        
        self.train(training)

    def begin_task(self, dataset):
        if self.task > 0:
            self.old_net = deepcopy(self.net.eval())
            # instantiate rkd projectors
            self.net.rkd_embed1 = nn.Linear(self.backbone_size, 2*self.backbone_size).to(self.device)
            self.net.rkd_embed2 = nn.Linear(self.backbone_size, 2*self.backbone_size).to(self.device)

            # spawn & train inversion
            datapoint = next(iter(dataset.train_loader))[0]
            self.inversion = GenerativeInversion(
                model=self.old_net,
                input_dims=datapoint.shape[1:],
                lr=self.args.inv_lr,
                batch_size=self.args.batch_size,
                tau = self.args.inv_tau,
                alpha_pr=self.args.inv_alpha_pr,
                alpha_rf=self.args.inv_alpha_rf,
                zdim=self.args.z_dim,
            ).to(self.device)
            self.inversion.configure_optimizers()
            self.train_inversion()
            self.inversion.eval()
            
        self.net.num_classes = self.cpt * (self.task + 1)
        self.reset_classifier(inherit=True, bias=False) # also includes rkd_embed1 and rkd_embed2 in optimizer

        if self.task == 0:
            self.lambda_lce = 1
            self.lambda_hkd = 0
            self.lambda_rkd = 0
        else:
            # update lambdas
            alpha = math.log2(self.cpt/2+1)
            beta = math.sqrt(self.task)
            self.lambda_lce = self.args.lambda_lce * (1 + 1/alpha) / beta
            self.lambda_hkd = self.args.lambda_hkd * alpha * beta
            self.lambda_rkd = self.args.lambda_rkd * alpha * beta
        print(f"TASK: {self.task} -> lambda_lce: {self.lambda_lce}, lambda_hkd: {self.lambda_hkd}, lambda_rkd: {self.lambda_rkd}")
        self.step_count = 0
        self.net.train()
        
    def end_task(self, dataset):
        if self.task:
            self.fit_buffer(dataset)
        self.task += 1
        # buffer_fitting

    # https://github.dev/jianzhangcs/R-DFCIL (module.py)
    # https://github.dev/GT-RIPL/AlwaysBeDreaming-DFCIL (datafree.py)
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.step_count += 1
        labels = labels.long()
        
        new_feat = self.net(inputs, returnt='features')
        new_out = self.net.classifier(new_feat)

        loss_hkd, loss_rkd = torch.tensor(0.).to(self.device), torch.tensor(0.).to(self.device)
        if self.task != 0:

            with torch.no_grad():
                synt_inputs, _ = self.inversion.sample(len(labels))

                old_resp = self.old_net(synt_inputs)
                old_feat = self.old_net(inputs, returnt='features')
            
            loss_hkd = F.l1_loss(old_resp, self.net(synt_inputs)[:, :self.old_net.num_classes]) / self.old_net.num_classes

            loss_rkd = self.rkd(new_feat, old_feat, self.net)
   
        old_classes = self.old_net.num_classes if self.old_net is not None else 0
        loss_lce = self.loss(new_out[:, old_classes:old_classes+self.cpt], labels % self.cpt)
        
        loss = self.lambda_lce * loss_lce + self.lambda_hkd * loss_hkd + self.lambda_rkd * loss_rkd
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item(), 0, 0, 0, 0