import torch
from torch import nn
from copy import deepcopy

def add_self_args(parser):
    parser.add_argument('--selfsup', type=str, default='barlow', choices=['barlow'])

def _get_projector_prenet(net, device=None):
    device = net.device if hasattr(net, 'device') else device if device is not None else "cpu"
    assert "resnet" in type(net).__name__.lower()

    sizes = [net.nf*8] + [256, net.nf*8]

    layers = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False).to(device))
        layers.append(nn.BatchNorm1d(sizes[i + 1]).to(device))
        layers.append(nn.ReLU(inplace=True).to(device))
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False).to(device))
    return nn.Sequential(*layers).to(device)

def init_model(model, args=None, device=None):
    model.projector = _get_projector_prenet(model, device=device)
    model.predictor = deepcopy(model.projector)

    return model

def get_self_func(args):
    if args.selfsup == 'barlow':
        return BarlowTwins
    else:
        raise NotImplementedError

def BarlowTwins(model, y1, y2, compute_logits=True):
    z1 = model.projector(model(y1,returnt="features") if compute_logits else y1)
    z2 = model.projector(model(y2,returnt="features") if compute_logits else y2)
    z_a = (z1 - z1.mean(0)) / z1.std(0)
    z_b = (z2 - z2.mean(0)) / z2.std(0)
    N, D = z_a.size(0), z_a.size(1)
    c_ = torch.mm(z_a.T, z_b) / N
    c_diff = (c_ - torch.eye(D).cuda()).pow(2)
    c_diff[~torch.eye(D, dtype=bool)] *= 2e-3
    loss = c_diff.sum()   
    return loss
