import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import math


def add_self_args(parser):
    parser.add_argument('--selfsup', type=str, default='barlow', choices=['simclr', 'simsiam', 'byol', 'barlow', 'mocov2'])

    # moco stuff
    parser.add_argument('--moco_q_size', type=int, default=65540)  # for simplicity
    parser.add_argument("--moco_temperature", type=float, default=0.1)
    parser.add_argument("--moco_base_tau_momentum", type=float, default=0.99)
    parser.add_argument("--moco_final_tau_momentum", type=float, default=1)


def _get_projector_prenet(net, device=None):
    device = net.device if hasattr(net, 'device') else device if device is not None else "cpu"
    assert "resnet" in type(net).__name__.lower()

    sizes = [net.nf * 8] + [256, net.nf * 8]

    layers = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False).to(device))
        layers.append(nn.BatchNorm1d(sizes[i + 1]).to(device))
        layers.append(nn.ReLU(inplace=True).to(device))
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False).to(device))
    return nn.Sequential(*layers).to(device)


def init_model(model, args, device=None):
    if 'moco' in args.selfsup:
        assert args.moco_q_size % args.batch_size == 0, "moco_q_size must be divisible by batch_size"
        queue = torch.randn(2, model.nf * 8, args.moco_q_size)
        queue = F.normalize(queue, dim=1)
        model.register_buffer('queue', queue)
        model.queue_ptr = 0
        for p in model.predictor.parameters():
            p.requires_grad = False

        # init momentum encoder
        model.momentum_encoder = deepcopy(model)
        for p in model.momentum_encoder.parameters():
            p.requires_grad = False
        model.momentum_encoder.eval()

        model.moco_temperature = args.moco_temperature
    elif args.selfsup == 'onproto':
        model.projector = nn.Linear(model.nf * 8, 128)
    else:
        model.projector = _get_projector_prenet(model, device=device)
        model.predictor = deepcopy(model.projector)

    return model


def begin_task_ssl(model, dataset, args):
    model.max_steps = len(dataset.train_loader) * args.n_epochs
    model.cur_step = 0
    model.cur_tau = args.moco_base_tau_momentum


def post_update_ssl(model, args):
    if not hasattr(post_update_ssl, '_module_resetted'):
        post_update_ssl._module_resetted = {}
    if 'moco' in args.selfsup:
        for (n, p), (mn, mp) in zip(model.named_parameters(), model.momentum_encoder.named_parameters()):
            if mp.shape != p.shape:
                assert n == mn
                assert n not in post_update_ssl._module_resetted
                post_update_ssl._module_resetted[n] = True
                mp.data = p.data
            else:
                mp.data = mp.data * model.cur_tau + p.data * (1. - model.cur_tau)

        # update tau
        model.cur_tau = args.moco_final_tau_momentum - (args.moco_final_tau_momentum - args.moco_base_tau_momentum) * (math.cos(math.pi * model.cur_step / model.max_steps) + 1) / 2

    return model


def get_self_func(args):
    if args.selfsup == 'simclr':
        return SimCLR
    elif args.selfsup == 'simsiam':
        return SimSiam
    elif args.selfsup == 'byol':
        return BYOL
    elif args.selfsup == 'barlow':
        return BarlowTwins
    elif args.selfsup == 'mocov2':
        return MoCoV2
    else:
        raise NotImplementedError

def BarlowTwins(model, y1, y2, compute_logits=True):
    z1 = model.projector(model(y1, returnt="features") if compute_logits else y1)
    z2 = model.projector(model(y2, returnt="features") if compute_logits else y2)
    z_a = (z1 - z1.mean(0)) / z1.std(0)
    z_b = (z2 - z2.mean(0)) / z2.std(0)
    N, D = z_a.size(0), z_a.size(1)
    c_ = torch.mm(z_a.T, z_b) / N
    c_diff = (c_ - torch.eye(D).to(model.device)).pow(2)
    c_diff[~torch.eye(D, dtype=bool)] *= 2e-3
    loss = c_diff.sum()
    return loss


def SimCLR(model, y1, y2, temp=100, eps=1e-6, compute_logits=True):
    z1 = model.projector(model(y1, returnt="features") if compute_logits else y1)
    z2 = model.projector(model(y2, returnt="features") if compute_logits else y2)
    z_a = (z1 - z1.mean(0)) / z1.std(0)
    z_b = (z2 - z2.mean(0)) / z2.std(0)

    out = torch.cat([z_a, z_b], dim=0)
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temp)
    neg = sim.sum(dim=1)

    row_sub = torch.Tensor(neg.shape).fill_(math.e**(1 / temp)).to(model.device)
    neg = torch.clamp(neg - row_sub, min=eps)
    pos = torch.exp(torch.sum(z_a * z_b, dim=-1) / temp)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / (neg + eps)).mean()
    return loss


def SimSiam(model, y1, y2, compute_logits=True):
    def D(p, z):
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()

    z1 = model.projector(model(y1, returnt="features") if compute_logits else y1)
    z2 = model.projector(model(y2, returnt="features") if compute_logits else y2)

    p1 = model.predictor(z1)
    p2 = model.predictor(z2)

    loss = (D(p1, z2).mean() + D(p2, z1).mean()) * 0.5
    return loss


def BYOL(model, y1, y2, compute_logits=True):
    def D(p, z):
        p = F.normalize(p, dim=-1, p=2)
        z = F.normalize(z, dim=-1, p=2)
        return 2 - 2 * (p * z).sum(dim=-1)

    z1 = model.projector(model(y1, returnt="features") if compute_logits else y1)
    z2 = model.projector(model(y2, returnt="features") if compute_logits else y2)
    p1, p2 = model.predictor(z1), model.predictor(z2)

    loss = (D(z1, p2.detach()).mean() + D(z2, p1.detach()).mean()) * 0.5
    return loss


def MoCoV2(model, y1, y2, compute_logits=True):
    # code from cassle
    assert compute_logits, "MoCoV2 requires logits to be computed by the encoders"

    def moco_loss_func(
        query: torch.Tensor, key: torch.Tensor, queue: torch.Tensor, temperature=0.1
    ) -> torch.Tensor:
        pos = torch.einsum("nc,nc->n", [query, key]).unsqueeze(-1)
        neg = torch.einsum("nc,ck->nk", [query, queue])
        logits = torch.cat([pos, neg], dim=1)
        logits /= temperature
        targets = torch.zeros(query.size(0), device=query.device, dtype=torch.long)
        return F.cross_entropy(logits, targets)

    @torch.no_grad()
    def _dequeue_and_enqueue(model, keys: torch.Tensor):
        batch_size = keys.shape[1]
        # assert model.queue.shape[-1] % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        keys = keys.permute(0, 2, 1)
        maxsize = model.queue.shape[-1] - model.queue_ptr
        real_batch_size = min(batch_size, maxsize)
        model.queue[:, :, model.queue_ptr: model.queue_ptr + real_batch_size] = keys[:, :, :real_batch_size]
        model.queue_ptr = (model.queue_ptr + real_batch_size) % model.queue.shape[-1]  # move pointer

    feats1, feats2 = model(y1, returnt="features"), model(y2, returnt="features")
    momentum_feats1, momentum_feats2 = model(y1, returnt="features"), model(y2, returnt="features")

    q1 = model.projector(feats1)
    q2 = model.projector(feats2)
    q1 = F.normalize(q1, dim=1)
    q2 = F.normalize(q2, dim=1)

    with torch.no_grad():
        k1 = model.predictor(momentum_feats1)
        k2 = model.predictor(momentum_feats2)
        k1 = F.normalize(k1, dim=1)
        k2 = F.normalize(k2, dim=1)

    queue = model.queue.clone().detach()
    nce_loss = (
        moco_loss_func(q1, k2, queue[1], model.moco_temperature)
        + moco_loss_func(q2, k1, queue[0], model.moco_temperature)
    ) / 2

    keys = torch.stack((k1, k2))
    _dequeue_and_enqueue(model, keys)

    return nce_loss
