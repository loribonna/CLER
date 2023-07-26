from copy import deepcopy
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Gaussiansmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(Gaussiansmoothing, self).__init__()
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).cuda()

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(
                    dim
                )
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, zdim, in_channel, img_sz):
        super().__init__()

        self.z_dim = zdim
        self.init_size = img_sz // 4
        self.l1 = nn.Sequential(nn.Linear(zdim, 128 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img

    def sample(self, size):
        device = next(self.parameters()).device
        z = torch.randn(size, self.z_dim).to(device)
        X = self.forward(z)
        return X

def freeze(module: nn.Module, mock_training: bool = False):
    """Freeze a torch Module
    1) save all parameters's current requires_grad state,
    2) disable requires_grad,
    3) turn on mock_training
    4) switch to evaluation mode.
    """

    state = {}
    for name, param in module.named_parameters():
        state[name] = param.requires_grad
        param.requires_grad = False
        param.grad = None

    if mock_training and hasattr(module, "mock_training"):
        module.mock_training = True

    module.eval()
    return state


def unfreeze(module: nn.Module, state= {}):
    """Unfreeze a torch Module
    1) restore all parameters's requires_grad state,
    2) switch to training mode.
    3) turn off mock_training
    """

    default = None if state else True
    for name, param in module.named_parameters():
        requires_grad = state.get(name, default)
        if requires_grad is not None:
            param.requires_grad = requires_grad

    module.train()

    if hasattr(module, "mock_training"):
        module.mock_training = False


class DeepInversionFeatureHook:
    """
    Implementation of the forward hook to track feature statistics and compute
    a loss on them. Will compute mean and variance, and will use l2 as a loss.
    """

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.value = None

    @property
    def r_feature(self):
        # compute generativeinversion's feature distribution regularization
        if self.value is None:
            return None

        nch = self.value.shape[1]
        mean = self.value.mean([0, 2, 3])
        value = self.value.permute(1, 0, 2, 3).contiguous().view([nch, -1])
        var = value.var(1, unbiased=False) + 1e-8

        r_mean = self.module.running_mean.data.type(var.type())
        r_var = self.module.running_var.data.type(var.type())

        r_feature_item1 = (var / (r_var + 1e-8)).log()
        r_feature_item2 = (r_var + (r_mean - mean).pow(2) + 1e-8) / var
        r_feature = 0.5 * (r_feature_item1 + r_feature_item2 - 1).mean()
        return r_feature

    @property
    def r_feature_di(self):
        # compute deepinversion's feature distribution regularization
        # compute generativeinversion's feature distribution regularization
        if self.value is None:
            return None

        nch = self.value.shape[1]
        mean = self.value.mean([0, 2, 3])
        value = self.value.permute(1, 0, 2, 3).contiguous().view([nch, -1])
        var = value.var(1, unbiased=False) + 1e-8

        r_mean = self.module.running_mean.data.type(var.type())
        r_var = self.module.running_var.data.type(var.type())

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        r_feature_item1 = torch.norm(r_mean - mean, 2)
        r_feature_item2 = torch.norm(r_var - var, 2)
        r_feature = r_feature_item1 + r_feature_item2
        return r_feature

    def hook_fn(self, module, input, output):
        self.value = input[0]

    def close(self):
        self.module, self.value = None, None
        self.hook.remove()

class GenerativeInversion(nn.Module):
    def __init__(
        self,
        model,
        zdim,
        batch_size,
        input_dims: Tuple[int, int, int] = (3, 32, 32),
        lr: float = 1e-3,
        tau: float = 1e3,
        alpha_pr: float = 1e-3,
        alpha_rf: float = 5.0,
    ):
        super().__init__()

        self.model = deepcopy(model)
        self.lr = lr
        self.tau = tau
        self.alpha_pr = alpha_pr
        self.alpha_rf = alpha_rf
        self.feature_hooks = []
        self.batch_size = batch_size

        self.generator = Generator(in_channel=input_dims[0], img_sz=input_dims[1], zdim=zdim)
        self.smoothing = Gaussiansmoothing(3, 5, 1)
        self.criterion_ce = nn.CrossEntropyLoss()

    def setup(self):
        freeze(self.model)
        self.register_feature_hooks()

    def register_feature_hooks(self):
        # Remove old before register
        for hook in self.feature_hooks:
            hook.remove()

        ## Create hooks for feature statistics catching
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d) and not 'ptx_net' in name:
                self.feature_hooks.append(DeepInversionFeatureHook(module))

    def criterion_pr(self, input):
        input_pad = F.pad(input, (2, 2, 2, 2), mode="reflect")
        input_smooth = self.smoothing(input_pad)
        return F.mse_loss(input, input_smooth)

    def criterion_rf(self):
        return torch.stack([h.r_feature for h in self.feature_hooks]).mean()

    def criterion_cb(self, output: torch.Tensor):
        logit_mu = output.softmax(dim=1).mean(dim=0)
        num_classes = output.shape[1]
        # ignore sign
        entropy = (logit_mu * logit_mu.log() / math.log(num_classes)).sum()
        return 1 + entropy

    @torch.no_grad()
    def sample(self, batch_size: int = None):
        _ = self.model.eval() if self.model.training else None
        batch_size = self.batch_size if batch_size is None else batch_size
        input = self.generator.sample(batch_size)
        target = self.model(input).argmax(dim=1)
        return input, target

    def train_step(self):
        input_ = self.generator.sample(self.batch_size)
        output = self.model(input_)
        target = output.data.argmax(dim=1)

        # content loss
        loss_ce = self.criterion_ce(output / self.tau, target)

        # label diversity loss
        loss_cb = self.criterion_cb(output)

        # locally smooth prior
        loss_pr = self.alpha_pr * self.criterion_pr(input_)

        # feature statistics regularization
        loss_rf = self.alpha_rf * self.criterion_rf()

        loss = loss_ce + loss_cb + loss_pr + loss_rf

        loss_dict = {
            "ce": loss_ce,
            "cb": loss_cb,
            "pr": loss_pr,
            "rf": loss_rf,
            "total": loss,
        }
        return loss, loss_dict

    def configure_optimizers(self):
        _ = self.setup(), unfreeze(self.generator)
        params = self.generator.parameters()
        self.opt = optim.Adam(params, lr=self.lr)

    def forward(self):
        loss, loss_dict = self.train_step()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss, loss_dict
        