import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import Callable, List, Optional


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, kernel_size=3, groups: int = 1, dilation: int = 1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride if kernel_size == 3 else 2,
                     padding=dilation if kernel_size == 3 else 3, bias=False, dilation=dilation, groups=groups)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def _small_block_forward(self, x: torch.Tensor) -> torch.Tensor:
    out = relu(self.bn(self.conv(x)))

    if self.use_maxpool:
        out = F.max_pool2d(out, 2, 2, ceil_mode=True)

    return out

class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.conv1 = conv3x3(in_planes=in_planes, out_planes=planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(in_planes=planes, out_planes=planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def set_small_forward(self, use_maxpool=True):
        self.forward = types.MethodType(_small_block_forward, self)
        self.use_maxpool = use_maxpool

        device = next(iter(self.conv1.parameters())).device

        self.conv = conv3x3(in_planes=self.in_planes, out_planes=self.planes).to(device)
        self.bn = nn.BatchNorm2d(self.planes).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        out = relu(out)
        return out

class Bottleneck(nn.Module):
    "Resnet v1.5 bottleneck"

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(in_planes=width, out_planes=width, stride=stride, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.stride = stride
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = relu(self.bn1(self.conv1(x)))
        out = relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)

        out = relu(out)

        return out

class ResNet(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def expand_classifier(self, n_classes):
        self.classifier = nn.Linear(self.classifier.in_features, n_classes)

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, first_kernel_size=3) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(in_planes=3, out_planes=nf * 1, kernel_size=first_kernel_size)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.enable_maxpool = first_kernel_size != 3
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.classifier = nn.Linear(nf * 8 * block.expansion, num_classes)

        self._features = lambda: exec(
            'raise NotImplementedError("Deprecated: use forward with returnt=\'features\'")')

    def get_params(self, discard_classifier=False) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for kk, pp in list(self.named_parameters()):
            if not discard_classifier or not 'classifier' in kk:
                params.append(pp.view(-1))
        return torch.cat(params)

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, returnt='out', **kwargs) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        out_0 = self.bn1(self.conv1(x))
        out_0 = relu(out_0)

        if self.enable_maxpool:
            out_0 = self.maxpool(out_0)
        out_1 = self.layer1(out_0) # 64, 32, 32
        out_2 = self.layer2(out_1) # 128, 16, 16
        out_3 = self.layer3(out_2) # 256, 8, 8
        out_4 = self.layer4(out_3) # 512, 4, 4

        feature = avg_pool2d(out_4, out_4.shape[2])  # 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        if returnt == 'features':
            return feature

        out = self.classifier(feature)

        if returnt == 'out':
            return out
        elif returnt == 'full':
            return out, [out_0, out_1, out_2, out_3, out_4, out]
        else:
            return (out, feature)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        return self.forward(x, returnt='features')

def resnet18(nclasses: int, nf: int = 64, first_k=3) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, first_kernel_size=first_k)


def resnet34(nclasses: int, nf: int = 64) -> ResNet:
    """
    Instantiates a ResNet34 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], nclasses, nf)


def resnet50(nclasses: int, nf: int = 64, first_k=3) -> ResNet:
    """
    Instantiates a ResNet50 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], nclasses, nf, first_kernel_size=first_k)


def lopeznet(nclasses: int) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, 20)
