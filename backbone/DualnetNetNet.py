from torch import nn
import torch
import torch.nn.functional as F
import math
from torch.nn.functional import relu

def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)

class block(nn.Module):
    def __init__(self, n_in, n_out):
        super(block, self).__init__()
        self.net = nn.Sequential(*[ nn.Linear(n_in, n_out), nn.ReLU()])
        self.net.apply(Xavier)
    
    def forward(self, x):
        return self.net(x)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        out = relu(out)
        return out

class DualnetNetNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf):
        super(DualnetNetNet, self).__init__()
        self.in_planes = nf
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(nf * 8 * block.expansion, int(num_classes))
        self.block = block
        
        sizes = [nf*8] + [256, nf*8]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        sizes = [nf*8] + [256, nf*8]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.predictor = nn.Sequential(*layers)

        self.f_conv1 = self._make_conv2d_layer(3, nf, max_pool=True, padding = 1)
        self.f_conv2 = self._make_conv2d_layer(nf*1, nf*2, padding = 1, max_pool=True)
        self.f_conv3 = self._make_conv2d_layer(nf*2, nf*4, padding = 1, max_pool=True)
        self.f_conv4 = self._make_conv2d_layer(nf*4, nf*8, padding = 1, max_pool=True)
        self.relu = nn.ReLU()

    @staticmethod
    def _make_conv2d_layer(in_maps, out_maps, max_pool=False, padding = 1):
        layers = [nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=padding),
                nn.BatchNorm2d(out_maps)]
        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, returnt="out"):
        h0_prerelu = self.bn1(self.conv1(x))
        h0_postrelu = relu(h0_prerelu)
        h0 = self.maxpool(h0_postrelu)
        h1 = self.layer1(h0)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)
        
        if returnt=="slow-features":
            feat = self.avgpool(h4)
            return feat.view(feat.size(0),-1)
        
        m1_ = self.f_conv1(x)
        m1 = F.relu(m1_) * h1
        m2_ = self.f_conv2(m1)
        m2 = F.relu(m2_) * h2
        m3_ = self.f_conv3(m2)
        m3 = F.relu(m3_) * h3
        m4_ = self.f_conv4(m3)
        m4 = F.relu(m4_) * h4
        out = self.avgpool(m4)
        #out = self.avgpool(h4)
        out = out.view(out.size(0), -1)

        if returnt=="features":
            return out

        y = self.linear(out)

        if returnt=="out":
            return y
        elif returnt=="full":
            return y, [h0_postrelu, h1, h2, h3, h4, y]
        elif returnt=="both":
            return y, out

    def BarlowTwins(self, y1, y2):
        
        z1 = self.projector(self(y1,returnt="slow-features"))
        z2 = self.projector(self(y2,returnt="slow-features"))
        z_a = (z1 - z1.mean(0)) / z1.std(0)
        z_b = (z2 - z2.mean(0)) / z2.std(0)
        N, D = z_a.size(0), z_a.size(1)
        c_ = torch.mm(z_a.T, z_b) / N
        c_diff = (c_ - torch.eye(D).cuda()).pow(2)
        c_diff[~torch.eye(D, dtype=bool)] *= 2e-3
        loss = c_diff.sum()   
        return loss
    
def DualnetNetNet18(num_classes, nf=20):
    return DualnetNetNet(BasicBlock, [2, 2, 2, 2], num_classes, nf)