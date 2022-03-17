import math
import torch
from torch import nn
from torch.nn import functional as F

padding = 0

def conv_layer(in_channels, out_channels, kernel, strides, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=strides, padding_mode="zeros", bias=False,
                     padding=padding)


def batch_norm(filters):
    return nn.BatchNorm2d(filters)


def relu():
    return nn.ReLU()

class Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride, kernel_size, short):
        super(Block, self).__init__()

        self.short = short
        self.bn1 = batch_norm(in_channels)
        self.relu1 = relu()
        self.conv1 = conv_layer(in_channels, out_channels, 1, stride, padding=0)

        self.conv2 = conv_layer(in_channels, out_channels, kernel_size, stride)
        self.bn2 = batch_norm(out_channels)
        self.relu2 = relu()
        self.conv3 = conv_layer(out_channels, out_channels, kernel_size, 1)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu1(x)

        x_short = x
        if self.short:
            x_short = self.conv1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        out = x + x_short
        return out

class FeatureModelResNet(nn.Module):

    def __init__(self, in_channels, strides=[1, 2, 2, 2], filters=[48, 48, 48, 48]):
        super(FeatureModelResNet, self).__init__()

        stride_prev = strides.pop(0)
        filters_prev = filters.pop(0)

        self.conv1 = conv_layer(in_channels, filters_prev, 3, stride_prev)

        module_list = nn.ModuleList()
        for s, f in zip(strides, filters):
            module_list.append(Block(filters_prev, f, s, 3, s != 1 or f != filters_prev))

            stride_prev = s
            filters_prev = f

        self.module_list = nn.Sequential(*module_list)

        self.bn1 = batch_norm(filters_prev)
        self.relu1 = relu()

    def forward(self, x):
        out = self.conv1(x)
        out = self.module_list(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = F.adaptive_avg_pool2d(out, (1, 1)) 
        out = out.view(out.shape[0], out.shape[1])
        out = F.normalize(out, p=2, dim=-1)
        return out




class FeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.layers = \
            torch.nn.Sequential(
                torch.nn.Conv2d(3, 36, kernel_size=5, padding=(2, 2)),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(36, 48, kernel_size=5, padding=(2, 2)),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.layers2 = \
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 36, kernel_size=5, padding=(2, 2)),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(36, 48, kernel_size=3, padding=(1, 1)),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
            )
    def forward(self, x):
        if x.shape[1] == 3:
            x = self.layers(x)
        elif x.shape[1] == 1:
            x = self.layers2(x)

        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        x = x.view(x.shape[0], -1)
        x = F.normalize(x, p=2, dim=1) # force the features have the same L2 norm.
        return x


class Classifier(torch.nn.Sequential):
    def __init__(self, output_num=2):
        super(Classifier, self).__init__()
        self.layers = \
            torch.nn.Sequential(
                torch.nn.Linear(48, 512),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(512, 512),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(512, output_num),
            )

    def forward(self, x):
        x = self.layers(x)
        prediction = F.softmax(x)
        return x, prediction



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm=lambda x: nn.InstanceNorm2d(x, affine=True),
                 kernel_size=1, dropout=False, compensate=False):
        super(Bottleneck, self).__init__()
        mid_planes = planes
        if compensate:
            mid_planes = int(2.5 * planes)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)

        self.conv2 = nn.Conv2d(planes, mid_planes, kernel_size=kernel_size,
                               stride=stride,
                               padding=(kernel_size - 1) // 2,  # used to be 0
                               bias=False)  # changed padding from (kernel_size - 1) // 2
        self.bn2 = norm(mid_planes)

        self.drop = nn.Dropout2d(p=0.2) if dropout else lambda x: x
        self.conv3 = nn.Conv2d(mid_planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if residual.size(-1) != out.size(-1):
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:, :, :-diff, :-diff]

        out += residual
        out = self.relu(out)

        return out


class BagNetEncoder(nn.Module):
    norms = {
        'in_aff': lambda x: nn.InstanceNorm2d(x, affine=True),
        'in': nn.InstanceNorm2d,
        'bn': nn.BatchNorm2d
    }

    def __init__(self, block, layers, strides=[1, 2, 2, 2], wide_factor=1,
                 kernel3=[0, 0, 0, 0], dropout=False, inp_channels=3,
                 compensate=False, norm='in_aff'):
        self.planes = int(64 * wide_factor)
        self.inplanes = int(64 * wide_factor)
        self.compensate = compensate
        self.dropout = dropout
        self.norm = norm
        super(BagNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(inp_channels, self.planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(self.planes, self.planes, kernel_size=3,
                               stride=1, padding=0, bias=False)
        self.bn1 = self.norms[self.norm](self.planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, self.planes, layers[0],
                                       stride=strides[0], kernel3=kernel3[0],
                                       prefix='layer1')
        self.layer2 = self._make_layer(block, self.planes * 2, layers[1],
                                       stride=strides[1], kernel3=kernel3[1],
                                       prefix='layer2')
        self.layer3 = self._make_layer(block, self.planes * 4, layers[2],
                                       stride=strides[2], kernel3=kernel3[2],
                                       prefix='layer3')
        self.layer4 = self._make_layer(block, self.planes * 8, layers[3],
                                       stride=strides[3], kernel3=kernel3[3],
                                       prefix='layer4')
        self.block = block

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.InstanceNorm2d) and self.norm == 'in_aff':
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0,
                    prefix=''):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norms[self.norm](planes * block.expansion),
            )

        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride, downsample,
                            kernel_size=kernel, dropout=self.dropout,
                            norm=self.norms[self.norm],
                            compensate=(self.compensate and kernel == 1)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel,
                                norm=self.norms[self.norm],
                                compensate=(self.compensate and kernel == 1)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

