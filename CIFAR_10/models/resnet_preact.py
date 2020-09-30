'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Code from: https://github.com/legolas123/cv-tricks.com/blob/master/xnornet_plusplus/models/resnet_preact_bin.py
'''
class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        output = input.sign()
        return output
    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.gt(1)] = 0 ###TODO
        grad_input[input.lt(-1)] = 0
        return grad_input

binactive = BinActive.apply
class BinConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=1, padding=0, groups=1, bias = False, dilation = 1, output_height=0, output_width=0):
        super(BinConv2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, dilation=dilation)
        self.alpha = nn.Parameter(torch.ones(output_height).reshape(1,-1,1))
        self.beta = nn.Parameter(torch.ones(output_width).reshape(1,1,-1))
        self.gamma = nn.Parameter(torch.ones(output_channels).reshape(-1,1,1))
    def forward(self,x):
        x = binactive(x)
        x = self.conv(x)
        return x.mul(self.gamma).mul(self.beta).mul(self.alpha)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, binarize = False, output_height=0, output_width=0):
    """3x3 convolution with padding"""
    if binarize:
        return BinConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, output_height=output_height, output_width=output_width)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, binarize = False, output_height=0, output_width=0):
    """1x1 convolution"""
    if binarize:
        return BinConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, output_height=output_height, output_width=output_width)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, output_height=0, output_width=0):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride, output_height=output_height, 
                                        output_width=output_width, binarize=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, output_height=output_height, 
                                        output_width=output_width, binarize=True)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion*planes, stride, output_height=output_height, 
                                        output_width=output_width, binarize=True)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out +=  shortcut
        return out


# class PreActBottleneck(nn.Module):
#     '''Pre-activation version of the original Bottleneck module.'''
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(PreActBottleneck, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(x))
#         shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
#         out = self.conv1(out)
#         out = self.conv2(F.relu(self.bn2(out)))
#         out = self.conv3(F.relu(self.bn3(out)))
#         out += shortcut
#         return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, output_height=32, output_width=32)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, output_height=16, output_width=16)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, output_height=8, output_width=8)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, output_height=4, output_width=4)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, output_height=0, output_width=0):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, output_height, output_width))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18():
    return PreActResNet(PreActBlock, [2,2,2,2])

# def PreActResNet34():
#     return PreActResNet(PreActBlock, [3,4,6,3])

# def PreActResNet50():
#     return PreActResNet(PreActBottleneck, [3,4,6,3])

# def PreActResNet101():
#     return PreActResNet(PreActBottleneck, [3,4,23,3])

# def PreActResNet152():
#     return PreActResNet(PreActBottleneck, [3,8,36,3])


def test():
    net = PreActResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()