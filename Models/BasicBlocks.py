import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, dilation = 1, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=int(dilation*(3-1)/2), dilation=dilation, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, dilation = 1, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes,dilation, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class MultipleBasicBlock(nn.Module):
    def __init__(self,input_feature,output_feature,
                 block, num_blocks,
                 intermediate_feature = 64, dense = True):
        super(MultipleBasicBlock, self).__init__()
        self.dense = dense
        self.num_block = num_blocks
        self.intermediate_feature = intermediate_feature

        self.block1= nn.Sequential(*[
            nn.Conv2d(input_feature, intermediate_feature,
                      kernel_size=7, stride=1, padding=3, bias=True),
            nn.ReLU(inplace=True)
        ])

        self.block2 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=2 else None
        self.block3 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=3 else None
        self.block4 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=4 else None
        self.block5 = nn.Sequential(*[nn.Conv2d(intermediate_feature, output_feature, (3, 3), 1, (1, 1))])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x) if self.num_block>=2 else x
        x = self.block3(x) if self.num_block>=3 else x
        x = self.block4(x) if self.num_block== 4 else x
        x = self.block5(x)
        return x

def MultipleBasicBlock_4(input_feature,output_feature, intermediate_feature = 64):
    model = MultipleBasicBlock(input_feature, output_feature,
                               BasicBlock,4 ,
                               intermediate_feature)
    return model

def MultipleBasicBlock_3(input_feature,output_feature, intermediate_feature = 64):
    model = MultipleBasicBlock(input_feature, output_feature,
                               BasicBlock,3 ,
                               intermediate_feature)
    return model

def MultipleBasicBlock_2(input_feature,output_feature, intermediate_feature = 64):
    model = MultipleBasicBlock(input_feature, output_feature,
                               BasicBlock,2 ,
                               intermediate_feature)
    return model
