import torch
import torch.nn as nn
from Utils import Stack

def Forward_singlePath_module(modulelist, input, name):
    stack = Stack()
    k = 0
    temp = []
    for layers in modulelist:  
        if k == 0:
            temp = layers(input)
        else:
            if isinstance(layers, nn.AvgPool2d) or isinstance(layers,nn.MaxPool2d):
                stack.push(temp)

            temp = layers(temp)
            if isinstance(layers, nn.Upsample):
                if name == 'offset':
                    temp = torch.cat((temp,stack.pop()),dim = 1)
                else:
                    temp += stack.pop()
        k += 1
    return temp

def Forward_flownets_module(model, input):
    output = model(input) * 20 * 0.5
    outputs = [nn.Upsample(scale_factor = 4, mode='bilinear')(output)]
    return outputs