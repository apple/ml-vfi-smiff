import torch.nn as nn

def conv_relu_conv(input_filter, output_filter, kernel_size, padding):
    layers = nn.Sequential(
        nn.Conv2d(input_filter, input_filter, kernel_size, 1, padding),
        nn.ReLU(inplace=False),
        nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
    )
    return layers

def conv_relu(input_filter, output_filter, kernel_size, padding):
    layers = nn.Sequential(*[
        nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),
        nn.ReLU(inplace=False)
    ])
    return layers

def conv_relu_maxpool(input_filter, output_filter, kernel_size, padding, kernel_size_pooling):
    layers = nn.Sequential(*[
        nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size_pooling)
    ])
    return layers

def conv_relu_unpool(input_filter, output_filter, kernel_size, padding, unpooling_factor):
    layers = nn.Sequential(*[
        nn.Upsample(scale_factor=unpooling_factor, mode='bilinear'),
        nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),
        nn.ReLU(inplace=False),
    ])
    return layers
    
def MonoNet(channel_in, channel_out):

    '''
    Generally, the MonoNet is aimed to provide a basic module for generating either offset, or filter, or occlusion.

    :param channel_in: number of channels that composed of multiple useful information like reference frame, previous coarser-scale result
    :param channel_out: number of output the offset or filter or occlusion
    :param name: to distinguish between offset, filter and occlusion, since they should use different activations in the last network layer

    :return: output the network model
    '''
    model = list()
    # encoder
    model.append(conv_relu(channel_in * 2, 16, (3, 3), (1, 1)))
    model.append(conv_relu_maxpool(16, 32, (3, 3), (1, 1), (2, 2)))  
    model.append(conv_relu_maxpool(32, 64, (3, 3), (1, 1), (2, 2)))
    model.append(conv_relu_maxpool(64, 128, (3, 3), (1, 1), (2, 2)))
    model.append(conv_relu_maxpool(128, 256, (3, 3), (1, 1), (2, 2)))
    model.append(conv_relu_maxpool(256, 512, (3, 3), (1, 1), (2, 2)))
    model.append(conv_relu(512, 512, (3, 3), (1, 1)))
    # decoder
    model.append(conv_relu_unpool(512, 256, (3, 3), (1, 1), 2))
    model.append(conv_relu_unpool(256, 128, (3, 3), (1, 1), 2))
    model.append(conv_relu_unpool(128, 64, (3, 3), (1, 1), 2))
    model.append(conv_relu_unpool(64, 32, (3, 3), (1, 1), 2))
    model.append(conv_relu_unpool(32,  16, (3, 3), (1, 1), 2))

    # output final purpose
    branch1 = list()
    branch2 = list()
    branch1.append(conv_relu_conv(16, channel_out,  (3, 3), (1, 1)))
    branch2.append(conv_relu_conv(16, channel_out,  (3, 3), (1, 1)))
    
    return  (nn.ModuleList(model), nn.ModuleList(branch1), nn.ModuleList(branch2))