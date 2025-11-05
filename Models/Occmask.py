import torch

class OccEst(torch.nn.Module):
    def __init__(self, input_channel, output_channel):
        super(OccEst, self).__init__()
        def Subnet_occlusion(input_channel, output_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.Sigmoid()
            )
        self.moduleOcclusion = Subnet_occlusion(input_channel, output_channel)

    def forward(self, input1, input2):
        tensorJoin = torch.cat((input1, input2), dim=1)
        Occlusion = self.moduleOcclusion(tensorJoin)
        return Occlusion

class OccEst_g(torch.nn.Module):
    def __init__(self, input_channel, output_channel):
        super(OccEst_g, self).__init__()
        def Subnet_occlusion(input_channel, output_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=output_channel, kernel_size=3, stride=1, padding=1)
            ) 
        self.moduleOcclusion = Subnet_occlusion(input_channel, output_channel)
        self.sm = torch.nn.Sigmoid()

    def forward(self, input1, input2):
        tensorJoin = torch.cat((input1, input2), dim=1)
        Occlusion = self.moduleOcclusion(tensorJoin)
        Occlusion = torch.nn.functional.adaptive_avg_pool2d(Occlusion,(1,1))
        Occlusion = self.sm(Occlusion)
        return Occlusion

class EstMask(torch.nn.Module):
    def __init__(self, input_channel, output_channel):
        super(EstMask, self).__init__()
        def Subnet_fus(input_channel, output_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=output_channel, kernel_size=3, stride=1, padding=1)
            )
        self.moduleFus = Subnet_fus(input_channel, output_channel)

    def forward(self, input1, input2):
        tensorJoin = torch.cat((input1, input2), dim=1)
        Occlusion = self.moduleFus(tensorJoin)
        return Occlusion

def Occnet(input_channel, output_channel):
    model = OccEst(input_channel, output_channel)
    return model 

def Occnet_g(input_channel, output_channel):
    model = OccEst_g(input_channel, output_channel)
    return model

def Mask_fus(input_channel, output_channel):
    model = EstMask(input_channel, output_channel)
    return model


