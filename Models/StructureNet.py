import torch
import torch.nn as nn

from Models import ExtractFeatures, DeformableConv, GlobalGenerator
      
class StructureGen(nn.Module):
    def __init__(self, feature_level = 3):
        super(StructureGen, self).__init__()
        self.feature_level = feature_level
        channel = 2 ** (6 + self.feature_level)
        self.extract_features = ExtractFeatures()
        self.dcn = DeformableConv(channel, dg = 16)
        self.generator = GlobalGenerator(channel, 4, n_downsampling=self.feature_level)

    def forward(self, input):
        img0_e, img1_e = input
        ft_img0 = list(self.extract_features(img0_e))[self.feature_level]
        ft_img1 = list(self.extract_features(img1_e))[self.feature_level]
        pre_gen_ft_imgt, _, _ = self.dcn(ft_img0, ft_img1)
        ref_imgt_e = self.generator(pre_gen_ft_imgt)
        ref_imgt = ref_imgt_e[:, :3]
        return  ref_imgt

def StructureModule(dict_path = None):
    structure_gen = StructureGen(feature_level = 3)
    structure_gen.cuda()
    if dict_path is not None:
        dict_path = torch.load(dict_path)
        structure_gen.load_state_dict(dict_path['state_dictGEN'])
    return structure_gen