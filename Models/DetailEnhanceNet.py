import torch
import torch.nn as nn

from Models import ValidationFeatures, ExtractAlignedFeatures, PCD_Align, TSA_Fusion, Reconstruct
      
class DetailEnhance(nn.Module):
    def __init__(self):
        super(DetailEnhance, self).__init__()
        self.extract_features = ValidationFeatures()
        self.extract_aligned_features = ExtractAlignedFeatures(n_res = 5)
        self.pcd_align = PCD_Align(groups = 8)
        self.tsa_fusion = TSA_Fusion(nframes = 3, center = 1)
        self.reconstruct = Reconstruct(n_res = 20)

    def forward(self, input):
        """
        Network forward tensor flow

        :param input: a tuple of input that will be unfolded
        :return: medium interpolation image
        """
        img0, img1, ref_imgt = input
        ref_align_ft = self.extract_aligned_features(ref_imgt)
        align_ft_0 = self.extract_aligned_features(img0)
        align_ft_1 = self.extract_aligned_features(img1)
        align_ft = [self.pcd_align(align_ft_0, ref_align_ft),
                    self.pcd_align(ref_align_ft, ref_align_ft),
                    self.pcd_align(align_ft_1, ref_align_ft)]
        align_ft = torch.stack(align_ft, dim=1)
        tsa_ft = self.tsa_fusion(align_ft)
        imgt = self.reconstruct(tsa_ft, ref_imgt)
        return imgt

def DetailEnhanceModule(dict_path = None):
    detail_enhance = DetailEnhance()
    detail_enhance.cuda()
    if dict_path is not None:
        dict_path = torch.load(dict_path)
        detail_enhance.load_state_dict(dict_path['state_dictDE'])
    return detail_enhance