import torch
import torch.nn as nn

from Models import MonoNet, S2DF_3dense, MultipleBasicBlock_4, Occnet
from Models import Pwc_dc_net
from Models import BdcnNet, StructureModule, DetailEnhanceModule
from Models import U2_netp

from Models import Forward_singlePath_module, Forward_flownets_module

from my_package.FlowProjection import  FlowProjectionModule
from my_package.FilterInterpolation import  FilterInterpolationModule2

class SmiffNet(torch.nn.Module):
    def __init__(self, training = True):
        super(SmiffNet, self).__init__()
        self.training = training
        self.initScaleNets_filter, self.initScaleNets_filter1, self.initScaleNets_filter2 = \
            MonoNet(channel_in = 3, channel_out = 4 * 4) # filter_size = 4*4
        
        self.ctxNet = S2DF_3dense()
        self.ctx_ch = 3 * 64 + 3
        self.rectifyNet = MultipleBasicBlock_4(3+3+3+3+3+2*2+16*2+2*self.ctx_ch, 3, 128) # rectifier module stage 1
        self.rectifyNet_it = MultipleBasicBlock_4(3+3+3+3+3+2*2+16*2+2*self.ctx_ch, 3, 128) # rectifier module stage 2
        self.occmask = Occnet(6, 1)
        self.occmask_it1 = Occnet(6, 1) # fusion mask for stage 1
        self.occmask_it2 = Occnet(6, 1) # fusion mask for stage 2
        self.occdain = Occnet(12, 1)
        self._initialize_weights()
        
        self.flownets = Pwc_dc_net()    
        self.bdcn = BdcnNet()
        self.structure_gen = StructureModule()
        self.detail_enhance = DetailEnhanceModule()
        self.detail_enhance_it = DetailEnhanceModule()
        
        self.salNet = U2_netp()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, input):
        s1 = torch.cuda.current_stream()
        s2 = torch.cuda.current_stream()   
        if self.training:
            assert input.size(0) == 3
            input_t0, input_t2, input_gt = torch.squeeze(input, dim=0)
        else:
            assert input.size(0) == 2
            input_t0, input_t2 = torch.squeeze(input,dim=0)
        
        # Motion-based interpolation
        cur_filter_input = torch.cat((input_t0, input_t2), dim = 1)
        with torch.cuda.stream(s1):
            cur_ctx_output = [
                self.ctxNet(cur_filter_input[:, :3, ...]),
                self.ctxNet(cur_filter_input[:, 3:, ...])
            ]
            filter_output = Forward_singlePath_module(
                self.initScaleNets_filter, cur_filter_input, 'filter'
            )
            cur_filter_output = [
                Forward_singlePath_module(self.initScaleNets_filter1, filter_output, name = None),
                Forward_singlePath_module(self.initScaleNets_filter2, filter_output, name = None)
            ]
        
        cur_offset_input = cur_filter_input
        with torch.cuda.stream(s2):
            cur_offset_outputs = [
                Forward_flownets_module(
                    self.flownets, 
                    cur_offset_input
                ),
                Forward_flownets_module(
                    self.flownets, 
                    torch.cat((cur_offset_input[:, 3:, ...], cur_offset_input[:, 0:3, ...]), dim = 1)
                )
            ]
        cur_offset_outputs = [
            self.FlowProject(cur_offset_outputs[0]),
            self.FlowProject(cur_offset_outputs[1])
        ]
        cur_offset_output = [cur_offset_outputs[0][0], cur_offset_outputs[1][0]]
        torch.cuda.synchronize()
        
        ctx_from_t0, _ = self.FilterInterpolate(
            cur_ctx_output[0], cur_offset_output[0].detach(), cur_filter_output[0].detach()
        )
        ctx_from_t2, _ = self.FilterInterpolate(
            cur_ctx_output[1], cur_offset_output[1].detach(), cur_filter_output[1].detach()
        )
        interp_image_from_t0, mask_t0 = self.FilterInterpolate(
            input_t0, cur_offset_output[0], cur_filter_output[0]
        )
        interp_image_from_t2, mask_t2 = self.FilterInterpolate(
            input_t2, cur_offset_output[1], cur_filter_output[1]
        )   
        flow_mask = self.occdain(
            torch.cat((interp_image_from_t0, cur_offset_output[0], mask_t0), dim = 1), 
            torch.cat((interp_image_from_t2, cur_offset_output[1], mask_t2), dim = 1)
        )
        interp_image_via_opticalflow = flow_mask * interp_image_from_t0 + \
                                   (1 - flow_mask) * interp_image_from_t2
        interp_image_via_opticalflow = (interp_image_via_opticalflow - 0.5) / 0.5
        
        # Structure-based interpolation
        str_input_t0 = (input_t0 - 0.5) / 0.5
        str_input_t2 = (input_t2 - 0.5) / 0.5
        str_enc_feat_t0 = torch.cat([str_input_t0, torch.tanh(self.bdcn(str_input_t0)[0])], dim = 1)
        str_enc_feat_t2 = torch.cat([str_input_t2, torch.tanh(self.bdcn(str_input_t2)[0])], dim = 1)
        interp_image_via_structure = self.structure_gen((str_enc_feat_t0, str_enc_feat_t2))
            
        # fusion via mask-0
        interp_ms_fus = self.mask_based_fusion(
            interp_image_via_opticalflow, interp_image_via_structure, self.occmask)

        #iter 1 via temporal-spatial based enhancement
        interp_temp_enh_iter1 = self.detail_enhance((str_input_t0, str_input_t2, interp_ms_fus))
        interp_spat_enh_iter1 = self.rectifyNet(
            torch.cat((
                interp_image_via_opticalflow * 0.5 + 0.5, 
                interp_image_via_structure * 0.5 + 0.5,
                interp_ms_fus * 0.5 + 0.5,
                interp_image_from_t0, interp_image_from_t2,
                cur_offset_output[0], cur_offset_output[1],
                cur_filter_output[0], cur_filter_output[1],
                ctx_from_t0, ctx_from_t2
            ), dim = 1)
        ) + interp_ms_fus * 0.5 + 0.5
        interp_spat_enh_iter1 = (interp_spat_enh_iter1 - 0.5) / 0.5
        
        # fusion via mask-1
        interp_temp_spat_fus_iter1 = self.mask_based_fusion(
            interp_temp_enh_iter1, interp_spat_enh_iter1, self.occmask_it1)
        
        #iter 2 via temporal-spatial based enhancement
        interp_temp_enh_iter2 = self.detail_enhance_it((str_input_t0, str_input_t2, interp_temp_spat_fus_iter1))
        interp_spat_enh_iter2 = self.rectifyNet_it(
            torch.cat((
                interp_temp_enh_iter1 * 0.5 + 0.5, 
                interp_spat_enh_iter1 * 0.5 + 0.5,
                interp_temp_spat_fus_iter1 * 0.5 + 0.5,
                interp_image_from_t0, interp_image_from_t2,
                cur_offset_output[0], cur_offset_output[1],
                cur_filter_output[0], cur_filter_output[1],
                ctx_from_t0, ctx_from_t2
            ), dim = 1)
        ) + interp_ms_fus * 0.5 + 0.5
        interp_spat_enh_iter2 = (interp_spat_enh_iter2 - 0.5) / 0.5
        
        # fusion via mask-2
        interp_temp_spat_fus_iter2 = self.mask_based_fusion(
            interp_temp_enh_iter2, interp_spat_enh_iter2, self.occmask_it2)   
        interp_temp_spat_fus_iter2 = interp_temp_spat_fus_iter2 * 0.5 + 0.5     
        
        if self.training:
            return interp_temp_spat_fus_iter2 - input_gt
        else:
            return interp_temp_spat_fus_iter2
        
    @staticmethod
    def FlowProject(inputs):
        outputs = [FlowProjectionModule(input.requires_grad)(input) for input in inputs]
        return outputs
    
    @staticmethod
    def FilterInterpolate(feat, offset, filter):
        feat_offset, mask = FilterInterpolationModule2()(feat, offset, filter)  
        mask = torch.mean(mask, dim = 1).unsqueeze(dim = 1)
        return feat_offset, mask
    
    def mask_based_fusion(self, input1, input2, fus_mask_func):
        mask = fus_mask_func(input1, input2)
        fus_res = mask * input1 + (1 - mask) * input2
        return fus_res