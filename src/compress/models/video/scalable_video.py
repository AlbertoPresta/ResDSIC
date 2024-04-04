import math

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda import amp

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import QReLU
from compressai.ops import quantize_ste
from compressai.registry import register_model

from ..base import CompressionModel
from ..utils import conv, deconv, gaussian_blur, gaussian_kernel2d, meshgrid2d
from .google import ScaleSpaceFlow
from .modules import Encoder, Decoder, HyperpriorMasked
from compress.layers.mask_layers import Mask

@register_model("ssf2020scalable")
class ScalableScaleSpaceFlow(ScaleSpaceFlow):
    def __init__(
        self,
        num_levels: int = 5,
        sigma0: float = 1.5,
        scale_field_shift: float = 1.0,
        scalable_ref = True,
        mask_policy = "two-levels",
        multiple_ref_decoder = True,
        multiple_ref_encoder = True, 
        multiple_flow_encoder = True,
        multiple_flow_decoder = True,
        multiple_res_encoder = True, 
        multiple_res_decoder = True,
        scalable_levels = 2
    ):
        super().__init__(num_levels=num_levels,sigma0=sigma0,scale_field_shift=scale_field_shift)

        self.multiple_ref_decoder = multiple_ref_decoder
        self.multiple_ref_encoder = multiple_ref_encoder 
        self.multiple_flow_encoder = multiple_flow_encoder
        self.multiple_flow_decoder = multiple_flow_decoder
        self.multiple_res_encoder = multiple_res_encoder
        self.multiple_res_decoder = multiple_res_decoder
        self.scalable_ref = scalable_ref

        assert isinstance(self.mask_policy,List)
        self.scalable_levels = scalable_levels 
        assert len(self.mask_policy) == 3 
        self.masking = [Mask(self.mask_policy[0],scalable_levels = self.scalable_levels) for _ in range(3)] 




        if self.scalable_ref:
            self.img_encoder = nn.ModuleList(Encoder(3) for _ in range(2)) if self.multiple_ref_encoder else Encoder(3, factor = 2) 
            self.img_decoder =  nn.ModuleList(Decoder(3) for _ in range(2)) if self.multiple_ref_decoder else Decoder(3,factor = 2)
            self.img_hyperprior = HyperpriorMasked(factor = 2, mask_policy=mask_policy[0],scalable_levels=scalable_levels)

        
        self.res_encoder = nn.ModuleList(Encoder(3) for _ in range(2)) if self.multiple_res_encoder else Encoder(3,factor = 2)
        self.res_decoder = nn.ModuleList(Decoder(3,in_planes=384) for _ in range(2)) if self.multiple_res_decoder else Decoder(3,factor=2,in_planes=384)
        self.res_hyperprior = HyperpriorMasked(factor = 2, mask_policy=mask_policy[1],scalable_levels=scalable_levels)


        self.motion_encoder = nn.ModuleList(Encoder(2*3) for _ in range(2)) if self.multiple_flow_encoder else Encoder(2*3,factor = 2)
        self.motion_decoder = nn.ModuleList( Decoder(3) for _ in range(2)) if self.multiple_flow_decoder else Decoder(3, factor = 2)
        self.motion_hyperprior = HyperpriorMasked(factor = 2, mask_policy=mask_policy[2],scalable_levels=scalable_levels)


    def define_quality(self,quality):
        if quality is None:
            list_quality = self.quality_list
        elif isinstance(quality,list):
            if quality[0] == 0:
                list_quality = quality 
            else:
                list_quality = [0] + quality
        else:
            list_quality = [quality] 
        return list_quality


    def forward(self, frames, quality = None, mask_pol = None, training = True ):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")



        if mask_pol is None:
            mask_pol = self.mask_policy
        list_quality = self.define_quality(quality)  
        reconstructions_base = []
        frames_likelihoods_base = []
    
        reconstructions_prog = []
        frames_likelihoods_prog = []
        
        frames_likelihoods_hype = []

        x_hat ,likelihoods = self.forward_keyframe(frames[0], 
                                                    quality = list_quality, 
                                                    mask_pol =mask_pol, 
                                                    training =training) # I-encoder! modeled as normal frame compression algorithm (choose a good one)
        
        
    
        reconstructions_base.append(x_hat[0])
        frames_likelihoods_base.append(likelihoods["keyframe"][0])


        reconstructions_prog.append(x_hat[1])
        frames_likelihoods_prog.append(likelihoods["keyframe"][1])

        frames_likelihoods_hype.append(likelihoods["keyhype"])
        x_ref_b = x_hat[0].detach()  # stop gradient flow (cf: google2020 paper)
        x_ref_p = x_hat[1].detach() # stop gradient flow (cf: google2020 paper)
        
        x_ref = [x_ref_b, x_ref_p]
        
        for i in range(1, len(frames)):
            x = frames[i]
            x_ref, likelihoods = self.forward_inter(x, 
                                                    x_ref, 
                                                    quality = list_quality, 
                                                    mask_pol =mask_pol, 
                                                    training =training)
            reconstructions_base.append(x_ref[0])
            frames_likelihoods_base.append(likelihoods[0])



            

            # fare la stessa cosa per la qualità 2
            x = frames[i]
            x_ref_prog, likelihoods = self.forward_inter(x, x_ref_prog, quality = 1, mask_pol =mask_pol, training =training)
            reconstructions_base.append(x_ref)
            frames_likelihoods_prog.append(likelihoods)           
            

            
        return {
            "x_hat": reconstructions,
            "likelihoods": frames_likelihoods,
        }



    def forward_keyframe(self, x, quality, mask_pol, training):  #sembra essere Balle18
        
        if self.multiple_ref_encoder:
            y_b = self.img_encoder[0](x)
            y_p = self.img_encoder[1](x)
            y = torch.cat([y_b,y_p],dim = 1) 
        else: 
            y = self.img_encoder(x)

        z = self.img_hyperprior.hyper_encoder(y)
        z_hat, z_likelihoods = self.img_hyperprior.entropy_bottleneck(z, training = training) 
        
        scales = self.img_hyperprior.hyper_decoder_scale(z_hat)
        means = self.img_hyperprior.hyper_decoder_mean(z_hat)    
        
        scales_b, scales_p = 0,0 # chunck scaes 
        means_b,means_p = 0,0 # chunk means 

        # this is the base       
        _, y_likelihoods_b = self.img_hyperprior.gaussian_conditional(y_b, scales_b, means_b)
        y_hat_b = quantize_ste(y - means_b) + means_b
        
        
        mask = self.img_hyperprior.compute_mask(scales if self.double_dim else scales_p, 
                                                mask_pol = mask_pol, 
                                                quality = quality) 
        
    
        _, y_likelihoods_p = self.img_hyperprior.gaussian_conditional((y_p - means_p)*mask, scales_p*mask)
        y_hat_p = quantize_ste(y_p - means_p)*mask  + means_p 
        y_hat_p = y_hat_b + y_hat_p
        
    
        x_hat_b = self.img_decoder[0](y_hat_b) if self.multiple_ref_decoder else  self.img_decoder(y_hat_b)
        x_hat_p = self.img_decoder[1](y_hat_p) if self.multiple_ref_decoder else  self.img_decoder(y_hat_p)
        

        x_hat = [x_hat_b, x_hat_p] 
        likelihoods = [y_likelihoods_b,y_likelihoods_p]
        return x_hat, {"keyhype": z_likelihoods,"keyframe": likelihoods}   
        

    def forward_inter(self, x_cur, x_ref, mask_pol, quality, training = True):
        # encode the motion information
        

        x = torch.cat((x_cur, x_ref[0]), dim=1)  if quality == 0 else torch.cat((x_cur, x_ref[1]), dim=1)# cat the input 
        
        
        if self.multiple_flow_encoder:
            y_motion_b = self.motion_encoder[0](x) 
            y_motion_p = self.motion_encoder[1](x)
            y_motion = torch.cat([y_motion_b,y_motion_p],dim = 1)
        else: 
            y_motion = self.motion_encoder(x)
        
        z_motion = self.motion_hyperprior.hyper_encoder(y_motion)
        z_hat_motion, z_likelihoods_motion = self.motion_hyperprior.entropy_bottleneck(z_motion, training = training)        
        


        # decode the space-scale flow information
        motion_info = self.motion_decoder(y_motion_hat) # scale space flow Decoder 
        x_pred = self.forward_prediction(x_ref, motion_info) # scale space warping

        # residual
        x_res = x_cur - x_pred
        y_res = self.res_encoder(x_res) # residual encoder, da aumentare 
        y_res_hat, res_likelihoods = self.res_hyperprior(y_res) # y_res_hat = v_hat^i

        # y_combine
        y_combine = torch.cat((y_res_hat, y_motion_hat), dim=1) #w_i + v_i
        x_res_hat = self.res_decoder(y_combine) # x_res_hat = r^_i

        # final reconstruction: prediction + residual
        x_rec = x_pred + x_res_hat # final reconstruction 

        return x_rec, {"motion": motion_likelihoods, "residual": res_likelihoods}


        










