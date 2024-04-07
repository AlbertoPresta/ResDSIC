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
        frames_likelihoods = []
    
        reconstructions_prog = []

        
        frames_likelihoods_hype = []

        x_hat ,likelihoods = self.forward_keyframe(frames[0], 
                                                    quality = list_quality, 
                                                    mask_pol =mask_pol, 
                                                    training =training) # I-encoder! modeled as normal frame compression algorithm (choose a good one)
        
        
    
        reconstructions_base.append(x_hat[0])
        frames_likelihoods.append(likelihoods["keyframe"])


        reconstructions_prog.append(x_hat[1])


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
            reconstructions_prog.append(x_ref[1])
            frames_likelihoods.append(likelihoods)
            frames_likelihoods_hype.append(likelihoods["hype"])
            
        reconstructions = [reconstructions_base,reconstructions_prog]
        return {
            "x_hat": reconstructions,
            "likelihoods": frames_likelihoods,
            "hyperprior": frames_likelihoods_hype
        }



    def forward_block(self,y,y_b,y_p,funct,training,mask_pol,quality):
        z_motion = self.motion_hyperprior.hyper_encoder(y)
        z_hat, z_likelihoods = funct.entropy_bottleneck(z_motion, 
                                                        training = training)        
        

        scales = funct.hyper_decoder_scale(z_hat)
        means = funct.hyper_decoder_mean(z_hat)    
        
        scales_b, scales_p = scales.chunk(2,1) # chunck scaes 
        means_b,means_p = means.chuck(2,1) # chunk means 

        # this is the base       
        _, y_likelihoods_b = funct.gaussian_conditional(y_b, 
                                                        scales_b, 
                                                        means_b)
        
        y_hat_b  = quantize_ste(y_b - means_b) + means_b
            

        mask = funct.compute_mask(scales if self.double_dim else scales_p, 
                                                mask_pol = mask_pol, 
                                                quality = quality) 
        
        _, y_likelihoods_p = funct.gaussian_conditional((y_p - means_p)*mask,
                                                        scales_p*mask)
        
        y_hat_p = quantize_ste(y_p - means_p)*mask  + means_p 
        y_hat_p = y_hat_b + y_hat_p

        return [y_hat_b,y_hat_p],[z_likelihoods,y_likelihoods_b,y_likelihoods_p]
        


    def forward_keyframe(self, x, quality, mask_pol, training):  #sembra essere Balle18
        
        if self.multiple_ref_encoder:
            y_b = self.img_encoder[0](x)
            y_p = self.img_encoder[1](x)
            y = torch.cat([y_b,y_p],dim = 1) 
        else: 
            y = self.img_encoder(x)
            y_b, y_p =y.chunck(2,1) #chuck the two components

        y_hat, likelihoods = self.forward_block(y,
                                                y_b,
                                                y_p,
                                                self.img_hyperprior,
                                                training,
                                                mask_pol,
                                                quality)

        y_hat_b,y_hat_p = y_hat[0],y_hat[1]
        z_likelihoods = likelihoods[0]
        y_likelihoods_b = likelihoods[1]
        y_likelihoods_p = likelihoods[2]
        
        x_hat_b = self.img_decoder[0](y_hat_b) if self.multiple_ref_decoder else  self.img_decoder(y_hat_b)
        x_hat_p = self.img_decoder[1](y_hat_p) if self.multiple_ref_decoder else  self.img_decoder(y_hat_p)
        

        x_hat = [x_hat_b, x_hat_p] 
        likelihoods = [y_likelihoods_b,y_likelihoods_p]
        return x_hat, {"keyhype": z_likelihoods,"keyframe": likelihoods}   
        



    def forward_inter(self, x_cur, x_ref, mask_pol, quality, training = True):
        # encode the motion information
        

        x_b = torch.cat((x_cur, x_ref[0]), dim=1)  
        x_p = torch.cat((x_cur, x_ref[1]), dim=1)# cat the input 
        
        y_motion_b = self.motion_encoder[0](x_b) if self.multiple_flow_encoder else self.motion_encoder(x_b)
        y_motion_p = self.motion_encoder[1](x_p) if self.multiple_flow_encoder else self.motion_encoder(x_p)
        y_motion = torch.cat([y_motion_b,y_motion_p],dim = 1)

        
        y_hat_motion, likelihoods_motion = self.forward_block(y_motion,
                                                            y_motion_b,
                                                               y_motion_p,
                                                             self.motion_hyperprior,
                                                             training,
                                                             mask_pol,
                                                             quality)

        y_hat_motion_b,y_hat_motion_p = y_hat_motion[0],y_hat_motion[1]
        z_likelihoods_motion = likelihoods_motion[0]
        y_likelihoods_motion_b = likelihoods_motion[1]
        y_likelihoods_motion_p = likelihoods_motion[2]
        
        motion_info_b = self.motion_decoder[0](y_hat_motion_b) if self.multiple_flow_decoder \
                        else self.motion_decoder(y_hat_motion_b) # scale space flow Decoder 
        x_pred_b = self.forward_prediction(x_ref[0], motion_info_b) # scale space warping


        # decode the space-scale flow information
        motion_info_p = self.motion_decoder[1](y_hat_motion_p) if self.multiple_flow_decoder \
                        else self.motion_decoder(y_hat_motion_p) # scale space flow Decoder 
        x_pred_p = self.forward_prediction(x_ref[1], motion_info_p)

        # residual BASE
        x_res_b = x_cur - x_pred_b
        x_res_p = x_cur - x_pred_p
        x_res = [x_res_b, x_res_p]

        y_res_b = self.res_encoder[0](x_res[0])  if self.multiple_res_encoder \
                                                else self.res_encoder(x_res[0])# residual encoder, da aumentare 
        
        
        y_res_p = self.res_encoder[1](x_res[1]) if self.multiple_res_encoder \
                                                else self.res_encoder(x_res[1]) 

        y_res = torch.cat([y_res_b,y_res_p],dim=1)

        y_hat_res, likelihoods_res = self.forward_block(y_res,
                                                            y_res_b,
                                                               y_res_p,
                                                             self.res_hyperprior,
                                                             training,
                                                             mask_pol,
                                                             quality)

        y_hat_res_b,y_hat_res_p = y_hat_res[0],y_hat_res[1]
        z_likelihoods_res = likelihoods_res[0]
        y_likelihoods_res_b = likelihoods_res[1]
        y_likelihoods_res_p = likelihoods_res[2]


        y_combine_b = torch.cat((y_hat_res_b,y_hat_motion_b), dim=1) #w_i + v_i
        x_res_hat_b = self.res_decoder[0](y_combine_b) if self.multiple_res_decoder  \
                                                    else self.res_decoder(y_combine_b) # x_res_hat = r^_i
        # final reconstruction: prediction + residual
        x_rec_b = x_pred_b + x_res_hat_b # final reconstruction




        y_combine_p = torch.cat((y_hat_res_p,y_hat_motion_p), dim=1)
        x_res_hat_p = self.res_decoder[1](y_combine_p) if self.multiple_res_decoder  \
                                                    else self.res_decoder(y_combine_p) # x_res_hat = r^_i

        # final reconstruction: prediction + residual
        x_rec_p = x_pred_p + x_res_hat_p # final reconstruction


        hype_likelihoods = {
            "motion": z_likelihoods_motion,
            "res": z_likelihoods_res,
        }
        base_likelihoods = {
            "motion": y_likelihoods_motion_b,
            "residual":y_likelihoods_res_b
        }

        prog_likelihoods = {
            "motion": y_likelihoods_motion_p,
            "residual":y_likelihoods_res_p
        }

        x_rec  = [x_rec_b,x_rec_p]
        return x_rec, {"base": base_likelihoods, 
                        "prog": prog_likelihoods,
                        "hype":hype_likelihoods}


        










