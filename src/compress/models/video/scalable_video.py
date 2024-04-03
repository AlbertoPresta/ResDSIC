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
from .modules import Encoder, Decoder, Hyperprior
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

        self.mask_policy = mask_policy
        self.scalable_levels = scalable_levels 
        self.masking = Mask(self.mask_policy,scalable_levels = self.scalable_levels)


        if self.scalable_ref:
            self.img_encoder = nn.ModuleList(Encoder(3) for _ in range(2)) if self.multiple_ref_encoder else Encoder(3, factor = 2) 
            self.img_decoder =  nn.ModuleList(Decoder(3) for _ in range(2)) if self.multiple_ref_decoder else Decoder(3,factor = 2)
            self.img_hyperprior = Hyperprior(factor = 2)

        
        self.res_encoder = nn.ModuleList(Encoder(3) for _ in range(2)) if self.multiple_res_encoder else Encoder(3,factor = 2)
        self.res_decoder = nn.ModuleList(Decoder(3,in_planes=384) for _ in range(2)) if self.multiple_res_decoder else Decoder(3,factor=2,in_planes=384)
        self.res_hyperprior = Hyperprior(factor = 2) #facotr 2 perché l'output e doppio 


        self.motion_encoder = nn.ModuleList(Encoder(2*3) for _ in range(2)) if self.multiple_flow_encoder else Encoder(2*3,factor = 2)
        self.motion_decoder = nn.ModuleList( Decoder(3) for _ in range(2)) if self.multiple_flow_decoder else Decoder(3, factor = 2)
        self.motion_hyperprior = Hyperprior(factor = 2) #facotr 2 perché l'output e doppio 


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
        frames_likelihoods_prog = []

        x_hat, likelihoods = self.forward_keyframe(frames[0], quality = 0, mask_pol =mask_pol, training =training) # I-encoder! modeled as normal frame compression algorithm (choose a good one)
        reconstructions_base.append(x_hat)
        frames_likelihoods.append(likelihoods)
        x_ref = x_hat.detach()  # stop gradient flow (cf: google2020 paper)

        if self.scalable_ref:
            x_hat_prog, likelihoods = self.forward_keyframe(frames[0], quality = 0, mask_pol =mask_pol, training =training) # I-encoder! modeled as normal frame compression algorithm (choose a good one)
            reconstructions_prog.append(x_hat_prog)
            frames_likelihoods_prog.append(likelihoods)
            x_ref_prog = x_hat_prog.detach()  # stop gradient flow (cf: google2020 paper)
        else: 
            reconstructions_prog.append(x_hat)
            frames_likelihoods_prog.append(likelihoods)
            x_ref_prog = x_hat.detach()  # stop gradient flow (cf: google2020 paper)           




        for i in range(1, len(frames)):
            x = frames[i]
            x_ref_prog, likelihoods = self.forward_inter(x, x_ref_prog, quality = 1, mask_pol =mask_pol, training =training)
            reconstructions_base.append(x_ref)
            frames_likelihoods_prog.append(likelihoods)

        
        
        
        return {
            "x_hat": reconstructions,
            "likelihoods": frames_likelihoods,
        }



        



        










