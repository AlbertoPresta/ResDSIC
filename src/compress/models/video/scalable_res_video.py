import math

from typing import List

import torch
import torch.nn as nn
from compressai.ops import quantize_ste
from compressai.registry import register_model
from .google import ScaleSpaceFlow
from .modules import Encoder, Decoder, HyperpriorMasked
from compress.layers.mask_layers import Mask

@register_model("ssf2020scalableres")
class ResScalableScaleSpaceFlow(ScaleSpaceFlow):
    def __init__(
        self,
        num_levels: int = 5,
        sigma0: float = 1.5,
        scale_field_shift: float = 1.0,
        mask_policy = "two-levels",
        multiple_res_encoder = True, 
        multiple_res_decoder = True,
        multiple_motion_encoder = True,
        multiple_motion_decoder = True,
        scalable_levels = 2,
        double_dim = False
    ):
        super().__init__(num_levels=num_levels,
                         sigma0=sigma0,
                         scale_field_shift=scale_field_shift)
        

        self.mask_policy = mask_policy 
        self.multiple_res_encoder = multiple_res_encoder
        self.multiple_res_decoder = multiple_res_decoder
        self.multiple_motion_encoder = multiple_motion_encoder 
        self.multiple_motion_decoder = multiple_motion_decoder
        self.scalable_levels = scalable_levels 
        self.double_dim = double_dim
        self.quality_list = [0,1]

        self.res_encoder = nn.ModuleList(Encoder(3) for _ in range(2)) \
                            if self.multiple_res_encoder \
                            else Encoder(3,factor = 2)
        self.res_decoder = nn.ModuleList(Decoder(3,in_planes=384) for _ in range(2))\
                            if self.multiple_res_decoder \
                            else Decoder(3,factor=2,in_planes=384)
        

        self.motion_encoder = nn.ModuleList(Encoder(2*3) for _ in range(2))    \
                                if self.multiple_motion_encoder \
                                    else Encoder(2*3,factor = 2)
        
        self.motion_decoder = nn.ModuleList( Decoder(3) for _ in range(2))\
                                 if self.multiple_motion_decoder   \
                                      else Decoder(3, factor = 2)
        

        if self.multiple_motion_encoder:
            self.motion_hyperprior = HyperpriorMasked(factor = 2,
                                                       mask_policy=mask_policy,
                                                       scalable_levels=scalable_levels)
                
        self.res_hyperprior = HyperpriorMasked(factor = 2, 
                                               mask_policy=mask_policy,
                                               scalable_levels=scalable_levels)
        

        self.forward_prediction = nn.ModuleList( Decoder(3) for _ in range(2))\
                                 if self.multiple_motion_decoder   \
                                      else Decoder(3, factor = 2)
        

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
        

    def forward(self, frames, quality = None, mask_pol = None, training = True):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")


        if mask_pol is None:
            mask_pol = self.mask_policy
        list_quality = self.define_quality(quality)  

        reconstructions_base = []
        reconstructions_prog = []
        frames_likelihoods_base = []
        frames_likelihoods_prog = []

        x_hat, likelihoods = self.forward_keyframe(frames[0]) # I-encoder! modeled as normal frame compression algorithm (choose a good one)

        
        reconstructions_base.append(x_hat)
        reconstructions_prog.append(x_hat)
        frames_likelihoods_base.append(likelihoods)
        frames_likelihoods_prog.append(likelihoods)
        x_ref_s = x_hat.detach()  # stop gradient flow (cf: google2020 paper)

        x_ref = [x_ref_s,x_ref_s]

        for i in range(1, len(frames)):
            x = frames[i]
            x_ref, likelihoods = self.forward_inter(x, 
                                                    x_ref,
                                                    quality= list_quality,
                                                    mask_pol = mask_pol,
                                                    training = training)

            reconstructions_base.append(x_ref[0]) 
            reconstructions_prog.append(x_ref[1])
            
            frames_likelihoods_base.append(likelihoods["base"])
            frames_likelihoods_prog.append(likelihoods["prog"])


        reconstructions = [reconstructions_base,reconstructions_prog]
        return {
            "x_hat": reconstructions,
            "likelihoods_base": frames_likelihoods_base,
            "likelihoods_prog": frames_likelihoods_prog,
        }
    def forward_inter(self, x_cur, x_ref,quality,mask_pol,training):
        # encode the motion information
        x_b = torch.cat((x_cur, x_ref[0]), dim=1) # cat the input 
        x_p = torch.cat((x_cur, x_ref[1]), dim=1) # cat the input 
        y_motion_b = self.motion_encoder[0](x_b) if self.multiple_motion_encoder \
                                                else self.motion_encoder(x_b) # scale space flow enoder, da aumentare 
        
        
        y_motion_p = self.motion_encoder[1](x_p) if self.multiple_motion_encoder \
                                                else self.motion_encoder(x_p) #
        y_motion = torch.cat([y_motion_b,y_motion_p],dim = 1)

        y_motion_hat, motion_likelihoods = self.forward_block(y_motion,
                                                        y_motion_b,
                                                        y_motion_p,
                                                        self.motion_hyperprior,
                                                        training,
                                                        mask_pol,
                                                        quality)



        y_motion_hat_b,y_motion_hat_p = y_motion_hat[0],y_motion_hat[1]
        z_likelihoods_motion = motion_likelihoods[0]
        motion_likelihoods_b = motion_likelihoods[1]
        motion_likelihoods_p = motion_likelihoods[2]
        #y_motion_hat_b, motion_likelihoods_b = self.motion_hyperprior(y_motion_b) # hyperprior, da lasciare in comune?


        motion_info_b = self.motion_decoder[0](y_motion_hat_b) if self.multiple_motion_decoder \
                                                            else self.motion_decoder(y_motion_hat_b)
        
        motion_info_p = self.motion_decoder[1](y_motion_hat_p) if self.multiple_motion_decoder \
                                                            else self.motion_decoder(y_motion_hat_p)
        
        x_pred_b = self.forward_prediction(x_ref[0], motion_info_b) # scale space warping

        x_pred_p = self.forward_prediction(x_ref[1], motion_info_p) # scale space warping

        # residual
        x_res_b = x_cur - x_pred_b
        x_res_p = x_cur - x_pred_p

        y_res_b = self.res_encoder[0](x_res_b) if self.multiple_res_encoder \
                                                else self.res_encoder(x_res_b) 
        y_res_p = self.res_encoder[1](x_res_p) if self.multiple_res_encoder \
                                                else  self.res_encoder(x_res_p) 
        


        y_res = torch.cat([y_res_b,y_res_p],dim = 1)

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


        y_combine_b = torch.cat((y_hat_res_b,y_motion_hat_b), dim=1) #w_i + v_i
        x_res_hat_b = self.res_decoder[0](y_combine_b) if self.multiple_res_decoder  \
                                                    else self.res_decoder(y_combine_b) # x_res_hat = r^_i
        # final reconstruction: prediction + residual
        x_rec_b = x_pred_b + x_res_hat_b # final reconstruction


        y_combine_p = torch.cat((y_hat_res_p,y_motion_hat_p), dim=1) #w_i + v_i
        x_res_hat_p = self.res_decoder[1](y_combine_p) if self.multiple_res_decoder  \
                                                    else self.res_decoder(y_combine_p) # x_res_hat = r^_i
        
        

        x_rec_p = x_pred_p + x_res_hat_p # final reconstruction

        x_rec = [x_rec_b,x_rec_p]

        base_likelihoods = {"motion": {"y":motion_likelihoods_b,
                                       "z":z_likelihoods_motion},
                            "residual": {
                                "y":y_likelihoods_res_b,
                                "z":z_likelihoods_res
                            }
                        }
                            


        prog_likelihoods = {"motion": {"y":motion_likelihoods_p,
                                       "z":z_likelihoods_motion},
                            "residual": {
                                "y":y_likelihoods_res_p,
                                "z":z_likelihoods_res
                            }
                        }


        return x_rec,{"base": base_likelihoods, 
                        "prog": prog_likelihoods}



    def forward_block(self,y,y_b,y_p,funct,training,mask_pol,quality):
        z_motion = funct.hyper_encoder(y)
        z_hat, z_likelihoods = funct.entropy_bottleneck(z_motion, 
                                                        training = training)        
        

        scales = funct.hyper_decoder_scale(z_hat)
        means = funct.hyper_decoder_mean(z_hat)    
        
        scales_b, scales_p = scales.chunk(2,1) # chunck scaes 
        means_b,means_p = means.chunk(2,1) # chunk means 

        # this is the base       
        _, y_likelihoods_b = funct.gaussian_conditional(y_b, 
                                                        scales_b, 
                                                        means_b)
        
        y_hat_b  = quantize_ste(y_b - means_b) + means_b
            

        mask = funct.masking(scale = scales_p, 
                             scale_prog = scales if self.double_dim else None, 
                                                mask_pol = mask_pol, 
                                                pr = quality[-1]) 
        
        _, y_likelihoods_p = funct.gaussian_conditional((y_p - means_p)*mask,
                                                        scales_p*mask)
        
        y_hat_p = quantize_ste(y_p - means_p)*mask  + means_p 
        y_hat_p = y_hat_b + y_hat_p

        return [y_hat_b,y_hat_p],[z_likelihoods,y_likelihoods_b,y_likelihoods_p]
    



    def compress(self, frames,quality, mask_pol):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")

        frame_strings = []
        shape_infos = []

        x_ref, out_keyframe = self.encode_keyframe(frames[0])

        frame_strings.append(out_keyframe["strings"])
        shape_infos.append(out_keyframe["shape"])

        for i in range(1, len(frames)):
            x = frames[i]
            x_ref, out_interframe = self.encode_inter(x, x_ref, quality, mask_pol)

            frame_strings.append(out_interframe["strings"])
            shape_infos.append(out_interframe["shape"])

        return frame_strings, shape_infos



    def encode_inter(self, x_cur, x_ref, quality, mask_pol):
        # encode the motion information
        x = torch.cat((x_cur, x_ref), dim=1)
        y_motion = self.motion_encoder(x)
        y_motion_hat, out_motion = self.motion_hyperprior.compress(y_motion)

        # decode the space-scale flow information
        motion_info = self.motion_decoder(y_motion_hat)
        x_pred = self.forward_prediction(x_ref, motion_info)

        # residual
        x_res = x_cur - x_pred
        y_res = self.res_encoder(x_res)


        y_res_hat, out_res = self.res_hyperprior.compress(y_res,quality,mask_pol)

        # y_combine
        y_combine = torch.cat((y_res_hat, y_motion_hat), dim=1)
        x_res_hat = self.res_decoder(y_combine)

        # final reconstruction: prediction + residual
        x_rec = x_pred + x_res_hat

        return x_rec, {
            "strings": {
                "motion": out_motion["strings"],
                "residual": out_res["strings"],
            },
            "shape": {"motion": out_motion["shape"], "residual": out_res["shape"]},
        }


    def decode_inter(self, x_ref, strings, shapes, quality, mask_pol):
        key = "motion"
        y_motion_hat = self.motion_hyperprior.decompress(strings[key], shapes[key])

        # decode the space-scale flow information
        motion_info = self.motion_decoder(y_motion_hat)
        x_pred = self.forward_prediction(x_ref, motion_info)

        # residual
        key = "residual"
        y_res_hat = self.res_hyperprior.decompress(strings[key], shapes[key],quality,mask_pol)

        # y_combine
        y_combine = torch.cat((y_res_hat, y_motion_hat), dim=1)
        x_res_hat = self.res_decoder(y_combine)

        # final reconstruction: prediction + residual
        x_rec = x_pred + x_res_hat

        return x_rec
    

    def decompress(self, strings, shapes, quality, mask_pol):
        if not isinstance(strings, List) or not isinstance(shapes, List):
            raise RuntimeError(f"Invalid number of frames: {len(strings)}.")

        assert len(strings) == len(
            shapes
        ), f"Number of information should match {len(strings)} != {len(shapes)}."

        dec_frames = []

        x_ref = self.decode_keyframe(strings[0], shapes[0])
        dec_frames.append(x_ref)

        for i in range(1, len(strings)):
            string = strings[i]
            shape = shapes[i]
            x_ref = self.decode_inter(x_ref, string, shape, quality, mask_pol)
            dec_frames.append(x_ref)

        return dec_frames