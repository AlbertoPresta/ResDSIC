import math

from typing import List

import torch
import torch.nn as nn
from compressai.ops import quantize_ste
from compressai.registry import register_model
from .google import ScaleSpaceFlow
from .modules import Encoder, Decoder, HyperpriorMasked, Hyperprior
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

        motion_input = "base",
        scalable_levels = 2,
        double_dim = False
    ):
        super().__init__(num_levels=num_levels,
                         sigma0=sigma0,
                         scale_field_shift=scale_field_shift)
        
        assert motion_input in ("UMSR","MMSR","SMSR")
        self.motion_input = motion_input
        self.mask_policy = mask_policy 
        self.multiple_res_encoder = multiple_res_encoder
        self.multiple_res_decoder = multiple_res_decoder
        #self.multiple_motion_encoder = multiple_motion_encoder 
        #self.multiple_motion_decoder = multiple_motion_decoder
        self.scalable_levels = scalable_levels 
        self.double_dim = double_dim
        self.quality_list = [0,1]

        self.res_encoder = nn.ModuleList(Encoder(3) for _ in range(2)) \
                            if self.multiple_res_encoder \
                            else Encoder(3,factor = 2)
        self.res_decoder = nn.ModuleList(Decoder(3,in_planes=384) for _ in range(2))\
                            if self.multiple_res_decoder \
                            else Decoder(3,factor=2,in_planes=384)
        

        
        #self.motion_encoder = nn.ModuleList(Encoder(2*3) for _ in range(2))    \
        #                        if self.multiple_motion_encoder \
        #                            else Encoder(2*3,factor = 2)
        
        #self.motion_decoder = nn.ModuleList( Decoder(3) for _ in range(2))\
        #                         if self.multiple_motion_decoder   \
        #                              else Decoder(3, factor = 2)
        

        if self.motion_input == "SMSR": 

            self.motion_encoder = nn.ModuleList(Encoder(2*3) for _ in range(2))   
            self.motion_decoder = nn.ModuleList( Decoder(3) for _ in range(2))

            self.motion_hyperprior = nn.ModuleList([Hyperprior(),
                                                    HyperpriorMasked(factor = 1, 
                                                                    mask_policy=mask_policy,
                                                                    scalable_levels=scalable_levels)])
                
                

                
        self.res_hyperprior = nn.ModuleList([Hyperprior(),
                                            HyperpriorMasked(factor = 1, 
                                                            mask_policy=mask_policy,
                                                            scalable_levels=scalable_levels)])
        

        #self.forward_prediction = nn.ModuleList( Decoder(3) for _ in range(2))\
        #                         if self.multiple_motion_decoder   \
        #                              else Decoder(3, factor = 2)
        

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
        if self.motion_input == "UMSR":
            x_u = torch.cat((x_cur, x_ref[0]), dim=1) # cat the input
            y_motion_u = self.motion_encoder(x_u) 
            y_motion_hat_u, motion_likelihoods = self.motion_hyperprior(y_motion_u)
            # decode the space-scale flow information
            motion_info_u = self.motion_decoder(y_motion_hat_u) # scale space flow Decoder 
            x_pred_u = self.forward_prediction(x_ref[0], motion_info_u) # scale space warping

            x_res_b = x_cur - x_pred_u
            x_res_p = x_cur - x_pred_u 


            y_res_b = self.res_encoder[0](x_res_b) if self.multiple_res_encoder \
                                                    else self.res_encoder(x_res_b) 
            y_res_p = self.res_encoder[1](x_res_p) if self.multiple_res_encoder \
                                                    else  self.res_encoder(x_res_p) 
        


            y_res_hat_b, res_likelihoods_b = self.res_hyperprior[0](y_res_b)
            y_res_hat_p, res_likelihoods_p = self.res_hyperprior[1](y_res_p,
                                                                    quality = quality[1], 
                                                                    mask_pol = mask_pol, 
                                                                    training = training) # y_res_hat = v_hat^i


            y_res_hat_p = y_res_hat_b + y_res_hat_p #residual scalable
            y_combine_b = torch.cat((y_res_hat_b,y_motion_hat_u), dim=1) #w_i + v_i
            y_combine_p = torch.cat((y_res_hat_p,y_motion_hat_u), dim=1) #w_i + v_i

            x_res_hat_b = self.res_decoder[0](y_combine_b) if self.multiple_res_decoder  \
                                                        else self.res_decoder(y_combine_b) 
            # final reconstruction: prediction + residual
            x_rec_b = x_pred_u + x_res_hat_b # final reconstruction

            x_res_hat_p = self.res_decoder[1](y_combine_p) if self.multiple_res_decoder  \
                                                        else self.res_decoder(y_combine_p) 
            
            x_rec_p = x_pred_u + x_res_hat_p # final reconstruction

            x_rec = [x_rec_b,x_rec_p]

            base_likelihoods = {"motion": motion_likelihoods,
                                "residual":   res_likelihoods_b,
                                   
                            }
                                
            prog_likelihoods = {"motion":motion_likelihoods,
                                "residual": res_likelihoods_p
                            }
            return x_rec,{"base": base_likelihoods, "prog": prog_likelihoods}
        
        elif self.motion_input == "MMSR":
            
            x_b = torch.cat((x_cur, x_ref[0]), dim=1) # cat the input 

            y_motion_b = self.motion_encoder(x_b) 
            y_motion_hat_b, motion_likelihoods_b = self.motion_hyperprior(y_motion_b)
            # decode the space-scale flow information
            motion_info_b = self.motion_decoder(y_motion_hat_b) # scale space flow Decoder 
            x_pred_b = self.forward_prediction(x_ref[0], motion_info_b) # scale space warping

            x_p = torch.cat((x_cur, x_ref[1]), dim=1) # cat the input 
            y_motion_p = self.motion_encoder(x_p) 
            y_motion_hat_p, motion_likelihoods_p = self.motion_hyperprior(y_motion_p)
            # decode the space-scale flow information
            motion_info_p = self.motion_decoder(y_motion_hat_p) # scale space flow Decoder 
            x_pred_p = self.forward_prediction(x_ref[1], motion_info_p) # scale space warping

            x_res_b = x_cur - x_pred_b
            x_res_p = x_cur - x_pred_p 


            y_res_b = self.res_encoder[0](x_res_b) if self.multiple_res_encoder \
                                                    else self.res_encoder(x_res_b) 
            y_res_p = self.res_encoder[1](x_res_p) if self.multiple_res_encoder \
                                                    else  self.res_encoder(x_res_p) 
        


            y_res_hat_b, res_likelihoods_b = self.res_hyperprior[0](y_res_b)
            y_res_hat_p, res_likelihoods_p = self.res_hyperprior[1](y_res_p,
                                                                    quality = quality[1], 
                                                                    mask_pol = mask_pol, 
                                                                    training = training) # y_res_hat = v_hat^i


            y_res_hat_p = y_res_hat_p + y_res_hat_b # residual scalability!!!

            y_combine_b = torch.cat((y_res_hat_b,y_motion_hat_u), dim=1) #w_i + v_i
            y_combine_p = torch.cat((y_res_hat_p,y_motion_hat_u), dim=1) #w_i + v_i

            
            x_res_hat_b = self.res_decoder[0](y_combine_b) if self.multiple_res_decoder  \
                                                        else self.res_decoder(y_combine_b) 
            # final reconstruction: prediction + residual
            x_rec_b = x_pred_b + x_res_hat_b # final reconstruction

            x_res_hat_p = self.res_decoder[1](y_combine_p) if self.multiple_res_decoder  \
                                                        else self.res_decoder(y_combine_p) 
            
            x_rec_p = x_pred_p + x_res_hat_p # final reconstruction

            x_rec = [x_rec_b,x_rec_p]

            base_likelihoods = {"motion": motion_likelihoods_b,
                                "residual":   res_likelihoods_b,     
                            }
                                

            prog_likelihoods = {"motion":motion_likelihoods_p,
                                "residual": res_likelihoods_p
                            }
            return x_rec,{"base": base_likelihoods, "prog": prog_likelihoods}
        else:
            x_b = torch.cat((x_cur, x_ref[0]), dim=1) # cat the input 

            y_motion_b = self.motion_encoder[0](x_b) 
            y_motion_hat_b, motion_likelihoods_b = self.motion_hyperprior[0](y_motion_b) 
            # decode the space-scale flow information
            motion_info_b = self.motion_decoder[0](y_motion_hat_b) # scale space flow Decoder 
            x_pred_b = self.forward_prediction(x_ref[0], motion_info_b) # scale space warping

            x_p = torch.cat((x_cur, x_ref[1]), dim=1) # cat the input 
            y_motion_p = self.motion_encoder[1](x_p) 
            y_motion_hat_p, motion_likelihoods_p = self.motion_hyperprior[1](y_motion_p,
                                                                            quality = quality[1], 
                                                                            mask_pol = mask_pol, 
                                                                            training = training) 
            
            
            y_motion_hat_p = y_motion_hat_p + y_motion_hat_b # motion scalable
            
            # decode the space-scale flow information
            motion_info_p = self.motion_decoder[1](y_motion_hat_p) # scale space flow Decoder 
            x_pred_p = self.forward_prediction(x_ref[1], motion_info_p) # scale space warping

            x_res_b = x_cur - x_pred_b
            x_res_p = x_cur - x_pred_p 

            y_res_b = self.res_encoder[0](x_res_b) if self.multiple_res_encoder \
                                                    else self.res_encoder(x_res_b) 
            y_res_p = self.res_encoder[1](x_res_p) if self.multiple_res_encoder \
                                                    else  self.res_encoder(x_res_p) 
        


            y_res_p = y_res_p + y_res_b #residul scalable


            y_res_hat_b, res_likelihoods_b = self.res_hyperprior[0](y_res_b, 
                                                                    quality = quality[0], 
                                                                    mask_pol = mask_pol, 
                                                                    training = training)
            y_res_hat_p, res_likelihoods_p = self.res_hyperprior[1](y_res_p,
                                                                    quality = quality[1], 
                                                                    mask_pol = mask_pol, 
                                                                    training = training) 


            y_combine_b = torch.cat((y_res_hat_b,y_motion_hat_u), dim=1) #w_i + v_i
            y_combine_p = torch.cat((y_res_hat_p,y_motion_hat_u), dim=1) #w_i + v_i

            
            x_res_hat_b = self.res_decoder[0](y_combine_b) if self.multiple_res_decoder  \
                                                        else self.res_decoder(y_combine_b) 
            # final reconstruction: prediction + residual
            x_rec_b = x_pred_b + x_res_hat_b # final reconstruction

            x_res_hat_p = self.res_decoder[1](y_combine_p) if self.multiple_res_decoder  \
                                                        else self.res_decoder(y_combine_p) 
            
            x_rec_p = x_pred_p + x_res_hat_p # final reconstruction

            x_rec = [x_rec_b,x_rec_p]

            base_likelihoods = {"motion": motion_likelihoods_b,
                                "residual":   res_likelihoods_b,
                                   
                            }
                                

            prog_likelihoods = {"motion":motion_likelihoods_p,
                                "residual": res_likelihoods_p
                            }
            return x_rec,{"base": base_likelihoods, "prog": prog_likelihoods}


    def encode_inter(self, x_cur, x_ref, quality, mask_pol):

        if self.motion_input in ("UMSR","MMSR"):
            x_u = torch.cat((x_cur, x_ref), dim=1) 
            y_motion_u = self.motion_encoder(x_u) 
            y_motion_hat, out_motion = self.motion_hyperprior.compress(y_motion_u)
            motion_info = self.motion_decoder(y_motion_hat)
            x_pred_u = self.forward_prediction(x_ref, motion_info) # scale space warping

            x_res = x_cur - x_pred_u

            # all'inizio è quello base!!

            y_res = self.res_encoder[0](x_res) if self.multiple_res_encoder else self.res_encoder(x_res) 
            y_res_hat, out_res = self.res_hyperprior[0].compress(y_res,quality,mask_pol)
            y_combine = torch.cat((y_res_hat, y_motion_hat), dim=1)
            x_res_hat = self.res_decoder[0](y_combine) if self.multiple_res_encoder else self.res_decoder(y_combine)

            # final reconstruction: prediction + residual
            x_rec_b = x_pred_u + x_res_hat


            if  quality == 0:
                return x_rec_b, {
                    "strings": {
                        "motion": out_motion["strings"],
                        "residual": out_res["strings"],
                    },
                    "shape": {"motion": out_motion["shape"], "residual": out_res["shape"]},
                }
            
            y_res = self.res_encoder[1](x_res) if self.multiple_res_encoder else self.res_encoder(x_res) 
            y_res_hat_p, out_res = self.res_hyperprior[1].compress(y_res,quality,mask_pol)

            y_res_hat_p = y_res_hat_p + y_res_hat #residual adding!!!!!

            y_combine = torch.cat((y_res_hat_p, y_motion_hat), dim=1)
            x_res_hat = self.res_decoder[1](y_combine) if self.multiple_res_encoder \
                                                            else self.res_decoder(y_combine)
            

            x_rec_p = x_pred_u + x_res_hat


            return x_rec_p if self.motion_input == "MMSR" else [x_rec_b,x_rec_p], {
                    "strings": {
                        "motion": out_motion["strings"],
                        "residual": out_res["strings"],
                    },
                    "shape": {"motion": out_motion["shape"], "residual": out_res["shape"]},
                }

        else: 
            raise NotImplementedError("Questo metodo deve essere implementato nella sottoclasse.")
            


    def decode_inter(self, x_ref, strings, shapes, quality, mask_pol):
        raise NotImplementedError("NOT yet implemented.")


    def compress(self, frames,quality, mask_pol):
        raise NotImplementedError("NOT yet implemented.")


    def decompress(self, strings, shapes, quality, mask_pol):
        raise NotImplementedError("NOT yet implemented.")