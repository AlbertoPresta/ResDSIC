import math
import torch
import torch.nn as nn

from compressai.ans import BufferedRansEncoder, RansDecoder
from compress.entropy_models import EntropyBottleneck, GaussianConditional #fff
from compress.layers import GDN
from ..utils import conv, deconv, update_registered_buffers
from compress.ops import ste_round
from compress.layers import conv3x3, subpel_conv3x3, Win_noShift_Attention
from .independent_entropy import ResWACNNIndependentEntropy 
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))



class ResWACNNConditionalIndependentEntropy(ResWACNNIndependentEntropy):
    def __init__(self, N=192,
                M=320,
                scalable_levels = 4,
                mask_policy = "learnable-mask",
                lmbda_list = None,
                lrp_prog = True,
                independent_lrp = False,
                multiple_decoder = True,
                joiner_policy = "concatenation",
                **kwargs):
        super().__init__(N = N, 
                         M = M,
                        scalable_levels = scalable_levels,
                        mask_policy = mask_policy,
                        lmbda_list = lmbda_list,
                        lrp_prog = lrp_prog,
                        independent_lrp = independent_lrp,
                        multiple_decoder = multiple_decoder,
                          **kwargs)

        self.joiner_policy = joiner_policy

        if self.multiple_decoder: # and self.joiner_policy in  ("concatenation","block_concatenation"):
            self.dimensions_M = [self.M, self.M*2 if  "concatenation" in self.joiner_policy else self.M]
            self.g_s = nn.ModuleList(
                        nn.Sequential(
                        Win_noShift_Attention(dim= self.dimensions_M[i], num_heads=8, window_size=4, shift_size=2),
                        deconv(self.dimensions_M[i], N, kernel_size=5, stride=2),
                        GDN(N, inverse=True),
                        deconv(N, N, kernel_size=5, stride=2),
                        GDN(N, inverse=True),
                        Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
                        deconv(N, N, kernel_size=5, stride=2),
                        GDN(N, inverse=True),
                        deconv(N, 3, kernel_size=5, stride=2),
                ) for i in range(2) 
            )


        if self.joiner_policy == "conditional":
            self.joiner = nn.ModuleList(
                nn.Sequential(
                    conv(64, 64, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(64, 64, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(64, 32, stride=1, kernel_size=3),
                ) for i in range(10)
            )
        elif self.joiner_policy == "cac":
            self.joiner_policy = nn.Conv2d(in_channels=M,
                                                       out_channels=M,
                                                         kernel_size=1, 
                                                         stride=1, 
                                                         padding=0)


    def merge(self,y_main,y_prog, slice = 0):
        if self.joiner_policy == "residual":
            return y_main + y_prog 
        elif self.joiner_policy == "concatenation" or self.joiner_policy == "cac":
            return y_main #torch.cat([y_main, y_prog], dim=1).to(y_main.device)
        elif self.joiner_policy == "block_concatenation":
            return torch.cat([y_main, y_prog], dim=1).to(y_main.device)
        else:
            y_hat_slice_support = torch.cat([y_main, y_prog], dim=1)
            return self.joiner[slice](y_hat_slice_support)

    # provo a tenere tutti insieme! poi vediamo 
    def forward(self, x, quality = None, mask_pol = None, training = True):

        if mask_pol is None:
            mask_pol = self.mask_policy

        list_quality = self.define_quality(quality) 

        y_base= self.split_ga(x)
        y = self.split_ga(y_base,begin = False)
        y_shape = y.shape[2:]

        y_progressive_support = self.concatenate(y_base,x)
        y_progressive = self.g_a_progressive(y_progressive_support)

        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)


        # calcoliamo mean and std per il progressive
        
        z_prog = self.h_a_prog(y_progressive) 
        _, z_likelihoods_prog = self.entropy_bottleneck_prog(z_prog) # this is different (could have different dix)

        z_offset_prog = self.entropy_bottleneck_prog._get_medians()
        z_tmp_prog = z_prog - z_offset_prog
        z_hat_prog = ste_round(z_tmp_prog) + z_offset_prog

        scales_prog = self.h_scale_s_prog(z_hat_prog)
        means_prog = self.h_mean_s_prog(z_hat_prog)

               


        y_slices = y.chunk(self.num_slices, 1)

        y_likelihoods_progressive = []
        y_likelihood_main = []
        x_hat_progressive = []

        
        for j,p in enumerate(list_quality):



            if p in list(self.lmbda_index_list.keys()): #se passo direttamente la lambda!S
                q = self.lmbda_index_list[p]
            else: 
                q = p 
            mask =  self.masking(latent_scales,scale_prog = scales_prog,pr = q, mask_pol = mask_pol)
            if "learnable-mask" in self.mask_policy: # and self.lmbda_index_list[p]!=0 and self.lmbda_index_list[p]!=len(self.lmbda_list) -1:
                mask = self.masking.apply_noise(mask,training)
                        
            y_progressive_slices = y_progressive.chunk(self.num_slices,dim = 1)
            mask_slices = mask.chunk(self.num_slices,dim = 1)
            y_hat_slices = []
            y_hat_complete = []
            y_hat_prog = []
            y_likelihood_prog = []

            for slice_index, y_slice in enumerate(y_slices):
                ####################################################################################################
                ##############################    PARTO DAL MAIN LATENT REPRESENTATION ##########################
                support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
                
                # encode the main latent representation 
                mean_support = torch.cat([latent_means] + support_slices, dim=1)
                scale_support = torch.cat([latent_scales] + support_slices, dim=1)
                mu, scale = self.extract_mu_and_scale(mean_support, scale_support,slice_index,y_shape)

                _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
                
                
                if j == 0:
                    y_likelihood_main.append(y_slice_likelihood)
                y_hat_slice = ste_round(y_slice - mu) + mu 
                

                lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
                lrp = self.lrp_transforms[slice_index](lrp_support)
                lrp = 0.5 * torch.tanh(lrp)
                y_hat_slice += lrp

                y_hat_slices.append(y_hat_slice)


                #############################################################
                #######################   TOCCA ALLA PARTE PROGRESSIVE ######
                ############################################################
                if  q != 0:
                    y_prog_slice = y_progressive_slices[slice_index]
                    block_mask = mask_slices[slice_index]
                    support_prog_slices = (y_hat_prog if self.max_support_slices < 0 else y_hat_prog[:self.max_support_slices])


                    #[latent_means] + support_slices
                    mean_support_prog = torch.cat([means_prog] + support_prog_slices, dim=1)
                    scale_support_prog = torch.cat([scales_prog] + support_prog_slices, dim=1)
                    mu_prog, scale_prog = self.extract_mu_and_scale(mean_support_prog, 
                                                                    scale_support_prog,
                                                                    slice_index,
                                                                    y_shape,
                                                                   prog = True
                                                                    )

                    y_prog_slice_m = y_prog_slice  - mu_prog 
                    y_prog_slice_m = y_prog_slice_m*block_mask
                    _, y_slice_likelihood_prog = self.gaussian_conditional_prog(y_prog_slice_m, scale_prog*block_mask)

                    y_likelihood_prog.append(y_slice_likelihood_prog)

                    y_hat_prog_slice = ste_round(y_prog_slice - mu_prog)*block_mask + mu_prog

                    if self.lrp_prog:
                        lrp_support = torch.cat([mean_support_prog, y_hat_prog_slice], dim=1)
                        if self.independent_lrp:
                            lrp = self.lrp_transforms_prog[slice_index](lrp_support)
                        else:
                            lrp = self.lrp_transforms[slice_index](lrp_support)
                        lrp = 0.5 * torch.tanh(lrp)
                        y_hat_prog_slice += lrp

                    y_hat_prog.append(y_hat_prog_slice)

                    y_hat_complete_slice = self.merge(y_hat_slice,y_hat_prog_slice,slice_index)
                    y_hat_complete.append(y_hat_complete_slice)
                else:
                    y_hat_complete.append(y_hat_slice)




            if self.joiner_policy == "concatenation" and quality != 0:
                y_hat_q = torch.cat(y_hat_complete + y_hat_prog,dim = 1) #questo va preso 
            else:
                y_hat_q = torch.cat(y_hat_complete,dim = 1) #questo va preso 

            if self.multiple_decoder:
                x_hat_q = self.g_s[0 if q == 0 else 1](y_hat_q)
            else:
                x_hat_q = self.g_s(y_hat_q)



            #y_hats.append(y_hat_q.unsqueeze(0))
            x_hat_progressive.append(x_hat_q.unsqueeze(0)) 

            if  q != 0:
                y_likelihood_progressive = torch.cat(y_likelihood_prog,dim = 1)
                y_likelihoods_progressive.append(y_likelihood_progressive.unsqueeze(0))


        x_hat_progressive = torch.cat(x_hat_progressive,dim = 0) #num_scalable-1, BS,3,W,H 
        y_likelihoods = torch.cat(y_likelihood_main, dim = 0).unsqueeze(0) # 1,BS,3,W,H solo per base 

        
        if len(y_likelihoods_progressive) == 0:
            y_likelihoods_prog = torch.ones_like(y_likelihoods).to(y_likelihoods.device)
        else:
            y_likelihoods_prog = torch.cat(y_likelihoods_progressive,dim = 0)

        if q == 0:
            z_likelihoods_prog = torch.ones_like(z_likelihoods_prog).to(z_likelihoods_prog.device)

        #y_hat = torch.cat(y_hats,dim = 0)

        return {
            "x_hat": x_hat_progressive,

            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods,"z_prog":z_likelihoods_prog,"y_prog":y_likelihoods_prog},
             "z_hat_prog":z_hat_prog ,"z_hat":z_hat
        }       
        

    def decompress(self, strings, shape, quality, mask_pol = None):

        if mask_pol is None:
            mask_pol = self.mask_policy



        if quality in list(self.lmbda_index_list.keys()): #se passo direttamente la lambda!S
            q = self.lmbda_index_list[quality]
        else: 
            q = quality 

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape[0])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)


        if q != 0:

            z_hat_prog =  self.entropy_bottleneck_prog.decompress(strings[2],shape[-1])#strings[-1]  #self.entropy_bottleneck.decompress(strings[-1],shape[-1])
            
            latent_scales_prog = self.h_scale_s_prog(z_hat_prog)
            latent_means_prog = self.h_mean_s_prog(z_hat_prog)

            progressive_strings = strings[-1]
            #y_string_prog = strings[2][0]


            mask =  self.masking(latent_scales,scale_prog = latent_scales_prog,pr = q, mask_pol = mask_pol)
            mask = torch.round(mask)
            mask_slices = mask.chunk(self.num_slices,dim = 1)


            #cdf_prog = self.gaussian_conditional_prog.quantized_cdf.tolist()
            #cdf_lengths_prog = self.gaussian_conditional_prog.cdf_length.reshape(-1).int().tolist()
            #offsets_prog = self.gaussian_conditional_prog.offset.reshape(-1).int().tolist()
            #decoder_prog = RansDecoder()
            #decoder_prog.set_stream(y_string_prog)



        y_hat_slices = []
        y_hat_prog = []
        y_hat_complete = []
        
        
        for slice_index in range(self.num_slices):

           
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            mu, scale =  self.extract_mu_and_scale(mean_support, scale_support,slice_index,y_shape)
            
            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(mu.shape)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)


            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp


            y_hat_slices.append(y_hat_slice)


            if q != 0:
                block_mask = mask_slices[slice_index]
                support_slices_prog = (y_hat_prog if self.max_support_slices < 0 else y_hat_prog[:self.max_support_slices])
                
                
                #[latent_means] + support_slices
                mean_support_prog = torch.cat([latent_means_prog] + support_slices_prog, dim=1)
                scale_support_prog = torch.cat([latent_scales_prog] + support_slices_prog, dim=1)
                mu_prog, scale_prog = self.extract_mu_and_scale(mean_support_prog,
                                                                 scale_support_prog,
                                                                 slice_index,
                                                                 y_shape,
                                                                 prog = True
                                                                 )

                index_prog = self.gaussian_conditional_prog.build_indexes(scale_prog*block_mask)
                index_prog = index_prog.int()


                #rv_prog = decoder_prog.decode_stream(index_prog.reshape(-1).tolist(), cdf_prog, cdf_lengths_prog, offsets_prog)
                #rv_prog = torch.Tensor(rv_prog).reshape(mu_prog.shape)
                #y_hat_slice_prog = self.gaussian_conditional_prog.dequantize(rv_prog, mu_prog)



                pr_strings = progressive_strings[slice_index]
                rv_prog = self.gaussian_conditional_prog.decompress(pr_strings, index_prog) # decoder_prog.decode_stream(index_prog.reshape(-1).tolist(), cdf_prog, cdf_lengths_prog, offsets_prog)
                
                y_hat_slice_prog = rv_prog.reshape(mu_prog.shape).to(mu_prog.device)
                y_hat_slice_prog = y_hat_slice_prog  + mu_prog
            
                if self.lrp_prog:
                    lrp_support = torch.cat([mean_support_prog, y_hat_slice_prog], dim=1)
                    if self.independent_lrp:
                        lrp = self.lrp_transforms_prog[slice_index](lrp_support)
                    else:
                        lrp = self.lrp_transforms[slice_index](lrp_support)
                    lrp = 0.5 * torch.tanh(lrp)
                    y_hat_slice_prog += lrp

                y_hat_prog.append(y_hat_slice_prog)
                y_hat_complete_slice = self.merge(y_hat_slice,y_hat_slice_prog,slice_index)
                y_hat_complete.append(y_hat_complete_slice)
            else:
                y_hat_complete.append(y_hat_slice)


            if self.joiner_policy == "concatenation" and q != 0:
                y_hat = torch.cat(y_hat_complete + y_hat_prog,dim = 1) #questo va preso 
            else:
                y_hat = torch.cat(y_hat_complete,dim = 1) #questo va preso


        if self.multiple_decoder:
            x_hat = self.g_s[0 if q == 0 else 1](y_hat).clamp_(0, 1)
        else:
            x_hat = self.g_s(y_hat).clamp_(0, 1)



        return {"x_hat": x_hat}