from .single_decoder import scalable_icd
import torch 
import math 
import torch.nn as nn
from compress.ops import ste_round
from compressai.ans import  RansDecoder
from ..utils import conv, deconv, update_registered_buffers
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class conditional_scalable_icd(scalable_icd):
    def __init__(self, 
                 N=192,
                M=320,
                mask_policy = "learnable-mask",
                lambda_list = [0.05],
                lrp_prog = True,
                independent_lrp = False,
                joiner_policy = "conditional",
                **kwargs):
        super().__init__(N = N, 
                         M = M,
                        mask_policy = mask_policy,
                        lambda_list = lambda_list,
                        lrp_prog = lrp_prog,
                        independent_lrp = independent_lrp,
                        **kwargs)
        

        self.joiner_policy = joiner_policy
        
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
        

    def print_information(self):
        print(" g_a: ",sum(p.numel() for p in self.g_a.parameters()))
        print(" g_a_progressive: ",sum(p.numel() for p in self.g_a_progressive.parameters()))

        print(" h_a: ",sum(p.numel() for p in self.h_a.parameters()))
        print(" h_a_prog: ",sum(p.numel() for p in self.h_a_prog.parameters()))

        print(" h_means_s: ",sum(p.numel() for p in self.h_mean_s.parameters()))
        print(" h_scale_s: ",sum(p.numel() for p in self.h_scale_s.parameters()))
        print(" h_means_s_prog: ",sum(p.numel() for p in self.h_mean_s_prog.parameters()))
        print(" h_scale_s_prog: ",sum(p.numel() for p in self.h_scale_s_prog.parameters()))

        print("cc_mean_transforms",sum(p.numel() for p in self.cc_mean_transforms.parameters()))
        print("cc_scale_transforms",sum(p.numel() for p in self.cc_scale_transforms.parameters()))
        print("cc_mean_transforms_prog",sum(p.numel() for p in self.cc_mean_transforms_prog.parameters()))
        print("cc_scale_transforms_prog",sum(p.numel() for p in self.cc_scale_transforms_prog.parameters()))

        print("lrp_transform",sum(p.numel() for p in self.lrp_transforms.parameters()))

        print("entropy_bottleneck",sum(p.numel() for p in self.entropy_bottleneck.parameters() if p.requires_grad == True))
        print("entropy_bottleneck PROG",sum(p.numel() for p in self.entropy_bottleneck_prog.parameters() if p.requires_grad == True))

        if "learnable-mask" in self.mask_policy:
            print("mask conv",sum(p.numel() for p in self.masking.mask_conv.parameters()))
            #print("gamma",sum(p.numel() for p in self.masking.gamma.parameters()))
        
        if self.independent_lrp:
             print("lrp_transform_prog",sum(p.numel() for p in self.lrp_transforms_prog.parameters()))

        
        print("joiner",sum(p.numel() for p in self.joiner.parameters()))
        print("**************************************************************************")
        model_tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_fr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad== False)
        print(" trainable parameters: ",model_tr_parameters)
        print(" freeze parameterss: ", model_fr_parameters)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    


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



    def forward(self, x, quality =None):


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

        list_quality = self.define_quality(quality)        


        y_slices = y.chunk(self.num_slices, 1)

        y_likelihoods_progressive = []
        y_likelihood_main = []

        x_hat_progressive = []

        y_hats = []
        
        for j,p in enumerate(list_quality): 

            if p in list(self.lmbda_index_list.keys()): #se passo direttamente la lambda!S
                quality = self.lmbda_index_list[p]
            else: 
                quality = p
            
            mask = self.masking(latent_scales,pr = quality)
            if "learnable-mask" in self.mask_policy: # and self.lmbda_index_list[p]!=0 and self.lmbda_index_list[p]!=len(self.lmbda_list) -1:
                mask = self.masking.apply_noise(mask,self.gaussian_conditional.training)
                        
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
                if  quality != 0:
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

                    #_, y_slice_likelihood_prog = self.gaussian_conditional_prog(y_prog_slice, scale_prog,mu_prog, mask = block_mask)
                    _, y_slice_likelihood_prog = self.gaussian_conditional_prog(y_prog_slice, scale_prog,mu_prog)

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

                    #y_hat_complete_slice = y_hat_slice + y_hat_prog_slice
                    y_hat_complete_slice = self.merge(y_hat_slice,y_hat_prog_slice,slice_index)
                    y_hat_complete.append(y_hat_complete_slice)
                else:
                    y_hat_complete.append(y_hat_slice)

            y_hat_q = torch.cat(y_hat_complete,dim = 1) #questo va preso 
            x_hat_q = self.g_s(y_hat_q) 

            y_hats.append(y_hat_q.unsqueeze(0))
            x_hat_progressive.append(x_hat_q.unsqueeze(0)) 

            if  quality != 0:
                y_likelihood_progressive = torch.cat(y_likelihood_prog,dim = 1)
                y_likelihoods_progressive.append(y_likelihood_progressive.unsqueeze(0))


        x_hat_progressive = torch.cat(x_hat_progressive,dim = 0) #num_scalable-1, BS,3,W,H 
        y_likelihoods = torch.cat(y_likelihood_main, dim = 0).unsqueeze(0) # 1,BS,3,W,H solo per base 

        
        if len(y_likelihoods_progressive) == 0:
            y_likelihoods_prog = torch.ones_like(y_likelihoods).to(y_likelihoods.device)
        else:
            y_likelihoods_prog = torch.cat(y_likelihoods_progressive,dim = 0)

        y_hat = torch.cat(y_hats,dim = 0)

        return {
            "x_hat": x_hat_progressive,

            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods,"z_prog":z_likelihoods_prog,"y_prog":y_likelihoods_prog},
            "y": y_hat, "z_hat_prog":z_hat_prog ,"z_hat":z_hat
        }

    def decompress(self, strings, shape, quality):

        if quality in list(self.lmbda_index_list.keys()):
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


            mask = self.masking(latent_scales,pr = quality)
            mask = torch.round(mask)
            mask_slices = mask.chunk(self.num_slices,dim = 1)

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

                support_slices_prog = (y_hat_prog if self.max_support_slices < 0 else y_hat_prog[:self.max_support_slices])
                
                block_mask = mask_slices[slice_index]
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
                rv_prog = self.gaussian_conditional_prog.decompress(pr_strings, index_prog, means= mu_prog) # decoder_prog.decode_stream(index_prog.reshape(-1).tolist(), cdf_prog, cdf_lengths_prog, offsets_prog)
                
                y_hat_slice_prog = rv_prog.reshape(mu_prog.shape).to(mu_prog.device)
            
                if self.lrp_prog:
                    lrp_support = torch.cat([mean_support_prog, y_hat_slice_prog], dim=1)
                    if self.independent_lrp:
                        lrp = self.lrp_transforms_prog[slice_index](lrp_support)
                    else:
                        lrp = self.lrp_transforms[slice_index](lrp_support)
                    lrp = 0.5 * torch.tanh(lrp)
                    y_hat_slice_prog += lrp

                y_hat_prog.append(y_hat_slice_prog)
                #y_hat_complete_slice = y_hat_slice_prog + y_hat_slice
                y_hat_complete_slice = self.merge(y_hat_slice, y_hat_slice_prog,slice_index)
                y_hat_complete.append(y_hat_complete_slice)
            else:
                y_hat_complete.append(y_hat_slice)


        y_hat = torch.cat(y_hat_complete, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}