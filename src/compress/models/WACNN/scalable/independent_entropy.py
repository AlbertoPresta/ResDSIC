import math
import torch
import torch.nn as nn
from compress.layers import GDN
from compressai.ans import BufferedRansEncoder, RansDecoder
from compress.layers import conv3x3, subpel_conv3x3, Win_noShift_Attention
from ..utils import conv
from compress.ops import ste_round
from compress.layers import conv3x3, subpel_conv3x3

from compress.layers.mask_layer import Mask
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compress.entropy_models import GaussianConditionalMask
from ..utils import conv, deconv, update_registered_buffers
from ..cnn import WACNN # import WACNN
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class ResWACNNIndependentEntropy(WACNN):
    def __init__(self, 
                 N=192,
                M=320,
                mask_policy = "learnable-mask",
                lambda_list = [0.05],
                independent_latent_hyperprior = False,
                independent_blockwise_hyperprior = False,
                independent_lrp = False,
                **kwargs):
        super().__init__(N = N, 
                         M = M,
                          **kwargs)

        assert lambda_list is not None 

        self.N = N 
        self.M = M 
        self.halve = 8
        self.level = 5 if self.halve == 8 else -1 
        self.factor = self.halve**2
        assert self.N%self.factor == 0 
        self.T = int(self.N//self.factor) + 3
        self.mask_policy = mask_policy

         


        self.scalable_levels = len(lambda_list)
        self.lmbda_list = lambda_list
        self.lmbda_index_list = dict(zip(self.lmbda_list, [i  for i in range(len(self.lmbda_list))] ))

        print("*****----> ",self.lmbda_index_list)



        self.masking = Mask(self.mask_policy, self.scalable_levels,self.M )
        self.independent_latent_hyperprior = independent_latent_hyperprior 
        self.independent_blockwise_hyperprior = independent_blockwise_hyperprior
        self.independent_lrp = independent_lrp

        self.g_a_progressive = nn.Sequential(
            conv(self.T, N, kernel_size=5, stride=2), # halve 2 so 128
            GDN(N),
            conv(N, N, kernel_size=5, stride=2), # halve 4 so 64 k**2 = 16  
            GDN(N),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4), # 
            conv(N, N, kernel_size=5, stride=2), #halve 8 so dim is 32  k**2 = 64 
            GDN(N),
            conv(N, M, kernel_size=5, stride=2), # 16 
        )

        if self.independent_latent_hyperprior:

            self.entropy_bottleneck_prog = EntropyBottleneck(self.N)

            self.h_a_prog = nn.Sequential(
                conv3x3(320, 320),
                nn.GELU(),
                conv3x3(320, 288),
                nn.GELU(),
                conv3x3(288, 256, stride=2),
                nn.GELU(),
                conv3x3(256, 224),
                nn.GELU(),
                conv3x3(224, 192, stride=2), #dddd
            )

            self.h_mean_s_prog = nn.Sequential(
                conv3x3(192, 192),
                nn.GELU(),
                subpel_conv3x3(192, 224, 2),
                nn.GELU(),
                conv3x3(224, 256),
                nn.GELU(),
                subpel_conv3x3(256, 288, 2),
                nn.GELU(),
                conv3x3(288, 320),
            )

            self.h_scale_s_prog = nn.Sequential(
                conv3x3(192, 192),
                nn.GELU(),
                subpel_conv3x3(192, 224, 2),
                nn.GELU(),
                conv3x3(224, 256),
                nn.GELU(),
                subpel_conv3x3(256, 288, 2),
                nn.GELU(),
                conv3x3(288, 320),
            )

        if self.independent_blockwise_hyperprior:    
            self.cc_mean_transforms_prog = nn.ModuleList(
                nn.Sequential(
                    conv(320 + 32*min(i, 5), 224, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(224, 176, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(176, 128, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(128, 64, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(64, 32, stride=1, kernel_size=3),
                ) for i in range(10)
            )
            self.cc_scale_transforms_prog = nn.ModuleList(
                nn.Sequential(
                    conv(320 + 32 * min(i, 5), 224, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(224, 176, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(176, 128, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(128, 64, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(64, 32, stride=1, kernel_size=3),
                ) for i in range(10)
                )


        if self.independent_lrp:
            self.lrp_transforms_prog = nn.ModuleList(
                nn.Sequential(
                    conv(320 + 32 * min(i+1, 6), 224, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(224, 176, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(176, 128, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(128, 64, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(64, 32, stride=1, kernel_size=3),
                ) for i in range(10)
            )


        self.g_s = nn.ModuleList(
                nn.Sequential(
                Win_noShift_Attention(dim= self.M, num_heads=8, window_size=4, shift_size=2),
                deconv(self.M, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                deconv(N, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
                deconv(N, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                deconv(N, 3, kernel_size=5, stride=2),
            ) for i in range(2) 
        )


        

    
        self.entropy_bottleneck = EntropyBottleneck(self.N) #utilizzo lo stesso modello, ma non lo stesso entropy bottleneck
        if self.independent_blockwise_hyperprior:
            self.gaussian_conditional_prog = GaussianConditional(None)




    def load_state_dict(self, state_dict, strict = False):
        if self.independent_blockwise_hyperprior:
            update_registered_buffers(
                self.gaussian_conditional_prog,
                "gaussian_conditional_prog",
                ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                state_dict,
            )

        if self.independent_latent_hyperprior:
            update_registered_buffers(
                self.entropy_bottleneck_prog,
                "entropy_bottleneck_prog",
                ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                state_dict,
            )       
        
        super().load_state_dict(state_dict, strict = strict)

    def concatenate(self, y_base, x):
        bs,c,w,h = y_base.shape 
        y_base = y_base.reshape(bs,c//self.factor, w*self.halve, h*self.halve).to(x.device)
        res = torch.cat([y_base,x],dim = 1).to(x.device)
        return res 


    def split_ga(self, x, begin = True):
        if begin:
            layers_intermedi = list(self.g_a.children())[:self.level + 1]
        else:
            layers_intermedi = list(self.g_a.children())[self.level + 1:]
        modello_intermedio = nn.Sequential(*layers_intermedi)
        return modello_intermedio(x)


    def print_information(self):
        print(" g_a: ",sum(p.numel() for p in self.g_a.parameters()))
        print(" g_a_progressive: ",sum(p.numel() for p in self.g_a_progressive.parameters()))

        print(" h_a: ",sum(p.numel() for p in self.h_a.parameters()))
        

        print(" h_means_s: ",sum(p.numel() for p in self.h_mean_s.parameters()))
        print(" h_scale_s: ",sum(p.numel() for p in self.h_scale_s.parameters()))
        if self.independent_latent_hyperprior:
            print("entropy_bottleneck PROG",sum(p.numel() for p in self.entropy_bottleneck_prog.parameters() if p.requires_grad == True))
            print(" h_a_prog: ",sum(p.numel() for p in self.h_a_prog.parameters()))
            print(" h_means_s_prog: ",sum(p.numel() for p in self.h_mean_s_prog.parameters()))
            print(" h_scale_s_prog: ",sum(p.numel() for p in self.h_scale_s_prog.parameters()))

        print("cc_mean_transforms",sum(p.numel() for p in self.cc_mean_transforms.parameters()))
        print("cc_scale_trangsforms",sum(p.numel() for p in self.cc_scale_transforms.parameters()))
        
        if self.independent_blockwise_hyperprior:
            print("cc_mean_transforms_prog",sum(p.numel() for p in self.cc_mean_transforms_prog.parameters()))
            print("cc_scale_transforms_prog",sum(p.numel() for p in self.cc_scale_transforms_prog.parameters()))

        print("lrp_transform",sum(p.numel() for p in self.lrp_transforms.parameters()))

        print("entropy_bottleneck",sum(p.numel() for p in self.entropy_bottleneck.parameters() if p.requires_grad == True))
        

        if "learnable-mask" in self.mask_policy:
            print("mask conv",sum(p.numel() for p in self.masking.mask_conv.parameters()))
            print("gamma",sum(p.numel() for p in self.masking.gamma.parameters()))
        
        if self.independent_lrp:
             print("lrp_transform_prog",sum(p.numel() for p in self.lrp_transforms_prog.parameters()))

        
        for i in range(2):
            print(" g_s_" + str(i),": ",sum(p.numel() for p in self.g_s[i].parameters()))

        print("**************************************************************************")
        model_tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_fr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad== False)
        print(" trainable parameters: ",model_tr_parameters)
        print(" freeze parameterss: ", model_fr_parameters)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        if self.independent_blockwise_hyperprior:
            updated = self.gaussian_conditional_prog.update_scale_table(scale_table, force=force)
            #updated |= super().update(force=force)

        if self.independent_latent_hyperprior:
            self.entropy_bottleneck_prog.update()
        self.entropy_bottleneck.update()
        return updated

    def extract_mu_and_scale(self,mean_support, scale_support,slice_index,y_shape, prog = False ):

            if prog is False:
                mu = self.cc_mean_transforms[slice_index](mean_support)
                mu = mu[:, :, :y_shape[0], :y_shape[1]]

                scale = self.cc_scale_transforms[slice_index](scale_support)
                scale = scale[:, :, :y_shape[0], :y_shape[1]] 

                return mu, scale 

            else:
                mu = self.cc_mean_transforms_prog[slice_index](mean_support)
                mu = mu[:, :, :y_shape[0], :y_shape[1]]

                scale = self.cc_scale_transforms_prog[slice_index](scale_support)
                scale = scale[:, :, :y_shape[0], :y_shape[1]] 

                return mu, scale 

             

    def hyperEncoderDecoder(self,y, progressive = False):
        if progressive:
            z_prog = self.h_a_prog(y)
            _, z_likelihoods_prog = self.entropy_bottleneck_prog(z_prog)

            z_offset_prog = self.entropy_bottleneck_prog._get_medians()
            z_tmp_prog = z_prog - z_offset_prog
            z_hat_prog = ste_round(z_tmp_prog) + z_offset_prog

            scales_prog = self.h_scale_s_prog(z_hat_prog)
            means_prog = self.h_mean_s_prog(z_hat_prog)
            return z_likelihoods_prog, scales_prog, means_prog
        else:
            z = self.h_a(y)
            _, z_likelihoods = self.entropy_bottleneck(z)

            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = ste_round(z_tmp) + z_offset

            scales = self.h_scale_s(z_hat)
            means = self.h_mean_s(z_hat)
            return z_likelihoods, scales, means

           

    # provo a tenere tutti insieme! poi vediamo 
    def forward(self, x, quality =None,  mask_pol = None):

        if mask_pol is None:
            mask_pol = self.mask_policy

        list_quality = self.define_quality(quality)  

        y_base= self.split_ga(x)
        y = self.split_ga(y_base,begin = False)
        y_shape = y.shape[2:]
        y_progressive_support = self.concatenate(y_base,x)
        y_progressive = self.g_a_progressive(y_progressive_support)


        z_likelihoods, latent_means, latent_scales = self.hyperEncoderDecoder(y)

        z_likelihoods_prog, means_prog, scales_prog = self.hyperEncoderDecoder(y_progressive, 
                                                                               progressive = self.independent_latent_hyperprior )

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

            mask = self.masking(latent_scales,pr = quality, mask_pol = mask_pol)
            if "learnable-mask" in self.mask_policy: # and self.lmbda_index_list[p]!=0 and self.lmbda_index_list[p]!=len(self.lmbda_list) -1:
                mask = self.masking.apply_noise(mask,self.gaussian_conditional.training)

            mask_slices = mask.chunk(self.num_slices,dim = 1)                 
            y_progressive_slices = y_progressive.chunk(self.num_slices,dim = 1)

            
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
                mu, scale = self.extract_mu_and_scale(mean_support, 
                                                      scale_support,
                                                      slice_index,
                                                      y_shape)

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
                                                        prog = self.independent_blockwise_hyperprior 
                                                               )
                    
                    
                                                         
                    if self.independent_blockwise_hyperprior:
                        _, y_slice_likelihood_prog = self.gaussian_conditional_prog(y_prog_slice, scale_prog*block_mask,mu_prog)
                    else:
                        _, y_slice_likelihood_prog = self.gaussian_conditional(y_prog_slice, scale_prog*block_mask,mu_prog)

                    y_likelihood_prog.append(y_slice_likelihood_prog)

                    y_hat_prog_slice = ste_round(y_prog_slice - mu_prog)*block_mask + mu_prog


                    lrp_support = torch.cat([mean_support_prog, y_hat_prog_slice], dim=1)
                    if self.independent_lrp:
                        lrp = self.lrp_transforms_prog[slice_index](lrp_support)
                    else:
                        lrp = self.lrp_transforms[slice_index](lrp_support)
                    lrp = 0.5 * torch.tanh(lrp)
                    y_hat_prog_slice += lrp

                    y_hat_prog.append(y_hat_prog_slice)

                    y_hat_complete_slice = y_hat_slice + y_hat_prog_slice
                    y_hat_complete.append(y_hat_complete_slice)
                else:
                    y_hat_complete.append(y_hat_slice)

            y_hat_q = torch.cat(y_hat_complete,dim = 1) #questo va preso 
            
            x_hat_q = self.g_s[0 if quality == 0 else 1](y_hat_q) 

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
            "y": y_hat
        }



    #/scratch/ResDSIC/models/zero__2_independent_two-levels_320_192_0.0035_0.065_False/0210__lambda__0.075__epoch__91_best_.pth.tar



    def define_quality(self,quality):
        if quality is None:
            list_quality = self.lmbda_list 
        elif isinstance(quality,list):
            list_quality = quality 
        else:
            list_quality = [quality] 
        return list_quality

    def hyperEncoderDecoderCompress(self,y, progressive = False):
        if progressive is False:
            z = self.h_a(y)
            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
            latent_scales = self.h_scale_s(z_hat)
            latent_means = self.h_mean_s(z_hat)

            return z_strings, latent_scales, latent_means,z
        else:
            z_prog = self.h_a_prog(y)
            z_string_prog = self.entropy_bottleneck_prog.compress(z_prog)
            z_hat_prog = self.entropy_bottleneck_prog.decompress(z_string_prog,z_prog.size()[-2:])
            latent_scales_prog = self.h_scale_s_prog(z_hat_prog)
            latent_means_prog = self.h_mean_s_prog(z_hat_prog)
            return z_string_prog, latent_scales_prog, latent_means_prog, z_prog


    def compress(self, x, quality = 0.0, mask_pol = None):


        if mask_pol is None:
            mask_pol = self.mask_policy
        y_base= self.split_ga(x)
        y = self.split_ga(y_base,begin = False)
        y_shape = y.shape[2:]


        y_progressive_support = self.concatenate(y_base,x)
        y_progressive = self.g_a_progressive(y_progressive_support)

        y_shape = y.shape[2:]


        z_strings, latent_scales, latent_means, z = self.hyperEncoderDecoderCompress(y)


        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()

        if self.independent_blockwise_hyperprior:
            cdf_prog = self.gaussian_conditional_prog.quantized_cdf.tolist()
            cdf_lengths_prog = self.gaussian_conditional_prog.cdf_length.reshape(-1).int().tolist()
            offsets_prog = self.gaussian_conditional_prog.offset.reshape(-1).int().tolist()
            encoder_prog = BufferedRansEncoder()
        else:
            cdf_prog = self.gaussian_conditional.quantized_cdf.tolist()
            cdf_lengths_prog = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
            offsets_prog = self.gaussian_conditional.offset.reshape(-1).int().tolist()
            encoder_prog = BufferedRansEncoder()   


        symbols_list = []
        indexes_list = []
        y_strings = []


        symbols_list_prog = []
        indexes_list_prog = []
        y_strings_prog = []


        if quality in list(self.lmbda_index_list.keys()):
            q = self.lmbda_index_list[quality] 
        else:
            q = quality

        if q != 0:

            z_string_prog, latent_scales_prog, latent_means_prog, z_prog = self.hyperEncoderDecoderCompress(y_progressive,
                                                                                                            self.independent_latent_hyperprior)


            mask = self.masking(latent_scales,scale_prog = latent_scales_prog, pr = quality, mask_pol = mask_pol)
            mask = torch.round(mask)
            mask_slices = mask.chunk(self.num_slices,dim = 1)


        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []

        y_progressive_slices = y_progressive.chunk(self.num_slices,dim = 1)
        y_hat_prog = []


        for slice_index, y_slice in enumerate(y_slices):
            #part con la parte main 
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            mu, scale = self.extract_mu_and_scale(mean_support, scale_support,slice_index,y_shape)

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu
       
            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

            # now progressive
            if q != 0:

                block_mask = mask_slices[slice_index]
                support_slices_prog = (y_hat_prog if self.max_support_slices < 0 else y_hat_prog[:self.max_support_slices])
                y_slice_prog = y_progressive_slices[slice_index]

                #[latent_means] + support_slices
                mean_support_prog = torch.cat([latent_means_prog] + support_slices_prog, dim=1)
                scale_support_prog = torch.cat([latent_scales_prog] + support_slices_prog, dim=1)
                mu_prog, scale_prog = self.extract_mu_and_scale(mean_support_prog, 
                                                                scale_support_prog,
                                                                slice_index,
                                                                y_shape,
                                                                prog = self.independent_blockwise_hyperprior)

                if self.independent_blockwise_hyperprior:
                    index_prog = self.gaussian_conditional_prog.build_indexes(scale_prog*block_mask) #saràda aggiungere la maschera
                else:
                    index_prog = self.gaussian_conditional.build_indexes(scale_prog*block_mask)
                index_prog = index_prog.int()

                y_q_slice_prog = y_slice_prog - mu_prog 
                y_q_slice_prog = y_q_slice_prog*block_mask

                if self.independent_blockwise_hyperprior:
                    y_q_slice_prog = self.gaussian_conditional_prog.quantize(y_q_slice_prog, "symbols")
                else:
                    y_q_slice_prog = self.gaussian_conditional.quantize(y_q_slice_prog, "symbols")
                y_hat_slice_prog = y_q_slice_prog + mu_prog

                symbols_list_prog.extend(y_q_slice_prog.reshape(-1).tolist())
                indexes_list_prog.extend(index_prog.reshape(-1).tolist())
        


                lrp_support = torch.cat([mean_support_prog,y_hat_slice_prog], dim=1)
                if self.independent_lrp:
                    lrp = self.lrp_transforms_prog[slice_index](lrp_support)
                else:
                    lrp = self.lrp_transforms[slice_index](lrp_support) ##dddd
                lrp = 0.5 * torch.tanh(lrp)
                y_hat_slice_prog += lrp

                y_hat_prog.append(y_hat_slice_prog)


        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)

        if q == 0:
            return {"strings": [y_strings, z_strings], "shape": [z.size()[-2:]]}
    
        
        encoder_prog.encode_with_indexes(symbols_list_prog, indexes_list_prog, cdf_prog, cdf_lengths_prog, offsets_prog)
        y_string_prog = encoder_prog.flush()
        y_strings_prog.append(y_string_prog)
        #print("finito encoding")
        
        return {"strings": [y_strings, z_strings, z_string_prog,y_strings_prog],  #preogressive_strings
                "shape": [z.size()[-2:],z_prog.size()[-2:]],          
                }
    

    def decompress(self, strings, shape, quality,mask_pol = None ):

        if mask_pol is None: 
            mask_pol = self.mask_policy

        if quality in list(self.lmbda_index_list.keys()):
            q = self.lmbda_index_list[quality] 
        else:
            q = quality

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape[0])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]
        y_string_prog = strings[-1][0]

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)


        if self.independent_blockwise_hyperprior:
            cdf_prog = self.gaussian_conditional_prog.quantized_cdf.tolist()
            cdf_lengths_prog = self.gaussian_conditional_prog.cdf_length.reshape(-1).int().tolist()
            offsets_prog = self.gaussian_conditional_prog.offset.reshape(-1).int().tolist()
        else:
            cdf_prog = self.gaussian_conditional.quantized_cdf.tolist()
            cdf_lengths_prog = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
            offsets_prog = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder_prog = RansDecoder()
        decoder_prog.set_stream(y_string_prog)


        if q != 0:

            
            #strings[-1]  #self.entropy_bottleneck.decompress(strings[-1],shape[-1])
            
            
            if self.independent_latent_hyperprior:
                z_hat_prog =  self.entropy_bottleneck_prog.decompress(strings[2],shape[-1])
                latent_scales_prog = self.h_scale_s_prog(z_hat_prog)
                latent_means_prog = self.h_mean_s_prog(z_hat_prog)
            else:
                z_hat_prog =  self.entropy_bottleneck.decompress(strings[2],shape[-1])
                latent_scales_prog = self.h_scale_s(z_hat_prog)
                latent_means_prog = self.h_mean_s(z_hat_prog)

            mask = self.masking(latent_scales,scale_prog = latent_scales_prog, pr = quality, mask_pol = mask_pol)
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
                                                                 prog =  self.independent_blockwise_hyperprior 
                                                                 )

                
                if self.independent_blockwise_hyperprior:
                    index_prog = self.gaussian_conditional_prog.build_indexes(scale_prog*block_mask)
                else:
                    index_prog = self.gaussian_conditional.build_indexes(scale_prog*block_mask)
                index_prog = index_prog.int()


                rv_prog = decoder_prog.decode_stream(index_prog.reshape(-1).tolist(), cdf_prog, cdf_lengths_prog, offsets_prog)
                rv_prog = torch.Tensor(rv_prog).reshape(mu_prog.shape)
                if self.independent_blockwise_hyperprior:
                    y_hat_slice_prog = self.gaussian_conditional_prog.dequantize(rv_prog, mu_prog)
                else:
                    y_hat_slice_prog = self.gaussian_conditional.dequantize(rv_prog, mu_prog)

                #pr_strings = progressive_strings[slice_index]
                #rv_prog = self.gaussian_conditional_prog.decompress(pr_strings, index_prog, means= mu_prog) # decoder_prog.decode_stream(index_prog.reshape(-1).tolist(), cdf_prog, cdf_lengths_prog, offsets_prog) 
                #y_hat_slice_prog = rv_prog.reshape(mu_prog.shape).to(mu_prog.device)
            
                lrp_support = torch.cat([mean_support_prog, y_hat_slice_prog], dim=1)
                if self.independent_lrp:
                    lrp = self.lrp_transforms_prog[slice_index](lrp_support)
                else:
                    lrp = self.lrp_transforms[slice_index](lrp_support)
                lrp = 0.5 * torch.tanh(lrp)
                y_hat_slice_prog += lrp

                y_hat_prog.append(y_hat_slice_prog)
                y_hat_complete_slice = y_hat_slice_prog + y_hat_slice
                y_hat_complete.append(y_hat_complete_slice)
            else:
                y_hat_complete.append(y_hat_slice)


        y_hat = torch.cat(y_hat_complete, dim=1)
        x_hat = self.g_s[0 if quality==0 else 1](y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}
    





"""
inputs = Input((64, 64, 1))
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

up1 = UpSampling2D(size=(2, 2))(conv3)
up1 = concatenate([up1, conv2], axis=-1)
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

up2 = UpSampling2D(size=(2, 2))(conv4)
up2 = concatenate([up2, conv1], axis=-1)
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

model = Model(inputs=[inputs], outputs=[outputs])




"""