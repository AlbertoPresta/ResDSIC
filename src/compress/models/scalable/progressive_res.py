import math
import torch
import torch.nn as nn
from compress.layers.mask_layers import Mask
from compressai.ans import BufferedRansEncoder, RansDecoder
from compress.entropy_models import EntropyBottleneck, GaussianConditional #fff
from compress.layers import GDN
from ..utils import conv, deconv, update_registered_buffers
from compress.ops import ste_round
from compress.layers import conv3x3, subpel_conv3x3, Win_noShift_Attention
from .progressive import ProgressiveWACNN




# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64






def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))



class ProgressiveResWACNN(ProgressiveWACNN):
    def __init__(self, 
                N=192,
                M=416,
                division_dimension = [320,416],
                dim_chunk = 32,
                multiple_decoder = True,
                mask_policy = None,
                lmbda_list = [0.075],
                shared_entropy_estimation = False,
                **kwargs):
        
        super().__init__(N = N, 
                         M = M,
                         dim_chunk = dim_chunk,
                         mask_policy=mask_policy,
                         lmbda_list=lmbda_list,
                         multiple_decoder=multiple_decoder,
                         division_dimension=division_dimension,
                         **kwargs)
        
        
        self.shared_entropy_estimation = shared_entropy_estimation
        
        
        if self.shared_entropy_estimation: 
            self.cc_mean_transforms = nn.ModuleList(
                nn.Sequential(
                    conv(self.M + 32*min(i, 5), 224, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(224, 176, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(176, 128, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(128, 64, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(64, 32, stride=1, kernel_size=3),
                ) for i in range(self.num_slice_cumulative_list[0])
            )
            self.cc_scale_transforms = nn.ModuleList(
                nn.Sequential(
                    conv(self.M + 32 * min(i, 5), 224, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(224, 176, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(176, 128, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(128, 64, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(64, 32, stride=1, kernel_size=3),
                ) for i in range(self.num_slice_cumulative_list[0])
                )
            self.lrp_transforms = nn.ModuleList(
                nn.Sequential(
                    conv(self.M + 32 * min(i+1, 6), 224, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(224, 176, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(176, 128, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(128, 64, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(64, 32, stride=1, kernel_size=3),
                ) for i in range(self.num_slice_cumulative_list[0])
            )           


        if self.multiple_decoder:
    
            self.g_s = nn.ModuleList(
                        nn.Sequential(
                        Win_noShift_Attention(dim= self.dimensions_M[0], num_heads=8, window_size=4, shift_size=2),
                        deconv(self.dimensions_M[0], N, kernel_size=5, stride=2),
                        GDN(N, inverse=True),
                        deconv(N, N, kernel_size=5, stride=2),
                        GDN(N, inverse=True),
                        Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
                        deconv(N, N, kernel_size=5, stride=2),
                        GDN(N, inverse=True),
                        deconv(N, 3, kernel_size=5, stride=2),
                ) for _ in range(2) # per adesso solo due, poi vediamo
            )
        else:

            self.g_s = nn.Sequential(
                Win_noShift_Attention(dim=self.dimensions_M[0], num_heads=8, window_size=4, shift_size=2),
                deconv(self.dimensions_M[0], N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                deconv(N, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
                deconv(N, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                deconv(N, 3, kernel_size=5, stride=2),
                )
            
            
    def print_information(self):
        print(" g_a: ",sum(p.numel() for p in self.g_a.parameters()))
        print(" h_a: ",sum(p.numel() for p in self.h_a.parameters()))
        print(" h_means_a: ",sum(p.numel() for p in self.h_mean_s.parameters()))
        print(" h_scale_a: ",sum(p.numel() for p in self.h_scale_s.parameters()))
        print("cc_mean_transforms",sum(p.numel() for p in self.cc_mean_transforms.parameters()))
        print("cc_scale_transforms",sum(p.numel() for p in self.cc_scale_transforms.parameters()))
        print("lrp_transform",sum(p.numel() for p in self.lrp_transforms.parameters()))
        if self.multiple_decoder:
            for i in range(2):
                print("g_s_" + str(i) + ": ",sum(p.numel() for p in self.g_s[i].parameters()))
        print("**************************************************************************")
        model_tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_fr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad== False)
        print(" trainable parameters: ",model_tr_parameters)
        print(" freeze parameterss: ", model_fr_parameters)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
    def forward(self,x, quality = None, mask_pol = None, training = True):
    


        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)


        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)


        y_slices = y.chunk(self.num_slices, 1) # total amount of slices

        y_hat_slices_base = []
        y_hat_slices_enhanced = []

        y_likelihood_base= []
        y_likelihood_enhanced = []
        y_hat_slices_only_enhanced = []


        for slice_index, y_slice in enumerate(y_slices):
            if self.shared_entropy_estimation is False:
                support_slices = (y_hat_slices_enhanced if self.max_support_slices < 0 \
                                                        else y_hat_slices_enhanced[:self.max_support_slices])
            else:
                indice = min(self.max_support_slices,slice_index%self.num_slice_cumulative_list[0])
                support_slices = (y_hat_slices_enhanced if self.max_support_slices < 0 \
                                                        else y_hat_slices_enhanced[:indice])               
            
            idx = slice_index%self.num_slice_cumulative_list[0]
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)           
            mu = self.cc_mean_transforms[idx](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]              
            scale = self.cc_scale_transforms[idx](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            if quality is None:
                _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu, training = training)
                y_hat_slice = ste_round(y_slice - mu) + mu
            else:
                assert 0 <= quality <= 1
                block_mask = self.extract_mask(scale, pr = quality)
                scale = scale*block_mask

                y_slice_m = y_slice  - mu
                y_slice_m = y_slice_m*block_mask

                _, y_slice_likelihood = self.gaussian_conditional(y_slice_m, scale*block_mask, training = training)
                y_hat_slice = ste_round(y_slice - mu)*block_mask + mu

            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.lrp_transforms[idx](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp               

            y_hat_slices_enhanced.append(y_hat_slice)
            y_likelihood_enhanced.append(y_slice_likelihood)

            if slice_index < self.num_slice_cumulative_list[0]:
                y_hat_slices_base.append(y_hat_slice)
                y_likelihood_base.append(y_slice_likelihood) 
            else:
                y_hat_slices_only_enhanced.append(y_hat_slice)


        y_likelihoods_base = torch.cat(y_likelihood_base, dim=1)
        y_likelihoods_enhanced = torch.cat(y_likelihood_enhanced, dim=1)


        y_hat_base = torch.cat(y_hat_slices_base,dim = 1)
        y_hat_enhanced = y_hat_base + torch.cat(y_hat_slices_only_enhanced,dim = 1) #residual 


        if self.multiple_decoder:
            x_hat_base = self.g_s[0](y_hat_base)
            x_hat_enhanced = self.g_s[1](y_hat_enhanced)
        else:
            x_hat_base = self.g_s(y_hat_base)

            #y_hat_enhanced = self.g_s_prog(y_hat_enhanced)
            x_hat_enhanced = self.g_s(y_hat_enhanced)

        #x_hat = torch.cat([x_hat_base, x_hat_enhanced],dim = 0) # [n_scalable_levels,...]
        x_hat_progressive = torch.cat([x_hat_base.unsqueeze(0), x_hat_enhanced.unsqueeze(0)],dim = 0)


        return {
            "x_hat": x_hat_progressive,
            "likelihoods": {"y": y_likelihoods_base,"y_prog":y_likelihoods_enhanced,"z": z_likelihoods},
            "z_hat":z_hat
        }   
               


    def extract_mask(self,scale,  pr = 0):
        shapes = scale.shape
        bs, ch, w,h = shapes
        assert scale is not None 
        #pr = pr*0.1  
        pr = 1 -pr 
        scale = scale.ravel()
        quantile = torch.quantile(scale, pr)
        res = scale >= quantile 
        #print("dovrebbero essere soli 1: ",torch.unique(res, return_counts = True))
        return res.reshape(bs,ch,w,h).to(torch.float).to(scale.device)
    


    def compress(self, x, quality = 0.0, mask_pol = None):

        if mask_pol is None:
            mask_pol = self.mask_policy

        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        y_hat_slices = []


        y_slices = y.chunk(self.num_slices, 1) # total amount of slices

        y_strings = []


        for slice_index,y_slice in enumerate(y_slices):
            if self.shared_entropy_estimation is False:
                support_slices = (y_hat_slices if self.max_support_slices < 0 \
                                                        else y_hat_slices[:self.max_support_slices])
            else:
                indice = min(self.max_support_slices,slice_index%self.num_slice_cumulative_list[0])
                support_slices = (y_hat_slices if self.max_support_slices < 0 \
                                                        else y_hat_slices[:indice]) 
                
                
            idx =  slice_index%self.num_slice_cumulative_list[0]  
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[idx](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[idx](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]           

            if slice_index < self.num_slice_cumulative_list[0] or quality == 0 or quality == 1:
                index = self.gaussian_conditional.build_indexes(scale)
                #y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
                y_q_string  = self.gaussian_conditional.compress(y_slice, index,mu)
            else:
                block_mask = self.extract_mask(scale, pr = quality)
                
                index = self.gaussian_conditional.build_indexes(scale*block_mask)
                #y_q_slice = self.gaussian_conditional.quantize(y_slice,"symbols",means = mu) 
                index = index.int()
                #y_q_slice = y_q_slice*block_mask
                y_q_string  = self.gaussian_conditional.compress((y_slice - mu)*block_mask, index)

            y_hat_slice = self.gaussian_conditional.decompress(y_q_string, index)
            y_hat_slice = y_hat_slice + mu

            y_strings.append(y_q_string)

            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.lrp_transforms[idx](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp



            y_hat_slices.append(y_hat_slice)
            if quality == 0 and  slice_index == self.num_slice_cumulative_list[0] - 1:
                break

        return {"strings": [y_strings, z_strings],
                "shape":z.size()[-2:]
                }







    def decompress(self, strings, shape, quality, mask_pol = None):


        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)


        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_string = strings[0]
        y_hat_slices = []


        num_slices = self.num_slice_cumulative_list[0 if quality == 0 else 1]


        y_hat_slices_base = []
        y_hat_slices_enh = []
        for slice_index in range(num_slices):
            
            if self.shared_entropy_estimation is False:
                support_slices = (y_hat_slices if self.max_support_slices < 0 \
                                                        else y_hat_slices[:self.max_support_slices])
            else:
                indice = min(self.max_support_slices,slice_index%self.num_slice_cumulative_list[0])
                support_slices = (y_hat_slices if self.max_support_slices < 0 \
                                                        else y_hat_slices[:indice]) 

            idx = slice_index%self.num_slice_cumulative_list[0]
            
            pr_strings = y_string[slice_index]

            
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[idx](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[idx](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            if slice_index <self.num_slice_cumulative_list[0] or quality == 0 or quality == 1 :
                index = self.gaussian_conditional.build_indexes(scale)
            else:
                block_mask = self.extract_mask(scale, pr = quality)
                index = self.gaussian_conditional.build_indexes(scale*block_mask)

                
                
            rv = self.gaussian_conditional.decompress(pr_strings, index)
            rv = rv.reshape(mu.shape)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[idx](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            if quality == 0 or slice_index <self.num_slice_cumulative_list[0]:
                y_hat_slices_base.append(y_hat_slice)
            else:
                y_hat_slices_enh.append(y_hat_slice)
            
            
            y_hat_slices.append(y_hat_slice)


        y_hat = torch.cat(y_hat_slices, dim=1) if quality == 0 else torch.cat(y_hat_slices_base, dim=1) + torch.cat(y_hat_slices_enh, dim=1)



        if self.multiple_decoder:
            x_hat = self.g_s[0 if quality == 0 else 1](y_hat).clamp_(0, 1)
        else:
            x_hat = self.g_s(y_hat).clamp_(0, 1) 
        return {"x_hat": x_hat}   





    def forward_single_quality(self,x, quality, mask_pol = None, training = False):

        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z) #ddd


        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1) # total amount of slices

        y_hat_slices = []
        y_hat_slices_base = []
        y_hat_slices_enh = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            
            if self.shared_entropy_estimation is False:
                support_slices = (y_hat_slices if self.max_support_slices < 0 \
                                                        else y_hat_slices[:self.max_support_slices])
            else:
                indice = min(self.max_support_slices,slice_index%self.num_slice_cumulative_list[0])
                support_slices = (y_hat_slices if self.max_support_slices < 0 \
                                                        else y_hat_slices[:indice])             

            idx = slice_index%self.num_slice_cumulative_list[0]

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)           
            mu = self.cc_mean_transforms[idx](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]              
            scale = self.cc_scale_transforms[idx](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            if slice_index < self.num_slice_cumulative_list[0] or quality == 0 or quality == 1:
                _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu, training = training)
                y_hat_slice = ste_round(y_slice - mu) + mu
            else:
                assert 0 <= quality <= 1
                block_mask = self.extract_mask(scale, pr = quality)
                scale = scale*block_mask

                y_prog_slice_m = y_slice  - mu
                y_prog_slice_m = y_prog_slice_m*block_mask

                _, y_slice_likelihood = self.gaussian_conditional(y_prog_slice_m, scale*block_mask, training = training)
                y_hat_slice = ste_round(y_slice - mu)*block_mask + mu

            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.lrp_transforms[idx](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp               

            y_hat_slices.append(y_hat_slice)
            y_likelihood.append(y_slice_likelihood)
            
            
            if quality == 0 or slice_index < self.num_slice_cumulative_list[0]:
                y_hat_slices_base.append(y_hat_slice)
            else:
                y_hat_slices_enh.append(y_hat_slice)


            if quality == 0 and  slice_index == self.num_slice_cumulative_list[0] - 1:
                break

        y_likelihoods = torch.cat(y_likelihood, dim=1)
        y_hat_enhanced = torch.cat(y_hat_slices,dim = 1) 


        if self.multiple_decoder:
            x_hat = self.g_s[0 if quality==0 else 1](y_hat_enhanced)
        else:
            if quality > 0:
                y_hat_enhanced = self.g_s_prog(y_hat_enhanced)
                x_hat = self.g_s(y_hat_enhanced)
            else: 
                x_hat = self.g_s(y_hat_enhanced)


        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods,"z": z_likelihoods},
            "z_hat":z_hat
        }     

