
from .model import TCM,SWAtten
from compress.layers.mask_layers import ChannelMask
import torch.nn as nn
from ..utils import conv
from compressai.entropy_models import  GaussianConditional
import torch 
from compress.ops import ste_round
from compressai.layers import (
    ResidualBlockUpsample,
    ResidualBlockWithStride,

)



class ProgTCM(TCM):

    def __init__(self, 
                 config=[2, 2, 2, 2, 2, 2],
                head_dim=[8, 16, 32, 32, 16, 8], 
                drop_path_rate=0, 
                N=128, 
                M=64,
                max_support_slices=5, 
                division_dimension = 320,
                dim_chunk = 32,
                mask_policy = "random",
                lmbda_list = [0.0025,0.05],
                multiple_decoder = True,
                multiple_encoder = False,
                inner_dimensions = [192,192],
                **kwargs):
        super().__init__(config, 
                         head_dim,
                         drop_path_rate,
                         N,
                         M,
                         dim_chunk,
                         max_support_slices,
                         **kwargs)
        
        self.lmbda_list = lmbda_list
        self.multiple_encoder = multiple_encoder
        self.inner_dimensions = inner_dimensions
        self.N = N 
        self.M = M 

        self.mask_policy = mask_policy
        self.dim_chunk = dim_chunk

        self.dim_chunk =  dim_chunk
        self.num_slices =  int(M//self.dim_chunk) 

        self.division_channel = division_dimension[0]
        self.dimensions_M = division_dimension

        self.multiple_decoder = multiple_decoder

        self.num_slices_list = [self.division_channel//self.dim_chunk, (self.M - self.division_channel)//self.dim_chunk ]
        self.num_slice_cumulative_list = [p//self.dim_chunk for p in self.dimensions_M]

        print(self.num_slice_cumulative_list," cumulative!!!!!")




        self.scalable_levels = len(self.lmbda_list)
        self.masking = ChannelMask(self.mask_policy, self.scalable_levels,self.dim_chunk,num_levels =self.num_slices_list[1] )


        self.gaussian_conditional = GaussianConditional(None)


        if self.multiple_decoder:
            self.g_s = nn.ModuleList(
                nn.Sequential(*[ResidualBlockUpsample(self.dimensions_M[0], 2*self.N, 2)] \
                                + self.m_up1 \
                                + self.m_up2 \
                                + self.m_up3) for _ in range(2))
        
        if self.multiple_encoder:
            self.g_a = nn.ModuleList(
            nn.Sequential(*[ResidualBlockWithStride(3, 2*N, 2)]  \
                        + self.m_down1 \
                        + self.m_down2 \
                        + self.m_down3)
            for i in range(2))
        


    def forward(self, x,quality = None, mask_pol = None, training = True):
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        y_hat_slices_base = []
        y_hat_slices_enhanced = []

        y_likelihood_base= []
        y_likelihood_enhanced = []


        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            if quality is None:
                _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu, training = training)
                y_hat_slice = ste_round(y_slice - mu) + mu
            else:
                current_index =slice_index%self.num_slice_cumulative_list[0]
                block_mask = self.masking(scale,slice_index = current_index, pr = quality, mask_pol = mask_pol)
                #block_mask = self.masking.apply_noise(block_mask,tr = training if "learnable" in self.mask_policy else False)
                scale = scale*block_mask

                y_slice_m = y_slice  - mu
                y_slice_m = y_slice_m*block_mask

                _, y_slice_likelihood = self.gaussian_conditional(y_slice_m, scale*block_mask, training = training)
                y_hat_slice = ste_round(y_slice - mu)*block_mask + mu
    


            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices_enhanced.append(y_hat_slice)
            y_likelihood_enhanced.append(y_slice_likelihood)

            if slice_index < self.num_slice_cumulative_list[0]:
                y_hat_slices_base.append(y_hat_slice)
                y_likelihood_base.append(y_slice_likelihood)  


        y_likelihoods_base = torch.cat(y_likelihood_base, dim=1)
        y_likelihoods_enhanced = torch.cat(y_likelihood_enhanced, dim=1)


        y_hat_base = torch.cat(y_hat_slices_base,dim = 1)
        y_hat_enhanced = torch.cat(y_hat_slices_enhanced,dim = 1) 

        if self.multiple_decoder:
            x_hat_base = self.g_s[0](y_hat_base)
            x_hat_enhanced = self.g_s[1](y_hat_enhanced)
        else:
            x_hat_base = self.g_s(y_hat_base)

            y_hat_enhanced = self.g_s_prog(y_hat_enhanced)
            x_hat_enhanced = self.g_s(y_hat_enhanced)

        #x_hat = torch.cat([x_hat_base, x_hat_enhanced],dim = 0) # [n_scalable_levels,...]
        x_hat_progressive = torch.cat([x_hat_base.unsqueeze(0), x_hat_enhanced.unsqueeze(0)],dim = 0)


        return {
            "x_hat": x_hat_progressive,
            "likelihoods": {"y": y_likelihoods_base,"y_prog":y_likelihoods_enhanced,"z": z_likelihoods},
            "z_hat":z_hat
        }


    def compress(self, x, quality = 0.0, mask_pol = None):

        if mask_pol is None:
            mask_pol = self.mask_policy

        if self.multiple_encoder is False:
            y = self.g_a(x)
        else:
            y_base = self.g_a[0](x)
            y_enh = self.g_a[1](x)
            y = torch.cat([y_base,y_enh],dim = 1).to(x.device)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        y_hat_slices = []


        y_slices = y.chunk(self.num_slices, 1) # total amount of slices
        masks = []
        y_strings = []


        for slice_index,y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            if slice_index < self.num_slice_cumulative_list[0] or quality == 0:
                index = self.gaussian_conditional.build_indexes(scale)
                y_q_string  = self.gaussian_conditional.compress(y_slice, index,mu)
            else:
                current_index =slice_index%self.num_slice_cumulative_list[0]
                block_mask = self.masking(scale,slice_index = current_index, pr = quality, mask_pol = mask_pol)
                index = self.gaussian_conditional.build_indexes(scale*block_mask).int()

                y_q_string  = self.gaussian_conditional.compress((y_slice - mu)*block_mask, index)

            y_hat_slice = self.gaussian_conditional.decompress(y_q_string, index)
            y_hat_slice = y_hat_slice + mu

            y_strings.append(y_q_string)

            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_strings.append(y_q_string)


            if quality == 0 and  slice_index == self.num_slice_cumulative_list[0] - 1:
                break

        return {"strings": [y_strings, z_strings],
                "shape":z.size()[-2:],
                "masks": masks
                }


    def decompress(self, strings, shape, quality, mask_pol = None, masks = None):

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_string = strings[0]
        y_hat_slices = []

        num_slices = self.num_slice_cumulative_list[0 if quality == 0 else 1]

        for slice_index in range(num_slices):

            pr_strings = y_string[slice_index]

            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            if slice_index <self.num_slice_cumulative_list[0] or quality == 0:
                index = self.gaussian_conditional.build_indexes(scale)
            else:
                current_index =slice_index%self.num_slice_cumulative_list[0]
                block_mask = self.masking(scale,slice_index = current_index, pr = quality, mask_pol = mask_pol)
                index = self.gaussian_conditional.build_indexes(scale*block_mask)
                   
            rv = self.gaussian_conditional.decompress(pr_strings, index)
            rv = rv.reshape(mu.shape)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)

        if self.multiple_decoder:
            x_hat = self.g_s[0 if quality == 0 else 1](y_hat).clamp_(0, 1)
        else:
            if quality == 0:
                x_hat = self.g_s(y_hat).clamp_(0, 1)
            else: 
                y_hat = self.g_s_prog(y_hat)   
                x_hat = self.g_s(y_hat).clamp_(0, 1) 

        return {"x_hat": x_hat} 
    



    def forward_single_quality(self,x, quality, mask_pol = None, training = False):

        if mask_pol is None:
            mask_pol = self.mask_policy


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
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
        
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]


            if slice_index < self.num_slice_cumulative_list[0] or quality == 0 or quality == 1:
                _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu, training = training)
                y_hat_slice = ste_round(y_slice - mu) + mu
            else:
                current_index = slice_index%self.num_slice_cumulative_list[0]
                block_mask = self.masking(scale,slice_index = current_index, pr = quality, mask_pol = mask_pol)
                #block_mask = self.masking.apply_noise(block_mask, False)
                scale = scale*block_mask

                y_prog_slice_m = y_slice  - mu
                y_prog_slice_m = y_prog_slice_m*block_mask

                _, y_slice_likelihood = self.gaussian_conditional(y_prog_slice_m, scale*block_mask, training = training)
                y_hat_slice = ste_round(y_slice - mu)*block_mask + mu

            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp               

            y_hat_slices.append(y_hat_slice)
            y_likelihood.append(y_slice_likelihood)


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

