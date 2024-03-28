from .model import TCM,SWAtten
from compress.layers.mask_layers import ChannelMask
import torch.nn as nn
from ..utils import conv
import torch 
from compress.ops import ste_round
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)





class ResTCM(TCM):

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
                joiner_policy = "res",
                support_progressive_slices = 2,
                **kwargs):
        super().__init__(config, 
                         head_dim,
                         drop_path_rate,
                         N,
                         M,
                         dim_chunk,
                         max_support_slices,
                         **kwargs)
        
        self.joiner_policy = joiner_policy
        self.support_progressive_slices = support_progressive_slices
        self.division_dimension = division_dimension
        self.mask_policy = mask_policy
        self.lmbda_list = lmbda_list 

        

        self.num_slices =  int(self.M//self.dim_chunk) 

        self.division_channel = division_dimension[0]
        self.dimensions_M = division_dimension

        self.multiple_decoder = multiple_decoder
        self.multiple_encoder = multiple_encoder

        self.num_slices_list = [self.division_channel//self.dim_chunk, (self.M - self.division_channel)//self.dim_chunk ]
        self.num_slice_cumulative_list = [p//self.dim_chunk for p in self.dimensions_M]

        self.ns0 = self.num_slice_cumulative_list[0] 
        self.ns1 = self.num_slice_cumulative_list[1] 
  

        self.scalable_levels = len(self.lmbda_list)
        self.masking = ChannelMask(self.mask_policy, 
                                   self.scalable_levels,
                                   self.dim_chunk,num_levels = self.num_slices_list[1] )
        
        if self.multiple_decoder:
            self.g_s = nn.ModuleList(
                nn.Sequential(*[ResidualBlockUpsample(self.dimensions_M[0], 2*self.N, 2)] \
                                + self.m_up1 \
                                + self.m_up2 \
                                + self.m_up3) for _ in range(2))


        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.division_dimension[0] + self.dim_chunk*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, self.dim_chunk, stride=1, kernel_size=3),
            ) for i in range(self.num_slice_cumulative_list[0])
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.division_dimension[0]  + self.dim_chunk*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, self.dim_chunk, stride=1, kernel_size=3),
            ) for i in range(self.num_slice_cumulative_list[0])
            )
        
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.division_dimension[0] + self.dim_chunk*min(i+1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, self.dim_chunk, stride=1, kernel_size=3),
            ) for i in range(self.num_slice_cumulative_list[0])
        ) 

        self.atten_mean = nn.ModuleList(
            nn.Sequential(
                SWAtten((self.division_dimension[0] + self.dim_chunk*min(i, 5)), 
                        (self.division_dimension[0] + self.dim_chunk*min(i, 5)), 
                        16, 
                        self.window_size,
                        0, 
                        inter_dim=128)
            ) for i in range(self.num_slice_cumulative_list[0])
        )
        self.atten_scale = nn.ModuleList(
            nn.Sequential(
                SWAtten((self.division_dimension[0] + self.dim_chunk*min(i, 5)),
                         (self.division_dimension[0] + self.dim_chunk*min(i, 5)),
                           16, 
                           self.window_size,
                           0, 
                           inter_dim=128)
            ) for i in range(self.num_slice_cumulative_list[0])
        )

        estremo_indice = self.support_progressive_slices + 1
        delta_dim = self.division_dimension[1] - self.division_dimension[0]

        self.cc_mean_transforms_prog = nn.ModuleList(
            nn.Sequential(
                conv(delta_dim + self.dim_chunk*min(i + 1, estremo_indice), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, self.dim_chunk, stride=1, kernel_size=3),
            ) for i in range(self.num_slice_cumulative_list[1] - self.num_slice_cumulative_list[0])
        )


        self.cc_scale_transforms_prog = nn.ModuleList(
            nn.Sequential(
                conv(delta_dim + self.dim_chunk*min(i + 1, estremo_indice), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, self.dim_chunk, stride=1, kernel_size=3),
            ) for i in range((self.num_slice_cumulative_list[1] - self.num_slice_cumulative_list[0]))
            )

        self.lrp_transforms_prog = nn.ModuleList(
            nn.Sequential(
                conv(delta_dim + self.dim_chunk* min(i+2, estremo_indice + 1), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, self.dim_chunk, stride=1, kernel_size=3),
            ) for i in range(self.num_slice_cumulative_list[0])
        ) 

        self.atten_mean_prog = nn.ModuleList(
            nn.Sequential(
                SWAtten((delta_dim + self.dim_chunk*min(i + 1, estremo_indice)), 
                        (delta_dim + self.dim_chunk*min(i + 1, estremo_indice)), 
                        16, 
                        self.window_size,
                        0, 
                        inter_dim=128)
            ) for i in range(self.num_slice_cumulative_list[1] - self.num_slice_cumulative_list[0])
        )


        self.atten_scale_prog = nn.ModuleList(
            nn.Sequential(
                SWAtten((delta_dim + self.dim_chunk*min(i + 1, estremo_indice)),
                         (delta_dim + self.dim_chunk*min(i + 1, estremo_indice)),
                           16, 
                           self.window_size,
                           0, 
                           inter_dim=128)
            ) for i in range(self.num_slice_cumulative_list[1] - self.num_slice_cumulative_list[0])
        )

        if self.joiner_policy == "channel_cond":
            self.joiner = nn.ModuleList(
                nn.Sequential(
                    conv(self.dim_chunk*2, 64, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(64, 64, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(64, 32, stride=1, kernel_size=3),
                ) for i in range(self.num_slice_cumulative_list[0])
            )  
        

    def define_quality(self,quality):
        if quality is None:
            list_quality = self.quality_list
        elif isinstance(quality,list):
            list_quality = quality 
        else:
            list_quality = [0, quality] 
        return list_quality


    def determine_support(self,y_hat_base,current_index,y_hat_quality):
        bi = y_hat_base[current_index]
        if current_index == 0 or self.support_progressive_slices == 0:
            return [bi]
        sup_ind = min(self.support_progressive_slices,current_index)
        psi_cum = y_hat_quality[current_index - sup_ind:current_index]
        return [bi] + psi_cum

    def merge(self,y_base,y_enhanced,slice_index):
        if self.joiner_policy == "res":
            return y_base + y_enhanced 
        elif self.joiner_policy == "conc":
            return torch.cat([y_base,y_enhanced],dim = 1)
        else: 
            c = torch.cat([y_base,y_enhanced],dim = 1)
            return self.joiner[slice_index](c)



    def forward(self, x, quality = None, mask_pol = None, training = True):

        if mask_pol is None:
            mask_pol = self.mask_policy
        list_quality = self.define_quality(quality)  
    
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

        y_hat_slices_base = []
        y_likelihood_base= []
        y_likelihood_enhanced = []
        x_hat_progressive = [] 

        for slice_index in range(self.ns0):
            y_slice = y_slices[slice_index]
            idx = slice_index%self.ns0
            indice = min(self.max_support_slices,idx)
            support_slices = (y_hat_slices_base if self.max_support_slices < 0 \
                                                        else y_hat_slices_base[:indice])
        
            scale_support = torch.cat([latent_scales[:,:self.dimensions_M[0]]] + support_slices, dim=1) 

            mean_support = torch.cat([latent_means[:,:self.dimensions_M[0]]] + support_slices, dim=1)
            mean_support = self.atten_mean[idx](mean_support)
            mu = self.cc_mean_transforms[idx](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales[:,:self.dimensions_M[0]]] + support_slices, dim=1)
            scale_support = self.atten_scale[idx](scale_support)
            scale = self.cc_scale_transforms[idx](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]


            _, y_slice_likelihood = self.gaussian_conditional(y_slice,
                                                            scale, 
                                                            mu, 
                                                            training = training)
            y_hat_slice = ste_round(y_slice - mu) + mu


            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.lrp_transforms[idx](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp           

            y_hat_slices_base.append(y_hat_slice)
            y_likelihood_enhanced.append(y_slice_likelihood)
            y_likelihood_base.append(y_slice_likelihood) 

        y_hat_b = torch.cat(y_hat_slices_base,dim = 1)
        x_hat_base = self.g_s[0](y_hat_b) if self.multiple_decoder else self.g_s(y_hat_b)
        x_hat_progressive.append(x_hat_base.unsqueeze(0))
        y_likelihoods_b = torch.cat(y_likelihood_base, dim=1)

        y_likelihood_total = []

        y_hat_total = []
        y_hat_total.append(y_hat_b)

        for _,q in enumerate(list_quality[1:]):
            y_likelihood_quality = []
            y_likelihood_quality = y_likelihood_quality +  y_likelihood_base
            y_hat_slices_quality = [] 

            for slice_index in range(self.ns0,self.ns1):
                y_slice = y_slices[slice_index]
                current_index = slice_index%self.ns0       

                support_slices = self.determine_support(y_hat_slices_base,
                                                         current_index,
                                                         y_hat_slices_quality) #dddd
                

                mean_support = torch.cat([latent_means[:,self.dimensions_M[0]:]] + support_slices, dim=1)
                mean_support = self.atten_mean_prog[current_index](mean_support)
                mu = self.cc_mean_transforms_prog[current_index](mean_support)
                mu = mu[:, :, :y_shape[0], :y_shape[1]]   
                
                scale_support = torch.cat([latent_scales[:,self.dimensions_M[0]:]] + support_slices, dim=1) 
                scale_support = self.atten_scale_prog[current_index](scale_support)
                scale = self.cc_scale_transforms_prog[current_index](scale_support)
                scale = scale[:, :, :y_shape[0], :y_shape[1]]

                block_mask = self.masking(scale,
                                        slice_index = current_index,
                                        pr = q, 
                                        mask_pol = mask_pol) 
                block_mask = self.masking.apply_noise(block_mask, 
                                                      training if "learnable" in mask_pol else False)


                y_slice_m = y_slice  - mu
                y_slice_m = y_slice_m*block_mask

                _, y_slice_likelihood = self.gaussian_conditional(y_slice_m, 
                                                                  scale*block_mask, 
                                                                  training = training)
                y_hat_slice = ste_round(y_slice - mu)*block_mask + mu

                lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
                lrp = self.lrp_transforms_prog[current_index](lrp_support)
                lrp = 0.5 * torch.tanh(lrp)
                y_hat_slice += lrp   

                # faccio il merge qua!!!!!
                y_hat_slice = self.merge(y_hat_slice,
                                         y_hat_slices_base[current_index],
                                         current_index)
 
                y_hat_slices_quality.append(y_hat_slice)
                y_likelihood_quality.append(y_slice_likelihood)


            y_hat_enhanced = torch.cat(y_hat_slices_quality,dim = 1) 
            if self.multiple_decoder:
                x_hat_current = self.g_s[1](y_hat_enhanced)
            else: 
                x_hat_current = self.g_s(y_hat_enhanced)

            y_likelihood_single_quality = torch.cat(y_likelihood_quality,dim = 1)
            y_likelihood_total.append(y_likelihood_single_quality.unsqueeze(0))
            x_hat_progressive.append(x_hat_current.unsqueeze(0)) #1,2,256,256

            y_hat_total.append(y_hat_enhanced)
        
        if len(y_likelihood_total)==0:
            y_likelihood_total = torch.ones_like(y_likelihoods_b).to(y_likelihoods_b.device)
        else:
            y_likelihood_total = torch.cat(y_likelihood_total,dim = 0)  #sliirrr
        x_hats = torch.cat(x_hat_progressive,dim = 0)
        
    
        return {
            "x_hat": x_hats,
            "likelihoods": {"y": y_likelihoods_b,"y_prog":y_likelihood_total,"z": z_likelihoods},
            "z_hat":z_hat,
            "y_hat":y_hat_total
        }
    


    def compress(self, x,quality = 0.0, mask_pol = None):


        mask_pol = self.mask_policy if mask_pol is None else mask_pol

        if self.multiple_encoder is False:
            y = self.g_a(x)
        else:
            y_base = self.g_a(x)
            y_enh = self.g_a_enh(x)
            y = torch.cat([y_base,y_enh],dim = 1).to(x.device)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)


        y_hat_slices = []


        y_slices = y.chunk(self.num_slices, 1) # total amount of slices

        y_strings = []
        masks = []


        for slice_index in range(self.ns0):
            y_slice = y_slices[slice_index]
            indice = min(self.max_support_slices,slice_index%self.ns0)
            support_slices = (y_hat_slices if self.max_support_slices < 0 \
                                                        else y_hat_slices[:indice]) 
            

            idx = slice_index%self.ns0
            mean_support = torch.cat([latent_means[:,:self.dimensions_M[0]]] + support_slices, dim=1)
            mean_support = self.atten_mean[idx](mean_support)
            mu = self.cc_mean_transforms[idx](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales[:,:self.dimensions_M[0]]] + support_slices, dim=1)
            scale_support = self.atten_scale[idx](scale_support)
            scale = self.cc_scale_transforms[idx](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_string  = self.gaussian_conditional.compress(y_slice, index,mu)

            y_hat_slice = self.gaussian_conditional.decompress(y_q_string, index)
            y_hat_slice = y_hat_slice + mu

           

            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.lrp_transforms[idx](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp


            y_strings.append(y_q_string)
            y_hat_slices.append(y_hat_slice)

        if quality == 0:
            return {"strings": [y_strings, z_strings],"shape":z.size()[-2:], "masks":masks}
        
        y_hat_slices_quality = []
        #y_hat_slices_quality = y_hat_slices + []

        for slice_index in range(self.ns0,self.ns1):

            y_slice = y_slices[slice_index]
            current_index = slice_index%self.ns0


            support_slices = self.determine_support(y_hat_slices,
                                                    current_index,
                                                    y_hat_slices_quality) 
                

            mean_support = torch.cat([latent_means[:,self.dimensions_M[0]:]] + support_slices, dim=1)
            mean_support = self.atten_mean_prog[current_index](mean_support)
            mu = self.cc_mean_transforms_prog[current_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]   
                
            scale_support = torch.cat([latent_scales[:,self.dimensions_M[0]:]] + support_slices, dim=1) 
            scale_support = self.atten_scale_prog[current_index](scale_support)
            scale = self.cc_scale_transforms_prog[current_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]


            block_mask = self.masking(scale,
                                    slice_index = current_index, 
                                    pr = quality,
                                    mask_pol = mask_pol)
            masks.append(block_mask)
            block_mask = self.masking.apply_noise(block_mask, False)
            index = self.gaussian_conditional.build_indexes(scale*block_mask).int()

            y_q_string  = self.gaussian_conditional.compress((y_slice - mu)*block_mask, index)
            y_strings.append(y_q_string)
            
            y_hat_slice = self.gaussian_conditional.decompress(y_q_string, index)
            y_hat_slice = y_hat_slice + mu

            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.lrp_transforms_prog[current_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp   

            y_hat_slice = self.merge(y_hat_slice,y_hat_slices[current_index],current_index)

            y_hat_slices_quality.append(y_hat_slice)

        return {"strings": [y_strings, z_strings],"shape":z.size()[-2:],"masks":masks}
    


    def decompress(self, strings, shape, quality, mask_pol = None, masks = None):
        mask_pol = self.mask_policy if mask_pol is None else mask_pol

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_string = strings[0]
        y_hat_slices = []

        for slice_index in range(self.num_slice_cumulative_list[0]):
            pr_strings = y_string[slice_index]
            idx = slice_index%self.num_slice_cumulative_list[0]
            indice = min(self.max_support_slices,idx)
            support_slices = (y_hat_slices if self.max_support_slices < 0 \
                              else y_hat_slices[:indice]) 
            
            mean_support = torch.cat([latent_means[:,:self.division_dimension[0]]] + support_slices,\
                                      dim=1)
            mean_support = self.atten_mean[idx](mean_support)
            mu = self.cc_mean_transforms[idx](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales[:,:self.dimensions_M[0]]] + support_slices,\
                                       dim=1)
            scale_support = self.atten_scale[idx](scale_support)
            scale = self.cc_scale_transforms[idx](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = self.gaussian_conditional.decompress(pr_strings, index )
            rv = rv.reshape(mu.shape)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu) 

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[idx](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        if quality == 0:
            y_hat_b = torch.cat(y_hat_slices, dim=1)
            x_hat = self.g_s[0](y_hat_b).clamp_(0, 1) if self.multiple_decoder else self.g_s(y_hat_b).clamp_(0, 1)
            return {"x_hat": x_hat}
        
        y_hat_slices_quality = []
        for slice_index in range(self.ns0,self.ns1):
            pr_strings = y_string[slice_index]
            current_index = slice_index%self.ns0
            support_slices = self.determine_support(y_hat_slices,
                                                    current_index,
                                                    y_hat_slices_quality)      
            

            mean_support = torch.cat([latent_means[:,self.dimensions_M[0]:]] + support_slices,\
                                      dim=1)
            mean_support = self.atten_mean_prog[current_index](mean_support)
            mu = self.cc_mean_transforms_prog[current_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]   
                
            scale_support = torch.cat([latent_scales[:,self.dimensions_M[0]:]] + support_slices,\
                                       dim=1) 
            scale_support = self.atten_scale_prog[current_index](scale_support)
            scale = self.cc_scale_transforms_prog[current_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            if mask_pol != "random":
                block_mask = self.masking(scale,
                                          slice_index = current_index, 
                                          pr = quality, 
                                          mask_pol = mask_pol)
                block_mask = self.masking.apply_noise(block_mask, False)
            else: 
                assert masks is not None 
                block_mask = masks[current_index]

            index = self.gaussian_conditional.build_indexes(scale*block_mask)
            rv = self.gaussian_conditional.decompress(pr_strings, index)
            rv = rv.reshape(mu.shape)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms_prog[current_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slice = self.merge(y_hat_slice,y_hat_slices[current_index],current_index)
            y_hat_slices_quality.append(y_hat_slice)

        y_hat_en = torch.cat(y_hat_slices_quality,dim = 1)
        if self.multiple_decoder:
            x_hat = self.g_s[1](y_hat_en).clamp_(0, 1)
        else:
            x_hat = self.g_s(y_hat_en).clamp_(0, 1) 
        return {"x_hat": x_hat}  
    

    def forward_single_quality(self,x, quality, mask_pol = None, training = False):

        mask_pol = self.mask_policy if mask_pol is None else mask_pol

        if self.multiple_encoder is False:
            y = self.g_a(x)
        else:
            y_base = self.g_a(x)
            y_enh = self.g_a_enh(x)
            y = torch.cat([y_base,y_enh],dim = 1).to(x.device) #dddd
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

        for slice_index in range(self.num_slice_cumulative_list[0]):
            y_slice = y_slices[slice_index]
            idx = slice_index%self.ns0
            indice = min(self.max_support_slices,idx)
            support_slices = (y_hat_slices if self.max_support_slices < 0 \
                                                        else y_hat_slices[:indice])


        
            scale_support = torch.cat([latent_scales[:,:self.dimensions_M[0]]] + support_slices, dim=1) 


            mean_support = torch.cat([latent_means[:,:self.dimensions_M[0]]] + support_slices, dim=1)
            mean_support = self.atten_mean[idx](mean_support)
            mu = self.cc_mean_transforms[idx](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales[:,:self.dimensions_M[0]]] + support_slices, dim=1)
            scale_support = self.atten_scale[idx](scale_support)
            scale = self.cc_scale_transforms[idx](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]


            _, y_slice_likelihood = self.gaussian_conditional(y_slice,
                                                            scale, 
                                                            mu, 
                                                            training = training)
            y_hat_slice = ste_round(y_slice - mu) + mu


            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.lrp_transforms[idx](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_likelihood.append(y_slice_likelihood)

        if quality == 0: #and  slice_index == self.num_slice_cumulative_list[0] - 1:
            y_hat = torch.cat(y_hat_slices,dim = 1)
            x_hat = self.g_s[0](y_hat).clamp_(0, 1) if self.multiple_decoder else self.g_s(y_hat).clamp_(0, 1)
            y_likelihoods = torch.cat(y_likelihood, dim=1)
            return {
                "x_hat": x_hat,
                "likelihoods": {"y": y_likelihoods,"z": z_likelihoods},
                "z_hat":z_hat
            }             

        y_hat_slices_quality = []

        y_likelihood_quality = []
        y_likelihood_quality = y_likelihood + []
    
        for slice_index in range(self.ns0,self.ns1):

            y_slice = y_slices[slice_index]
            current_index = slice_index%self.ns0
            support_slices = self.determine_support(y_hat_slices,
                                                    current_index,
                                                    y_hat_slices_quality) 
            
            mean_support = torch.cat([latent_means[:,self.dimensions_M[0]:]] + support_slices, dim=1)
            mean_support = self.atten_mean_prog[current_index](mean_support)
            mu = self.cc_mean_transforms_prog[current_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]   
                
            scale_support = torch.cat([latent_scales[:,self.dimensions_M[0]:]] + support_slices, dim=1) 
            scale_support = self.atten_scale_prog[current_index](scale_support)
            scale = self.cc_scale_transforms_prog[current_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]


            block_mask = self.masking(scale,
                                    slice_index = current_index,
                                    pr = quality,
                                     mask_pol = mask_pol) #scale, slice_index = 0,  pr = 0, mask_pol = None
            block_mask = self.masking.apply_noise(block_mask, False)

            y_slice_m = y_slice  - mu
            y_slice_m = y_slice_m*block_mask

            _, y_slice_likelihood = self.gaussian_conditional(y_slice_m, 
                                                                scale*block_mask, 
                                                                training = training)
            y_hat_slice = ste_round(y_slice - mu)*block_mask + mu

            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.lrp_transforms_prog[current_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp   

            # faccio il merge qua!!!!!
            y_hat_slice = self.merge(y_hat_slice,y_hat_slices[current_index],current_index)   #ddd

            y_hat_slices_quality.append(y_hat_slice)    

        y_likelihoods = torch.cat(y_likelihood_quality,dim = 1)
        y_hat = torch.cat(y_hat_slices_quality,dim = 1)  
    
        if self.multiple_decoder:
            x_hat = self.g_s[1](y_hat).clamp_(0, 1)
        else:
            #y_hat_t = self.g_s_prog(y_hat)
            x_hat = self.g_s(y_hat).clamp_(0, 1) 


        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods,"z": z_likelihoods},
            "z_hat":z_hat
        }     

