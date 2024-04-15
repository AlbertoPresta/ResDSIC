import math
import torch
import torch.nn as nn
from compress.layers import GDN
from ..utils import conv, deconv
from compress.ops import ste_round

from compress.layers.mask_layers import ChannelMask
from .progressive_res import ProgressiveResWACNN

from compress.layers import conv3x3, subpel_conv3x3, Win_noShift_Attention


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64






def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))



class ChannelProgresssiveWACNN(ProgressiveResWACNN):
    def __init__(self, 
                N=192,
                M=640,
                division_dimension = [320,640],
                dim_chunk = 32,
                multiple_decoder = True,
                multiple_encoder = True,
                multiple_hyperprior = False,
                mask_policy = "two-levels",
                lmbda_list = [0.005,0.05],
                shared_entropy_estimation = False,
                joiner_policy = "res",
                support_progressive_slices = 0,
                double_dim = False,
                **kwargs):
        
        super().__init__(N = N, 
                         M = M,
                         dim_chunk = dim_chunk,
                         mask_policy=mask_policy,
                         lmbda_list=lmbda_list,
                         multiple_decoder=multiple_decoder,
                         division_dimension=division_dimension,
                         shared_entropy_estimation=shared_entropy_estimation,
                         joiner_policy=joiner_policy,
                         **kwargs)
        

        assert joiner_policy in ("res","cond","channel_cond","channel_res")
        self.support_progressive_slices = support_progressive_slices
        self.shared_entropy_estimation = shared_entropy_estimation
        self.multiple_encoder = multiple_encoder
        self.division_dimension = division_dimension
        assert self.shared_entropy_estimation is False 
        self.multiple_hyperprior = multiple_hyperprior





        self.double_dim = double_dim
        print("double dimension----->",self.double_dim) #dddd
        self.scalable_levels = len(self.lmbda_list)
        self.masking = ChannelMask(self.mask_policy, 
                                   self.scalable_levels,
                                   self.dim_chunk,
                                   num_levels =self.num_slices_list[1],
                                    double_dim=self.double_dim )
        

        self.quality_list = [i for i in range(self.scalable_levels)]

        self.ns0 = self.num_slice_cumulative_list[0] 
        self.ns1 = self.num_slice_cumulative_list[1] 

        estremo_indice = self.support_progressive_slices + 1
        delta_dim = self.division_dimension[1] - self.division_dimension[0]

        if self.multiple_encoder:
            self.g_a = nn.ModuleList(
                    nn.Sequential(
                    conv(3, N, kernel_size=5, stride=2), # halve 128
                    GDN(N),
                    conv(N, N, kernel_size=5, stride=2), # halve 64
                    GDN(N),
                    Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4), # 
                    conv(N, N, kernel_size=5, stride=2), #32 
                    GDN(N),
                    conv(N, self.dimensions_M[0], kernel_size=5, stride=2), # 16
                    Win_noShift_Attention(dim=self.dimensions_M[0], num_heads=8, window_size=4, shift_size=2),
                ) for _ in range(2)
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
        
        self.cc_mean_transforms = nn.ModuleList(
                nn.Sequential(
                    conv(self.division_dimension[0] + 32*min(i, 5), 224, stride=1, kernel_size=3),
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
                    conv(self.division_dimension[0] + 32 * min(i, 5), 224, stride=1, kernel_size=3),
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
                    conv(self.division_dimension[0] + 32 * min(i+1, 6), 224, stride=1, kernel_size=3),
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

        if self.multiple_hyperprior:

            self.h_mean_s = nn.ModuleList(
                nn.Sequential(
                conv3x3(self.N, 192),
                nn.GELU(),
                subpel_conv3x3(192, 224, 2),
                nn.GELU(),
                conv3x3(224, 256),
                nn.GELU(),
                subpel_conv3x3(256, 288, 2),
                nn.GELU(),
                conv3x3(288, self.dimensions_M[0]),
            ) for i in range(2))

            self.h_scale_s = nn.ModuleList(
                    nn.Sequential(
                    conv3x3(self.N, 192),
                    nn.GELU(),
                    subpel_conv3x3(192, 224, 2),
                    nn.GELU(),
                    conv3x3(224, 256),
                    nn.GELU(),
                    subpel_conv3x3(256, 288, 2),
                    nn.GELU(),
                    conv3x3(288, self.dimensions_M[0]),
                ) for i in range(2))
        

        self.cc_mean_transforms_prog = nn.ModuleList(
                    nn.Sequential(
                        conv(delta_dim + 32*min(i + 1, estremo_indice), 224, stride=1, kernel_size=3),
                        nn.GELU(),
                        conv(224, 176, stride=1, kernel_size=3),
                        nn.GELU(),
                        conv(176, 128, stride=1, kernel_size=3),
                        nn.GELU(),
                        conv(128, 64, stride=1, kernel_size=3),
                        nn.GELU(),
                        conv(64, 32, stride=1, kernel_size=3),
                    ) for i in range(self.num_slice_cumulative_list[1] - self.num_slice_cumulative_list[0] )
                )
        self.cc_scale_transforms_prog = nn.ModuleList(
                    nn.Sequential(
                        conv(delta_dim + 32*min(i + 1, estremo_indice), 224, stride=1, kernel_size=3),
                        nn.GELU(),
                        conv(224, 176, stride=1, kernel_size=3),
                        nn.GELU(),
                        conv(176, 128, stride=1, kernel_size=3),
                        nn.GELU(),
                        conv(128, 64, stride=1, kernel_size=3),
                        nn.GELU(),
                        conv(64, 32, stride=1, kernel_size=3),
                    ) for i in range(self.num_slice_cumulative_list[1]- self.num_slice_cumulative_list[0])
                    )

        self.lrp_transforms_prog = nn.ModuleList(
                nn.Sequential(
                    conv(delta_dim + 32 * min(i+2, estremo_indice + 1), 224, stride=1, kernel_size=3),
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



    def freeze_base_net(self,multiple_hyperprior,freeze_dec):

        for p in self.g_s[0].parameters():
            p.requires_grad = False
        for p in self.g_a[0].parameters():
            p.requires_grad = False 
        for p in self.lrp_transforms.parameters():
            p.requires_grad = False 
        for p in self.cc_mean_transforms.parameters():
            p.requires_grad = False 
        for p in self.cc_scale_transforms.parameters():
            p.requires_grad = False 
        
        if multiple_hyperprior:
            
            for p in self.h_scale_s[0].parameters():
                p.requires_grad = False 
            for p in self.h_mean_s[0].parameters():
                p.requires_grad = False
        
        if freeze_dec:
            for p in self.g_s[1].parameters():
                p.requires_grad = False       
        
            
    def print_information(self):
        if self.multiple_encoder is False:
            print(" g_a: ",sum(p.numel() for p in self.g_a.parameters()))
        else:
            print(" g_a: ",sum(p.numel() for p in self.g_a.parameters()))
           
        print(" h_a: ",sum(p.numel() for p in self.h_a.parameters()))
        print(" h_means_a: ",sum(p.numel() for p in self.h_mean_s.parameters()))
        print(" h_scale_a: ",sum(p.numel() for p in self.h_scale_s.parameters()))
        print("cc_mean_transforms",sum(p.numel() for p in self.cc_mean_transforms.parameters()))
        print("cc_scale_transforms",sum(p.numel() for p in self.cc_scale_transforms.parameters()))


        if self.mask_policy == "single-learnable-mask-quantile" or self.mask_policy == "three-levels-learnable":
            print("mask",sum(p.numel() for p in self.masking.mask_conv.parameters()))
        if "gamma" in self.mask_policy:
            print("mask",sum(p.numel() for p in self.masking.mask_conv.parameters()))
            for i in range(len(self.masking.gamma)):
                print("gamma " + str(i),sum(p.numel() for p in self.masking.gamma[i]))

        if self.shared_entropy_estimation is False: 
            print("cc_mean_transforms_prog",sum(p.numel() for p in self.cc_mean_transforms_prog.parameters()))
            print("cc_scale_transforms_prog",sum(p.numel() for p in self.cc_scale_transforms_prog.parameters()))  

        print("lrp_transform",sum(p.numel() for p in self.lrp_transforms.parameters()))
        if self.multiple_decoder:
            for i in range(2):
                print("g_s_" + str(i) + ": ",sum(p.numel() for p in self.g_s[i].parameters()))
        else: 
            print("g_s",sum(p.numel() for p in self.g_s.parameters()))

        if self.joiner_policy == "cond":
            print("joiner",sum(p.numel() for p in self.joiner.parameters()))  
        print("**************************************************************************")
        model_tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_fr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad== False)
        print(" trainable parameters: ",model_tr_parameters)
        print(" freeze parameterss: ", model_fr_parameters)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    


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


    def only_mask(self):

        for p in self.parameters():
            p.requires_grad = False 
        for _,p in self.named_parameters():
            p.requires_grad = False 
        
        for p in self.masking.mask_conv.parameters():
            p.requires_grad = True
        for n,p in self.masking.mask_conv.named_parameters():
            p.requires_grad = True
        

        if "gamma" in self.mask_policy:
            for i in range(len(self.masking.gamma)): #ddd
                self.masking.gamma[i].requires_grad_(True)                        

    def compute_hyperprior(self,y, quality = 0):

        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        if self.multiple_hyperprior is False or quality == 0:
            latent_scales = self.h_scale_s(z_hat) if self.multiple_hyperprior is False else self.h_scale_s[0](z_hat) 
            latent_means = self.h_mean_s(z_hat) if self.multiple_hyperprior is False else self.h_mean_s[0](z_hat)
            return latent_means, latent_scales, z_likelihoods
        else:
            latent_scales_base = self.h_scale_s[0](z_hat)
            latent_means_base = self.h_mean_s[0](z_hat)

            latent_scales_enh = self.h_scale_s[1](z_hat)
            latent_means_enh = self.h_mean_s[1](z_hat)

            latent_means = torch.cat([latent_means_base,latent_means_enh],dim = 1)
            latent_scales = torch.cat([latent_scales_base,latent_scales_enh],dim = 1)
            return latent_means, latent_scales, z_likelihoods

    
    def forward(self,x, quality = None, mask_pol = None, training = True):
    
        if mask_pol is None:
            mask_pol = self.mask_policy
        list_quality = self.define_quality(quality)  
        if self.multiple_encoder is False:
            y = self.g_a(x)
        else:
            y_base = self.g_a[0](x)
            y_enh = self.g_a[1](x)
            y = torch.cat([y_base,y_enh],dim = 1).to(x.device)
        y_shape = y.shape[2:]


        latent_means, latent_scales, z_likelihoods = self.compute_hyperprior(y, quality)
 
        y_slices = y.chunk(self.num_slices, 1) # total amount of slices

        y_hat_slices_base = []
        y_likelihood_base= []
        y_likelihood_enhanced = []
        x_hat_progressive = []

        scales_baseline = []


        for slice_index in range(self.ns0):
            y_slice = y_slices[slice_index]
            idx = slice_index%self.ns0

            indice = min(self.max_support_slices,idx)
            support_slices = (y_hat_slices_base if self.max_support_slices < 0 \
                                                        else y_hat_slices_base[:indice])               
            
            mean_support = torch.cat([latent_means[:,:self.dimensions_M[0]]] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales[:,:self.dimensions_M[0]]] + support_slices, dim=1) 

            
            mu = self.cc_mean_transforms[idx](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]  

            scale = self.cc_scale_transforms[idx](scale_support)#self.extract_scale(idx,slice_index,scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            scales_baseline.append(scale)
            _, y_slice_likelihood = self.gaussian_conditional(y_slice,
                                                            scale, 
                                                            mu, 
                                                            training = training)
            y_hat_slice = ste_round(y_slice - mu) + mu


            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.lrp_transforms[idx](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp               

            
            y_likelihood_enhanced.append(y_slice_likelihood)

            y_hat_slices_base.append(y_hat_slice)
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
                scale_support = torch.cat([latent_scales[:,self.dimensions_M[0]:]] + support_slices, dim=1) 

            
                mu = self.cc_mean_transforms_prog[current_index](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
                mu = mu[:, :, :y_shape[0], :y_shape[1]]  

                scale = self.cc_scale_transforms_prog[current_index](scale_support)#self.extract_scale(idx,slice_index,scale_support)
                scale = scale[:, :, :y_shape[0], :y_shape[1]]


                if self.double_dim is False: # or mask_pol in ("two-levels","point-base-std", "three-levels-std"):

                    block_mask = self.masking(scale,
                                            scale_base = None,
                                            slice_index = current_index,
                                            pr = q, 
                                            mask_pol = mask_pol) 
                else:
                    
                    sc_base = torch.cat([scale,y_hat_slices_base[current_index]],dim = 1)

                    block_mask = self.masking(scale = scale,
                                            scale_base = sc_base,
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
                y_hat_slice = self.merge(y_hat_slice,y_hat_slices_base[current_index],current_index)
 
                y_hat_slices_quality.append(y_hat_slice)
                #y_hat_slices_only_quality.append(y_hat_slice)
                y_likelihood_quality.append(y_slice_likelihood)

            
            y_hat_enhanced = torch.cat(y_hat_slices_quality,dim = 1) 
            if self.multiple_decoder:
                x_hat_current = self.g_s[1](y_hat_enhanced)
            else: 
                #y_hat_enhanced = self.g_s_prog(y_hat_enhanced)
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
            "y_hat":y_hat_total
        }
    


    def compress(self, x, quality = 0.0, mask_pol = None):


        mask_pol = self.mask_policy if mask_pol is None else mask_pol

        if self.multiple_encoder is False:
            y = self.g_a(x)
        else:
            y_base = self.g_a[0](x)
            y_enh = self.g_a[1](x)
            y = torch.cat([y_base,y_enh],dim = 1).to(x.device)
        y_shape = y.shape[2:]

    
        z = self.h_a(y)

        z_strings =  self.entropy_bottleneck.compress(z)
        
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        latent_scales_base = self.h_scale_s(z_hat) if self.multiple_hyperprior is False else self.h_scale_s[0](z_hat)
        latent_means_base = self.h_mean_s(z_hat) if self.multiple_hyperprior is False else self.h_mean_s[0](z_hat)

        if self.multiple_hyperprior is False or quality == 0:
            latent_means =  latent_means_base # torch.cat([latent_means_base,latent_means_enh],dim = 1)
            latent_scales = latent_scales_base #torch.cat([latent_scales_base,latent_scales_enh],dim = 1) 
        else:
            latent_scales_enh = self.h_scale_s[1](z_hat) 
            latent_means_enh = self.h_mean_s[1](z_hat)
            latent_means = torch.cat([latent_means_base,latent_means_enh],dim = 1)
            latent_scales = torch.cat([latent_scales_base,latent_scales_enh],dim = 1) 

        y_hat_slices = []

        y_slices = y.chunk(self.num_slices, 1) # total amount of slices

        y_strings = []
        masks = []

        scales_baseline = []

        for slice_index in range(self.ns0):
            y_slice = y_slices[slice_index]
            indice = min(self.max_support_slices,slice_index%self.ns0)
            support_slices = (y_hat_slices if self.max_support_slices < 0 \
                                                        else y_hat_slices[:indice])               
            
            idx = slice_index%self.ns0
            mean_support = torch.cat([latent_means[:,:self.division_dimension[0]]] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales[:,:self.division_dimension[0]]] + support_slices, dim=1) 

            
            mu = self.cc_mean_transforms[idx](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]  
            scale = self.cc_scale_transforms[idx](scale_support)#self.extract_scale(idx,slice_index,scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            scales_baseline.append(scale)

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_string  = self.gaussian_conditional.compress(y_slice, index,mu)

            y_hat_slice = self.gaussian_conditional.decompress(y_q_string, index)
            y_hat_slice = y_hat_slice + mu

            y_strings.append(y_q_string)

            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.lrp_transforms[idx](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        if quality <= 0:
            return {"strings": [y_strings, z_strings],"shape":z.size()[-2:], "masks":masks}
        
        y_hat_slices_quality = []
        #y_hat_slices_quality = y_hat_slices + []

        for slice_index in range(self.ns0,self.ns1):

            y_slice = y_slices[slice_index]
            current_index = slice_index%self.ns0

            support_slices = self.determine_support(y_hat_slices,current_index,y_hat_slices_quality) 
            mean_support = torch.cat([latent_means[:,self.division_dimension[0]:]] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales[:,self.division_dimension[0]:]] + support_slices, dim=1)  

            mu = self.cc_mean_transforms_prog[current_index](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]  

            scale = self.cc_scale_transforms_prog[current_index](scale_support)#self.extract_scale(idx,slice_index,scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            if self.double_dim is False: # or mask_pol in ("two-levels","point-base-std", "three-levels-std"):
                block_mask = self.masking(scale,
                                            scale_base = None,
                                            slice_index = current_index,
                                            pr = quality, 
                                            mask_pol = mask_pol) 
            else:
                sc_base = torch.cat([scale,y_hat_slices[current_index]],dim = 1)
                block_mask = self.masking(scale = scale,
                                            scale_base = sc_base,
                                            slice_index = current_index,
                                            pr = quality, 
                                            mask_pol = mask_pol)   
            masks.append(block_mask)
            block_mask = self.masking.apply_noise(block_mask, False)
            index = self.gaussian_conditional.build_indexes(scale*block_mask).int() #ffff

            y_q_string  = self.gaussian_conditional.compress((y_slice - mu)*block_mask, index)
            y_strings.append(y_q_string)
            
            y_hat_slice = self.gaussian_conditional.decompress(y_q_string, index)
            y_hat_slice = y_hat_slice + mu

            

            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.lrp_transforms_prog[current_index](lrp_support) #ddd
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slice = self.merge(y_hat_slice,y_hat_slices[current_index],current_index)

            y_hat_slices_quality.append(y_hat_slice)

        return {"strings": [y_strings, z_strings],"shape":z.size()[-2:],"masks":masks}

    def decompress(self, strings, shape, quality, mask_pol = None, masks = None):

        mask_pol = self.mask_policy if mask_pol is None else mask_pol

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales_base = self.h_scale_s(z_hat) if self.multiple_hyperprior is False else self.h_scale_s[0](z_hat)
        latent_means_base = self.h_mean_s(z_hat) if self.multiple_hyperprior is False else self.h_mean_s[0](z_hat)

        
        if self.multiple_hyperprior is False or quality == 0:
            latent_scales = latent_scales_base #torch.zeros_like(latent_scales_base).to(latent_scales_base.device) 
            latent_means = latent_means_base #torch.zeros_like(latent_means_base).to(latent_means_base.device) 
        else:
            latent_scales_enh = self.h_scale_s[1](z_hat) 
            latent_means_enh = self.h_mean_s[1](z_hat)
            latent_means = torch.cat([latent_means_base,latent_means_enh],dim = 1)
            latent_scales = torch.cat([latent_scales_base,latent_scales_enh],dim = 1) 

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_string = strings[0]
        y_hat_slices = []

        scales_baseline = []
        for slice_index in range(self.num_slice_cumulative_list[0]): #ddd
            pr_strings = y_string[slice_index]
            idx = slice_index%self.num_slice_cumulative_list[0]
            indice = min(self.max_support_slices,idx)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:indice]) 
            
            mean_support = torch.cat([latent_means[:,:self.division_dimension[0]]] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales[:,:self.division_dimension[0]]] + support_slices, dim=1) 

            
            mu = self.cc_mean_transforms[idx](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]  
            scale = self.cc_scale_transforms[idx](scale_support)#self.extract_scale(idx,slice_index,scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            scales_baseline.append(scale)

            index = self.gaussian_conditional.build_indexes(scale)


            rv = self.gaussian_conditional.decompress(pr_strings, index )
            rv = rv.reshape(mu.shape)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[idx](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            #if quality == 0 or slice_index <self.num_slice_cumulative_list[0]:
            y_hat_slices.append(y_hat_slice)
            #else:
            #    y_hat_slices_enh.append(y_hat_slice)
        if quality == 0:
            y_hat_b = torch.cat(y_hat_slices, dim=1)
            x_hat = self.g_s[0](y_hat_b).clamp_(0, 1) if self.multiple_decoder else self.g_s(y_hat_b).clamp_(0, 1)
            return {"x_hat": x_hat}

        y_hat_slices_quality = []
        for slice_index in range(self.ns0,self.ns1):
            pr_strings = y_string[slice_index]
            current_index = slice_index%self.ns0

            support_slices = self.determine_support(y_hat_slices,current_index,y_hat_slices_quality) 
            mean_support = torch.cat([latent_means[:,self.division_dimension[0]:]] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales[:,self.division_dimension[0]:]] + support_slices, dim=1)             
        
            mu = self.cc_mean_transforms_prog[current_index](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
            scale = self.cc_scale_transforms_prog[current_index](scale_support)#self.extract_scale(idx,slice_index,scale_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]  
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            if self.double_dim is False: # or mask_pol in ("two-levels","point-base-std", "three-levels-std"):
                block_mask = self.masking(scale,
                                            scale_base = None,
                                            slice_index = current_index,
                                            pr = quality, 
                                            mask_pol = mask_pol) 
            else:
                sc_base = torch.cat([scale,y_hat_slices[current_index]],dim = 1)
                block_mask = self.masking(scale = scale,
                                            scale_base = sc_base,
                                            slice_index = current_index,
                                            pr = quality, 
                                            mask_pol = mask_pol)   

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
            y_base = y 
            y_enh = y
        else:
            y_base = self.g_a[0](x)
            y_enh = self.g_a[1](x)
            y = torch.cat([y_base,y_enh],dim = 1).to(x.device) #dddd
        y_shape = y.shape[2:]
        latent_means, latent_scales, z_likelihoods = self.compute_hyperprior(y, quality)

        y_slices = y.chunk(self.num_slices, 1) # total amount of slicesy,

        y_hat_slices = []
        y_likelihood = []

        scales_base = []

        for slice_index in range(self.num_slice_cumulative_list[0]):
            y_slice = y_slices[slice_index]
            idx = slice_index%self.num_slice_cumulative_list[0]
            indice = min(self.max_support_slices,idx)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:indice]) 
            
            mean_support = torch.cat([latent_means[:,:self.division_dimension[0]]] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales[:,:self.division_dimension[0]]] + support_slices, dim=1) 

            
            mu = self.cc_mean_transforms[idx](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]  
            scale = self.cc_scale_transforms[idx](scale_support)#self.extract_scale(idx,slice_index,scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            scales_base.append(scale)


            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu, training = training)
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
                "y_hat":y_hat

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
            
            mean_support = torch.cat([latent_means[:,self.division_dimension[0]:]] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales[:,self.division_dimension[0]:]] + support_slices, dim=1)  

            mu = self.cc_mean_transforms_prog[current_index](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]  

            scale = self.cc_scale_transforms_prog[current_index](scale_support)#self.extract_scale(idx,slice_index,scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            sc_base = torch.cat([scale,y_hat_slices[current_index]],dim = 1) #fff
            block_mask = self.masking(scale,
                                      scale_base = sc_base ,
                                      slice_index = current_index, 
                                      pr = quality,
                                        mask_pol = mask_pol) #scale, slice_index = 0,  pr = 0, mask_pol = None
            block_mask = self.masking.apply_noise(block_mask, False)


            y_slice_m = y_slice  - mu
            y_slice_m = y_slice_m*block_mask

            _, y_slice_likelihood = self.gaussian_conditional(y_slice_m, scale*block_mask, training = training)
            y_hat_slice = ste_round(y_slice - mu)*block_mask + mu

            y_likelihood_quality.append(y_slice_likelihood)

            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.lrp_transforms_prog[current_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp   

            # faccio il merge qua!!!!!
            y_hat_slice = self.merge(y_hat_slice,y_hat_slices[current_index],current_index)   #ddd

            y_hat_slices_quality.append(y_hat_slice)    



        y_likelihoods = torch.cat(y_likelihood_quality,dim = 1) #ddddd
        y_hat = torch.cat(y_hat_slices_quality,dim = 1)  
    
        if self.multiple_decoder:
            x_hat = self.g_s[1](y_hat).clamp_(0, 1)
        else:
            #y_hat_t = self.g_s_prog(y_hat)
            x_hat = self.g_s(y_hat).clamp_(0, 1) 


        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods,"z": z_likelihoods},
            "y_hat":y_hat

        }     



